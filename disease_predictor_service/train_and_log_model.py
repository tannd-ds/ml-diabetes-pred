import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

CATEGORICAL_FEATURES = []
BOOLEAN_FEATURES = []
NUMERICAL_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]
ALL_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

def main():
    print("Loading data...")
    data_path = Path(__file__).parent / "data" / "diabetes.csv"
    df = pd.read_csv(data_path)

    # Basic Preprocessing (handle potential inconsistencies if any from the new dataset)
    # Example: Ensure target 'Outcome' is present and numeric
    if 'Outcome' not in df.columns:
        raise ValueError("Target column 'Outcome' not found in the dataset.")
    # df['Outcome'] = pd.to_numeric(df['Outcome'], errors='coerce') # Optional: if Outcome might have non-numeric strings
    # df.dropna(subset=['Outcome'], inplace=True) # Optional: drop rows where Outcome became NaN

    print("Preprocessing data...")
    for bf in BOOLEAN_FEATURES:
        # Simple fillna with False for ALL boolean features 
        # # Real scenario needs careful consideration (imputation, specific category, etc.)
        df[bf] = df[bf].fillna(False).astype(int)
    
    for nf in NUMERICAL_FEATURES:
        df[nf] = df[nf].astype(float)
        if df[nf].isnull().any():
            df[nf] = df[nf].fillna(df[nf].median()) # median imputation for Nones for num features

    X = df[NUMERICAL_FEATURES + BOOLEAN_FEATURES]
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Defining preprocessing and model pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('bool', 'passthrough', BOOLEAN_FEATURES)
        ],
        remainder='drop' # Drop other columns
    )

    base_model = LogisticRegression(solver='liblinear', random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', base_model)
    ])

    # Note: The parameter names for GridSearchCV are prefixed with 'classifier__'
    # because 'classifier' is the name of the model in the pipeline.
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2']
    }

    # use a simple GridSearchCV
    grid_search = GridSearchCV(pipeline,
                               param_grid,
                               cv=5,
                               scoring='accuracy',
                               verbose=1)


    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("features", ALL_FEATURES)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 5)

        print("Training the pipeline with GridSearchCV...")
        grid_search.fit(X_train, y_train)

        # Log best parameters and score from GridSearchCV
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

        best_pipeline = grid_search.best_estimator_

        print("Evaluating the best model on the test set...")
        y_pred = best_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy with best model: {accuracy:.4f}")
        mlflow.log_metric("test_accuracy", accuracy)

        # Log the pipeline using mlflow.sklearn.log_model
        print("Logging best pipeline to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="sk_pipeline",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=X_train.head(5),
        )
        model_uri = f"runs:/{run_id}/sk_pipeline"
        print(f"MLflow Model URI: {model_uri}")

    # Save the best pipeline to a local file
    model_dir = Path(__file__).parent / "ml_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = model_dir / "diabetes_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path) # Save the best pipeline
    print(f"Best pipeline saved to: {pipeline_path}")

    print("\n--- Instructions for service integration ---")
    print("1. Ensure 'mlruns' directory (created by MLflow) and 'ml_model/diabetes_pipeline.joblib' are copied to the Docker image.")
    print("2. The service will initially load from 'ml_model/diabetes_pipeline.joblib'.")
    print(f"3. To load from MLflow directly (later), use Model URI: {model_uri}")
    print("   (This requires the 'mlruns' directory to be accessible to the service at runtime with the correct structure)")


if __name__ == "__main__":
    main() 