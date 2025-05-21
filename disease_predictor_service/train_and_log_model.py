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
import requests
import os

CATEGORICAL_FEATURES = []
BOOLEAN_FEATURES = []
NUMERICAL_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]
ALL_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data_management_service:8001/training_dataset/diabetes_features")

def main():
    print(f"Fetching training data from Data Management Service at {DATA_SERVICE_URL}...")
    try:
        response = requests.get(DATA_SERVICE_URL)
        response.raise_for_status()
        data_list = response.json() 

        if not data_list:
            print("No data received from the data management service. Exiting.")
            return

        df = pd.DataFrame(data_list)
        print(f"Successfully fetched and loaded {len(df)} records into DataFrame.")
        print(f"Columns available: {df.columns.tolist()}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {DATA_SERVICE_URL}: {e}")
        print("Please ensure the Data Management Service is running and accessible, and the URL is correct.")
        return
    except Exception as e: 
        print(f"An error occurred while processing data from the service: {e}")
        return

    # Define ALL_FEATURES based on the columns present in the DataFrame, excluding 'Outcome' for X
    if 'Outcome' not in df.columns:
        print("Error: 'Outcome' column not found in fetched data. Cannot proceed with training.")
        return
    
    # TODO: Ensure all NUMERICAL_FEATURES are present in the dataframe from the API
    
    # Basic Preprocessing
    for nf in NUMERICAL_FEATURES:
        if df[nf].isnull().any():
            df[nf] = pd.to_numeric(df[nf], errors='coerce')
            df[nf] = df[nf].fillna(df[nf].median()) 

    X = df[NUMERICAL_FEATURES + BOOLEAN_FEATURES]
    y = df['Outcome']
    
    all_features_for_logging = X.columns.tolist() 

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
        mlflow.log_param("features_used", all_features_for_logging)
        mlflow.log_param("target_variable", "Outcome")
        mlflow.log_param("data_source_url", DATA_SERVICE_URL)
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

        print("Logging best pipeline to MLflow...")
        input_example_df = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="sk_pipeline",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=input_example_df,
        )
        model_uri = f"runs:/{run_id}/sk_pipeline"
        print(f"MLflow Model URI: {model_uri}")

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