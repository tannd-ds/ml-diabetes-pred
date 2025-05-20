import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

def create_dummy_data(num_samples=200):
    """ For dev purposes only. """
    data = {}
    for feature in NUMERICAL_FEATURES:
        data[feature] = np.random.rand(num_samples) * 100 
        if feature == 'Age': data[feature] = np.random.randint(20, 80, num_samples)
        if feature == 'BMI': data[feature] = np.random.uniform(18, 40, num_samples)

    for feature in BOOLEAN_FEATURES:
        data[feature] = np.random.choice([True, False, None], size=num_samples, p=[0.4, 0.4, 0.2]) # Allow Nones

    target = (data['Glucose'] > 70) & (data['BMI'] > 25) | (data['Age'] > 50)
    df = pd.DataFrame(data)
    df['Outcome'] = target.astype(int)
    return df

def main():
    print("Generating dummy data...")
    df = create_dummy_data(num_samples=500)

    for bf in BOOLEAN_FEATURES:
        # Simple fillna with False for ALL boolean features 
        # # Real scenario needs careful consideration (imputation, specific category, etc.)
        df[bf] = df[bf].fillna(False).astype(int)
    
    for nf in NUMERICAL_FEATURES:
        if df[nf].isnull().any():
            df[nf] = df[nf].fillna(df[nf].median()) # median imputation for Nones for num features

    X = df[NUMERICAL_FEATURES + BOOLEAN_FEATURES]
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Defining preprocessing and model pipeline...")
    # Preprocessor: Scale numerical, passthrough boolean (already 0/1)
    # If we had actual categorical, OHE would be here.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('bool', 'passthrough', BOOLEAN_FEATURES)
        ],
        remainder='drop' # Drop other columns
    )

    model = LogisticRegression(solver='liblinear', random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("features", ALL_FEATURES)
        mlflow.log_param("random_state", 42)

        print("Training the pipeline...")
        pipeline.fit(X_train, y_train)

        print("Evaluating the model...")
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        mlflow.log_metric("test_accuracy", accuracy)

        # Log the pipeline using mlflow.sklearn.log_model
        # This packages the preprocessor and model together.
        print("Logging pipeline to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="sk_pipeline",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=X_train.head(5),
        )
        model_uri = f"runs:/{run_id}/sk_pipeline"
        print(f"MLflow Model URI: {model_uri}")

    # Save pipeline to a local file for direct loading by the service
    # This is a separate step from MLflow logging but useful for easy local access
    model_dir = Path(__file__).parent / "ml_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = model_dir / "diabetes_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")

    print("\n--- Instructions for service integration ---")
    print("1. Ensure 'mlruns' directory (created by MLflow) and 'ml_model/diabetes_pipeline.joblib' are copied to the Docker image.")
    print("2. The service will initially load from 'ml_model/diabetes_pipeline.joblib'.")
    print(f"3. To load from MLflow directly (later), use Model URI: {model_uri}")
    print("   (This requires the 'mlruns' directory to be accessible to the service at runtime with the correct structure)")


if __name__ == "__main__":
    main() 