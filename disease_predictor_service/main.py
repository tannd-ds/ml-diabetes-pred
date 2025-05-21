from pathlib import Path
from models import DiabetesPredictionInput, DiabetesPredictionOutput
import pandas as pd
from fastapi import FastAPI, HTTPException, status
import contextlib
import joblib


diabetes_model_pipeline = None
MODEL_PATH = Path(__file__).parent / "ml_model" / "diabetes_pipeline.joblib"

@contextlib.asynccontextmanager
async def lifespan(app):
    global diabetes_model_pipeline
    print(f"Loading ML model from: {MODEL_PATH}")
    try:
        diabetes_model_pipeline = joblib.load(MODEL_PATH)
        print("ML model loaded successfully.")
    except FileNotFoundError:
        diabetes_model_pipeline = None # Ensure it's None if not found
        print(f"Error: Model file not found at {MODEL_PATH}. Prediction endpoint will not work.")
    except Exception as e:
        diabetes_model_pipeline = None
        print(f"Error loading ML model: {e}. Prediction endpoint will not work.")

    yield
    print("Application shutdown.")

app = FastAPI(
    title="Disease Predictor Service",
    description="Provides AI/ML models for disease prediction. Initial focus: Type 2 Diabetes.",
    version="0.1.0",
    lifespan=lifespan
)


@app.post("/predict/typ2_diabetes", response_model=DiabetesPredictionOutput, status_code=status.HTTP_200_OK)
async def predict_diabetes(input_data: DiabetesPredictionInput):
    global diabetes_model_pipeline
    if diabetes_model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Cannot perform predictions."
        )

    try:
        # Pydantic model > dictionary > DataFrame
        input_dict = input_data.model_dump()
        # The order of columns in the DataFrame should match what the model was trained on.
        df = pd.DataFrame([input_dict], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])

        prediction = diabetes_model_pipeline.predict(df)
        # Get probabilities (returns array of shape (n_samples, n_classes))
        probabilities = diabetes_model_pipeline.predict_proba(df)

        return DiabetesPredictionOutput(
            predicted_outcome=int(prediction[0]),
            probability_outcome_0=float(probabilities[0, 0]),
            probability_outcome_1=float(probabilities[0, 1])
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing prediction: {str(e)}"
        )


@app.get("/")
async def root():
    return {"message": "Disease Predictor Service is running. Focus: Type 2 Diabetes."}