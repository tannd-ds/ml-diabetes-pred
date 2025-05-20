from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Disease Predictor Service",
    description="Provides AI/ML models for disease prediction. Initial focus: Type 2 Diabetes.",
    version="0.1.0"
)

class DiabetesPredictionInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


class PredictionOutput(BaseModel):
    disease: str
    prediction_probability: float
    risk_level: str
    recommendations: List[str] = []


@app.post("/predict/type2_diabetes", response_model=PredictionOutput)
async def predict_type2_diabetes(data: DiabetesPredictionInput):
    """
    Predicts the likelihood of Type 2 Diabetes based on input features.
    NOTE: This is a placeholder and will be replaced with an actual ML model inference.
    """
    print(f"Received data for prediction: {data.dict()}")
    predicted_prob = 0.1
    risk = "Low"

    score = 0
    if data.Glucose and data.Glucose > 100: score +=1
    if data.BMI and data.BMI > 25: score +=1
    if data.DiabetesPedigreeFunction: score +=1
    if data.Age: score +=1

    if score >= 4 or (data.Glucose and data.Glucose > 125 and data.BMI and data.BMI > 30):
        predicted_prob = 0.85
        risk = "High"
    elif score >= 2:
        predicted_prob = 0.65
        risk = "Medium"
    
    recs = ["Consult with a healthcare professional for a comprehensive assessment."]
    if risk == "Medium":
        recs.append("Consider lifestyle modifications like diet and exercise.")
    if risk == "High":
        recs.append("Immediate medical consultation is advised. Further tests may be required.")

    return PredictionOutput(
        disease="Type 2 Diabetes",
        prediction_probability=predicted_prob,
        risk_level=risk,
        recommendations=recs
    )

@app.get("/")
async def root():
    return {"message": "Disease Predictor Service is running. Focus: Type 2 Diabetes."}