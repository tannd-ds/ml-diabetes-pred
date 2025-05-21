from pydantic import BaseModel

class DiabetesPredictionInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

    class Config:
        json_schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148.0,
                "BloodPressure": 72.0,
                "SkinThickness": 35.0,
                "Insulin": 0.0, # Typical value for missing insulin in datasets
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }

class DiabetesPredictionOutput(BaseModel):
    predicted_outcome: int
    probability_outcome_0: float
    probability_outcome_1: float

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_outcome": 1,
                "probability_outcome_0": 0.3,
                "probability_outcome_1": 0.7
            }
        }