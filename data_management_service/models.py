from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timezone
import uuid

class Patient(BaseModel):
    patient_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date_of_birth: date

    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    class Config:
        from_attributes = True # For ORM integration


class HealthRecordData(BaseModel):
    """
    Flexible structure for various health data points.
    NOTE: Current recorded fields are from the diabetes.csv dataset.
    """
    Pregnancies: Optional[int] = None
    Glucose: Optional[float] = None
    BloodPressure: Optional[int] = None
    SkinThickness: Optional[float] = None
    Insulin: Optional[float] = None
    BMI: Optional[float] = None
    DiabetesPedigreeFunction: Optional[float] = None
    Outcome: Optional[int] = None


class HealthRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    record_type: str
    source: Optional[str] = None # e.g., "EHR", "PatientReported", "SensorWearableX"
    
    data: HealthRecordData

    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    class Config:
        from_attributes = True # For ORM integration


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
        # Example to show in API docs
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
