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
