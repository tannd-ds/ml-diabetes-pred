from fastapi import FastAPI, HTTPException, status, Depends
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
import contextlib
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta # For age calculation

from models import (
    Patient as PatientPydantic, 
    HealthRecord as HealthRecordPydantic, 
    HealthRecordData, 
)
from database import create_db_tables, get_async_db_session
import crud

MODEL_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Attempting to create database tables...")
    await create_db_tables()
    print("Database tables check/creation process complete.")
    yield
    print("Application shutdown.")

app = FastAPI(
    title="Data Management Service",
    description="Manages patient data and health records using PostgreSQL. Initial focus: Type 2 Diabetes related data.",
    version="0.1.0",
    lifespan=lifespan
)


# --- Patient Endpoints ---
@app.post("/patients/", response_model=PatientPydantic, status_code=status.HTTP_201_CREATED)
async def create_patient(patient_data: PatientPydantic, db: AsyncSession = Depends(get_async_db_session)):
    """Creates a new patient record in the database."""
    db_patient_orm = await crud.create_db_patient(db=db, patient=patient_data)
    return db_patient_orm

@app.get("/patients/", response_model=List[PatientPydantic])
async def list_patients(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_async_db_session)):
    """Lists all patient records from the database."""
    db_patients_orm = await crud.get_db_patients(db=db, skip=skip, limit=limit)
    return db_patients_orm

@app.get("/patients/{patient_id}", response_model=PatientPydantic)
async def get_patient(patient_id: str, db: AsyncSession = Depends(get_async_db_session)):
    """Retrieves a specific patient by their ID from the database."""
    db_patient_orm = await crud.get_db_patient(db=db, patient_id=patient_id)
    if db_patient_orm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return db_patient_orm

# Health Record Endpoints
@app.post("/health_records/", response_model=HealthRecordPydantic, status_code=status.HTTP_201_CREATED)
async def create_health_record(record_data: HealthRecordPydantic, db: AsyncSession = Depends(get_async_db_session)):
    """Creates a new health record for a patient in the database."""
    # Check if patient exists first
    db_patient_orm = await crud.get_db_patient(db=db, patient_id=record_data.patient_id)
    if db_patient_orm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient with ID {record_data.patient_id} not found.")
    
    db_health_record_orm = await crud.create_db_health_record(db=db, record=record_data)
    return db_health_record_orm

@app.get("/health_records/patient/{patient_id}", response_model=List[HealthRecordPydantic])
async def get_health_records_for_patient(patient_id: str, skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_async_db_session)):
    """Retrieves all health records for a specific patient from the database."""
    db_patient_orm = await crud.get_db_patient(db=db, patient_id=patient_id)
    if db_patient_orm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient with ID {patient_id} not found.")
        
    db_health_records_orm = await crud.get_db_health_records_for_patient(db=db, patient_id=patient_id, skip=skip, limit=limit)
    return db_health_records_orm

@app.get("/health_records/{record_id}", response_model=HealthRecordPydantic)
async def get_health_record(record_id: str, db: AsyncSession = Depends(get_async_db_session)):
    """Retrieves a specific health record by its ID from the database."""
    db_health_record_orm = await crud.get_db_health_record(db=db, record_id=record_id)
    if db_health_record_orm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Health record not found")
    return db_health_record_orm

@app.get("/training_dataset/diabetes_features", response_model=List[Dict[str, Any]])
async def get_diabetes_training_dataset(db: AsyncSession = Depends(get_async_db_session)):
    """
    Provides a dataset suitable for training the diabetes prediction model.
    It fetches patient and health record data, calculates age, and attempts to 
    construct a feature set for each patient based on their LATEST AVAILABLE RECORD 
    (if multiple records exist per patient from the query).
    """
    all_records_data = await crud.get_db_all_patient_health_records_for_training(db) # Gets all, joined

    training_data = []
    processed_patient_ids = set()

    for health_record_orm, patient_orm in all_records_data:
        if patient_orm.patient_id in processed_patient_ids:
            continue

        record_data_dict = health_record_orm.data

        # Calculate Age
        today = datetime.now(timezone.utc).date()
        age = relativedelta(today, patient_orm.date_of_birth).years

        feature_dict = {"Age": age}

        for feature_name in MODEL_FEATURES:
            # Skip Age as it's calculated
            if feature_name == "Age":
                continue

            value = record_data_dict.get(feature_name)
            feature_dict[feature_name] = value

        training_data.append(feature_dict)
        processed_patient_ids.add(patient_orm.patient_id)
        
    if not training_data:
        pass # Returning empty list if no data
    else:
        # NOTE: This is for DEV purposes only.
        if len(training_data) < 5:
            print("Loading diabetes.csv file as training data...")
            import pandas as pd
            df = pd.read_csv("data/diabetes.csv")
            training_data = df.to_dict(orient="records")

    return training_data

@app.get("/")
async def root():
    return {"message": "Data Management Service is running with PostgreSQL backend."}
