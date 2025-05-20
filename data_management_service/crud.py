from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update as sqlalchemy_update, join
from typing import List, Optional, Any
from datetime import datetime, timezone

from models import Patient as PatientPydantic, HealthRecord as HealthRecordPydantic
from database import PatientORM, HealthRecordORM

# --- Patient CRUD ---
async def create_db_patient(db: AsyncSession, patient: PatientPydantic) -> PatientORM:
    db_patient = PatientORM(
        patient_id=patient.patient_id, # Using Pydantic generated ID
        date_of_birth=patient.date_of_birth,
        # created_at and updated_at have defaults in ORM
    )
    db.add(db_patient)
    await db.commit()
    await db.refresh(db_patient)
    return db_patient

async def get_db_patient(db: AsyncSession, patient_id: str) -> Optional[PatientORM]:
    result = await db.execute(select(PatientORM).filter(PatientORM.patient_id == patient_id))
    return result.scalars().first()

async def get_db_patients(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[PatientORM]:
    result = await db.execute(select(PatientORM).offset(skip).limit(limit))
    return result.scalars().all()

# --- HealthRecord CRUD --- 
async def create_db_health_record(db: AsyncSession, record: HealthRecordPydantic) -> HealthRecordORM:
    db_record = HealthRecordORM(
        record_id=record.record_id, # Using Pydantic generated ID
        patient_id=record.patient_id,
        record_type=record.record_type,
        source=record.source,
        data=record.data.model_dump() # Serialize Pydantic sub-model to dict for JSON field
    )
    db.add(db_record)
    await db.commit()
    await db.refresh(db_record)
    return db_record

async def get_db_health_record(db: AsyncSession, record_id: str) -> Optional[HealthRecordORM]:
    result = await db.execute(select(HealthRecordORM).filter(HealthRecordORM.record_id == record_id))
    return result.scalars().first()

async def get_db_health_records_for_patient(db: AsyncSession, patient_id: str, skip: int = 0, limit: int = 100) -> List[HealthRecordORM]:
    result = await db.execute(
        select(HealthRecordORM)
        .filter(HealthRecordORM.patient_id == patient_id)
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def get_db_all_patient_health_records_for_training(db, record_type_filter=None):
    """
    Fetches health records and associated patient data. 
    Can optionally filter by health record_type.
    This is a basic version; complex aggregation or "latest record per patient" logic
    would require more sophisticated querying or processing in the service layer.
    """
    stmt = (
        select(HealthRecordORM, PatientORM)
        .join(PatientORM, HealthRecordORM.patient_id == PatientORM.patient_id)
    )
    if record_type_filter:
        stmt = stmt.filter(HealthRecordORM.record_type == record_type_filter)
    
    stmt = stmt.order_by(PatientORM.patient_id, HealthRecordORM.created_at.desc())

    result = await db.execute(stmt)
    return result.all() # Returns a list of Row objects (tuples) containing (HealthRecordORM_instance, PatientORM_instance)

# Note: Update and Delete operations would be added here as needed.
# Example update for patient's updated_at field (if not handled by onupdate):
# async def update_db_patienttimestamp(db: AsyncSession, patient_id: str):
#     stmt = (
#         sqlalchemy_update(PatientORM)
#         .where(PatientORM.patient_id == patient_id)
#         .values(updated_at=datetime.now(timezone.utc))
#         .execution_options(synchronize_session="fetch")
#     )
#     await db.execute(stmt)
#     await db.commit() 