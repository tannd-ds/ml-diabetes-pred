import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, mapped_column
from sqlalchemy import String, Date, DateTime, JSON
from datetime import datetime, date, timezone
import uuid

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@db:5432/appdb")

engine = create_async_engine(DATABASE_URL, echo=True) # echo=True for logging SQL
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class PatientORM(Base):
    __tablename__ = "patients"

    patient_id = mapped_column(
        String, 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    date_of_birth = mapped_column(
        Date, 
        nullable=False
    )
    created_at = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc), 
        onupdate=lambda: datetime.now(timezone.utc)
    )


class HealthRecordORM(Base):
    __tablename__ = "health_records"

    record_id = mapped_column(
        String, 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    patient_id = mapped_column(
        String, 
        index=True, 
        nullable=False
    )
    record_type = mapped_column(
        String, 
        nullable=False
    )
    source = mapped_column(
        String, 
        nullable=True
    )
    data = mapped_column(
        JSON, 
        nullable=False
    )
    created_at = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc), 
        onupdate=lambda: datetime.now(timezone.utc)
    )
    

async def get_async_db_session():
    async with AsyncSessionLocal() as session:
        yield session


async def create_db_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"Database tables created (if they didn't exist) for {DATABASE_URL}")