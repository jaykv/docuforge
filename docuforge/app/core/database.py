"""
Database connection and session management.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator
from ..config import settings
from ..models.base import Base

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DEBUG,
)

# Create async session factory
async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_database():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database():
    """Close database connections."""
    await engine.dispose()


async def update_job_status(job_id: str, status: str, error_message: str = None):
    """Update job status in database."""
    from ..models.job import Job
    
    async with async_session_factory() as session:
        job = await session.get(Job, job_id)
        if job:
            job.status = status
            if error_message:
                job.error_message = error_message
            await session.commit()


async def get_job_by_id(job_id: str) -> dict:
    """Get job by ID."""
    from ..models.job import Job
    
    async with async_session_factory() as session:
        job = await session.get(Job, job_id)
        if job:
            return {
                "job_id": job.id,
                "status": job.status,
                "operation": job.operation,
                "result": job.result,
                "error_message": job.error_message,
                "duration": job.duration,
                "usage": job.usage,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
            }
        return None


async def store_job_metadata(job_id: str, operation: str, request_data: dict):
    """Store job metadata in database."""
    from ..models.job import Job
    
    async with async_session_factory() as session:
        job = Job(
            id=job_id,
            operation=operation,
            document_url=request_data.get("document_url"),
            options=request_data.get("options", {}),
            advanced_options=request_data.get("advanced_options", {}),
            experimental_options=request_data.get("experimental_options", {}),
        )
        session.add(job)
        await session.commit()
