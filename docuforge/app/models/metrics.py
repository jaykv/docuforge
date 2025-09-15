"""
Processing metrics model for monitoring and analytics.
"""

from sqlalchemy import Column, String, Float, Boolean, Text, JSON, DateTime
from sqlalchemy.sql import func
from .base import Base
import uuid


class ProcessingMetrics(Base):
    """Model for storing processing step metrics."""
    
    __tablename__ = "processing_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, nullable=False)
    step_name = Column(String, nullable=False)
    duration = Column(Float, nullable=False)  # seconds
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, default={})
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<ProcessingMetrics(job_id={self.job_id}, step={self.step_name}, success={self.success})>"
