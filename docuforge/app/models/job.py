"""
Job model for tracking document processing jobs.
"""

from sqlalchemy import Column, String, JSON, Text, Integer, DateTime
from sqlalchemy.sql import func
from .base import BaseModel


class Job(BaseModel):
    """Model for tracking document processing jobs."""
    
    __tablename__ = "jobs"
    
    operation = Column(String, nullable=False)  # parse, extract, split, edit
    status = Column(String, default="processing")  # processing, completed, failed, cancelled
    document_url = Column(String, nullable=False)
    options = Column(JSON, default={})
    advanced_options = Column(JSON, default={})
    experimental_options = Column(JSON, default={})
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    duration = Column(Integer, nullable=True)  # seconds
    usage = Column(JSON, nullable=True)  # pages, credits, etc.
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Job(id={self.id}, operation={self.operation}, status={self.status})>"
