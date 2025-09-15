"""
Response schemas for API endpoints.
"""

from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
from datetime import datetime
from .common import JobStatus, UsageInfo, ErrorInfo, Citation


class UploadResponse(BaseModel):
    """Response schema for file upload."""
    file_id: str
    presigned_url: HttpUrl


class JobResponse(BaseModel):
    """Response schema for job operations."""
    job_id: str
    status: JobStatus = JobStatus.PROCESSING
    duration: Optional[int] = None
    pdf_url: Optional[HttpUrl] = None
    studio_link: Optional[HttpUrl] = None
    usage: Optional[UsageInfo] = None
    result: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ParseResponse(JobResponse):
    """Response schema for document parsing."""
    pass


class ExtractResponse(JobResponse):
    """Response schema for structured extraction."""
    citations: Optional[List[Citation]] = None


class SplitResponse(JobResponse):
    """Response schema for document splitting."""
    table_of_contents: Optional[List[Dict[str, Any]]] = None


class EditResponse(JobResponse):
    """Response schema for document editing."""
    validation: Optional[Dict[str, Any]] = None


class CancelResponse(BaseModel):
    """Response schema for job cancellation."""
    job_id: str
    status: str = "cancelled"
    message: str = "Job cancelled successfully"


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = "healthy"
    version: str
    timestamp: float
    services: Optional[Dict[str, str]] = None


class VersionResponse(BaseModel):
    """Response schema for version information."""
    version: str
    api_version: str
    build_date: str
    features: List[str]


class WebhookResponse(BaseModel):
    """Response schema for webhook configuration."""
    webhook_id: str
    status: str = "configured"
    message: str = "Webhook configured successfully"
