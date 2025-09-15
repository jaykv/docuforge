"""
Request schemas for API endpoints.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List
from .common import BaseProcessingOptions


class ParseRequest(BaseModel):
    """Request schema for document parsing."""
    document_url: HttpUrl
    options: Optional[BaseProcessingOptions] = BaseProcessingOptions()
    advanced_options: Optional[Dict[str, Any]] = {}
    experimental_options: Optional[Dict[str, Any]] = {}
    priority: bool = True


class ExtractRequest(BaseModel):
    """Request schema for structured data extraction."""
    document_url: HttpUrl
    schema: Any = Field(..., description="JSON schema for extraction")
    system_prompt: str = "Be precise and thorough."
    generate_citations: bool = False
    options: Optional[BaseProcessingOptions] = BaseProcessingOptions()
    advanced_options: Optional[Dict[str, Any]] = {}
    experimental_options: Optional[Dict[str, Any]] = {}
    array_extract: Optional[Dict[str, Any]] = {"enabled": False}
    use_chunking: bool = False
    include_images: bool = False
    spreadsheet_agent: bool = False
    experimental_table_citations: bool = True
    priority: bool = True
    citations_options: Optional[Dict[str, Any]] = {"numerical_confidence": False}


class SplitRequest(BaseModel):
    """Request schema for document splitting."""
    document_url: HttpUrl
    split_criteria: Dict[str, Any] = Field(
        default={"method": "auto"},
        description="Criteria for splitting the document"
    )
    options: Optional[BaseProcessingOptions] = BaseProcessingOptions()
    advanced_options: Optional[Dict[str, Any]] = {}
    experimental_options: Optional[Dict[str, Any]] = {}
    priority: bool = True


class EditRequest(BaseModel):
    """Request schema for document editing."""
    document_url: HttpUrl
    edit_instructions: Dict[str, Any] = Field(
        ...,
        description="Instructions for editing the document"
    )
    options: Optional[BaseProcessingOptions] = BaseProcessingOptions()
    advanced_options: Optional[Dict[str, Any]] = {}
    experimental_options: Optional[Dict[str, Any]] = {}
    priority: bool = True


class WebhookConfig(BaseModel):
    """Webhook configuration."""
    url: HttpUrl
    events: List[str] = ["job.completed", "job.failed"]
    secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = {}


class ConfigureWebhookRequest(BaseModel):
    """Request schema for webhook configuration."""
    webhook: WebhookConfig
