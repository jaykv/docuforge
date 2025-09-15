"""
Webhook configuration API endpoints.
"""

from fastapi import APIRouter, Depends
from ..core.auth import get_current_user
from ..schemas.requests import ConfigureWebhookRequest
from ..schemas.responses import WebhookResponse, VersionResponse
from ..config import settings
import uuid

router = APIRouter()


@router.post("/configure_webhook", response_model=WebhookResponse)
async def configure_webhook(
    request: ConfigureWebhookRequest,
    current_user: dict = Depends(get_current_user)
) -> WebhookResponse:
    """Configure webhook notifications."""
    
    # Generate webhook ID
    webhook_id = str(uuid.uuid4())
    
    # Store webhook configuration (would be stored in database)
    # For now, just return success response
    
    return WebhookResponse(
        webhook_id=webhook_id,
        status="configured",
        message="Webhook configured successfully"
    )


@router.get("/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    """Get API version information."""
    
    return VersionResponse(
        version=settings.APP_VERSION,
        api_version="1.0.0",
        build_date="2024-01-01",
        features=[
            "document_parsing",
            "structured_extraction", 
            "document_splitting",
            "document_editing",
            "agentic_ocr_correction",
            "multi_engine_ocr",
            "webhook_notifications"
        ]
    )
