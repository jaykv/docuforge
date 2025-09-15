"""
Document processing API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPAuthorizationCredentials
from typing import Dict
import uuid

from ..core.auth import get_current_user, validate_auth_token
from ..core.storage import upload_to_storage, generate_presigned_url
from ..core.database import store_job_metadata
from ..functions import inngest_client
from ..schemas.requests import ParseRequest, ExtractRequest, SplitRequest, EditRequest
from ..schemas.responses import (
    UploadResponse, ParseResponse, ExtractResponse, 
    SplitResponse, EditResponse
)
from ..config import settings

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
) -> UploadResponse:
    """Upload a document and return file_id and presigned_url."""
    
    # Validate file size
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
        )
    
    # Validate file format
    file_extension = file.filename.split(".")[-1].lower() if file.filename else ""
    if f".{file_extension}" not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {settings.SUPPORTED_FORMATS}"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Upload to storage
    storage_path = await upload_to_storage(file, file_id)
    
    # Generate presigned URL for download
    presigned_url = await generate_presigned_url(storage_path)
    
    return UploadResponse(
        file_id=file_id,
        presigned_url=presigned_url
    )


@router.post("/parse", response_model=ParseResponse)
async def parse_document(
    request: ParseRequest,
    current_user: dict = Depends(get_current_user)
) -> ParseResponse:
    """Parse document into structured chunks."""
    
    # Create job
    job_id = str(uuid.uuid4())
    
    # Store job metadata
    await store_job_metadata(job_id, "parse", request.dict())
    
    # Emit event to trigger processing
    await inngest_client.send_event({
        "name": "document.uploaded",
        "data": {
            "job_id": job_id,
            "document_url": str(request.document_url),
            "options": request.options.dict(),
            "advanced_options": request.advanced_options,
            "experimental_options": request.experimental_options,
            "operation": "parse",
            "priority": request.priority
        }
    })
    
    return ParseResponse(job_id=job_id)


@router.post("/extract", response_model=ExtractResponse)
async def extract_structured_data(
    request: ExtractRequest,
    current_user: dict = Depends(get_current_user)
) -> ExtractResponse:
    """Extract structured data using schema."""
    
    job_id = str(uuid.uuid4())
    
    # Store job metadata
    await store_job_metadata(job_id, "extract", request.dict())
    
    # First trigger document processing if needed
    await inngest_client.send_event({
        "name": "document.uploaded",
        "data": {
            "job_id": job_id,
            "document_url": str(request.document_url),
            "options": request.options.dict(),
            "operation": "extract",
            "priority": request.priority
        }
    })
    
    # Then trigger extraction
    await inngest_client.send_event({
        "name": "extract.requested",
        "data": {
            "job_id": job_id,
            "schema": request.schema,
            "system_prompt": request.system_prompt,
            "generate_citations": request.generate_citations,
            "array_extract": request.array_extract,
            "citations_options": request.citations_options
        }
    })
    
    return ExtractResponse(job_id=job_id)


@router.post("/split", response_model=SplitResponse)
async def split_document(
    request: SplitRequest,
    current_user: dict = Depends(get_current_user)
) -> SplitResponse:
    """Split document into sections with table of contents."""
    
    job_id = str(uuid.uuid4())
    
    # Store job metadata
    await store_job_metadata(job_id, "split", request.dict())
    
    # First trigger document processing
    await inngest_client.send_event({
        "name": "document.uploaded",
        "data": {
            "job_id": job_id,
            "document_url": str(request.document_url),
            "options": request.options.dict(),
            "operation": "split",
            "priority": request.priority
        }
    })
    
    # Then trigger splitting
    await inngest_client.send_event({
        "name": "split.requested",
        "data": {
            "job_id": job_id,
            "split_criteria": request.split_criteria,
            "options": request.options.dict()
        }
    })
    
    return SplitResponse(job_id=job_id)


@router.post("/edit", response_model=EditResponse)
async def edit_document(
    request: EditRequest,
    current_user: dict = Depends(get_current_user)
) -> EditResponse:
    """Edit document with form filling and modifications."""
    
    job_id = str(uuid.uuid4())
    
    # Store job metadata
    await store_job_metadata(job_id, "edit", request.dict())
    
    # First trigger document processing
    await inngest_client.send_event({
        "name": "document.uploaded",
        "data": {
            "job_id": job_id,
            "document_url": str(request.document_url),
            "options": request.options.dict(),
            "operation": "edit",
            "priority": request.priority
        }
    })
    
    # Then trigger editing
    await inngest_client.send_event({
        "name": "edit.requested",
        "data": {
            "job_id": job_id,
            "edit_instructions": request.edit_instructions,
            "options": request.options.dict()
        }
    })
    
    return EditResponse(job_id=job_id)


# Async versions of endpoints
@router.post("/parse_async", response_model=ParseResponse)
async def parse_document_async(
    request: ParseRequest,
    current_user: dict = Depends(get_current_user)
) -> ParseResponse:
    """Async version of parse endpoint."""
    return await parse_document(request, current_user)


@router.post("/extract_async", response_model=ExtractResponse)
async def extract_structured_data_async(
    request: ExtractRequest,
    current_user: dict = Depends(get_current_user)
) -> ExtractResponse:
    """Async version of extract endpoint."""
    return await extract_structured_data(request, current_user)


@router.post("/split_async", response_model=SplitResponse)
async def split_document_async(
    request: SplitRequest,
    current_user: dict = Depends(get_current_user)
) -> SplitResponse:
    """Async version of split endpoint."""
    return await split_document(request, current_user)


@router.post("/edit_async", response_model=EditResponse)
async def edit_document_async(
    request: EditRequest,
    current_user: dict = Depends(get_current_user)
) -> EditResponse:
    """Async version of edit endpoint."""
    return await edit_document(request, current_user)
