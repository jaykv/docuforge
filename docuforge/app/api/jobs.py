"""
Job management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from ..core.auth import get_current_user
from ..core.database import get_job_by_id, update_job_status
from ..functions import inngest_client
from ..schemas.responses import JobResponse, CancelResponse
from ..schemas.common import JobStatus

router = APIRouter()


@router.get("/job/{job_id}", response_model=JobResponse)
async def get_job_result(
    job_id: str,
    current_user: dict = Depends(get_current_user)
) -> JobResponse:
    """Retrieve job results."""
    
    # Get job from database
    job = await get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(**job)


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(10, ge=1, le=100, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    current_user: dict = Depends(get_current_user)
) -> List[JobResponse]:
    """List jobs with optional filtering."""
    
    # This would typically query the database with filters
    # For now, return empty list as placeholder
    return []


@router.post("/cancel/{job_id}", response_model=CancelResponse)
async def cancel_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
) -> CancelResponse:
    """Cancel a running job."""
    
    # Check if job exists
    job = await get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if job can be cancelled
    if job["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel job with status: {job['status']}"
        )
    
    # Cancel in Inngest
    await inngest_client.send_event({
        "name": "job.cancel",
        "data": {"job_id": job_id}
    })
    
    # Update job status
    await update_job_status(job_id, "cancelled")
    
    return CancelResponse(job_id=job_id)
