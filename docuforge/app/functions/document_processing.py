"""
Main document processing pipeline using Inngest.
"""

import tempfile
import time
from typing import Dict, Any
from inngest import Context
from ..functions import inngest_client
from ..core.database import update_job_status, get_db_session
from ..models.metrics import ProcessingMetrics
from ..services.ocr_service import (
    download_document,
    preprocess_document,
    analyze_document_layout,
    tesseract_ocr,
    paddle_ocr,
    easy_ocr,
    ensemble_ocr_results,
)
from ..services.ai_service import agentic_error_correction
from ..utils.monitoring import monitor_function_execution
import structlog

logger = structlog.get_logger()


@inngest_client.create_function(
    fn_id="document-processing-pipeline",
    trigger=inngest_client.trigger.event(event="document.uploaded"),
    retries=3,
    concurrency=10
)
@monitor_function_execution
async def process_document(ctx: Context) -> Dict[str, Any]:
    """Main document processing pipeline."""
    document_url = ctx.event.data["document_url"]
    job_id = ctx.event.data["job_id"]
    options = ctx.event.data.get("options", {})
    
    logger.info(f"Starting document processing for job {job_id}")

    try:
        # Step 1: Download and validate document
        document_info = await ctx.step.run(
            "download-document",
            lambda: download_document(document_url, job_id)
        )

        # Step 2: Preprocess document
        preprocessed = await ctx.step.run(
            "preprocess-document",
            lambda: preprocess_document(document_info, options)
        )

        # Step 3: Layout analysis
        layout_info = await ctx.step.run(
            "analyze-layout",
            lambda: analyze_document_layout(preprocessed)
        )

        # Step 4: OCR processing (parallel)
        ocr_results = await ctx.step.parallel([
            ("tesseract-ocr", lambda: tesseract_ocr(preprocessed, layout_info)),
            ("paddle-ocr", lambda: paddle_ocr(preprocessed, layout_info)),
            ("easy-ocr", lambda: easy_ocr(preprocessed, layout_info))
        ])

        # Step 5: Ensemble OCR results
        ensemble_result = await ctx.step.run(
            "ensemble-ocr",
            lambda: ensemble_ocr_results(ocr_results)
        )

        # Step 6: Agentic correction
        corrected_result = await ctx.step.run(
            "agentic-correction",
            lambda: agentic_error_correction(ensemble_result, options)
        )

        # Step 7: Post-processing
        final_result = await ctx.step.run(
            "post-process",
            lambda: post_process_document(corrected_result, options)
        )

        # Step 8: Store results
        await ctx.step.run(
            "store-results",
            lambda: store_processing_results(job_id, final_result)
        )

        # Emit completion event
        await ctx.step.send_event(
            "document-processed",
            {
                "name": "document.processed",
                "data": {
                    "job_id": job_id,
                    "status": "completed",
                    "result": final_result
                }
            }
        )

        logger.info(f"Document processing completed for job {job_id}")
        return final_result

    except Exception as e:
        logger.error(f"Document processing failed for job {job_id}: {str(e)}")
        
        # Handle errors and emit failure event
        await ctx.step.send_event(
            "document-failed",
            {
                "name": "document.failed",
                "data": {
                    "job_id": job_id,
                    "error": str(e),
                    "status": "failed"
                }
            }
        )
        
        # Update job status
        await update_job_status(job_id, "failed", str(e))
        raise


@inngest_client.create_function(
    fn_id="parallel-ocr-processing",
    trigger=inngest_client.trigger.event(event="ocr.parallel"),
    retries=2
)
async def parallel_ocr_processing(ctx: Context) -> Dict[str, Any]:
    """Parallel OCR processing with multiple engines."""
    document_path = ctx.event.data["document_path"]
    layout_info = ctx.event.data["layout_info"]
    
    # Run OCR engines in parallel
    results = await ctx.step.parallel([
        ("tesseract", lambda: tesseract_ocr(document_path, layout_info)),
        ("paddle", lambda: paddle_ocr(document_path, layout_info)),
        ("easy", lambda: easy_ocr(document_path, layout_info))
    ])
    
    return {"ocr_results": results}


@inngest_client.create_function(
    fn_id="agentic-correction",
    trigger=inngest_client.trigger.event(event="correction.agentic"),
    retries=2
)
async def agentic_correction(ctx: Context) -> Dict[str, Any]:
    """Agentic error correction using AI models."""
    ocr_result = ctx.event.data["ocr_result"]
    options = ctx.event.data.get("options", {})
    
    corrected_result = await ctx.step.run(
        "ai-correction",
        lambda: agentic_error_correction(ocr_result, options)
    )
    
    return {"corrected_result": corrected_result}


async def post_process_document(result: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process document results."""
    # Add chunking, formatting, and other post-processing
    processed_result = {
        "content": result.get("content", ""),
        "metadata": result.get("metadata", {}),
        "confidence": result.get("confidence", 0.0),
        "processing_time": result.get("processing_time", 0),
        "chunks": [],  # Will be populated based on chunking options
    }
    
    # Apply chunking if requested
    chunk_mode = options.get("chunking", {}).get("chunk_mode", "disabled")
    if chunk_mode != "disabled":
        processed_result["chunks"] = await apply_chunking(result, chunk_mode)
    
    return processed_result


async def apply_chunking(result: Dict[str, Any], chunk_mode: str) -> list:
    """Apply chunking strategy to document."""
    content = result.get("content", "")
    
    if chunk_mode == "page":
        # Split by pages
        return [{"page": i, "content": page} for i, page in enumerate(content.split("\n\n"), 1)]
    elif chunk_mode == "section":
        # Split by sections (basic implementation)
        sections = content.split("\n\n\n")
        return [{"section": i, "content": section} for i, section in enumerate(sections, 1)]
    else:
        # Variable chunking (default)
        chunk_size = 1000  # characters
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append({
                "chunk": len(chunks) + 1,
                "content": content[i:i + chunk_size]
            })
        return chunks


async def store_processing_results(job_id: str, result: Dict[str, Any]):
    """Store processing results in database."""
    from ..models.job import Job
    from ..core.database import async_session_factory
    
    async with async_session_factory() as session:
        job = await session.get(Job, job_id)
        if job:
            job.result = result
            job.status = "completed"
            job.completed_at = time.time()
            await session.commit()
