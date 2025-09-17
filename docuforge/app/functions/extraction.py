"""
Structured data extraction functions.
"""

from typing import Dict, Any
import inngest
from ..functions import inngest_client
from ..services.ai_service import extract_with_schema, generate_citations
from ..core.database import get_db_session
import structlog

logger = structlog.get_logger()


@inngest_client.create_function(
    fn_id="extract-structured-data",
    trigger=inngest.TriggerEvent(event="extract.requested"),
)
async def extract_structured_data(ctx: inngest.Context, step: inngest.Step) -> Dict[str, Any]:
    """Extract structured data using schemas and prompts."""
    job_id = ctx.event.data["job_id"]
    schema = ctx.event.data["schema"]
    system_prompt = ctx.event.data.get("system_prompt", "Be precise and thorough.")
    
    logger.info(f"Starting structured data extraction for job {job_id}")

    # Step 1: Load processed document
    document = await ctx.step.run(
        "load-document",
        lambda: load_processed_document(job_id)
    )

    # Step 2: Schema-based extraction
    extracted_data = await ctx.step.run(
        "schema-extraction",
        lambda: extract_with_schema(document, schema, system_prompt)
    )

    # Step 3: Generate citations if requested
    if ctx.event.data.get("generate_citations", False):
        citations = await ctx.step.run(
            "generate-citations",
            lambda: generate_citations(extracted_data, document)
        )
        extracted_data["citations"] = citations

    logger.info(f"Structured data extraction completed for job {job_id}")
    return extracted_data


async def load_processed_document(job_id: str) -> Dict[str, Any]:
    """Load processed document from database."""
    from ..models.job import Job
    from ..core.database import async_session_factory
    
    async with async_session_factory() as session:
        job = await session.get(Job, job_id)
        if job and job.result:
            return job.result
        else:
            raise ValueError(f"No processed document found for job {job_id}")
