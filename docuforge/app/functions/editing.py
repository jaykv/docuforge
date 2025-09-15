"""
Document editing functions.
"""

from typing import Dict, Any
from inngest import Context
from ..functions import inngest_client
from ..functions.extraction import load_processed_document
import structlog

logger = structlog.get_logger()


@inngest_client.create_function(
    fn_id="edit-document",
    trigger=inngest_client.trigger.event(event="edit.requested"),
    retries=2
)
async def edit_document(ctx: Context) -> Dict[str, Any]:
    """Edit document with form filling and modifications."""
    job_id = ctx.event.data["job_id"]
    edit_instructions = ctx.event.data.get("edit_instructions", {})
    
    logger.info(f"Starting document editing for job {job_id}")

    # Step 1: Load processed document
    document = await ctx.step.run(
        "load-document",
        lambda: load_processed_document(job_id)
    )

    # Step 2: Apply edits
    edited_document = await ctx.step.run(
        "apply-edits",
        lambda: apply_document_edits(document, edit_instructions)
    )

    # Step 3: Validate edits
    validation_result = await ctx.step.run(
        "validate-edits",
        lambda: validate_document_edits(edited_document, edit_instructions)
    )

    logger.info(f"Document editing completed for job {job_id}")
    return {
        "edited_document": edited_document,
        "validation": validation_result
    }


async def apply_document_edits(document: Dict[str, Any], edit_instructions: Dict[str, Any]) -> Dict[str, Any]:
    """Apply edits to document."""
    edited_document = document.copy()
    content = edited_document.get("content", "")
    
    # Apply text replacements
    replacements = edit_instructions.get("replacements", [])
    for replacement in replacements:
        old_text = replacement.get("old_text", "")
        new_text = replacement.get("new_text", "")
        if old_text and old_text in content:
            content = content.replace(old_text, new_text)
    
    # Apply form field filling
    form_fields = edit_instructions.get("form_fields", {})
    for field_name, field_value in form_fields.items():
        # Simple form field replacement (would be more sophisticated in real implementation)
        placeholder = f"[{field_name}]"
        if placeholder in content:
            content = content.replace(placeholder, str(field_value))
    
    edited_document["content"] = content
    edited_document["edit_metadata"] = {
        "edits_applied": len(replacements) + len(form_fields),
        "edit_timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
    }
    
    return edited_document


async def validate_document_edits(edited_document: Dict[str, Any], edit_instructions: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that edits were applied correctly."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    content = edited_document.get("content", "")
    
    # Check that required fields were filled
    required_fields = edit_instructions.get("required_fields", [])
    for field in required_fields:
        placeholder = f"[{field}]"
        if placeholder in content:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Required field '{field}' was not filled")
    
    # Check for any remaining placeholders
    import re
    remaining_placeholders = re.findall(r'\[([^\]]+)\]', content)
    if remaining_placeholders:
        validation_result["warnings"].append(f"Unfilled placeholders found: {remaining_placeholders}")
    
    return validation_result
