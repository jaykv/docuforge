"""
Document splitting functions.
"""

from typing import Dict, Any, List
from inngest import Context
from ..functions import inngest_client
from ..functions.extraction import load_processed_document
import structlog

logger = structlog.get_logger()


@inngest_client.create_function(
    fn_id="split-document",
    trigger=inngest_client.trigger.event(event="split.requested"),
    retries=2
)
async def split_document(ctx: Context) -> Dict[str, Any]:
    """Split document into sections with table of contents."""
    job_id = ctx.event.data["job_id"]
    options = ctx.event.data.get("options", {})
    
    logger.info(f"Starting document splitting for job {job_id}")

    # Step 1: Load processed document
    document = await ctx.step.run(
        "load-document",
        lambda: load_processed_document(job_id)
    )

    # Step 2: Identify section boundaries
    sections = await ctx.step.run(
        "identify-sections",
        lambda: identify_document_sections(document, options)
    )

    # Step 3: Generate table of contents
    toc = await ctx.step.run(
        "generate-toc",
        lambda: generate_table_of_contents(sections)
    )

    # Step 4: Split into individual sections
    split_sections = await ctx.step.run(
        "split-sections",
        lambda: split_into_sections(document, sections)
    )

    logger.info(f"Document splitting completed for job {job_id}")
    return {
        "table_of_contents": toc,
        "sections": split_sections
    }


async def identify_document_sections(document: Dict[str, Any], options: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify section boundaries in document."""
    content = document.get("content", "")
    
    # Simple section identification based on headers
    sections = []
    lines = content.split("\n")
    current_section = {"title": "Introduction", "start_line": 0, "content": ""}
    
    for i, line in enumerate(lines):
        # Check if line looks like a header (simple heuristic)
        if line.strip() and (line.isupper() or line.startswith("#") or 
                           any(keyword in line.lower() for keyword in ["chapter", "section", "part"])):
            # Save previous section
            if current_section["content"].strip():
                current_section["end_line"] = i - 1
                sections.append(current_section)
            
            # Start new section
            current_section = {
                "title": line.strip(),
                "start_line": i,
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"
    
    # Add final section
    if current_section["content"].strip():
        current_section["end_line"] = len(lines) - 1
        sections.append(current_section)
    
    return sections


async def generate_table_of_contents(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate table of contents from sections."""
    toc = []
    for i, section in enumerate(sections):
        toc.append({
            "section_number": i + 1,
            "title": section["title"],
            "page": section.get("page", 1),  # Would be calculated from layout analysis
            "level": 1  # Would be determined from header analysis
        })
    return toc


async def split_into_sections(document: Dict[str, Any], sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split document into individual sections."""
    split_sections = []
    
    for i, section in enumerate(sections):
        split_sections.append({
            "section_id": i + 1,
            "title": section["title"],
            "content": section["content"].strip(),
            "metadata": {
                "start_line": section["start_line"],
                "end_line": section.get("end_line", section["start_line"]),
                "word_count": len(section["content"].split()),
                "character_count": len(section["content"])
            }
        })
    
    return split_sections
