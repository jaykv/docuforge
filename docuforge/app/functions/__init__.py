"""
Inngest functions for document processing workflows.
"""

from inngest import Inngest

from ..config import settings

# Initialize Inngest client
inngest_client = Inngest(
    app_id="docuforge",
    is_production=not settings.INNGEST_DEV
)

# Import all functions
from .document_processing import (
    process_document,
    parallel_ocr_processing,
    agentic_correction
)
from .extraction import extract_structured_data
from .splitting import split_document
from .editing import edit_document

# Function registry
FUNCTIONS = [
    process_document,
    parallel_ocr_processing,
    agentic_correction,
    extract_structured_data,
    split_document,
    edit_document
]



