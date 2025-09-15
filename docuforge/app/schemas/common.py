"""
Common schemas and enums.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class OCRMode(str, Enum):
    """OCR processing modes."""
    STANDARD = "standard"
    AGENTIC = "agentic"


class ExtractionMode(str, Enum):
    """Data extraction modes."""
    OCR = "ocr"
    METADATA = "metadata"
    HYBRID = "hybrid"


class ChunkMode(str, Enum):
    """Document chunking modes."""
    VARIABLE = "variable"
    SECTION = "section"
    PAGE = "page"
    BLOCK = "block"
    DISABLED = "disabled"
    PAGE_SECTIONS = "page_sections"


class JobStatus(str, Enum):
    """Job processing status."""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_mode: ChunkMode = ChunkMode.VARIABLE
    chunk_size: Optional[int] = None


class TableSummaryConfig(BaseModel):
    """Configuration for table summarization."""
    enabled: bool = False
    method: str = "ai"
    include_headers: bool = True


class FigureSummaryConfig(BaseModel):
    """Configuration for figure summarization."""
    enabled: bool = False
    enhanced: bool = False
    override: bool = False
    method: str = "vision_ai"


class BaseProcessingOptions(BaseModel):
    """Base processing options for all operations."""
    ocr_mode: OCRMode = OCRMode.STANDARD
    extraction_mode: ExtractionMode = ExtractionMode.OCR
    chunking: ChunkingConfig = ChunkingConfig()
    table_summary: TableSummaryConfig = TableSummaryConfig()
    figure_summary: FigureSummaryConfig = FigureSummaryConfig()
    filter_blocks: List[str] = []
    force_url_result: bool = False


class UsageInfo(BaseModel):
    """Usage information for billing/tracking."""
    pages_processed: int = 0
    credits_used: int = 0
    processing_time: float = 0.0
    api_calls: int = 0


class ErrorInfo(BaseModel):
    """Error information."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class Citation(BaseModel):
    """Citation information for extracted data."""
    field: str
    value: str
    source: str
    confidence: float
    page: int
    position: Dict[str, Any]
