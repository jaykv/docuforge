"""
Document model for storing document metadata.
"""

from sqlalchemy import Column, String, Integer, Boolean, JSON
from .base import BaseModel


class Document(BaseModel):
    """Model for storing document metadata."""
    
    __tablename__ = "documents"
    
    file_id = Column(String, unique=True, nullable=False)
    original_filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    num_pages = Column(Integer, nullable=True)
    processed = Column(Boolean, default=False)
    ocr_result = Column(JSON, nullable=True)
    layout_analysis = Column(JSON, nullable=True)
    embeddings_stored = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Document(id={self.id}, file_id={self.file_id}, filename={self.original_filename})>"
