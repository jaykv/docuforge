"""
OCR processing service with multiple engines.
"""

import tempfile
import os
import time
from typing import Dict, Any, List
import httpx
import cv2
import numpy as np
from PIL import Image
import pytesseract
import structlog

logger = structlog.get_logger()


async def download_document(document_url: str, job_id: str) -> Dict[str, Any]:
    """Download document from URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(document_url)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            # Get file info
            file_size = len(response.content)
            content_type = response.headers.get("content-type", "application/pdf")
            
            logger.info(f"Downloaded document for job {job_id}: {file_size} bytes")
            
            return {
                "path": temp_path,
                "size": file_size,
                "content_type": content_type,
                "pages": 1  # Would be determined by actual PDF analysis
            }
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise


async def preprocess_document(document_info: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess document for OCR."""
    try:
        document_path = document_info["path"]
        
        # Convert PDF to images if needed
        if document_info["content_type"] == "application/pdf":
            image_paths = await convert_pdf_to_images(document_path)
        else:
            image_paths = [document_path]
        
        # Preprocess images
        preprocessed_images = []
        for image_path in image_paths:
            preprocessed_image = await preprocess_image(image_path, options)
            preprocessed_images.append(preprocessed_image)
        
        return {
            "original_path": document_path,
            "preprocessed_images": preprocessed_images,
            "num_pages": len(preprocessed_images)
        }
    except Exception as e:
        logger.error(f"Error preprocessing document: {e}")
        raise


async def convert_pdf_to_images(pdf_path: str) -> List[str]:
    """Convert PDF to images."""
    try:
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path, dpi=300)
        image_paths = []
        
        for i, image in enumerate(images):
            temp_path = f"{pdf_path}_page_{i}.png"
            image.save(temp_path, "PNG")
            image_paths.append(temp_path)
        
        return image_paths
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        # Fallback: return original path
        return [pdf_path]


async def preprocess_image(image_path: str, options: Dict[str, Any]) -> str:
    """Preprocess image for better OCR results."""
    try:
        # Load image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Save preprocessed image
        preprocessed_path = f"{image_path}_preprocessed.png"
        cv2.imwrite(preprocessed_path, thresh)
        
        return preprocessed_path
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image_path  # Return original if preprocessing fails


async def analyze_document_layout(preprocessed: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze document layout."""
    try:
        # Simple layout analysis (would use LayoutLM or YOLO in production)
        layout_info = {
            "text_regions": [],
            "table_regions": [],
            "figure_regions": [],
            "header_regions": [],
            "footer_regions": []
        }
        
        # For each page, identify regions
        for i, image_path in enumerate(preprocessed["preprocessed_images"]):
            page_layout = await analyze_page_layout(image_path)
            layout_info["text_regions"].extend(page_layout.get("text_regions", []))
            layout_info["table_regions"].extend(page_layout.get("table_regions", []))
            layout_info["figure_regions"].extend(page_layout.get("figure_regions", []))
        
        return layout_info
    except Exception as e:
        logger.error(f"Error analyzing document layout: {e}")
        return {"text_regions": [], "table_regions": [], "figure_regions": []}


async def analyze_page_layout(image_path: str) -> Dict[str, Any]:
    """Analyze layout of a single page."""
    # Simplified layout analysis
    return {
        "text_regions": [{"x": 0, "y": 0, "width": 100, "height": 100, "confidence": 0.9}],
        "table_regions": [],
        "figure_regions": []
    }


async def tesseract_ocr(preprocessed: Dict[str, Any], layout_info: Dict[str, Any]) -> Dict[str, Any]:
    """Perform OCR using Tesseract."""
    try:
        start_time = time.time()
        all_text = []
        
        for image_path in preprocessed["preprocessed_images"]:
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(Image.open(image_path), config=custom_config)
            all_text.append(text)
        
        processing_time = time.time() - start_time
        
        return {
            "engine": "tesseract",
            "text": "\n\n".join(all_text),
            "confidence": 0.85,  # Would calculate actual confidence
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Tesseract OCR error: {e}")
        return {"engine": "tesseract", "text": "", "confidence": 0.0, "error": str(e)}


async def paddle_ocr(preprocessed: Dict[str, Any], layout_info: Dict[str, Any]) -> Dict[str, Any]:
    """Perform OCR using PaddleOCR."""
    try:
        start_time = time.time()
        
        # PaddleOCR would be initialized here
        # For now, return mock result
        mock_text = "Mock PaddleOCR result for document processing."
        
        processing_time = time.time() - start_time
        
        return {
            "engine": "paddle",
            "text": mock_text,
            "confidence": 0.88,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"PaddleOCR error: {e}")
        return {"engine": "paddle", "text": "", "confidence": 0.0, "error": str(e)}


async def easy_ocr(preprocessed: Dict[str, Any], layout_info: Dict[str, Any]) -> Dict[str, Any]:
    """Perform OCR using EasyOCR."""
    try:
        start_time = time.time()
        
        # EasyOCR would be initialized here
        # For now, return mock result
        mock_text = "Mock EasyOCR result for document processing."
        
        processing_time = time.time() - start_time
        
        return {
            "engine": "easy",
            "text": mock_text,
            "confidence": 0.82,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"EasyOCR error: {e}")
        return {"engine": "easy", "text": "", "confidence": 0.0, "error": str(e)}


async def ensemble_ocr_results(ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine OCR results using ensemble method."""
    try:
        from ..config import settings
        
        weights = settings.OCR_ENSEMBLE_WEIGHTS
        total_confidence = 0
        weighted_text = ""
        
        # Simple ensemble: weighted average based on confidence
        for i, (engine_name, result) in enumerate(ocr_results):
            if i < len(weights) and result.get("text"):
                weight = weights[i] * result.get("confidence", 0)
                total_confidence += weight
                
                # For simplicity, use the highest confidence result
                if weight > total_confidence * 0.4:  # If this result has significant weight
                    weighted_text = result["text"]
        
        return {
            "ensemble_text": weighted_text,
            "confidence": total_confidence / sum(weights) if weights else 0,
            "individual_results": dict(ocr_results),
            "method": "weighted_confidence"
        }
    except Exception as e:
        logger.error(f"Error in OCR ensemble: {e}")
        # Fallback to first available result
        for engine_name, result in ocr_results:
            if result.get("text"):
                return {
                    "ensemble_text": result["text"],
                    "confidence": result.get("confidence", 0),
                    "individual_results": dict(ocr_results),
                    "method": "fallback"
                }
        return {"ensemble_text": "", "confidence": 0, "individual_results": {}, "method": "failed"}
