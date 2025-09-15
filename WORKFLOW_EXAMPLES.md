# Docuforge - Inngest Workflow Examples

## Overview

This document provides comprehensive examples of Inngest functions for document processing workflows in the Docuforge. These examples demonstrate real-world patterns, error handling, parallel processing, and event-driven architecture.

## Table of Contents

1. [Basic Document Processing](#1-basic-document-processing)
2. [Parallel OCR Processing](#2-parallel-ocr-processing)
3. [Agentic Error Correction](#3-agentic-error-correction)
4. [Structured Data Extraction](#4-structured-data-extraction)
5. [Document Splitting](#5-document-splitting)
6. [Batch Processing](#6-batch-processing)
7. [Error Handling Patterns](#7-error-handling-patterns)
8. [Event Chaining](#8-event-chaining)

## 1. Basic Document Processing

### 1.1 Main Document Processing Pipeline

```python
# app/functions/document_processing.py
from inngest import Context
from ..functions import inngest_client
from ..services.ocr_service import OCRService
from ..services.storage_service import StorageService
from ..core.database import get_db_session
import structlog

logger = structlog.get_logger()

@inngest_client.create_function(
    fn_id="document-processing-pipeline",
    trigger=inngest.TriggerEvent(event="document.uploaded"),
    retries=3,
    concurrency=10
)
async def process_document(ctx: Context) -> dict:
    """
    Main document processing pipeline
    
    Event data expected:
    - job_id: str
    - document_url: str
    - options: dict
    - operation: str (parse, extract, split, edit)
    """
    
    job_id = ctx.event.data["job_id"]
    document_url = ctx.event.data["document_url"]
    options = ctx.event.data.get("options", {})
    operation = ctx.event.data.get("operation", "parse")
    
    logger.info("Starting document processing", job_id=job_id, operation=operation)
    
    try:
        # Step 1: Download and validate document
        document_info = await ctx.step.run(
            "download-document",
            lambda: download_and_validate_document(document_url, job_id)
        )
        
        # Step 2: Preprocess document (image enhancement, PDF conversion)
        preprocessed = await ctx.step.run(
            "preprocess-document",
            lambda: preprocess_document(document_info, options)
        )
        
        # Step 3: Layout analysis
        layout_info = await ctx.step.run(
            "analyze-layout",
            lambda: analyze_document_layout(preprocessed, options)
        )
        
        # Step 4: OCR processing (trigger parallel processing)
        ocr_event_id = await ctx.step.send_event(
            "trigger-ocr",
            {
                "name": "ocr.parallel_process",
                "data": {
                    "job_id": job_id,
                    "document_info": preprocessed,
                    "layout_info": layout_info,
                    "options": options
                }
            }
        )
        
        # Step 5: Wait for OCR completion and get results
        ocr_results = await ctx.step.wait_for_event(
            "wait-for-ocr",
            event="ocr.completed",
            match="data.job_id",
            timeout="10m"
        )
        
        # Step 6: Agentic correction (if enabled)
        if options.get("ocr_mode") == "agentic":
            corrected_result = await ctx.step.run(
                "agentic-correction",
                lambda: agentic_error_correction(ocr_results.data["result"], options)
            )
        else:
            corrected_result = ocr_results.data["result"]
        
        # Step 7: Post-processing based on operation type
        final_result = await ctx.step.run(
            "post-process",
            lambda: post_process_document(corrected_result, operation, options)
        )
        
        # Step 8: Store results
        await ctx.step.run(
            "store-results",
            lambda: store_processing_results(job_id, final_result, operation)
        )
        
        # Step 9: Emit completion event
        await ctx.step.send_event(
            "document-processed",
            {
                "name": "document.processed",
                "data": {
                    "job_id": job_id,
                    "operation": operation,
                    "status": "completed",
                    "result": final_result,
                    "duration": ctx.attempt.duration_ms
                }
            }
        )
        
        logger.info("Document processing completed", job_id=job_id)
        return final_result
        
    except Exception as e:
        logger.error("Document processing failed", job_id=job_id, error=str(e))
        
        # Update job status
        await ctx.step.run(
            "update-job-status",
            lambda: update_job_status(job_id, "failed", str(e))
        )
        
        # Emit failure event
        await ctx.step.send_event(
            "document-failed",
            {
                "name": "document.failed",
                "data": {
                    "job_id": job_id,
                    "operation": operation,
                    "error": str(e),
                    "status": "failed"
                }
            }
        )
        
        raise

# Helper functions
async def download_and_validate_document(document_url: str, job_id: str) -> dict:
    """Download and validate document"""
    storage_service = StorageService()
    
    # Download document
    local_path = await storage_service.download_document(document_url, job_id)
    
    # Validate file format and size
    file_info = await storage_service.validate_document(local_path)
    
    return {
        "local_path": local_path,
        "file_info": file_info,
        "job_id": job_id
    }

async def preprocess_document(document_info: dict, options: dict) -> dict:
    """Preprocess document for OCR"""
    from ..services.image_service import ImageService
    
    image_service = ImageService()
    
    # Convert PDF to images if needed
    if document_info["file_info"]["mime_type"] == "application/pdf":
        images = await image_service.pdf_to_images(
            document_info["local_path"],
            dpi=options.get("dpi", 300)
        )
    else:
        images = [document_info["local_path"]]
    
    # Enhance images
    enhanced_images = []
    for image_path in images:
        enhanced = await image_service.enhance_image(
            image_path,
            options.get("image_enhancement", {})
        )
        enhanced_images.append(enhanced)
    
    return {
        **document_info,
        "images": enhanced_images,
        "num_pages": len(enhanced_images)
    }

async def analyze_document_layout(document_info: dict, options: dict) -> dict:
    """Analyze document layout"""
    from ..services.layout_service import LayoutService
    
    layout_service = LayoutService()
    
    layout_results = []
    for image_path in document_info["images"]:
        layout = await layout_service.analyze_layout(
            image_path,
            model=options.get("layout_model", "default")
        )
        layout_results.append(layout)
    
    return {
        "layouts": layout_results,
        "document_type": layout_service.classify_document_type(layout_results)
    }
```

## 2. Parallel OCR Processing

### 2.1 Multi-Engine OCR Function

```python
# app/functions/ocr_processing.py
from inngest import Context
from ..functions import inngest_client
from ..services.ocr_service import TesseractOCR, PaddleOCR, EasyOCR
import asyncio

@inngest_client.create_function(
    fn_id="parallel-ocr-processing",
    trigger=inngest.TriggerEvent(event="ocr.parallel_process"),
    retries=2,
    concurrency=5
)
async def parallel_ocr_processing(ctx: Context) -> dict:
    """
    Process document with multiple OCR engines in parallel
    
    Event data expected:
    - job_id: str
    - document_info: dict
    - layout_info: dict
    - options: dict
    """
    
    job_id = ctx.event.data["job_id"]
    document_info = ctx.event.data["document_info"]
    layout_info = ctx.event.data["layout_info"]
    options = ctx.event.data["options"]
    
    logger.info("Starting parallel OCR processing", job_id=job_id)
    
    try:
        # Step 1: Run multiple OCR engines in parallel
        ocr_results = await ctx.step.parallel([
            ("tesseract-ocr", lambda: run_tesseract_ocr(document_info, layout_info, options)),
            ("paddle-ocr", lambda: run_paddle_ocr(document_info, layout_info, options)),
            ("easy-ocr", lambda: run_easy_ocr(document_info, layout_info, options))
        ])
        
        # Step 2: Ensemble OCR results
        ensemble_result = await ctx.step.run(
            "ensemble-ocr-results",
            lambda: ensemble_ocr_results(ocr_results, options)
        )
        
        # Step 3: Extract tables and figures
        structured_content = await ctx.step.run(
            "extract-structured-content",
            lambda: extract_tables_and_figures(ensemble_result, layout_info, options)
        )
        
        # Step 4: Generate confidence scores
        confidence_scores = await ctx.step.run(
            "calculate-confidence",
            lambda: calculate_confidence_scores(ensemble_result, ocr_results)
        )
        
        final_result = {
            "ocr_result": ensemble_result,
            "structured_content": structured_content,
            "confidence_scores": confidence_scores,
            "individual_results": {
                "tesseract": ocr_results[0][1],
                "paddle": ocr_results[1][1],
                "easy": ocr_results[2][1]
            }
        }
        
        # Step 5: Emit completion event
        await ctx.step.send_event(
            "ocr-completed",
            {
                "name": "ocr.completed",
                "data": {
                    "job_id": job_id,
                    "result": final_result,
                    "status": "completed"
                }
            }
        )
        
        logger.info("Parallel OCR processing completed", job_id=job_id)
        return final_result
        
    except Exception as e:
        logger.error("Parallel OCR processing failed", job_id=job_id, error=str(e))
        
        # Emit failure event
        await ctx.step.send_event(
            "ocr-failed",
            {
                "name": "ocr.failed",
                "data": {
                    "job_id": job_id,
                    "error": str(e),
                    "status": "failed"
                }
            }
        )
        
        raise

# OCR Engine Functions
async def run_tesseract_ocr(document_info: dict, layout_info: dict, options: dict) -> dict:
    """Run Tesseract OCR"""
    tesseract = TesseractOCR()
    
    results = []
    for i, image_path in enumerate(document_info["images"]):
        layout = layout_info["layouts"][i]
        
        # Configure Tesseract based on layout
        config = tesseract.get_config_for_layout(layout, options)
        
        # Run OCR
        result = await tesseract.process_image(image_path, config)
        results.append(result)
    
    return {
        "engine": "tesseract",
        "results": results,
        "confidence": tesseract.calculate_overall_confidence(results)
    }

async def run_paddle_ocr(document_info: dict, layout_info: dict, options: dict) -> dict:
    """Run PaddleOCR"""
    paddle = PaddleOCR()
    
    results = []
    for i, image_path in enumerate(document_info["images"]):
        layout = layout_info["layouts"][i]
        
        # Run OCR with layout awareness
        result = await paddle.process_image(image_path, layout, options)
        results.append(result)
    
    return {
        "engine": "paddle",
        "results": results,
        "confidence": paddle.calculate_overall_confidence(results)
    }

async def run_easy_ocr(document_info: dict, layout_info: dict, options: dict) -> dict:
    """Run EasyOCR"""
    easy = EasyOCR()
    
    results = []
    for i, image_path in enumerate(document_info["images"]):
        # Run OCR
        result = await easy.process_image(image_path, options)
        results.append(result)
    
    return {
        "engine": "easy",
        "results": results,
        "confidence": easy.calculate_overall_confidence(results)
    }

def ensemble_ocr_results(ocr_results: list, options: dict) -> dict:
    """Combine results from multiple OCR engines"""
    from ..services.ensemble_service import EnsembleService
    
    ensemble = EnsembleService()
    
    # Extract individual results
    tesseract_result = ocr_results[0][1]
    paddle_result = ocr_results[1][1]
    easy_result = ocr_results[2][1]
    
    # Combine using weighted voting based on confidence
    combined_result = ensemble.weighted_ensemble(
        [tesseract_result, paddle_result, easy_result],
        weights=options.get("ensemble_weights", [0.4, 0.4, 0.2])
    )
    
    return combined_result
```

## 3. Agentic Error Correction

### 3.1 AI-Powered OCR Correction

```python
# app/functions/agentic_correction.py
from inngest import Context
from ..functions import inngest_client
from ..services.ai_service import OpenAIService, AnthropicService

@inngest_client.create_function(
    fn_id="agentic-ocr-correction",
    trigger=inngest.TriggerEvent(event="ocr.agentic_correction"),
    retries=2,
    concurrency=3
)
async def agentic_ocr_correction(ctx: Context) -> dict:
    """
    Use AI to correct OCR errors and improve accuracy

    Event data expected:
    - job_id: str
    - ocr_result: dict
    - original_images: list
    - options: dict
    """

    job_id = ctx.event.data["job_id"]
    ocr_result = ctx.event.data["ocr_result"]
    original_images = ctx.event.data["original_images"]
    options = ctx.event.data.get("options", {})

    logger.info("Starting agentic OCR correction", job_id=job_id)

    try:
        # Step 1: Identify potential errors
        error_analysis = await ctx.step.run(
            "analyze-ocr-errors",
            lambda: analyze_ocr_errors(ocr_result, options)
        )

        # Step 2: Generate correction prompts
        correction_prompts = await ctx.step.run(
            "generate-correction-prompts",
            lambda: generate_correction_prompts(ocr_result, error_analysis, options)
        )

        # Step 3: Run AI correction in parallel for different sections
        corrected_sections = await ctx.step.parallel([
            (f"correct-section-{i}", lambda section=section: correct_text_section(section, options))
            for i, section in enumerate(correction_prompts)
        ])

        # Step 4: Validate corrections against original images
        validated_corrections = await ctx.step.run(
            "validate-corrections",
            lambda: validate_corrections_with_vision(
                corrected_sections, original_images, options
            )
        )

        # Step 5: Merge corrected sections
        final_corrected_result = await ctx.step.run(
            "merge-corrections",
            lambda: merge_corrected_sections(validated_corrections, ocr_result)
        )

        # Step 6: Calculate improvement metrics
        improvement_metrics = await ctx.step.run(
            "calculate-improvements",
            lambda: calculate_improvement_metrics(ocr_result, final_corrected_result)
        )

        result = {
            "corrected_text": final_corrected_result,
            "original_text": ocr_result,
            "improvement_metrics": improvement_metrics,
            "correction_details": {
                "errors_found": len(error_analysis["errors"]),
                "corrections_made": len([c for c in corrected_sections if c[1]["changed"]]),
                "confidence_improvement": improvement_metrics["confidence_delta"]
            }
        }

        # Step 7: Emit completion event
        await ctx.step.send_event(
            "correction-completed",
            {
                "name": "agentic_correction.completed",
                "data": {
                    "job_id": job_id,
                    "result": result,
                    "status": "completed"
                }
            }
        )

        logger.info("Agentic correction completed", job_id=job_id)
        return result

    except Exception as e:
        logger.error("Agentic correction failed", job_id=job_id, error=str(e))
        raise

async def correct_text_section(section_data: dict, options: dict) -> dict:
    """Correct a specific text section using AI"""
    ai_service = OpenAIService() if options.get("ai_provider") == "openai" else AnthropicService()

    prompt = f"""
    You are an expert OCR error correction system. Please correct the following text that was extracted from a document using OCR.

    Original OCR Text:
    {section_data['text']}

    Context Information:
    - Document type: {section_data.get('document_type', 'unknown')}
    - Section type: {section_data.get('section_type', 'text')}
    - Language: {section_data.get('language', 'en')}

    Common OCR errors to look for:
    - Character substitutions (e.g., 'rn' -> 'm', '0' -> 'O')
    - Missing or extra spaces
    - Incorrect punctuation
    - Number/letter confusion

    Please provide:
    1. The corrected text
    2. A list of specific corrections made
    3. Confidence score (0-1) for the corrections

    Respond in JSON format:
    {
        "corrected_text": "...",
        "corrections": [{"original": "...", "corrected": "...", "reason": "..."}],
        "confidence": 0.95,
        "changed": true/false
    }
    """

    response = await ai_service.generate_completion(
        prompt,
        model=options.get("correction_model", "gpt-4"),
        temperature=0.1,
        max_tokens=2000
    )

    return json.loads(response)

async def validate_corrections_with_vision(
    corrected_sections: list,
    original_images: list,
    options: dict
) -> list:
    """Validate corrections using vision models"""
    vision_service = OpenAIService()

    validated_sections = []

    for i, (section_id, correction_data) in enumerate(corrected_sections):
        if not correction_data["changed"]:
            validated_sections.append((section_id, correction_data))
            continue

        # Use vision model to validate correction
        validation_prompt = f"""
        Compare the corrected text with what you see in this image region.

        Corrected text: {correction_data['corrected_text']}

        Does the corrected text accurately represent what's visible in the image?
        Respond with: {{"valid": true/false, "confidence": 0.0-1.0, "notes": "..."}}
        """

        # Get the relevant image region (simplified - would need actual region extraction)
        image_path = original_images[min(i, len(original_images) - 1)]

        validation_result = await vision_service.analyze_image_with_text(
            image_path,
            validation_prompt,
            model="gpt-4-vision-preview"
        )

        validation_data = json.loads(validation_result)

        # Keep correction only if validated
        if validation_data["valid"] and validation_data["confidence"] > 0.7:
            validated_sections.append((section_id, correction_data))
        else:
            # Revert to original if validation fails
            original_data = correction_data.copy()
            original_data["corrected_text"] = original_data.get("original_text", "")
            original_data["changed"] = False
            validated_sections.append((section_id, original_data))

    return validated_sections
```

## 4. Structured Data Extraction

### 4.1 Schema-Based Extraction Function

```python
# app/functions/structured_extraction.py
from inngest import Context
from ..functions import inngest_client
from ..services.extraction_service import ExtractionService

@inngest_client.create_function(
    fn_id="structured-data-extraction",
    trigger=inngest.TriggerEvent(event="document.extract"),
    retries=2,
    concurrency=5
)
async def extract_structured_data(ctx: Context) -> dict:
    """
    Extract structured data from document based on schema

    Event data expected:
    - job_id: str
    - document_text: str
    - extraction_schema: dict
    - options: dict
    """

    job_id = ctx.event.data["job_id"]
    document_text = ctx.event.data["document_text"]
    extraction_schema = ctx.event.data["extraction_schema"]
    options = ctx.event.data.get("options", {})

    logger.info("Starting structured data extraction", job_id=job_id)

    try:
        # Step 1: Preprocess text for extraction
        preprocessed_text = await ctx.step.run(
            "preprocess-text",
            lambda: preprocess_text_for_extraction(document_text, options)
        )

        # Step 2: Extract entities and relationships
        entities = await ctx.step.run(
            "extract-entities",
            lambda: extract_named_entities(preprocessed_text, extraction_schema)
        )

        # Step 3: Extract structured fields based on schema
        structured_fields = await ctx.step.parallel([
            (field_name, lambda field=field_config: extract_field_data(
                preprocessed_text, field_name, field, entities
            ))
            for field_name, field_config in extraction_schema["fields"].items()
        ])

        # Step 4: Validate extracted data
        validated_data = await ctx.step.run(
            "validate-extracted-data",
            lambda: validate_extraction_results(structured_fields, extraction_schema)
        )

        # Step 5: Post-process and format results
        formatted_result = await ctx.step.run(
            "format-results",
            lambda: format_extraction_results(validated_data, extraction_schema)
        )

        # Step 6: Generate confidence scores
        confidence_scores = await ctx.step.run(
            "calculate-confidence",
            lambda: calculate_extraction_confidence(formatted_result, entities)
        )

        final_result = {
            "extracted_data": formatted_result,
            "confidence_scores": confidence_scores,
            "entities": entities,
            "schema_version": extraction_schema.get("version", "1.0"),
            "extraction_metadata": {
                "total_fields": len(extraction_schema["fields"]),
                "extracted_fields": len([f for f in formatted_result.values() if f is not None]),
                "overall_confidence": confidence_scores.get("overall", 0.0)
            }
        }

        # Step 7: Store extraction results
        await ctx.step.run(
            "store-extraction-results",
            lambda: store_extraction_results(job_id, final_result)
        )

        # Step 8: Emit completion event
        await ctx.step.send_event(
            "extraction-completed",
            {
                "name": "extraction.completed",
                "data": {
                    "job_id": job_id,
                    "result": final_result,
                    "status": "completed"
                }
            }
        )

        logger.info("Structured data extraction completed", job_id=job_id)
        return final_result

    except Exception as e:
        logger.error("Structured data extraction failed", job_id=job_id, error=str(e))
        raise

async def extract_field_data(text: str, field_name: str, field_config: dict, entities: dict) -> dict:
    """Extract specific field data based on configuration"""
    extraction_service = ExtractionService()

    field_type = field_config.get("type", "text")
    extraction_method = field_config.get("method", "regex")

    if extraction_method == "regex":
        result = extraction_service.extract_with_regex(
            text, field_config["pattern"], field_type
        )
    elif extraction_method == "nlp":
        result = extraction_service.extract_with_nlp(
            text, field_config, entities
        )
    elif extraction_method == "ai":
        result = await extraction_service.extract_with_ai(
            text, field_config, field_name
        )
    else:
        raise ValueError(f"Unknown extraction method: {extraction_method}")

    return {
        "field_name": field_name,
        "value": result["value"],
        "confidence": result["confidence"],
        "method": extraction_method,
        "metadata": result.get("metadata", {})
    }
```

## 5. Document Splitting

### 5.1 Intelligent Document Splitting

```python
# app/functions/document_splitting.py
from inngest import Context
from ..functions import inngest_client
from ..services.splitting_service import SplittingService

@inngest_client.create_function(
    fn_id="document-splitting",
    trigger=inngest.TriggerEvent(event="document.split"),
    retries=2,
    concurrency=3
)
async def split_document(ctx: Context) -> dict:
    """
    Split document into logical sections or pages

    Event data expected:
    - job_id: str
    - document_data: dict
    - split_criteria: dict
    - options: dict
    """

    job_id = ctx.event.data["job_id"]
    document_data = ctx.event.data["document_data"]
    split_criteria = ctx.event.data["split_criteria"]
    options = ctx.event.data.get("options", {})

    logger.info("Starting document splitting", job_id=job_id)

    try:
        # Step 1: Analyze document structure
        structure_analysis = await ctx.step.run(
            "analyze-document-structure",
            lambda: analyze_document_structure(document_data, split_criteria)
        )

        # Step 2: Identify split points
        split_points = await ctx.step.run(
            "identify-split-points",
            lambda: identify_split_points(structure_analysis, split_criteria)
        )

        # Step 3: Split document into sections
        document_sections = await ctx.step.run(
            "split-into-sections",
            lambda: split_into_sections(document_data, split_points, options)
        )

        # Step 4: Process each section in parallel
        processed_sections = await ctx.step.parallel([
            (f"process-section-{i}", lambda section=section: process_document_section(
                section, i, options
            ))
            for i, section in enumerate(document_sections)
        ])

        # Step 5: Generate section metadata
        section_metadata = await ctx.step.run(
            "generate-section-metadata",
            lambda: generate_section_metadata(processed_sections, split_criteria)
        )

        # Step 6: Create output files
        output_files = await ctx.step.run(
            "create-output-files",
            lambda: create_split_output_files(processed_sections, job_id, options)
        )

        final_result = {
            "sections": processed_sections,
            "metadata": section_metadata,
            "output_files": output_files,
            "split_summary": {
                "total_sections": len(processed_sections),
                "split_criteria": split_criteria,
                "original_pages": document_data.get("num_pages", 0)
            }
        }

        # Step 7: Emit completion event
        await ctx.step.send_event(
            "splitting-completed",
            {
                "name": "splitting.completed",
                "data": {
                    "job_id": job_id,
                    "result": final_result,
                    "status": "completed"
                }
            }
        )

        logger.info("Document splitting completed", job_id=job_id)
        return final_result

    except Exception as e:
        logger.error("Document splitting failed", job_id=job_id, error=str(e))
        raise

async def process_document_section(section_data: dict, section_index: int, options: dict) -> dict:
    """Process individual document section"""
    splitting_service = SplittingService()

    # Extract text and images for this section
    section_text = splitting_service.extract_section_text(section_data)
    section_images = splitting_service.extract_section_images(section_data)

    # Generate section summary
    summary = await splitting_service.generate_section_summary(
        section_text, options.get("summary_length", "short")
    )

    # Classify section type
    section_type = splitting_service.classify_section_type(section_text, section_images)

    return {
        "section_index": section_index,
        "section_type": section_type,
        "text": section_text,
        "images": section_images,
        "summary": summary,
        "word_count": len(section_text.split()),
        "page_range": section_data.get("page_range", []),
        "metadata": {
            "has_tables": splitting_service.has_tables(section_data),
            "has_images": len(section_images) > 0,
            "language": splitting_service.detect_language(section_text)
        }
    }
```

## 6. Batch Processing

### 6.1 Batch Document Processing

```python
# app/functions/batch_processing.py
from inngest import Context
from ..functions import inngest_client

@inngest_client.create_function(
    fn_id="batch-document-processing",
    trigger=inngest.TriggerEvent(event="batch.process"),
    retries=1,
    concurrency=1  # Limit concurrency for batch jobs
)
async def process_document_batch(ctx: Context) -> dict:
    """
    Process multiple documents in a batch

    Event data expected:
    - batch_id: str
    - document_urls: list
    - batch_options: dict
    """

    batch_id = ctx.event.data["batch_id"]
    document_urls = ctx.event.data["document_urls"]
    batch_options = ctx.event.data.get("batch_options", {})

    logger.info("Starting batch processing", batch_id=batch_id, document_count=len(document_urls))

    try:
        # Step 1: Initialize batch tracking
        batch_info = await ctx.step.run(
            "initialize-batch",
            lambda: initialize_batch_processing(batch_id, document_urls, batch_options)
        )

        # Step 2: Process documents in parallel (with concurrency limit)
        batch_size = batch_options.get("batch_size", 5)
        document_batches = [
            document_urls[i:i + batch_size]
            for i in range(0, len(document_urls), batch_size)
        ]

        all_results = []
        for batch_index, doc_batch in enumerate(document_batches):
            # Process each batch of documents
            batch_results = await ctx.step.parallel([
                (f"process-doc-{doc_index}", lambda url=doc_url: process_single_document_in_batch(
                    url, batch_id, batch_index, doc_index, batch_options
                ))
                for doc_index, doc_url in enumerate(doc_batch)
            ])

            all_results.extend(batch_results)

            # Optional: Add delay between batches to avoid overwhelming the system
            if batch_index < len(document_batches) - 1:
                await ctx.step.sleep("batch-delay", batch_options.get("batch_delay", 5))

        # Step 3: Aggregate results
        aggregated_results = await ctx.step.run(
            "aggregate-batch-results",
            lambda: aggregate_batch_results(all_results, batch_options)
        )

        # Step 4: Generate batch report
        batch_report = await ctx.step.run(
            "generate-batch-report",
            lambda: generate_batch_report(aggregated_results, batch_info)
        )

        # Step 5: Store batch results
        await ctx.step.run(
            "store-batch-results",
            lambda: store_batch_results(batch_id, aggregated_results, batch_report)
        )

        # Step 6: Emit completion event
        await ctx.step.send_event(
            "batch-completed",
            {
                "name": "batch.completed",
                "data": {
                    "batch_id": batch_id,
                    "results": aggregated_results,
                    "report": batch_report,
                    "status": "completed"
                }
            }
        )

        logger.info("Batch processing completed", batch_id=batch_id)
        return aggregated_results

    except Exception as e:
        logger.error("Batch processing failed", batch_id=batch_id, error=str(e))

        # Emit failure event
        await ctx.step.send_event(
            "batch-failed",
            {
                "name": "batch.failed",
                "data": {
                    "batch_id": batch_id,
                    "error": str(e),
                    "status": "failed"
                }
            }
        )

        raise

async def process_single_document_in_batch(
    document_url: str,
    batch_id: str,
    batch_index: int,
    doc_index: int,
    batch_options: dict
) -> dict:
    """Process a single document within a batch"""

    # Generate unique job ID for this document
    job_id = f"{batch_id}-{batch_index}-{doc_index}"

    try:
        # Trigger individual document processing
        # This reuses the main document processing function
        result = await process_document_internal(
            job_id=job_id,
            document_url=document_url,
            options=batch_options.get("document_options", {}),
            operation=batch_options.get("operation", "parse")
        )

        return {
            "job_id": job_id,
            "document_url": document_url,
            "status": "completed",
            "result": result,
            "batch_index": batch_index,
            "doc_index": doc_index
        }

    except Exception as e:
        logger.error("Document processing failed in batch",
                    job_id=job_id, error=str(e))

        return {
            "job_id": job_id,
            "document_url": document_url,
            "status": "failed",
            "error": str(e),
            "batch_index": batch_index,
            "doc_index": doc_index
        }
```

## 7. Error Handling Patterns

### 7.1 Comprehensive Error Handling

```python
# app/functions/error_handling.py
from inngest import Context
from ..functions import inngest_client
import traceback

@inngest_client.create_function(
    fn_id="robust-document-processor",
    trigger=inngest.TriggerEvent(event="document.process_robust"),
    retries=3,
    concurrency=5
)
async def robust_document_processor(ctx: Context) -> dict:
    """
    Document processor with comprehensive error handling
    """

    job_id = ctx.event.data["job_id"]

    try:
        # Step 1: Validate input with error handling
        validated_input = await ctx.step.run(
            "validate-input",
            lambda: validate_input_with_fallback(ctx.event.data),
            retries=2
        )

        # Step 2: Download document with retry logic
        document_info = await ctx.step.run(
            "download-document",
            lambda: download_document_with_retry(validated_input["document_url"], job_id),
            retries=3
        )

        # Step 3: Process with circuit breaker pattern
        processing_result = await ctx.step.run(
            "process-document",
            lambda: process_with_circuit_breaker(document_info, validated_input["options"]),
            retries=2
        )

        return processing_result

    except Exception as e:
        # Comprehensive error handling
        error_info = await ctx.step.run(
            "handle-error",
            lambda: handle_processing_error(e, job_id, ctx.event.data)
        )

        # Emit error event for monitoring
        await ctx.step.send_event(
            "error-occurred",
            {
                "name": "processing.error",
                "data": {
                    "job_id": job_id,
                    "error_type": error_info["error_type"],
                    "error_message": error_info["error_message"],
                    "retry_count": ctx.attempt.count,
                    "is_retryable": error_info["is_retryable"]
                }
            }
        )

        # Re-raise if not retryable or max retries exceeded
        if not error_info["is_retryable"] or ctx.attempt.count >= 3:
            raise

        # Return partial result if available
        return error_info.get("partial_result", {})

def handle_processing_error(error: Exception, job_id: str, event_data: dict) -> dict:
    """Categorize and handle different types of errors"""

    error_type = type(error).__name__
    error_message = str(error)
    stack_trace = traceback.format_exc()

    # Categorize error types
    if isinstance(error, (ConnectionError, TimeoutError)):
        error_category = "network"
        is_retryable = True
        recovery_action = "retry_with_backoff"
    elif isinstance(error, FileNotFoundError):
        error_category = "file_not_found"
        is_retryable = False
        recovery_action = "notify_user"
    elif isinstance(error, MemoryError):
        error_category = "resource"
        is_retryable = True
        recovery_action = "reduce_quality_and_retry"
    elif "rate limit" in error_message.lower():
        error_category = "rate_limit"
        is_retryable = True
        recovery_action = "exponential_backoff"
    else:
        error_category = "unknown"
        is_retryable = True
        recovery_action = "standard_retry"

    # Log error details
    logger.error(
        "Processing error occurred",
        job_id=job_id,
        error_type=error_type,
        error_category=error_category,
        error_message=error_message,
        stack_trace=stack_trace,
        is_retryable=is_retryable
    )

    return {
        "error_type": error_type,
        "error_category": error_category,
        "error_message": error_message,
        "is_retryable": is_retryable,
        "recovery_action": recovery_action,
        "stack_trace": stack_trace
    }

@inngest_client.create_function(
    fn_id="error-recovery-handler",
    trigger=inngest.TriggerEvent(event="processing.error"),
    retries=1
)
async def handle_error_recovery(ctx: Context) -> dict:
    """Handle error recovery based on error type"""

    error_data = ctx.event.data
    job_id = error_data["job_id"]
    error_type = error_data["error_type"]

    if error_data["retry_count"] >= 3:
        # Max retries exceeded - notify user and cleanup
        await ctx.step.run(
            "notify-failure",
            lambda: notify_user_of_failure(job_id, error_data)
        )

        await ctx.step.run(
            "cleanup-resources",
            lambda: cleanup_failed_job_resources(job_id)
        )

        return {"status": "failed", "action": "user_notified"}

    elif error_type == "rate_limit":
        # Implement exponential backoff
        delay = min(300, 2 ** error_data["retry_count"] * 10)  # Max 5 minutes

        await ctx.step.sleep("rate-limit-backoff", delay)

        # Retry the original request
        await ctx.step.send_event(
            "retry-processing",
            {
                "name": "document.process_robust",
                "data": {
                    **ctx.event.data,
                    "retry_attempt": error_data["retry_count"] + 1
                }
            }
        )

        return {"status": "retrying", "delay": delay}

    else:
        # Standard retry logic
        return {"status": "handled", "action": "standard_retry"}
```

## 8. Event Chaining

### 8.1 Complex Workflow Orchestration

```python
# app/functions/workflow_orchestration.py
from inngest import Context
from ..functions import inngest_client

@inngest_client.create_function(
    fn_id="document-workflow-orchestrator",
    trigger=inngest.TriggerEvent(event="workflow.start"),
    retries=1
)
async def orchestrate_document_workflow(ctx: Context) -> dict:
    """
    Orchestrate complex document processing workflow

    Event data expected:
    - workflow_id: str
    - documents: list
    - workflow_config: dict
    """

    workflow_id = ctx.event.data["workflow_id"]
    documents = ctx.event.data["documents"]
    workflow_config = ctx.event.data["workflow_config"]

    logger.info("Starting workflow orchestration", workflow_id=workflow_id)

    try:
        # Step 1: Initialize workflow
        workflow_state = await ctx.step.run(
            "initialize-workflow",
            lambda: initialize_workflow_state(workflow_id, documents, workflow_config)
        )

        # Step 2: Process documents in parallel
        document_results = await ctx.step.parallel([
            (f"process-doc-{i}", lambda doc=doc: trigger_document_processing(doc, workflow_id))
            for i, doc in enumerate(documents)
        ])

        # Step 3: Wait for all document processing to complete
        completed_documents = []
        for doc_id, _ in document_results:
            completion_event = await ctx.step.wait_for_event(
                f"wait-for-{doc_id}",
                event="document.processed",
                match="data.job_id",
                timeout="30m"
            )
            completed_documents.append(completion_event.data)

        # Step 4: Aggregate results based on workflow type
        if workflow_config["type"] == "merge":
            final_result = await ctx.step.run(
                "merge-documents",
                lambda: merge_document_results(completed_documents, workflow_config)
            )
        elif workflow_config["type"] == "compare":
            final_result = await ctx.step.run(
                "compare-documents",
                lambda: compare_document_results(completed_documents, workflow_config)
            )
        elif workflow_config["type"] == "extract_and_combine":
            final_result = await ctx.step.run(
                "extract-and-combine",
                lambda: extract_and_combine_data(completed_documents, workflow_config)
            )
        else:
            final_result = {"documents": completed_documents}

        # Step 5: Generate workflow report
        workflow_report = await ctx.step.run(
            "generate-workflow-report",
            lambda: generate_workflow_report(final_result, workflow_state)
        )

        # Step 6: Emit workflow completion
        await ctx.step.send_event(
            "workflow-completed",
            {
                "name": "workflow.completed",
                "data": {
                    "workflow_id": workflow_id,
                    "result": final_result,
                    "report": workflow_report,
                    "status": "completed"
                }
            }
        )

        logger.info("Workflow orchestration completed", workflow_id=workflow_id)
        return final_result

    except Exception as e:
        logger.error("Workflow orchestration failed", workflow_id=workflow_id, error=str(e))

        # Emit workflow failure
        await ctx.step.send_event(
            "workflow-failed",
            {
                "name": "workflow.failed",
                "data": {
                    "workflow_id": workflow_id,
                    "error": str(e),
                    "status": "failed"
                }
            }
        )

        raise

# Event-driven workflow continuation
@inngest_client.create_function(
    fn_id="workflow-step-processor",
    trigger=inngest.TriggerEvent(event="workflow.step"),
    retries=2
)
async def process_workflow_step(ctx: Context) -> dict:
    """Process individual workflow steps based on events"""

    step_data = ctx.event.data
    workflow_id = step_data["workflow_id"]
    step_type = step_data["step_type"]

    if step_type == "conditional_processing":
        # Process based on conditions
        condition_result = await ctx.step.run(
            "evaluate-condition",
            lambda: evaluate_workflow_condition(step_data["condition"], step_data["context"])
        )

        if condition_result["proceed"]:
            # Trigger next step
            await ctx.step.send_event(
                "trigger-next-step",
                {
                    "name": f"workflow.{condition_result['next_step']}",
                    "data": {
                        "workflow_id": workflow_id,
                        "previous_result": condition_result,
                        **step_data.get("next_step_data", {})
                    }
                }
            )

        return condition_result

    elif step_type == "data_transformation":
        # Transform data between steps
        transformed_data = await ctx.step.run(
            "transform-data",
            lambda: transform_workflow_data(step_data["input_data"], step_data["transformation_config"])
        )

        # Emit transformed data event
        await ctx.step.send_event(
            "data-transformed",
            {
                "name": "workflow.data_transformed",
                "data": {
                    "workflow_id": workflow_id,
                    "transformed_data": transformed_data,
                    "step_id": step_data["step_id"]
                }
            }
        )

        return transformed_data

    else:
        raise ValueError(f"Unknown workflow step type: {step_type}")
```

This comprehensive set of workflow examples demonstrates the power and flexibility of Inngest for building complex document processing pipelines with proper error handling, parallel processing, and event-driven orchestration.
```
```
