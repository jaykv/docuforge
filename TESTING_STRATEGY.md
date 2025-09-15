# Docuforge - Testing Strategy (Inngest)

## Overview

This document outlines a comprehensive testing strategy for the Inngest-based Docuforge, covering unit tests, integration tests, function tests, and end-to-end testing. The strategy emphasizes testing Inngest functions, event-driven workflows, and API endpoints.

## Table of Contents

1. [Testing Philosophy](#1-testing-philosophy)
2. [Test Types and Structure](#2-test-types-and-structure)
3. [Inngest Function Testing](#3-inngest-function-testing)
4. [API Testing](#4-api-testing)
5. [Integration Testing](#5-integration-testing)
6. [Performance Testing](#6-performance-testing)
7. [Test Data Management](#7-test-data-management)
8. [CI/CD Testing Pipeline](#8-cicd-testing-pipeline)

## 1. Testing Philosophy

### 1.1 Testing Principles
- **Test Pyramid**: Focus on unit tests, fewer integration tests, minimal E2E tests
- **Event-Driven Testing**: Test event flows and function interactions
- **Isolation**: Each test should be independent and repeatable
- **Fast Feedback**: Tests should run quickly and provide clear failure messages
- **Real-World Scenarios**: Test with realistic document types and sizes

### 1.2 Test Coverage Goals
- **Unit Tests**: 90%+ code coverage
- **Function Tests**: 100% of Inngest functions tested
- **API Tests**: 100% of endpoints tested
- **Integration Tests**: All critical workflows tested
- **Performance Tests**: All processing functions benchmarked

## 2. Test Types and Structure

### 2.1 Directory Structure
```
tests/
├── unit/                    # Unit tests
│   ├── test_services/       # Service layer tests
│   ├── test_models/         # Database model tests
│   ├── test_utils/          # Utility function tests
│   └── test_core/           # Core functionality tests
├── functions/               # Inngest function tests
│   ├── test_document_processing.py
│   ├── test_ocr_functions.py
│   ├── test_extraction.py
│   └── test_workflow_orchestration.py
├── api/                     # API endpoint tests
│   ├── test_documents.py
│   ├── test_jobs.py
│   ├── test_auth.py
│   └── test_webhooks.py
├── integration/             # Integration tests
│   ├── test_full_workflows.py
│   ├── test_event_flows.py
│   └── test_database_operations.py
├── performance/             # Performance tests
│   ├── test_load_testing.py
│   ├── test_function_performance.py
│   └── test_concurrent_processing.py
├── fixtures/                # Test data and fixtures
│   ├── documents/           # Sample documents
│   ├── schemas/             # Test schemas
│   └── responses/           # Expected responses
└── conftest.py              # Pytest configuration
```

### 2.2 Test Configuration

**conftest.py:**
```python
import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from inngest.testing import InngestTestEngine

from app.main import app
from app.config import settings
from app.core.database import get_db_session, Base
from app.functions import inngest_client

# Test database configuration
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test_docuforge_clone"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def test_session(test_engine):
    """Create test database session"""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
async def client(test_session):
    """Create test HTTP client"""
    app.dependency_overrides[get_db_session] = lambda: test_session
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()

@pytest.fixture
def inngest_test_engine():
    """Create Inngest test engine"""
    return InngestTestEngine

@pytest.fixture
def sample_document():
    """Provide sample document for testing"""
    return {
        "url": "https://example.com/test.pdf",
        "filename": "test.pdf",
        "size": 1024000,
        "mime_type": "application/pdf"
    }

@pytest.fixture
def sample_job_data():
    """Provide sample job data for testing"""
    return {
        "job_id": "test-job-123",
        "operation": "parse",
        "options": {
            "ocr_mode": "standard",
            "extraction_mode": "ocr"
        }
    }
```

## 3. Inngest Function Testing

### 3.1 Basic Function Testing

**tests/functions/test_document_processing.py:**
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from inngest.testing import InngestTestEngine

from app.functions.document_processing import process_document

@pytest.mark.asyncio
async def test_process_document_success(inngest_test_engine, sample_job_data):
    """Test successful document processing"""
    test_engine = inngest_test_engine(process_document)
    
    event_data = {
        "job_id": sample_job_data["job_id"],
        "document_url": "https://example.com/test.pdf",
        "options": sample_job_data["options"],
        "operation": "parse"
    }
    
    # Mock external dependencies
    with patch("app.functions.document_processing.download_and_validate_document") as mock_download, \
         patch("app.functions.document_processing.preprocess_document") as mock_preprocess, \
         patch("app.functions.document_processing.analyze_document_layout") as mock_layout:
        
        # Setup mocks
        mock_download.return_value = {
            "local_path": "/tmp/test.pdf",
            "file_info": {"mime_type": "application/pdf", "size": 1024000},
            "job_id": sample_job_data["job_id"]
        }
        
        mock_preprocess.return_value = {
            "images": ["/tmp/page_1.png"],
            "num_pages": 1,
            "local_path": "/tmp/test.pdf"
        }
        
        mock_layout.return_value = {
            "layouts": [{"type": "document", "regions": []}],
            "document_type": "text"
        }
        
        # Execute function
        result = await test_engine.execute(
            event_name="document.uploaded",
            event_data=event_data
        )
        
        # Assertions
        assert result.status == "completed"
        assert "result" in result.data
        
        # Verify mocks were called
        mock_download.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_layout.assert_called_once()

@pytest.mark.asyncio
async def test_process_document_failure_handling(inngest_test_engine, sample_job_data):
    """Test document processing failure handling"""
    test_engine = inngest_test_engine(process_document)
    
    event_data = {
        "job_id": sample_job_data["job_id"],
        "document_url": "https://invalid-url.com/test.pdf",
        "options": sample_job_data["options"],
        "operation": "parse"
    }
    
    # Mock download failure
    with patch("app.functions.document_processing.download_and_validate_document") as mock_download:
        mock_download.side_effect = Exception("Download failed")
        
        # Execute function and expect failure
        with pytest.raises(Exception, match="Download failed"):
            await test_engine.execute(
                event_name="document.uploaded",
                event_data=event_data
            )

@pytest.mark.asyncio
async def test_process_document_with_events(inngest_test_engine, sample_job_data):
    """Test document processing with event emissions"""
    test_engine = inngest_test_engine(process_document)
    
    event_data = {
        "job_id": sample_job_data["job_id"],
        "document_url": "https://example.com/test.pdf",
        "options": sample_job_data["options"],
        "operation": "parse"
    }
    
    # Mock all dependencies
    with patch("app.functions.document_processing.download_and_validate_document"), \
         patch("app.functions.document_processing.preprocess_document"), \
         patch("app.functions.document_processing.analyze_document_layout"), \
         patch("app.functions.document_processing.store_processing_results"):
        
        result = await test_engine.execute(
            event_name="document.uploaded",
            event_data=event_data
        )
        
        # Check that events were emitted
        emitted_events = test_engine.get_emitted_events()
        
        # Should have emitted OCR trigger and completion events
        ocr_events = [e for e in emitted_events if e.name == "ocr.parallel_process"]
        completion_events = [e for e in emitted_events if e.name == "document.processed"]
        
        assert len(ocr_events) == 1
        assert len(completion_events) == 1
        assert completion_events[0].data["job_id"] == sample_job_data["job_id"]
```

### 3.2 Parallel Function Testing

**tests/functions/test_ocr_functions.py:**
```python
import pytest
from unittest.mock import patch, AsyncMock
from inngest.testing import InngestTestEngine

from app.functions.ocr_processing import parallel_ocr_processing

@pytest.mark.asyncio
async def test_parallel_ocr_processing(inngest_test_engine):
    """Test parallel OCR processing with multiple engines"""
    test_engine = inngest_test_engine(parallel_ocr_processing)
    
    event_data = {
        "job_id": "test-job-123",
        "document_info": {
            "images": ["/tmp/page_1.png", "/tmp/page_2.png"],
            "num_pages": 2
        },
        "layout_info": {
            "layouts": [{"type": "text"}, {"type": "text"}],
            "document_type": "document"
        },
        "options": {"ocr_engines": ["tesseract", "paddle", "easy"]}
    }
    
    # Mock OCR engines
    with patch("app.functions.ocr_processing.run_tesseract_ocr") as mock_tesseract, \
         patch("app.functions.ocr_processing.run_paddle_ocr") as mock_paddle, \
         patch("app.functions.ocr_processing.run_easy_ocr") as mock_easy, \
         patch("app.functions.ocr_processing.ensemble_ocr_results") as mock_ensemble:
        
        # Setup mock returns
        mock_tesseract.return_value = {
            "engine": "tesseract",
            "results": [{"text": "Page 1", "confidence": 0.9}],
            "confidence": 0.9
        }
        
        mock_paddle.return_value = {
            "engine": "paddle",
            "results": [{"text": "Page 1", "confidence": 0.85}],
            "confidence": 0.85
        }
        
        mock_easy.return_value = {
            "engine": "easy",
            "results": [{"text": "Page 1", "confidence": 0.8}],
            "confidence": 0.8
        }
        
        mock_ensemble.return_value = {
            "text": "Page 1",
            "confidence": 0.88,
            "pages": [{"text": "Page 1", "confidence": 0.88}]
        }
        
        # Execute function
        result = await test_engine.execute(
            event_name="ocr.parallel_process",
            event_data=event_data
        )
        
        # Assertions
        assert result.status == "completed"
        assert "ocr_result" in result.data
        assert "individual_results" in result.data
        
        # Verify all engines were called
        mock_tesseract.assert_called_once()
        mock_paddle.assert_called_once()
        mock_easy.assert_called_once()
        mock_ensemble.assert_called_once()

@pytest.mark.asyncio
async def test_ocr_engine_failure_resilience(inngest_test_engine):
    """Test OCR processing resilience when one engine fails"""
    test_engine = inngest_test_engine(parallel_ocr_processing)
    
    event_data = {
        "job_id": "test-job-123",
        "document_info": {"images": ["/tmp/page_1.png"]},
        "layout_info": {"layouts": [{"type": "text"}]},
        "options": {}
    }
    
    # Mock one engine failure
    with patch("app.functions.ocr_processing.run_tesseract_ocr") as mock_tesseract, \
         patch("app.functions.ocr_processing.run_paddle_ocr") as mock_paddle, \
         patch("app.functions.ocr_processing.run_easy_ocr") as mock_easy:
        
        # Tesseract fails, others succeed
        mock_tesseract.side_effect = Exception("Tesseract failed")
        mock_paddle.return_value = {"engine": "paddle", "results": [], "confidence": 0.8}
        mock_easy.return_value = {"engine": "easy", "results": [], "confidence": 0.75}
        
        # Should handle partial failure gracefully
        result = await test_engine.execute(
            event_name="ocr.parallel_process",
            event_data=event_data
        )
        
        # Function should still complete with available results
        assert result.status == "completed"
        assert "individual_results" in result.data
        
        # Should have results from working engines
        individual_results = result.data["individual_results"]
        assert "paddle" in individual_results
        assert "easy" in individual_results
```

## 4. API Testing

### 4.1 Document Upload Testing

**tests/api/test_documents.py:**
```python
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import io

@pytest.mark.asyncio
async def test_upload_document_success(client: AsyncClient):
    """Test successful document upload"""

    # Create test file
    test_file = io.BytesIO(b"fake pdf content")
    files = {"file": ("test.pdf", test_file, "application/pdf")}
    headers = {"Authorization": "Bearer test-token"}

    with patch("app.api.documents.generate_presigned_url") as mock_presigned:
        mock_presigned.return_value = "https://storage.example.com/upload-url"

        response = await client.post("/api/v1/upload", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "presigned_url" in data
        assert data["presigned_url"] == "https://storage.example.com/upload-url"

@pytest.mark.asyncio
async def test_upload_document_invalid_file_type(client: AsyncClient):
    """Test upload with invalid file type"""

    test_file = io.BytesIO(b"fake content")
    files = {"file": ("test.exe", test_file, "application/x-executable")}
    headers = {"Authorization": "Bearer test-token"}

    response = await client.post("/api/v1/upload", files=files, headers=headers)

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

@pytest.mark.asyncio
async def test_parse_document_success(client: AsyncClient):
    """Test document parsing endpoint"""

    payload = {
        "document_url": "https://example.com/test.pdf",
        "options": {
            "ocr_mode": "standard",
            "extraction_mode": "ocr"
        }
    }
    headers = {"Authorization": "Bearer test-token"}

    with patch("app.functions.inngest_client.send_event") as mock_send_event:
        mock_send_event.return_value = AsyncMock()

        response = await client.post("/api/v1/parse", json=payload, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "processing"

        # Verify event was sent
        mock_send_event.assert_called_once()
        call_args = mock_send_event.call_args[0][0]
        assert call_args["name"] == "document.uploaded"

@pytest.mark.asyncio
async def test_get_job_status(client: AsyncClient):
    """Test job status retrieval"""

    job_id = "test-job-123"
    headers = {"Authorization": "Bearer test-token"}

    with patch("app.services.job_service.get_job_by_id") as mock_get_job:
        mock_get_job.return_value = {
            "id": job_id,
            "status": "completed",
            "operation": "parse",
            "result": {"text": "Sample text"},
            "created_at": "2024-01-01T00:00:00Z"
        }

        response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["status"] == "completed"
        assert "result" in data

@pytest.mark.asyncio
async def test_extract_structured_data(client: AsyncClient):
    """Test structured data extraction endpoint"""

    payload = {
        "document_url": "https://example.com/invoice.pdf",
        "extraction_schema": {
            "fields": {
                "invoice_number": {"type": "string", "method": "regex"},
                "total_amount": {"type": "number", "method": "nlp"}
            }
        }
    }
    headers = {"Authorization": "Bearer test-token"}

    with patch("app.functions.inngest_client.send_event") as mock_send_event:
        response = await client.post("/api/v1/extract", json=payload, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "processing"

@pytest.mark.asyncio
async def test_authentication_required(client: AsyncClient):
    """Test that authentication is required for protected endpoints"""

    payload = {"document_url": "https://example.com/test.pdf"}

    # Request without authorization header
    response = await client.post("/api/v1/parse", json=payload)

    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]

@pytest.mark.asyncio
async def test_rate_limiting(client: AsyncClient):
    """Test API rate limiting"""

    headers = {"Authorization": "Bearer test-token"}
    payload = {"document_url": "https://example.com/test.pdf"}

    # Make multiple rapid requests
    responses = []
    for _ in range(10):
        response = await client.post("/api/v1/parse", json=payload, headers=headers)
        responses.append(response)

    # Should eventually hit rate limit
    rate_limited = any(r.status_code == 429 for r in responses)
    assert rate_limited, "Rate limiting should be triggered"
```

### 4.2 Webhook Testing

**tests/api/test_webhooks.py:**
```python
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_webhook_configuration(client: AsyncClient):
    """Test webhook configuration endpoint"""

    payload = {
        "url": "https://example.com/webhook",
        "events": ["document.processed", "document.failed"],
        "secret": "webhook-secret"
    }
    headers = {"Authorization": "Bearer test-token"}

    response = await client.post("/api/v1/webhooks", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["url"] == payload["url"]
    assert data["events"] == payload["events"]

@pytest.mark.asyncio
async def test_webhook_delivery(client: AsyncClient):
    """Test webhook delivery mechanism"""

    webhook_data = {
        "event": "document.processed",
        "data": {
            "job_id": "test-job-123",
            "status": "completed",
            "result": {"text": "Sample text"}
        }
    }

    with patch("app.services.webhook_service.deliver_webhook") as mock_deliver:
        mock_deliver.return_value = {"status": "delivered", "response_code": 200}

        # Simulate webhook delivery
        from app.services.webhook_service import WebhookService
        webhook_service = WebhookService()

        result = await webhook_service.deliver_webhook(
            "https://example.com/webhook",
            webhook_data,
            "webhook-secret"
        )

        assert result["status"] == "delivered"
        assert result["response_code"] == 200
```

## 5. Integration Testing

### 5.1 Full Workflow Testing

**tests/integration/test_full_workflows.py:**
```python
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import asyncio

@pytest.mark.asyncio
async def test_complete_document_processing_workflow(client: AsyncClient):
    """Test complete document processing from upload to completion"""

    # Step 1: Upload document
    test_file = io.BytesIO(b"fake pdf content")
    files = {"file": ("test.pdf", test_file, "application/pdf")}
    headers = {"Authorization": "Bearer test-token"}

    with patch("app.api.documents.generate_presigned_url") as mock_presigned:
        mock_presigned.return_value = "https://storage.example.com/upload-url"

        upload_response = await client.post("/api/v1/upload", files=files, headers=headers)
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

    # Step 2: Trigger parsing
    parse_payload = {
        "document_url": f"https://storage.example.com/{file_id}",
        "options": {"ocr_mode": "standard"}
    }

    with patch("app.functions.inngest_client.send_event") as mock_send_event:
        parse_response = await client.post("/api/v1/parse", json=parse_payload, headers=headers)
        assert parse_response.status_code == 200
        job_id = parse_response.json()["job_id"]

    # Step 3: Simulate processing completion
    with patch("app.services.job_service.get_job_by_id") as mock_get_job:
        # First call - processing
        mock_get_job.return_value = {
            "id": job_id,
            "status": "processing",
            "operation": "parse"
        }

        status_response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert status_response.json()["status"] == "processing"

        # Second call - completed
        mock_get_job.return_value = {
            "id": job_id,
            "status": "completed",
            "operation": "parse",
            "result": {
                "text": "Extracted text content",
                "confidence": 0.95,
                "pages": 1
            }
        }

        final_response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert final_response.json()["status"] == "completed"
        assert "result" in final_response.json()

@pytest.mark.asyncio
async def test_batch_processing_workflow(client: AsyncClient):
    """Test batch document processing workflow"""

    headers = {"Authorization": "Bearer test-token"}

    # Create batch processing request
    batch_payload = {
        "documents": [
            "https://example.com/doc1.pdf",
            "https://example.com/doc2.pdf",
            "https://example.com/doc3.pdf"
        ],
        "batch_options": {
            "operation": "parse",
            "batch_size": 2
        }
    }

    with patch("app.functions.inngest_client.send_event") as mock_send_event:
        response = await client.post("/api/v1/batch", json=batch_payload, headers=headers)

        assert response.status_code == 200
        batch_id = response.json()["batch_id"]

        # Verify batch event was sent
        mock_send_event.assert_called_once()
        call_args = mock_send_event.call_args[0][0]
        assert call_args["name"] == "batch.process"
        assert len(call_args["data"]["document_urls"]) == 3

@pytest.mark.asyncio
async def test_error_handling_workflow(client: AsyncClient):
    """Test error handling in complete workflow"""

    headers = {"Authorization": "Bearer test-token"}

    # Trigger processing with invalid URL
    payload = {
        "document_url": "https://invalid-domain.com/nonexistent.pdf",
        "options": {"ocr_mode": "standard"}
    }

    with patch("app.functions.inngest_client.send_event") as mock_send_event:
        response = await client.post("/api/v1/parse", json=payload, headers=headers)
        job_id = response.json()["job_id"]

    # Simulate error in processing
    with patch("app.services.job_service.get_job_by_id") as mock_get_job:
        mock_get_job.return_value = {
            "id": job_id,
            "status": "failed",
            "operation": "parse",
            "error": "Failed to download document"
        }

        error_response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert error_response.json()["status"] == "failed"
        assert "error" in error_response.json()
```

### 5.2 Event Flow Testing

**tests/integration/test_event_flows.py:**
```python
import pytest
from inngest.testing import InngestTestEngine
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_document_processing_event_flow():
    """Test complete event flow for document processing"""

    # Import all functions that participate in the flow
    from app.functions.document_processing import process_document
    from app.functions.ocr_processing import parallel_ocr_processing

    # Create test engines for each function
    doc_engine = InngestTestEngine(process_document)
    ocr_engine = InngestTestEngine(parallel_ocr_processing)

    # Mock external dependencies
    with patch("app.functions.document_processing.download_and_validate_document"), \
         patch("app.functions.document_processing.preprocess_document"), \
         patch("app.functions.document_processing.analyze_document_layout"), \
         patch("app.functions.ocr_processing.run_tesseract_ocr"), \
         patch("app.functions.ocr_processing.run_paddle_ocr"), \
         patch("app.functions.ocr_processing.run_easy_ocr"):

        # Step 1: Trigger main document processing
        doc_result = await doc_engine.execute(
            event_name="document.uploaded",
            event_data={
                "job_id": "test-job-123",
                "document_url": "https://example.com/test.pdf",
                "options": {"ocr_mode": "standard"},
                "operation": "parse"
            }
        )

        # Step 2: Check that OCR event was emitted
        emitted_events = doc_engine.get_emitted_events()
        ocr_events = [e for e in emitted_events if e.name == "ocr.parallel_process"]
        assert len(ocr_events) == 1

        # Step 3: Process OCR event
        ocr_result = await ocr_engine.execute(
            event_name="ocr.parallel_process",
            event_data=ocr_events[0].data
        )

        # Step 4: Check OCR completion event
        ocr_emitted = ocr_engine.get_emitted_events()
        completion_events = [e for e in ocr_emitted if e.name == "ocr.completed"]
        assert len(completion_events) == 1

        # Verify event data flow
        assert completion_events[0].data["job_id"] == "test-job-123"
        assert "result" in completion_events[0].data

@pytest.mark.asyncio
async def test_error_event_propagation():
    """Test error event propagation through the system"""

    from app.functions.document_processing import process_document

    doc_engine = InngestTestEngine(process_document)

    # Mock download failure
    with patch("app.functions.document_processing.download_and_validate_document") as mock_download:
        mock_download.side_effect = Exception("Network error")

        # Execute and expect failure
        with pytest.raises(Exception):
            await doc_engine.execute(
                event_name="document.uploaded",
                event_data={
                    "job_id": "test-job-123",
                    "document_url": "https://invalid.com/test.pdf",
                    "options": {},
                    "operation": "parse"
                }
            )

        # Check error events were emitted
        emitted_events = doc_engine.get_emitted_events()
        error_events = [e for e in emitted_events if e.name == "document.failed"]
        assert len(error_events) == 1
        assert "error" in error_events[0].data
```

## 6. Performance Testing

### 6.1 Function Performance Testing

**tests/performance/test_function_performance.py:**
```python
import pytest
import time
import asyncio
from unittest.mock import patch
from inngest.testing import InngestTestEngine

@pytest.mark.asyncio
async def test_document_processing_performance():
    """Test document processing performance benchmarks"""

    from app.functions.document_processing import process_document

    test_engine = InngestTestEngine(process_document)

    # Mock fast responses
    with patch("app.functions.document_processing.download_and_validate_document"), \
         patch("app.functions.document_processing.preprocess_document"), \
         patch("app.functions.document_processing.analyze_document_layout"):

        start_time = time.time()

        result = await test_engine.execute(
            event_name="document.uploaded",
            event_data={
                "job_id": "perf-test-123",
                "document_url": "https://example.com/test.pdf",
                "options": {"ocr_mode": "standard"},
                "operation": "parse"
            }
        )

        execution_time = time.time() - start_time

        # Performance assertions
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert result.status == "completed"

@pytest.mark.asyncio
async def test_concurrent_function_execution():
    """Test concurrent function execution performance"""

    from app.functions.document_processing import process_document

    test_engine = InngestTestEngine(process_document)

    # Mock dependencies
    with patch("app.functions.document_processing.download_and_validate_document"), \
         patch("app.functions.document_processing.preprocess_document"), \
         patch("app.functions.document_processing.analyze_document_layout"):

        # Create multiple concurrent executions
        tasks = []
        for i in range(10):
            task = test_engine.execute(
                event_name="document.uploaded",
                event_data={
                    "job_id": f"concurrent-test-{i}",
                    "document_url": f"https://example.com/test{i}.pdf",
                    "options": {"ocr_mode": "standard"},
                    "operation": "parse"
                }
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time

        # All should complete successfully
        assert all(r.status == "completed" for r in results)

        # Concurrent execution should be faster than sequential
        assert execution_time < 10.0  # Should complete within 10 seconds

@pytest.mark.asyncio
async def test_memory_usage_monitoring():
    """Test memory usage during function execution"""

    import psutil
    import os

    from app.functions.document_processing import process_document

    test_engine = InngestTestEngine(process_document)

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    with patch("app.functions.document_processing.download_and_validate_document"), \
         patch("app.functions.document_processing.preprocess_document"), \
         patch("app.functions.document_processing.analyze_document_layout"):

        await test_engine.execute(
            event_name="document.uploaded",
            event_data={
                "job_id": "memory-test-123",
                "document_url": "https://example.com/test.pdf",
                "options": {"ocr_mode": "standard"},
                "operation": "parse"
            }
        )

        # Check memory usage after execution
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for mocked execution)
        assert memory_increase < 100, f"Memory increased by {memory_increase}MB"
```

## 7. Test Data Management

### 7.1 Test Fixtures and Data

**tests/fixtures/document_fixtures.py:**
```python
import pytest
import os
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "documents"

@pytest.fixture
def sample_pdf_path():
    """Path to sample PDF document"""
    return TEST_DATA_DIR / "sample.pdf"

@pytest.fixture
def sample_image_path():
    """Path to sample image document"""
    return TEST_DATA_DIR / "sample.png"

@pytest.fixture
def sample_invoice_pdf():
    """Path to sample invoice PDF"""
    return TEST_DATA_DIR / "invoice.pdf"

@pytest.fixture
def sample_multi_page_pdf():
    """Path to multi-page PDF document"""
    return TEST_DATA_DIR / "multi_page.pdf"

@pytest.fixture
def sample_extraction_schema():
    """Sample extraction schema for testing"""
    return {
        "version": "1.0",
        "fields": {
            "invoice_number": {
                "type": "string",
                "method": "regex",
                "pattern": r"Invoice #?(\d+)",
                "required": True
            },
            "total_amount": {
                "type": "number",
                "method": "nlp",
                "description": "Total amount due",
                "required": True
            },
            "due_date": {
                "type": "date",
                "method": "regex",
                "pattern": r"Due Date:?\s*(\d{1,2}/\d{1,2}/\d{4})",
                "required": False
            },
            "vendor_name": {
                "type": "string",
                "method": "ai",
                "description": "Name of the vendor/company",
                "required": True
            }
        }
    }

@pytest.fixture
def sample_ocr_result():
    """Sample OCR result for testing"""
    return {
        "text": "Sample document text content",
        "confidence": 0.95,
        "pages": [
            {
                "page_number": 1,
                "text": "Sample document text content",
                "confidence": 0.95,
                "words": [
                    {"text": "Sample", "confidence": 0.98, "bbox": [10, 10, 50, 30]},
                    {"text": "document", "confidence": 0.96, "bbox": [55, 10, 120, 30]},
                    {"text": "text", "confidence": 0.94, "bbox": [125, 10, 160, 30]},
                    {"text": "content", "confidence": 0.92, "bbox": [165, 10, 220, 30]}
                ]
            }
        ],
        "metadata": {
            "engine": "tesseract",
            "processing_time": 2.5,
            "language": "en"
        }
    }

@pytest.fixture
def sample_layout_analysis():
    """Sample layout analysis result"""
    return {
        "layouts": [
            {
                "page_number": 1,
                "regions": [
                    {
                        "type": "text",
                        "bbox": [10, 10, 500, 200],
                        "confidence": 0.95
                    },
                    {
                        "type": "table",
                        "bbox": [10, 220, 500, 400],
                        "confidence": 0.88
                    }
                ]
            }
        ],
        "document_type": "invoice",
        "confidence": 0.92
    }
```

### 7.2 Database Test Data

**tests/fixtures/database_fixtures.py:**
```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.job import Job
from app.models.document import Document
import uuid
from datetime import datetime

@pytest.fixture
async def sample_job(test_session: AsyncSession):
    """Create sample job in test database"""
    job = Job(
        id=str(uuid.uuid4()),
        operation="parse",
        status="processing",
        document_url="https://example.com/test.pdf",
        options={"ocr_mode": "standard"},
        created_at=datetime.utcnow()
    )

    test_session.add(job)
    await test_session.commit()
    await test_session.refresh(job)

    return job

@pytest.fixture
async def completed_job(test_session: AsyncSession):
    """Create completed job with results"""
    job = Job(
        id=str(uuid.uuid4()),
        operation="parse",
        status="completed",
        document_url="https://example.com/test.pdf",
        options={"ocr_mode": "standard"},
        result={
            "text": "Sample extracted text",
            "confidence": 0.95,
            "pages": 1
        },
        created_at=datetime.utcnow()
    )

    test_session.add(job)
    await test_session.commit()
    await test_session.refresh(job)

    return job

@pytest.fixture
async def sample_document(test_session: AsyncSession):
    """Create sample document record"""
    document = Document(
        id=str(uuid.uuid4()),
        filename="test.pdf",
        file_size=1024000,
        mime_type="application/pdf",
        storage_path="documents/test.pdf",
        processed=True,
        created_at=datetime.utcnow()
    )

    test_session.add(document)
    await test_session.commit()
    await test_session.refresh(document)

    return document
```

### 7.3 Mock Data Generators

**tests/utils/mock_generators.py:**
```python
import random
import string
from faker import Faker
from typing import Dict, List, Any

fake = Faker()

def generate_mock_document_url() -> str:
    """Generate mock document URL"""
    return f"https://storage.example.com/{fake.uuid4()}.pdf"

def generate_mock_job_data(operation: str = "parse") -> Dict[str, Any]:
    """Generate mock job data"""
    return {
        "job_id": fake.uuid4(),
        "operation": operation,
        "document_url": generate_mock_document_url(),
        "options": {
            "ocr_mode": random.choice(["standard", "agentic"]),
            "extraction_mode": random.choice(["ocr", "hybrid"]),
            "language": random.choice(["en", "es", "fr", "de"])
        }
    }

def generate_mock_ocr_result(num_pages: int = 1) -> Dict[str, Any]:
    """Generate mock OCR result"""
    pages = []
    full_text = ""

    for page_num in range(1, num_pages + 1):
        page_text = fake.text(max_nb_chars=1000)
        full_text += page_text + "\n"

        words = []
        for word in page_text.split()[:20]:  # Limit words for testing
            words.append({
                "text": word,
                "confidence": round(random.uniform(0.8, 1.0), 2),
                "bbox": [
                    random.randint(10, 100),
                    random.randint(10, 100),
                    random.randint(150, 250),
                    random.randint(150, 250)
                ]
            })

        pages.append({
            "page_number": page_num,
            "text": page_text,
            "confidence": round(random.uniform(0.85, 0.98), 2),
            "words": words
        })

    return {
        "text": full_text.strip(),
        "confidence": round(random.uniform(0.85, 0.98), 2),
        "pages": pages,
        "metadata": {
            "engine": random.choice(["tesseract", "paddle", "easy"]),
            "processing_time": round(random.uniform(1.0, 10.0), 2),
            "language": "en"
        }
    }

def generate_mock_extraction_result(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Generate mock extraction result based on schema"""
    extracted_data = {}
    confidence_scores = {}

    for field_name, field_config in schema["fields"].items():
        field_type = field_config["type"]

        if field_type == "string":
            extracted_data[field_name] = fake.company() if "vendor" in field_name else fake.word()
        elif field_type == "number":
            extracted_data[field_name] = round(random.uniform(10.0, 1000.0), 2)
        elif field_type == "date":
            extracted_data[field_name] = fake.date().isoformat()
        else:
            extracted_data[field_name] = fake.text(max_nb_chars=50)

        confidence_scores[field_name] = round(random.uniform(0.7, 0.95), 2)

    return {
        "extracted_data": extracted_data,
        "confidence_scores": confidence_scores,
        "overall_confidence": round(sum(confidence_scores.values()) / len(confidence_scores), 2)
    }
```

## 8. CI/CD Testing Pipeline

### 8.1 GitHub Actions Workflow

**.github/workflows/test.yml:**
```yaml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_USER: test
          POSTGRES_DB: test_docuforge_clone
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: latest
        virtualenvs-create: true
        virtualenvs-in-project: true

    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ hashFiles('**/poetry.lock') }}

    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --no-interaction --no-root

    - name: Install project
      run: poetry install --no-interaction

    - name: Run linting
      run: |
        poetry run black --check app/ tests/
        poetry run isort --check-only app/ tests/
        poetry run flake8 app/ tests/
        poetry run mypy app/

    - name: Run unit tests
      run: |
        poetry run pytest tests/unit/ -v --cov=app --cov-report=xml
      env:
        DATABASE_URL: postgresql+asyncpg://test:test@localhost:5432/test_docuforge_clone
        REDIS_URL: redis://localhost:6379/0

    - name: Run function tests
      run: |
        poetry run pytest tests/functions/ -v
      env:
        DATABASE_URL: postgresql+asyncpg://test:test@localhost:5432/test_docuforge_clone
        REDIS_URL: redis://localhost:6379/0

    - name: Run API tests
      run: |
        poetry run pytest tests/api/ -v
      env:
        DATABASE_URL: postgresql+asyncpg://test:test@localhost:5432/test_docuforge_clone
        REDIS_URL: redis://localhost:6379/0

    - name: Run integration tests
      run: |
        poetry run pytest tests/integration/ -v
      env:
        DATABASE_URL: postgresql+asyncpg://test:test@localhost:5432/test_docuforge_clone
        REDIS_URL: redis://localhost:6379/0

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  performance-test:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install poetry
        poetry install

    - name: Run performance tests
      run: |
        poetry run pytest tests/performance/ -v --benchmark-only
      env:
        DATABASE_URL: postgresql+asyncpg://test:test@localhost:5432/test_docuforge_clone
```

### 8.2 Test Reporting and Metrics

**scripts/test_reporting.py:**
```python
#!/usr/bin/env python3
"""
Test reporting and metrics collection script
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_tests_with_metrics():
    """Run tests and collect metrics"""

    # Run tests with coverage and timing
    result = subprocess.run([
        "poetry", "run", "pytest",
        "--cov=app",
        "--cov-report=json",
        "--cov-report=html",
        "--durations=10",
        "--json-report",
        "--json-report-file=test-report.json",
        "-v"
    ], capture_output=True, text=True)

    # Load test results
    with open("test-report.json") as f:
        test_data = json.load(f)

    # Load coverage data
    with open("coverage.json") as f:
        coverage_data = json.load(f)

    # Generate summary report
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_summary": {
            "total_tests": test_data["summary"]["total"],
            "passed": test_data["summary"]["passed"],
            "failed": test_data["summary"]["failed"],
            "skipped": test_data["summary"]["skipped"],
            "duration": test_data["duration"]
        },
        "coverage_summary": {
            "total_coverage": coverage_data["totals"]["percent_covered"],
            "lines_covered": coverage_data["totals"]["covered_lines"],
            "lines_missing": coverage_data["totals"]["missing_lines"]
        },
        "slowest_tests": [
            {
                "name": test["nodeid"],
                "duration": test["duration"]
            }
            for test in sorted(test_data["tests"], key=lambda x: x["duration"], reverse=True)[:5]
        ]
    }

    # Save summary
    with open("test-summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Test Summary:")
    print(f"  Total Tests: {summary['test_summary']['total_tests']}")
    print(f"  Passed: {summary['test_summary']['passed']}")
    print(f"  Failed: {summary['test_summary']['failed']}")
    print(f"  Coverage: {summary['coverage_summary']['total_coverage']:.1f}%")
    print(f"  Duration: {summary['test_summary']['duration']:.2f}s")

    return result.returncode == 0

if __name__ == "__main__":
    success = run_tests_with_metrics()
    sys.exit(0 if success else 1)
```

This comprehensive testing strategy ensures robust testing of the Inngest-based Docuforge across all layers, from individual functions to complete workflows, with proper CI/CD integration and performance monitoring.
```

This comprehensive testing strategy ensures robust testing of all aspects of the Inngest-based Docuforge, from individual functions to complete workflows and performance characteristics.
```
