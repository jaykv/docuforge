# Docuforge - Technical Specification (Inngest)

## Overview

This document provides detailed technical specifications for the Inngest-based Docuforge implementation. It covers dependencies, configuration, code structure, and deployment requirements.

## 1. System Requirements

### 1.1 Minimum Hardware Requirements
- **CPU**: 8 cores (16 recommended for production)
- **RAM**: 16GB (32GB recommended for production)
- **Storage**: 100GB SSD (500GB+ for production)
- **GPU**: Optional NVIDIA GPU for ML acceleration (RTX 3080+ recommended)

### 1.2 Software Requirements
- **Python**: 3.9+ (3.11 recommended)
- **Node.js**: 18+ (for Inngest CLI)
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Kubernetes**: 1.24+ (for production)

## 2. Dependencies

### 2.1 Core Python Dependencies
```toml
[tool.poetry.dependencies]
python = "^3.9"

# Web Framework
fastapi = "^0.104.1"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"

# Inngest
inngest = "^0.3.0"

# Database
asyncpg = "^0.29.0"
sqlalchemy = {extras = ["asyncio"], version = "^2.0.23"}
alembic = "^1.13.0"

# Storage
boto3 = "^1.34.0"
minio = "^7.2.0"

# Cache
redis = {extras = ["hiredis"], version = "^5.0.1"}
aioredis = "^2.0.1"

# Vector Database
qdrant-client = "^1.7.0"

# Document Processing
opencv-python = "^4.8.1"
pillow = "^10.1.0"
pytesseract = "^0.3.10"
paddleocr = "^2.7.3"
easyocr = "^1.7.0"
pdf2image = "^1.16.3"
pymupdf = "^1.23.8"

# AI/ML
openai = "^1.3.7"
anthropic = "^0.7.7"
transformers = "^4.36.0"
torch = "^2.1.1"
torchvision = "^0.16.1"
sentence-transformers = "^2.2.2"
layoutlm = "^0.1.0"
ultralytics = "^8.0.206"

# NLP
spacy = "^3.7.2"
spacy-transformers = "^1.3.4"

# Utilities
httpx = "^0.25.2"
aiofiles = "^23.2.1"
python-multipart = "^0.0.6"
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
passlib = {extras = ["bcrypt"], version = "^1.7.4"}
celery = "^5.3.4"
structlog = "^23.2.0"
prometheus-client = "^0.19.0"
```

### 2.2 Development Dependencies
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-asyncio = "^0.21.1"
pytest-cov = "^4.1.0"
black = "^23.11.0"
isort = "^5.12.0"
flake8 = "^6.1.0"
mypy = "^1.7.1"
pre-commit = "^3.6.0"
httpx = "^0.25.2"
factory-boy = "^3.3.0"
```

### 2.3 System Dependencies
```dockerfile
# Ubuntu/Debian packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*
```

## 3. Configuration Management

### 3.1 Environment Variables
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Docuforge Clone"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Inngest
    INNGEST_EVENT_KEY: str
    INNGEST_SIGNING_KEY: str
    INNGEST_BASE_URL: str = "http://localhost:8288"
    INNGEST_DEV: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/docuforge_clone"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 20
    
    # Storage (MinIO/S3)
    STORAGE_ENDPOINT: str = "localhost:9000"
    STORAGE_ACCESS_KEY: str = "minioadmin"
    STORAGE_SECRET_KEY: str = "minioadmin"
    STORAGE_BUCKET: str = "documents"
    STORAGE_SECURE: bool = False
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "documents"
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Processing
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS: list = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    OCR_TIMEOUT: int = 300  # 5 minutes
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # Monitoring
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 3.2 Inngest Configuration
```python
from inngest import Inngest
from inngest.fast_api import serve

# Initialize Inngest client
inngest_client = Inngest(
    app_id="docuforge",
    event_key=settings.INNGEST_EVENT_KEY,
    signing_key=settings.INNGEST_SIGNING_KEY,
    base_url=settings.INNGEST_BASE_URL,
    is_production=not settings.INNGEST_DEV
)

# Function registration
from .functions import (
    process_document,
    extract_structured_data,
    split_document,
    edit_document,
    cancel_job
)

# Serve functions
app.mount("/api/inngest", serve(
    inngest_client,
    [
        process_document,
        extract_structured_data,
        split_document,
        edit_document,
        cancel_job
    ]
))
```

## 4. Database Schema

### 4.1 SQLAlchemy Models
```python
from sqlalchemy import Column, String, DateTime, JSON, Text, Integer, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    operation = Column(String, nullable=False)  # parse, extract, split, edit
    status = Column(String, default="processing")  # processing, completed, failed, cancelled
    document_url = Column(String, nullable=False)
    options = Column(JSON, default={})
    advanced_options = Column(JSON, default={})
    experimental_options = Column(JSON, default={})
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    duration = Column(Integer, nullable=True)  # seconds
    usage = Column(JSON, nullable=True)  # pages, credits, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ProcessingMetrics(Base):
    __tablename__ = "processing_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, nullable=False)
    step_name = Column(String, nullable=False)
    duration = Column(Float, nullable=False)  # seconds
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, default={})
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
```

### 4.2 Database Migrations
```python
# alembic/versions/001_initial_schema.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'jobs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('operation', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('document_url', sa.String(), nullable=False),
        sa.Column('options', sa.JSON(), nullable=True),
        sa.Column('advanced_options', sa.JSON(), nullable=True),
        sa.Column('experimental_options', sa.JSON(), nullable=True),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('usage', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index(op.f('ix_jobs_status'), 'jobs', ['status'], unique=False)
    op.create_index(op.f('ix_jobs_operation'), 'jobs', ['operation'], unique=False)
    op.create_index(op.f('ix_jobs_created_at'), 'jobs', ['created_at'], unique=False)
```

## 5. Project Structure

### 5.1 Directory Layout
```
docuforge/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration settings
│   ├── dependencies.py        # FastAPI dependencies
│   │
│   ├── api/                   # API endpoints
│   │   ├── __init__.py
│   │   ├── auth.py           # Authentication endpoints
│   │   ├── documents.py      # Document processing endpoints
│   │   ├── jobs.py           # Job management endpoints
│   │   └── webhooks.py       # Webhook configuration
│   │
│   ├── core/                 # Core business logic
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication logic
│   │   ├── storage.py       # File storage operations
│   │   ├── database.py      # Database operations
│   │   └── cache.py         # Redis cache operations
│   │
│   ├── functions/           # Inngest functions
│   │   ├── __init__.py
│   │   ├── document_processing.py  # Main processing pipeline
│   │   ├── ocr_functions.py        # OCR-specific functions
│   │   ├── extraction.py           # Data extraction functions
│   │   ├── splitting.py            # Document splitting
│   │   └── editing.py              # Document editing
│   │
│   ├── models/              # Database models
│   │   ├── __init__.py
│   │   ├── job.py          # Job model
│   │   ├── document.py     # Document model
│   │   └── metrics.py      # Metrics model
│   │
│   ├── schemas/             # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── requests.py     # Request schemas
│   │   ├── responses.py    # Response schemas
│   │   └── common.py       # Common schemas
│   │
│   ├── services/           # Business services
│   │   ├── __init__.py
│   │   ├── ocr_service.py         # OCR processing
│   │   ├── layout_service.py      # Layout analysis
│   │   ├── ai_service.py          # AI/ML operations
│   │   ├── vector_service.py      # Vector operations
│   │   └── webhook_service.py     # Webhook notifications
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── image_processing.py    # Image utilities
│       ├── pdf_processing.py     # PDF utilities
│       ├── text_processing.py    # Text utilities
│       └── monitoring.py         # Metrics and logging
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── conftest.py        # Test configuration
│   ├── test_api/          # API tests
│   ├── test_functions/    # Inngest function tests
│   ├── test_services/     # Service tests
│   └── test_utils/        # Utility tests
│
├── alembic/               # Database migrations
│   ├── versions/
│   ├── env.py
│   └── alembic.ini
│
├── docker/                # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
│
├── k8s/                   # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
│
├── scripts/               # Utility scripts
│   ├── setup.sh          # Environment setup
│   ├── migrate.py         # Database migration
│   └── seed_data.py       # Test data seeding
│
├── .env.example           # Environment variables template
├── .gitignore
├── README.md
├── pyproject.toml         # Poetry configuration
└── requirements.txt       # Pip requirements (generated)
```

### 5.2 Core Application Structure
```python
# app/main.py
from fastapi import FastAPI, Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .config import settings
from .api import auth, documents, jobs, webhooks
from .functions import inngest_client, serve_inngest
from .core.database import engine
from .utils.monitoring import setup_logging

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Setup monitoring
if not settings.DEBUG:
    Instrumentator().instrument(app).expose(app)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])

# Mount Inngest functions
app.mount("/api/inngest", serve_inngest())

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Initialize storage buckets
    from .core.storage import init_storage
    await init_storage()

    # Initialize vector database
    from .services.vector_service import init_vector_db
    await init_vector_db()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.APP_VERSION}
```

## 6. Inngest Function Implementation

### 6.1 Function Registration
```python
# app/functions/__init__.py
from inngest import Inngest
from inngest.fast_api import serve
from ..config import settings

# Initialize Inngest client
inngest_client = Inngest(
    app_id="docuforge",
    event_key=settings.INNGEST_EVENT_KEY,
    signing_key=settings.INNGEST_SIGNING_KEY,
    base_url=settings.INNGEST_BASE_URL,
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

def serve_inngest():
    """Create Inngest serve instance"""
    return serve(inngest_client, FUNCTIONS)
```

### 6.2 Error Handling and Monitoring
```python
# app/functions/base.py
import structlog
from typing import Any, Dict
from inngest import Context
from ..models.metrics import ProcessingMetrics
from ..core.database import get_db_session

logger = structlog.get_logger()

class FunctionBase:
    """Base class for Inngest functions with common functionality"""

    @staticmethod
    async def log_step_metrics(
        ctx: Context,
        step_name: str,
        duration: float,
        success: bool,
        error_message: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Log step execution metrics"""
        job_id = ctx.event.data.get("job_id")

        async with get_db_session() as session:
            metric = ProcessingMetrics(
                job_id=job_id,
                step_name=step_name,
                duration=duration,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )
            session.add(metric)
            await session.commit()

    @staticmethod
    async def handle_function_error(
        ctx: Context,
        error: Exception,
        step_name: str = "unknown"
    ):
        """Handle function errors with proper logging and metrics"""
        job_id = ctx.event.data.get("job_id")

        logger.error(
            "Function error",
            job_id=job_id,
            step_name=step_name,
            error=str(error),
            exc_info=True
        )

        # Update job status
        from ..core.database import update_job_status
        await update_job_status(job_id, "failed", str(error))

        # Emit failure event
        await ctx.step.send_event(
            "job-failed",
            {
                "name": "job.failed",
                "data": {
                    "job_id": job_id,
                    "step_name": step_name,
                    "error": str(error)
                }
            }
        )

        raise error
```

## 7. Testing Strategy

### 7.1 Test Configuration
```python
# tests/conftest.py
import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.config import settings
from app.core.database import get_db_session, Base

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test_docuforge_clone"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop tables
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

@pytest.fixture
async def client(test_session):
    """Create test client with database override"""
    app.dependency_overrides[get_db_session] = lambda: test_session

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()

@pytest.fixture
def mock_inngest_client():
    """Mock Inngest client for testing"""
    from unittest.mock import AsyncMock
    return AsyncMock()
```

### 7.2 API Tests
```python
# tests/test_api/test_documents.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_upload_document(client: AsyncClient):
    """Test document upload endpoint"""
    files = {"file": ("test.pdf", b"fake pdf content", "application/pdf")}
    headers = {"Authorization": "Bearer test-token"}

    response = await client.post("/api/v1/upload", files=files, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert "presigned_url" in data

@pytest.mark.asyncio
async def test_parse_document(client: AsyncClient):
    """Test document parsing endpoint"""
    payload = {
        "document_url": "https://example.com/test.pdf",
        "options": {
            "ocr_mode": "standard",
            "extraction_mode": "ocr"
        }
    }
    headers = {"Authorization": "Bearer test-token"}

    response = await client.post("/api/v1/parse", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "processing"
```

### 7.3 Function Tests
```python
# tests/test_functions/test_document_processing.py
import pytest
from unittest.mock import AsyncMock, patch
from inngest.testing import InngestTestEngine

from app.functions.document_processing import process_document

@pytest.mark.asyncio
async def test_process_document_success():
    """Test successful document processing"""
    test_engine = InngestTestEngine(process_document)

    event_data = {
        "job_id": "test-job-123",
        "document_url": "https://example.com/test.pdf",
        "options": {"ocr_mode": "standard"}
    }

    with patch("app.services.ocr_service.download_document") as mock_download:
        mock_download.return_value = {"path": "/tmp/test.pdf", "pages": 1}

        result = await test_engine.execute(
            event_name="document.uploaded",
            event_data=event_data
        )

        assert result.status == "completed"
        assert "result" in result.data

@pytest.mark.asyncio
async def test_process_document_failure():
    """Test document processing failure handling"""
    test_engine = InngestTestEngine(process_document)

    event_data = {
        "job_id": "test-job-456",
        "document_url": "https://invalid-url.com/test.pdf",
        "options": {}
    }

    with patch("app.services.ocr_service.download_document") as mock_download:
        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            await test_engine.execute(
                event_name="document.uploaded",
                event_data=event_data
            )
```

## 8. Monitoring and Observability

### 8.1 Metrics Collection
```python
# app/utils/monitoring.py
import time
import structlog
from prometheus_client import Counter, Histogram, Gauge
from functools import wraps

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

FUNCTION_DURATION = Histogram(
    'inngest_function_duration_seconds',
    'Inngest function execution duration',
    ['function_name', 'status']
)

ACTIVE_JOBS = Gauge(
    'active_jobs_total',
    'Number of active processing jobs',
    ['operation']
)

OCR_ACCURACY = Histogram(
    'ocr_accuracy_score',
    'OCR accuracy confidence scores',
    ['engine']
)

def setup_logging():
    """Configure structured logging"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def monitor_function_execution(func):
    """Decorator to monitor Inngest function execution"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        status = "success"

        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            FUNCTION_DURATION.labels(
                function_name=function_name,
                status=status
            ).observe(duration)

    return wrapper
```

### 8.2 Health Checks
```python
# app/api/health.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from redis import Redis

from ..core.database import get_db_session
from ..core.cache import get_redis_client
from ..services.vector_service import get_qdrant_client

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client)
):
    """Detailed health check with dependency status"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {}
    }

    # Check database
    try:
        await db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # Check Redis
    try:
        await redis.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # Check Qdrant
    try:
        client = get_qdrant_client()
        await client.get_collections()
        health_status["services"]["qdrant"] = "healthy"
    except Exception as e:
        health_status["services"]["qdrant"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status
```

## 9. Security Specifications

### 9.1 Authentication & Authorization
```python
# app/core/auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    return {"user_id": user_id, "scopes": payload.get("scopes", [])}
```

This technical specification provides comprehensive details for implementing the Inngest-based Docuforge, covering all aspects from dependencies and configuration to testing and security.
```
```
