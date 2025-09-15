# Docuforge - Local Development Guide (Inngest)

## Overview

This guide provides step-by-step instructions for setting up a local development environment for the Inngest-based Docuforge. The setup leverages Inngest Dev Server for function development and Docker Compose for supporting services.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start](#2-quick-start)
3. [Development Environment Setup](#3-development-environment-setup)
4. [Inngest Development Workflow](#4-inngest-development-workflow)
5. [Testing and Debugging](#5-testing-and-debugging)
6. [Common Development Tasks](#6-common-development-tasks)

## 1. Prerequisites

### 1.1 System Requirements
- **OS**: macOS, Linux, or Windows with WSL2
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space
- **CPU**: 4 cores minimum

### 1.2 Required Software
```bash
# Install Node.js (for Inngest CLI)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python 3.9+
sudo apt-get install python3.9 python3.9-venv python3.9-dev

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Poetry (Python dependency management)
curl -sSL https://install.python-poetry.org | python3 -

# Install Inngest CLI
npm install -g inngest-cli
```

### 1.3 IDE Setup (VS Code Recommended)
```bash
# Install VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension ms-python.flake8
code --install-extension bradlc.vscode-tailwindcss
code --install-extension ms-vscode.vscode-json
```

## 2. Quick Start

### 2.1 Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd docuforge

# Copy environment template
cp .env.example .env

# Install Python dependencies
poetry install

# Start supporting services
docker-compose up -d postgres redis minio qdrant

# Run database migrations
poetry run alembic upgrade head

# Start Inngest Dev Server (in separate terminal)
inngest-cli dev

# Start the API server
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2.2 Verify Installation
```bash
# Check API health
curl http://localhost:8000/health

# Check Inngest dashboard
open http://localhost:8288

# Check API documentation
open http://localhost:8000/docs

# Run a simple test
poetry run pytest tests/test_api/test_health.py -v
```

## 3. Development Environment Setup

### 3.1 Environment Configuration

**Create .env file:**
```bash
# Application Settings
APP_NAME=Docuforge Clone Dev
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production

# Inngest Configuration
INNGEST_EVENT_KEY=dev-event-key
INNGEST_SIGNING_KEY=dev-signing-key
INNGEST_DEV=true
INNGEST_BASE_URL=http://localhost:8288

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/docuforge_clone_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# MinIO (Local S3)
STORAGE_ENDPOINT=localhost:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin
STORAGE_BUCKET=documents-dev
STORAGE_SECURE=false

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333

# AI Services (Optional for development)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Development Settings
LOG_LEVEL=DEBUG
MAX_FILE_SIZE=52428800  # 50MB for development
OCR_TIMEOUT=120  # 2 minutes for development
```

### 3.2 Docker Compose for Development

**docker-compose.dev.yml:**
```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: docuforge_clone_dev
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./scripts/init-dev-db.sql:/docker-entrypoint-initdb.d/init-dev-db.sql
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_dev_data:/data
    restart: unless-stopped

  # MinIO Object Storage
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_dev_data:/data
    restart: unless-stopped

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_dev_data:/qdrant/storage
    restart: unless-stopped

  # Mailhog (Email testing)
  mailhog:
    image: mailhog/mailhog:latest
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI
    restart: unless-stopped

volumes:
  postgres_dev_data:
  redis_dev_data:
  minio_dev_data:
  qdrant_dev_data:
```

### 3.3 Python Environment Setup

**pyproject.toml configuration:**
```toml
[tool.poetry]
name = "docuforge"
version = "0.1.0"
description = "Docuforge with Inngest"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
fastapi = "^0.104.1"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
inngest = "^0.3.0"
# ... other dependencies from technical spec

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

[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### 3.4 Pre-commit Hooks

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## 4. Inngest Development Workflow

### 4.1 Starting Development Environment

**Terminal 1 - Supporting Services:**
```bash
# Start databases and storage
docker-compose -f docker-compose.dev.yml up -d

# Check service status
docker-compose -f docker-compose.dev.yml ps
```

**Terminal 2 - Inngest Dev Server:**
```bash
# Start Inngest development server
inngest-cli dev

# The dev server will:
# - Start on http://localhost:8288
# - Provide a web UI for function monitoring
# - Auto-reload functions when code changes
# - Show real-time function execution logs
```

**Terminal 3 - API Server:**
```bash
# Activate virtual environment
poetry shell

# Start FastAPI with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the development script
python scripts/dev_server.py
```

### 4.2 Function Development

**Creating a new Inngest function:**
```python
# app/functions/example_function.py
from inngest import Context
from ..functions import inngest_client

@inngest_client.create_function(
    fn_id="example-processing",
    trigger=inngest.TriggerEvent(event="example.process"),
    retries=2
)
async def process_example(ctx: Context) -> dict:
    """Example processing function"""
    data = ctx.event.data
    
    # Step 1: Validate input
    validated_data = await ctx.step.run(
        "validate-input",
        lambda: validate_example_data(data)
    )
    
    # Step 2: Process data
    result = await ctx.step.run(
        "process-data",
        lambda: process_example_data(validated_data)
    )
    
    # Step 3: Store result
    await ctx.step.run(
        "store-result",
        lambda: store_example_result(result)
    )
    
    return result

def validate_example_data(data: dict) -> dict:
    """Validate input data"""
    # Add validation logic
    return data

def process_example_data(data: dict) -> dict:
    """Process the data"""
    # Add processing logic
    return {"processed": True, "data": data}

def store_example_result(result: dict) -> None:
    """Store the result"""
    # Add storage logic
    pass
```

**Testing the function:**
```python
# tests/test_functions/test_example_function.py
import pytest
from inngest.testing import InngestTestEngine
from app.functions.example_function import process_example

@pytest.mark.asyncio
async def test_process_example():
    """Test example processing function"""
    test_engine = InngestTestEngine(process_example)
    
    event_data = {
        "input": "test data",
        "options": {"validate": True}
    }
    
    result = await test_engine.execute(
        event_name="example.process",
        event_data=event_data
    )
    
    assert result.status == "completed"
    assert result.data["processed"] is True
```

### 4.3 Triggering Functions

**From API endpoint:**
```python
# app/api/example.py
from fastapi import APIRouter
from ..functions import inngest_client

router = APIRouter()

@router.post("/trigger-example")
async def trigger_example_processing(data: dict):
    """Trigger example processing"""
    
    # Send event to Inngest
    await inngest_client.send_event({
        "name": "example.process",
        "data": data
    })
    
    return {"status": "triggered", "message": "Processing started"}
```

**From command line (for testing):**
```bash
# Using curl to trigger function
curl -X POST http://localhost:8000/api/v1/trigger-example \
  -H "Content-Type: application/json" \
  -d '{"input": "test data", "options": {"validate": true}}'

# Using httpie (if installed)
http POST localhost:8000/api/v1/trigger-example input="test data" options:='{"validate": true}'
```

## 5. Testing and Debugging

### 5.1 Running Tests

**Run all tests:**
```bash
# Run full test suite
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_functions/test_document_processing.py -v

# Run tests matching pattern
poetry run pytest -k "test_upload" -v

# Run tests with debugging output
poetry run pytest -s -v
```

**Test configuration:**
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --asyncio-mode=auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### 5.2 Debugging Inngest Functions

**Using Inngest Dev Server UI:**
1. Open http://localhost:8288 in browser
2. Navigate to "Functions" tab to see all registered functions
3. Click on a function to see execution history
4. View step-by-step execution details
5. Check logs and error messages

**Adding debug logging:**
```python
# app/functions/debug_example.py
import structlog
from inngest import Context

logger = structlog.get_logger()

@inngest_client.create_function(
    fn_id="debug-example",
    trigger=inngest.TriggerEvent(event="debug.test")
)
async def debug_example(ctx: Context) -> dict:
    """Example function with debug logging"""

    logger.info("Function started", event_data=ctx.event.data)

    try:
        # Step with detailed logging
        result = await ctx.step.run(
            "process-step",
            lambda: process_with_logging(ctx.event.data)
        )

        logger.info("Function completed", result=result)
        return result

    except Exception as e:
        logger.error("Function failed", error=str(e), exc_info=True)
        raise

def process_with_logging(data: dict) -> dict:
    """Process data with detailed logging"""
    logger.debug("Processing data", input_data=data)

    # Simulate processing
    result = {"processed": True, "input": data}

    logger.debug("Processing complete", output_data=result)
    return result
```

**Using debugger with functions:**
```python
# For debugging, you can use pdb or IDE breakpoints
import pdb

@inngest_client.create_function(
    fn_id="debug-with-breakpoint",
    trigger=inngest.TriggerEvent(event="debug.breakpoint")
)
async def debug_with_breakpoint(ctx: Context) -> dict:
    """Function with debugger breakpoint"""

    data = ctx.event.data

    # Set breakpoint for debugging
    pdb.set_trace()  # Remove in production!

    result = await ctx.step.run(
        "debug-step",
        lambda: {"debug": True, "data": data}
    )

    return result
```

### 5.3 Database Debugging

**Database inspection:**
```bash
# Connect to development database
poetry run python -c "
from app.core.database import engine
import asyncio
async def test_db():
    async with engine.begin() as conn:
        result = await conn.execute('SELECT version()')
        print(result.fetchone())
asyncio.run(test_db())
"

# Run database migrations
poetry run alembic upgrade head

# Create new migration
poetry run alembic revision --autogenerate -m "Add new table"

# Check migration status
poetry run alembic current
poetry run alembic history
```

**Redis debugging:**
```bash
# Connect to Redis CLI
docker exec -it docuforge-redis-1 redis-cli

# Check Redis keys
KEYS *

# Monitor Redis commands
MONITOR

# Check memory usage
INFO memory
```

## 6. Common Development Tasks

### 6.1 Adding New API Endpoints

**Create new router:**
```python
# app/api/new_feature.py
from fastapi import APIRouter, Depends, HTTPException
from ..schemas.requests import NewFeatureRequest
from ..schemas.responses import NewFeatureResponse
from ..functions import inngest_client

router = APIRouter()

@router.post("/new-feature", response_model=NewFeatureResponse)
async def create_new_feature(
    request: NewFeatureRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create new feature"""

    # Validate request
    if not request.data:
        raise HTTPException(status_code=400, detail="Data is required")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Trigger Inngest function
    await inngest_client.send_event({
        "name": "new_feature.process",
        "data": {
            "job_id": job_id,
            "user_id": current_user["user_id"],
            "request_data": request.dict()
        }
    })

    return NewFeatureResponse(
        job_id=job_id,
        status="processing",
        message="Feature processing started"
    )
```

**Register router in main app:**
```python
# app/main.py
from .api import new_feature

app.include_router(
    new_feature.router,
    prefix="/api/v1",
    tags=["new-feature"]
)
```

### 6.2 Database Operations

**Create new model:**
```python
# app/models/new_model.py
from sqlalchemy import Column, String, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from .base import Base
import uuid

class NewModel(Base):
    __tablename__ = "new_models"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    data = Column(JSON, default={})
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

**Create database service:**
```python
# app/services/new_model_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..models.new_model import NewModel

class NewModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, data: dict) -> NewModel:
        """Create new model instance"""
        model = NewModel(name=name, data=data)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def get_by_id(self, model_id: str) -> NewModel:
        """Get model by ID"""
        result = await self.session.execute(
            select(NewModel).where(NewModel.id == model_id)
        )
        return result.scalar_one_or_none()

    async def list_active(self) -> list[NewModel]:
        """List all active models"""
        result = await self.session.execute(
            select(NewModel).where(NewModel.active == True)
        )
        return result.scalars().all()
```

### 6.3 Adding New Inngest Functions

**Function template:**
```python
# app/functions/new_function.py
from inngest import Context
from ..functions import inngest_client
from ..services.new_model_service import NewModelService
from ..core.database import get_db_session

@inngest_client.create_function(
    fn_id="new-feature-processor",
    trigger=inngest.TriggerEvent(event="new_feature.process"),
    retries=3
)
async def process_new_feature(ctx: Context) -> dict:
    """Process new feature request"""

    job_id = ctx.event.data["job_id"]
    user_id = ctx.event.data["user_id"]
    request_data = ctx.event.data["request_data"]

    try:
        # Step 1: Validate input
        validated_data = await ctx.step.run(
            "validate-input",
            lambda: validate_new_feature_input(request_data)
        )

        # Step 2: Process feature
        result = await ctx.step.run(
            "process-feature",
            lambda: process_feature_logic(validated_data)
        )

        # Step 3: Store result
        await ctx.step.run(
            "store-result",
            lambda: store_feature_result(job_id, user_id, result)
        )

        # Step 4: Send notification
        await ctx.step.send_event(
            "feature-completed",
            {
                "name": "new_feature.completed",
                "data": {
                    "job_id": job_id,
                    "user_id": user_id,
                    "result": result
                }
            }
        )

        return result

    except Exception as e:
        # Handle error
        await ctx.step.send_event(
            "feature-failed",
            {
                "name": "new_feature.failed",
                "data": {
                    "job_id": job_id,
                    "user_id": user_id,
                    "error": str(e)
                }
            }
        )
        raise

async def store_feature_result(job_id: str, user_id: str, result: dict):
    """Store feature processing result"""
    async with get_db_session() as session:
        service = NewModelService(session)
        await service.create(
            name=f"feature_result_{job_id}",
            data={"user_id": user_id, "result": result}
        )
```

### 6.4 Environment Management

**Development scripts:**
```bash
# scripts/dev_setup.sh
#!/bin/bash
set -e

echo "Setting up development environment..."

# Install dependencies
poetry install

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Run migrations
poetry run alembic upgrade head

# Create test data
poetry run python scripts/create_test_data.py

echo "Development environment ready!"
echo "API: http://localhost:8000"
echo "Inngest: http://localhost:8288"
echo "MinIO Console: http://localhost:9001"
```

**Hot reloading setup:**
```python
# scripts/dev_server.py
import subprocess
import sys
from pathlib import Path

def start_dev_server():
    """Start development server with hot reloading"""

    # Start Inngest dev server in background
    inngest_process = subprocess.Popen([
        "inngest-cli", "dev"
    ])

    try:
        # Start FastAPI server
        subprocess.run([
            "uvicorn", "app.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload-dir", "app",
            "--reload-exclude", "*.pyc"
        ])
    finally:
        # Clean up Inngest process
        inngest_process.terminate()

if __name__ == "__main__":
    start_dev_server()
```

This local development guide provides comprehensive instructions for setting up and working with the Inngest-based development environment, including function development, testing, debugging, and common development tasks.
```
