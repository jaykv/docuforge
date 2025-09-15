# Docuforge - Inngest Implementation

## 🚀 Overview

This is a comprehensive implementation of a Docuforge built with **Inngest.io** for workflow orchestration. The system provides a 1-to-1 API-compatible document processing service that can parse, extract, split, and edit documents using advanced OCR, layout analysis, and AI-powered correction.

### ✨ Key Features

- **🔄 Event-Driven Architecture**: Built on Inngest.io for reliable, scalable workflow orchestration
- **📄 Complete API Compatibility**: 1-to-1 compatible with Docuforge API endpoints
- **🤖 Agentic OCR Correction**: AI-powered error correction using GPT-4 and Claude
- **⚡ Parallel Processing**: Multi-engine OCR processing with ensemble results
- **🏗️ Production Ready**: Self-hosting capabilities with Docker Compose and Kubernetes
- **🧪 Comprehensive Testing**: Full test suite with function, API, and integration tests
- **📊 Monitoring & Observability**: Built-in metrics, logging, and health checks

### 🏛️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Inngest       │    │   Processing    │
│   API Layer     │───▶│   Functions     │───▶│   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis Cache   │    │   MinIO/S3      │
│   Database      │    │   & Queue       │    │   Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📚 Documentation

### Core Documentation
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Complete project roadmap and architecture
- **[Technical Specification](TECHNICAL_SPECIFICATION.md)** - Detailed technical requirements and code structure
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production and local deployment instructions
- **[Local Development Guide](LOCAL_DEVELOPMENT_GUIDE.md)** - Development environment setup
- **[Workflow Examples](WORKFLOW_EXAMPLES.md)** - Comprehensive Inngest function examples
- **[Testing Strategy](TESTING_STRATEGY.md)** - Complete testing approach and examples

### Quick Links
- [🚀 Quick Start](#quick-start)
- [🔧 API Endpoints](#api-endpoints)
- [⚙️ Configuration](#configuration)
- [🐳 Docker Setup](#docker-setup)
- [☸️ Kubernetes Deployment](#kubernetes-deployment)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for Inngest CLI)
- Python 3.9+

### 1. Clone and Setup
```bash
git clone <repository-url>
cd docuforge

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 2. Start Services
```bash
# Start supporting services
docker-compose up -d postgres redis minio qdrant

# Install dependencies
poetry install

# Run database migrations
poetry run alembic upgrade head

# Start Inngest Dev Server (separate terminal)
inngest-cli dev

# Start API server
poetry run uvicorn app.main:app --reload
```

### 3. Verify Installation
```bash
# Check API health
curl http://localhost:8000/health

# Check Inngest dashboard
open http://localhost:8288

# Check API documentation
open http://localhost:8000/docs
```

## 🔧 API Endpoints

### Document Processing

#### Upload Document
```bash
POST /api/v1/upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

#### Parse Document
```bash
POST /api/v1/parse
Content-Type: application/json

{
  "document_url": "https://example.com/document.pdf",
  "options": {
    "ocr_mode": "agentic",
    "extraction_mode": "hybrid"
  }
}
```

#### Extract Structured Data
```bash
POST /api/v1/extract
Content-Type: application/json

{
  "document_url": "https://example.com/invoice.pdf",
  "extraction_schema": {
    "fields": {
      "invoice_number": {"type": "string", "method": "regex"},
      "total_amount": {"type": "number", "method": "ai"}
    }
  }
}
```

#### Split Document
```bash
POST /api/v1/split
Content-Type: application/json

{
  "document_url": "https://example.com/document.pdf",
  "split_criteria": {
    "method": "page_range",
    "ranges": [[1, 3], [4, 6]]
  }
}
```

### Job Management

#### Get Job Status
```bash
GET /api/v1/jobs/{job_id}

curl "http://localhost:8000/api/v1/jobs/12345" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### List Jobs
```bash
GET /api/v1/jobs?status=completed&limit=10

curl "http://localhost:8000/api/v1/jobs?status=completed" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## ⚙️ Configuration

### Environment Variables

```bash
# Application
APP_NAME=Docuforge Clone
DEBUG=false
SECRET_KEY=your-secret-key

# Inngest
INNGEST_EVENT_KEY=your-inngest-event-key
INNGEST_SIGNING_KEY=your-inngest-signing-key
INNGEST_DEV=false
INNGEST_BASE_URL=http://inngest:8288

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/docuforge_clone

# Redis
REDIS_URL=redis://redis:6379

# Storage
STORAGE_ENDPOINT=minio:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin
STORAGE_BUCKET=documents

# AI Services
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### OCR Configuration

```python
# OCR engine settings
OCR_ENGINES = ["tesseract", "paddle", "easy"]
OCR_ENSEMBLE_WEIGHTS = [0.4, 0.4, 0.2]
OCR_TIMEOUT = 300  # seconds
OCR_DPI = 300

# Agentic correction settings
AGENTIC_CORRECTION_MODEL = "gpt-4"
AGENTIC_CORRECTION_TEMPERATURE = 0.1
VISION_VALIDATION_MODEL = "gpt-4-vision-preview"
```

## 🐳 Docker Setup

### Development
```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f api

# Scale API instances
docker-compose up -d --scale api=3
```

### Production
```bash
# Build production image
docker build -f docker/Dockerfile -t docuforge:latest .

# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Health check
curl http://localhost:8000/health
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster
- Helm 3.8+
- Ingress controller

### Deploy Inngest
```bash
# Add Inngest Helm repo
helm repo add inngest https://inngest.github.io/helm-charts
helm repo update

# Install Inngest
helm install inngest inngest/inngest \
  -f k8s/values-inngest.yaml \
  -n docuforge
```

### Deploy Application
```bash
# Create namespace and secrets
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml

# Deploy application
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n docuforge
kubectl get ingress -n docuforge
```

## 🧪 Testing

### Run Tests
```bash
# All tests
poetry run pytest

# Specific test types
poetry run pytest tests/unit/ -v
poetry run pytest tests/functions/ -v
poetry run pytest tests/api/ -v
poetry run pytest tests/integration/ -v

# With coverage
poetry run pytest --cov=app --cov-report=html
```

### Test Inngest Functions
```bash
# Test individual function
poetry run pytest tests/functions/test_document_processing.py::test_process_document_success -v

# Test with Inngest Dev Server
inngest-cli dev &
poetry run pytest tests/functions/ -v
```

## 📊 Monitoring

### Metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Inngest Dashboard**: http://localhost:8288

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Detailed health with dependencies
curl http://localhost:8000/health/detailed

# Inngest health
curl http://localhost:8288/health
```

### Logs
```bash
# Application logs
docker-compose logs -f api

# Inngest function logs
docker-compose logs -f inngest

# Database logs
docker-compose logs -f postgres
```

## 🔧 Development

### Adding New Functions
```python
# app/functions/new_function.py
from inngest import Context
from ..functions import inngest_client

@inngest_client.create_function(
    fn_id="new-function",
    trigger=inngest.TriggerEvent(event="new.event"),
    retries=2
)
async def new_function(ctx: Context) -> dict:
    # Function implementation
    return {"status": "completed"}
```

### Adding New API Endpoints
```python
# app/api/new_endpoint.py
from fastapi import APIRouter
from ..functions import inngest_client

router = APIRouter()

@router.post("/new-endpoint")
async def new_endpoint(data: dict):
    await inngest_client.send_event({
        "name": "new.event",
        "data": data
    })
    return {"status": "triggered"}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Code Quality
```bash
# Format code
poetry run black app/ tests/
poetry run isort app/ tests/

# Lint code
poetry run flake8 app/ tests/
poetry run mypy app/

# Run pre-commit hooks
pre-commit run --all-files
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: See the [docs](.) directory
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

## 🙏 Acknowledgments

- **Inngest.io** for the excellent workflow orchestration platform
- **Docuforge** for the API design inspiration
- **FastAPI** for the robust web framework
- **PostgreSQL**, **Redis**, **MinIO** for the infrastructure components
