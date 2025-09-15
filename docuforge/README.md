# DocuForge 🔥

**AI-Powered Document Processing Service with Inngest Orchestration**

DocuForge is a sophisticated document processing service that provides OCR, layout analysis, and AI-powered correction capabilities. Built with FastAPI and Inngest for reliable, scalable workflow orchestration.

## ✨ Features

- **🔄 Event-Driven Architecture**: Built on Inngest.io for reliable, scalable workflow orchestration
- **📄 Complete API Compatibility**: 1-to-1 compatible with Docuforge API endpoints
- **🤖 Agentic OCR Correction**: AI-powered error correction using GPT-4 and Claude
- **⚡ Parallel Processing**: Multi-engine OCR processing with ensemble results
- **🏗️ Production Ready**: Self-hosting capabilities with Docker Compose
- **📊 Monitoring & Observability**: Built-in metrics, logging, and health checks

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.9+ (for local development)
- Node.js 18+ (for Inngest CLI)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd docuforge

# Copy environment template
cp .env.example .env

# Edit environment variables (add your API keys)
nano .env
```

### 2. Start Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 3. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check Inngest dashboard
open http://localhost:8288

# Check API documentation
open http://localhost:8000/docs

# Check monitoring dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
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
  "schema": {
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
GET /api/v1/job/{job_id}

curl "http://localhost:8000/api/v1/job/12345" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Cancel Job
```bash
POST /api/v1/cancel/{job_id}

curl -X POST "http://localhost:8000/api/v1/cancel/12345" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 🏗️ Architecture

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

## 🛠️ Development

### Local Development Setup

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install Inngest CLI
npm install -g inngest-cli

# Start Inngest Dev Server (separate terminal)
inngest-cli dev

# Start API server
poetry run uvicorn app.main:app --reload

# Run tests
poetry run pytest
```

### Project Structure

```
docuforge/
├── app/
│   ├── api/                   # API endpoints
│   ├── core/                  # Core business logic
│   ├── functions/             # Inngest functions
│   ├── models/                # Database models
│   ├── schemas/               # Pydantic schemas
│   ├── services/              # Business services
│   └── utils/                 # Utility functions
├── tests/                     # Test suite
├── docker-compose.yml         # Local development setup
├── Dockerfile                 # Container configuration
└── pyproject.toml            # Poetry configuration
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test types
poetry run pytest tests/api/ -v
poetry run pytest tests/functions/ -v
```

## 📊 Monitoring

- **API Documentation**: http://localhost:8000/docs
- **Inngest Dashboard**: http://localhost:8288
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

## ⚙️ Configuration

Key environment variables:

```bash
# AI Service API Keys (required for agentic correction)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Inngest Configuration
INNGEST_EVENT_KEY=your-inngest-event-key
INNGEST_SIGNING_KEY=your-inngest-signing-key

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/docuforge

# Storage
STORAGE_ENDPOINT=localhost:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin
```

## 🚀 Production Deployment

For production deployment, see the comprehensive guides in the parent directory:
- [Deployment Guide](../DEPLOYMENT_GUIDE.md)
- [Technical Specification](../TECHNICAL_SPECIFICATION.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Inngest.io** for the excellent workflow orchestration platform
- **FastAPI** for the robust web framework
- **PostgreSQL**, **Redis**, **MinIO** for the infrastructure components

---

**DocuForge** - Forging the future of document processing with AI 🔥
