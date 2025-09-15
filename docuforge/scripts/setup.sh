#!/bin/bash

# DocuForge Setup Script

set -e

echo "🔥 Setting up DocuForge..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it with your API keys and configuration."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p logs

# Create basic Prometheus configuration
cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'docuforge-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'inngest'
    static_configs:
      - targets: ['inngest:8288']
    metrics_path: /metrics
    scrape_interval: 10s
EOF

# Create Grafana datasource configuration
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

echo "🐳 Starting Docker services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service health..."

# Check API health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API service is healthy"
else
    echo "❌ API service is not responding"
fi

# Check Inngest
if curl -f http://localhost:8288 > /dev/null 2>&1; then
    echo "✅ Inngest service is healthy"
else
    echo "❌ Inngest service is not responding"
fi

echo ""
echo "🎉 DocuForge setup complete!"
echo ""
echo "📚 Access points:"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Inngest Dashboard: http://localhost:8288"
echo "   Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   Prometheus: http://localhost:9090"
echo "   MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo ""
echo "🔧 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Restart services: docker-compose restart api"
echo "   3. Test the API: curl http://localhost:8000/health"
echo ""
echo "📖 For more information, see README.md"
