# Docuforge - Deployment Guide (Inngest)

## Overview

This guide provides comprehensive deployment instructions for the Inngest-based Docuforge, covering both local development and production environments. The deployment leverages Inngest's self-hosting capabilities with Docker Compose for local development and Kubernetes for production.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Development Setup](#2-local-development-setup)
3. [Production Deployment](#3-production-deployment)
4. [Monitoring and Observability](#4-monitoring-and-observability)
5. [Troubleshooting](#5-troubleshooting)

## 1. Prerequisites

### 1.1 System Requirements

**Local Development:**
- Docker 20.10+
- Docker Compose 2.0+
- Node.js 18+ (for Inngest CLI)
- Python 3.9+
- 8GB RAM minimum (16GB recommended)

**Production:**
- Kubernetes 1.24+
- Helm 3.8+
- kubectl configured for your cluster
- 16GB RAM minimum per node
- SSD storage for databases

### 1.2 Required Tools

```bash
# Install Inngest CLI
npm install -g inngest-cli

# Install Helm (if not already installed)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install kubectl (if not already installed)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

## 2. Local Development Setup

### 2.1 Environment Configuration

Create environment file:
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

**Required Environment Variables:**
```bash
# Application
APP_NAME=Docuforge Clone
DEBUG=true
SECRET_KEY=your-secret-key-here

# Inngest
INNGEST_EVENT_KEY=your-inngest-event-key
INNGEST_SIGNING_KEY=your-inngest-signing-key
INNGEST_DEV=true

# Database
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/docuforge_clone

# Redis
REDIS_URL=redis://redis:6379

# Storage (MinIO)
STORAGE_ENDPOINT=minio:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin
STORAGE_BUCKET=documents
STORAGE_SECURE=false

# Qdrant
QDRANT_URL=http://qdrant:6333

# AI Services (Optional for development)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### 2.2 Docker Compose Setup

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # Main API application
  api:
    build: 
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/docuforge_clone
      - REDIS_URL=redis://redis:6379
      - STORAGE_ENDPOINT=minio:9000
      - QDRANT_URL=http://qdrant:6333
      - INNGEST_EVENT_KEY=${INNGEST_EVENT_KEY}
      - INNGEST_SIGNING_KEY=${INNGEST_SIGNING_KEY}
      - INNGEST_DEV=true
    volumes:
      - ./app:/app/app
      - ./tests:/app/tests
    depends_on:
      - postgres
      - redis
      - minio
      - qdrant
    restart: unless-stopped

  # Inngest Dev Server
  inngest:
    image: inngest/inngest:latest
    ports:
      - "8288:8288"
    environment:
      - INNGEST_DEV=true
      - INNGEST_PORT=8288
    restart: unless-stopped

  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: docuforge_clone
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
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
      - minio_data:/data
    restart: unless-stopped

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  # Prometheus (Optional - for monitoring)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  # Grafana (Optional - for dashboards)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  minio_data:
  qdrant_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: docuforge-network
```

### 2.3 Application Dockerfile

**docker/Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 2.4 Starting Local Development

```bash
# Clone repository
git clone <repository-url>
cd docuforge

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Run database migrations
docker-compose exec api alembic upgrade head

# Create initial buckets in MinIO
docker-compose exec api python scripts/setup_storage.py

# Access services:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Inngest Dashboard: http://localhost:8288
# - MinIO Console: http://localhost:9001
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### 2.5 Development Workflow

```bash
# Start Inngest dev server (in separate terminal)
inngest-cli dev

# Run tests
docker-compose exec api pytest

# Run specific test
docker-compose exec api pytest tests/test_api/test_documents.py -v

# Check code quality
docker-compose exec api black app/ tests/
docker-compose exec api isort app/ tests/
docker-compose exec api flake8 app/ tests/
docker-compose exec api mypy app/

# Database operations
docker-compose exec api alembic revision --autogenerate -m "Add new table"
docker-compose exec api alembic upgrade head

# View function logs
docker-compose logs -f inngest

# Restart specific service
docker-compose restart api
```

## 3. Production Deployment

### 3.1 Kubernetes Cluster Setup

**Prerequisites:**
- Kubernetes cluster with at least 3 nodes
- Ingress controller (nginx-ingress recommended)
- Cert-manager for TLS certificates
- Storage class for persistent volumes

### 3.2 Namespace and RBAC

**k8s/namespace.yaml:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: docuforge
  labels:
    name: docuforge
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: docuforge-sa
  namespace: docuforge
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: docuforge-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: docuforge-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: docuforge-role
subjects:
- kind: ServiceAccount
  name: docuforge-sa
  namespace: docuforge
```

### 3.3 ConfigMap and Secrets

**k8s/configmap.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: docuforge-config
  namespace: docuforge
data:
  APP_NAME: "Docuforge Clone"
  DEBUG: "false"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  DATABASE_POOL_SIZE: "20"
  DATABASE_MAX_OVERFLOW: "30"
  REDIS_DB: "0"
  REDIS_MAX_CONNECTIONS: "20"
  STORAGE_BUCKET: "documents"
  STORAGE_SECURE: "true"
  QDRANT_COLLECTION: "documents"
  MAX_FILE_SIZE: "104857600"  # 100MB
  OCR_TIMEOUT: "300"
  ACCESS_TOKEN_EXPIRE_MINUTES: "30"
  ALGORITHM: "HS256"
  PROMETHEUS_PORT: "9090"
  LOG_LEVEL: "INFO"
  INNGEST_DEV: "false"
```

**k8s/secret.yaml:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: docuforge-secrets
  namespace: docuforge
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret-key>
  INNGEST_EVENT_KEY: <base64-encoded-inngest-event-key>
  INNGEST_SIGNING_KEY: <base64-encoded-inngest-signing-key>
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  STORAGE_ACCESS_KEY: <base64-encoded-storage-access-key>
  STORAGE_SECRET_KEY: <base64-encoded-storage-secret-key>
  OPENAI_API_KEY: <base64-encoded-openai-key>
  ANTHROPIC_API_KEY: <base64-encoded-anthropic-key>
```

### 3.4 Inngest Deployment with Helm

**Add Inngest Helm repository:**
```bash
helm repo add inngest https://inngest.github.io/helm-charts
helm repo update
```

**values-inngest.yaml:**
```yaml
# Inngest Helm values
image:
  repository: inngest/inngest
  tag: "latest"
  pullPolicy: IfNotPresent

replicaCount: 2

env:
  - name: INNGEST_POSTGRES_URL
    valueFrom:
      secretKeyRef:
        name: inngest-secrets
        key: postgres-url
  - name: INNGEST_REDIS_URL
    valueFrom:
      secretKeyRef:
        name: inngest-secrets
        key: redis-url
  - name: INNGEST_LOG_LEVEL
    value: "info"

service:
  type: ClusterIP
  port: 8288

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: inngest.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: inngest-tls
      hosts:
        - inngest.yourdomain.com

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

nodeSelector: {}
tolerations: []
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - inngest
        topologyKey: kubernetes.io/hostname

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 20Gi

postgresql:
  enabled: true
  auth:
    postgresPassword: "secure-password"
    database: "inngest"
  primary:
    persistence:
      enabled: true
      storageClass: "fast-ssd"
      size: 50Gi

redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      enabled: true
      storageClass: "fast-ssd"
      size: 10Gi
```

**Deploy Inngest:**
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create Inngest secrets
kubectl create secret generic inngest-secrets \
  --from-literal=postgres-url="postgresql://user:pass@postgres:5432/inngest" \
  --from-literal=redis-url="redis://redis:6379" \
  -n docuforge

# Install Inngest with Helm
helm install inngest inngest/inngest \
  -f values-inngest.yaml \
  -n docuforge
```

### 3.5 Application Deployment

**k8s/deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docuforge-api
  namespace: docuforge
  labels:
    app: docuforge-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: docuforge-api
  template:
    metadata:
      labels:
        app: docuforge-api
    spec:
      serviceAccountName: docuforge-sa
      containers:
      - name: api
        image: docuforge:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: INNGEST_BASE_URL
          value: "http://inngest:8288"
        envFrom:
        - configMapRef:
            name: docuforge-config
        - secretRef:
            name: docuforge-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: tmp-volume
          mountPath: /tmp
      volumes:
      - name: tmp-volume
        emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - docuforge-api
              topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: Service
metadata:
  name: docuforge-api-service
  namespace: docuforge
  labels:
    app: docuforge-api
spec:
  selector:
    app: docuforge-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

### 3.6 Ingress Configuration

**k8s/ingress.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: docuforge-ingress
  namespace: docuforge
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: docuforge-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: docuforge-api-service
            port:
              number: 80
```

### 3.7 Horizontal Pod Autoscaler

**k8s/hpa.yaml:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: docuforge-hpa
  namespace: docuforge
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: docuforge-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### 3.8 Production Deployment Commands

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -n docuforge
kubectl get services -n docuforge
kubectl get ingress -n docuforge

# Check pod logs
kubectl logs -f deployment/docuforge-api -n docuforge

# Check Inngest status
kubectl logs -f deployment/inngest -n docuforge

# Scale deployment
kubectl scale deployment docuforge-api --replicas=5 -n docuforge

# Update deployment (rolling update)
kubectl set image deployment/docuforge-api api=docuforge:v1.1.0 -n docuforge

# Check rollout status
kubectl rollout status deployment/docuforge-api -n docuforge

# Rollback if needed
kubectl rollout undo deployment/docuforge-api -n docuforge
```

## 4. Monitoring and Observability

### 4.1 Prometheus Configuration

**monitoring/prometheus.yml:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'docuforge-api'
    static_configs:
      - targets: ['docuforge-api-service:9090']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'inngest'
    static_configs:
      - targets: ['inngest:8288']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - docuforge
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

### 4.2 Grafana Dashboards

**monitoring/grafana/dashboards/docuforge.json:**
```json
{
  "dashboard": {
    "title": "Docuforge Clone Metrics",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Function Execution Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(inngest_function_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Jobs",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(active_jobs_total)",
            "legendFormat": "Total Active Jobs"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

### 4.3 Alerting Rules

**monitoring/alert_rules.yml:**
```yaml
groups:
  - name: docuforge-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighFunctionDuration
        expr: histogram_quantile(0.95, rate(inngest_function_duration_seconds_bucket[5m])) > 300
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Function execution taking too long"
          description: "95th percentile duration is {{ $value }} seconds"

      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} is restarting frequently"
```

## 5. Troubleshooting

### 5.1 Common Issues

**Issue: Inngest functions not executing**
```bash
# Check Inngest pod logs
kubectl logs -f deployment/inngest -n docuforge

# Verify event key and signing key
kubectl get secret docuforge-secrets -o yaml -n docuforge

# Check function registration
curl http://inngest.yourdomain.com/v1/functions
```

**Issue: Database connection errors**
```bash
# Check database connectivity
kubectl exec -it deployment/docuforge-api -n docuforge -- \
  python -c "import asyncpg; print('DB connection test')"

# Verify database URL
kubectl get secret docuforge-secrets -o jsonpath='{.data.DATABASE_URL}' -n docuforge | base64 -d
```

**Issue: High memory usage**
```bash
# Check memory usage
kubectl top pods -n docuforge

# Increase memory limits
kubectl patch deployment docuforge-api -n docuforge -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

### 5.2 Performance Tuning

**Database Optimization:**
```sql
-- Add indexes for better query performance
CREATE INDEX CONCURRENTLY idx_jobs_status_created ON jobs(status, created_at);
CREATE INDEX CONCURRENTLY idx_jobs_operation_status ON jobs(operation, status);
CREATE INDEX CONCURRENTLY idx_documents_processed ON documents(processed, created_at);
```

**Redis Configuration:**
```bash
# Increase Redis memory limit
kubectl patch deployment redis -n docuforge -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"redis","args":["redis-server","--maxmemory","2gb","--maxmemory-policy","allkeys-lru"]}]}}}}'
```

### 5.3 Backup and Recovery

**Database Backup:**
```bash
# Create database backup
kubectl exec -it postgres-pod -n docuforge -- \
  pg_dump -U user docuforge_clone > backup-$(date +%Y%m%d).sql

# Restore from backup
kubectl exec -i postgres-pod -n docuforge -- \
  psql -U user docuforge_clone < backup-20240101.sql
```

**Storage Backup:**
```bash
# Backup MinIO data
kubectl exec -it minio-pod -n docuforge -- \
  mc mirror /data /backup/minio-$(date +%Y%m%d)
```

This deployment guide provides comprehensive instructions for deploying the Inngest-based Docuforge in both development and production environments, with proper monitoring, scaling, and troubleshooting procedures.
```
```
