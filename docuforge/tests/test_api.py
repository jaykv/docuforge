"""
Basic API tests for DocuForge.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_version_endpoint():
    """Test version endpoint."""
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "api_version" in data
    assert "features" in data


def test_docs_endpoint():
    """Test API documentation endpoint."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_upload_endpoint_without_auth():
    """Test upload endpoint without authentication."""
    response = client.post("/api/v1/upload")
    assert response.status_code == 403  # Should require authentication


def test_parse_endpoint_without_auth():
    """Test parse endpoint without authentication."""
    response = client.post("/api/v1/parse", json={
        "document_url": "https://example.com/test.pdf"
    })
    assert response.status_code == 403  # Should require authentication


if __name__ == "__main__":
    pytest.main([__file__])
