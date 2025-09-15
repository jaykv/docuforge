"""
DocuForge - Main FastAPI application.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import time

from .config import settings
from .api import documents, jobs, webhooks
from .functions import serve_inngest
from .core.database import init_database, close_database
from .core.storage import init_storage
from .utils.monitoring import setup_logging
from .schemas.responses import HealthResponse

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Document Processing Service with Inngest Orchestration",
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
app.include_router(documents.router, prefix=settings.API_PREFIX, tags=["documents"])
app.include_router(jobs.router, prefix=settings.API_PREFIX, tags=["jobs"])
app.include_router(webhooks.router, prefix=settings.API_PREFIX, tags=["webhooks"])

# Mount Inngest functions
app.mount("/api/inngest", serve_inngest())


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        # Initialize database
        await init_database()
        
        # Initialize storage buckets
        await init_storage()
        
        print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} started successfully!")
        print(f"📚 API Documentation: http://localhost:{settings.API_PORT}/docs")
        print(f"🔧 Inngest Dashboard: {settings.INNGEST_BASE_URL}")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    await close_database()
    print(f"👋 {settings.APP_NAME} shutdown complete")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=time.time()
    )


@app.get("/health/detailed", response_model=HealthResponse)
async def detailed_health_check():
    """Detailed health check with dependency status."""
    health_status = HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=time.time(),
        services={}
    )
    
    # Check database connectivity
    try:
        from .core.database import async_session_factory
        async with async_session_factory() as session:
            await session.execute("SELECT 1")
        health_status.services["database"] = "healthy"
    except Exception as e:
        health_status.services["database"] = f"unhealthy: {str(e)}"
        health_status.status = "unhealthy"
    
    # Check storage connectivity
    try:
        from .core.storage import minio_client
        minio_client.list_buckets()
        health_status.services["storage"] = "healthy"
    except Exception as e:
        health_status.services["storage"] = f"unhealthy: {str(e)}"
        health_status.status = "unhealthy"
    
    if health_status.status == "unhealthy":
        return JSONResponse(
            status_code=503,
            content=health_status.dict()
        )
    
    return health_status


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    import structlog
    logger = structlog.get_logger()
    
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID to all requests."""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
