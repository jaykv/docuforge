"""
Monitoring and metrics utilities.
"""

import time
import structlog
from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
from typing import Any, Dict

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
    """Configure structured logging."""
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
    """Decorator to monitor Inngest function execution."""
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


async def log_step_metrics(
    job_id: str,
    step_name: str,
    duration: float,
    success: bool,
    error_message: str = None,
    metadata: Dict[str, Any] = None
):
    """Log step execution metrics."""
    from ..models.metrics import ProcessingMetrics
    from ..core.database import async_session_factory

    async with async_session_factory() as session:
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


def record_ocr_accuracy(engine: str, confidence: float):
    """Record OCR accuracy metrics."""
    OCR_ACCURACY.labels(engine=engine).observe(confidence)


def update_active_jobs(operation: str, delta: int):
    """Update active jobs counter."""
    if delta > 0:
        ACTIVE_JOBS.labels(operation=operation).inc(delta)
    else:
        ACTIVE_JOBS.labels(operation=operation).dec(abs(delta))
