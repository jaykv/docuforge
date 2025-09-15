"""
File storage operations using MinIO/S3.
"""

from minio import Minio
from minio.error import S3Error
from fastapi import UploadFile
import aiofiles
import tempfile
import os
from typing import Optional
from ..config import settings
import structlog

logger = structlog.get_logger()

# Initialize MinIO client
minio_client = Minio(
    settings.STORAGE_ENDPOINT,
    access_key=settings.STORAGE_ACCESS_KEY,
    secret_key=settings.STORAGE_SECRET_KEY,
    secure=settings.STORAGE_SECURE,
)


async def init_storage():
    """Initialize storage buckets."""
    try:
        # Check if bucket exists, create if not
        if not minio_client.bucket_exists(settings.STORAGE_BUCKET):
            minio_client.make_bucket(settings.STORAGE_BUCKET)
            logger.info(f"Created bucket: {settings.STORAGE_BUCKET}")
        else:
            logger.info(f"Bucket already exists: {settings.STORAGE_BUCKET}")
    except S3Error as e:
        logger.error(f"Error initializing storage: {e}")
        raise


async def upload_to_storage(file: UploadFile, file_id: str) -> str:
    """Upload file to storage and return storage path."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Generate storage path
        file_extension = os.path.splitext(file.filename)[1]
        storage_path = f"documents/{file_id}{file_extension}"

        # Upload to MinIO
        minio_client.fput_object(
            settings.STORAGE_BUCKET,
            storage_path,
            temp_file_path,
            content_type=file.content_type,
        )

        # Clean up temporary file
        os.unlink(temp_file_path)

        logger.info(f"Uploaded file to storage: {storage_path}")
        return storage_path

    except Exception as e:
        logger.error(f"Error uploading file to storage: {e}")
        raise


async def download_from_storage(storage_path: str, local_path: str):
    """Download file from storage to local path."""
    try:
        minio_client.fget_object(
            settings.STORAGE_BUCKET,
            storage_path,
            local_path,
        )
        logger.info(f"Downloaded file from storage: {storage_path}")
    except Exception as e:
        logger.error(f"Error downloading file from storage: {e}")
        raise


async def generate_presigned_url(storage_path: str, expires_hours: int = 24) -> str:
    """Generate presigned URL for file access."""
    try:
        from datetime import timedelta
        
        url = minio_client.presigned_get_object(
            settings.STORAGE_BUCKET,
            storage_path,
            expires=timedelta(hours=expires_hours),
        )
        return url
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        raise


async def delete_from_storage(storage_path: str):
    """Delete file from storage."""
    try:
        minio_client.remove_object(settings.STORAGE_BUCKET, storage_path)
        logger.info(f"Deleted file from storage: {storage_path}")
    except Exception as e:
        logger.error(f"Error deleting file from storage: {e}")
        raise
