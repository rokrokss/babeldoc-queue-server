# SPDX-License-Identifier: AGPL-3.0-or-later.

import os
import logging
from typing import Optional, BinaryIO
import tempfile
import uuid
from pathlib import Path
import shutil
import time
from functools import wraps
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
DEFAULT_RETRY_ATTEMPTS = 3
CHUNK_SIZE = 8192 * 1024

def retry_on_failure(max_retries: int = DEFAULT_RETRY_ATTEMPTS):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    time.sleep(2 ** attempt)
            return None
        return wrapper
    return decorator

class GCSManager:
    def __init__(self):
        self.client = None
        self.bucket = None
        self.bucket_name = os.getenv("GCS_BUCKET_NAME")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.timeout = int(os.getenv("GCS_TIMEOUT", DEFAULT_TIMEOUT))
        
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable is required")
        
        if not self.credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID environment variable is required")
        
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            self.client = storage.Client.from_service_account_json(
                self.credentials_path,
                project=self.project_id
            )
            
            self.bucket = self.client.bucket(self.bucket_name)
            if not self.bucket.exists():
                raise ValueError(f"Bucket {self.bucket_name} does not exist")
            
            logger.info(f"GCS client initialized successfully for bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Error initializing GCS client: {e}")
            raise
    
    def _get_blob(self, gcs_key: str) -> storage.Blob:
        return self.bucket.blob(gcs_key)
    
    def _validate_gcs_key(self, gcs_key: str) -> bool:
        if not gcs_key or not isinstance(gcs_key, str):
            return False
        if gcs_key.startswith('/') or gcs_key.endswith('/'):
            return False
        return True
    
    @retry_on_failure(max_retries=3)
    def upload_file(self, file_path: str, gcs_key: str) -> bool:
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            if not self._validate_gcs_key(gcs_key):
                logger.error(f"Invalid GCS key: {gcs_key}")
                return False
            
            blob = self._get_blob(gcs_key)
            blob.upload_from_filename(file_path, timeout=self.timeout)
            logger.info(f"File uploaded successfully: {file_path} -> {gcs_key}")
            return True
            
        except GoogleCloudError as e:
            logger.error(f"GCS error uploading file {gcs_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file {gcs_key}: {e}")
            return False
    
    @retry_on_failure(max_retries=3)
    def upload_fileobj(self, file_obj: BinaryIO, gcs_key: str) -> bool:
        try:
            if not self._validate_gcs_key(gcs_key):
                logger.error(f"Invalid GCS key: {gcs_key}")
                return False
            
            blob = self._get_blob(gcs_key)
            file_obj.seek(0)
            
            blob.upload_from_file(file_obj, timeout=self.timeout)
            logger.info(f"File object uploaded successfully: {gcs_key}")
            return True
            
        except GoogleCloudError as e:
            logger.error(f"GCS error uploading file object {gcs_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file object {gcs_key}: {e}")
            return False
    
    @retry_on_failure(max_retries=3)
    def download_file(self, gcs_key: str, local_path: str) -> bool:
        try:
            if not self._validate_gcs_key(gcs_key):
                logger.error(f"Invalid GCS key: {gcs_key}")
                return False
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            blob = self._get_blob(gcs_key)
            if not blob.exists():
                logger.error(f"File not found in GCS: {gcs_key}")
                return False
            
            blob.download_to_filename(local_path, timeout=self.timeout)
            logger.info(f"File downloaded successfully: {gcs_key} -> {local_path}")
            return True
            
        except GoogleCloudError as e:
            logger.error(f"GCS error downloading file {gcs_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading file {gcs_key}: {e}")
            return False
    
    @retry_on_failure(max_retries=3)
    def download_to_temp(self, gcs_key: str) -> Optional[str]:
        try:
            if not self._validate_gcs_key(gcs_key):
                logger.error(f"Invalid GCS key: {gcs_key}")
                return None
            
            blob = self._get_blob(gcs_key)
            if not blob.exists():
                logger.error(f"File not found in GCS: {gcs_key}")
                return None
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
            temp_path = temp_file.name
            temp_file.close()
            
            blob.download_to_filename(temp_path, timeout=self.timeout)
            logger.info(f"File downloaded to temp: {gcs_key} -> {temp_path}")
            return temp_path
            
        except GoogleCloudError as e:
            logger.error(f"GCS error downloading file {gcs_key} to temp: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading file {gcs_key} to temp: {e}")
            return None
    
    @retry_on_failure(max_retries=3)
    def delete_file(self, gcs_key: str) -> bool:
        try:
            if not self._validate_gcs_key(gcs_key):
                logger.error(f"Invalid GCS key: {gcs_key}")
                return False
            
            blob = self._get_blob(gcs_key)
            blob.delete()
            logger.info(f"File deleted successfully: {gcs_key}")
            return True
            
        except NotFound:
            logger.warning(f"File not found for deletion: {gcs_key}")
            return True
        except GoogleCloudError as e:
            logger.error(f"GCS error deleting file {gcs_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting file {gcs_key}: {e}")
            return False
    
    @retry_on_failure(max_retries=3)
    def delete_files_with_prefix(self, prefix: str) -> bool:
        try:
            if not prefix or not isinstance(prefix, str):
                logger.error(f"Invalid prefix: {prefix}")
                return False
            
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            if not blobs:
                logger.info(f"No files found with prefix: {prefix}")
                return True
            
            deleted_count = 0
            failed_count = 0
            
            for blob in blobs:
                try:
                    blob.delete()
                    deleted_count += 1
                    logger.debug(f"Deleted file: {blob.name}")
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to delete file {blob.name}: {e}")
            
            logger.info(f"Deleted {deleted_count} files with prefix: {prefix}")
            if failed_count > 0:
                logger.warning(f"Failed to delete {failed_count} files with prefix: {prefix}")
                return False
            
            return True
            
        except GoogleCloudError as e:
            logger.error(f"GCS error deleting files with prefix {prefix}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting files with prefix {prefix}: {e}")
            return False
    
    @retry_on_failure(max_retries=2)
    def file_exists(self, gcs_key: str) -> bool:
        try:
            if not self._validate_gcs_key(gcs_key):
                return False
            
            blob = self._get_blob(gcs_key)
            return blob.exists()
            
        except GoogleCloudError as e:
            logger.error(f"GCS error checking file existence {gcs_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking file existence {gcs_key}: {e}")
            return False
    
    @retry_on_failure(max_retries=2)
    def get_file_info(self, gcs_key: str) -> Optional[dict]:
        try:
            if not self._validate_gcs_key(gcs_key):
                return None
            
            blob = self._get_blob(gcs_key)
            if not blob.exists():
                return None
            
            blob.reload()
            return {
                "name": blob.name,
                "size": blob.size,
                "created": blob.time_created,
                "updated": blob.updated,
                "content_type": blob.content_type,
                "md5_hash": blob.md5_hash
            }
            
        except GoogleCloudError as e:
            logger.error(f"GCS error getting file info {gcs_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting file info {gcs_key}: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def generate_signed_url(self, gcs_key: str, expiration: int = 3600) -> Optional[str]:
        try:
            if not self._validate_gcs_key(gcs_key):
                logger.error(f"Invalid GCS key: {gcs_key}")
                return None
            
            blob = self._get_blob(gcs_key)
            if not blob.exists():
                logger.error(f"File not found for signed URL: {gcs_key}")
                return None
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET"
            )
            logger.info(f"Signed URL generated for {gcs_key}")
            return url
            
        except GoogleCloudError as e:
            logger.error(f"GCS error generating signed URL for {gcs_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating signed URL for {gcs_key}: {e}")
            return None
    
    def get_bucket_info(self) -> dict:
        try:
            self.bucket.reload()
            return {
                "name": self.bucket.name,
                "location": self.bucket.location,
                "storage_class": self.bucket.storage_class,
                "created": self.bucket.time_created
            }
        except Exception as e:
            logger.error(f"Error getting bucket info: {e}")
            return {}

gcs_manager = None

def get_gcs_manager() -> GCSManager:
    global gcs_manager
    if gcs_manager is None:
        try:
            gcs_manager = GCSManager()
        except Exception as e:
            logger.error(f"Failed to initialize GCS manager: {e}")
            raise
    return gcs_manager

def generate_gcs_key(prefix: str, filename: str) -> str:
    if not prefix or not filename:
        raise ValueError("Prefix and filename are required")
    
    safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    return f"{prefix}/{timestamp}_{unique_id}_{safe_filename}"

def cleanup_temp_file(temp_path: str):
    if not temp_path:
        return
    
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Temp file cleaned up: {temp_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp file {temp_path}: {e}")

def cleanup_temp_directory(temp_dir: str):
    if not temp_dir:
        return
    
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Temp directory cleaned up: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory {temp_dir}: {e}")

def cleanup_task_related_temp_files(task_id: str):
    if not task_id:
        return
    
    try:
        temp_dir = tempfile.gettempdir()
        cleaned_count = 0
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if task_id in file:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.debug(f"Task-related temp file cleaned up: {file_path}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up task temp file {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} task-related temp files for {task_id}")
        
    except Exception as e:
        logger.error(f"Error cleaning up task-related temp files for {task_id}: {e}")

def validate_environment():
    required_vars = [
        "GCS_BUCKET_NAME",
        "GOOGLE_APPLICATION_CREDENTIALS", 
        "GCP_PROJECT_ID"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not os.path.exists(cred_path):
        raise ValueError(f"Google credentials file not found: {cred_path}")
    
    logger.info("Environment validation passed")

try:
    validate_environment()
except Exception as e:
    logger.error(f"Environment validation failed: {e}")
    if os.getenv("ENVIRONMENT") == "production":
        raise 