"""
S3 upload utility using pre-signed URLs.
No AWS SDK needed - just HTTP PUT to signed URLs.
"""

import io
import logging
import time
import uuid
from typing import Optional, Tuple
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class S3Uploader:
    """
    Simple S3 uploader using pre-signed URLs.
    Philosophy: Keep it simple - just PUT to the URL.
    """
    
    def __init__(self):
        self.session = requests.Session()
        # Connection pooling for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0  # We handle retries ourselves
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def upload_image(
        self, 
        image: Image.Image, 
        presigned_url: str,
        jpeg_quality: int = 65,
        max_retries: int = 3,
        timeout: int = 30
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Upload image to S3 using pre-signed URL.
        
        Args:
            image: PIL Image to upload
            presigned_url: Pre-signed S3 URL for PUT
            jpeg_quality: JPEG compression quality (1-100)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            
        Returns:
            Tuple[bool, Optional[str], Optional[int]]: (success, error_message, http_status)
        """
        
        # Convert image to JPEG bytes
        try:
            img_buffer = io.BytesIO()
            # Always save as JPEG for consistency
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert images with alpha channel to RGB
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(img_buffer, format='JPEG', quality=jpeg_quality)
            img_data = img_buffer.getvalue()
            
            logger.info(f"Image converted to JPEG, size: {len(img_data) / 1024:.2f} KB")
            
        except Exception as e:
            error_msg = f"Failed to convert image to JPEG: {e}"
            logger.error(error_msg)
            return False, error_msg, None
        
        # Upload with exponential backoff retry
        for attempt in range(max_retries):
            try:
                logger.info(f"Uploading to S3 (attempt {attempt + 1}/{max_retries})")
                
                response = self.session.put(
                    presigned_url,
                    data=img_data,
                    headers={
                        'Content-Type': 'image/jpeg',
                        'Content-Length': str(len(img_data))
                    },
                    timeout=timeout
                )
                
                # Success: 200 OK or 204 No Content
                if response.status_code in (200, 204):
                    logger.info(f"Upload successful, status: {response.status_code}")
                    return True, None, response.status_code
                
                # Client error (4xx) - don't retry
                if 400 <= response.status_code < 500:
                    error_msg = f"Upload failed with client error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    return False, error_msg, response.status_code
                
                # Server error (5xx) - retry
                logger.warning(f"Upload attempt {attempt + 1} failed with status {response.status_code}")
                
            except requests.exceptions.Timeout:
                logger.warning(f"Upload attempt {attempt + 1} timed out")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during upload: {e}")
                return False, str(e), None
            
            # Exponential backoff: 1s, 2s, 4s
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        error_msg = f"Upload failed after {max_retries} attempts"
        logger.error(error_msg)
        return False, error_msg, None
    
    def generate_s3_key(self, prefix: Optional[str] = None) -> str:
        """
        Generate a unique S3 key for the image.
        
        Args:
            prefix: Optional prefix for the key
            
        Returns:
            str: S3 key like "prefix/uuid.jpeg" or just "uuid.jpeg"
        """
        unique_id = str(uuid.uuid4())
        if prefix:
            return f"{prefix}/{unique_id}.jpeg"
        return f"{unique_id}.jpeg"


# Global instance
_uploader = None

def get_uploader() -> S3Uploader:
    """Get the global S3 uploader instance"""
    global _uploader
    if _uploader is None:
        _uploader = S3Uploader()
    return _uploader


def upload_to_s3(
    image: Image.Image,
    presigned_url: str,
    jpeg_quality: int = 65,
    image_hash: Optional[str] = None,
) -> Tuple[bool, Optional[str], str, Optional[int]]:
    """
    Simple API for S3 upload using the provided pre-signed URL as-is.

    Args:
        image: PIL Image to upload
        presigned_url: Pre-signed S3 URL for PUT (includes the exact desired filename)
        jpeg_quality: JPEG compression quality
        image_hash: Unused; kept for backward compatibility

    Returns:
        Tuple[bool, Optional[str], str, Optional[int]]: (success, error_message, final_url, http_status)
        On success, returns (True, None, presigned_url, status_code)
        On failure, returns (False, error_message, None, status_code)
    """
    # Use the provided URL without modifying the filename
    final_url = presigned_url

    uploader = get_uploader()
    success, error, status_code = uploader.upload_image(image, final_url, jpeg_quality)

    if success:
        return True, None, final_url, status_code
    else:
        return False, error, None, status_code
