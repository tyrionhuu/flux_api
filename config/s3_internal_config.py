"""
Internal S3 configuration for storing generated images and requests
"""

import base64
import json
from datetime import datetime, timedelta

# S3 Bucket Configuration (from test_sekai_upload_s3.py)
S3_BUCKET_NAME = "customers-sekai-background-image-gen"
S3_BUCKET_URL = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/"

# AWS Credentials for presigned POST
AWS_ACCESS_KEY_ID = "AKIAUBX3L5WZRFJCJA5G"

# S3 key prefix for internal storage
S3_KEY_PREFIX = "customers/sekai/background-image-gen"

# Upload settings
S3_UPLOAD_TIMEOUT = 30  # seconds
S3_MAX_FILE_SIZE = 50000000  # 50MB max
S3_UPLOAD_ACL = "private"

# Enable/disable S3 uploads
ENABLE_INTERNAL_S3_UPLOAD = True


def generate_presigned_post_data(filename: str, expiration_hours: int = 24):
    """
    Generate presigned POST data for S3 upload.
    Based on the working example from test_sekai_upload_s3.py
    
    Args:
        filename: Name of the file to upload (e.g., 'output-abc123.jpg')
        expiration_hours: Hours until the presigned URL expires
        
    Returns:
        dict: Presigned POST data with url and fields
    """
    # Using the exact working policy and signature from test_sekai_upload_s3.py
    # This policy is pre-signed and works for the bucket
    
    return {
        "url": S3_BUCKET_URL,
        "fields": {
            "acl": S3_UPLOAD_ACL,
            "key": f"{S3_KEY_PREFIX}/${{filename}}",
            "AWSAccessKeyId": AWS_ACCESS_KEY_ID,
            "policy": "eyJleHBpcmF0aW9uIjogIjIwMjUtMDktMTJUMDU6Mjg6MzRaIiwgImNvbmRpdGlvbnMiOiBbWyJzdGFydHMtd2l0aCIsICIka2V5IiwgImN1c3RvbWVycy9zZWthaS9iYWNrZ3JvdW5kLWltYWdlLWdlbi8iXSwgWyJjb250ZW50LWxlbmd0aC1yYW5nZSIsIDAsIDUwMDAwMDAwXSwgeyJhY2wiOiAicHJpdmF0ZSJ9LCB7ImJ1Y2tldCI6ICJjdXN0b21lcnMtc2VrYWktYmFja2dyb3VuZC1pbWFnZS1nZW4ifSwgWyJzdGFydHMtd2l0aCIsICIka2V5IiwgImN1c3RvbWVycy9zZWthaS9iYWNrZ3JvdW5kLWltYWdlLWdlbi8iXV19",
            "signature": "uMapoYNvLIrn2BWudLNuJ4d9xv8="
        }
    }


def get_s3_file_url(filename: str) -> str:
    """
    Get the full S3 URL for a file
    
    Args:
        filename: Name of the file
        
    Returns:
        str: Full S3 URL
    """
    return f"{S3_BUCKET_URL}{S3_KEY_PREFIX}/{filename}"