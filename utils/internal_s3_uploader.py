"""
Internal S3 uploader for storing generated images and request logs
Uses presigned POST similar to test_sekai_upload_s3.py
"""

import io
import json
import logging
import os
from typing import Dict, Any, Optional, Tuple
import requests
from PIL import Image
from config.s3_internal_config import (
    generate_presigned_post_data,
    get_s3_file_url,
    S3_UPLOAD_TIMEOUT,
    ENABLE_INTERNAL_S3_UPLOAD
)

logger = logging.getLogger(__name__)


def upload_json_to_s3(data: Dict[str, Any], filename: str) -> Tuple[bool, Optional[str]]:
    """
    Upload JSON data to S3 using presigned POST
    
    Args:
        data: Dictionary to upload as JSON
        filename: Filename for the JSON file (e.g., 'input-abc123.json')
        
    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)
    """
    if not ENABLE_INTERNAL_S3_UPLOAD:
        logger.info("Internal S3 upload is disabled")
        return True, None
        
    try:
        # Generate presigned POST data
        post_data = generate_presigned_post_data(filename)
        
        # Convert data to JSON
        json_data = json.dumps(data, indent=2).encode('utf-8')
        
        # Prepare the multipart form data
        files = {
            'file': (filename, io.BytesIO(json_data), 'application/json')
        }
        
        # Update the key field to use the actual filename
        fields = post_data['fields'].copy()
        fields['key'] = fields['key'].replace('${filename}', filename)
        
        # Upload to S3
        response = requests.post(
            post_data['url'],
            data=fields,
            files=files,
            timeout=S3_UPLOAD_TIMEOUT
        )
        
        if response.status_code == 204:
            logger.info(f"Successfully uploaded JSON to S3: {filename}")
            return True, None
        else:
            error_msg = f"Failed to upload JSON to S3. Status: {response.status_code}, Response: {response.text[:200]}"
            logger.error(error_msg)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Error uploading JSON to S3: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def upload_image_to_s3(image: Image.Image, filename: str, jpeg_quality: int = 70) -> Tuple[bool, Optional[str]]:
    """
    Upload PIL Image to S3 using presigned POST
    
    Args:
        image: PIL Image to upload
        filename: Filename for the image (e.g., 'output-abc123.jpg')
        jpeg_quality: JPEG quality (1-100)
        
    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)
    """
    if not ENABLE_INTERNAL_S3_UPLOAD:
        logger.info("Internal S3 upload is disabled")
        return True, None
        
    try:
        # Generate presigned POST data
        post_data = generate_presigned_post_data(filename)
        
        # Convert image to JPEG bytes
        img_buffer = io.BytesIO()
        if image.mode in ('RGBA', 'LA', 'P'):
            # Convert images with alpha channel to RGB
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(img_buffer, format='JPEG', quality=jpeg_quality, optimize=True)
        img_buffer.seek(0)
        
        # Prepare the multipart form data
        files = {
            'file': (filename, img_buffer, 'image/jpeg')
        }
        
        # Update the key field to use the actual filename
        fields = post_data['fields'].copy()
        fields['key'] = fields['key'].replace('${filename}', filename)
        
        # Upload to S3
        response = requests.post(
            post_data['url'],
            data=fields,
            files=files,
            timeout=S3_UPLOAD_TIMEOUT
        )
        
        if response.status_code == 204:
            logger.info(f"Successfully uploaded image to S3: {filename}")
            return True, None
        else:
            error_msg = f"Failed to upload image to S3. Status: {response.status_code}, Response: {response.text[:200]}"
            logger.error(error_msg)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Error uploading image to S3: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def upload_image_file_to_s3(image_path: str, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Upload image file to S3 using presigned POST
    
    Args:
        image_path: Path to the image file
        filename: Filename for the image in S3
        
    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)
    """
    if not ENABLE_INTERNAL_S3_UPLOAD:
        logger.info("Internal S3 upload is disabled")
        return True, None
        
    try:
        # Generate presigned POST data
        post_data = generate_presigned_post_data(filename)
        
        # Read the image file
        with open(image_path, 'rb') as f:
            files = {
                'file': (filename, f, 'image/jpeg')
            }
            
            # Update the key field to use the actual filename
            fields = post_data['fields'].copy()
            fields['key'] = fields['key'].replace('${filename}', filename)
            
            # Upload to S3
            response = requests.post(
                post_data['url'],
                data=fields,
                files=files,
                timeout=S3_UPLOAD_TIMEOUT
            )
        
        if response.status_code == 204:
            logger.info(f"Successfully uploaded image file to S3: {filename}")
            return True, None
        else:
            error_msg = f"Failed to upload image file to S3. Status: {response.status_code}, Response: {response.text[:200]}"
            logger.error(error_msg)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Error uploading image file to S3: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def get_uploaded_file_urls(image_hash: str) -> Dict[str, str]:
    """
    Get the S3 URLs for uploaded files
    
    Args:
        image_hash: The hash used in the filenames
        
    Returns:
        dict: URLs for input JSON, output image, and output JSON
    """
    return {
        "input_url": get_s3_file_url(f"input-{image_hash}.json"),
        "output_url": get_s3_file_url(f"output-{image_hash}.jpg"),
        "output_json_url": get_s3_file_url(f"output-{image_hash}.json")
    }