#!/usr/bin/env python3
"""
Test script for internal S3 upload functionality
"""

import json
import time
import hashlib
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.internal_s3_uploader import (
    upload_json_to_s3,
    upload_image_to_s3,
    get_uploaded_file_urls
)


def test_internal_s3_upload():
    """Test uploading JSON and image to internal S3"""
    
    print("Testing internal S3 upload functionality...")
    print("-" * 50)
    
    # Generate a test hash
    test_hash = hashlib.md5(f"test_{time.time()}".encode()).hexdigest()
    print(f"Test hash: {test_hash}")
    
    # Test 1: Upload JSON data
    print("\n1. Testing JSON upload...")
    test_data = {
        "test": True,
        "timestamp": time.time(),
        "prompt": "Test prompt for S3 upload",
        "model": "FLUX",
        "generation_time": 5.23
    }
    
    json_filename = f"input-{test_hash}.json"
    success, error = upload_json_to_s3(test_data, json_filename)
    
    if success:
        print(f"✅ JSON upload successful: {json_filename}")
    else:
        print(f"❌ JSON upload failed: {error}")
        return False
    
    # Test 2: Upload test image
    print("\n2. Testing image upload...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color='blue')
    
    # Draw something on it to make it unique
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.text((10, 10), f"Test {test_hash[:8]}", fill='white')
    
    image_filename = f"output-{test_hash}.jpg"
    success, error = upload_image_to_s3(test_image, image_filename, jpeg_quality=70)
    
    if success:
        print(f"✅ Image upload successful: {image_filename}")
    else:
        print(f"❌ Image upload failed: {error}")
        return False
    
    # Test 3: Get S3 URLs
    print("\n3. Getting S3 URLs...")
    urls = get_uploaded_file_urls(test_hash)
    print(f"Input URL: {urls['input_url']}")
    print(f"Output URL: {urls['output_url']}")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed successfully!")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    try:
        success = test_internal_s3_upload()
        if success:
            print("\n🎉 Internal S3 upload is working correctly!")
        else:
            print("\n⚠️ Some tests failed. Check the errors above.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)