#!/usr/bin/env python3
"""
Quick test to verify S3 upload fix
"""

import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.async_tasks import run_async_task
from utils.internal_s3_uploader import upload_json_to_s3, upload_image_file_to_s3
from PIL import Image
import hashlib

def test_async_upload():
    """Test the async upload with the fix"""
    
    print("Testing async S3 upload with os import fix...")
    
    # Create test data
    test_hash = hashlib.md5(f"test_fix_{time.time()}".encode()).hexdigest()
    
    # Create and save a test image
    test_image = Image.new('RGB', (100, 100), color='green')
    test_image_path = f"/tmp/test_{test_hash}.jpg"
    test_image.save(test_image_path, 'JPEG', quality=70)
    
    def async_upload():
        """Async function similar to the one in the API"""
        import os  # Import inside function like in the fix
        try:
            # Upload JSON
            input_data = {"test": "data", "hash": test_hash}
            input_filename = f"input-{test_hash}.json"
            success, error = upload_json_to_s3(input_data, input_filename)
            if success:
                print(f"✅ JSON uploaded: {input_filename}")
            else:
                print(f"❌ JSON upload failed: {error}")
            
            # Upload image using file path
            output_filename = f"output-{test_hash}.jpg"
            if os.path.exists(test_image_path):  # This should work now
                success, error = upload_image_file_to_s3(test_image_path, output_filename)
                if success:
                    print(f"✅ Image uploaded: {output_filename}")
                else:
                    print(f"❌ Image upload failed: {error}")
            else:
                print("❌ Test image file not found")
                
        except Exception as e:
            print(f"❌ Error in async upload: {str(e)}")
    
    # Run async
    run_async_task(async_upload)
    
    # Wait for completion
    print("Waiting for async upload to complete...")
    time.sleep(3)
    
    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print(f"\nTest hash: {test_hash}")
    print("Check the output above to verify both uploads succeeded")

if __name__ == "__main__":
    test_async_upload()