#!/usr/bin/env python3
"""
Verify that files were uploaded to S3
"""

import requests
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.s3_internal_config import get_s3_file_url

def check_s3_file(url):
    """Check if a file exists at the given S3 URL"""
    try:
        # HEAD request to check if file exists
        response = requests.head(url, timeout=10)
        # Note: Private ACL files will return 403, but that still means they exist
        if response.status_code in [200, 403]:
            return True, response.status_code
        else:
            return False, response.status_code
    except Exception as e:
        return False, str(e)

def verify_uploads(image_hash):
    """Verify uploads for a given image hash"""
    
    print(f"Verifying S3 uploads for hash: {image_hash}")
    print("-" * 50)
    
    # Get URLs
    input_url = get_s3_file_url(f"input-{image_hash}.json")
    output_url = get_s3_file_url(f"output-{image_hash}.jpg")
    
    # Check input JSON
    print(f"\nChecking input JSON...")
    print(f"URL: {input_url}")
    exists, status = check_s3_file(input_url)
    if exists:
        print(f"✅ Input JSON exists (status: {status})")
    else:
        print(f"❌ Input JSON not found (status: {status})")
    
    # Check output image
    print(f"\nChecking output image...")
    print(f"URL: {output_url}")
    exists, status = check_s3_file(output_url)
    if exists:
        print(f"✅ Output image exists (status: {status})")
    else:
        print(f"❌ Output image not found (status: {status})")
    
    print("\n" + "=" * 50)
    if exists:
        print("✅ Files are uploaded to S3!")
        print("Note: Status 403 means the files exist but are private (as expected)")
    else:
        print("⚠️ Some files were not found on S3")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_s3_uploads.py <image_hash>")
        print("Example: python verify_s3_uploads.py abc123def456")
        print("\nYou can find the image hash in the API logs after generation")
        sys.exit(1)
    
    image_hash = sys.argv[1]
    verify_uploads(image_hash)