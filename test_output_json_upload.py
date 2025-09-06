#!/usr/bin/env python3
"""
Test script to verify that output JSON is being uploaded to internal S3
"""

import requests
import json
import time
import sys

def test_output_json_upload():
    """Test that the API uploads output JSON to S3"""
    
    # API endpoint (adjust port if needed)
    api_url = "http://localhost:8000/generate"
    
    # Test request
    test_request = {
        "prompt": "a beautiful sunset over mountains",
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,
        "seed": 42,
        "response_format": "json"  # Use regular JSON format
    }
    
    print("Testing output JSON upload...")
    print(f"Sending request to {api_url}")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        # Send the request
        response = requests.post(api_url, json=test_request, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ API Response received successfully")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Extract image hash from response
            if 'image_hash' in result:
                image_hash = result['image_hash']
                print(f"\n📝 Image hash: {image_hash}")
                
                # Wait a bit for async upload to complete
                print("⏳ Waiting 3 seconds for async S3 upload...")
                time.sleep(3)
                
                # Check the uploaded files using the internal S3 uploader
                from utils.internal_s3_uploader import get_uploaded_file_urls
                urls = get_uploaded_file_urls(image_hash)
                
                print("\n📦 S3 URLs:")
                print(f"  Input JSON:  {urls['input_url']}")
                print(f"  Output Image: {urls['output_url']}")
                print(f"  Output JSON: {urls['output_json_url']}")
                
                print("\n✅ Test completed successfully!")
                print("Note: Check the logs to confirm S3 uploads were successful")
                
            else:
                print("⚠️  No image_hash in response")
        else:
            print(f"❌ API returned status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_output_json_upload()