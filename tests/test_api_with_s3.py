#!/usr/bin/env python3
"""
Test the API with internal S3 upload
"""

import requests
import json
import time
import sys

# API endpoint
API_URL = "http://localhost:8000/generate"

def test_api_generation():
    """Test API generation with S3 upload"""
    
    print("Testing API generation with internal S3 upload...")
    print("-" * 50)
    
    # Test request
    request_data = {
        "prompt": "A beautiful sunset over mountains",
        "width": 512,
        "height": 512,
        "seed": 42,
        "num_inference_steps": 10,  # Reduced for faster testing
        "response_format": "binary"  # Using binary to test internal S3
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    print("\nSending request to API...")
    
    try:
        response = requests.post(API_URL, json=request_data, timeout=120)
        
        if response.status_code == 200:
            # Save the image
            with open("test_output.jpg", "wb") as f:
                f.write(response.content)
            print(f"✅ Image generated successfully!")
            print(f"   Saved to: test_output.jpg")
            print(f"   Response size: {len(response.content)} bytes")
            
            # Wait a bit for async S3 upload to complete
            print("\nWaiting 5 seconds for async S3 upload to complete...")
            time.sleep(5)
            
            print("\n✅ Test successful! Check the logs for S3 upload status.")
            print("   Look for lines containing 'Internal S3:' in the API logs")
            
            return True
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Note: Make sure the API is running (python main_fp4_sekai.py)")
    print("=" * 50)
    
    success = test_api_generation()
    
    if success:
        print("\n🎉 API test with S3 upload completed!")
        print("\nTo verify S3 uploads, check the API logs for:")
        print("  - 'Triggered async internal S3 upload for hash: ...'")
        print("  - 'Internal S3: Successfully uploaded input-...json'")
        print("  - 'Internal S3: Successfully uploaded output-...jpg'")
    else:
        print("\n⚠️ API test failed")
        sys.exit(1)