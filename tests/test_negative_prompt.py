#!/usr/bin/env python3
"""
Test script for negative prompt functionality
"""

import requests
import json
import time

# API endpoint (adjust if needed)
API_URL = "http://localhost:8000/generate"

def test_negative_prompt():
    """Test the negative prompt functionality"""
    
    # Test case 1: Without negative prompt
    print("Test 1: Generating image WITHOUT negative prompt...")
    payload1 = {
        "prompt": "A beautiful sunset over mountains",
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,
        "response_format": "json"
    }
    
    try:
        response1 = requests.post(API_URL, json=payload1)
        print(f"  Status: {response1.status_code}")
        if response1.status_code == 200:
            print("  ✅ Success - Image generated without negative prompt")
        else:
            print(f"  ❌ Failed: {response1.text}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()
    
    # Test case 2: With negative prompt
    print("Test 2: Generating image WITH negative prompt...")
    payload2 = {
        "prompt": "A beautiful sunset over mountains",
        "negative_prompt": "dark, cloudy, rainy, stormy, ugly, blurry",
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,
        "response_format": "json"
    }
    
    try:
        response2 = requests.post(API_URL, json=payload2)
        print(f"  Status: {response2.status_code}")
        if response2.status_code == 200:
            print("  ✅ Success - Image generated with negative prompt")
            result = response2.json()
            print(f"  Image saved at: {result.get('image_url')}")
        else:
            print(f"  ❌ Failed: {response2.text}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()
    
    # Test case 3: Queue submission with negative prompt
    print("Test 3: Submitting to queue WITH negative prompt...")
    queue_url = "http://localhost:8000/submit-request"
    payload3 = {
        "prompt": "A serene lake with mountains",
        "negative_prompt": "people, buildings, pollution, boats",
        "width": 512,
        "height": 512,
    }
    
    try:
        response3 = requests.post(queue_url, json=payload3)
        print(f"  Status: {response3.status_code}")
        if response3.status_code == 200:
            result = response3.json()
            request_id = result.get("request_id")
            print(f"  ✅ Success - Request queued with ID: {request_id}")
            
            # Check status
            time.sleep(2)
            status_url = f"http://localhost:8000/request-status/{request_id}"
            status_response = requests.get(status_url)
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"  Status: {status_data.get('status')}")
                print(f"  Negative prompt in queue: {status_data.get('negative_prompt')}")
        else:
            print(f"  ❌ Failed: {response3.text}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    print("🧪 Testing Negative Prompt Support for Sekai API\n")
    print("=" * 50)
    test_negative_prompt()