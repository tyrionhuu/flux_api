#!/usr/bin/env python3
"""
Test script for Sekai API updates.
Tests NSFW detection and S3 upload functionality.
"""

import json
import requests
import time
import sys
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8080"  # nginx load balancer
# API_BASE_URL = "http://localhost:8000"  # single instance for dev


def test_generate_with_nsfw_check(prompt: str, enable_nsfw: bool = True) -> dict:
    """Test image generation with NSFW detection"""
    print(f"\n{'='*60}")
    print(f"Testing: {prompt}")
    print(f"NSFW check: {enable_nsfw}")
    print(f"{'='*60}")
    
    payload = {
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "num_inference_steps": 15,
        "response_format": "json",
        "enable_nsfw_check": enable_nsfw
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=payload, timeout=120)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: Image generated")
            print(f"   Filename: {result.get('filename')}")
            print(f"   Generation time: {result.get('generation_time')}")
            return result
        else:
            print(f"❌ Failed: {response.text}")
            return {"error": response.text}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {"error": str(e)}


def test_s3_upload(prompt: str, s3_url: str) -> dict:
    """Test image generation with S3 upload"""
    print(f"\n{'='*60}")
    print(f"Testing S3 Upload: {prompt}")
    print(f"S3 URL (truncated): {s3_url[:50]}...")
    print(f"{'='*60}")
    
    payload = {
        "prompt": prompt,
        "width": 512,
        "height": 1024,
        "num_inference_steps": 15,
        "response_format": "s3",
        "s3_prefix": s3_url,
        "enable_nsfw_check": True,
        "upscale": "true"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=payload, timeout=180)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: Image uploaded to S3")
            print(f"   S3 URL: {result.get('data', {}).get('s3_url', '')[:50]}...")
            print(f"   NSFW Score: {result.get('data', {}).get('nsfw_score', 'N/A')}")
            print(f"   Image Hash: {result.get('data', {}).get('image_hash', 'N/A')}")
            print(f"   S3 Upload Status: {result.get('data', {}).get('s3_upload_status', 'N/A')}")
            return result
        else:
            print(f"❌ Failed: {response.text}")
            return {"error": response.text}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {"error": str(e)}


def test_nsfw_scoring() -> dict:
    """Test that NSFW scoring works without blocking"""
    print(f"\n{'='*60}")
    print("Testing NSFW Scoring (Non-blocking)")
    print(f"{'='*60}")
    
    # Test that content is generated regardless of NSFW score
    test_prompt = "A beautiful landscape with mountains and a lake at sunset"
    
    payload = {
        "prompt": test_prompt,
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,  # Faster for testing
        "response_format": "json",
        "enable_nsfw_check": True
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=payload, timeout=120)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Content generated successfully")
            print(f"   Note: NSFW scoring is informational only, does not block generation")
            return {"success": True}
        else:
            print(f"❌ Unexpected response: {response.text}")
            return {"error": response.text}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {"error": str(e)}


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print(" SEKAI API TEST SUITE")
    print("="*60)
    
    # Test 1: Basic generation with NSFW check
    print("\n[Test 1] Basic generation with NSFW check")
    test_generate_with_nsfw_check(
        "A cute cat playing with a ball of yarn",
        enable_nsfw=True
    )
    
    # Test 2: Generation without NSFW check
    print("\n[Test 2] Generation without NSFW check")
    test_generate_with_nsfw_check(
        "A futuristic city skyline at night",
        enable_nsfw=False
    )
    
    # Test 3: NSFW scoring (non-blocking)
    print("\n[Test 3] NSFW scoring (non-blocking)")
    test_nsfw_scoring()
    
    # Test 4: S3 upload (requires valid pre-signed URL)
    print("\n[Test 4] S3 Upload Test")
    print("⚠️  Note: This test requires a valid pre-signed S3 URL")
    print("   Skipping S3 test (set S3_TEST_URL environment variable to enable)")
    
    import os
    s3_test_url = os.environ.get("S3_TEST_URL")
    if s3_test_url:
        test_s3_upload(
            "A serene Japanese garden with cherry blossoms",
            s3_test_url
        )
    
    print("\n" + "="*60)
    print(" TEST SUITE COMPLETE")
    print("="*60)


def test_single_endpoint():
    """Test a single endpoint for development"""
    print("\n Testing single /generate endpoint...")
    
    payload = {
        "prompt": "A peaceful mountain landscape",
        "width": 512,
        "height": 512,
        "num_inference_steps": 15,
        "response_format": "json",
        "enable_nsfw_check": True
    }
    
    try:
        print(f"Sending request to {API_BASE_URL}/generate")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/generate", json=payload, timeout=120)
        elapsed = time.time() - start_time
        
        print(f"\nResponse received in {elapsed:.2f}s")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Failed!")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        test_single_endpoint()
    else:
        run_all_tests()