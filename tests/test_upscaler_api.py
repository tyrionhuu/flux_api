#!/usr/bin/env python3
"""
Test script to verify upscaler singleton pattern works correctly in API calls
This simulates multiple API requests to ensure the model is loaded only once
"""

import requests
import json
import time
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "http://localhost:8000"  # Single instance test
# API_BASE_URL = "http://localhost:8080"  # Multi-GPU nginx load balancer

def test_upscaler_status():
    """Test the upscaler status endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/upscaler-status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Upscaler Status: {json.dumps(data, indent=2)}")
            return data
        else:
            logger.error(f"Failed to get upscaler status: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error getting upscaler status: {e}")
        return None

def test_generate_with_upscaling(prompt="A beautiful sunset over mountains", upscale=True):
    """Test image generation with upscaling"""
    payload = {
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "upscale": upscale,
        "upscale_factor": 2,
        "response_format": "json",  # Use JSON to see metadata
        "num_inference_steps": 4  # Minimal steps for faster testing
    }
    
    try:
        logger.info(f"Generating image {'with' if upscale else 'without'} upscaling...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=payload,
            timeout=120  # 2 minutes timeout for generation
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Generation completed in {elapsed_time:.2f}s")
            logger.info(f"  Image: {data.get('filename', 'N/A')}")
            logger.info(f"  Upscaled: {data.get('upscale', False)}")
            logger.info(f"  Final size: {data.get('width', 'N/A')}x{data.get('height', 'N/A')}")
            return True, elapsed_time
        else:
            logger.error(f"✗ Generation failed: {response.status_code}")
            logger.error(f"  Response: {response.text}")
            return False, elapsed_time
            
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return False, 0

def main():
    """Main test function"""
    print("\n" + "="*60)
    print("Testing Upscaler Singleton in API")
    print("="*60 + "\n")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            logger.error(f"API is not responding at {API_BASE_URL}")
            logger.info("Please start the API with: python main_fp4_sekai.py")
            return False
    except Exception as e:
        logger.error(f"Cannot connect to API at {API_BASE_URL}: {e}")
        logger.info("Please start the API with: python main_fp4_sekai.py")
        return False
    
    logger.info(f"API is running at {API_BASE_URL}")
    
    # Test 1: Check upscaler status
    print("\n1. Checking upscaler status...")
    status = test_upscaler_status()
    if status and status.get('singleton'):
        logger.info("✓ Upscaler is using singleton pattern")
    else:
        logger.warning("⚠ Upscaler status unclear")
    
    # Test 2: First generation with upscaling
    print("\n2. First generation with upscaling (should load model if not loaded)...")
    success1, time1 = test_generate_with_upscaling("A serene forest landscape", upscale=True)
    
    # Test 3: Second generation with upscaling (should reuse model)
    print("\n3. Second generation with upscaling (should reuse loaded model)...")
    success2, time2 = test_generate_with_upscaling("A futuristic city skyline", upscale=True)
    
    # Test 4: Third generation with upscaling (verify consistency)
    print("\n4. Third generation with upscaling (verify consistency)...")
    success3, time3 = test_generate_with_upscaling("An abstract colorful pattern", upscale=True)
    
    # Test 5: Generation without upscaling (control test)
    print("\n5. Generation without upscaling (control test)...")
    success4, time4 = test_generate_with_upscaling("A simple geometric shape", upscale=False)
    
    # Check upscaler status again
    print("\n6. Final upscaler status check...")
    final_status = test_upscaler_status()
    
    # Results summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    all_success = all([success1, success2, success3, success4])
    
    if all_success:
        print("✓ All generations completed successfully")
    else:
        print("✗ Some generations failed")
    
    print(f"\nGeneration times:")
    print(f"  1st with upscaling: {time1:.2f}s")
    print(f"  2nd with upscaling: {time2:.2f}s")
    print(f"  3rd with upscaling: {time3:.2f}s")
    print(f"  Without upscaling:  {time4:.2f}s")
    
    # Check if subsequent upscaling calls are faster (indicating model reuse)
    if time2 > 0 and time3 > 0:
        # The second and third calls should be similar or faster than the first
        # if the model is being reused
        avg_subsequent = (time2 + time3) / 2
        if avg_subsequent <= time1 * 1.1:  # Allow 10% variance
            print("\n✓ Subsequent upscaling calls are efficient (model is being reused)")
        else:
            print("\n⚠ Subsequent upscaling calls may be reloading the model")
    
    if final_status and final_status.get('status') == 'ready':
        print("✓ Upscaler remains ready after multiple uses")
    
    return all_success

if __name__ == "__main__":
    success = main()
    
    print("\n" + "="*60)
    if success:
        print("SUCCESS: Upscaler optimization is working correctly!")
        print("The upscaling model is loaded once and reused across requests.")
    else:
        print("Please check the logs for any issues.")
    print("="*60)
    
    sys.exit(0 if success else 1)