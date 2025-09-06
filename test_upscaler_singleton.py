#!/usr/bin/env python3
"""
Test script to verify upscaler singleton pattern works correctly
"""

import sys
import time
import logging

# Configure logging to see upscaler initialization messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the current directory to path to import modules
sys.path.insert(0, '/data/pingzhi/flux_api')

from models.upscaler import get_upscaler

def test_singleton_pattern():
    """Test that the upscaler is only loaded once"""
    print("\n" + "="*60)
    print("Testing Upscaler Singleton Pattern")
    print("="*60 + "\n")
    
    # First call - should initialize the upscaler
    print("1. Getting upscaler instance (first time)...")
    start_time = time.time()
    upscaler1 = get_upscaler()
    first_load_time = time.time() - start_time
    print(f"   First load took: {first_load_time:.2f} seconds")
    print(f"   Instance ID: {id(upscaler1)}")
    print(f"   Is ready: {upscaler1.is_ready()}")
    
    # Second call - should return the same instance quickly
    print("\n2. Getting upscaler instance (second time)...")
    start_time = time.time()
    upscaler2 = get_upscaler()
    second_load_time = time.time() - start_time
    print(f"   Second load took: {second_load_time:.2f} seconds")
    print(f"   Instance ID: {id(upscaler2)}")
    print(f"   Is ready: {upscaler2.is_ready()}")
    
    # Third call - to be absolutely sure
    print("\n3. Getting upscaler instance (third time)...")
    start_time = time.time()
    upscaler3 = get_upscaler()
    third_load_time = time.time() - start_time
    print(f"   Third load took: {third_load_time:.2f} seconds")
    print(f"   Instance ID: {id(upscaler3)}")
    print(f"   Is ready: {upscaler3.is_ready()}")
    
    # Verify all instances are the same
    print("\n" + "="*60)
    print("Verification Results:")
    print("="*60)
    
    same_instance = (upscaler1 is upscaler2) and (upscaler2 is upscaler3)
    print(f"✓ All instances are the same object: {same_instance}")
    
    if same_instance:
        print("✓ Singleton pattern is working correctly!")
    else:
        print("✗ Singleton pattern is NOT working - instances are different!")
    
    # Check load time improvement
    print(f"\n✓ First load time: {first_load_time:.2f}s")
    print(f"✓ Second load time: {second_load_time:.2f}s (should be near 0)")
    print(f"✓ Third load time: {third_load_time:.2f}s (should be near 0)")
    
    if second_load_time < 0.1 and third_load_time < 0.1:
        print("✓ Subsequent loads are fast (< 0.1s) - model is being reused!")
    else:
        print("⚠ Subsequent loads are taking time - check if model is being reloaded")
    
    # Show model info
    print("\nModel Information:")
    model_info = upscaler1.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    return same_instance

if __name__ == "__main__":
    success = test_singleton_pattern()
    
    print("\n" + "="*60)
    if success:
        print("SUCCESS: Upscaler singleton pattern is working correctly!")
        print("The model is loaded only once and reused for all requests.")
    else:
        print("FAILURE: Upscaler singleton pattern is not working.")
        print("The model is being loaded multiple times.")
    print("="*60)
    
    sys.exit(0 if success else 1)