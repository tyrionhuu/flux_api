#!/usr/bin/env python3
"""
Standalone test for NSFW detection to verify it's working properly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
from utils.nsfw_detector import check_image_nsfw

def test_nsfw_detection():
    """Test NSFW detection with a simple generated image"""
    
    print("Testing NSFW Detection Module")
    print("="*50)
    
    # Create a simple test image (random pixels)
    print("\n1. Creating test image (random pixels)...")
    width, height = 224, 224
    random_pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    test_image = Image.fromarray(random_pixels, 'RGB')
    print(f"   Created {width}x{height} RGB image")
    
    # Test NSFW detection
    print("\n2. Running NSFW detection...")
    try:
        score = check_image_nsfw(test_image, timeout=10.0)
        print(f"   ✅ NSFW Score: {score:.3f}")
        
        if score == 1.0:
            print("\n   ⚠️ Score is 1.0 - this might indicate:")
            print("      - Detection failed (check logs)")
            print("      - Model not loaded properly")
            print("      - Timeout occurred")
        else:
            print(f"\n   ✅ Detection working! Score: {score:.3f}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Create a solid color image (should be safe)
    print("\n3. Testing with solid blue image...")
    blue_pixels = np.full((height, width, 3), [0, 0, 255], dtype=np.uint8)
    blue_image = Image.fromarray(blue_pixels, 'RGB')
    
    try:
        score = check_image_nsfw(blue_image, timeout=10.0)
        print(f"   ✅ NSFW Score: {score:.3f}")
        
        if score < 0.5:
            print("   ✅ Solid color correctly identified as safe")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n" + "="*50)
    print("Test Complete!")
    return True


if __name__ == "__main__":
    success = test_nsfw_detection()
    sys.exit(0 if success else 1)