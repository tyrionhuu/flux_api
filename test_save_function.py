#!/usr/bin/env python3
"""
Test script for save_image_with_unique_name function
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_save_function():
    """Test the save_image_with_unique_name function"""
    try:
        from utils.image_utils import save_image_with_unique_name
        from PIL import Image

        print("🚀 Testing save_image_with_unique_name function...")

        # Create a test image
        test_image = Image.new("RGB", (512, 512), color="red")

        # Test saving
        print("✅ Testing image saving...")
        filename = save_image_with_unique_name(test_image)
        print(f"   Saved image to: {filename}")

        # Check if file exists
        if os.path.exists(filename):
            print(f"   ✅ File exists: {os.path.getsize(filename)} bytes")
        else:
            print(f"   ❌ File does not exist")

        # Clean up
        if os.path.exists(filename):
            os.remove(filename)
            print(f"   🧹 Cleaned up test file")

        print(f"\n🎯 Save function test completed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_save_function()
    if success:
        print("\n🎉 All tests completed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
