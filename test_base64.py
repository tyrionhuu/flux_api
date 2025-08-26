#!/usr/bin/env python3
"""
Test script to verify base64 image conversion functionality
"""

from PIL import Image
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_base64_conversion():
    """Test the base64 image conversion function"""
    try:
        from utils.image_utils import image_to_base64
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Convert to base64
        base64_string = image_to_base64(test_image, "PNG")
        
        print("✅ Base64 conversion successful!")
        print(f"Base64 string length: {len(base64_string)}")
        print(f"Starts with: {base64_string[:50]}...")
        print(f"Contains data URI prefix: {'data:image/png;base64,' in base64_string}")
        
        return True
        
    except Exception as e:
        print(f"❌ Base64 conversion failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing base64 image conversion...")
    success = test_base64_conversion()
    
    if success:
        print("\n🎉 All tests passed! The API should now return base64 images.")
        print("\nYou can now use your curl command and get the image directly in the response:")
        print("curl -X POST 'http://localhost:8000/generate' \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"prompt\": \"A beautiful landscape\", \"width\": 512, \"height\": 512}'")
        print("\nThe response will include an 'image_base64' field with the encoded image data.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1)
