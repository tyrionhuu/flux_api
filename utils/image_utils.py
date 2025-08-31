"""
Image utilities for the Diffusion API
"""

from typing import Any

from PIL import Image


def extract_image_from_result(result: Any) -> Image.Image:
    """Extract image from Diffusion pipeline result"""
    try:
        # Handle different possible return types from Diffusion pipeline
        if hasattr(result, "images") and result.images:
            return result.images[0]
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            # If result is a tuple/list, try to find the image
            for item in result:
                if isinstance(item, Image.Image):
                    return item
                elif hasattr(item, "images") and item.images:
                    return item.images[0]
        elif isinstance(result, Image.Image):
            return result

        # Fallback: create a simple placeholder image
        print("Warning: Could not extract image from result, using placeholder")
        raise ValueError("Could not extract image from result")

    except Exception as e:
        print(f"Error extracting image from result: {e}")
        raise ValueError(f"Error extracting image from result: {e}")


def save_image_with_unique_name(
    image: Image.Image, directory: str = "generated_images"
) -> str:
    """Save image with a unique filename and return the filename"""
    import os
    import uuid

    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/{uuid.uuid4()}.png"
    image.save(filename)
    return filename
