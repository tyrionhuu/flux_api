"""
Image utilities for the FLUX API
"""

from typing import Any, Optional

from PIL import Image

from config.fp4_settings import DEFAULT_IMAGE_SIZE, PLACEHOLDER_COLORS


def extract_image_from_result(result: Any) -> Image.Image:
    """Extract image from FLUX pipeline result"""
    try:
        # Handle different possible return types from FLUX pipeline
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
        return Image.new(
            "RGB", DEFAULT_IMAGE_SIZE, color=PLACEHOLDER_COLORS["placeholder"]
        )

    except Exception as e:
        print(f"Error extracting image from result: {e}")
        return Image.new("RGB", DEFAULT_IMAGE_SIZE, color=PLACEHOLDER_COLORS["error"])


def create_placeholder_image(
    text: str = "Placeholder", color: Optional[str] = None
) -> Image.Image:
    """Create a placeholder image with optional text"""
    if color is None:
        color = PLACEHOLDER_COLORS["default"]

    img = Image.new("RGB", DEFAULT_IMAGE_SIZE, color=color)
    return img


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
