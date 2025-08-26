"""
Image utilities for the FLUX API
"""

from PIL import Image
from typing import Any, Optional
from config.fp4_settings import DEFAULT_IMAGE_SIZE, PLACEHOLDER_COLORS
import os
import uuid
from fastapi import UploadFile, HTTPException


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


def validate_uploaded_image(file: UploadFile) -> None:
    """Validate uploaded image file"""
    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Check file size (max 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")

    # Check file extension
    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    file_extension = os.path.splitext(file.filename or "")[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format. Allowed: {', '.join(allowed_extensions)}",
        )


def save_uploaded_image(file: UploadFile, directory: str = "uploads/images") -> str:
    """Save uploaded image and return the file path"""
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Generate unique filename
    file_extension = os.path.splitext(file.filename or "")[1].lower()
    if not file_extension:
        file_extension = ".png"

    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(directory, unique_filename)

    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)
        return file_path
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded image: {str(e)}"
        )


def load_and_preprocess_image(
    file_path: str, target_size: tuple = (512, 512)
) -> Image.Image:
    """Load and preprocess uploaded image for model input"""
    try:
        # Load image
        image = Image.open(file_path)

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)

        return image
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load and preprocess image: {str(e)}"
        )


def cleanup_uploaded_image(file_path: str) -> None:
    """Clean up uploaded image file after processing"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Failed to cleanup uploaded image {file_path}: {e}")
