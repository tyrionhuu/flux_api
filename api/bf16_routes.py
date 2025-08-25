"""
API routes for the BF16 FLUX API (Port 8001)
This reuses the existing route logic to avoid code duplication.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from models.bf16_flux_model import BF16FluxModelManager
from api.models import GenerateRequest, ModelStatusResponse
from config.bf16_settings import DEFAULT_LORA_NAME, DEFAULT_LORA_WEIGHT
from utils.image_utils import extract_image_from_result, save_image_with_unique_name
from utils.system_utils import get_system_memory
import time

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global bf16 model manager instance - singleton pattern
_bf16_model_manager_instance = None


def get_bf16_model_manager():
    global _bf16_model_manager_instance
    if _bf16_model_manager_instance is None:
        _bf16_model_manager_instance = BF16FluxModelManager()
    return _bf16_model_manager_instance


bf16_model_manager = get_bf16_model_manager()


@router.get("/download/{filename}")
def download_image(filename: str):
    """Download a generated image file"""
    import os
    from pathlib import Path
    from fastapi.responses import FileResponse

    # Security: only allow files from generated_images directory
    safe_filename = os.path.basename(filename)  # Remove any path traversal
    file_path = Path("generated_images") / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    if not file_path.suffix.lower() in [
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".webp",
    ]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=safe_filename,
        headers={"Content-Disposition": f"attachment; filename={safe_filename}"},
    )


@router.get("/")
def read_root():
    """Root endpoint for testing"""
    return {
        "message": "BF16 FLUX API is running!",
        "endpoints": [
            "/static-image",
            "/generate",
            "/loras",
            "/apply-lora",
            "/remove-lora",
            "/lora-status",
        ],
        "model_loaded": bf16_model_manager.is_loaded(),
        "model_type": bf16_model_manager.model_type,
    }


@router.post("/generate")
async def generate_image(request: GenerateRequest):
    """Generate image using BF16 FLUX model with optional LoRA support"""
    try:
        # Validate request parameters
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Handle multiple LoRA support
        loras_to_apply = []
        remove_all_loras = False

        # Check for new multiple LoRA format first
        if request.loras is not None:
            if len(request.loras) == 0:
                remove_all_loras = True
            else:
                for lora_config in request.loras:
                    if not lora_config.name or not lora_config.name.strip():
                        raise HTTPException(
                            status_code=400, detail="LoRA name cannot be empty"
                        )
                    if lora_config.weight < 0 or lora_config.weight > 2.0:
                        raise HTTPException(
                            status_code=400,
                            detail="LoRA weight must be between 0 and 2.0",
                        )
                    loras_to_apply.append(
                        {"name": lora_config.name.strip(), "weight": lora_config.weight}
                    )
        # Legacy support for single LoRA
        elif request.lora_name:
            if not request.lora_name.strip():
                raise HTTPException(
                    status_code=400, detail="LoRA name cannot be empty if provided"
                )
            if (
                request.lora_weight is None
                or request.lora_weight < 0
                or request.lora_weight > 2.0
            ):
                raise HTTPException(
                    status_code=400, detail="LoRA weight must be between 0 and 2.0"
                )
            loras_to_apply.append(
                {"name": request.lora_name.strip(), "weight": request.lora_weight}
            )

        # Apply default only when loras is None and no legacy field
        if (
            not loras_to_apply
            and not remove_all_loras
            and request.loras is None
            and not request.lora_name
        ):
            loras_to_apply = [
                {"name": DEFAULT_LORA_NAME, "weight": DEFAULT_LORA_WEIGHT}
            ]

        # Clean up input
        prompt = request.prompt.strip()

        # First, ensure the model is loaded
        if not bf16_model_manager.is_loaded():
            logger.info("BF16 model not loaded, loading it first...")
            if not bf16_model_manager.load_model():
                raise HTTPException(
                    status_code=500, detail="Failed to load BF16 FLUX model"
                )
            logger.info("BF16 model loaded successfully")

        # Now check if LoRAs are already applied
        current_lora = bf16_model_manager.get_lora_info()
        lora_applied = None
        lora_weight_applied = None

        if loras_to_apply:
            # Apply multiple LoRAs simultaneously
            logger.info(f"Applying {len(loras_to_apply)} LoRAs to loaded BF16 model")
            try:
                # Validate all LoRA names first
                for lora_config in loras_to_apply:
                    if not lora_config["name"]:
                        raise HTTPException(
                            status_code=400, detail="LoRA name cannot be empty"
                        )

                    # Allow uploaded file paths (uploads/lora_files/...)
                    if lora_config["name"].startswith("uploaded_lora_"):
                        # This is an uploaded file, validate it exists
                        import os

                        upload_path = f"uploads/lora_files/{lora_config['name']}"
                        if not os.path.exists(upload_path):
                            raise HTTPException(
                                status_code=400,
                                detail=f"Uploaded LoRA file not found: {lora_config['name']}",
                            )
                    elif "/" not in lora_config["name"]:
                        # Must be a Hugging Face repository ID or local path
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid LoRA name format for '{lora_config['name']}'. Must be a Hugging Face repository ID (e.g., 'username/model-name'), local path, or uploaded file path",
                        )

                # Apply all LoRAs at once using the new method
                if not bf16_model_manager.apply_multiple_loras(loras_to_apply):
                    logger.error(
                        f"Multiple LoRA application failed - Model: {bf16_model_manager.is_loaded()}, Pipeline: {bf16_model_manager.get_pipeline() is None}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to apply LoRAs. Please check if the LoRAs exist and are compatible.",
                    )
                else:
                    logger.info(f"All {len(loras_to_apply)} LoRAs applied successfully")

                # Get the updated LoRA info
                current_lora = bf16_model_manager.get_lora_info()
                if current_lora:
                    lora_applied = current_lora.get("name")
                    lora_weight_applied = current_lora.get("weight")
                    logger.info(
                        f"Multiple LoRAs applied successfully to BF16 model. Current LoRAs: {lora_applied} with total weight {lora_weight_applied}"
                    )
            except Exception as lora_error:
                logger.error(
                    f"Exception during LoRA application to BF16 model: {lora_error} (Type: {type(lora_error).__name__})"
                )
                if "not found" in str(lora_error).lower() or "404" in str(lora_error):
                    raise HTTPException(
                        status_code=400,
                        detail=f"One or more LoRAs not found. Please check the repository IDs.",
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to apply LoRAs to BF16 model: {str(lora_error)}",
                    )
        elif remove_all_loras:
            if bf16_model_manager.get_lora_info():
                logger.info(
                    "Removing all LoRAs from BF16 model as requested by client (empty list)"
                )
                bf16_model_manager.remove_lora()
                current_lora = None
                lora_applied = None
                lora_weight_applied = None
        else:
            if current_lora:
                lora_applied = current_lora.get("name")
                lora_weight_applied = current_lora.get("weight")

        result = generate_image_internal(
            prompt,
            "BF16_FLUX",
            lora_applied,
            lora_weight_applied,
            request.width or 512,
            request.height or 512,
            request.seed,
            request.upscale or False,
            request.upscale_factor or 2,
        )

        # Trigger cleanup after successful image generation
        try:
            from utils.cleanup_service import cleanup_after_generation

            cleanup_after_generation()
        except Exception as cleanup_error:
            logger.warning(
                f"Failed to trigger cleanup after generation: {cleanup_error}"
            )

        return result
    except Exception as e:
        logger.error(f"Request processing failed: {e} (Type: {type(e).__name__})")
        raise HTTPException(
            status_code=500, detail=f"Request processing failed: {str(e)}"
        )


@router.post("/load-model")
def load_model():
    """Load the BF16 FLUX model"""
    try:
        if bf16_model_manager.load_model():
            logger.info("BF16 FLUX model loaded successfully")
            return {"message": "BF16 FLUX model loaded successfully"}
        else:
            logger.error(
                f"Failed to load BF16 FLUX model - Model: {bf16_model_manager.is_loaded()}, Pipeline: {bf16_model_manager.get_pipeline() is not None}"
            )
            raise HTTPException(
                status_code=500, detail="Failed to load BF16 FLUX model"
            )
    except Exception as e:
        logger.error(f"Exception during model loading: {e} (Type: {type(e).__name__})")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


@router.get("/model-status")
def get_model_status():
    """Get the status of the BF16 FLUX model"""
    try:
        # Get current LoRA info
        lora_info = bf16_model_manager.get_lora_info()

        # Get system memory info
        system_used_gb, system_total_gb = get_system_memory()

        return ModelStatusResponse(
            model_loaded=bf16_model_manager.is_loaded(),
            model_type=bf16_model_manager.model_type,
            selected_gpu=bf16_model_manager.gpu_manager.selected_gpu,
            vram_usage_gb=f"{bf16_model_manager.gpu_manager.get_vram_usage():.2f}GB",
            system_memory_used_gb=f"{system_used_gb:.2f}GB",
            system_memory_total_gb=f"{system_total_gb:.2f}GB",
            lora_loaded=lora_info.get("name") if lora_info else None,
            lora_weight=lora_info.get("weight") if lora_info else None,
        )
    except Exception as e:
        logger.error(f"Exception during status check: {e} (Type: {type(e).__name__})")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/apply-lora")
def apply_lora(lora_name: str, lora_weight: float = 1.0):
    """Apply LoRA to the BF16 FLUX model"""
    try:
        if not bf16_model_manager.is_loaded():
            raise HTTPException(status_code=400, detail="Model not loaded")

        if bf16_model_manager.apply_lora(lora_name, lora_weight):
            logger.info(
                f"LoRA {lora_name} applied successfully with weight {lora_weight}"
            )
            return {"message": f"LoRA {lora_name} applied successfully"}
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to apply LoRA {lora_name}"
            )
    except Exception as e:
        logger.error(
            f"Exception during LoRA application: {e} (Type: {type(e).__name__})"
        )
        raise HTTPException(
            status_code=500, detail=f"LoRA application failed: {str(e)}"
        )


@router.post("/remove-lora")
def remove_lora():
    """Remove LoRA from the BF16 FLUX model"""
    try:
        if not bf16_model_manager.is_loaded():
            raise HTTPException(status_code=400, detail="Model not loaded")

        if bf16_model_manager.remove_lora():
            logger.info("LoRA removed successfully")
            return {"message": "LoRA removed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove LoRA")
    except Exception as e:
        logger.error(f"Exception during LoRA removal: {e} (Type: {type(e).__name__})")
        raise HTTPException(status_code=500, detail=f"LoRA removal failed: {str(e)}")


@router.get("/lora-status")
def get_lora_status():
    """Get the current LoRA status for the BF16 FLUX model"""
    try:
        lora_info = bf16_model_manager.get_lora_info()
        if lora_info:
            return {
                "lora_applied": True,
                "lora_name": lora_info["name"],
                "lora_weight": lora_info["weight"],
                "model_type": lora_info["model_type"],
            }
        else:
            return {"lora_applied": False, "model_type": "bf16"}
    except Exception as e:
        logger.error(
            f"Exception during LoRA status check: {e} (Type: {type(e).__name__})"
        )
        raise HTTPException(
            status_code=500, detail=f"LoRA status check failed: {str(e)}"
        )


@router.get("/loras")
def get_available_loras():
    """Get available LoRAs for the BF16 FLUX model"""
    try:
        # This would typically query a database or API for available LoRAs
        # For now, return a placeholder response
        return {"message": "LoRA discovery not implemented yet", "model_type": "bf16"}
    except Exception as e:
        logger.error(f"Exception during LoRA discovery: {e} (Type: {type(e).__name__})")
        raise HTTPException(status_code=500, detail=f"LoRA discovery failed: {str(e)}")


def generate_image_internal(
    prompt: str,
    model_type_name: str = "BF16_FLUX",
    lora_applied: Optional[str] = None,
    lora_weight: Optional[float] = None,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    upscale: bool = False,
    upscale_factor: int = 2,
):
    """Internal function to generate images for BF16 model - adapted from original"""
    # Append "Use GHIBLISTYLE" to the start of the user prompt
    enhanced_prompt = f"Use GHIBLISTYLE, {prompt}"
    logger.info(f"Starting BF16 image generation for prompt: {enhanced_prompt}")

    # Model should already be loaded at this point
    if not bf16_model_manager.is_loaded():
        raise HTTPException(status_code=500, detail="BF16 model not loaded")

    # Apply LoRA if specified and different from the currently applied one
    if lora_applied:
        current_info = bf16_model_manager.get_lora_info()
        should_apply = (
            not current_info
            or current_info.get("name") != lora_applied
            or (lora_weight is not None and current_info.get("weight") != lora_weight)
        )
        if should_apply:
            logger.info(
                f"Applying LoRA {lora_applied} (weight={lora_weight or 1.0}) to loaded BF16 model"
            )
            try:
                if not bf16_model_manager.apply_lora(lora_applied, lora_weight or 1.0):
                    logger.error(
                        f"Failed to apply LoRA {lora_applied} to loaded BF16 model"
                    )
                else:
                    logger.info(
                        f"LoRA {lora_applied} applied successfully to loaded BF16 model"
                    )
            except Exception as lora_error:
                logger.error(
                    f"Exception during LoRA application to loaded BF16 model: {lora_error} (Type: {type(lora_error).__name__})"
                )

    try:
        logger.info(f"Generating {model_type_name} image for prompt: {prompt}")

        # Start timing
        start_time = time.time()

        # Generate image with BF16 FLUX
        if bf16_model_manager.get_pipeline() is None:
            raise HTTPException(
                status_code=500,
                detail=f"{model_type_name} model not properly loaded",
            )

        # Generate the image
        result = bf16_model_manager.generate_image(
            enhanced_prompt,
            10,  # Fixed num_inference_steps
            4.0,  # Fixed guidance_scale
            width,
            height,
            seed,
            None,  # No negative prompt
        )

        image = extract_image_from_result(result)

        # End timing
        end_time = time.time()
        generation_time = end_time - start_time

        # Apply upscaling if requested
        try:
            from models.upscaler import apply_upscaling

            image_filename, upscaled_image_path, final_width, final_height = (
                apply_upscaling(
                    image, upscale, upscale_factor, save_image_with_unique_name
                )
            )
        except Exception as upscale_error:
            logger.error(f"Upscaling failed with error: {upscale_error}")
            # Fall back to saving original image without upscaling
            image_filename = save_image_with_unique_name(image)
            final_width = width
            final_height = height
            upscaled_image_path = None
            logger.info(f"Falling back to original image: {image_filename}")

        # Get system information
        vram_usage = bf16_model_manager.gpu_manager.get_vram_usage()
        system_memory_used, system_memory_total = get_system_memory()

        # Get the actual LoRA status from the model manager
        actual_lora_info = bf16_model_manager.get_lora_info()

        # Create download URL for the generated image
        import os

        filename = os.path.basename(image_filename)
        download_url = f"/download/{filename}"

        return {
            "message": f"Generated {model_type_name} image for prompt: {enhanced_prompt}",
            "image_url": image_filename,
            "download_url": download_url,
            "filename": filename,
            "generation_time": f"{generation_time:.2f}s",
            "lora_applied": actual_lora_info.get("name") if actual_lora_info else None,
            "lora_weight": actual_lora_info.get("weight") if actual_lora_info else None,
            "width": final_width,
            "height": final_height,
            "seed": seed,
        }

    except Exception as e:
        logger.error(f"Image generation failed: {e} (Type: {type(e).__name__})")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


@router.post("/upload-lora")
async def upload_lora_file(file: UploadFile = File(...)):
    """Upload a LoRA file to the server"""
    import os
    import shutil
    from pathlib import Path

    # Check file type
    allowed_extensions = [".safetensors", ".bin", ".pt", ".pth"]
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Check file size (max 1GB)
    max_size = 1024 * 1024 * 1024  # 1GB
    if not file.size or file.size > max_size:
        raise HTTPException(
            status_code=400, detail="File too large. Maximum size is 1GB."
        )

    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads/lora_files")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = int(time.time())
        safe_filename = f"uploaded_lora_{timestamp}{file_extension}"
        file_path = uploads_dir / safe_filename

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"LoRA file uploaded successfully: {file_path}")

        # Trigger cleanup after upload
        try:
            from utils.cleanup_service import cleanup_after_upload

            cleanup_after_upload()
        except Exception as cleanup_error:
            logger.warning(f"Failed to trigger cleanup after upload: {cleanup_error}")

        return {
            "message": "LoRA file uploaded successfully",
            "filename": safe_filename,
            "file_path": str(file_path),
            "size": file.size,
        }

    except Exception as e:
        logger.error(f"Error uploading LoRA file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload LoRA file: {str(e)}"
        )
