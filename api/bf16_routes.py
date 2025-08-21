"""
API routes for the BF16 FLUX API (Port 8001)
This reuses the existing route logic to avoid code duplication.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from models.bf16_flux_model import BF16FluxModelManager
from api.models import GenerateRequest, GenerateResponse, ModelStatusResponse
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

        if request.lora_name and not request.lora_name.strip():
            raise HTTPException(
                status_code=400, detail="LoRA name cannot be empty if provided"
            )

        if request.lora_weight < 0 or request.lora_weight > 2.0:
            raise HTTPException(
                status_code=400, detail="LoRA weight must be between 0 and 2.0"
            )

        # Clean up input
        prompt = request.prompt.strip()
        lora_name = request.lora_name.strip() if request.lora_name else None
        lora_weight = request.lora_weight

        # First, ensure the model is loaded
        if not bf16_model_manager.is_loaded():
            logger.info("BF16 model not loaded, loading it first...")
            if not bf16_model_manager.load_model():
                raise HTTPException(
                    status_code=500, detail="Failed to load BF16 FLUX model"
                )
            logger.info("BF16 model loaded successfully")

        # Now check if LoRA is already applied
        current_lora = bf16_model_manager.get_lora_info()
        lora_applied = None
        lora_weight_applied = None

        if lora_name:
            # Validate LoRA name format (should be a valid Hugging Face repo ID)
            if not lora_name or "/" not in lora_name:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid LoRA name format. Must be a Hugging Face repository ID (e.g., 'username/model-name')",
                )

            # Only apply LoRA if it's different from the current one
            if (
                not current_lora
                or current_lora.get("name") != lora_name
                or current_lora.get("weight") != lora_weight
            ):
                logger.info(f"Applying new LoRA: {lora_name} with weight {lora_weight}")
                try:
                    if bf16_model_manager.apply_lora(lora_name, lora_weight):
                        lora_applied = lora_name
                        lora_weight_applied = lora_weight
                        logger.info(
                            f"LoRA {lora_name} applied successfully with weight {lora_weight}"
                        )
                    else:
                        logger.error(
                            f"LoRA application failed: {lora_name} - Model: {bf16_model_manager.is_loaded()}, Pipeline: {bf16_model_manager.get_pipeline() is not None}"
                        )
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to apply LoRA {lora_name}. Please check if the LoRA exists and is compatible.",
                        )
                except Exception as lora_error:
                    logger.error(
                        f"Exception during LoRA application: {lora_error} (Type: {type(lora_error).__name__})"
                    )
                    if "not found" in str(lora_error).lower() or "404" in str(
                        lora_error
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=f"LoRA {lora_name} not found. Please check the repository ID.",
                        )
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to apply LoRA {lora_name}: {str(lora_error)}",
                        )
            else:
                # Use the already applied LoRA
                lora_applied = current_lora.get("name")
                lora_weight = current_lora.get("weight")
                logger.info(
                    f"Using already applied LoRA: {lora_applied} with weight {lora_weight}"
                )
        else:
            # No LoRA specified, check if one is currently applied
            if current_lora:
                lora_applied = current_lora.get("name")
                lora_weight = current_lora.get("weight")
                logger.info(
                    f"Using currently applied LoRA: {lora_applied} with weight {lora_weight}"
                )

        result = generate_image_internal(
            prompt,
            "BF16_FLUX",
            lora_applied,
            lora_weight_applied,
            request.width or 512,
            request.height or 512,
            request.seed,
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
):
    """Internal function to generate images for BF16 model - adapted from original"""
    # Append "Use GHIBLISTYLE" to the start of the user prompt
    enhanced_prompt = f"Use GHIBLISTYLE, {prompt}"
    logger.info(f"Starting BF16 image generation for prompt: {enhanced_prompt}")

    # Model should already be loaded at this point
    if not bf16_model_manager.is_loaded():
        raise HTTPException(status_code=500, detail="BF16 model not loaded")

    # Apply LoRA if specified and not already applied
    if lora_applied and not bf16_model_manager.get_lora_info():
        logger.info(f"Applying LoRA {lora_applied} to loaded BF16 model")
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

        # Save image with unique name to default directory
        image_filename = save_image_with_unique_name(image)

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
            "width": width,
            "height": height,
            "seed": seed,
        }

    except Exception as e:
        logger.error(f"Image generation failed: {e} (Type: {type(e).__name__})")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )
