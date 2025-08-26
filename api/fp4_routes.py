"""
API routes for the FLUX API
"""

import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.models import GenerateRequest
from config.fp4_settings import (DEFAULT_LORA_NAME, DEFAULT_LORA_WEIGHT,
                                 STATIC_IMAGES_DIR)
from models.fp4_flux_model import FluxModelManager
from utils.image_utils import (extract_image_from_result,
                               save_image_with_unique_name)
from utils.queue_manager import QueueManager
from utils.system_utils import get_system_memory

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global model manager instance - singleton pattern with thread safety
import threading

_model_manager_instance = None
_model_manager_lock = threading.Lock()


def get_model_manager():
    global _model_manager_instance
    if _model_manager_instance is None:
        with _model_manager_lock:
            # Double-check pattern for thread safety
            if _model_manager_instance is None:
                _model_manager_instance = FluxModelManager()
    return _model_manager_instance


model_manager = get_model_manager()

# Global queue manager instance
queue_manager = QueueManager(max_concurrent=2, max_queue_size=100)


@router.get("/debug-version")
def debug_version():
    """Debug endpoint to check code version"""
    import time

    return {
        "status": "debug",
        "service": "FLUX API",
        "code_version": "enhanced_v2",
        "timestamp": time.time(),
        "thread_safe_model_check": True,
    }


@router.post("/test-form")
async def test_form_endpoint(
    prompt: str = Form(...),
    test_param: Optional[str] = Form(None),
):
    """Test endpoint to verify form handling works"""
    return {
        "status": "success",
        "message": "Form test successful",
        "prompt": prompt,
        "test_param": test_param,
        "timestamp": time.time()
    }


@router.get("/")
def read_root():
    """Root endpoint for testing"""
    logger.info("=== ROOT ENDPOINT CALLED ===")
    return {
        "message": "FLUX API is running!",
        "endpoints": [
            "/static-image",
            "/generate",
            "/generate-with-image",
            "/loras",
            "/apply-lora",
            "/remove-lora",
            "/lora-status",
        ],
        "model_loaded": model_manager.is_loaded(),
        "model_type": model_manager.model_type,
        "server_port": os.environ.get("FP4_API_PORT", "8000"),
        "timestamp": time.time(),
    }


@router.get("/static-image")
def get_static_image():
    """Serve static images"""
    image_path = f"{STATIC_IMAGES_DIR}/sample.jpg"
    return FileResponse(image_path)


@router.get("/download/{filename}")
def download_image(filename: str):
    """Download a generated image file"""
    import os
    from pathlib import Path

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


@router.post("/generate")
async def generate_image(request: GenerateRequest):
    """Generate image using FLUX model with optional LoRA support - supports multiple LoRAs"""
    try:
        # Debug logging for incoming requests
        logger.info(f"=== GENERATE ENDPOINT CALLED ===")
        logger.info(f"Request received: {request}")
        logger.info(f"Prompt: {request.prompt}")
        logger.info(f"Dimensions: {request.width}x{request.height}")
        logger.info(f"LoRAs: {request.loras}")
        logger.info(f"Seed: {request.seed}")
        
        # Validate request parameters
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Handle multiple LoRA support
        loras_to_apply = []
        remove_all_loras = False

        # Check for new multiple LoRA format first
        if request.loras is not None:
            if len(request.loras) == 0:
                # Explicitly requested to use NO LoRA
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

        # Apply default LoRA only when client did not send loras at all (None) and no legacy fields
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

        # First, ensure the model is loaded with thread safety and force check
        with _model_manager_lock:
            # Force a more thorough check - sometimes the state gets inconsistent
            model_actually_loaded = (
                model_manager.is_loaded()
                and model_manager.get_pipeline() is not None
                and hasattr(model_manager.get_pipeline(), "transformer")
            )

            if not model_actually_loaded:
                logger.info(
                    "Model not properly loaded or pipeline unavailable, loading it first..."
                )
                # Check again if another thread loaded it while we were waiting
                if not (
                    model_manager.is_loaded()
                    and model_manager.get_pipeline() is not None
                ):
                    if not model_manager.load_model():
                        raise HTTPException(
                            status_code=500, detail="Failed to load FLUX model"
                        )
                    logger.info("Model loaded successfully")
                else:
                    logger.info("Model was loaded by another thread while waiting")

        # Now check if LoRAs are already applied
        current_lora = model_manager.get_lora_info()
        lora_applied = None
        lora_weight_applied = None

        if loras_to_apply:
            # Apply multiple LoRAs simultaneously
            logger.info(f"Applying {len(loras_to_apply)} LoRAs to loaded model")
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
                if not model_manager.apply_multiple_loras(loras_to_apply):
                    logger.error(
                        f"Multiple LoRA application failed - Model: {model_manager.is_loaded()}, Pipeline: {model_manager.get_pipeline() is None}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to apply LoRAs. Please check if the LoRAs exist and are compatible.",
                    )
                else:
                    logger.info(f"All {len(loras_to_apply)} LoRAs applied successfully")

                # Get the updated LoRA info
                current_lora = model_manager.get_lora_info()
                if current_lora:
                    lora_applied = current_lora.get("name")
                    lora_weight_applied = current_lora.get("weight")
                    logger.info(
                        f"Multiple LoRAs applied successfully. Current LoRAs: {lora_applied} with total weight {lora_weight_applied}"
                    )
            except Exception as lora_error:
                logger.error(
                    f"Exception during LoRA application: {lora_error} (Type: {type(lora_error).__name__})"
                )
                if "not found" in str(lora_error).lower() or "404" in str(lora_error):
                    raise HTTPException(
                        status_code=400,
                        detail=f"One or more LoRAs not found. Please check the repository IDs.",
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to apply LoRAs: {str(lora_error)}",
                    )
        elif remove_all_loras:
            # Explicit removal requested
            if model_manager.get_lora_info():
                logger.info("Removing all LoRAs as requested by client (empty list)")
                model_manager.remove_lora()
                current_lora = None
                lora_applied = None
                lora_weight_applied = None
        else:
            # No-op
            if current_lora:
                lora_applied = current_lora.get("name")
                lora_weight_applied = current_lora.get("weight")

        result = generate_image_internal(
            prompt,
            "FLUX",
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


@router.post("/generate-with-image")
async def generate_with_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    num_inference_steps: Optional[int] = Form(25),
    guidance_scale: Optional[float] = Form(2.5),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(512),
    seed: Optional[int] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    prompt_prefix: Optional[str] = Form(None),
):
    """Generate image using image + text input (image-to-image generation)"""
    try:
        from PIL import Image
        import io
        
        # Debug logging for form parameters
        logger.info(f"Received generate-with-image request - prompt: {prompt}, num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, width: {width}, height: {height}, seed: {seed}")
        logger.info(f"Image file: {image.filename}, content_type: {image.content_type}, size: {image.size}")
        
        # Apply prompt prefix if provided, otherwise use the original prompt
        if prompt_prefix:
            enhanced_prompt = f"{prompt_prefix}, {prompt}"
        else:
            enhanced_prompt = prompt
            
        # Validate parameters
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
            
        if num_inference_steps and (num_inference_steps < 1 or num_inference_steps > 100):
            raise HTTPException(status_code=400, detail="num_inference_steps must be between 1 and 100")
            
        if guidance_scale and (guidance_scale < 0.1 or guidance_scale > 20.0):
            raise HTTPException(status_code=400, detail="guidance_scale must be between 0.1 and 20.0")
            
        if width and (width < 256 or width > 1024):
            raise HTTPException(status_code=400, detail="width must be between 256 and 1024")
            
        if height and (height < 256 or height > 1024):
            raise HTTPException(status_code=400, detail="height must be between 256 and 1024")
            
        # Load input image
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
            
        # Validate image file type
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Check file size (max 10MB)
        if image.size and image.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
            
        # Check file extension
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        file_extension = os.path.splitext(image.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format. Allowed: {', '.join(allowed_extensions)}"
            )
            
        try:
            raw = await image.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as img_error:
            logger.error(f"Failed to load image: {img_error}")
            raise HTTPException(status_code=400, detail=f"Failed to load image: {str(img_error)}")
        
        # Ensure model is loaded
        with _model_manager_lock:
            if not model_manager.is_loaded():
                logger.info("Model not loaded, loading it first...")
                if not model_manager.load_model():
                    raise HTTPException(status_code=500, detail="Failed to load model")
        
        # Start timing
        generation_start_time = time.time()
        
        # Generate image using the new method
        try:
            logger.info(f"Calling model_manager.generate_image_with_image with prompt: {enhanced_prompt}")
            result = model_manager.generate_image_with_image(
                prompt=enhanced_prompt,
                image=img,
                num_inference_steps=num_inference_steps or 25,
                guidance_scale=guidance_scale or 2.5,
                width=width or 512,
                height=height or 512,
                seed=seed,
                negative_prompt=negative_prompt,
            )
            logger.info(f"Model generation completed successfully, result type: {type(result)}")
        except Exception as gen_error:
            logger.error(f"Model generation failed: {gen_error}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(gen_error)}")
        
        # Calculate generation time
        generation_time = time.time() - generation_start_time
        
        # Extract and save the generated image
        try:
            generated_image = extract_image_from_result(result)
            if generated_image is None:
                raise RuntimeError("Failed to extract image from generation result")
            logger.info(f"Successfully extracted generated image: {type(generated_image)}")
        except Exception as extract_error:
            logger.error(f"Failed to extract image from result: {extract_error}")
            raise HTTPException(status_code=500, detail=f"Failed to extract generated image: {str(extract_error)}")
            
        try:
            image_filename = save_image_with_unique_name(generated_image)
            logger.info(f"Successfully saved image to: {image_filename}")
        except Exception as save_error:
            logger.error(f"Failed to save generated image: {save_error}")
            raise HTTPException(status_code=500, detail=f"Failed to save generated image: {str(save_error)}")
        
        # Convert image to base64 for direct response
        try:
            from utils.image_utils import image_to_base64
            image_base64 = image_to_base64(generated_image, "PNG")
        except Exception as base64_error:
            logger.warning(f"Failed to convert image to base64: {base64_error}")
            image_base64 = None
        
        # Return the result with download URL and base64
        return {
            "status": "success",
            "message": f"Image generated successfully for prompt: {enhanced_prompt}",
            "download_url": f"/download/{image_filename}",
            "filename": image_filename,
            "image_base64": image_base64,  # Base64 encoded image data
            "generation_time": f"{generation_time:.2f}s"
        }
        
    except Exception as e:
        logger.error(f"Error in generate_with_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-model")
def load_model():
    """Load the FLUX model"""
    try:
        with _model_manager_lock:
            # Enhanced check for proper model loading
            model_actually_loaded = (
                model_manager.is_loaded()
                and model_manager.get_pipeline() is not None
                and hasattr(model_manager.get_pipeline(), "transformer")
            )

            if model_actually_loaded:
                logger.info("FLUX model already properly loaded")
                return {"message": "FLUX model already loaded"}

            logger.info("Loading FLUX model...")
            if model_manager.load_model():
                logger.info("FLUX model loaded successfully")
                return {"message": "FLUX model loaded successfully"}
            else:
                logger.error(
                    f"Failed to load FLUX model - Model: {model_manager.is_loaded()}, Pipeline: {model_manager.get_pipeline() is not None}"
                )
                raise HTTPException(status_code=500, detail="Failed to load FLUX model")
    except Exception as e:
        logger.error(f"Exception during model loading: {e} (Type: {type(e).__name__})")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


@router.get("/model-status")
def get_model_status():
    """Get the current model status"""
    status = model_manager.get_model_status()
    system_memory_used, system_memory_total = get_system_memory()

    # Enhanced status with detailed model state
    pipeline = model_manager.get_pipeline()
    has_transformer = pipeline is not None and hasattr(pipeline, "transformer")

    status.update(
        {
            "system_memory_used_gb": f"{system_memory_used:.2f}GB",
            "system_memory_total_gb": f"{system_memory_total:.2f}GB",
            "lora_loaded": (model_manager.get_lora_info() or {}).get("name"),
            "lora_weight": (model_manager.get_lora_info() or {}).get("weight"),
            "pipeline_loaded": pipeline is not None,
            "has_transformer": has_transformer,
            "model_actually_ready": (
                model_manager.is_loaded() and pipeline is not None and has_transformer
            ),
        }
    )

    return status


@router.get("/gpu-info")
def get_gpu_info():
    """Get detailed GPU information"""
    return model_manager.gpu_manager.get_device_info()


@router.get("/loras")
def list_loras():
    """List all available LoRA files - only Hugging Face LoRAs supported"""
    return {
        "available_loras": [],
        "note": "Only Hugging Face LoRAs are supported. Use /apply-lora with a Hugging Face repository ID.",
    }


@router.post("/apply-lora")
async def apply_lora(lora_name: str, weight: float = 1.0):
    """Apply a LoRA to the current model - lora_name should be a Hugging Face repo ID (e.g., aleksa-codes/flux-ghibsky-illustration)"""
    if not model_manager.is_loaded():
        logger.error(f"Cannot apply LoRA {lora_name}: Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        if model_manager.apply_lora(lora_name, weight):
            logger.info(f"LoRA {lora_name} applied successfully with weight {weight}")
            return {
                "message": f"LoRA {lora_name} applied successfully with weight {weight}",
                "lora_name": lora_name,
                "weight": weight,
                "status": "applied",
            }
        else:
            logger.error(
                f"Failed to apply LoRA {lora_name} - Model: {model_manager.is_loaded()}, Pipeline: {model_manager.get_pipeline() is not None}"
            )
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
async def remove_lora():
    """Remove the currently applied LoRA from the model"""
    if not model_manager.is_loaded():
        logger.error(f"Cannot remove LoRA: Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        if model_manager.remove_lora():
            logger.info("LoRA removed successfully")
            return {"message": "LoRA removed successfully", "status": "removed"}
        else:
            logger.error(
                f"Failed to remove LoRA - Model: {model_manager.is_loaded()}, Pipeline: {model_manager.get_pipeline() is not None}"
            )
            raise HTTPException(status_code=500, detail="Failed to remove LoRA")
    except Exception as e:
        logger.error(f"Exception during LoRA removal: {e} (Type: {type(e).__name__})")
        raise HTTPException(status_code=500, detail=f"LoRA removal failed: {str(e)}")


@router.get("/lora-status")
def get_lora_status():
    """Get the current LoRA status"""
    return {
        "current_lora": model_manager.get_lora_info(),
        "note": "Only Hugging Face LoRAs are supported. Use /apply-lora to apply a LoRA.",
    }


# Queue management endpoints
@router.post("/submit-request")
async def submit_generation_request(request: GenerateRequest):
    """Submit a generation request to the queue"""
    try:
        # Validate request
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        if request.lora_name and not request.lora_name.strip():
            raise HTTPException(
                status_code=400, detail="LoRA name cannot be empty if provided"
            )

        if request.lora_weight is not None and (
            request.lora_weight < 0 or request.lora_weight > 2.0
        ):
            raise HTTPException(
                status_code=400, detail="LoRA weight must be between 0 and 2.0"
            )

        # Submit to queue
        request_id = await queue_manager.submit_request(
            prompt=request.prompt.strip(),
            lora_name=(
                request.loras[0].name
                if request.loras and len(request.loras) > 0
                else None
            ),
            lora_weight=(
                request.loras[0].weight
                if request.loras and len(request.loras) > 0
                else 1.0
            ),
            loras=(
                [{"name": lora.name, "weight": lora.weight} for lora in request.loras]
                if request.loras
                else None
            ),
            num_inference_steps=10,  # Fixed value
            guidance_scale=4.0,  # Fixed value
            width=request.width or 512,
            height=request.height or 512,
            seed=request.seed,
        )

        return {
            "message": "Request submitted successfully",
            "request_id": request_id,
            "status": "queued",
        }

    except Exception as e:
        logger.error(f"Failed to submit request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit request: {str(e)}"
        )


@router.get("/request-status/{request_id}")
async def get_request_status(request_id: str):
    """Get the status of a specific request"""
    try:
        request = await queue_manager.get_request_status(request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        return {
            "request_id": request.id,
            "status": request.status.value,
            "prompt": request.prompt,
            "lora_name": request.lora_name,
            "lora_weight": request.lora_weight,
            "created_at": request.created_at,
            "started_at": request.started_at,
            "completed_at": request.completed_at,
            "result": request.result,
            "error": request.error,
            # Generation parameters
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get request status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get request status: {str(e)}"
        )


@router.delete("/cancel-request/{request_id}")
async def cancel_request(request_id: str):
    """Cancel a pending request"""
    try:
        success = await queue_manager.cancel_request(request_id)
        if not success:
            raise HTTPException(
                status_code=404, detail="Request not found or already processing"
            )

        return {"message": "Request cancelled successfully", "request_id": request_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel request: {str(e)}"
        )


@router.get("/queue-stats")
async def get_queue_stats():
    """Get current queue statistics"""
    try:
        return queue_manager.get_queue_stats()
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get queue stats: {str(e)}"
        )


def generate_image_internal(
    prompt: str,
    model_type_name: str = "FLUX",
    lora_applied: Optional[str] = None,
    lora_weight: Optional[float] = None,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    upscale: bool = False,
    upscale_factor: int = 2,
    prompt_prefix: Optional[str] = None,
):
    """Internal function to generate images - used by both endpoints"""
    # Apply prompt prefix if provided, otherwise use the original prompt
    if prompt_prefix:
        enhanced_prompt = f"{prompt_prefix}, {prompt}"
    else:
        enhanced_prompt = prompt
    logger.info(f"Starting image generation for prompt: {enhanced_prompt}")

    # Model should already be loaded at this point
    if not model_manager.is_loaded():
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Apply LoRA if specified and different from the currently applied one
    if lora_applied:
        current_info = model_manager.get_lora_info()
        should_apply = (
            not current_info
            or current_info.get("name") != lora_applied
            or (lora_weight is not None and current_info.get("weight") != lora_weight)
        )
        if should_apply:
            logger.info(
                f"Applying LoRA {lora_applied} (weight={lora_weight or 1.0}) to loaded model"
            )
            try:
                if not model_manager.apply_lora(lora_applied, lora_weight or 1.0):
                    logger.error(f"Failed to apply LoRA {lora_applied} to loaded model")
                else:
                    logger.info(
                        f"LoRA {lora_applied} applied successfully to loaded model"
                    )
            except Exception as lora_error:
                logger.error(
                    f"Exception during LoRA application to loaded model: {lora_error} (Type: {type(lora_error).__name__})"
                )

    try:
        logger.info(f"Generating {model_type_name} image for prompt: {prompt}")

        # Start timing
        start_time = time.time()

        # Generate image with FLUX
        if model_manager.get_pipeline() is None:
            raise HTTPException(
                status_code=500,
                detail=f"{model_type_name} model not properly loaded",
            )

        # Generate the image
        result = model_manager.generate_image(
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
        vram_usage = model_manager.gpu_manager.get_vram_usage()
        system_memory_used, system_memory_total = get_system_memory()

        # Get the actual LoRA status from the model manager
        actual_lora_info = model_manager.get_lora_info()

        # Create download URL for the generated image
        import os

        filename = os.path.basename(image_filename)
        download_url = f"/download/{filename}"

        # Convert image to base64 for direct response
        try:
            from utils.image_utils import image_to_base64
            image_base64 = image_to_base64(image, "PNG")
        except Exception as base64_error:
            logger.warning(f"Failed to convert image to base64: {base64_error}")
            image_base64 = None

        return {
            "message": f"Generated {model_type_name} image for prompt: {enhanced_prompt}",
            "image_url": image_filename,
            "download_url": download_url,
            "filename": filename,
            "image_base64": image_base64,  # Base64 encoded image data
            "generation_time": f"{generation_time:.2f}s",
            "lora_applied": actual_lora_info.get("name") if actual_lora_info else None,
            "lora_weight": actual_lora_info.get("weight") if actual_lora_info else None,
            "width": final_width,
            "height": final_height,
            "seed": seed,
        }

    except Exception as e:
        logger.error(
            f"Error generating {model_type_name} image: {e} (Type: {type(e).__name__})"
        )
        raise HTTPException(
            status_code=500,
            detail=f"{model_type_name} image generation failed: {str(e)}",
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


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload a reference image to the server without generating."""
    try:
        from utils.image_utils import (save_uploaded_image,
                                       validate_uploaded_image)

        validate_uploaded_image(file)
        file_path = save_uploaded_image(file)  # saves to uploads/images by default

        import os

        filename = os.path.basename(file_path)

        return {
            "message": "Image uploaded successfully",
            "filename": filename,
            "file_path": file_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@router.post("/upload-image-generate")
async def upload_image_and_generate(
    file: UploadFile = File(None),
    prompt: str = Form(...),
    loras: Optional[str] = Form(None),
    lora_name: Optional[str] = Form(None),
    lora_weight: Optional[float] = Form(None),
    width: int = Form(512),
    height: int = Form(512),
    seed: Optional[int] = Form(None),
    upscale: Optional[str] = Form("false"),
    upscale_factor: int = Form(2),
    image_strength: float = Form(0.8),
    image_guidance_scale: float = Form(1.5),
    uploaded_image_path: Optional[str] = Form(None),
):
    """Generate image using uploaded image (file or previously uploaded path) and prompt with optional LoRA support"""
    try:
        from utils.image_utils import (cleanup_uploaded_image,
                                       load_and_preprocess_image,
                                       save_uploaded_image,
                                       validate_uploaded_image)

        temp_file_to_cleanup = None

        # Resolve input image source: either a freshly uploaded file or a server-side uploaded path
        if uploaded_image_path:
            # Use existing uploaded image on server (no validation possible here beyond path safety)
            input_image_path = uploaded_image_path
        else:
            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either file or uploaded_image_path must be provided",
                )
            # Validate and persist temporarily
            validate_uploaded_image(file)
            input_image_path = save_uploaded_image(file)
            temp_file_to_cleanup = input_image_path

        try:
            # Load and preprocess the uploaded image
            input_image = load_and_preprocess_image(input_image_path, (width, height))

            # Parse LoRA configuration
            loras_to_apply = []
            remove_all_loras = False

            if loras:
                import json

                try:
                    lora_configs = json.loads(loras)
                    for lora_config in lora_configs:
                        if (
                            not lora_config.get("name")
                            or not lora_config["name"].strip()
                        ):
                            raise HTTPException(
                                status_code=400, detail="LoRA name cannot be empty"
                            )
                        if (
                            lora_config.get("weight", 1.0) < 0
                            or lora_config.get("weight", 1.0) > 2.0
                        ):
                            raise HTTPException(
                                status_code=400,
                                detail="LoRA weight must be between 0 and 2.0",
                            )
                        loras_to_apply.append(
                            {
                                "name": lora_config["name"].strip(),
                                "weight": lora_config["weight"],
                            }
                        )
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=400, detail="Invalid LoRA configuration format"
                    )
            elif lora_name:
                if not lora_name.strip():
                    raise HTTPException(
                        status_code=400, detail="LoRA name cannot be empty if provided"
                    )
                if lora_weight is None or lora_weight < 0 or lora_weight > 2.0:
                    raise HTTPException(
                        status_code=400, detail="LoRA weight must be between 0 and 2.0"
                    )
                loras_to_apply.append(
                    {"name": lora_name.strip(), "weight": lora_weight}
                )

            # Apply default LoRA if none specified
            if not loras_to_apply and not remove_all_loras:
                loras_to_apply = [
                    {"name": DEFAULT_LORA_NAME, "weight": DEFAULT_LORA_WEIGHT}
                ]

            # Validate dimensions
            if width < 256 or width > 1024 or height < 256 or height > 1024:
                raise HTTPException(
                    status_code=400,
                    detail="Width and height must be between 256 and 1024",
                )

            # Validate upscale factor
            if upscale and upscale_factor not in [2, 4]:
                raise HTTPException(
                    status_code=400, detail="Upscale factor must be 2 or 4"
                )

            # Validate image strength and guidance
            if image_strength < 0.0 or image_strength > 1.0:
                raise HTTPException(
                    status_code=400, detail="Image strength must be between 0.0 and 1.0"
                )
            if image_guidance_scale < 1.0 or image_guidance_scale > 20.0:
                raise HTTPException(
                    status_code=400,
                    detail="Image guidance scale must be between 1.0 and 20.0",
                )

            # Clean up input
            prompt = prompt.strip()
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt cannot be empty")

            # Ensure model is loaded
            with _model_manager_lock:
                if not model_manager.is_loaded():
                    logger.info("Model not loaded, loading it first...")
                    model_manager.load_model()

                if not model_manager.is_loaded():
                    raise HTTPException(status_code=500, detail="Failed to load model")

            # Apply LoRAs if specified
            if loras_to_apply:
                for lora_config in loras_to_apply:
                    model_manager.apply_lora(lora_config["name"], lora_config["weight"])
            elif remove_all_loras:
                if model_manager.get_lora_info():
                    logger.info(
                        "Removing all LoRAs as requested by client (empty list)"
                    )
                    model_manager.remove_lora()

            # Generate image using the model (placeholder: regular generate with prompt)
            logger.info(
                f"Starting image generation with uploaded image and prompt: {prompt}"
            )

            pipeline = model_manager.get_pipeline()
            if not pipeline:
                raise HTTPException(
                    status_code=500, detail="Model pipeline not available"
                )

            if seed is not None:
                import torch

                torch.manual_seed(seed)

            generation_start_time = time.time()

            # Use image-to-image generation with the uploaded image
            result = model_manager.generate_image_with_image(
                prompt=prompt,
                image=input_image,  # Use the loaded input image
                num_inference_steps=10,
                guidance_scale=image_guidance_scale,
                seed=seed,
            )

            generation_time = time.time() - generation_start_time

            generated_image = extract_image_from_result(result)

            image_filename = save_image_with_unique_name(generated_image)

            if upscale and upscale.lower() in ["true", "1", "yes", "on"]:
                try:
                    from models.upscaler import apply_upscaling

                    image_filename, upscaled_image_path, final_width, final_height = (
                        apply_upscaling(
                            generated_image,
                            upscale,
                            upscale_factor,
                            save_image_with_unique_name,
                        )
                    )
                    logger.info(
                        f"Image upscaled by {upscale_factor}x: {image_filename}"
                    )
                except Exception as upscale_error:
                    logger.warning(f"Upscaling failed: {upscale_error}")
                    image_filename = save_image_with_unique_name(generated_image)

            # Create download URL
            import os

            filename = os.path.basename(image_filename)
            download_url = f"/download/{filename}"

            # Convert image to base64 for direct response
            try:
                from utils.image_utils import image_to_base64
                image_base64 = image_to_base64(generated_image, "PNG")
            except Exception as base64_error:
                logger.warning(f"Failed to convert image to base64: {base64_error}")
                image_base64 = None

            return {
                "message": f"Generated image from uploaded image and prompt: {prompt}",
                "image_url": image_filename,
                "download_url": download_url,
                "filename": filename,
                "image_base64": image_base64,  # Base64 encoded image data
                "generation_time": f"{generation_time:.2f}s",
                "width": width,
                "height": height,
                "seed": seed,
                "image_strength": image_strength,
                "image_guidance_scale": image_guidance_scale,
                "upscaled": upscale and upscale.lower() in ["true", "1", "yes", "on"],
                "upscale_factor": upscale_factor if upscale else None,
            }

        finally:
            # Only cleanup if we created a temp file in this request
            if temp_file_to_cleanup:
                cleanup_uploaded_image(temp_file_to_cleanup)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload-image-generate: {e}")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )
