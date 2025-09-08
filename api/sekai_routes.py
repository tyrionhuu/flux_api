"""
API routes for the FLUX API
"""

import asyncio
import logging
import os
import time
import traceback
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from models.fp4_flux_model import FluxModelManager
from utils.image_utils import extract_image_from_result, save_image_with_unique_name
from utils.system_utils import get_system_memory
from utils.queue_manager import QueueManager
from utils.request_queue import RequestQueueManager
from utils.async_tasks import run_async_task
from utils.internal_s3_uploader import (
    upload_json_to_s3,
    upload_image_to_s3,
    upload_image_file_to_s3,
    get_uploaded_file_urls
)
from config.sekai_settings import (
    STATIC_IMAGES_DIR,
    LORA_1_NAME,
    LORA_1_WEIGHT,
    LORA_2_NAME,
    LORA_2_WEIGHT,
    LORA_3_NAME,
    LORA_3_WEIGHT,
    SAVE_AS_JPEG,
    JPEG_QUALITY,
    MAX_CONCURRENT_REQUESTS,
    MAX_QUEUE_SIZE,
    REQUEST_TIMEOUT,
)
from api.models import GenerateRequest

DEFAULT_LORA_LIST = [
    {"name": LORA_1_NAME, "weight": LORA_1_WEIGHT},
    {"name": LORA_2_NAME, "weight": LORA_2_WEIGHT},
    {"name": LORA_3_NAME, "weight": LORA_3_WEIGHT},
]

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

# Preload upscaler model at startup (optional but recommended)
# This ensures the upscaler is ready before first use
PRELOAD_UPSCALER = os.environ.get("PRELOAD_UPSCALER", "true").lower() == "true"

if PRELOAD_UPSCALER:
    try:
        from models.upscaler import get_upscaler
        logger.info("Preloading upscaler model at startup...")
        upscaler = get_upscaler()
        if upscaler.is_ready():
            logger.info("Upscaler model preloaded successfully and ready for use")
        else:
            logger.warning("Upscaler model preloaded but not ready - will retry on first use")
    except Exception as e:
        logger.warning(f"Failed to preload upscaler model: {e}")
        logger.info("Upscaler will be loaded on first use")

# Global queue manager instance (for the old queue endpoints - not used in main /generate)
queue_manager = QueueManager(max_concurrent=2, max_queue_size=100)

# Global request queue manager for synchronous processing with async queuing
request_queue_manager = RequestQueueManager(
    max_concurrent=MAX_CONCURRENT_REQUESTS,
    max_queue_size=MAX_QUEUE_SIZE,
    request_timeout=REQUEST_TIMEOUT
)


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


@router.get("/")
def read_root():
    """Root endpoint for testing"""
    return {
        "message": "FLUX API is running!",
        "endpoints": [
            "/static-image",
            "/generate",
            "/loras",
            "/apply-lora",
            "/remove-lora",
            "/lora-status",
        ],
        "model_loaded": model_manager.is_loaded(),
        "model_type": model_manager.model_type,
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


def generate_image_sync(request: GenerateRequest):
    """Synchronous image generation function for use with the queue"""
    # This function contains the actual generation logic
    # It will be called by the queue manager in a thread pool
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
        loras_to_apply = DEFAULT_LORA_LIST

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
                    try:
                        if not model_manager.load_model():
                            logger.error("Model loading failed - load_model() returned False")
                            raise HTTPException(
                                status_code=500, detail="Failed to load FLUX model"
                            )
                        logger.info("Model loaded successfully")
                    except HTTPException:
                        raise  # Re-raise HTTP exceptions
                    except Exception as e:
                        logger.error(f"Model loading fucked up: {e}\n{traceback.format_exc()}")
                        raise HTTPException(
                            status_code=500, detail="Failed to load FLUX model"
                        )
                else:
                    logger.info("Model was loaded by another thread while waiting")

    # Now check if LoRAs are already applied
    current_lora = model_manager.get_lora_info()
    lora_applied = None
    lora_weight_applied = None

    # Check if current LoRAs match what we want to apply
    def loras_match(current, desired):
            """Check if current LoRAs match desired LoRAs"""
            if not current or not desired:
                return False
            
            # Get current LoRA names and weights
            if isinstance(current.get("name"), list):
                # Multiple LoRAs currently applied
                current_loras = current.get("name", [])
                current_weight = current.get("weight", 0)
                
                # Check if same number of LoRAs
                if len(current_loras) != len(desired):
                    return False
                
                # Check if all LoRAs match (order matters for merged LoRAs)
                for i, lora_config in enumerate(desired):
                    if i >= len(current_loras):
                        return False
                    if current_loras[i] != lora_config["name"]:
                        return False
                
                # Check if combined weight matches
                desired_weight = sum(lora["weight"] for lora in desired)
                if abs(current_weight - desired_weight) > 0.001:  # Small tolerance for float comparison
                    return False
                
                return True
            else:
                # Single LoRA currently applied
                if len(desired) != 1:
                    return False
                return (
                    current.get("name") == desired[0]["name"] 
                    and abs(current.get("weight", 0) - desired[0]["weight"]) < 0.001
                )
    
    # Only apply LoRAs if they don't match current state
    should_apply_loras = loras_to_apply and not loras_match(current_lora, loras_to_apply)
    
    if should_apply_loras:
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
                    f"LoRA application shit the bed: {lora_error}\n{traceback.format_exc()}"
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
    elif loras_to_apply and not should_apply_loras:
        # LoRAs already match what we want - no need to reapply
        logger.info(f"LoRAs already applied and match desired state, skipping re-application")
        if current_lora:
            lora_applied = current_lora.get("name")
            lora_weight_applied = current_lora.get("weight")
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

    # Pass both the merged LoRA info and the individual LoRA details
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
        request.response_format or "binary",
        request.s3_prefix,
        request.enable_nsfw_check if request.enable_nsfw_check is not None else True,
        request.num_inference_steps or 15,
        request.negative_prompt,
        loras_list=loras_to_apply if loras_to_apply else None,  # Pass individual LoRA details
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


@router.post("/generate")
async def generate_image(request: GenerateRequest):
    """Generate image using FLUX model with optional LoRA support - with request queuing"""
    try:
        # Use the request queue manager to handle queuing
        logger.info(f"Received generation request for prompt: {request.prompt[:50]}...")
        
        # Check queue status before processing
        queue_stats = request_queue_manager.get_queue_stats()
        logger.info(
            f"Queue status - Active: {queue_stats['active_requests']}, "
            f"Queued: {queue_stats['queued_requests']}"
        )
        
        # Process the request through the queue
        try:
            result = await request_queue_manager.process_request(
                generate_image_sync,
                request
            )
            return result
        except asyncio.QueueFull:
            logger.warning("Request queue is full")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Queue is full. Please try again later.",
                headers={
                    "Retry-After": "30",  # Suggest retry after 30 seconds
                    "X-Queue-Full": "true",
                    "X-Max-Queue-Size": str(queue_stats['max_queue_size'])
                }
            )
        except asyncio.TimeoutError:
            logger.error("Request timed out while waiting in queue")
            raise HTTPException(
                status_code=504,
                detail="Request timed out while waiting in queue. Please try again.",
                headers={"X-Timeout": "true"}
            )
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Request processing blew up: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Request processing failed: {str(e)}"
        )


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
        logger.error(f"Model loading died: {e}\n{traceback.format_exc()}")
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
        logger.error(f"LoRA apply broke ({lora_name}, weight={weight}): {e}\n{traceback.format_exc()}")
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
        logger.error(f"LoRA removal failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"LoRA removal failed: {str(e)}")


@router.get("/lora-status")
def get_lora_status():
    """Get the current LoRA status"""
    return {
        "current_lora": model_manager.get_lora_info(),
        "note": "Only Hugging Face LoRAs are supported. Use /apply-lora to apply a LoRA.",
    }


@router.get("/upscaler-status")
def get_upscaler_status():
    """Get the current upscaler status"""
    try:
        from models.upscaler import get_upscaler
        upscaler = get_upscaler()
        return {
            "status": "ready" if upscaler.is_ready() else "not_ready",
            "model_info": upscaler.get_model_info(),
            "singleton": True,
            "note": "Upscaler is loaded once and kept in memory for all requests"
        }
    except Exception as e:
        logger.error(f"Failed to get upscaler status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "singleton": True,
            "note": "Upscaler singleton pattern is enabled but model may not be available"
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
            num_inference_steps=20,  # Fixed value
            guidance_scale=4.0,  # Fixed value
            width=request.width or 512,
            height=request.height or 512,
            seed=request.seed,
            negative_prompt=request.negative_prompt,
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
            "negative_prompt": request.negative_prompt,
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


@router.get("/request-queue-stats")
async def get_request_queue_stats():
    """Get current request queue statistics for the main /generate endpoint"""
    try:
        stats = request_queue_manager.get_queue_stats()
        return {
            "queue_stats": stats,
            "status": "healthy" if stats["active_requests"] < MAX_CONCURRENT_REQUESTS or stats["queued_requests"] < MAX_QUEUE_SIZE else "busy"
        }
    except Exception as e:
        logger.error(f"Failed to get request queue stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get request queue stats: {str(e)}"
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
    response_format: str = "binary",
    s3_prefix: Optional[str] = None,
    enable_nsfw_check: bool = True,
    num_inference_steps: int = 15,
    negative_prompt: Optional[str] = None,
    loras_list: Optional[list] = None,  # New parameter for individual LoRA details
):
    """Internal function to generate images - used by both endpoints"""
    # Use the original prompt without modifications
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
            num_inference_steps,  # Use the parameter value
            4.0,  # Fixed guidance_scale
            width,
            height,
            seed,
            negative_prompt,  # Pass negative prompt
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
                    image, upscale, upscale_factor, 
                    lambda img: save_image_with_unique_name(img, save_as_jpeg=SAVE_AS_JPEG, jpeg_quality=JPEG_QUALITY),
                    save_as_jpeg=SAVE_AS_JPEG, jpeg_quality=JPEG_QUALITY
                )
            )
            # Update image reference to upscaled version if available
            if upscale and upscaled_image_path:
                # Load the upscaled image for NSFW check and S3 upload
                from PIL import Image as PILImage
                image = PILImage.open(upscaled_image_path)
        except Exception as upscale_error:
            logger.error(f"Upscaling failed with error: {upscale_error}")
            # Fall back to saving original image without upscaling
            image_filename = save_image_with_unique_name(image, save_as_jpeg=SAVE_AS_JPEG, jpeg_quality=JPEG_QUALITY)
            final_width = width
            final_height = height
            upscaled_image_path = None
            logger.info(f"Falling back to original image: {image_filename}")
        
        # Calculate image hash for consistent naming
        import hashlib
        import io
        from PIL import Image as PILImage
        img_buffer = io.BytesIO()
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = PILImage.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image_to_hash = rgb_image
        else:
            image_to_hash = image
        image_to_hash.save(img_buffer, format='JPEG', quality=JPEG_QUALITY)
        image_hash = hashlib.md5(img_buffer.getvalue()).hexdigest()
        
        # Build the response JSON first (before uploading to S3)
        # This will be used both for the API response and S3 upload
        output_json = None
        
        # Build response based on format
        if response_format == "s3":
            # This will be built later after S3 upload
            pass
        elif response_format == "binary":
            # For binary responses, create metadata JSON for S3 upload
            filename = os.path.basename(image_filename)
            output_json = {
                "message": f"Generated {model_type_name} image for prompt: {enhanced_prompt}",
                "response_format": "binary",
                "filename": filename,
                "generation_time": f"{generation_time:.2f}s",
                "lora_applied": model_manager.get_lora_info().get("name") if model_manager.get_lora_info() else None,
                "lora_weight": model_manager.get_lora_info().get("weight") if model_manager.get_lora_info() else None,
                "width": final_width,
                "height": final_height,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
                "negative_prompt": negative_prompt,
                "upscale": upscale,
                "upscale_factor": upscale_factor if upscale else None,
                "nsfw_score": 0.0,  # Will be updated after NSFW check
                "image_hash": image_hash
            }
        else:
            # Regular JSON response format
            filename = os.path.basename(image_filename)
            download_url = f"/download/{filename}"
            actual_lora_info = model_manager.get_lora_info()
            
            output_json = {
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
                "num_inference_steps": num_inference_steps,
                "negative_prompt": negative_prompt,
                "upscale": upscale,
                "upscale_factor": upscale_factor if upscale else None,
                "nsfw_score": 0.0,  # Will be updated after NSFW check
                "image_hash": image_hash
            }
        
        # Async upload to internal S3 bucket (fire and forget)
        # This happens regardless of response_format
        def upload_to_internal_s3(output_json_to_upload):
            """Background task to upload input, output image, and output JSON to S3"""
            import os  # Import inside function for thread safety
            try:
                # Prepare input request data
                input_data = {
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "model_type": model_type_name,
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "num_inference_steps": num_inference_steps,
                    "negative_prompt": negative_prompt,
                    "lora_applied": lora_applied,  # Keep for backward compatibility
                    "lora_weight": lora_weight,     # Keep for backward compatibility (summed weight)
                    "loras_details": loras_list,    # NEW: Individual LoRA names and weights
                    "upscale": upscale,
                    "upscale_factor": upscale_factor,
                    "response_format": response_format,
                    "s3_prefix": s3_prefix,
                    "enable_nsfw_check": enable_nsfw_check,
                    "timestamp": time.time(),
                    "generation_time": generation_time
                }
                
                # Upload input JSON
                input_filename = f"input-{image_hash}.json"
                success, error = upload_json_to_s3(input_data, input_filename)
                if success:
                    logger.info(f"Internal S3: Successfully uploaded {input_filename}")
                else:
                    logger.error(f"Internal S3: Failed to upload {input_filename}: {error}")
                
                # Upload output image
                output_image_filename = f"output-{image_hash}.jpg"
                # Try to use the saved file first (more efficient)
                if image_filename and os.path.exists(image_filename):
                    success, error = upload_image_file_to_s3(image_filename, output_image_filename)
                else:
                    # Fall back to PIL image
                    success, error = upload_image_to_s3(image, output_image_filename, JPEG_QUALITY)
                    
                if success:
                    logger.info(f"Internal S3: Successfully uploaded {output_image_filename}")
                else:
                    logger.error(f"Internal S3: Failed to upload {output_image_filename}: {error}")
                
                # Upload output JSON (the API response)
                if output_json_to_upload:
                    output_json_filename = f"output-{image_hash}.json"
                    success, error = upload_json_to_s3(output_json_to_upload, output_json_filename)
                    if success:
                        logger.info(f"Internal S3: Successfully uploaded {output_json_filename}")
                    else:
                        logger.error(f"Internal S3: Failed to upload {output_json_filename}: {error}")
                    
                # Log the S3 URLs for reference
                urls = get_uploaded_file_urls(image_hash)
                logger.info(f"Internal S3 URLs - Input: {urls['input_url']}, Output Image: {urls['output_url']}, Output JSON: {urls.get('output_json_url', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Internal S3 upload error: {str(e)}")
        
        # NSFW Detection (do this before S3 upload to include in output JSON)
        nsfw_score = 0.0
        nsfw_error_message = None
        if enable_nsfw_check:
            try:
                from utils.nsfw_detector import check_image_nsfw
                logger.info("Running NSFW detection...")
                nsfw_score = check_image_nsfw(image, timeout=5.0)
                logger.info(f"NSFW check complete: score={nsfw_score:.3f}")
                        
            except Exception as nsfw_error:
                logger.error(f"NSFW detection error: {nsfw_error}")
                # On error, set score to 1.0 but continue processing
                nsfw_score = 1.0
                nsfw_error_message = "NSFW detection encountered an error; score set to 1.0"
                logger.warning(
                    "NSFW detection failed, setting score to 1.0 and continuing"
                )
        
        # Update NSFW score in output_json if it was created
        if output_json:
            output_json["nsfw_score"] = nsfw_score
            if nsfw_error_message:
                output_json["nsfw_error"] = nsfw_error_message
        
        # Trigger async upload for non-S3 formats
        if response_format != "s3" and output_json:
            run_async_task(lambda: upload_to_internal_s3(output_json))
            logger.info(f"Triggered async internal S3 upload for hash: {image_hash}")
        
        # S3 Upload for response_format="s3"
        if response_format == "s3":
            if not s3_prefix:
                raise HTTPException(
                    status_code=400,
                    detail="s3_prefix is required when response_format='s3'"
                )
            
            try:
                from utils.s3_uploader import upload_to_s3
                logger.info(f"Uploading to S3 with presigned URL...")
                
                # Use the already calculated image hash from above
                # Upload using the presigned URL as provided (no filename rewrite)
                success, error, s3_url, http_status = upload_to_s3(
                    image,
                    s3_prefix,
                    jpeg_quality=JPEG_QUALITY,
                    image_hash=image_hash
                )
                
                if not success:
                    logger.error(f"S3 upload failed: {error}, HTTP status: {http_status}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"S3 upload failed: {error}"
                    )
                
                logger.info(f"S3 upload successful, HTTP status: {http_status}")
                
                # Build minimal output JSON for S3 format (only expose necessary fields)
                output_json = {
                    "data": {
                        "s3_url": s3_url,
                        "nsfw_score": nsfw_score,
                        "image_hash": image_hash,
                        "s3_upload_status": http_status,
                    }
                }
                if nsfw_error_message:
                    output_json["data"]["nsfw_error"] = nsfw_error_message
                
                # Trigger async upload with output JSON
                run_async_task(lambda: upload_to_internal_s3(output_json))
                logger.info(f"Triggered async internal S3 upload for hash: {image_hash}")
                
                # Log the final response JSON
                logger.info(f"API Response: {output_json}")
                
                return output_json
                
            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except Exception as s3_error:
                logger.error(f"S3 upload error: {s3_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"S3 upload failed: {str(s3_error)}"
                )

        # Get system information
        vram_usage = model_manager.gpu_manager.get_vram_usage()
        system_memory_used, system_memory_total = get_system_memory()

        # Get the actual LoRA status from the model manager
        actual_lora_info = model_manager.get_lora_info()

        # Create download URL for the generated image
        import os

        filename = os.path.basename(image_filename)
        download_url = f"/download/{filename}"

        # Return binary image if requested
        if response_format == "binary":
            from fastapi.responses import FileResponse
            # Determine media type based on configuration
            media_type = "image/jpeg" if SAVE_AS_JPEG else "image/png"
            # Log the output JSON even though we're returning binary
            logger.info(f"API Response (metadata for binary): {output_json}")
            return FileResponse(
                image_filename,
                media_type=media_type,
                filename=filename,
                headers={"Content-Disposition": f"inline; filename={filename}"}
            )

        # Log the final response JSON
        logger.info(f"API Response: {output_json}")
        
        return output_json

    except Exception as e:
        logger.error(f"Image generation crapped out: {e}\n{traceback.format_exc()}")
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
        from utils.image_utils import validate_uploaded_image, save_uploaded_image

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
    upscale: bool = Form(False),
    upscale_factor: int = Form(2),
    image_strength: float = Form(0.8),
    image_guidance_scale: float = Form(1.5),
    uploaded_image_path: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None)
):
    """Generate image using uploaded image (file or previously uploaded path) and prompt with optional LoRA support"""
    try:
        from utils.image_utils import (
            validate_uploaded_image,
            save_uploaded_image,
            load_and_preprocess_image,
            cleanup_uploaded_image,
        )

        temp_file_to_cleanup = None

        # Resolve input image source: either a freshly uploaded file or a server-side uploaded path
        if uploaded_image_path:
            # Use existing uploaded image on server (no validation possible here beyond path safety)
            input_image_path = uploaded_image_path
        else:
            if file is None:
                raise HTTPException(status_code=400, detail="Either file or uploaded_image_path must be provided")
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
                        if not lora_config.get("name") or not lora_config["name"].strip():
                            raise HTTPException(status_code=400, detail="LoRA name cannot be empty")
                        if lora_config.get("weight", 1.0) < 0 or lora_config.get("weight", 1.0) > 2.0:
                            raise HTTPException(status_code=400, detail="LoRA weight must be between 0 and 2.0")
                        loras_to_apply.append({
                            "name": lora_config["name"].strip(),
                            "weight": lora_config["weight"]
                        })
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid LoRA configuration format")
            elif lora_name:
                if not lora_name.strip():
                    raise HTTPException(status_code=400, detail="LoRA name cannot be empty if provided")
                if lora_weight is None or lora_weight < 0 or lora_weight > 2.0:
                    raise HTTPException(status_code=400, detail="LoRA weight must be between 0 and 2.0")
                loras_to_apply.append({
                    "name": lora_name.strip(),
                    "weight": lora_weight
                })

            # Apply default LoRA if none specified
            if not loras_to_apply and not remove_all_loras:
                loras_to_apply = DEFAULT_LORA_LIST

            # Validate dimensions
            if width < 256 or width > 1024 or height < 256 or height > 1024:
                raise HTTPException(status_code=400, detail="Width and height must be between 256 and 1024")

            # Validate upscale factor
            if upscale and upscale_factor not in [2, 4]:
                raise HTTPException(status_code=400, detail="Upscale factor must be 2 or 4")

            # Validate image strength and guidance
            if image_strength < 0.0 or image_strength > 1.0:
                raise HTTPException(status_code=400, detail="Image strength must be between 0.0 and 1.0")
            if image_guidance_scale < 1.0 or image_guidance_scale > 20.0:
                raise HTTPException(status_code=400, detail="Image guidance scale must be between 1.0 and 20.0")

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
                    logger.info("Removing all LoRAs as requested by client (empty list)")
                    model_manager.remove_lora()

            # Generate image using the model (placeholder: regular generate with prompt)
            logger.info(f"Starting image generation with uploaded image and prompt: {prompt}")

            pipeline = model_manager.get_pipeline()
            if not pipeline:
                raise HTTPException(status_code=500, detail="Model pipeline not available")

            if seed is not None:
                import torch
                torch.manual_seed(seed)

            generation_start_time = time.time()

            # For now use standard generate; future: img2img
            result = model_manager.generate_image(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=image_guidance_scale,
                width=width,
                height=height,
                seed=seed,
                negative_prompt=negative_prompt,
            )

            generation_time = time.time() - generation_start_time

            generated_image = extract_image_from_result(result)

            image_filename = save_image_with_unique_name(generated_image, save_as_jpeg=SAVE_AS_JPEG, jpeg_quality=JPEG_QUALITY)

            if upscale:
                try:
                    from models.upscaler import apply_upscaling
                    image_filename, upscaled_image_path, final_width, final_height = apply_upscaling(
                        generated_image, upscale, upscale_factor, 
                        lambda img: save_image_with_unique_name(img, save_as_jpeg=SAVE_AS_JPEG, jpeg_quality=JPEG_QUALITY),
                        save_as_jpeg=SAVE_AS_JPEG, jpeg_quality=JPEG_QUALITY
                    )
                    logger.info(f"Image upscaled by {upscale_factor}x: {image_filename}")
                except Exception as upscale_error:
                    logger.warning(f"Upscaling failed: {upscale_error}")
                    image_filename = save_image_with_unique_name(generated_image, save_as_jpeg=SAVE_AS_JPEG, jpeg_quality=JPEG_QUALITY)

            # Create download URL
            import os
            filename = os.path.basename(image_filename)
            download_url = f"/download/{filename}"

            return {
                "message": f"Generated image from uploaded image and prompt: {prompt}",
                "image_url": image_filename,
                "download_url": download_url,
                "filename": filename,
                "generation_time": f"{generation_time:.2f}s",
                "width": width,
                "height": height,
                "seed": seed,
                "image_strength": image_strength,
                "image_guidance_scale": image_guidance_scale,
                "upscaled": upscale,
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
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
