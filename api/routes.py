"""
API routes for the FLUX API
"""

import logging
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from models.flux_model import FluxModelManager
from utils.image_utils import extract_image_from_result, save_image_with_unique_name
from utils.system_utils import get_system_memory
from utils.queue_manager import QueueManager
from config.settings import STATIC_IMAGES_DIR
from api.models import GenerateRequest, GenerateResponse, ModelStatusResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global model manager instance - singleton pattern
_model_manager_instance = None


def get_model_manager():
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = FluxModelManager()
    return _model_manager_instance


model_manager = get_model_manager()

# Global queue manager instance
queue_manager = QueueManager(max_concurrent=2, max_queue_size=100)


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


@router.post("/generate")
async def generate_image(request: GenerateRequest):
    """Generate image using FLUX model with optional LoRA support - lora_name should be a Hugging Face repo ID"""
    try:
        # Validate request parameters
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
            
        if request.lora_name and not request.lora_name.strip():
            raise HTTPException(status_code=400, detail="LoRA name cannot be empty if provided")
            
        if request.lora_weight < 0 or request.lora_weight > 2.0:
            raise HTTPException(status_code=400, detail="LoRA weight must be between 0 and 2.0")
            
        # Clean up input
        prompt = request.prompt.strip()
        lora_name = request.lora_name.strip() if request.lora_name else None
        lora_weight = request.lora_weight
        
        # First, ensure the model is loaded
        if not model_manager.is_loaded():
            logger.info("Model not loaded, loading it first...")
            if not model_manager.load_model():
                raise HTTPException(status_code=500, detail="Failed to load FLUX model")
            logger.info("Model loaded successfully")

        # Now check if LoRA is already applied
        current_lora = model_manager.get_lora_info()
        lora_applied = None
        lora_weight_applied = None

        if lora_name:
            # Validate LoRA name format (should be a valid Hugging Face repo ID)
            if not lora_name or "/" not in lora_name:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid LoRA name format. Must be a Hugging Face repository ID (e.g., 'username/model-name')"
                )
                
            # Only apply LoRA if it's different from the current one
            if (
                not current_lora
                or current_lora.get("name") != lora_name
                or current_lora.get("weight") != lora_weight
            ):
                logger.info(
                    f"Applying new LoRA: {lora_name} with weight {lora_weight}"
                )
                try:
                    if model_manager.apply_lora(lora_name, lora_weight):
                        lora_applied = lora_name
                        lora_weight_applied = lora_weight
                        logger.info(
                            f"LoRA {lora_name} applied successfully with weight {lora_weight}"
                        )
                    else:
                        logger.error(
                            f"LoRA application failed: {lora_name} - Model: {model_manager.is_loaded()}, Pipeline: {model_manager.get_pipeline() is not None}"
                        )
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Failed to apply LoRA {lora_name}. Please check if the LoRA exists and is compatible."
                        )
                except Exception as lora_error:
                    logger.error(
                        f"Exception during LoRA application: {lora_error} (Type: {type(lora_error).__name__})"
                    )
                    if "not found" in str(lora_error).lower() or "404" in str(lora_error):
                        raise HTTPException(
                            status_code=400, 
                            detail=f"LoRA {lora_name} not found. Please check the repository ID."
                        )
                    else:
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Failed to apply LoRA {lora_name}: {str(lora_error)}"
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
            prompt, "FLUX", lora_applied, lora_weight_applied,
            request.num_inference_steps, request.guidance_scale,
            request.width, request.height, request.seed, request.negative_prompt
        )
        return result
    except Exception as e:
        logger.error(f"Request processing failed: {e} (Type: {type(e).__name__})")
        raise HTTPException(
            status_code=500, detail=f"Request processing failed: {str(e)}"
        )


@router.post("/load-model")
def load_model():
    """Load the FLUX model"""
    try:
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

    status.update(
        {
            "system_memory_used_gb": f"{system_memory_used:.2f}GB",
            "system_memory_total_gb": f"{system_memory_total:.2f}GB",
            "lora_loaded": (model_manager.get_lora_info() or {}).get("name"),
            "lora_weight": (model_manager.get_lora_info() or {}).get("weight"),
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
            raise HTTPException(status_code=400, detail="LoRA name cannot be empty if provided")
            
        if request.lora_weight < 0 or request.lora_weight > 2.0:
            raise HTTPException(status_code=400, detail="LoRA weight must be between 0 and 2.0")
            
        # Submit to queue
        request_id = await queue_manager.submit_request(
            prompt=request.prompt.strip(),
            lora_name=request.lora_name.strip() if request.lora_name else None,
            lora_weight=request.lora_weight,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
            negative_prompt=request.negative_prompt
        )
        
        return {
            "message": "Request submitted successfully",
            "request_id": request_id,
            "status": "queued"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit request: {str(e)}")


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
            "negative_prompt": request.negative_prompt
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get request status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get request status: {str(e)}")


@router.delete("/cancel-request/{request_id}")
async def cancel_request(request_id: str):
    """Cancel a pending request"""
    try:
        success = await queue_manager.cancel_request(request_id)
        if not success:
            raise HTTPException(status_code=404, detail="Request not found or already processing")
            
        return {"message": "Request cancelled successfully", "request_id": request_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel request: {str(e)}")


@router.get("/queue-stats")
async def get_queue_stats():
    """Get current queue statistics"""
    try:
        return queue_manager.get_queue_stats()
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")


def generate_image_internal(
    prompt: str,
    model_type_name: str = "FLUX",
    lora_applied: Optional[str] = None,
    lora_weight: Optional[float] = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
):
    """Internal function to generate images - used by both endpoints"""
    logger.info(f"Starting image generation for prompt: {prompt}")

    # Model should already be loaded at this point
    if not model_manager.is_loaded():
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Apply LoRA if specified and not already applied
    if lora_applied and not model_manager.get_lora_info():
        logger.info(f"Applying LoRA {lora_applied} to loaded model")
        try:
            if not model_manager.apply_lora(lora_applied, lora_weight or 1.0):
                logger.error(f"Failed to apply LoRA {lora_applied} to loaded model")
            else:
                logger.info(f"LoRA {lora_applied} applied successfully to loaded model")
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
            prompt, 
            num_inference_steps, 
            guidance_scale, 
            width, 
            height, 
            seed, 
            negative_prompt
        )

        image = extract_image_from_result(result)

        # End timing
        end_time = time.time()
        generation_time = end_time - start_time

        # Save image with unique name
        image_filename = save_image_with_unique_name(image)

        # Get system information
        vram_usage = model_manager.gpu_manager.get_vram_usage()
        system_memory_used, system_memory_total = get_system_memory()

        # Get the actual LoRA status from the model manager
        actual_lora_info = model_manager.get_lora_info()

        return {
            "message": f"Generated {model_type_name} image for prompt: {prompt}",
            "image_url": image_filename,
            "generation_time": f"{generation_time:.2f}s",
            "vram_usage_gb": f"{vram_usage:.2f}GB",
            "system_memory_used_gb": f"{system_memory_used:.2f}GB",
            "system_memory_total_gb": f"{system_memory_total:.2f}GB",
            "model_type": model_manager.model_type,
            "lora_applied": actual_lora_info.get("name") if actual_lora_info else None,
            "lora_weight": actual_lora_info.get("weight") if actual_lora_info else None,
            # Generation parameters used
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
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
