"""
API routes for the Diffusion API
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from loguru import logger

from api.models import GenerateRequest, SwitchModelRequest, SwitchModelResponse
from config.settings import DEFAULT_GUIDANCE_SCALE, DEFAULT_INFERENCE_STEPS
from models.models import DiffusionModelManager
from utils.image_utils import (extract_image_from_result,
                               save_image_with_unique_name)
from utils.queue_manager import QueueManager
from utils.system_utils import get_system_memory

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
                _model_manager_instance = DiffusionModelManager()
    return _model_manager_instance


model_manager = get_model_manager()

# Global queue manager instance
queue_manager = QueueManager(max_concurrent=2, max_queue_size=100)


async def update_lora_index(
    stored_name: str, original_name: str, timestamp: int, size: int
):
    """Update the LoRA index.json file with a new entry"""
    try:
        index_file = Path("uploads/lora_files/index.json")

        # Load existing index or create new one
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    index_data = json.loads(f.read())
            except (json.JSONDecodeError, FileNotFoundError):
                index_data = {"entries": []}
        else:
            index_data = {"entries": []}

        # Add new entry
        new_entry = {
            "stored_name": stored_name,
            "original_name": original_name,
            "timestamp": timestamp,
            "size": size,
        }

        # Check if entry already exists (avoid duplicates)
        existing_entry = next(
            (
                entry
                for entry in index_data["entries"]
                if entry["stored_name"] == stored_name
            ),
            None,
        )

        if not existing_entry:
            index_data["entries"].append(new_entry)

            # Save updated index
            with open(index_file, "w") as f:
                json.dump(index_data, f, indent=2)

            logger.info(f"Updated index.json with new LoRA: {stored_name}")
        else:
            logger.info(f"LoRA entry already exists in index: {stored_name}")

    except Exception as e:
        logger.error(f"Failed to update LoRA index: {e}")
        # Don't fail the upload if index update fails


async def remove_lora_from_index(stored_name: str):
    """Remove a LoRA entry from the index.json file"""
    try:
        index_file = Path("uploads/lora_files/index.json")

        if not index_file.exists():
            return

        with open(index_file, "r") as f:
            index_data = json.loads(f.read())

        # Remove the entry
        index_data["entries"] = [
            entry
            for entry in index_data["entries"]
            if entry["stored_name"] != stored_name
        ]

        # Save updated index
        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Removed LoRA entry from index: {stored_name}")

    except Exception as e:
        logger.error(f"Failed to remove LoRA from index: {e}")


@router.get("/")
def read_root():
    """Serve the frontend HTML"""
    html_path = "frontend/templates/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return {
            "message": "Diffusion API is running!",
            "endpoints": [
                "/generate",
                "/loras",
                "/apply-lora",
                "/remove-lora",
                "/lora-status",
                "/apply-lora-permanent",
                "/fused-lora-status",
                "/unfuse-loras",
            ],
            "model_loaded": model_manager.is_loaded(),
        }


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
    """Generate image using Diffusion model

    Note: LoRAs must be applied separately via /apply-lora endpoint or configured at startup via fusion mode.
    This endpoint uses whatever LoRA is currently applied to the model.
    """
    try:
        # Validate request parameters
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Clean up input
        prompt = request.prompt.strip()

        # First, ensure the model is loaded with thread safety
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
                            status_code=500, detail="Failed to load Diffusion model"
                        )
                    logger.info("Model loaded successfully")
                else:
                    logger.info("Model was loaded by another thread while waiting")

        # Get current LoRA info (if any was applied via /apply-lora or fusion mode)
        current_lora = model_manager.get_lora_info()
        lora_applied = current_lora.get("name") if current_lora else None
        lora_weight_applied = current_lora.get("weight") if current_lora else None

        if lora_applied:
            logger.info(
                f"Using currently applied LoRA: {lora_applied} (weight: {lora_weight_applied})"
            )
        else:
            logger.info("No LoRA currently applied, generating without LoRA")

        result = generate_image_internal(
            prompt,
            lora_applied,
            lora_weight_applied,
            request.width or 512,
            request.height or 512,
            request.seed,
            request.upscale or False,
            request.upscale_factor or 2,
            request.guidance_scale or DEFAULT_GUIDANCE_SCALE,
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


@router.post("/generate-and-return-image")
async def generate_and_return_image(request: GenerateRequest):
    """Generate image and return it directly as binary data

    Note: LoRAs must be applied separately via /apply-lora endpoint or configured at startup via fusion mode.
    This endpoint uses whatever LoRA is currently applied to the model.
    Returns the image bytes directly instead of saving to disk.
    """
    try:
        # Validate request parameters
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Clean up input
        prompt = request.prompt.strip()

        # First, ensure the model is loaded with thread safety
        with _model_manager_lock:
            model_actually_loaded = (
                model_manager.is_loaded()
                and model_manager.get_pipeline() is not None
                and hasattr(model_manager.get_pipeline(), "transformer")
            )

            if not model_actually_loaded:
                logger.info(
                    "Model not properly loaded or pipeline unavailable, loading it first..."
                )
                if not (
                    model_manager.is_loaded()
                    and model_manager.get_pipeline() is not None
                ):
                    if not model_manager.load_model():
                        raise HTTPException(
                            status_code=500, detail="Failed to load Diffusion model"
                        )
                    logger.info("Model loaded successfully")
                else:
                    logger.info("Model was loaded by another thread while waiting")

        # Get current LoRA info (if any was applied via /apply-lora or fusion mode)
        current_lora = model_manager.get_lora_info()
        lora_applied = current_lora.get("name") if current_lora else None
        lora_weight_applied = current_lora.get("weight") if current_lora else None

        if lora_applied:
            logger.info(
                f"Using currently applied LoRA: {lora_applied} (weight: {lora_weight_applied})"
            )
        else:
            logger.info("No LoRA currently applied, generating without LoRA")

        # Generate image and get result with file path
        result = generate_image_internal(
            prompt,
            lora_applied,
            lora_weight_applied,
            request.width or 512,
            request.height or 512,
            request.seed,
            request.upscale or False,
            request.upscale_factor or 2,
            request.guidance_scale or DEFAULT_GUIDANCE_SCALE,
        )

        # Extract the download URL and read the file
        download_url = result.get("download_url")
        if not download_url:
            raise HTTPException(
                status_code=500,
                detail="No download URL received from generate endpoint",
            )

        # Extract filename from download_url (format: /download/{filename})
        filename = download_url.split("/")[-1]
        file_path = Path("generated_images") / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Generated image not found")

        # Read the image file into memory
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        # Delete the temporary file
        try:
            file_path.unlink()
            logger.info(f"Deleted temporary image file: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary image file: {cleanup_error}")

        # Return the image bytes directly
        return Response(content=image_bytes, media_type="image/png")

    except Exception as e:
        logger.error(f"Request processing failed: {e} (Type: {type(e).__name__})")
        raise HTTPException(
            status_code=500, detail=f"Request processing failed: {str(e)}"
        )


@router.post("/load-model")
def load_model():
    """Load the Diffusion model"""
    try:
        with _model_manager_lock:
            # Enhanced check for proper model loading
            model_actually_loaded = (
                model_manager.is_loaded()
                and model_manager.get_pipeline() is not None
                and hasattr(model_manager.get_pipeline(), "transformer")
            )

            if model_actually_loaded:
                logger.info("Diffusion model already properly loaded")
                return {"message": "Diffusion model already loaded"}

            logger.info("Loading Diffusion model...")
            if model_manager.load_model():
                logger.info("Diffusion model loaded successfully")
                return {"message": "Diffusion model loaded successfully"}
            else:
                logger.error(
                    f"Failed to load Diffusion model - Model: {model_manager.is_loaded()}, Pipeline: {model_manager.get_pipeline() is not None}"
                )
                raise HTTPException(
                    status_code=500, detail="Failed to load Diffusion model"
                )
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
            "fusion_mode": model_manager.is_fusion_mode_enabled(),
        }
    )

    return status


@router.get("/gpu-info")
def get_gpu_info():
    """Get detailed GPU information"""
    return model_manager.gpu_manager.get_device_info()


@router.get("/loras")
async def get_available_loras():
    """Get list of available LoRA files (uploaded and default)"""
    try:
        # Get uploaded LoRAs from index.json
        uploaded_loras = []
        index_file = Path("uploads/lora_files/index.json")

        # Ensure directory exists
        index_file.parent.mkdir(parents=True, exist_ok=True)

        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    index_data = json.loads(f.read())
                    if "entries" in index_data and isinstance(
                        index_data["entries"], list
                    ):
                        uploaded_loras = index_data["entries"]
                        logger.info(
                            f"Loaded {len(uploaded_loras)} LoRAs from index.json"
                        )
                    else:
                        logger.warning("Invalid index.json format")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse index.json: {e}")
                uploaded_loras = []
        else:
            # Initialize empty index file
            try:
                with open(index_file, "w") as f:
                    json.dump({"entries": []}, f, indent=2)
            except Exception as init_err:
                logger.warning(f"Failed to initialize LoRA index.json: {init_err}")

        # Add default LoRA
        default_loras = [
            {
                "name": "/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors",
                "display_name": "/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors (Default)",
                "type": "default",
                "weight": 1.0,
            }
        ]

        return {
            "uploaded": uploaded_loras,
            "default": default_loras,
            "total_count": len(uploaded_loras) + len(default_loras),
        }

    except Exception as e:
        logger.error(f"Error getting available LoRAs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get available LoRAs: {str(e)}"
        )


@router.post("/upload-lora")
async def upload_lora_file(file: UploadFile = File(...)):
    """Upload a LoRA file to the server"""

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

        # Prevent duplicate upload by original filename (case-insensitive)
        index_file = uploads_dir / "index.json"
        current_entries = {"entries": []}
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    current_entries = json.loads(f.read()) or {"entries": []}
            except Exception:
                current_entries = {"entries": []}

        original_name_lower = file.filename.lower()
        if any(
            (e.get("original_name", "").lower() == original_name_lower)
            for e in current_entries.get("entries", [])
        ):
            raise HTTPException(
                status_code=409, detail=f"LoRA '{file.filename}' already uploaded"
            )

        # Generate unique filename
        timestamp = int(time.time())
        safe_filename = f"uploaded_lora_{timestamp}{file_extension}"
        file_path = uploads_dir / safe_filename

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"LoRA file uploaded successfully: {file_path}")

        # Update the index.json file
        try:
            await update_lora_index(safe_filename, file.filename, timestamp, file.size)
        except Exception as index_error:
            logger.warning(f"Failed to update LoRA index after upload: {index_error}")

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


@router.post("/apply-lora")
async def apply_lora(lora_name: str, weight: float = 1.0):
    """Apply a LoRA to the current model - lora_name should be a Hugging Face repo ID (e.g., aleksa-codes/flux-ghibsky-illustration)"""
    # Check if fusion mode is enabled
    if model_manager.is_fusion_mode_enabled():
        raise HTTPException(
            status_code=403,
            detail="Runtime LoRA changes are disabled in fusion mode. LoRAs were configured at startup and cannot be modified.",
        )

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


@router.delete("/remove-lora/{filename}")
async def remove_lora_file(filename: str):
    """Remove a LoRA file and its entry from the index"""
    try:
        uploads_dir = Path("uploads/lora_files")
        file_path = uploads_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="LoRA file not found")

        # Remove the file
        file_path.unlink()

        # Remove entry from index.json
        await remove_lora_from_index(filename)

        logger.info(f"LoRA file removed: {filename}")
        return {"message": f"LoRA file {filename} removed successfully"}

    except Exception as e:
        logger.error(f"Error removing LoRA file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to remove LoRA file: {str(e)}"
        )


@router.post("/remove-lora")
async def remove_lora():
    """Remove the currently applied LoRA from the model"""
    # Check if fusion mode is enabled
    if model_manager.is_fusion_mode_enabled():
        raise HTTPException(
            status_code=403,
            detail="Runtime LoRA changes are disabled in fusion mode. LoRAs were configured at startup and cannot be modified.",
        )

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
    """Submit a generation request to the queue

    Note: Uses currently applied LoRA (via /apply-lora or fusion mode).
    LoRA parameters cannot be specified per-request.
    """
    try:
        # Validate request
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Get current LoRA info
        current_lora = model_manager.get_lora_info()
        lora_name = current_lora.get("name") if current_lora else None
        lora_weight = current_lora.get("weight") if current_lora else None

        # Submit to queue
        request_id = await queue_manager.submit_request(
            prompt=request.prompt.strip(),
            lora_name=lora_name,
            lora_weight=lora_weight,
            loras=None,  # Not supported in queue mode
            num_inference_steps=request.num_inference_steps or DEFAULT_INFERENCE_STEPS,
            guidance_scale=request.guidance_scale or DEFAULT_GUIDANCE_SCALE,
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
            "guidance_scale": request.guidance_scale or DEFAULT_GUIDANCE_SCALE,
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
    lora_applied: Optional[str] = None,
    lora_weight: Optional[float] = None,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    upscale: bool = False,
    upscale_factor: int = 2,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
):
    """Internal function to generate images - used by both endpoints"""
    logger.info(f"Starting image generation for prompt: {prompt}")

    # Model should already be loaded at this point
    if not model_manager.is_loaded():
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Log which model type is being used for generation
    current_model_type = getattr(model_manager, "current_model_type", "unknown")
    logger.info(f"Using {current_model_type} model for image generation")

    # Apply LoRA if specified and different from the currently applied one
    if lora_applied and not model_manager.is_fused():
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
    elif model_manager.is_fused():
        logger.info("Using fused LoRAs for generation")

    try:
        logger.info(
            f"Generating image for prompt: {prompt} with {current_model_type} model"
        )

        # Start timing
        start_time = time.time()

        # Generate image with Diffusion
        if model_manager.get_pipeline() is None:
            raise HTTPException(
                status_code=500,
            )

        # Generate the image
        result = model_manager.generate_image(
            prompt,
            DEFAULT_INFERENCE_STEPS,  # Fixed num_inference_steps
            guidance_scale,  # Use parameter value
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

        # Get the actual LoRA status from the model manager
        actual_lora_info = model_manager.get_lora_info()

        filename = os.path.basename(image_filename)
        download_url = f"/download/{filename}"

        logger.info(
            f"Image generation completed successfully with {current_model_type} model"
        )

        # Get the current model type for the response
        current_model_type = getattr(model_manager, "current_model_type", "unknown")

        return {
            "message": f"Generated image for prompt: {prompt}",
            "image_url": image_filename,
            "download_url": download_url,
            "filename": filename,
            "generation_time": f"{generation_time:.2f}s",
            "model_type": current_model_type,
            "lora_applied": actual_lora_info.get("name") if actual_lora_info else None,
            "lora_weight": actual_lora_info.get("weight") if actual_lora_info else None,
            "width": final_width,
            "height": final_height,
            "seed": seed,
        }

    except Exception as e:
        logger.error(
            f"Error generating image with {current_model_type} model: {e} (Type: {type(e).__name__})"
        )
        raise HTTPException(
            status_code=500,
        )


@router.post("/switch-model")
async def switch_model(request: SwitchModelRequest):
    """Switch between different model types (flux/qwen)"""
    try:
        model_type = request.model_type

        with _model_manager_lock:
            # Get current model type before switching
            current_status = model_manager.get_model_status()
            previous_model_type = current_status.get("model_type", "unknown")

            # Switch the model
            if model_manager.switch_model(model_type):
                logger.info(
                    f"Successfully switched from {previous_model_type} to {model_type}"
                )
                return SwitchModelResponse(
                    message=f"Model switched to {model_type} successfully",
                    model_type=model_type,
                    previous_model_type=previous_model_type,
                )
            else:
                raise HTTPException(
                    status_code=500, detail=f"Failed to switch to {model_type} model"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/apply-lora-permanent")
async def apply_lora_permanent(request: dict):
    """Apply LoRAs permanently (fusion)"""
    # Check if fusion mode is enabled
    if model_manager.is_fusion_mode_enabled():
        raise HTTPException(
            status_code=403,
            detail="Runtime LoRA changes are disabled in fusion mode. LoRAs were configured at startup and cannot be modified.",
        )

    try:
        if "loras" not in request:
            raise HTTPException(
                status_code=400, detail="Missing 'loras' field in request"
            )

        lora_configs = request["loras"]
        if not isinstance(lora_configs, list):
            raise HTTPException(status_code=400, detail="'loras' must be a list")

        if len(lora_configs) == 0:
            raise HTTPException(status_code=400, detail="No LoRAs provided for fusion")

        # Validate LoRA configurations
        for config in lora_configs:
            if not isinstance(config, dict):
                raise HTTPException(
                    status_code=400, detail="Each LoRA config must be a dictionary"
                )
            if "name" not in config or "weight" not in config:
                raise HTTPException(
                    status_code=400,
                    detail="Each LoRA config must have 'name' and 'weight' fields",
                )
            if not config["name"] or not config["name"].strip():
                raise HTTPException(status_code=400, detail="LoRA name cannot be empty")
            if (
                not isinstance(config["weight"], (int, float))
                or config["weight"] < 0
                or config["weight"] > 2.0
            ):
                raise HTTPException(
                    status_code=400,
                    detail="LoRA weight must be a number between 0 and 2.0",
                )

        # Apply fusion
        if model_manager.fuse_loras(lora_configs):
            fused_info = model_manager.get_fused_lora_info()
            return {
                "message": f"Successfully fused {len(lora_configs)} LoRAs",
                "fused_info": fused_info,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to fuse LoRAs")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying LoRA fusion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/fused-lora-status")
async def get_fused_lora_status():
    """Get current fusion status"""
    try:
        is_fused = model_manager.is_fused()
        fused_info = model_manager.get_fused_lora_info()

        return {"is_fused": is_fused, "fused_info": fused_info}

    except Exception as e:
        logger.error(f"Error getting fusion status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/fusion-mode-status")
async def get_fusion_mode_status():
    """Get fusion mode status (startup fusion lock)"""
    try:
        fusion_mode = model_manager.is_fusion_mode_enabled()
        lora_info = model_manager.get_lora_info() if fusion_mode else None

        return {
            "fusion_mode": fusion_mode,
            "description": (
                "Fusion mode prevents runtime LoRA changes. LoRAs were configured at startup."
                if fusion_mode
                else "Fusion mode is disabled. Runtime LoRA changes are allowed."
            ),
            "lora_info": lora_info,
        }

    except Exception as e:
        logger.error(f"Error getting fusion mode status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/unfuse-loras")
async def unfuse_loras():
    """Unfuse LoRAs"""
    # Check if fusion mode is enabled
    if model_manager.is_fusion_mode_enabled():
        raise HTTPException(
            status_code=403,
            detail="Runtime LoRA changes are disabled in fusion mode. LoRAs were configured at startup and cannot be modified.",
        )

    try:
        if model_manager.unfuse_loras():
            return {"message": "LoRAs unfused successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to unfuse LoRAs")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unfusing LoRAs: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
