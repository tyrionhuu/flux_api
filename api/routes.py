"""
API routes for the FLUX API
"""

import io
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import loguru
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from api.models import ApplyLoRARequest, GenerateRequest
from config.settings import (DEFAULT_GUIDANCE_SCALE, DEFAULT_LORA_NAME,
                             DEFAULT_LORA_WEIGHT, INFERENCE_STEPS)
from models.flux_model import FluxModelManager
from models.upscaler import apply_upscaling
from utils.birefnet_remover import remove_background_birefnet
from utils.cleanup_service import (cleanup_after_generation,
                                   cleanup_after_upload)
from utils.image_utils import (extract_image_from_result,
                               save_image_with_unique_name,
                               save_uploaded_image, validate_uploaded_image)
from utils.infer_utils import kontext_preprocess
from utils.queue_manager import QueueManager
from utils.system_utils import get_system_memory

logger = loguru.logger.bind(name="api.routes")


def handle_api_error(
    operation: str, error: Exception, status_code: int = 500
) -> HTTPException:
    """Standardized error handling for API endpoints"""
    logger.error(f"Error in {operation}: {error}")
    return HTTPException(
        status_code=status_code, detail=f"{operation} failed: {str(error)}"
    )


router = APIRouter()


def _get_bg_removal_strength(strength: Optional[float]) -> Optional[float]:
    """
    Get background removal strength parameter
    BiRefNet current version doesn't support strength adjustment, but keeps API compatibility

    Args:
        strength: Background removal strength (0.0-1.0)

    Returns:
        Processed strength value, currently returns original value
    """
    if strength is None:
        return None
    try:
        # Ensure strength value is within valid range
        return max(0.0, min(1.0, float(strength)))
    except (ValueError, TypeError):
        return None


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
def read_root(request: Request):
    """Root endpoint for testing"""
    return {
        "message": "FLUX API is running!",
        "endpoints": [
            "/static-image",
            "/generate",
            "/generate-and-return-image",
            "/generate-with-image",
            "/generate-with-image-and-return",
            "/loras",
            "/apply-lora",
            "/remove-lora",
            "/lora-status",
            "/apply-lora-permanent",
            "/fused-lora-status",
            "/unfuse-loras",
        ],
        "model_loaded": model_manager.is_loaded(),
        "model_type": model_manager.model_type,
        "server_port": getattr(request.app.state, "port", 9200),
        "timestamp": time.time(),
    }


@router.get("/download/{filename:path}")
def download_image(filename: str):
    """Download a generated image file"""

    # Security: only allow files from generated_images directory
    safe_filename = os.path.basename(filename)  # Remove any path traversal

    # Use absolute path to avoid working directory issues
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = Path(base_dir) / "generated_images" / safe_filename

    logger.info(f"Download request for filename: {filename}")
    logger.debug(f"Safe filename: {safe_filename}")
    logger.debug(f"Base directory: {base_dir}")
    logger.debug(f"Full file path: {file_path}")
    logger.info(f"File exists: {file_path.exists()}")

    if not file_path.exists():
        # Log the directory contents for debugging
        generated_images_dir = Path(base_dir) / "generated_images"
        if generated_images_dir.exists():
            files = list(generated_images_dir.glob("*.png"))
            logger.error(
                f"File not found. Available files in {generated_images_dir}: {[f.name for f in files]}"
            )
        else:
            logger.error(
                f"Generated images directory does not exist: {generated_images_dir}"
            )
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


def _ensure_model_loaded():
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
                model_manager.is_loaded() and model_manager.get_pipeline() is not None
            ):
                if not model_manager.load_model():
                    raise HTTPException(
                        status_code=500, detail="Failed to load FLUX model"
                    )
                logger.info("Model loaded successfully")
            else:
                logger.info("Model was loaded by another thread while waiting")


def _extract_loras_from_request(request: GenerateRequest):
    loras_to_apply = []
    remove_all_loras = False

    if request.loras is not None:
        if len(request.loras) == 0:
            remove_all_loras = True
        else:
            for lora_config in request.loras:
                if not lora_config.name or not lora_config.name.strip():
                    raise HTTPException(
                        status_code=400, detail="LoRA name cannot be empty"
                    )
                if (
                    not hasattr(lora_config, "weight")
                    or lora_config.weight < 0
                    or lora_config.weight > 2.0
                ):
                    raise HTTPException(
                        status_code=400, detail="LoRA weight must be between 0 and 2.0"
                    )
                loras_to_apply.append(
                    {
                        "name": lora_config.name.strip(),
                        "weight": float(lora_config.weight),
                    }
                )
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
            {"name": request.lora_name.strip(), "weight": float(request.lora_weight)}
        )

    # Only apply default LoRA if explicitly requested or if no LoRAs specified
    if (
        not loras_to_apply
        and not remove_all_loras
        and request.loras is None
        and not request.lora_name
        and hasattr(request, "use_default_lora")
        and request.use_default_lora
        and DEFAULT_LORA_NAME is not None
    ):
        loras_to_apply = [{"name": DEFAULT_LORA_NAME, "weight": DEFAULT_LORA_WEIGHT}]

    return loras_to_apply, remove_all_loras


def _apply_loras(loras_to_apply, remove_all_loras):
    current_lora = model_manager.get_lora_info()
    lora_applied = None
    lora_weight_applied = None

    logger.info(
        f"LoRA application request - loras_to_apply: {loras_to_apply}, remove_all_loras: {remove_all_loras}"
    )
    logger.info(f"Current LoRA state: {current_lora}")

    if loras_to_apply:
        logger.info(f"Applying {len(loras_to_apply)} LoRAs to loaded model")
        for lora_config in loras_to_apply:
            logger.info(f"Processing LoRA: {lora_config}")
            if not lora_config["name"]:
                raise HTTPException(status_code=400, detail="LoRA name cannot be empty")

            # Handle uploaded LoRAs
            if lora_config["name"].startswith("uploaded_lora_"):
                upload_path = f"uploads/lora_files/{lora_config['name']}"
                if not os.path.exists(upload_path):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded LoRA file not found: {lora_config['name']} at {upload_path}",
                    )
                logger.info(f"Found uploaded LoRA at: {upload_path}")
            # Handle Hugging Face LoRAs
            elif "/" in lora_config["name"]:
                logger.info(f"Processing Hugging Face LoRA: {lora_config['name']}")
            # Handle local path LoRAs
            elif os.path.exists(lora_config["name"]):
                logger.info(f"Processing local LoRA: {lora_config['name']}")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid LoRA name format for '{lora_config['name']}'. Must be a Hugging Face repository ID (e.g., 'username/model-name'), local path, or uploaded file path"
                    ),
                )

        # Apply the LoRAs
        if not model_manager.apply_multiple_loras(loras_to_apply):
            logger.error(
                f"Multiple LoRA application failed - Model: {model_manager.is_loaded()}, Pipeline: {model_manager.get_pipeline() is None}"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to apply LoRAs. Please check if the LoRAs exist and are compatible.",
            )
        else:
            # Log successful LoRA application
            applied_loras = model_manager.get_lora_info()
            if applied_loras:
                logger.info(f"Successfully applied LoRAs: {applied_loras}")
            else:
                logger.info(
                    "LoRAs applied successfully (no current LoRA info available)"
                )

        current_lora = model_manager.get_lora_info()
        if current_lora:
            lora_applied = current_lora.get("name")
            lora_weight_applied = current_lora.get("weight")
            logger.info(
                f"Multiple LoRAs applied successfully. Current LoRAs: {lora_applied} with total weight {lora_weight_applied}"
            )
        else:
            logger.warning("LoRAs were applied but get_lora_info() returned None")
    elif remove_all_loras:
        if model_manager.get_lora_info():
            logger.info("Removing all LoRAs as requested by client (empty list)")
            model_manager.remove_lora()
        else:
            logger.info("No LoRAs to remove")
    else:
        if current_lora:
            lora_applied = current_lora.get("name")
            lora_weight_applied = current_lora.get("weight")
            logger.info(
                f"Using existing LoRA: {lora_applied} with weight {lora_weight_applied}"
            )
        else:
            logger.info("No LoRAs specified and no existing LoRAs")

    return lora_applied, lora_weight_applied


async def _queue_txt2img_and_get_result(
    prompt: str,
    width: int,
    height: int,
    seed: Optional[int],
    upscale: bool,
    upscale_factor: int,
    lora_applied: Optional[str],
    lora_weight_applied: Optional[float],
    loras_to_apply: Optional[list],
    num_inference_steps: int,
    guidance_scale: float,
    downscale: Optional[bool] = True,
):
    def processor(_req, _ctx):
        return generate_image_internal(
            prompt,
            "FLUX",
            lora_applied,
            lora_weight_applied,
            width or 512,
            height or 512,
            seed,
            upscale or False,
            upscale_factor or 2,
            downscale,
            num_inference_steps,
            guidance_scale,
        )

    return await queue_manager.submit_and_wait(
        prompt=prompt,
        loras=loras_to_apply if loras_to_apply else None,
        lora_name=lora_applied,
        lora_weight=lora_weight_applied or 1.0,
        width=width or 512,
        height=height or 512,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        processor=processor,
        context={},
    )


def _read_and_cleanup_generated(download_url: str) -> bytes:
    filename = download_url.split("/")[-1]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = Path(base_dir) / "generated_images" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Generated image file not found")
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    try:
        os.remove(file_path)
    except Exception as cleanup_error:
        logger.warning(f"Failed to cleanup temporary image file: {cleanup_error}")
    return image_bytes


@router.post("/generate-and-return-image")
async def generate_and_return_image(request: GenerateRequest):
    """Generate image and return it directly as binary data"""
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

        # Handle LoRA support via helper
        loras_to_apply, remove_all_loras = _extract_loras_from_request(request)

        # Clean up input
        prompt = request.prompt.strip()

        _ensure_model_loaded()

        lora_applied, lora_weight_applied = _apply_loras(
            loras_to_apply, remove_all_loras
        )

        # Auto-size for txt2img: default to 1024x1024 and ignore client-provided size
        result = await _queue_txt2img_and_get_result(
            prompt,
            1024,
            1024,
            request.seed,
            request.upscale or False,
            request.upscale_factor or 2,
            lora_applied,
            lora_weight_applied,
            loras_to_apply,
            request.num_inference_steps or INFERENCE_STEPS,
            request.guidance_scale or DEFAULT_GUIDANCE_SCALE,
            request.downscale if hasattr(request, "downscale") else True,
        )

        # Extract the download URL from the result
        download_url = result.get("download_url")

        if not download_url:
            raise HTTPException(
                status_code=500,
                detail="No download URL received from generate endpoint",
            )

        # Optional background removal on the final image
        if getattr(request, "remove_background", False):
            download_url = _apply_background_removal_to_saved(
                download_url, getattr(request, "bg_strength", None)
            )

        return Response(
            content=_read_and_cleanup_generated(download_url), media_type="image/png"
        )

    except Exception as e:
        raise handle_api_error("generate_and_return_image", e)


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

        # Handle LoRA support via helper
        loras_to_apply, remove_all_loras = _extract_loras_from_request(request)

        # Clean up input
        prompt = request.prompt.strip()

        _ensure_model_loaded()

        lora_applied, lora_weight_applied = _apply_loras(
            loras_to_apply, remove_all_loras
        )

        def processor(_req, _ctx):
            return generate_image_internal(
                prompt,
                "FLUX",
                lora_applied,
                lora_weight_applied,
                1024,
                1024,
                request.seed,
                request.upscale or False,
                request.upscale_factor or 2,
                request.downscale if hasattr(request, "downscale") else True,
                _req.num_inference_steps,
                _req.guidance_scale,
            )

        result = await queue_manager.submit_and_wait(
            prompt=prompt,
            loras=loras_to_apply if loras_to_apply else None,
            lora_name=lora_applied,
            lora_weight=lora_weight_applied or 1.0,
            width=1024,
            height=1024,
            seed=request.seed,
            num_inference_steps=request.num_inference_steps or INFERENCE_STEPS,
            guidance_scale=request.guidance_scale or DEFAULT_GUIDANCE_SCALE,
            processor=processor,
            context={},
        )

        # Optionally apply background removal by mutating download_url
        try:
            if (
                getattr(request, "remove_background", False)
                and isinstance(result, dict)
                and result.get("download_url")
            ):
                new_download_url = _apply_background_removal_to_saved(
                    result["download_url"]
                )
                result["download_url"] = new_download_url
                result["image_url"] = new_download_url
        except Exception as e:
            logger.error(f"Background removal (txt2img) failed: {e}")

        # Trigger cleanup after successful image generation
        try:

            cleanup_after_generation()
        except Exception as cleanup_error:
            logger.warning(
                f"Failed to trigger cleanup after generation: {cleanup_error}"
            )

        return result
    except Exception as e:
        raise handle_api_error("request processing", e)


@router.post("/generate-with-image-and-return")
async def generate_with_image_and_return(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    num_inference_steps: Optional[int] = Form(INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(DEFAULT_GUIDANCE_SCALE),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(512),
    seed: Optional[int] = Form(None),
    remove_background: Optional[bool] = Form(False),
    bg_strength: Optional[float] = Form(None),
    lora_name: Optional[str] = Form(None),
    lora_weight: Optional[float] = Form(None),
    loras_json: Optional[str] = Form(None),
    use_default_lora: Optional[bool] = Form(False),
    upscale: Optional[bool] = Form(False),
    upscale_factor: Optional[int] = Form(2),
    downscale: Optional[bool] = Form(True),
):
    """Generate image from uploaded image and return it directly as binary data"""
    try:

        # Debug logging for form parameters
        logger.info(
            f"Received generate-with-image request - prompt: {prompt}, num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, width: {width}, height: {height}, seed: {seed}, downscale: {downscale}"
        )
        logger.info(
            f"Image file: {image.filename}, content_type: {image.content_type}, size: {image.size}"
        )

        if width and (width < 256 or width > 1024):
            raise HTTPException(
                status_code=400, detail="width must be between 256 and 1024"
            )

        if height and (height < 256 or height > 1024):
            raise HTTPException(
                status_code=400, detail="height must be between 256 and 1024"
            )

        # Load input image
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")

        # Validate image file type
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Check file size (max 10MB)
        if image.size and image.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="Image file too large (max 10MB)"
            )

        # Check file extension
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        file_extension = os.path.splitext(image.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format. Allowed: {', '.join(allowed_extensions)}",
            )

        try:
            raw = await image.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            pre_img, tgt_w, tgt_h = kontext_preprocess(img, downscale=downscale)
        except Exception as img_error:
            logger.error(f"Failed to load image: {img_error}")
            raise HTTPException(
                status_code=400, detail=f"Failed to load image: {str(img_error)}"
            )

        # Ensure model is loaded
        with _model_manager_lock:
            if not model_manager.is_loaded():
                logger.info("Model not loaded, loading it first...")
                if not model_manager.load_model():
                    raise HTTPException(status_code=500, detail="Failed to load model")

        # Parse LoRA parameters from form-data
        loras_to_apply = []
        remove_all_loras = False
        try:
            if loras_json is not None:
                parsed = json.loads(loras_json)
                if not isinstance(parsed, list):
                    raise ValueError("loras_json must be a JSON array")
                if len(parsed) == 0:
                    remove_all_loras = True
                else:
                    for item in parsed:
                        name = (item.get("name") or "").strip()
                        weight = float(item.get("weight", 1.0))
                        if not name:
                            raise ValueError("LoRA name cannot be empty")
                        if weight < 0 or weight > 2.0:
                            raise ValueError("LoRA weight must be between 0 and 2.0")
                        loras_to_apply.append({"name": name, "weight": weight})
            elif lora_name:
                name = lora_name.strip()
                if not name:
                    raise ValueError("LoRA name cannot be empty")
                w = 1.0 if lora_weight is None else float(lora_weight)
                if w < 0 or w > 2.0:
                    raise ValueError("LoRA weight must be between 0 and 2.0")
                loras_to_apply.append({"name": name, "weight": w})
            elif use_default_lora:
                loras_to_apply = [
                    {"name": DEFAULT_LORA_NAME, "weight": DEFAULT_LORA_WEIGHT}
                ]
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid LoRA parameters: {str(e)}"
            )

        # Apply LoRAs if requested
        if loras_to_apply or remove_all_loras:
            _ensure_model_loaded()
            _apply_loras(loras_to_apply, remove_all_loras)

        # Start timing
        generation_start_time = time.time()

        # Generate image via queue processor
        def processor(_req, _ctx):
            logger.info(
                f"Calling model_manager.generate_image_with_image with prompt: {prompt}"
            )
            logger.info(
                f"Using parameters from request: num_inference_steps={_req.num_inference_steps}, guidance_scale={_req.guidance_scale}, width={_req.width}, height={_req.height}, seed={_req.seed}"
            )
            return model_manager.generate_image_with_image(
                prompt=prompt,
                image=pre_img,
                num_inference_steps=_req.num_inference_steps,
                guidance_scale=_req.guidance_scale,
                width=tgt_w,
                height=tgt_h,
                seed=_req.seed,
            )

        try:
            result = await queue_manager.submit_and_wait(
                prompt=prompt,
                width=tgt_w,
                height=tgt_h,
                seed=seed,
                num_inference_steps=num_inference_steps or INFERENCE_STEPS,
                guidance_scale=guidance_scale or DEFAULT_GUIDANCE_SCALE,
                processor=processor,
                context={},
            )
        except Exception as gen_error:
            logger.error(f"Model generation failed: {gen_error}")
            raise HTTPException(
                status_code=500, detail=f"Image generation failed: {str(gen_error)}"
            )

        # Extract and save the generated image
        try:
            generated_image = extract_image_from_result(result)
            if generated_image is None:
                raise RuntimeError("Failed to extract image from generation result")
        except Exception as extract_error:
            logger.error(f"Failed to extract image from result: {extract_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract generated image: {str(extract_error)}",
            )

        # Optionally apply upscaling before returning bytes
        final_image_path: Optional[str]
        upscaled_image_path: Optional[str]
        try:
            if upscale:
                _, upscaled_image_path, final_w, final_h = apply_upscaling(
                    generated_image, upscale_factor or 2, save_image_with_unique_name
                )
                final_image_path = upscaled_image_path or save_image_with_unique_name(
                    generated_image
                )
                logger.info(
                    f"Upscaling applied: factor={upscale_factor}, path={final_image_path}, size={final_w}x{final_h}"
                )
            else:
                final_image_path = save_image_with_unique_name(generated_image)
                logger.info(f"Saved image without upscaling to: {final_image_path}")
        except Exception as save_error:
            logger.error(f"Failed to save generated image: {save_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save generated image: {str(save_error)}",
            )

        # Read the final image file and return bytes (binary response)
        file_path = final_image_path
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, detail="Generated image file not found"
            )

        with open(file_path, "rb") as f:
            out_bytes = f.read()

        # Clean up the file after reading
        try:
            os.remove(file_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary image file: {cleanup_error}")

        # If background removal requested: process bytes result via temp file path.
        # For consistency with other endpoints, try to reuse saved file path if available in logs; otherwise return original.
        if remove_background:
            try:
                # Save bytes to a temp image, process, and return processed bytes
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                tmp_path = (
                    Path(base_dir)
                    / "generated_images"
                    / f"tmp_{int(time.time()*1000)}.png"
                )
                with open(tmp_path, "wb") as f:
                    f.write(out_bytes)
                with Image.open(tmp_path).convert("RGB") as im:
                    out_im = remove_background_birefnet(im, bg_strength)
                new_rel = save_image_with_unique_name(out_im)
                new_abs = Path(base_dir) / new_rel
                with open(new_abs, "rb") as f:
                    processed_bytes = f.read()
                try:
                    os.remove(tmp_path)
                    os.remove(new_abs)
                except Exception:
                    pass
                return Response(content=processed_bytes, media_type="image/png")
            except Exception as e:
                logger.error(f"Background removal failed (and-return): {e}")
                # fallthrough to return original image_bytes
        return Response(content=out_bytes, media_type="image/png")

    except Exception as e:
        raise handle_api_error("generate_with_image_and_return", e)


@router.post("/generate-with-image")
async def generate_with_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    num_inference_steps: Optional[int] = Form(INFERENCE_STEPS),
    guidance_scale: Optional[float] = Form(DEFAULT_GUIDANCE_SCALE),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(512),
    seed: Optional[int] = Form(None),
    remove_background: Optional[bool] = Form(False),
    bg_strength: Optional[float] = Form(None),
    lora_name: Optional[str] = Form(None),
    lora_weight: Optional[float] = Form(None),
    loras_json: Optional[str] = Form(None),
    use_default_lora: Optional[bool] = Form(False),
    upscale: Optional[bool] = Form(False),
    upscale_factor: Optional[int] = Form(2),
    downscale: Optional[bool] = Form(True),
):
    """Generate image using image + text input (image-to-image generation)"""
    try:
        # Debug logging for form parameters
        logger.info(
            f"Received generate-with-image request - prompt: {prompt}, num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, width: {width}, height: {height}, seed: {seed}, downscale: {downscale}"
        )
        logger.info(
            f"Image file: {image.filename}, content_type: {image.content_type}, size: {image.size}"
        )

        # Validate parameters
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        if width and (width < 256 or width > 1024):
            raise HTTPException(
                status_code=400, detail="width must be between 256 and 1024"
            )

        if height and (height < 256 or height > 1024):
            raise HTTPException(
                status_code=400, detail="height must be between 256 and 1024"
            )

        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")

        # Validate image file type
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Check file size (max 10MB)
        if image.size and image.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="Image file too large (max 10MB)"
            )

        # Check file extension
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        file_extension = os.path.splitext(image.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format. Allowed: {', '.join(allowed_extensions)}",
            )

        try:
            raw = await image.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            pre_img, tgt_w, tgt_h = kontext_preprocess(img, downscale=downscale)
        except Exception as img_error:
            logger.error(f"Failed to load image: {img_error}")
            raise HTTPException(
                status_code=400, detail=f"Failed to load image: {str(img_error)}"
            )

        # Ensure model is loaded
        with _model_manager_lock:
            if not model_manager.is_loaded():
                logger.info("Model not loaded, loading it first...")
                if not model_manager.load_model():
                    raise HTTPException(status_code=500, detail="Failed to load model")

        # Parse LoRA parameters from form-data
        loras_to_apply = []
        remove_all_loras = False
        try:
            if loras_json is not None:
                parsed = json.loads(loras_json)
                if not isinstance(parsed, list):
                    raise ValueError("loras_json must be a JSON array")
                if len(parsed) == 0:
                    remove_all_loras = True
                else:
                    for item in parsed:
                        name = (item.get("name") or "").strip()
                        weight = float(item.get("weight", 1.0))
                        if not name:
                            raise ValueError("LoRA name cannot be empty")
                        if weight < 0 or weight > 2.0:
                            raise ValueError("LoRA weight must be between 0 and 2.0")
                        loras_to_apply.append({"name": name, "weight": weight})
            elif lora_name:
                name = lora_name.strip()
                if not name:
                    raise ValueError("LoRA name cannot be empty")
                w = 1.0 if lora_weight is None else float(lora_weight)
                if w < 0 or w > 2.0:
                    raise ValueError("LoRA weight must be between 0 and 2.0")
                loras_to_apply.append({"name": name, "weight": w})
            elif use_default_lora:
                loras_to_apply = [
                    {"name": DEFAULT_LORA_NAME, "weight": DEFAULT_LORA_WEIGHT}
                ]
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid LoRA parameters: {str(e)}"
            )

        # Apply LoRAs if requested
        if loras_to_apply or remove_all_loras:
            _ensure_model_loaded()
            _apply_loras(loras_to_apply, remove_all_loras)

        # Start timing
        generation_start_time = time.time()

        def processor(_req, _ctx):
            logger.info(
                f"Using parameters from request: num_inference_steps={_req.num_inference_steps}, guidance_scale={_req.guidance_scale}, width={_req.width}, height={_req.height}, seed={_req.seed}"
            )
            return model_manager.generate_image_with_image(
                prompt=prompt,
                image=pre_img,
                num_inference_steps=_req.num_inference_steps,
                guidance_scale=_req.guidance_scale,
                width=_req.width,
                height=_req.height,
                seed=_req.seed,
            )

        try:
            result = await queue_manager.submit_and_wait(
                prompt=prompt,
                width=tgt_w,
                height=tgt_h,
                seed=seed,
                num_inference_steps=num_inference_steps or INFERENCE_STEPS,
                guidance_scale=guidance_scale or DEFAULT_GUIDANCE_SCALE,
                processor=processor,
                context={},
            )
        except Exception as gen_error:
            logger.error(f"Model generation failed: {gen_error}")
            raise HTTPException(
                status_code=500, detail=f"Image generation failed: {str(gen_error)}"
            )

        # Calculate generation time

        # Extract and save the generated image
        try:
            generated_image = extract_image_from_result(result)
            if generated_image is None:
                raise RuntimeError("Failed to extract image from generation result")
        except Exception as extract_error:
            logger.error(f"Failed to extract image from result: {extract_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract generated image: {str(extract_error)}",
            )

        # Optionally apply upscaling to saved file
        try:
            if upscale:
                image_filename, upscaled_image_filename, final_w, final_h = (
                    apply_upscaling(
                        generated_image,
                        upscale_factor or 2,
                        save_image_with_unique_name,
                    )
                )
                image_filename = upscaled_image_filename or image_filename
                tgt_w, tgt_h = final_w, final_h
                logger.info(
                    f"Upscaling applied: factor={upscale_factor}, path={image_filename}, size={tgt_w}x{tgt_h}"
                )
            else:
                image_filename = save_image_with_unique_name(generated_image)
                logger.info(f"Successfully saved image to: {image_filename}")

        except Exception as save_error:
            logger.error(f"Failed to save generated image: {save_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save generated image: {str(save_error)}",
            )

        # Apply background removal if requested
        if remove_background:
            try:
                # Process the saved image for background removal
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                abs_image_path = Path(base_dir) / image_filename

                with Image.open(abs_image_path).convert("RGB") as im:
                    out_im = remove_background_birefnet(im, bg_strength)

                # Save the processed image with a new name
                new_rel = save_image_with_unique_name(out_im)
                new_filename = os.path.basename(new_rel)

                # Clean up the original image
                os.remove(abs_image_path)

                # Update variables to use the processed image
                image_filename = new_rel
                filename = new_filename
                logger.info(f"Background removal completed, new image saved: {new_rel}")

            except Exception as e:
                logger.error(f"Background removal failed: {e}")
                filename = os.path.basename(image_filename)

        # Return JSON response in the same format as /generate endpoint
        if not remove_background:
            filename = os.path.basename(image_filename)
        download_url = f"/generated_images/{filename}"

        # Format generation time
        generation_time = time.time() - generation_start_time
        formatted_generation_time = f"{generation_time:.2f}s"
        actual_lora_info = model_manager.get_lora_info()

        return {
            "message": f"Generated image from uploaded image for prompt: {prompt}",
            "image_url": image_filename,
            "download_url": download_url,
            "filename": filename,
            "generation_time": formatted_generation_time,
            "lora_applied": actual_lora_info.get("name") if actual_lora_info else None,
            "lora_weight": actual_lora_info.get("weight") if actual_lora_info else None,
            "width": tgt_w,
            "height": tgt_h,
            "seed": seed,
        }

    except Exception as e:
        raise handle_api_error("generate_with_image", e)


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
        raise handle_api_error("model loading", e)


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


@router.post("/apply-lora")
async def apply_lora(request: ApplyLoRARequest):
    """Apply a LoRA to the current model - supports both local files and Hugging Face repositories"""
    if not model_manager.is_loaded():
        logger.error(f"Cannot apply LoRA {request.lora_name}: Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Determine the LoRA source and construct the appropriate identifier
        lora_identifier = request.lora_name

        # If it's a Hugging Face LoRA, use the repo_id and filename
        if request.repo_id and request.filename:
            lora_identifier = f"{request.repo_id}/{request.filename}"
            logger.info(f"Applying Hugging Face LoRA: {lora_identifier}")
        else:
            logger.info(f"Applying local LoRA: {lora_identifier}")

        if model_manager.apply_lora(lora_identifier, request.weight):
            logger.info(
                f"LoRA {lora_identifier} applied successfully with weight {request.weight}"
            )
            return {
                "message": f"LoRA {lora_identifier} applied successfully with weight {request.weight}",
                "lora_name": lora_identifier,
                "weight": request.weight,
                "status": "applied",
            }
        else:
            logger.error(
                f"Failed to apply LoRA {lora_identifier} - Model: {model_manager.is_loaded()}, Pipeline: {model_manager.get_pipeline() is not None}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to apply LoRA {lora_identifier}"
            )
    except Exception as e:
        raise handle_api_error("LoRA application", e)


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
        raise handle_api_error("LoRA removal", e)


# ============================================================================
# LoRA Fusion Endpoints
# ============================================================================


@router.post("/apply-lora-permanent")
async def apply_lora_permanent(request: dict):
    """Apply LoRAs permanently (fusion)"""
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
        logger.error(f"Error applying LoRAs permanently: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/fused-lora-status")
async def get_fused_lora_status():
    """Get the current fused LoRA status"""
    try:
        is_fused = model_manager.is_fused()
        fused_info = model_manager.get_fused_lora_info()
        return {"is_fused": is_fused, "fused_info": fused_info}
    except Exception as e:
        logger.error(f"Error getting fused LoRA status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/unfuse-loras")
async def unfuse_loras():
    """Unfuse LoRAs"""
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
            num_inference_steps=request.num_inference_steps or INFERENCE_STEPS,
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
        raise handle_api_error("submit request", e)


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
        raise handle_api_error("get request status", e)


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
    except Exception as e:
        raise handle_api_error("cancel request", e)


@router.get("/queue-stats")
async def get_queue_stats():
    """Get current queue statistics"""
    try:
        return queue_manager.get_queue_stats()
    except Exception as e:
        raise handle_api_error("get queue stats", e)


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
    downscale: Optional[bool] = None,
    num_inference_steps: int = INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
):
    """Internal function to generate images - used by both endpoints"""
    logger.info(f"Starting image generation for prompt: {prompt}")

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
            prompt,
            num_inference_steps,
            guidance_scale,
            width,
            height,
            seed,
        )
        image = extract_image_from_result(result)
        image_filename = save_image_with_unique_name(image)
        final_width = width
        final_height = height

        if upscale:
            image_filename, upscaled_image_filename, final_width, final_height = (
                apply_upscaling(image, upscale_factor, image_filename)
            )
            image_filename = upscaled_image_filename

        # Get the actual LoRA status from the model manager
        actual_lora_info = model_manager.get_lora_info()
        filename = os.path.basename(image_filename)
        download_url = f"/download/{filename}"

        # End timing
        end_time = time.time()
        generation_time = end_time - start_time

        return {
            "message": f"Generated {model_type_name} image for prompt: {prompt}",
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
        raise handle_api_error(f"{model_type_name} image generation", e)


@router.get("/loras")
async def get_available_loras():
    """Get list of available LoRA files (uploaded and default)"""
    try:
        # Get uploaded LoRAs from index.json
        uploaded_loras = []
        index_file = Path("uploads/lora_files/index.json")

        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    index_data = json.loads(f.read())
                    if "entries" in index_data and isinstance(
                        index_data["entries"], list
                    ):
                        uploaded_loras = index_data["entries"]
                        logger.info(
                            f"Loaded {len(uploaded_loras)} uploaded LoRAs from index.json"
                        )
                    else:
                        logger.warning("Invalid index.json format")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse index.json: {e}")
                uploaded_loras = []

        # No default LoRAs - users must explicitly select LoRAs
        default_loras = []

        return {
            "uploaded": uploaded_loras,
            "huggingface": [],
            "default": default_loras,
            "total_count": len(uploaded_loras) + len(default_loras),
        }

    except Exception as e:
        raise handle_api_error("get available LoRAs", e)


@router.post("/cache-hf-lora")
async def cache_huggingface_lora(request: dict):
    """Cache a Hugging Face LoRA and add it to the index"""
    try:
        repo_id = request.get("repo_id")
        filename = request.get("filename")
        display_name = request.get("display_name", f"{repo_id}/{filename}")

        if not repo_id or not filename:
            raise HTTPException(
                status_code=400, detail="repo_id and filename are required"
            )

        # Generate a unique stored name for the cached LoRA
        import hashlib
        import os
        import time

        # Create a hash-based name to avoid conflicts
        hash_input = f"{repo_id}_{filename}_{int(time.time())}"
        stored_name = (
            f"hf_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}.safetensors"
        )

        # Download and cache the LoRA using the existing model manager
        model_manager = get_model_manager()

        # Use the existing HF download logic from the model
        try:
            import shutil
            import tempfile

            from huggingface_hub import hf_hub_download

            # Create temp directory for download
            temp_dir = tempfile.mkdtemp(prefix="hf_lora_cache_")

            # Download the LoRA file
            logger.info(
                f"Downloading HF LoRA: repo_id={repo_id}, filename={filename}, temp_dir={temp_dir}"
            )
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=temp_dir,
            )

            logger.info(f"Downloaded file path: {downloaded_path}")

            # Ensure the downloaded file exists
            if not os.path.exists(downloaded_path):
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")

            logger.info(f"File exists, size: {os.path.getsize(downloaded_path)} bytes")

            # Move to uploads directory
            uploads_dir = Path("uploads/lora_files")
            uploads_dir.mkdir(parents=True, exist_ok=True)

            final_path = uploads_dir / stored_name

            # Copy instead of move to avoid issues
            shutil.copy2(downloaded_path, final_path)

            # Verify the file was copied successfully
            if not final_path.exists():
                raise FileNotFoundError(f"Failed to copy file to: {final_path}")

            # Get file size
            file_size = final_path.stat().st_size

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            # Add to index
            await update_lora_index(
                stored_name=stored_name,
                original_name=display_name,
                timestamp=int(time.time()),
                size=file_size,
            )

            logger.info(
                f"Successfully cached Hugging Face LoRA: {repo_id}/{filename} as {stored_name}"
            )

            return {
                "message": f"LoRA cached successfully",
                "stored_name": stored_name,
                "original_name": display_name,
                "size": file_size,
                "type": "huggingface",
            }

        except Exception as e:
            logger.error(f"Failed to download and cache HF LoRA: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to cache LoRA: {str(e)}"
            )

    except Exception as e:
        raise handle_api_error("cache Hugging Face LoRA", e)


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
            cleanup_after_upload()
        except Exception as cleanup_error:
            logger.warning(f"Failed to trigger cleanup after upload: {cleanup_error}")

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
        raise handle_api_error("upload LoRA file", e)


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
        raise handle_api_error("remove LoRA file", e)


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload a reference image to the server without generating."""
    try:
        validate_uploaded_image(file)
        file_path = save_uploaded_image(file)  # saves to uploads/images by default

        filename = os.path.basename(file_path)

        return {
            "message": "Image uploaded successfully",
            "filename": filename,
            "file_path": file_path,
        }
    except Exception as e:
        raise handle_api_error("upload image", e)


def _apply_background_removal_to_saved(
    download_url: str, bg_strength: Optional[float] = None
) -> str:
    """
    Apply BiRefNet background removal to saved image

    Args:
        download_url: Image download URL
        bg_strength: Background removal strength (0.0-1.0), currently unused

    Returns:
        Processed image download URL
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = download_url.split("/")[-1]
        abs_path = Path(base_dir) / "generated_images" / filename

        # Open image and convert to RGB format (BiRefNet requires RGB input)
        with Image.open(abs_path).convert("RGB") as im:
            # Use BiRefNet for background removal
            output_im = remove_background_birefnet(im, bg_strength)

        # Save processed image
        new_rel = save_image_with_unique_name(output_im)  # saves into generated_images
        return f"/generated_images/{Path(new_rel).name}"

    except Exception as e:
        logger.error(f"BiRefNet background removal processing failed: {e}")
        return download_url
