"""
API routes for the FLUX API
"""

import time
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from models.flux_model import FluxModelManager
from utils.image_utils import extract_image_from_result, save_image_with_unique_name
from utils.system_utils import get_system_memory
from config.settings import STATIC_IMAGES_DIR

# Create router
router = APIRouter()

# Global model manager instance
model_manager = FluxModelManager()


@router.get("/")
def read_root():
    """Root endpoint for testing"""
    return {
        "message": "FLUX API is running!",
        "endpoints": ["/static-image", "/generate"],
        "model_loaded": model_manager.is_loaded(),
        "model_type": model_manager.model_type,
    }


@router.get("/static-image")
def get_static_image():
    """Serve static images"""
    image_path = f"{STATIC_IMAGES_DIR}/sample.jpg"
    return FileResponse(image_path)


@router.post("/generate")
async def generate_image(request: Request):
    """Generate image using FLUX model - accepts both JSON and form data"""
    try:
        # Try to parse as JSON first
        try:
            json_data = await request.json()
            prompt = json_data.get("prompt")
            if not prompt:
                raise HTTPException(
                    status_code=400, detail="Missing 'prompt' field in JSON"
                )
        except:
            # If JSON parsing fails, try form data
            try:
                form_data = await request.form()
                prompt = form_data.get("prompt")
                if not prompt:
                    raise HTTPException(
                        status_code=400, detail="Missing 'prompt' field in form data"
                    )
                # Convert to string if it's not already
                if not isinstance(prompt, str):
                    prompt = str(prompt)
            except:
                raise HTTPException(
                    status_code=400,
                    detail="Could not parse request as JSON or form data",
                )

        return generate_image_internal(prompt, "FLUX")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Request processing failed: {str(e)}"
        )


@router.post("/load-model")
def load_model():
    """Load the FLUX model"""
    if model_manager.load_model():
        return {"message": "FLUX model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load FLUX model")


@router.get("/model-status")
def get_model_status():
    """Get the current model status"""
    status = model_manager.get_model_status()
    system_memory_used, system_memory_total = get_system_memory()
    
    status.update({
        "system_memory_used_gb": f"{system_memory_used:.2f}GB",
        "system_memory_total_gb": f"{system_memory_total:.2f}GB",
    })
    
    return status


@router.get("/gpu-info")
def get_gpu_info():
    """Get detailed GPU information"""
    return model_manager.gpu_manager.get_device_info()


def generate_image_internal(prompt: str, model_type_name: str = "FLUX"):
    """Internal function to generate images - used by both endpoints"""
    if not model_manager.is_loaded():
        # Try to load the model if not already loaded
        if not model_manager.load_model():
            raise HTTPException(status_code=500, detail="Failed to load FLUX model")

    try:
        print(f"Generating {model_type_name} image for prompt: {prompt}")

        # Start timing
        start_time = time.time()

        # Generate image with FLUX
        if model_manager.get_pipeline() is None:
            raise HTTPException(
                status_code=500,
                detail=f"{model_type_name} model not properly loaded",
            )

        # Generate the image
        result = model_manager.generate_image(prompt)
        image = extract_image_from_result(result)

        # End timing
        end_time = time.time()
        generation_time = end_time - start_time

        # Save image with unique name
        image_filename = save_image_with_unique_name(image)

        # Get system information
        vram_usage = model_manager.gpu_manager.get_vram_usage()
        system_memory_used, system_memory_total = get_system_memory()

        return {
            "message": f"Generated {model_type_name} image for prompt: {prompt}",
            "image_url": image_filename,
            "generation_time": f"{generation_time:.2f}s",
            "vram_usage_gb": f"{vram_usage:.2f}GB",
            "system_memory_used_gb": f"{system_memory_used:.2f}GB",
            "system_memory_total_gb": f"{system_memory_total:.2f}GB",
            "model_type": model_manager.model_type,
        }

    except Exception as e:
        print(f"Error generating {model_type_name} image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"{model_type_name} image generation failed: {str(e)}",
        )
