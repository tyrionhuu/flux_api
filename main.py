from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from PIL import Image
import time
import uuid
import psutil
from typing import Optional, Union, Tuple, Any

app = FastAPI()

# Request model for the generate endpoint
class GenerateRequest(BaseModel):
    prompt: str

# Global variables for model management
pipe: Optional[FluxPipeline] = None
model_loaded = False
model_type = "none"
test_mode = True  # Enable test mode for development

def load_flux_model():
    """Load the FLUX model with BF16 precision"""
    global pipe, model_loaded, model_type
    
    if test_mode:
        print("Test mode: Simulating FLUX model load")
        model_loaded = True
        model_type = "bf16_test"
        print("FLUX model loaded successfully (test mode)!")
        return True
    
    try:
        print("Loading FLUX model...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        model_loaded = True
        model_type = "bf16"
        print("FLUX model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading FLUX model: {e}")
        return False

def load_quantized_model():
    """Load the quantized model (placeholder for now)"""
    global pipe, model_loaded, model_type
    
    if test_mode:
        print("Test mode: Simulating quantized FLUX model load")
        model_loaded = True
        model_type = "quantized_test"
        print("Quantized FLUX model loaded successfully (test mode)!")
        return True
    
    try:
        print("Loading quantized FLUX model...")
        # For now, we'll use the regular model as a placeholder
        # In the future, this will load the actual quantized model
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        model_loaded = True
        model_type = "quantized_placeholder"
        print("Quantized FLUX model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading quantized FLUX model: {e}")
        return False

def get_vram_usage():
    """Get current VRAM usage in GB"""
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    except:
        return 0.0

def get_system_memory():
    """Get system memory usage in GB"""
    try:
        memory = psutil.virtual_memory()
        return memory.used / 1024**3, memory.total / 1024**3
    except:
        return 0.0, 0.0

def create_test_image(prompt: str):
    """Create a test image for development purposes"""
    try:
        # Create a simple test image with the prompt text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a 512x512 image
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add text to the image
        text = f"Test Image: {prompt[:50]}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (512 - text_width) // 2
        y = (512 - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        return img
    except Exception as e:
        print(f"Error creating test image: {e}")
        # Create a simple colored image as fallback
        img = Image.new('RGB', (512, 512), color='lightblue')
        return img

def extract_image_from_result(result: Any) -> Image.Image:
    """Extract image from FLUX pipeline result"""
    try:
        # Handle different possible return types from FLUX pipeline
        if hasattr(result, 'images') and result.images:
            return result.images[0]
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            # If result is a tuple/list, try to find the image
            for item in result:
                if isinstance(item, Image.Image):
                    return item
                elif hasattr(item, 'images') and item.images:
                    return item.images[0]
        elif isinstance(result, Image.Image):
            return result
        
        # Fallback: create a placeholder image
        print("Warning: Could not extract image from result, using placeholder")
        return create_test_image("Placeholder - extraction failed")
        
    except Exception as e:
        print(f"Error extracting image from result: {e}")
        return create_test_image("Error - extraction failed")

def generate_image_internal(prompt: str, model_type_name: str = "FLUX"):
    """Internal function to generate images - used by both endpoints"""
    global pipe, model_loaded
    
    if not model_loaded:
        # Try to load the model if not already loaded
        if not load_flux_model():
            raise HTTPException(status_code=500, detail="Failed to load FLUX model")
    
    try:
        print(f"Generating {model_type_name} image for prompt: {prompt}")
        
        # Start timing
        start_time = time.time()
        
        if test_mode:
            # Test mode: create a test image
            if "quantized" in model_type_name.lower():
                image = create_test_image(f"Quantized: {prompt}")
                print("Test mode: Created quantized test image")
            else:
                image = create_test_image(prompt)
                print("Test mode: Created test image")
        else:
            # Real mode: generate image with FLUX
            if pipe is None:
                raise HTTPException(status_code=500, detail=f"{model_type_name} model not properly loaded")
            result = pipe(prompt)
            image = extract_image_from_result(result)
        
        # End timing
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Save image with unique name
        os.makedirs("generated_images", exist_ok=True)
        image_filename = f"generated_images/{uuid.uuid4()}.png"
        image.save(image_filename)
        
        # Get VRAM usage
        vram_usage = get_vram_usage()
        system_memory_used, system_memory_total = get_system_memory()
        
        return {
            "message": f"Generated {model_type_name} image for prompt: {prompt}",
            "image_url": image_filename,
            "generation_time": f"{generation_time:.2f}s",
            "vram_usage_gb": f"{vram_usage:.2f}GB",
            "system_memory_used_gb": f"{system_memory_used:.2f}GB",
            "system_memory_total_gb": f"{system_memory_total:.2f}GB",
            "model_type": model_type
        }
        
    except Exception as e:
        print(f"Error generating {model_type_name} image: {e}")
        raise HTTPException(status_code=500, detail=f"{model_type_name} image generation failed: {str(e)}")

# Placeholder endpoint to test FastAPI
@app.get("/static-image")
def get_static_image():
    # Path to your sample image
    image_path = "static/sample.jpg"
    return FileResponse(image_path)

@app.post("/generate")
async def generate_image(request: Request):
    """Generate image using FLUX model - accepts both JSON and form data"""
    try:
        # Try to parse as JSON first
        try:
            json_data = await request.json()
            prompt = json_data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Missing 'prompt' field in JSON")
        except:
            # If JSON parsing fails, try form data
            try:
                form_data = await request.form()
                prompt = form_data.get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="Missing 'prompt' field in form data")
                # Convert to string if it's not already
                if not isinstance(prompt, str):
                    prompt = str(prompt)
            except:
                raise HTTPException(status_code=400, detail="Could not parse request as JSON or form data")
        
        return generate_image_internal(prompt, "FLUX")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

@app.post("/generate-quantized")
async def generate_image_quantized(request: Request):
    """Generate image using quantized FLUX model - accepts both JSON and form data"""
    try:
        # Try to parse as JSON first
        try:
            json_data = await request.json()
            prompt = json_data.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="Missing 'prompt' field in JSON")
        except:
            # If JSON parsing fails, try form data
            try:
                form_data = await request.form()
                prompt = form_data.get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="Missing 'prompt' field in form data")
                # Convert to string if it's not already
                if not isinstance(prompt, str):
                    prompt = str(prompt)
            except:
                raise HTTPException(status_code=400, detail="Could not parse request as JSON or form data")
        
        return generate_image_internal(prompt, "Quantized FLUX")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

# Root endpoint for testing
@app.get("/")
def read_root():
    return {
        "message": "FLUX API is running!", 
        "endpoints": [
            "/static-image", 
            "/generate", 
            "/generate-quantized"
        ],
        "model_loaded": model_loaded,
        "model_type": model_type,
        "test_mode": test_mode
    }

# Model management endpoints
@app.post("/load-model")
def load_model():
    """Load the FLUX model"""
    if load_flux_model():
        return {"message": "FLUX model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load FLUX model")

@app.post("/load-quantized-model")
def load_quantized_model_endpoint():
    """Load the quantized FLUX model"""
    if load_quantized_model():
        return {"message": "Quantized FLUX model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load quantized FLUX model")

@app.get("/model-status")
def get_model_status():
    """Get the current model status"""
    return {
        "model_loaded": model_loaded,
        "model_type": model_type,
        "test_mode": test_mode,
        "vram_usage_gb": f"{get_vram_usage():.2f}GB",
        "system_memory_used_gb": f"{get_system_memory()[0]:.2f}GB"
    }
