"""
Configuration settings for the FLUX API
"""

# Port Configuration (overridable via environment)
import os

FP4_API_PORT = int(os.environ.get("FP4_API_PORT", "8000"))

# Model Configuration
NUNCHAKU_MODEL_ID = "nunchaku-tech/nunchaku-flux.1-dev"

# Quantized Model Files (for reference - not currently used)
# FP4_WEIGHTS_FILE = "svdq-fp4_r32-flux.1-schnell.safetensors"
# INT4_WEIGHTS_FILE = "svdq-int4_r32-flux.1-schnell.safetensors"

# Model Types
MODEL_TYPE_QUANTIZED_GPU = "flux_quantized_gpu"

# Device Configuration
DEFAULT_GPU = 0
CUDA_TEST_TENSOR_SIZE = (100, 100)

# Image Configuration
DEFAULT_IMAGE_SIZE = (512, 512)
PLACEHOLDER_COLORS = {
    "default": "lightblue",
    "error": "red",
    "placeholder": "lightblue",
}

# Image Output Format Configuration
SAVE_AS_JPEG = False  # Keep PNG for fp4
JPEG_QUALITY = 70     # JPEG quality if enabled

# API Configuration
API_TITLE = "FLUX API"
API_DESCRIPTION = "High-performance image generation using FLUX model"
API_VERSION = "1.0.0"

# Cache Configuration
HUGGINGFACE_CACHE_DIR = "~/.cache/huggingface/hub"

# File Paths
GENERATED_IMAGES_DIR = "/data/pingzhi/generated_images"
STATIC_IMAGES_DIR = "static"

# LoRA Configuration
DEFAULT_LORA_NAME = "/home/user/checkpoints/Studio_Ghibli_Flux.safetensors"  # Default LoRA to apply
DEFAULT_LORA_WEIGHT = 1.0
MIN_LORA_WEIGHT = 0.0
MAX_LORA_WEIGHT = 2.0
