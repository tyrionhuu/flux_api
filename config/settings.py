"""
Configuration settings for the Diffusion API
"""

# Port Configuration (overridable via environment)
import os

API_PORT = int(os.environ.get("API_PORT", "8000"))

# Model Configuration
NUNCHAKU_MODEL_ID = "nunchaku-tech/nunchaku-flux.1-dev"

# Generation Configuration
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_INFERENCE_STEPS = 20

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

# API Configuration
API_TITLE = "Diffusion API"
API_DESCRIPTION = "High-performance image generation using FLUX model"
API_VERSION = "1.0.0"

# Cache Configuration
HUGGINGFACE_CACHE_DIR = "~/.cache/huggingface/hub"

# File Paths
GENERATED_IMAGES_DIR = "generated_images"
STATIC_IMAGES_DIR = "static"

# LoRA Configuration
# Removed DEFAULT_LORA_NAME and DEFAULT_LORA_WEIGHT - now configurable per LoRA
MIN_LORA_WEIGHT = 0.0
MAX_LORA_WEIGHT = 2.0

