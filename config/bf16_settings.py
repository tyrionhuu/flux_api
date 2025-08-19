"""
Configuration settings for the bf16 FLUX API (Port 8001)
"""

# Model Configuration - bf16 version
BF16_MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# Model Types
MODEL_TYPE_BF16_GPU = "flux_bf16_gpu"

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
API_TITLE = "FLUX API (bf16)"
API_DESCRIPTION = "High-performance image generation using FLUX.1-schnell bf16 model"
API_VERSION = "1.0.0"

# Cache Configuration
HUGGINGFACE_CACHE_DIR = "~/.cache/huggingface/hub"

# File Paths
GENERATED_IMAGES_DIR = "generated_images"
STATIC_IMAGES_DIR = "static"

# LoRA Configuration
DEFAULT_LORA_NAME = (
    "aleksa-codes/flux-ghibsky-illustration/lora.safetensors"  # Default LoRA to apply
)
DEFAULT_LORA_WEIGHT = 1.0
MIN_LORA_WEIGHT = 0.0
MAX_LORA_WEIGHT = 2.0

# Port Configuration
BF16_API_PORT = 8001
