"""
Configuration settings for the FLUX API
"""

# Model Configuration
NUNCHAKU_MODEL_ID = "nunchaku-tech/nunchaku-flux.1-kontext-dev"
DEFAULT_GUIDANCE_SCALE = 3.5
INFERENCE_STEPS = 20

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
API_TITLE = "FLUX API"
API_DESCRIPTION = "High-performance image generation using FLUX model"
API_VERSION = "1.0.0"

# Cache Configuration
HUGGINGFACE_CACHE_DIR = "~/.cache/huggingface/hub"

# File Paths
GENERATED_IMAGES_DIR = "generated_images"
STATIC_IMAGES_DIR = "static"

# LoRA Configuration
DEFAULT_LORA_NAME = None  # No default LoRA applied
DEFAULT_LORA_WEIGHT = 1.0
MIN_LORA_WEIGHT = 0.0
MAX_LORA_WEIGHT = 2.0
