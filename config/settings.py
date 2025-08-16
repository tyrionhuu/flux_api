"""
Configuration settings for the FLUX API
"""

# Model Configuration
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
NUNCHAKU_MODEL_ID = "nunchaku-tech/nunchaku-flux.1-dev"

# Quantized Model Files
FP4_WEIGHTS_FILE = "svdq-fp4_r32-flux.1-dev.safetensors"
INT4_WEIGHTS_FILE = "svdq-int4_r32-flux.1-dev.safetensors"

# Model Types
MODEL_TYPE_STANDARD_CPU = "flux_cpu"
MODEL_TYPE_STANDARD_GPU = "flux_gpu"
MODEL_TYPE_QUANTIZED_GPU = "flux_quantized_gpu"

# Device Configuration
DEFAULT_GPU = 0
DEFAULT_DEVICE_MAP = "balanced"
DEFAULT_TORCH_DTYPE_CPU = "float32"
DEFAULT_TORCH_DTYPE_GPU = "bfloat16"

# Memory Configuration
LOW_CPU_MEMORY_USAGE = True
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
