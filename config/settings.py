"""
Configuration settings for the Diffusion API
"""

# Port Configuration (overridable via environment)
import os

API_PORT = int(os.environ.get("API_PORT", "8000"))

# Model Configuration
NUNCHAKU_FLUX_MODEL_ID = "nunchaku-tech/nunchaku-flux.1-dev"
NUNCHAKU_QWEN_IMAGE_MODEL_ID = "nunchaku-tech/nunchaku-qwen-image"
# Generation Configuration
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_INFERENCE_STEPS = 20

# API Configuration
API_TITLE = "Diffusion API"
API_DESCRIPTION = "High-performance image generation API"
API_VERSION = "1.0.0"

# Cache Configuration
HUGGINGFACE_CACHE_DIR = "~/.cache/huggingface/hub"

# File Paths
GENERATED_IMAGES_DIR = "generated_images"

# LoRA Configuration
# Removed DEFAULT_LORA_NAME and DEFAULT_LORA_WEIGHT - now configurable per LoRA
MIN_LORA_WEIGHT = 0.0
MAX_LORA_WEIGHT = 2.0

# MODEL_TYPE = "flux"
MODEL_TYPE = "qwen"
