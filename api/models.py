"""
Pydantic models for API requests and responses
"""

from typing import Optional

from pydantic import BaseModel, Field

from config.settings import DEFAULT_GUIDANCE_SCALE, DEFAULT_INFERENCE_STEPS


class LoRAConfig(BaseModel):
    """Model for individual LoRA configuration"""

    name: str = Field(..., description="Hugging Face repository ID or local path")
    weight: float = Field(1.0, ge=0.0, le=2.0, description="LoRA weight (0.0 to 2.0)")


class GenerateRequest(BaseModel):
    """Request model for image generation with optional LoRA parameters"""

    prompt: str = Field(..., description="Text prompt for image generation")
    # Support for multiple LoRAs
    loras: Optional[list[LoRAConfig]] = Field(
        None,
        description="List of LoRAs to apply. Each LoRA has a name and weight. If not specified, default LoRA will be used.",
    )
    # Legacy support for single LoRA (deprecated but maintained for backward compatibility)
    lora_name: Optional[str] = Field(
        None,
        description="[DEPRECATED] Single LoRA name. Use 'loras' list instead for multiple LoRA support.",
    )
    lora_weight: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="[DEPRECATED] Single LoRA weight. Use 'loras' list instead for multiple LoRA support.",
    )
    width: Optional[int] = Field(
        512, ge=256, le=1024, description="Image width in pixels (256-1024)"
    )
    height: Optional[int] = Field(
        512, ge=256, le=1024, description="Image height in pixels (256-1024)"
    )
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=2**32 - 1,
        description="Random seed for reproducible results (0-4294967295)",
    )
    upscale: Optional[bool] = Field(
        False, description="Whether to upscale the generated image using Remacri ESRGAN"
    )
    upscale_factor: Optional[int] = Field(
        2, ge=2, le=4, description="Upscaling factor: 2 for 2x, 4 for 4x (default: 2)"
    )
    guidance_scale: Optional[float] = Field(
        DEFAULT_GUIDANCE_SCALE,
        ge=-10.0,
        le=10.0,
        description=f"Guidance scale for image generation (-10.0 to 10.0, default: {DEFAULT_GUIDANCE_SCALE})",
    )


class GenerateResponse(BaseModel):
    """Response model for image generation"""

    message: str = Field(..., description="Success message")
    image_url: str = Field(..., description="Path to the generated image on server")
    download_url: str = Field(..., description="URL endpoint to download the image")
    filename: str = Field(..., description="Generated image filename")
    generation_time: str = Field(..., description="Time taken to generate the image")
    vram_usage_gb: str = Field(..., description="VRAM usage in GB")
    system_memory_used_gb: str = Field(..., description="System memory used in GB")
    system_memory_total_gb: str = Field(..., description="Total system memory in GB")
    model_type: str = Field(..., description="Type of model used")
    lora_applied: Optional[str] = Field(None, description="LoRA file that was applied")
    lora_weight: Optional[float] = Field(None, description="LoRA weight that was used")
    # Generation parameters used
    num_inference_steps: int = Field(..., description="Number of inference steps used")
    guidance_scale: float = Field(..., description="Guidance scale used")
    width: int = Field(..., description="Image width generated")
    height: int = Field(..., description="Image height generated")
    seed: Optional[int] = Field(None, description="Random seed used (if any)")


class LoRAInfo(BaseModel):
    """Model for LoRA information"""

    name: str = Field(..., description="LoRA file name")
    weight: float = Field(..., description="LoRA weight applied")
    status: str = Field(..., description="Status of LoRA application")


class ModelStatusResponse(BaseModel):
    """Response model for model status"""

    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_type: str = Field(..., description="Type of model loaded")
    selected_gpu: Optional[int] = Field(None, description="Selected GPU ID")
    vram_usage_gb: str = Field(..., description="VRAM usage in GB")
    system_memory_used_gb: str = Field(..., description="System memory used in GB")
    system_memory_total_gb: str = Field(..., description="Total system memory in GB")
    lora_loaded: Optional[str] = Field(None, description="Currently loaded LoRA file")
    lora_weight: Optional[float] = Field(None, description="Current LoRA weight")
