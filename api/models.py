"""
Pydantic models for API requests and responses
"""

from typing import Optional

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """Model for individual LoRA configuration"""

    name: str = Field(..., description="Hugging Face repository ID or local path")
    weight: float = Field(1.0, ge=0.0, le=2.0, description="LoRA weight (0.0 to 2.0)")


class ApplyLoRARequest(BaseModel):
    """Request model for applying a single LoRA to the model"""

    lora_name: str = Field(..., description="LoRA name to apply")
    weight: float = Field(1.0, ge=0.0, le=2.0, description="LoRA weight (0.0 to 2.0)")
    repo_id: Optional[str] = Field(
        None, description="Hugging Face repository ID (for HF LoRAs)"
    )
    filename: Optional[str] = Field(
        None, description="Specific filename in the repository (for HF LoRAs)"
    )


class GenerateRequest(BaseModel):
    """Request model for image generation with optional LoRA parameters"""

    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(
        "", description="Text to steer the model away from (optional)"
    )
    # Support for multiple LoRAs
    loras: Optional[list[LoRAConfig]] = Field(
        None,
        description="List of LoRAs to apply. Each LoRA has a name and weight. If not specified, no LoRA will be applied.",
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
    num_inference_steps: Optional[int] = Field(
        20,
        ge=1,
        le=100,
        description="Number of inference steps for generation (1-100, default: 20)",
    )
    guidance_scale: Optional[float] = Field(
        3.5,
        ge=-10.0,
        le=10.0,
        description="Guidance scale for generation (-10.0 to 10.0, default: 3.5)",
    )
    true_cfg_scale: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=20.0,
        description="True CFG scale (>1.0 enables negative_prompt in FluxKontext)",
    )
    upscale: Optional[bool] = Field(
        False, description="Whether to upscale the generated image using Remacri ESRGAN"
    )
    upscale_factor: Optional[int] = Field(
        2, ge=2, le=4, description="Upscaling factor: 2 for 2x, 4 for 4x (default: 2)"
    )
    downscale: Optional[bool] = Field(
        True,
        description="Whether to automatically downscale large images (>512px) by half before processing",
    )
    remove_background: Optional[bool] = Field(
        False, description="Whether to remove background from the final image"
    )
    bg_strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Background removal strength (0.0 gentle → 1.0 aggressive)",
    )


class ImageUploadGenerateRequest(BaseModel):
    """Request model for image generation with uploaded image and prompt"""

    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(
        "", description="Text to steer the model away from (optional)"
    )
    # Support for multiple LoRAs
    loras: Optional[list[LoRAConfig]] = Field(
        None,
        description="List of LoRAs to apply. Each LoRA has a name and weight. If not specified, no LoRA will be applied.",
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
    num_inference_steps: Optional[int] = Field(
        20,
        ge=1,
        le=100,
        description="Number of inference steps for generation (1-100, default: 20)",
    )
    guidance_scale: Optional[float] = Field(
        3.5,
        ge=-10.0,
        le=10.0,
        description="Guidance scale for generation (-10.0 to 10.0, default: 3.5)",
    )
    true_cfg_scale: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=20.0,
        description="True CFG scale (>1.0 enables negative_prompt in FluxKontext)",
    )
    upscale: Optional[bool] = Field(
        False, description="Whether to upscale the generated image using Remacri ESRGAN"
    )
    upscale_factor: Optional[int] = Field(
        2, ge=2, le=4, description="Upscaling factor: 2 for 2x, 4 for 4x (default: 2)"
    )
    downscale: Optional[bool] = Field(
        True,
        description="Whether to automatically downscale large images (>512px) by half before processing",
    )
    remove_background: Optional[bool] = Field(
        False, description="Whether to remove background from the final image"
    )
    bg_strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Background removal strength (0.0 gentle → 1.0 aggressive)",
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
