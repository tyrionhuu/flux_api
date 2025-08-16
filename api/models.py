"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    """Request model for image generation with optional LoRA parameters"""
    prompt: str = Field(..., description="Text prompt for image generation")
    lora_name: Optional[str] = Field(None, description="Hugging Face repository ID (e.g., aleksa-codes/flux-ghibsky-illustration)")
    lora_weight: float = Field(1.0, ge=0.0, le=2.0, description="LoRA weight (0.0 to 2.0)")


class GenerateResponse(BaseModel):
    """Response model for image generation"""
    message: str = Field(..., description="Success message")
    image_url: str = Field(..., description="Path to the generated image")
    generation_time: str = Field(..., description="Time taken to generate the image")
    vram_usage_gb: str = Field(..., description="VRAM usage in GB")
    system_memory_used_gb: str = Field(..., description="System memory used in GB")
    system_memory_total_gb: str = Field(..., description="Total system memory in GB")
    model_type: str = Field(..., description="Type of model used")
    lora_applied: Optional[str] = Field(None, description="LoRA file that was applied")
    lora_weight: Optional[float] = Field(None, description="LoRA weight that was used")


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
