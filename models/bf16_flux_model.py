"""
BF16 FLUX model management for the FLUX API (Port 8001)
This extends the base FluxModelManager to avoid code duplication.
"""

import logging
import torch
from typing import Optional, Any
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from config.bf16_settings import (
    BF16_MODEL_ID,
    MODEL_TYPE_BF16_GPU,
    DEFAULT_LORA_NAME,
    DEFAULT_LORA_WEIGHT,
)
from models.fp4_flux_model import FluxModelManager

# Configure logging
logger = logging.getLogger(__name__)


class BF16FluxModelManager(FluxModelManager):
    """Manages BF16 FLUX model loading - extends base FluxModelManager"""

    def __init__(self):
        super().__init__()
        self.model_type = "none"

    def load_model(self) -> bool:
        """Load the BF16 FLUX model with GPU-only support"""
        try:
            logger.info("Loading BF16 FLUX model...")

            # Check if CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available. This model requires GPU support."
                )

            # Check for balanced multi-GPU mode
            visible_gpu_count = torch.cuda.device_count()
            logger.info(f"Visible GPUs: {visible_gpu_count}")

            device_map = None
            if visible_gpu_count > 1:
                # Balanced multi-GPU mode
                logger.info(
                    f"Using balanced multi-GPU mode across {visible_gpu_count} GPUs"
                )
                device_map = "balanced"
                device = "cuda"
            else:
                # Single GPU mode
                device = "cuda:0"
                logger.info(f"Using single GPU mode: {device}")
                try:
                    torch.cuda.set_device(0)
                except Exception:
                    pass

                # Verify device is set correctly
                current_device = torch.cuda.current_device()
                logger.info(f"Current CUDA device: {current_device}, Target device: 0")
                if current_device != 0:
                    logger.warning(
                        f"Device mismatch! Current: {current_device}, Target device: 0"
                    )
                    torch.cuda.set_device(0)
                    current_device = torch.cuda.current_device()
                    logger.info(f"Device after force set: {current_device}")

            # Load the standard FLUX.1-dev model in bf16 precision
            logger.info("Loading BF16 FLUX.1-dev model...")

            try:
                # Create FluxPipeline with bf16 precision
                logger.info(f"Loading model to device: {device}")
                logger.info(
                    f"Current CUDA device before loading: {torch.cuda.current_device()}"
                )

                if device_map == "balanced":
                    # Multi-GPU balanced mode
                    logger.info("Loading BF16 pipeline with balanced device map")
                    self.pipe = FluxPipeline.from_pretrained(
                        BF16_MODEL_ID,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map,
                    )
                else:
                    # Single GPU mode
                    self.pipe = FluxPipeline.from_pretrained(
                        BF16_MODEL_ID,
                        torch_dtype=torch.bfloat16,
                    ).to(device)

                logger.info(
                    f"Model loaded. Current CUDA device after loading: {torch.cuda.current_device()}"
                )
                logger.info(
                    f"Pipeline device: {self.pipe.device if hasattr(self.pipe, 'device') else 'unknown'}"
                )

                # Verify device consistency
                logger.debug(
                    f"Device consistency - Target: {device}, Pipeline: {self.pipe.device if hasattr(self.pipe, 'device') else 'unknown'}"
                )

                self.model_type = MODEL_TYPE_BF16_GPU
                logger.info("BF16 FLUX.1-dev model loaded successfully!")

            except Exception as bf16_error:
                logger.error(
                    f"Error loading BF16 FLUX.1-dev model: {bf16_error} (Type: {type(bf16_error).__name__})"
                )
                raise RuntimeError(
                    f"Failed to load BF16 FLUX.1-dev model: {bf16_error}."
                )

            self.model_loaded = True
            # Reset LoRA state when loading a new model
            self.current_lora = None
            self.current_weight = 1.0
            logger.info(f"BF16 FLUX model loaded successfully on {device}!")

            # Perform CUDA Graph warm-up for better performance
            self._warmup_cuda_graph()
            
            # Apply default LoRA
            self._apply_default_lora_bf16()

            return True

        except Exception as e:
            logger.error(
                f"Error loading BF16 FLUX model: {e} (Type: {type(e).__name__})"
            )
            return False

    def _apply_default_lora_bf16(self):
        """Apply the default LoRA after BF16 model loading"""
        try:
            if not self.pipe:
                logger.warning("Pipeline not loaded, cannot apply default LoRA")
                return False
                
            logger.info(f"Applying default LoRA to BF16 model: {DEFAULT_LORA_NAME} with weight {DEFAULT_LORA_WEIGHT}")
            
            # Load and apply LoRA using diffusers method
            logger.info(f"   - Loading default LoRA weights from {DEFAULT_LORA_NAME}")
            try:
                # For BF16 model, we need to use just the repo name without /lora.safetensors
                lora_repo = DEFAULT_LORA_NAME.replace("/lora.safetensors", "")
                self.pipe.load_lora_weights(lora_repo)
                logger.info(f"   - Default LoRA weights loaded successfully")
            except Exception as load_error:
                logger.warning(f"   - Failed to load default LoRA weights: {load_error}")
                return False

            # Set LoRA scale
            logger.info(f"   - Setting default LoRA adapter weight to {DEFAULT_LORA_WEIGHT}")
            try:
                adapter_name = lora_repo.split("/")[-1] if "/" in lora_repo else lora_repo
                self.pipe.set_adapters([adapter_name], adapter_weights=[DEFAULT_LORA_WEIGHT])
                logger.info(f"   - Default LoRA adapter weight set successfully")
            except Exception as adapter_error:
                logger.warning(f"   - Failed to set default LoRA adapter weight: {adapter_error}")
                return False

            self.current_lora = DEFAULT_LORA_NAME
            self.current_weight = DEFAULT_LORA_WEIGHT
            logger.info(f"Default LoRA {DEFAULT_LORA_NAME} applied successfully to BF16 model with weight {DEFAULT_LORA_WEIGHT}")
            return True

        except Exception as e:
            logger.warning(f"Failed to apply default LoRA to BF16 model: {e} - continuing without default LoRA")
            return False

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> Any:
        """Generate image using the loaded BF16 model - GPU only with multi-GPU support"""
        if not self.model_loaded or self.pipe is None:
            raise RuntimeError("Model not loaded")

        # Ensure we're using GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU required for image generation.")

        # Set device for generation based on mode
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count > 1:
            logger.info(
                f"Generating with balanced multi-GPU mode ({visible_gpu_count} GPUs)"
            )
        else:
            try:
                torch.cuda.set_device(0)
                logger.info("Generating on cuda:0 (single GPU)")
            except Exception as gpu_error:
                logger.error(
                    f"GPU error during device selection: {gpu_error} (Type: {type(gpu_error).__name__})"
                )
                raise RuntimeError(
                    f"GPU error: {gpu_error}. GPU required for image generation."
                )

        # Generate the image using the FluxPipeline
        try:
            logger.info(f"Generating BF16 image with prompt: {prompt}")

            # Try with user-specified parameters first
            try:
                logger.info(
                    f"Generating with parameters: steps={num_inference_steps}, guidance={guidance_scale}, size={width}x{height}"
                )

                # Set seed if provided
                if seed is not None:
                    torch.manual_seed(seed)
                    logger.info(f"Using seed: {seed}")

                # Prepare generation kwargs
                generation_kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }

                # Add negative prompt if provided
                if negative_prompt:
                    generation_kwargs["negative_prompt"] = negative_prompt

                result = self.pipe(**generation_kwargs)
                logger.info("BF16 image generation completed successfully")
                return result

            except Exception as memory_error:
                if "CUDA" in str(memory_error) or "memory" in str(memory_error).lower():
                    logger.warning(
                        f"CUDA memory error detected: {memory_error} (Type: {type(memory_error).__name__})"
                    )
                    logger.info("Trying with reduced parameters...")

                    # Fallback with reduced parameters
                    fallback_steps = min(num_inference_steps // 2, 10)
                    fallback_guidance = max(guidance_scale * 0.6, 1.0)

                    generation_kwargs = {
                        "prompt": prompt,
                        "num_inference_steps": fallback_steps,
                        "guidance_scale": fallback_guidance,
                    }

                    if negative_prompt:
                        generation_kwargs["negative_prompt"] = negative_prompt

                    result = self.pipe(**generation_kwargs)
                    logger.info(
                        f"BF16 image generation completed with reduced parameters: steps={fallback_steps}, guidance={fallback_guidance}"
                    )
                    return result
                else:
                    # Re-raise if it's not a memory error
                    logger.error(
                        f"Non-memory error during BF16 image generation: {memory_error} (Type: {type(memory_error).__name__})"
                    )
                    raise memory_error

        except Exception as e:
            logger.error(
                f"Error in BF16 image generation: {e} (Type: {type(e).__name__})"
            )
            raise RuntimeError(f"Failed to generate BF16 image: {e}")

    def apply_lora(self, lora_name: str, lora_weight: float = 1.0) -> bool:
        """Apply LoRA to the BF16 model"""
        try:
            if not self.model_loaded or self.pipe is None:
                logger.error("Model not loaded, cannot apply LoRA")
                return False

            logger.info(
                f"Applying LoRA {lora_name} with weight {lora_weight} to BF16 model..."
            )

            # Load and apply LoRA
            self.pipe.load_lora_weights(lora_name)

            # Set LoRA scale
            self.pipe.set_adapters([lora_name], adapter_weights=[lora_weight])

            # Update state
            self.current_lora = lora_name
            self.current_weight = lora_weight

            logger.info(f"LoRA {lora_name} applied successfully to BF16 model")
            return True

        except Exception as e:
            logger.error(f"Failed to apply LoRA to BF16 model: {e}")
            return False

    def remove_lora(self) -> bool:
        """Remove LoRA from the BF16 model"""
        try:
            if not self.model_loaded or self.pipe is None:
                logger.error("Model not loaded, cannot remove LoRA")
                return False

            logger.info("Removing LoRA from BF16 model...")

            # Remove LoRA adapters
            self.pipe.set_adapters([])

            # Update state
            self.current_lora = None
            self.current_weight = 1.0

            logger.info("LoRA removed successfully from BF16 model")
            return True

        except Exception as e:
            logger.error(f"Failed to remove LoRA from BF16 model: {e}")
            return False

    def get_lora_info(self) -> Optional[dict]:
        """Get current LoRA information for BF16 model"""
        if self.current_lora:
            return {
                "name": self.current_lora,
                "weight": self.current_weight,
                "model_type": "bf16",
            }
        return None
