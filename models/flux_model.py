"""
FLUX model management for the FLUX API
"""

import logging
import torch
from typing import Optional, Any
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from config.settings import (
    NUNCHAKU_MODEL_ID,
    MODEL_TYPE_QUANTIZED_GPU,
)
from utils.gpu_manager import GPUManager

# Configure logging
logger = logging.getLogger(__name__)


class FluxModelManager:
    """Manages FLUX model loading and quantization"""

    def __init__(self):
        self.pipe: Optional[FluxPipeline] = (
            None  # Will be FluxPipeline with Nunchaku transformer
        )
        self.model_loaded = False
        self.model_type = "none"
        self.gpu_manager = GPUManager()
        # LoRA state
        self.current_lora: Optional[str] = None
        self.current_weight: float = 1.0

    def load_model(self) -> bool:
        """Load the FLUX model with GPU-only support and quantization"""
        try:
            logger.info("Loading FLUX model...")

            # Check if CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available. This model requires GPU support."
                )

            # Get optimal GPU device
            device, selected_gpu = self.gpu_manager.get_optimal_device()

            if device == "cpu":
                raise RuntimeError(
                    "GPU required. CPU mode is not supported for this model."
                )

            logger.info(f"Using device: {device}")

            # Always load Nunchaku model (this has LoRA support)
            logger.info("Loading Nunchaku model with LoRA support...")

            try:
                from nunchaku import NunchakuFluxTransformer2dModel
                from nunchaku.utils import get_precision

                precision = get_precision()  # auto-detect precision
                logger.info(f"Detected precision: {precision}")

                # Load the Nunchaku transformer on the same device
                transformer_result = NunchakuFluxTransformer2dModel.from_pretrained(
                    f"{NUNCHAKU_MODEL_ID}/svdq-{precision}_r32-flux.1-dev.safetensors"
                )

                # Handle the tuple return: (transformer, config_dict)
                if isinstance(transformer_result, tuple):
                    transformer = transformer_result[0].to(
                        device
                    )  # Extract transformer from tuple
                else:
                    transformer = transformer_result.to(
                        device
                    )  # Direct transformer object

                # Create FluxPipeline with the Nunchaku transformer
                self.pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                ).to(device)

                # Verify device consistency
                logger.debug(
                    f"Device consistency - Target: {device}, Transformer: {next(transformer.parameters()).device}, Pipeline: {self.pipe.device if hasattr(self.pipe, 'device') else 'unknown'}"
                )

                self.model_type = MODEL_TYPE_QUANTIZED_GPU
                logger.info("Nunchaku model loaded successfully with LoRA support!")

            except Exception as nunchaku_error:
                logger.error(
                    f"Error loading Nunchaku model: {nunchaku_error} (Type: {type(nunchaku_error).__name__})"
                )
                raise RuntimeError(
                    f"Failed to load Nunchaku model: {nunchaku_error}. Nunchaku is required for this model."
                )

            self.model_loaded = True
            # Reset LoRA state when loading a new model
            self.current_lora = None
            self.current_weight = 1.0
            logger.info(f"FLUX model loaded successfully on {device}!")
            return True

        except Exception as e:
            logger.error(f"Error loading FLUX model: {e} (Type: {type(e).__name__})")
            return False

    # Removed _integrate_quantized_weights - now using Nunchaku pipeline directly

    def generate_image(self, prompt: str) -> Any:
        """Generate image using the loaded Nunchaku model - GPU only"""
        if not self.model_loaded or self.pipe is None:
            raise RuntimeError("Model not loaded")

        # Ensure we're using GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU required for image generation.")

        if self.gpu_manager.selected_gpu is not None:
            try:
                self.gpu_manager.set_device(self.gpu_manager.selected_gpu)
                logger.info(f"Generating on GPU {self.gpu_manager.selected_gpu}")
            except Exception as gpu_error:
                logger.error(
                    f"GPU error during device selection: {gpu_error} (Type: {type(gpu_error).__name__})"
                )
                raise RuntimeError(
                    f"GPU error: {gpu_error}. GPU required for image generation."
                )
        else:
            logger.error(f"No GPU selected for image generation")
            raise RuntimeError("No GPU selected. GPU required for image generation.")

        # Generate the image using the FluxPipeline (same as the example)
        try:
            logger.info(f"Generating image with prompt: {prompt}")

            # Try with reduced parameters first to avoid memory issues
            try:
                logger.info("Attempting generation with standard parameters...")
                result = self.pipe(
                    prompt,
                    num_inference_steps=25,  # Same as example
                    guidance_scale=3.5,  # Same as example
                )
                logger.info("Image generation completed successfully")
                return result

            except Exception as memory_error:
                if "CUDA" in str(memory_error) or "memory" in str(memory_error).lower():
                    logger.warning(
                        f"CUDA memory error detected: {memory_error} (Type: {type(memory_error).__name__})"
                    )
                    logger.info("Trying with reduced parameters...")

                    # Fallback with reduced parameters
                    result = self.pipe(
                        prompt,
                        num_inference_steps=10,  # Reduced steps
                        guidance_scale=2.0,  # Reduced guidance
                    )
                    logger.info("Image generation completed with reduced parameters")
                    return result
                else:
                    # Re-raise if it's not a memory error
                    logger.error(
                        f"Non-memory error during image generation: {memory_error} (Type: {type(memory_error).__name__})"
                    )
                    raise memory_error

        except Exception as e:
            logger.error(f"Error in image generation: {e} (Type: {type(e).__name__})")
            raise RuntimeError(f"Failed to generate image: {e}")

    def get_model_status(self) -> dict:
        """Get the current model status"""
        return {
            "model_loaded": self.model_loaded,
            "model_type": self.model_type,
            "selected_gpu": self.gpu_manager.selected_gpu,
            "vram_usage_gb": f"{self.gpu_manager.get_vram_usage():.2f}GB",
        }

    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model_loaded

    def get_pipeline(self) -> Optional[FluxPipeline]:
        """Get the loaded FluxPipeline with Nunchaku transformer"""
        return self.pipe

    def apply_lora(self, lora_name: str, weight: float = 1.0) -> bool:
        """Apply LoRA to the FluxPipeline using Nunchaku transformer methods"""
        if not self.model_loaded or self.pipe is None:
            logger.error(
                f"Cannot apply LoRA {lora_name}: Model not loaded or pipeline not available"
            )
            return False

        try:
            logger.info(f"Applying LoRA {lora_name} with weight {weight}")

            # The LoRA methods are on the transformer, not the pipeline
            if hasattr(self.pipe, "transformer") and hasattr(
                self.pipe.transformer, "update_lora_params"
            ):
                # Load LoRA parameters from HuggingFace repository
                logger.info(
                    f"   - Loading LoRA parameters from {lora_name}/lora.safetensors"
                )
                try:
                    self.pipe.transformer.update_lora_params(
                        f"{lora_name}/lora.safetensors"
                    )
                    logger.info(f"   - LoRA parameters loaded successfully")
                except Exception as load_error:
                    logger.error(f"   - Failed to load LoRA parameters: {load_error}")
                    return False

                # Set LoRA strength
                logger.info(f"   - Setting LoRA strength to {weight}")
                try:
                    self.pipe.transformer.set_lora_strength(weight)
                    logger.info(f"   - LoRA strength set successfully")
                except Exception as strength_error:
                    logger.error(f"   - Failed to set LoRA strength: {strength_error}")
                    return False

                self.current_lora = lora_name
                self.current_weight = weight
                logger.info(
                    f"LoRA {lora_name} applied successfully with weight {weight}"
                )
                return True
            else:
                logger.error(
                    f"FluxPipeline transformer does not have LoRA support methods"
                )
                return False

        except Exception as e:
            logger.error(
                f"Error applying LoRA {lora_name}: {e} (Type: {type(e).__name__})"
            )
            return False

    def remove_lora(self) -> bool:
        """Remove currently applied LoRA from the pipeline"""
        if not self.model_loaded or self.pipe is None:
            logger.error(
                f"Cannot remove LoRA: Model not loaded or pipeline not available"
            )
            return False

        try:
            if not self.current_lora:
                logger.info("No LoRA currently applied")
                return True

            logger.info(f"Removing LoRA: {self.current_lora}")

            # Use Nunchaku transformer's built-in method to remove LoRA
            if hasattr(self.pipe, "transformer") and hasattr(
                self.pipe.transformer, "set_lora_strength"
            ):
                logger.info(f"   - Setting LoRA strength to 0 to disable")
                self.pipe.transformer.set_lora_strength(
                    0
                )  # Set strength to 0 to disable
            else:
                logger.warning(f"Transformer does not have set_lora_strength method")

            self.current_lora = None
            self.current_weight = 1.0

            logger.info(f"LoRA removed successfully")
            return True
        except Exception as e:
            logger.error(f"Error removing LoRA: {e} (Type: {type(e).__name__})")
            return False

    def get_lora_info(self) -> Optional[dict]:
        """Get information about the currently applied LoRA"""
        if not self.current_lora:
            return None

        return {
            "name": self.current_lora,
            "weight": self.current_weight,
            "status": "applied",
        }
