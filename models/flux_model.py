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

            # Check for balanced multi-GPU mode
            visible_gpu_count = torch.cuda.device_count()
            logger.info(f"Visible GPUs: {visible_gpu_count}")
            
            device_map = None
            if visible_gpu_count > 1:
                # Balanced multi-GPU mode
                logger.info(f"Using balanced multi-GPU mode across {visible_gpu_count} GPUs")
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
                    logger.warning(f"Device mismatch! Current: {current_device}, Target device: 0")
                    torch.cuda.set_device(0)
                    current_device = torch.cuda.current_device()
                    logger.info(f"Device after force set: {current_device}")

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
                if device_map == "balanced":
                    # Multi-GPU balanced mode
                    logger.info("Loading pipeline with balanced device map")
                    self.pipe = FluxPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        transformer=transformer,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map,
                    )
                else:
                    # Single GPU mode
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

            # Perform CUDA Graph warm-up for better performance
            self._warmup_cuda_graph()

            return True

        except Exception as e:
            logger.error(f"Error loading FLUX model: {e} (Type: {type(e).__name__})")
            return False

    def _warmup_cuda_graph(self):
        """Perform CUDA Graph warm-up for better performance"""
        try:
            if self.pipe is not None and torch.cuda.is_available():
                logger.info("Performing CUDA Graph warm-up...")

                # Warm-up with multiple iterations to optimize CUDA graphs
                for i in range(2):
                    logger.info(f"CUDA Graph warm-up iteration {i+1}/2")
                    _ = self.pipe(
                        "warmup prompt", num_inference_steps=5, guidance_scale=1.0
                    )

                    # Clear CUDA cache between warm-up iterations
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                logger.info("CUDA Graph warm-up completed successfully")
            else:
                logger.warning(
                    "Skipping CUDA Graph warm-up - pipeline not loaded or CUDA not available"
                )
        except Exception as e:
            logger.warning(
                f"CUDA Graph warm-up failed: {e} - continuing without warm-up"
            )

    # Removed _integrate_quantized_weights - now using Nunchaku pipeline directly

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
        """Generate image using the loaded Nunchaku model - GPU only"""
        if not self.model_loaded or self.pipe is None:
            raise RuntimeError("Model not loaded")

        # Ensure we're using GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU required for image generation.")

        # Set device for generation based on mode
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count > 1:
            logger.info(f"Generating with balanced multi-GPU mode ({visible_gpu_count} GPUs)")
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

        # Generate the image using the FluxPipeline (same as the example)
        try:
            logger.info(f"Generating image with prompt: {prompt}")

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
                logger.info("Image generation completed successfully")
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
                        f"Image generation completed with reduced parameters: steps={fallback_steps}, guidance={fallback_guidance}"
                    )
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
