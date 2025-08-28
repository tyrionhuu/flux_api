"""
FLUX model management for the FLUX API
"""

import logging
import os
import shutil
import tempfile
from typing import Any, Optional, Union

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from config.fp4_settings import (MODEL_TYPE_QUANTIZED_GPU, NUNCHAKU_MODEL_ID)
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
        # LoRA state - can be single LoRA (str) or multiple LoRAs (list)
        self.current_lora: Optional[Union[str, list]] = None
        self.current_weight: float = 1.0
        # Track temporary LoRA files for cleanup
        self._temp_lora_paths: list = []

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self._cleanup_temp_loras()
        except:
            pass  # Ignore errors during cleanup

    def load_model(self) -> bool:
        """Load the FLUX model with GPU-only support and quantization"""
        try:
            # Safety check: avoid reloading if already loaded
            if self.model_loaded and self.pipe is not None:
                logger.info("FLUX model already loaded, skipping reload")
                return True

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

            # No default LoRA - users must explicitly specify LoRAs if they want them

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
                        "warmup prompt", num_inference_steps=10, guidance_scale=0
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
        num_inference_steps: int = 10,
        guidance_scale: float = 0,
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
                    "width": width,
                    "height": height,
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
                        "width": width,
                        "height": height,
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
        """Check if the model is loaded and pipeline is available"""
        return self.model_loaded and self.pipe is not None

    def get_pipeline(self) -> Optional[FluxPipeline]:
        """Get the loaded FluxPipeline with Nunchaku transformer"""
        return self.pipe

    def _is_ready_with_lora(self) -> bool:
        """Quick readiness check for LoRA-capable pipeline."""
        if not self.model_loaded or self.pipe is None:
            logger.error(
                "Cannot apply LoRAs: Model not loaded or pipeline not available"
            )
            return False
        if not (
            hasattr(self.pipe, "transformer")
            and hasattr(self.pipe.transformer, "update_lora_params")
        ):
            logger.error("FluxPipeline transformer does not have LoRA support methods")
            return False
        return True

    def _get_pipe_transformer(self):
        """Get the transformer from the pipeline with type safety."""
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")
        if not hasattr(self.pipe, "transformer"):
            raise RuntimeError("Pipeline does not have transformer")
        return self.pipe.transformer

    def _parse_hf_input(self, lora_input: str) -> tuple[str, str]:
        """Parse Hugging Face input to extract repo_id and filename.

        Args:
            lora_input: Can be:
                - Full URL: https://huggingface.co/username/repo/blob/main/filename.safetensors
                - Repo path: username/repo
                - Repo path with file: username/repo/filename.safetensors

        Returns:
            Tuple of (repo_id, filename) where filename may be empty string if not specified
        """
        # Remove protocol and domain if present
        if lora_input.startswith("https://huggingface.co/"):
            lora_input = lora_input[23:]  # Remove "https://huggingface.co/"
        elif lora_input.startswith("http://huggingface.co/"):
            lora_input = lora_input[22:]  # Remove "http://huggingface.co/"

        # Remove /blob/main/ if present
        if "/blob/main/" in lora_input:
            lora_input = lora_input.replace("/blob/main/", "/")

        # Split by '/' to get components
        parts = lora_input.split("/")

        if len(parts) >= 3:
            # Format: username/repo/filename
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = parts[2]
        elif len(parts) == 2:
            # Format: username/repo
            repo_id = lora_input
            filename = ""
        else:
            # Single part or invalid format
            repo_id = lora_input
            filename = ""

        return repo_id, filename

    def _check_lora_compatibility(self, lora_path: str) -> bool:
        """Check if a LoRA file is compatible with FLUX models"""
        try:
            # Check if file exists
            if not os.path.exists(lora_path):
                logger.error(f"   - LoRA file not found: {lora_path}")
                return False

            # Check file size (basic validation)
            file_size = os.path.getsize(lora_path)
            if file_size < 1024:  # Less than 1KB
                logger.error(
                    f"   - LoRA file too small ({file_size} bytes), likely corrupted"
                )
                return False

            # Check file extension
            if not lora_path.endswith((".safetensors", ".bin", ".pt", ".pth")):
                logger.error(
                    f"   - Unsupported file format: {os.path.splitext(lora_path)[1]}"
                )
                return False

            # Basic compatibility check - try to load the file
            try:
                if lora_path.endswith(".safetensors"):
                    from safetensors import safe_open

                    with safe_open(lora_path, framework="pt") as f:
                        lora_weights = {key: f.get_tensor(key) for key in f.keys()}
                else:
                    lora_weights = torch.load(
                        lora_path, map_location="cpu", weights_only=True
                    )

                # Check if it's a dictionary and not empty
                if not isinstance(lora_weights, dict) or not lora_weights:
                    logger.error(
                        f"   - LoRA file does not contain valid weight dictionary"
                    )
                    return False

                # Check for FLUX-compatible LoRA keys
                has_compatible_keys = False
                expected_patterns = ["lora", "adapter", "model", "weights"]
                for key in lora_weights.keys():
                    key_lower = key.lower()
                    for pattern in expected_patterns:
                        if pattern in key_lower:
                            has_compatible_keys = True
                            break
                    if has_compatible_keys:
                        break

                if not has_compatible_keys:
                    logger.error(
                        f"   - LoRA file does not appear to be compatible with FLUX models"
                    )
                    return False

                logger.info(f"   - LoRA compatibility check passed")
                return True

            except Exception as load_error:
                logger.error(
                    f"   - Failed to load LoRA file for compatibility check: {str(load_error)}"
                )
                return False

        except Exception as e:
            logger.error(f"   - LoRA compatibility check failed: {str(e)}")
            return False

    def _apply_lora_to_transformer(self, lora_source: str, weight: float) -> bool:
        """Apply a LoRA (local path or repo id) and set its strength."""
        try:
            if not self._is_ready_with_lora():
                return False

            # Get the transformer safely
            transformer = self._get_pipe_transformer()

            # Get the actual file path (this handles uploaded files, local files, and HF downloads)
            lora_path = self._get_lora_path(lora_source)
            if not lora_path:
                logger.error(f"   - Could not resolve path for LoRA: {lora_source}")
                return False

            # Check LoRA compatibility before applying
            if not self._check_lora_compatibility(lora_path):
                logger.error(f"   - LoRA compatibility check failed: {lora_source}")
                return False

            logger.info(f"   - Loading LoRA parameters from: {lora_path}")
            transformer.update_lora_params(lora_path)
            logger.info(f"   - Setting LoRA strength to {weight}")
            transformer.set_lora_strength(weight)
            logger.info("   - LoRA applied successfully")
            return True
        except Exception as e:
            logger.error(f"   - Failed to apply LoRA: {e}")
            return False

    def apply_lora(self, lora_name: str, lora_weight: float = 1.0) -> bool:
        """Apply a single LoRA to the pipeline - for backward compatibility"""
        return self.apply_multiple_loras([{"name": lora_name, "weight": lora_weight}])

    def apply_multiple_loras(self, lora_configs: list) -> bool:
        """Apply multiple LoRAs simultaneously to the pipeline by combining them"""
        try:
            if not self._is_ready_with_lora():
                return False

            if not lora_configs:
                logger.info("No LoRAs to apply")
                return True

            logger.info(f"Applying {len(lora_configs)} LoRAs to FLUX pipeline...")

            if len(lora_configs) == 1:
                # Single LoRA - apply directly
                lora_config = lora_configs[0]
                return self._apply_lora_to_transformer(
                    lora_config["name"], lora_config["weight"]
                )

            else:
                # Multiple LoRAs - merge them into a single LoRA
                logger.info(f"   - Merging {len(lora_configs)} LoRAs...")

                # Try to merge the LoRAs
                merged_lora_path = self._merge_loras(lora_configs)

                if merged_lora_path:
                    # Calculate combined weight (sum of all weights)
                    combined_weight = sum(lora["weight"] for lora in lora_configs)

                    # Apply the merged LoRA
                    success = self._apply_lora_to_transformer(
                        merged_lora_path, combined_weight
                    )

                    if success:
                        # Store info about all LoRAs for reference
                        self.current_lora = lora_configs
                        self.current_weight = combined_weight

                        # Track the temporary file for cleanup
                        if not hasattr(self, "_temp_lora_paths"):
                            self._temp_lora_paths = []
                        self._temp_lora_paths.append(merged_lora_path)

                        logger.info(
                            f"   - Merged LoRA applied successfully (weight: {combined_weight})"
                        )
                        return True
                    else:
                        # Clean up on failure
                        if os.path.exists(merged_lora_path):
                            temp_dir = os.path.dirname(merged_lora_path)
                            if os.path.exists(temp_dir):
                                shutil.rmtree(temp_dir)
                        return False
                else:
                    # Fallback: use the first LoRA with combined weight
                    logger.warning(f"   - LoRA merging failed, using fallback approach")

                    primary_lora = lora_configs[0]
                    combined_weight = sum(lora["weight"] for lora in lora_configs)

                    # Apply the primary LoRA with combined weight
                    success = self._apply_lora_to_transformer(
                        primary_lora["name"], combined_weight
                    )

                    if success:
                        # Store info about all LoRAs for reference
                        self.current_lora = lora_configs
                        self.current_weight = combined_weight
                        logger.info(
                            f"   - Fallback LoRA applied (weight: {combined_weight})"
                        )
                        return True
                    else:
                        return False

        except Exception as e:
            logger.error(
                f"Error applying multiple LoRAs: {e} (Type: {type(e).__name__})"
            )
            return False

    def _apply_single_lora(self, lora_name: str, weight: float) -> bool:
        """Internal method to apply a single LoRA - extracted for reuse"""
        # Backward-compatibility wrapper
        return self._apply_lora_to_transformer(lora_name, weight)

    def remove_lora(self) -> bool:
        """Remove currently applied LoRA(s) from the pipeline"""
        if not self.model_loaded or self.pipe is None:
            logger.error(
                f"Cannot remove LoRA: Model not loaded or pipeline not available"
            )
            return False

        try:
            if not self.current_lora:
                logger.info("No LoRA currently applied")
                return True

            logger.info(f"Removing LoRA(s)...")

            # Use Nunchaku transformer's built-in method to remove LoRA
            try:
                transformer = self._get_pipe_transformer()
                transformer.set_lora_strength(0)  # Set strength to 0 to disable
            except RuntimeError:
                logger.warning("Pipeline transformer not available")
                return False

            # Clean up temporary LoRA files
            self._cleanup_temp_loras()

            # Reset LoRA state
            self.current_lora = None
            self.current_weight = 1.0

            logger.info(f"LoRA(s) removed successfully")
            return True
        except Exception as e:
            logger.error(f"Error removing LoRA: {e}")
            return False

    def get_lora_info(self) -> Optional[dict]:
        """Get information about the currently applied LoRA"""
        if not self.current_lora:
            return None

        if isinstance(self.current_lora, str):
            return {
                "name": self.current_lora,
                "weight": self.current_weight,
                "status": "applied",
            }
        elif isinstance(self.current_lora, list):
            return {
                "name": [lora["name"] for lora in self.current_lora],
                "weight": self.current_weight,
                "status": "applied",
            }
        return None

    def _merge_loras(self, lora_configs: list) -> Optional[str]:
        """Merge multiple LoRAs into a single LoRA file by combining their weights"""
        try:
            if len(lora_configs) <= 1:
                return None

            logger.info(f"Merging {len(lora_configs)} LoRAs into a single LoRA...")

            # Create a temporary directory for the merged LoRA
            temp_dir = tempfile.mkdtemp(prefix="merged_lora_")
            merged_lora_path = os.path.join(temp_dir, "merged_lora.safetensors")

            try:
                # Load all LoRAs
                lora_data_list = []
                lora_weights_list = []

                for i, lora_config in enumerate(lora_configs):
                    lora_name = lora_config["name"]
                    weight = lora_config["weight"]

                    logger.info(
                        f"   - Loading LoRA {i+1}: {lora_name} (weight: {weight})"
                    )

                    lora_path = self._get_lora_path(lora_name)
                    if not lora_path:
                        logger.error(
                            f"   - Could not resolve path for LoRA: {lora_name}"
                        )
                        return None

                    # Load the LoRA weights
                    try:
                        if lora_path.endswith(".safetensors"):
                            lora_data = safe_load_file(lora_path)
                        else:
                            lora_data = torch.load(
                                lora_path, map_location="cpu", weights_only=False
                            )
                        lora_data_list.append(lora_data)
                        lora_weights_list.append(weight)
                        logger.info(f"   - LoRA {i+1} loaded successfully")
                    except Exception as load_error:
                        logger.error(f"   - Failed to load LoRA {i+1}: {load_error}")
                        return None

                # Merge the LoRA weights
                # Build union of all keys across LoRAs
                all_keys = set()
                for data_dict in lora_data_list:
                    all_keys.update(data_dict.keys())

                logger.info(f"   - Merging {len(all_keys)} tensor keys...")

                merged_lora = {}
                for key in all_keys:
                    # Find first LoRA that has this key to get base tensor/shape
                    first_idx = next(
                        (i for i, d in enumerate(lora_data_list) if key in d), None
                    )
                    if first_idx is None:
                        continue
                    base_tensor = lora_data_list[first_idx][key]
                    merged_tensor = base_tensor * lora_weights_list[first_idx]

                    for i in range(len(lora_data_list)):
                        if i == first_idx:
                            continue
                        if key in lora_data_list[i]:
                            merged_tensor = (
                                merged_tensor
                                + lora_data_list[i][key] * lora_weights_list[i]
                            )

                    merged_lora[key] = merged_tensor

                # Save the merged LoRA
                safe_save_file(merged_lora, merged_lora_path)
                logger.info(f"   - Merged LoRA saved: {merged_lora_path}")

                return merged_lora_path

            except Exception as merge_error:
                logger.error(f"Error during LoRA merging: {merge_error}")
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return None

        except Exception as e:
            logger.error(f"Error in LoRA merging: {e}")
            return None

    def _get_lora_path(self, lora_name: str) -> Optional[str]:
        """Get the full path to a LoRA file, downloading from HF if needed"""
        try:
            # Check if it's an uploaded file
            if lora_name.startswith("uploaded_lora_"):
                upload_path = f"uploads/lora_files/{lora_name}"
                if os.path.exists(upload_path):
                    logger.info(f"   - Found uploaded LoRA file: {upload_path}")
                    return upload_path
                else:
                    logger.error(f"   - Uploaded LoRA file not found: {upload_path}")
                    return None

            # Check if it's a local path
            if os.path.exists(lora_name):
                return lora_name

            # Check if it's a relative path
            if lora_name.startswith("./") or lora_name.startswith("../"):
                abs_path = os.path.abspath(lora_name)
                if os.path.exists(abs_path):
                    return abs_path

            # Check if it's a Hugging Face URL or repo - try to download it
            if "/" in lora_name and not os.path.exists(lora_name):
                # Parse the input to extract repo_id and filename
                repo_id, filename = self._parse_hf_input(lora_name)
                logger.info(
                    f"   - Parsed input: repo_id='{repo_id}', filename='{filename}'"
                )

                try:
                    # Create a temporary directory for the downloaded LoRA
                    temp_dir = tempfile.mkdtemp(prefix="hf_lora_")

                    # Use huggingface_hub to download the LoRA
                    from huggingface_hub import (hf_hub_download,
                                                 list_repo_files)

                    logger.info(f"   - Downloading to temp dir: {temp_dir}")

                    # Initialize downloaded_path variable
                    downloaded_path = None

                    # If we have a specific filename, try to download it directly
                    if filename:
                        try:
                            downloaded_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=filename,
                                cache_dir=temp_dir,
                            )
                            logger.info(f"   - Downloaded: {filename}")
                        except Exception as specific_error:
                            logger.warning(
                                f"   - Failed to download {filename}, falling back to auto-detection"
                            )
                            filename = None

                    # If no specific filename or it failed, auto-detect the file
                    if not filename or downloaded_path is None:
                        try:
                            repo_files = list_repo_files(repo_id)

                            # Look for common LoRA file patterns
                            lora_filename = None
                            for file in repo_files:
                                if (
                                    file.endswith(".safetensors")
                                    or file.endswith(".bin")
                                ) and any(
                                    pattern in file.lower()
                                    for pattern in [
                                        "lora",
                                        "adapter",
                                        "model",
                                        "weights",
                                        "super-realism",
                                    ]
                                ):
                                    lora_filename = file
                                    break

                            # If no specific LoRA file found, try common names
                            if not lora_filename:
                                common_names = [
                                    "lora.safetensors",
                                    "model.safetensors",
                                    "pytorch_lora_weights.safetensors",
                                    "adapter_model.safetensors",
                                ]
                                for name in common_names:
                                    if name in repo_files:
                                        lora_filename = name
                                        break

                            if not lora_filename:
                                logger.error(
                                    f"   - No suitable LoRA file found in repository"
                                )
                                return None

                            # Download the auto-detected file
                            downloaded_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=lora_filename,
                                cache_dir=temp_dir,
                            )
                            logger.info(f"   - Downloaded: {lora_filename}")

                        except Exception as list_error:
                            logger.warning(
                                f"   - Could not list repo files, using fallback"
                            )
                            # Final fallback to common filename
                            try:
                                downloaded_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename="lora.safetensors",
                                    cache_dir=temp_dir,
                                )
                                logger.info(
                                    f"   - Downloaded: lora.safetensors (fallback)"
                                )
                            except Exception as fallback_error:
                                logger.error(
                                    f"   - All download attempts failed: {fallback_error}"
                                )
                                raise fallback_error

                    # Verify the downloaded file
                    if not os.path.exists(downloaded_path) or not os.path.isfile(
                        downloaded_path
                    ):
                        logger.error(f"   - Downloaded file not found or invalid")
                        return None

                    if os.path.getsize(downloaded_path) == 0:
                        logger.error(f"   - Downloaded file is empty")
                        return None

                    lora_path = downloaded_path

                    # Track for cleanup
                    if not hasattr(self, "_temp_lora_paths"):
                        self._temp_lora_paths = []
                    self._temp_lora_paths.append(lora_path)

                    return lora_path

                except ImportError:
                    logger.error(
                        f"   - huggingface_hub not available. Install with: pip install huggingface_hub"
                    )
                    return None
                except Exception as download_error:
                    logger.error(f"   - Failed to download LoRA: {download_error}")
                    # Clean up temp directory on failure
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    return None

            return None

        except Exception as e:
            logger.error(f"Error resolving LoRA path: {e}")
            return None

    def _cleanup_temp_loras(self):
        """Clean up any temporary LoRA files"""
        try:
            if hasattr(self, "_temp_lora_paths"):
                for temp_path in self._temp_lora_paths:
                    if os.path.exists(temp_path):
                        temp_dir = os.path.dirname(temp_path)
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                            logger.info(f"Cleaned up temporary LoRA: {temp_dir}")
                self._temp_lora_paths = []
        except Exception as e:
            logger.warning(f"Error cleaning up temporary LoRAs: {e}")
