"""
LoRA Fusion Utilities for Production Deployment
Handles LoRA configuration parsing and validation for startup fusion
"""

import json
import os
from typing import Dict, List, Optional, Union

from loguru import logger


class LoRAFusionConfig:
    """Configuration class for LoRA fusion at startup"""

    def __init__(self):
        self.fusion_mode = False
        self.lora_name = ""
        self.lora_weight = 1.0
        self.loras_config = []
        self.validated = False

    def parse_from_env(self) -> bool:
        """Parse LoRA configuration from environment variables"""
        try:
            # Parse fusion mode
            self.fusion_mode = os.getenv("FUSION_MODE", "false").lower() == "true"

            if not self.fusion_mode:
                logger.info("LoRA fusion mode disabled")
                return True

            # Parse single LoRA configuration
            self.lora_name = os.getenv("LORA_NAME", "").strip()
            self.lora_weight = float(os.getenv("LORA_WEIGHT", "1.0"))

            # Parse multiple LoRAs configuration
            loras_config_str = os.getenv("LORAS_CONFIG", "").strip()
            if loras_config_str:
                try:
                    self.loras_config = json.loads(loras_config_str)
                    if not isinstance(self.loras_config, list):
                        raise ValueError("LORAS_CONFIG must be a JSON array")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid LORAS_CONFIG format: {e}")
                    return False

            # Validate configuration
            return self._validate_config()

        except Exception as e:
            logger.error(f"Error parsing LoRA configuration: {e}")
            return False

    def _validate_config(self) -> bool:
        """Validate LoRA configuration"""
        if not self.fusion_mode:
            return True

        # Check for conflicting configurations
        if self.lora_name and self.loras_config:
            logger.error("Cannot specify both LORA_NAME and LORAS_CONFIG")
            return False

        if not self.lora_name and not self.loras_config:
            logger.error("Fusion mode requires either LORA_NAME or LORAS_CONFIG")
            return False

        # Validate single LoRA
        if self.lora_name:
            if not self.lora_name.strip():
                logger.error("LORA_NAME cannot be empty")
                return False

            if not (0.0 <= self.lora_weight <= 2.0):
                logger.error("LORA_WEIGHT must be between 0.0 and 2.0")
                return False

        # Validate multiple LoRAs
        if self.loras_config:
            if not isinstance(self.loras_config, list):
                logger.error("LORAS_CONFIG must be a list")
                return False

            for i, lora_config in enumerate(self.loras_config):
                if not isinstance(lora_config, dict):
                    logger.error(f"LoRA config {i} must be a dictionary")
                    return False

                if "name" not in lora_config:
                    logger.error(f"LoRA config {i} missing 'name' field")
                    return False

                if "weight" not in lora_config:
                    logger.error(f"LoRA config {i} missing 'weight' field")
                    return False

                if not lora_config["name"].strip():
                    logger.error(f"LoRA config {i} has empty name")
                    return False

                weight = lora_config["weight"]
                if not (0.0 <= weight <= 2.0):
                    logger.error(f"LoRA config {i} weight must be between 0.0 and 2.0")
                    return False

        self.validated = True
        logger.info("LoRA configuration validated successfully")
        return True

    def get_lora_configs(self) -> List[Dict[str, Union[str, float]]]:
        """Get LoRA configurations in the format expected by the model manager"""
        if not self.fusion_mode or not self.validated:
            return []

        if self.lora_name:
            return [{"name": self.lora_name, "weight": self.lora_weight}]

        return self.loras_config

    def is_fusion_mode_enabled(self) -> bool:
        """Check if fusion mode is enabled"""
        return self.fusion_mode and self.validated

    def get_config_summary(self) -> Dict[str, Union[str, bool, List]]:
        """Get a summary of the current configuration"""
        return {
            "fusion_mode": self.fusion_mode,
            "lora_name": self.lora_name,
            "lora_weight": self.lora_weight,
            "loras_config": self.loras_config,
            "validated": self.validated,
        }


def prepare_lora_for_fusion(lora_configs: List[Dict[str, Union[str, float]]]) -> bool:
    """Prepare LoRA files for fusion (download, validate, etc.)"""
    try:
        for lora_config in lora_configs:
            lora_name = lora_config["name"]

            # Check if it's a local file
            if os.path.exists(lora_name):
                logger.info(f"Found local LoRA file: {lora_name}")
                continue

            # Check if it's an uploaded file
            if lora_name.startswith("uploaded_lora_"):
                upload_path = f"uploads/lora_files/{lora_name}"
                if os.path.exists(upload_path):
                    logger.info(f"Found uploaded LoRA file: {upload_path}")
                    continue
                else:
                    logger.error(f"Uploaded LoRA file not found: {upload_path}")
                    return False

            # Check if it's a Hugging Face repository
            if "/" in lora_name and not os.path.exists(lora_name):
                logger.info(f"Hugging Face LoRA repository: {lora_name}")
                # The model manager will handle HF downloads
                continue

            # Unknown format
            logger.error(f"Unknown LoRA format: {lora_name}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error preparing LoRA files: {e}")
        return False


def apply_startup_lora(model_manager, lora_config: "LoRAFusionConfig") -> bool:
    """Apply LoRA fusion during model startup"""
    try:
        if not lora_config.is_fusion_mode_enabled():
            logger.info("LoRA fusion mode not enabled")
            return True

        lora_configs = lora_config.get_lora_configs()
        if not lora_configs:
            logger.info("No LoRAs to apply")
            return True

        # Prepare LoRA files
        if not prepare_lora_for_fusion(lora_configs):
            logger.error("Failed to prepare LoRA files")
            return False

        # Apply LoRAs
        logger.info(f"Applying {len(lora_configs)} LoRAs in fusion mode")
        success = model_manager.apply_multiple_loras(lora_configs)

        if success:
            # Enable fusion mode to prevent runtime changes
            model_manager.set_fusion_mode(True)
            logger.info("LoRA fusion completed successfully")
        else:
            logger.error("Failed to apply LoRAs in fusion mode")

        return success

    except Exception as e:
        logger.error(f"Error applying startup LoRA: {e}")
        return False
