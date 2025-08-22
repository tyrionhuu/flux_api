"""
ESRGAN Upscaler Module for FLUX API Pipeline
Based on the original ESRGAN implementation with state dict key mapping
"""

import os
import os.path as osp
import cv2
import numpy as np
import torch
import logging
from collections import OrderedDict
from typing import Optional, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

try:
    import ESRGAN.RRDBNet_arch as arch
    ESRGAN_AVAILABLE = True
except ImportError:
    logger.warning("ESRGAN not available.")
    ESRGAN_AVAILABLE = False


class ESRGANUpscaler:
    """ESRGAN upscaler for image enhancement"""
    
    def __init__(self, model_path: str = '/data/weights/ESRGAN/foolhardy_Remacri.pth', 
                 device: str = 'cuda', scale_factor: int = 4):
        """
        Initialize ESRGAN upscaler
        
        Args:
            model_path: Path to ESRGAN model weights
            device: Device to run on ('cuda' or 'cpu')
            scale_factor: Upscaling factor (default 4x)
        """
        if not ESRGAN_AVAILABLE:
            raise ImportError("ESRGAN not available.")
            
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scale_factor = scale_factor
        self.model = None
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ESRGAN model not found: {model_path}")
            
        self._load_model()
    
    def _validate_model_loaded(self) -> bool:
        """
        Validate that the model has essential layers loaded
        
        Returns:
            True if model appears to be properly loaded
        """
        if self.model is None:
            return False
            
        # Check if essential layers exist
        essential_layers = ['conv_first', 'conv_last']
        missing_essential = []
        
        for layer_name in essential_layers:
            if not hasattr(self.model, layer_name):
                missing_essential.append(layer_name)
        
        if missing_essential:
            logger.error(f"Missing essential layers: {missing_essential}")
            return False
            
        # Try a simple forward pass with dummy data
        try:
            dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def _load_model(self):
        """Load and initialize the ESRGAN model"""
        try:
            logger.info(f"Loading ESRGAN model from: {self.model_path}")
            
            # Initialize model architecture
            self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            # Map legacy keys to new architecture
            state_dict = self._map_esrgan_state_dict_keys(state_dict)
            
            # Load state dict with strict=False to handle mismatches
            load_info = self.model.load_state_dict(state_dict, strict=False)
            
            # Handle loading mismatches
            if hasattr(load_info, 'missing_keys') and hasattr(load_info, 'unexpected_keys'):
                missing_keys = load_info.missing_keys
                unexpected_keys = load_info.unexpected_keys
                if missing_keys or unexpected_keys:
                    logger.warning('ESRGAN model loading mismatches:')
                    if missing_keys:
                        logger.warning(f'  Missing keys: {missing_keys}')
                    if unexpected_keys:
                        logger.warning(f'  Unexpected keys: {unexpected_keys}')
                    
                    # Check if critical layers are loaded
                    if len(missing_keys) > 0:
                        logger.info(f"Model loaded with {len(missing_keys)} missing keys, but continuing...")
            else:
                # PyTorch < 1.6 compatibility
                missing_keys, unexpected_keys = load_info
                if missing_keys or unexpected_keys:
                    logger.warning('ESRGAN model loading mismatches:')
                    if missing_keys:
                        logger.warning(f'  Missing keys: {missing_keys}')
                    if unexpected_keys:
                        logger.warning(f'  Unexpected keys: {unexpected_keys}')
                    
                    # Check if critical layers are loaded
                    if len(missing_keys) > 0:
                        logger.info(f"Model loaded with {len(missing_keys)} missing keys, but continuing...")
            
            # Set to evaluation mode and move to device
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Validate the loaded model
            if not self._validate_model_loaded():
                logger.warning("Model loaded but validation failed. Model may not work correctly.")
            
            logger.info(f"ESRGAN model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load ESRGAN model: {e}")
            raise
    
    def _map_esrgan_state_dict_keys(self, state_dict: dict) -> OrderedDict:
        """
        Map legacy ESRGAN keys to RRDBNet keys
        
        Args:
            state_dict: Original state dict with legacy keys
            
        Returns:
            Mapped state dict with new keys
        """
        new_state_dict = OrderedDict()
        
        # Create a mapping from expected keys to actual keys
        key_mapping = {}
        
        for key, tensor in state_dict.items():
            original_key = key
            
            # Strip DistributedDataParallel prefix if present
            if key.startswith('module.'):
                key = key[len('module.'):]

            if key.startswith('model.'):
                parts = key.split('.')
                # parts[0] == 'model'
                stage = parts[1]
                
                if stage == '0':
                    # model.0.* -> conv_first.*
                    key = key.replace('model.0', 'conv_first')
                elif stage == '1' and len(parts) > 3 and parts[2] == 'sub':
                    # model.1.sub.{i}.RDBk.convj.(0.)?{weight|bias} OR model.1.sub.23.{weight|bias}
                    block_idx = parts[3]
                    # Special mapping: model.1.sub.23.{weight,bias} -> trunk_conv.{weight,bias}
                    if block_idx == '23' and len(parts) == 5 and parts[4] in ('weight', 'bias'):
                        key = f'trunk_conv.{parts[4]}'
                    else:
                        rest = '.'.join(parts[4:])
                        # Drop occasional sequential '.0.' in conv layers
                        rest = rest.replace('.0.', '.')
                        rest = rest.replace('.0.weight', '.weight').replace('.0.bias', '.bias')
                        key = f'RRDB_trunk.{block_idx}.{rest}'
                elif stage == '23':
                    key = key.replace('model.23', 'trunk_conv')
                elif stage == '3':
                    key = key.replace('model.3', 'upconv1')
                elif stage == '6':
                    key = key.replace('model.6', 'upconv2')
                elif stage == '8':
                    key = key.replace('model.8', 'HRconv')
                elif stage == '10':
                    key = key.replace('model.10', 'conv_last')
                
                # Clean any leftover '.0.' patterns
                key = key.replace('.0.', '.')
                key = key.replace('.0.weight', '.weight').replace('.0.bias', '.bias')

            # Handle RRDB_trunk keys that need index mapping
            if key.startswith('RRDB_trunk.'):
                parts = key.split('.')
                if len(parts) >= 3 and parts[1].isdigit():
                    # RRDB_trunk.{index}.RDB{k}.conv{j}.{weight|bias}
                    # Map to RRDB_trunk.RDB{k}.conv{j}.{weight|bias}
                    if len(parts) >= 4 and parts[2].startswith('RDB'):
                        new_key = f"RRDB_trunk.{'.'.join(parts[2:])}"
                        key = new_key

            new_state_dict[key] = tensor
            
        return new_state_dict
    
    def upscale_image(self, image: Union[np.ndarray, str, Path], 
                     output_path: Optional[str] = None) -> np.ndarray:
        """
        Upscale an image using ESRGAN
        
        Args:
            image: Input image as numpy array, file path, or Path object
            output_path: Optional path to save the upscaled image
            
        Returns:
            Upscaled image as numpy array
        """
        if self.model is None:
            raise RuntimeError("ESRGAN model not loaded")
        
        # Validate model before inference
        if not self._validate_model_loaded():
            raise RuntimeError("ESRGAN model validation failed. Model may not work correctly.")
        
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                if not str(image).strip():  # Check for empty path
                    raise ValueError("Image path cannot be empty")
                img = cv2.imread(str(image), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to load image: {image}")
            else:
                img = image.copy()
            
            # Get original dimensions
            h0, w0 = img.shape[:2]
            logger.info(f"Input image size: {w0}x{h0}")
            
            # Preprocess image
            img = img * 1.0 / 255  # Normalize to [0, 1]
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_input = img.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(img_input).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            
            # Postprocess output
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            
            # Get output dimensions
            h1, w1 = output.shape[:2]
            logger.info(f"Upscaled image size: {w1}x{h1}")
            
            # Save if output path provided
            if output_path:
                if not str(output_path).strip():  # Check for empty path
                    raise ValueError("Output path cannot be empty")
                
                # Handle case where output_path is just a filename (no directory)
                output_dir = os.path.dirname(output_path)
                if output_dir:  # Only create directory if there is one
                    os.makedirs(output_dir, exist_ok=True)
                    
                cv2.imwrite(output_path, output)
                logger.info(f"Upscaled image saved to: {output_path}")
            
            return output
            
        except Exception as e:
            logger.error(f"Upscaling failed: {e}")
            raise
    
    def upscale_batch(self, image_paths: list, output_dir: str) -> list:
        """
        Upscale multiple images in batch
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save upscaled images
            
        Returns:
            List of output paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {path}")
                
                # Generate output filename
                base_name = osp.splitext(osp.basename(path))[0]
                output_path = osp.join(output_dir, f'{base_name}_upscaled.png')
                
                # Upscale image
                self.upscale_image(path, output_path)
                output_paths.append(output_path)
                
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                continue
        
        return output_paths
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "scale_factor": self.scale_factor,
            "model_loaded": self.model is not None,
            "available": ESRGAN_AVAILABLE
        }


# Convenience function for quick upscaling
def quick_upscale(image_path: str, output_path: str, 
                  model_path: str = '/data/weights/ESRGAN/foolhardy_Remacri.pth',
                  device: str = 'cuda') -> np.ndarray:
    """
    Quick upscale function for single images
    
    Args:
        image_path: Input image path
        output_path: Output image path
        model_path: ESRGAN model path
        device: Device to use
        
    Returns:
        Upscaled image array
    """
    upscaler = ESRGANUpscaler(model_path, device)
    return upscaler.upscale_image(image_path, output_path)


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize upscaler
        upscaler = ESRGANUpscaler()
        
        # Test with a sample image
        test_image = "test.png"
        if os.path.exists(test_image) and os.path.getsize(test_image) > 0:
            result = upscaler.upscale_image(test_image, "test_image_upscaled.png")
            print(f"Upscaling completed. Result shape: {result.shape}")
        else:
            print(f"Test image {test_image} not found or empty. Skipping test.")
            print("You can test the upscaler by providing a valid image path.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
