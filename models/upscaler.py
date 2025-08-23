"""
FLUX Framework Upscaler using Remacri ESRGAN model
Provides high-quality 2x and 4x upscaling capabilities
"""

import os
import logging
import torch
import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Union
import time

# Configure logging
logger = logging.getLogger(__name__)

# Import ESRGAN architecture
try:
    import ESRGAN.RRDBNet_arch as arch
    ESRGAN_AVAILABLE = True
except ImportError:
    ESRGAN_AVAILABLE = False
    logger.warning("ESRGAN not available. Upscaling will not work.")


class FLUXUpscaler:
    """High-quality image upscaler using Remacri ESRGAN model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the FLUX upscaler
        
        Args:
            model_path: Path to the Remacri ESRGAN model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Default model path
        if model_path is None:
            model_path = "/data/weights/ESRGAN/foolhardy_Remacri.pth"
        
        self.model_path = model_path
        
        # Check if ESRGAN is available
        if not ESRGAN_AVAILABLE:
            logger.error("ESRGAN not available. Cannot initialize upscaler.")
            return
        
        # Load the model
        if not self._load_model():
            logger.error("Failed to load model during initialization")
            return
    
    def _load_model(self) -> bool:
        """Load the Remacri ESRGAN model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info(f"Loading Remacri ESRGAN model from: {self.model_path}")
            
            # Initialize RRDBNet architecture
            self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
            
            if self.model is None:
                logger.error("Failed to initialize RRDBNet architecture")
                return False
            
            # Load checkpoint and map keys if needed
            checkpoint = torch.load(self.model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            # Map legacy ESRGAN keys to RRDBNet keys
            state_dict = self._map_esrgan_state_dict_keys(state_dict)
            
            # Load state dict
            load_info = self.model.load_state_dict(state_dict, strict=False)
            
            # Check for loading mismatches
            if hasattr(load_info, 'missing_keys') and hasattr(load_info, 'unexpected_keys'):
                missing_keys = load_info.missing_keys
                unexpected_keys = load_info.unexpected_keys
                if missing_keys or unexpected_keys:
                    logger.warning('Model loading mismatches:')
                    if missing_keys:
                        logger.warning(f'  Missing keys: {missing_keys}')
                    if unexpected_keys:
                        logger.warning(f'  Unexpected keys: {unexpected_keys}')
            else:
                # PyTorch < 1.6 returns (missing, unexpected)
                missing_keys, unexpected_keys = load_info
                if missing_keys or unexpected_keys:
                    logger.warning('Model loading mismatches:')
                    if missing_keys:
                        logger.warning(f'  Missing keys: {missing_keys}')
                    if unexpected_keys:
                        logger.warning(f'  Unexpected keys: {unexpected_keys}')
            
            # Set model to evaluation mode and move to device
            self.model.eval()
            self.model = self.model.to(self.device)
            
            self.model_loaded = True
            logger.info("Remacri ESRGAN model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Remacri ESRGAN model: {e}")
            return False
    
    def _map_esrgan_state_dict_keys(self, state_dict: dict) -> OrderedDict:
        """
        Map legacy ESRGAN keys to RRDBNet keys
        
        Examples:
          - model.0.* -> conv_first.*
          - model.1.sub.{i}.RDBk.convj.0.{weight,bias} -> RRDB_trunk.{i}.RDBk.convj.{weight,bias}
          - model.23.* -> trunk_conv.*
          - model.3.*  -> upconv1.*
          - model.6.*  -> upconv2.*
          - model.8.*  -> HRconv.*
          - model.10.* -> conv_last.*
        """
        new_state_dict = OrderedDict()
        
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
            
            new_state_dict[key] = tensor
        
        return new_state_dict
    
    def _high_quality_resize(self, image: np.ndarray, target_width: int, target_height: int, 
                            method: str = 'lanczos') -> np.ndarray:
        """
        Use high-quality resampling methods to resize images
        
        Args:
            image: Image as numpy array (H, W, C)
            target_width: Target width
            target_height: Target height
            method: Resampling method ('lanczos', 'bicubic', 'hamming')
        
        Returns:
            Resized image
        """
        # Convert numpy array to PIL image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Select resampling method
        if method == 'lanczos':
            resample = Image.Resampling.LANCZOS
        elif method == 'bicubic':
            resample = Image.Resampling.BICUBIC
        elif method == 'hamming':
            resample = Image.Resampling.HAMMING
        else:
            resample = Image.Resampling.LANCZOS  # Default to Lanczos
        
        # Resize image
        resized_image = pil_image.resize((target_width, target_height), resample)
        
        # Convert back to numpy array
        return np.array(resized_image)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for ESRGAN model"""
        # Normalize to [0, 1] range
        img = image.astype(np.float32) / 255.0
        
        # Convert BGR to RGB and transpose to (C, H, W)
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def _postprocess_image(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess ESRGAN model output"""
        # Remove batch dimension and clamp to [0, 1]
        output_np = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        # Transpose back to (H, W, C) and convert BGR to RGB
        output_np = np.transpose(output_np[[2, 1, 0], :, :], (1, 2, 0))
        
        # Convert to [0, 255] range
        output_np = (output_np * 255.0).round().astype(np.uint8)
        
        return output_np
    
    def upscale_image(self, image: np.ndarray, scale_factor: int = 2, 
                     method: str = 'lanczos') -> np.ndarray:
        """
        Upscale image using Remacri ESRGAN model
        
        Args:
            image: Input image as numpy array (BGR format)
            scale_factor: Upscaling factor (2 or 4)
            method: Downsampling method for 2x output when using 4x model
        
        Returns:
            Upscaled image
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Cannot perform upscaling.")
        
        if scale_factor not in [2, 4]:
            raise ValueError("Scale factor must be 2 or 4")
        
        try:
            start_time = time.time()
            logger.info(f"Starting {scale_factor}x upscaling...")
            
            # Get original dimensions
            h0, w0 = image.shape[:2]
            
            # Preprocess image
            img_tensor = self._preprocess_image(image)
            
            # Perform upscaling
            if self.model is None:
                raise RuntimeError("Model not loaded. Cannot perform upscaling.")
                
            with torch.no_grad():
                output = self.model(img_tensor)
            
            # Postprocess output
            output_4x = self._postprocess_image(output)
            
            if scale_factor == 2:
                # Downsample 4x result to 2x using high-quality resampling
                target_width = w0 * 2
                target_height = h0 * 2
                
                output_2x = self._high_quality_resize(output_4x, target_width, target_height, method)
                
                upscaling_time = time.time() - start_time
                logger.info(f"2x upscaling completed in {upscaling_time:.2f}s")
                
                return output_2x
            else:
                # Return 4x result
                upscaling_time = time.time() - start_time
                logger.info(f"4x upscaling completed in {upscaling_time:.2f}s")
                
                return output_4x
                
        except Exception as e:
            logger.error(f"Upscaling failed: {e}")
            raise RuntimeError(f"Upscaling failed: {e}")
    
    def upscale_file(self, input_path: str, output_path: str, scale_factor: int = 2,
                    method: str = 'lanczos') -> bool:
        """
        Upscale image file and save result
        
        Args:
            input_path: Path to input image
            output_path: Path to save upscaled image
            scale_factor: Upscaling factor (2 or 4)
            method: Downsampling method for 2x output
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read input image
            if not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False
            
            logger.info(f"Reading image from: {input_path}")
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error(f"Failed to read image: {input_path}")
                return False
            
            # Perform upscaling
            upscaled_image = self.upscale_image(image, scale_factor, method)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save upscaled image
            success = cv2.imwrite(output_path, upscaled_image)
            
            if success is not None and success:
                logger.info(f"Upscaled image saved to: {output_path}")
                return True
            else:
                logger.error(f"Failed to save upscaled image: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"File upscaling failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "device": str(self.device),
            "esrgan_available": ESRGAN_AVAILABLE,
            "model_type": "Remacri ESRGAN"
        }
    
    def is_ready(self) -> bool:
        """Check if the upscaler is ready to use"""
        return self.model_loaded and ESRGAN_AVAILABLE


# Convenience function for quick upscaling
def quick_upscale(input_path: str, output_path: str, scale_factor: int = 2,
                  model_path: Optional[str] = None) -> bool:
    """
    Quick upscaling function for single images
    
    Args:
        input_path: Path to input image
        output_path: Path to save upscaled image
        scale_factor: Upscaling factor (2 or 4)
        model_path: Path to Remacri model weights
    
    Returns:
        True if successful, False otherwise
    """
    try:
        upscaler = FLUXUpscaler(model_path)
        if not upscaler.is_ready():
            logger.error("Upscaler not ready")
            return False
        
        return upscaler.upscale_file(input_path, output_path, scale_factor)
        
    except Exception as e:
        logger.error(f"Quick upscaling failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("🚀 FLUX Upscaler - Remacri ESRGAN Implementation")
    
    # Test the upscaler
    upscaler = FLUXUpscaler()
    
    if upscaler.is_ready():
        print("✅ Upscaler ready!")
        print(f"Model info: {upscaler.get_model_info()}")
    else:
        print("❌ Upscaler not ready")
        print("Make sure ESRGAN is installed and model weights are available")


# API Integration Functions
def apply_upscaling(image, upscale: bool, upscale_factor: int, save_original_func) -> Tuple[str, Optional[str], int, int]:
    """
    Apply upscaling to the generated image if requested
    
    Args:
        image: The generated image (PIL Image or numpy array)
        upscale: Whether to apply upscaling
        upscale_factor: Upscaling factor (2 or 4)
        save_original_func: Function to save the original image
    
    Returns:
        Tuple of (final_image_path, upscaled_image_path, final_width, final_height)
    """
    # Convert PIL Image to numpy array if needed
    if hasattr(image, 'size'):  # PIL Image
        import numpy as np
        image_array = np.array(image)
        # Convert RGB to BGR for OpenCV compatibility
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = image_array[:, :, [2, 1, 0]]  # RGB to BGR
    else:  # Already numpy array
        image_array = image
    
    if not upscale:
        # Save original image without upscaling
        image_filename = save_original_func(image)
        # Get original dimensions from the image
        h, w = image_array.shape[:2]
        return image_filename, None, w, h
    
    try:
        logger.info(f"Starting upscaling with factor {upscale_factor}x...")
        upscaler = FLUXUpscaler()
        
        # Check if upscaler is ready and log detailed status
        if not upscaler.is_ready():
            logger.warning("Upscaler not ready, using original image")
            logger.info(f"Upscaler status: {upscaler.get_model_info()}")
            image_filename = save_original_func(image)
            # Get original dimensions from the image
            h, w = image_array.shape[:2]
            return image_filename, None, w, h
        
        # Save the original image first
        original_image_filename = save_original_func(image)
        
        # Create upscaled filename
        base_name = os.path.splitext(original_image_filename)[0]
        upscaled_image_path = f"{base_name}_upscaled_{upscale_factor}x.png"
        
        # Perform upscaling
        logger.info(f"Calling upscaler.upscale_file with: {original_image_filename} -> {upscaled_image_path}, factor: {upscale_factor}")
        success = upscaler.upscale_file(
            original_image_filename, 
            upscaled_image_path, 
            upscale_factor
        )
        logger.info(f"Upscaling result: {success}")
        
        if success:
            logger.info(f"Upscaling completed successfully: {upscaled_image_path}")
            # Get upscaled dimensions by reading the actual upscaled image
            try:
                upscaled_image = cv2.imread(upscaled_image_path, cv2.IMREAD_COLOR)
                if upscaled_image is not None:
                    upscaled_h, upscaled_w = upscaled_image.shape[:2]
                    logger.info(f"Upscaled image dimensions: {upscaled_w}×{upscaled_h}")
                else:
                    # Fallback to calculated dimensions
                    upscaled_h, upscaled_w = image_array.shape[:2]
                    if upscale_factor == 2:
                        upscaled_w *= 2
                        upscaled_h *= 2
                    elif upscale_factor == 4:
                        upscaled_w *= 4
                        upscaled_h *= 4
                    logger.info(f"Calculated upscaled dimensions: {upscaled_w}×{upscaled_h}")
            except Exception as dim_error:
                logger.warning(f"Could not read upscaled image dimensions: {dim_error}")
                # Fallback to calculated dimensions
                upscaled_h, upscaled_w = image_array.shape[:2]
                if upscale_factor == 2:
                    upscaled_w *= 2
                    upscaled_h *= 2
                elif upscale_factor == 4:
                    upscaled_w *= 4
                    upscaled_h *= 4
                logger.info(f"Fallback calculated dimensions: {upscaled_w}×{upscaled_h}")
            
            return upscaled_image_path, upscaled_image_path, upscaled_w, upscaled_h
        else:
            logger.warning("Upscaling failed, using original image")
            # Even though upscaling failed, return the calculated upscaled dimensions
            # so the user knows what the target resolution would be
            h, w = image_array.shape[:2]
            if upscale_factor == 2:
                w *= 2
                h *= 2
            elif upscale_factor == 4:
                w *= 4
                h *= 4
            logger.info(f"Returning calculated target dimensions: {w}×{h}")
            return original_image_filename, None, w, h
            
    except Exception as upscale_error:
        logger.error(f"Upscaling error: {upscale_error}")
        # Fall back to original image
        image_filename = save_original_func(image)
        # Even though upscaling failed, return the calculated upscaled dimensions
        # so the user knows what the target resolution would be
        h, w = image_array.shape[:2]
        if upscale_factor == 2:
            w *= 2
            h *= 2
        elif upscale_factor == 4:
            w *= 4
            h *= 4
        logger.info(f"Exception handler returning calculated target dimensions: {w}×{h}")
        return image_filename, None, w, h
