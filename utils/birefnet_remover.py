"""
BiRefNet Background Removal Module
Uses ZhengPeng7/BiRefNet_HR model for background removal with singleton pattern to avoid repeated model loadingre'ma'c'ri
"""

import os
import time
import threading
from typing import Optional
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import loguru

logger = loguru.logger

class BiRefNetRemover:
    """BiRefNet background remover using singleton pattern to avoid repeated model loading"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BiRefNetRemover, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.model_name = "ZhengPeng7/BiRefNet_HR"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        
        # 图像预处理参数
        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"BiRefNetRemover initialized, device: {self.device}")
    
    def load_model(self) -> bool:
        """Load BiRefNet model"""
        if self.model is not None:
            logger.info("BiRefNet model already loaded, skipping reload")
            return True
            
        try:
            logger.info(f"Starting to load BiRefNet model: {self.model_name}")
            load_start_time = time.time()
            
            # Load model
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set model parameters
            torch.set_float32_matmul_precision(['high', 'highest'][0])
            self.model.to(self.device)
            self.model.eval()
            self.model.half()
            
            load_end_time = time.time()
            load_duration = load_end_time - load_start_time
            
            logger.info(f"BiRefNet model loaded successfully, time: {load_duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BiRefNet model: {e}")
            self.model = None
            return False
    
    def remove_background(self, image: Image.Image, bg_strength: Optional[float] = None) -> Image.Image:
        """
        Remove background from image
        
        Args:
            image: PIL Image object
            bg_strength: Background removal strength (0.0-1.0), currently unused but kept for API compatibility
            
        Returns:
            Processed PIL Image object (RGBA format with transparent background)
        """
        if self.model is None:
            if not self.load_model():
                logger.error("Model loading failed, cannot perform background removal")
                return image
        
        try:
            # Record processing start time
            process_start_time = time.time()
            
            # Preprocess image
            input_images = self.transform_image(image).unsqueeze(0).to(self.device).half()
            
            # Perform prediction
            with torch.no_grad():
                preds = self.model(input_images)[-1].sigmoid().cpu()
            
            # Post-process
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image.size)
            
            # Apply mask to original image
            result_image = image.copy()
            result_image.putalpha(mask)
            
            process_end_time = time.time()
            process_duration = process_end_time - process_start_time
            
            logger.info(f"Background removal completed, time: {process_duration:.2f}s")
            return result_image
            
        except Exception as e:
            logger.error(f"Background removal processing failed: {e}")
            return image
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_model_loaded(),
            "image_size": self.image_size
        }


# Global singleton instance
_birefnet_remover = None
_birefnet_lock = threading.Lock()

def get_birefnet_remover() -> BiRefNetRemover:
    """Get BiRefNet background remover singleton instance"""
    global _birefnet_remover
    if _birefnet_remover is None:
        with _birefnet_lock:
            if _birefnet_remover is None:
                _birefnet_remover = BiRefNetRemover()
    return _birefnet_remover


def remove_background_birefnet(image: Image.Image, bg_strength: Optional[float] = None) -> Image.Image:
    """
    Convenience function to remove background using BiRefNet
    
    Args:
        image: PIL Image object
        bg_strength: Background removal strength (0.0-1.0), currently unused
        
    Returns:
        Processed PIL Image object (RGBA format with transparent background)
    """
    remover = get_birefnet_remover()
    return remover.remove_background(image, bg_strength)
