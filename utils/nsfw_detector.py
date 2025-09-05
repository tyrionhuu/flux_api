"""
NSFW content detection using Falconsai/nsfw_image_detection model.
Provides NSFW scoring for informational purposes - does not block generation.
"""

import asyncio
import logging
import time
from typing import Optional
from PIL import Image
import torch

logger = logging.getLogger(__name__)

class NSFWDetector:
    """
    Singleton NSFW detector with lazy loading and timeout protection.
    Provides NSFW scores for monitoring - does not block content generation.
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.classifier = None
            self.device = 0 if torch.cuda.is_available() else -1
            logger.info(f"NSFW Detector initialized, device: {'CUDA' if self.device >= 0 else 'CPU'}")
    
    def _load_model(self):
        """Lazy load the model on first use"""
        if self.classifier is not None:
            return
        
        try:
            from transformers import pipeline
            logger.info("Loading NSFW detection model...")
            
            self.classifier = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=self.device
            )
            
            logger.info("NSFW detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NSFW model: {e}")
            self.classifier = None
    
    async def detect_nsfw(self, image: Image.Image, timeout: float = 5.0) -> float:
        """
        Detect NSFW content in image with timeout protection.
        Note: Score is for informational purposes only, does not block generation.
        
        Returns:
            float: NSFW probability score (0.0-1.0)
            1.0 on any error/timeout (indicates detection failure)
        """
        try:
            # Ensure model is loaded
            if self.classifier is None:
                async with self._lock:
                    if self.classifier is None:
                        await asyncio.get_event_loop().run_in_executor(None, self._load_model)
            
            if self.classifier is None:
                logger.error("NSFW classifier not available, returning unsafe")
                return 1.0
            
            # Run detection with timeout
            start_time = time.time()
            
            async def _run_detection():
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self.classifier, image)
                # Extract NSFW score from results
                # Results format: [{'label': 'nsfw', 'score': 0.9}, {'label': 'normal', 'score': 0.1}]
                for item in results:
                    if item['label'] == 'nsfw':
                        return item['score']
                return 0.0
            
            # Apply timeout
            nsfw_score = await asyncio.wait_for(_run_detection(), timeout=timeout)
            
            elapsed = time.time() - start_time
            logger.info(f"NSFW detection completed in {elapsed:.2f}s, score: {nsfw_score:.3f}")
            
            return nsfw_score
            
        except asyncio.TimeoutError:
            logger.warning(f"NSFW detection timed out after {timeout}s, returning unsafe")
            return 1.0
        except Exception as e:
            logger.error(f"NSFW detection failed: {e}, returning unsafe")
            return 1.0
    
    def detect_nsfw_sync(self, image: Image.Image, timeout: float = 5.0) -> float:
        """
        Synchronous wrapper for NSFW detection.
        Works in both sync and async contexts by using thread pool.
        """
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._detect_nsfw_blocking, image, timeout)
                return future.result(timeout=timeout + 1)
        except Exception as e:
            logger.error(f"Sync NSFW detection failed: {e}")
            return 1.0
    
    def _detect_nsfw_blocking(self, image: Image.Image, timeout: float = 5.0) -> float:
        """
        Blocking version of NSFW detection for thread executor.
        Simple version without signal-based timeout (incompatible with threads).
        """
        try:
            # Ensure model is loaded
            if self.classifier is None:
                self._load_model()
            
            if self.classifier is None:
                logger.error("NSFW classifier not available")
                return 1.0
            
            # Run detection directly (blocking)
            # Note: Timeout is handled by the ThreadPoolExecutor.result() call
            import time
            start_time = time.time()
            
            results = self.classifier(image)
            
            elapsed = time.time() - start_time
            logger.info(f"NSFW detection took {elapsed:.2f}s")
            
            # Extract NSFW score from results
            for item in results:
                if item['label'] == 'nsfw':
                    score = item['score']
                    logger.info(f"NSFW score: {score:.3f}")
                    return score
            
            # If no NSFW label found, return 0
            logger.info("No NSFW label found in results, returning 0.0")
            return 0.0
                
        except Exception as e:
            logger.error(f"NSFW detection error: {e}")
            return 1.0


# Global singleton instance
_detector = None

def get_detector() -> NSFWDetector:
    """Get the global NSFW detector instance"""
    global _detector
    if _detector is None:
        _detector = NSFWDetector()
    return _detector


def check_image_nsfw(image: Image.Image, timeout: float = 5.0) -> float:
    """
    Simple API for NSFW checking.
    Note: This only reports scores, does not block generation.
    
    Returns:
        float: NSFW confidence score (0.0-1.0)
    """
    detector = get_detector()
    score = detector.detect_nsfw_sync(image, timeout)
    
    return score