#!/usr/bin/env python3
"""
Test script to replicate and profile upscaling performance issue
Tests upscaling a 512x1024 image to 2x and profiles each step
"""

import os
import sys
import time
import torch
import numpy as np
import cv2
from PIL import Image
import cProfile
import pstats
from io import StringIO
import logging

# Add the project directory to the path
sys.path.insert(0, '/data/pingzhi/flux_api')

# Import the upscaler module exactly as used in production
from models.upscaler import FLUXUpscaler, get_upscaler

# Configure logging to see detailed timing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image(width=512, height=1024):
    """Create a test image similar to what FLUX would generate"""
    # Create a random RGB image (similar to generated images)
    np.random.seed(42)
    # Generate realistic-looking image data (0-255 range, uint8)
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Add some patterns to make it more realistic
    # Add a gradient
    for i in range(height):
        image[i, :, 0] = (image[i, :, 0] * 0.7 + (i / height * 255) * 0.3).astype(np.uint8)
    
    return image

def profile_upscaling_detailed(upscaler, image, scale_factor=2):
    """Profile upscaling with detailed timing for each step"""
    
    print(f"\n{'='*60}")
    print(f"Detailed Profiling: {scale_factor}x upscaling of {image.shape[1]}x{image.shape[0]} image")
    print(f"{'='*60}")
    
    # Warm up GPU (first run is always slower)
    print("\n🔥 Warming up GPU...")
    _ = upscaler.upscale_image(image, scale_factor)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("\n📊 Starting detailed profiling...")
    
    # Profile the main upscale_image function
    total_start = time.time()
    
    # Step 1: Preprocess image
    h0, w0 = image.shape[:2]
    
    preprocess_start = time.time()
    img = image.astype(np.float32) / 255.0
    img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
    img_tensor = torch.from_numpy(img).float().unsqueeze(0)
    img_tensor = img_tensor.to(upscaler.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    preprocess_time = time.time() - preprocess_start
    print(f"  1. Preprocessing (to GPU): {preprocess_time:.3f}s")
    
    # Step 2: ESRGAN inference (4x upscaling)
    inference_start = time.time()
    with torch.no_grad():
        output = upscaler.model(img_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    print(f"  2. ESRGAN inference (4x): {inference_time:.3f}s")
    
    # Step 3: Postprocess to numpy
    postprocess_start = time.time()
    output_np = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_np = np.transpose(output_np[[2, 1, 0], :, :], (1, 2, 0))
    output_4x = (output_np * 255.0).round().astype(np.uint8)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    postprocess_time = time.time() - postprocess_start
    print(f"  3. Postprocessing (to CPU): {postprocess_time:.3f}s")
    
    # Step 4: Downsampling to 2x (if needed)
    if scale_factor == 2:
        downsample_start = time.time()
        
        # This is the exact code from _high_quality_resize
        target_width = w0 * 2
        target_height = h0 * 2
        
        # Sub-step 4.1: Convert to PIL
        pil_convert_start = time.time()
        pil_image = Image.fromarray(output_4x.astype(np.uint8))
        pil_convert_time = time.time() - pil_convert_start
        
        # Sub-step 4.2: PIL resize
        pil_resize_start = time.time()
        resized_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        pil_resize_time = time.time() - pil_resize_start
        
        # Sub-step 4.3: Convert back to numpy
        numpy_convert_start = time.time()
        output_2x = np.array(resized_image)
        numpy_convert_time = time.time() - numpy_convert_start
        
        downsample_time = time.time() - downsample_start
        
        print(f"  4. Downsampling 4x→2x (CPU): {downsample_time:.3f}s")
        print(f"     - PIL conversion: {pil_convert_time:.3f}s")
        print(f"     - PIL resize (LANCZOS): {pil_resize_time:.3f}s")
        print(f"     - NumPy conversion: {numpy_convert_time:.3f}s")
    
    total_time = time.time() - total_start
    
    print(f"\n📈 Total time: {total_time:.3f}s")
    
    # Calculate percentages
    print(f"\n📊 Time breakdown:")
    print(f"  - Preprocessing: {preprocess_time/total_time*100:.1f}%")
    print(f"  - ESRGAN inference: {inference_time/total_time*100:.1f}%")
    print(f"  - Postprocessing: {postprocess_time/total_time*100:.1f}%")
    if scale_factor == 2:
        print(f"  - Downsampling: {downsample_time/total_time*100:.1f}%")
    
    return total_time

def test_with_production_method(image, scale_factor=2):
    """Test using the exact production method (get_upscaler singleton)"""
    print(f"\n{'='*60}")
    print("Testing with PRODUCTION method (singleton)")
    print(f"{'='*60}")
    
    # Use the exact method from production
    upscaler = get_upscaler()
    
    if not upscaler.is_ready():
        print("❌ Upscaler not ready!")
        return None, []
    
    print(f"✅ Upscaler ready: {upscaler.get_model_info()}")
    
    # Warm up
    print("\n🔥 Warming up...")
    _ = upscaler.upscale_image(image, scale_factor)
    
    # Test multiple runs to get average
    times = []
    print(f"\n⏱️  Running 3 timing tests...")
    for i in range(3):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        start = time.time()
        result = upscaler.upscale_image(image, scale_factor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n📊 Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    
    return upscaler, times

def test_gpu_vs_cpu_resize():
    """Compare GPU vs CPU resizing performance"""
    print(f"\n{'='*60}")
    print("GPU vs CPU Resize Comparison")
    print(f"{'='*60}")
    
    # Create a 4x upscaled image (2048x4096 from 512x1024)
    test_image = np.random.randint(0, 256, (4096, 2048, 3), dtype=np.uint8)
    target_size = (1024, 2048)  # 2x of original
    
    # Test CPU resize (current method)
    print("\n🖥️  CPU Resize (PIL LANCZOS):")
    cpu_times = []
    for i in range(3):
        start = time.time()
        pil_image = Image.fromarray(test_image)
        resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        result_cpu = np.array(resized)
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)
        print(f"  Run {i+1}: {cpu_time:.3f}s")
    print(f"  Average: {np.mean(cpu_times):.3f}s")
    
    # Test GPU resize (potential optimization)
    if torch.cuda.is_available():
        print("\n🎮 GPU Resize (torch.nn.functional.interpolate):")
        
        # Convert to tensor and move to GPU
        tensor_image = torch.from_numpy(test_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor_image = tensor_image.cuda()
        
        gpu_times = []
        for i in range(3):
            torch.cuda.synchronize()
            start = time.time()
            resized_tensor = torch.nn.functional.interpolate(
                tensor_image, 
                size=target_size[::-1],  # (H, W)
                mode='bicubic',
                align_corners=False
            )
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)
            print(f"  Run {i+1}: {gpu_time:.3f}s")
        
        print(f"  Average: {np.mean(gpu_times):.3f}s")
        
        # Including transfer time
        print("\n  With CPU↔GPU transfer:")
        transfer_times = []
        for i in range(3):
            torch.cuda.synchronize()
            start = time.time()
            resized_tensor = torch.nn.functional.interpolate(
                tensor_image, 
                size=target_size[::-1],
                mode='bicubic',
                align_corners=False
            )
            result_gpu = (resized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            torch.cuda.synchronize()
            transfer_time = time.time() - start
            transfer_times.append(transfer_time)
            print(f"  Run {i+1}: {transfer_time:.3f}s")
        print(f"  Average: {np.mean(transfer_times):.3f}s")
        
        print(f"\n🚀 Speedup: {np.mean(cpu_times)/np.mean(gpu_times):.1f}x (pure GPU)")
        print(f"🚀 Speedup: {np.mean(cpu_times)/np.mean(transfer_times):.1f}x (with transfers)")

def main():
    print("🔬 FLUX Upscaling Performance Test")
    print("=" * 60)
    
    # Create test image (512x1024 as mentioned)
    print("\n📸 Creating test image (512x1024)...")
    test_image = create_test_image(512, 1024)
    print(f"  Image shape: {test_image.shape}")
    print(f"  Image dtype: {test_image.dtype}")
    
    # Save test image for verification
    cv2.imwrite('/tmp/test_input.jpg', test_image)
    print(f"  Saved test image to /tmp/test_input.jpg")
    
    # Test 1: Production method
    upscaler, times_2x = test_with_production_method(test_image, scale_factor=2)
    
    # Test 2: Detailed profiling
    if upscaler:
        profile_upscaling_detailed(upscaler, test_image, scale_factor=2)
    
    # Test 3: GPU vs CPU resize comparison
    test_gpu_vs_cpu_resize()
    
    # Test 4: Compare 2x vs 4x upscaling
    if upscaler:
        print(f"\n{'='*60}")
        print("Comparing 2x vs 4x upscaling")
        print(f"{'='*60}")
        
        # Test 4x upscaling
        print("\n🔍 4x upscaling:")
        times_4x = []
        for i in range(3):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            start = time.time()
            _ = upscaler.upscale_image(test_image, scale_factor=4)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times_4x.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")
        print(f"  Average 4x: {np.mean(times_4x):.3f}s")
        
        print("\n📊 Summary:")
        print(f"  2x upscaling: {np.mean(times_2x):.3f}s")
        print(f"  4x upscaling: {np.mean(times_4x):.3f}s")
        print(f"  Difference: {np.mean(times_2x) - np.mean(times_4x):.3f}s")
        print(f"  → This difference is the CPU downsampling overhead!")

if __name__ == "__main__":
    main()