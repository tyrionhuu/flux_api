#!/usr/bin/env python3
"""
Test script to analyze GPU memory usage of the upscaler model
"""

import os
import sys
import torch
import numpy as np
import gc

# Add the project directory to the path
sys.path.insert(0, '/data/pingzhi/flux_api')

from models.upscaler import FLUXUpscaler

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0

def analyze_memory_usage():
    """Analyze memory usage at each stage"""
    print("🔍 GPU Memory Usage Analysis")
    print("=" * 60)
    
    # Initial state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    allocated, reserved = get_gpu_memory_info()
    print(f"\n1. Initial state:")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")
    
    # Load model
    print(f"\n2. Loading ESRGAN model...")
    upscaler = FLUXUpscaler()
    
    allocated, reserved = get_gpu_memory_info()
    print(f"   After model load:")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")
    
    # Check model size
    model_params = sum(p.numel() for p in upscaler.model.parameters())
    model_size_mb = (model_params * 4) / (1024 * 1024)  # Assuming float32
    print(f"   Model parameters: {model_params:,}")
    print(f"   Model size (theoretical): {model_size_mb:.2f} MB")
    
    # Create test images of different sizes
    sizes = [
        (512, 512),
        (512, 1024),
        (1024, 1024),
    ]
    
    for width, height in sizes:
        print(f"\n3. Testing {width}x{height} image:")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        allocated_before, reserved_before = get_gpu_memory_info()
        print(f"   Before inference:")
        print(f"   Allocated: {allocated_before:.2f} GB")
        print(f"   Reserved: {reserved_before:.2f} GB")
        
        # Create test image
        test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Preprocess
        img = test_image.astype(np.float32) / 255.0
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Calculate input tensor size
        input_size_mb = img_tensor.numel() * 4 / (1024 * 1024)
        print(f"   Input tensor size: {input_size_mb:.2f} MB")
        
        # Move to GPU
        img_tensor = img_tensor.cuda()
        
        allocated_input, reserved_input = get_gpu_memory_info()
        print(f"   After input to GPU:")
        print(f"   Allocated: {allocated_input:.2f} GB (+{(allocated_input-allocated_before):.2f} GB)")
        print(f"   Reserved: {reserved_input:.2f} GB")
        
        # Run inference
        with torch.no_grad():
            output = upscaler.model(img_tensor)
        
        # Calculate output size
        output_size_mb = output.numel() * 4 / (1024 * 1024)
        print(f"   Output tensor size: {output_size_mb:.2f} MB")
        print(f"   Output shape: {output.shape}")
        
        allocated_after, reserved_after = get_gpu_memory_info()
        print(f"   After inference:")
        print(f"   Allocated: {allocated_after:.2f} GB (+{(allocated_after-allocated_input):.2f} GB)")
        print(f"   Reserved: {reserved_after:.2f} GB")
        
        # Calculate intermediate activations estimate
        intermediate_gb = (allocated_after - allocated_before - input_size_mb/1024 - output_size_mb/1024)
        print(f"   Estimated intermediate activations: {intermediate_gb:.2f} GB")
        
        # Clear output
        del output
        del img_tensor
        
    # Analyze model architecture
    print(f"\n4. Model Architecture Analysis:")
    print(f"   Model type: RRDBNet")
    
    # Count layers
    num_conv_layers = 0
    num_params_by_layer = {}
    
    for name, module in upscaler.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            num_conv_layers += 1
            params = sum(p.numel() for p in module.parameters())
            num_params_by_layer[name] = params
    
    print(f"   Number of Conv2d layers: {num_conv_layers}")
    print(f"   Top 5 largest layers:")
    sorted_layers = sorted(num_params_by_layer.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, params in sorted_layers:
        size_mb = (params * 4) / (1024 * 1024)
        print(f"     {name}: {params:,} params ({size_mb:.2f} MB)")
    
    # Test with different precision
    print(f"\n5. Testing memory with different batch sizes:")
    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    for batch_size in [1, 2, 4]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Create batch
        batch_images = np.stack([test_image] * batch_size)
        batch_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2) / 255.0
        batch_tensor = batch_tensor[:, [2, 1, 0], :, :]  # BGR to RGB
        
        batch_size_mb = batch_tensor.numel() * 4 / (1024 * 1024)
        
        allocated_before, _ = get_gpu_memory_info()
        
        try:
            batch_tensor = batch_tensor.cuda()
            with torch.no_grad():
                output = upscaler.model(batch_tensor)
            
            allocated_after, _ = get_gpu_memory_info()
            memory_used = allocated_after - allocated_before
            
            print(f"   Batch size {batch_size}: {memory_used:.2f} GB used")
            print(f"     Per image: {memory_used/batch_size:.2f} GB")
            
            del output
            del batch_tensor
            
        except torch.cuda.OutOfMemoryError:
            print(f"   Batch size {batch_size}: OOM")
    
    # Check for memory fragmentation
    print(f"\n6. Memory Fragmentation Check:")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated_final, reserved_final = get_gpu_memory_info()
        print(f"   After empty_cache:")
        print(f"   Allocated: {allocated_final:.2f} GB")
        print(f"   Reserved: {reserved_final:.2f} GB")
        
        if reserved_final > allocated_final + 0.5:
            print(f"   ⚠️ High fragmentation detected: {reserved_final - allocated_final:.2f} GB reserved but not allocated")

def check_esrgan_architecture():
    """Check ESRGAN RRDBNet architecture details"""
    print("\n7. ESRGAN Architecture Deep Dive:")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, '/data/pingzhi/flux_api')
    import ESRGAN.RRDBNet_arch as arch
    
    # Create model
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    
    # Calculate memory for intermediate feature maps
    # RRDBNet uses 23 RRDB blocks, each with 3 dense blocks
    # Feature maps grow through the network
    
    print("   RRDBNet configuration:")
    print(f"   - Input channels: 3")
    print(f"   - Output channels: 3")
    print(f"   - Feature channels: 64")
    print(f"   - Number of RRDB blocks: 23")
    print(f"   - Growth channels: 32")
    
    # For a 512x1024 input, after 4x upscaling = 2048x4096
    # Intermediate feature maps can be large
    input_h, input_w = 512, 1024
    feature_channels = 64
    
    # Calculate sizes at different stages
    print(f"\n   Memory usage for {input_w}x{input_h} input:")
    
    # Initial conv
    size_mb = input_h * input_w * feature_channels * 4 / (1024**2)
    print(f"   After initial conv (64 channels): {size_mb:.2f} MB")
    
    # RRDB blocks (feature maps stay same size but many intermediate tensors)
    # Each RRDB has 3 dense blocks, each dense block has 5 convs
    # Peak memory includes all intermediate activations
    num_rrdb = 23
    num_dense_per_rrdb = 3
    num_conv_per_dense = 5
    
    # Worst case: all intermediate activations kept in memory
    total_intermediate_mb = size_mb * num_rrdb * num_dense_per_rrdb * num_conv_per_dense
    print(f"   Worst-case intermediate activations: {total_intermediate_mb/1024:.2f} GB")
    
    # After upsampling layers
    up1_h, up1_w = input_h * 2, input_w * 2
    up1_mb = up1_h * up1_w * feature_channels * 4 / (1024**2)
    print(f"   After first upsample (2x): {up1_mb:.2f} MB")
    
    up2_h, up2_w = input_h * 4, input_w * 4
    up2_mb = up2_h * up2_w * feature_channels * 4 / (1024**2)
    print(f"   After second upsample (4x): {up2_mb:.2f} MB")
    
    # Final output
    output_mb = up2_h * up2_w * 3 * 4 / (1024**2)
    print(f"   Final output (3 channels): {output_mb:.2f} MB")
    
    # Peak memory usage estimate
    peak_gb = (size_mb + up1_mb + up2_mb + output_mb) / 1024
    print(f"\n   Minimum memory for forward pass: {peak_gb:.2f} GB")
    print(f"   But PyTorch may keep more intermediate tensors for autograd graphs")

if __name__ == "__main__":
    analyze_memory_usage()
    check_esrgan_architecture()