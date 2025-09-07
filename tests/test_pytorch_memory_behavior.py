#!/usr/bin/env python3
"""
Test PyTorch's memory allocation behavior to understand why it reserves so much
"""

import torch
import torch.nn as nn
import numpy as np
import gc

def print_memory_stats(label):
    """Print detailed memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n{label}:")
        print(f"  Allocated: {allocated:.3f} GB")
        print(f"  Reserved:  {reserved:.3f} GB")
        print(f"  Free in reserved pool: {(reserved - allocated):.3f} GB")
        
        # Get more detailed stats
        stats = torch.cuda.memory_stats()
        print(f"  Active blocks: {stats.get('active.all.current', 0)}")
        print(f"  Allocation requests: {stats.get('allocation.all.current', 0)}")
        
def test_simple_convolutions():
    """Test memory behavior with simple convolutions"""
    print("=" * 60)
    print("TEST 1: Simple Convolution Layers")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print_memory_stats("Initial")
    
    # Create a simple conv network
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 3, 3, padding=1)
    ).cuda()
    
    print_memory_stats("After model creation")
    
    # Test different input sizes
    sizes = [(256, 256), (512, 512), (512, 1024)]
    
    for h, w in sizes:
        torch.cuda.empty_cache()
        
        # Create input
        x = torch.randn(1, 3, h, w).cuda()
        print_memory_stats(f"After creating {h}x{w} input")
        
        # Forward pass
        with torch.no_grad():
            y = model(x)
            
        print_memory_stats(f"After forward pass")
        
        # Check peak memory
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak allocated: {peak:.3f} GB")
        
        # Clean up
        del x, y
        torch.cuda.empty_cache()
        
def test_deep_network():
    """Test memory with deep network (similar to RRDB)"""
    print("\n" + "=" * 60)
    print("TEST 2: Deep Network (RRDB-like)")
    print("=" * 60)
    
    class DenseBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
            
        def forward(self, x):
            x1 = torch.relu(self.conv1(x))
            x2 = torch.relu(self.conv2(torch.cat([x, x1], dim=1)[:, :x.shape[1]]))
            x3 = torch.relu(self.conv3(torch.cat([x, x1, x2], dim=1)[:, :x.shape[1]]))
            return x3 + x  # Residual connection
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create a deep network with multiple dense blocks
    model = nn.Sequential(*[DenseBlock(64) for _ in range(10)]).cuda()
    
    print_memory_stats("After creating deep model")
    
    # Test with 512x1024 input
    x = torch.randn(1, 64, 512, 1024).cuda()
    print_memory_stats("After creating input")
    
    with torch.no_grad():
        y = model(x)
    
    print_memory_stats("After forward pass")
    
    # Clean up
    del x, y
    torch.cuda.empty_cache()
    print_memory_stats("After cleanup")

def test_memory_allocation_pattern():
    """Test PyTorch's memory allocation pattern"""
    print("\n" + "=" * 60)
    print("TEST 3: Memory Allocation Pattern")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("\nAllocating tensors of increasing sizes:")
    
    sizes_mb = [1, 10, 50, 100, 500, 1000]
    
    for size_mb in sizes_mb:
        torch.cuda.empty_cache()
        
        # Calculate tensor size
        elements = (size_mb * 1024 * 1024) // 4  # float32
        
        # Allocate tensor
        tensor = torch.zeros(elements).cuda()
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        print(f"\n  Requested: {size_mb:4d} MB")
        print(f"  Allocated: {allocated:7.1f} MB")
        print(f"  Reserved:  {reserved:7.1f} MB")
        print(f"  Ratio:     {reserved/allocated:5.2f}x")
        
        del tensor

def test_upsampling_memory():
    """Test memory usage during upsampling operations"""
    print("\n" + "=" * 60)
    print("TEST 4: Upsampling Operations (like ESRGAN)")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Simulate ESRGAN-like upsampling
    h, w = 512, 1024
    
    # Input tensor
    x = torch.randn(1, 64, h, w).cuda()
    print_memory_stats(f"Input {h}x{w}x64")
    
    # First upsample (2x)
    x_up1 = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
    print_memory_stats(f"After 2x upsample to {h*2}x{w*2}x64")
    
    # Conv after upsample
    conv = nn.Conv2d(64, 64, 3, padding=1).cuda()
    x_conv1 = conv(x_up1)
    print_memory_stats(f"After conv on {h*2}x{w*2}x64")
    
    # Second upsample (4x total)
    x_up2 = torch.nn.functional.interpolate(x_conv1, scale_factor=2, mode='nearest')
    print_memory_stats(f"After 4x upsample to {h*4}x{w*4}x64")
    
    # Final conv to RGB
    conv_final = nn.Conv2d(64, 3, 3, padding=1).cuda()
    output = conv_final(x_up2)
    print_memory_stats(f"Final output {h*4}x{w*4}x3")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output size: {output.numel() * 4 / 1024**2:.1f} MB")

def test_memory_fragmentation():
    """Test memory fragmentation behavior"""
    print("\n" + "=" * 60)
    print("TEST 5: Memory Fragmentation")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    
    print("\nAllocating and freeing many tensors:")
    
    tensors = []
    
    # Allocate many tensors of different sizes
    for i in range(100):
        size = np.random.randint(1, 100) * 1024 * 1024 // 4  # 1-100 MB
        tensor = torch.zeros(size).cuda()
        tensors.append(tensor)
        
        if i % 20 == 19:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  After {i+1} allocations: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB")
    
    # Free half of them randomly
    import random
    indices = list(range(100))
    random.shuffle(indices)
    
    for i in indices[:50]:
        del tensors[i]
        tensors[i] = None
    
    print_memory_stats("After freeing 50% randomly")
    
    # Try to allocate a large tensor
    try:
        large = torch.zeros(500 * 1024 * 1024 // 4).cuda()  # 500 MB
        print("  ✓ Could allocate 500MB tensor")
        del large
    except:
        print("  ✗ Could NOT allocate 500MB tensor (fragmentation!)")
    
    # Clean up
    tensors = []
    torch.cuda.empty_cache()
    print_memory_stats("After full cleanup")

def check_pytorch_allocator_config():
    """Check PyTorch memory allocator configuration"""
    print("\n" + "=" * 60)
    print("PyTorch CUDA Allocator Configuration")
    print("=" * 60)
    
    import os
    
    # Check environment variables
    env_vars = [
        'PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_LAUNCH_BLOCKING',
        'PYTORCH_NO_CUDA_MEMORY_CACHING'
    ]
    
    print("\nEnvironment variables:")
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Check PyTorch allocator info
    if hasattr(torch.cuda, 'get_allocator_backend'):
        print(f"\nAllocator backend: {torch.cuda.get_allocator_backend()}")
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Memory allocation strategy
    print("\nDefault behavior:")
    print("  - PyTorch uses a caching allocator")
    print("  - Reserves memory in large blocks (512MB default)")
    print("  - Doesn't return memory to OS immediately")
    print("  - This improves performance but uses more memory")
    
    print("\nTo reduce memory reservation, set:")
    print("  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
    print("  This limits the maximum size of memory blocks")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    print("🔍 PyTorch Memory Allocation Behavior Analysis")
    print("=" * 60)
    
    # Run tests
    test_simple_convolutions()
    test_deep_network()
    test_memory_allocation_pattern()
    test_upsampling_memory()
    test_memory_fragmentation()
    check_pytorch_allocator_config()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
PyTorch reserves much more memory than needed because:

1. **Caching Allocator**: PyTorch doesn't return memory to the OS immediately.
   It keeps a pool of allocated memory for reuse.

2. **Block Allocation**: Memory is allocated in large blocks (default 512MB).
   Even small tensors can trigger large reservations.

3. **Fragmentation Prevention**: Over-allocation helps prevent fragmentation
   and improves performance for future allocations.

4. **Peak Memory**: During forward pass, many intermediate tensors exist
   simultaneously, especially with deep networks.

For ESRGAN's 11.5GB reservation on 512x1024 input:
- The deep RRDB architecture creates many intermediate tensors
- Each upsampling doubles the spatial dimensions (4x memory)
- PyTorch reserves enough for worst-case peak usage

This is NORMAL behavior for PyTorch, but can be optimized.
    """)