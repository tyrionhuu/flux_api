"""
GPU management utilities for the FLUX API
"""

import torch
from typing import Optional, Tuple
from config.settings import DEFAULT_GPU, CUDA_TEST_TENSOR_SIZE


class GPUManager:
    """Manages GPU operations and selection"""
    
    def __init__(self):
        self.selected_gpu: Optional[int] = None
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    def test_cuda_compatibility(self) -> bool:
        """Test if CUDA is actually working (not just available)"""
        if not torch.cuda.is_available():
            return False
            
        try:
            # Test if we can actually use CUDA
            test_tensor = torch.randn(*CUDA_TEST_TENSOR_SIZE, device="cuda:0")
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"CUDA compatibility issue: {e}")
            return False
    
    def select_best_gpu(self) -> Optional[int]:
        """Select GPU with most free memory"""
        if not torch.cuda.is_available():
            return None

        gpu_memory = []
        for i in range(self.device_count):
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free_memory = (
                torch.cuda.get_device_properties(i).total_memory
                - torch.cuda.memory_allocated(i)
            ) / 1024**3
            gpu_memory.append((i, free_memory, total_memory))
            print(f"GPU {i}: {free_memory:.1f}GB free / {total_memory:.1f}GB total")

        # Select GPU with most free memory
        best_gpu = max(gpu_memory, key=lambda x: x[1])
        self.selected_gpu = best_gpu[0]
        print(f"Selected GPU {self.selected_gpu} with {best_gpu[1]:.1f}GB free memory")
        return self.selected_gpu
    
    def set_device(self, gpu_id: int) -> bool:
        """Set the current CUDA device"""
        try:
            if 0 <= gpu_id < self.device_count:
                torch.cuda.set_device(gpu_id)
                self.selected_gpu = gpu_id
                return True
            return False
        except Exception as e:
            print(f"Error setting GPU device {gpu_id}: {e}")
            return False
    
    def get_device_info(self) -> dict:
        """Get detailed GPU information"""
        gpu_info = []
        
        if torch.cuda.is_available():
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
                free_memory = total_memory - allocated_memory

                gpu_info.append({
                    "gpu_id": i,
                    "name": props.name,
                    "total_memory_gb": f"{total_memory:.1f}",
                    "allocated_memory_gb": f"{allocated_memory:.1f}",
                    "free_memory_gb": f"{free_memory:.1f}",
                    "compute_capability": f"{props.major}.{props.minor}",
                })

        return {
            "cuda_available": torch.cuda.is_available(),
            "device_count": self.device_count,
            "selected_gpu": self.selected_gpu,
            "gpus": gpu_info,
        }
    
    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        try:
            if torch.cuda.is_available() and self.selected_gpu is not None:
                return torch.cuda.memory_allocated(self.selected_gpu) / 1024**3
            return 0.0
        except:
            return 0.0
    
    def get_optimal_device(self) -> Tuple[str, Optional[int]]:
        """Get the optimal device (GPU or CPU) for the current setup"""
        if self.test_cuda_compatibility():
            selected_gpu = self.select_best_gpu()
            if selected_gpu is not None:
                self.set_device(selected_gpu)
                return f"cuda:{selected_gpu}", selected_gpu
            else:
                return "cpu", None
        else:
            return "cpu", None
