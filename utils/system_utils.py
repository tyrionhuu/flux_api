"""
System utilities for the Diffusion API
"""

from typing import Tuple

import psutil


def get_system_memory() -> Tuple[float, float]:
    """Get system memory usage in GB"""
    try:
        memory = psutil.virtual_memory()
        return memory.used / 1024**3, memory.total / 1024**3
    except:
        return 0.0, 0.0


def get_system_info() -> dict:
    """Get comprehensive system information"""
    try:
        memory_used, memory_total = get_system_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        return {
            "cpu_percent": cpu_percent,
            "memory_used_gb": memory_used,
            "memory_total_gb": memory_total,
            "memory_percent": (
                (memory_used / memory_total) * 100 if memory_total > 0 else 0
            ),
        }
    except Exception as e:
        print(f"Error getting system info: {e}")
        return {
            "cpu_percent": 0.0,
            "memory_used_gb": 0.0,
            "memory_total_gb": 0.0,
            "memory_percent": 0.0,
        }
