"""
System utilities for the FLUX API
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
