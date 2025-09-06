"""
Async task handler for background operations
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Global thread pool for background tasks
_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="async_task")


def run_async_task(func: Callable, *args, **kwargs) -> None:
    """
    Run a function asynchronously in the background (fire and forget)
    
    Args:
        func: Function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    """
    def wrapper():
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in async task {func.__name__}: {str(e)}")
    
    # Submit task to executor
    future = _executor.submit(wrapper)
    
    # Add a callback to log any exceptions that might occur
    def log_exception(fut):
        try:
            fut.result()
        except Exception as e:
            logger.error(f"Unhandled exception in async task: {str(e)}")
    
    future.add_done_callback(log_exception)
    

def shutdown_executor():
    """
    Shutdown the thread pool executor gracefully
    Call this during application shutdown
    """
    _executor.shutdown(wait=True)
    logger.info("Async task executor shut down")