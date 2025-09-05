"""
Request queue manager for handling synchronous image generation with async queuing
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    """Represents a queued request"""
    request_id: str
    future: asyncio.Future
    created_at: float
    position: int = 0


class RequestQueueManager:
    """
    Manages request queuing for single-worker image generation.
    Ensures only one request is processed at a time while queuing others.
    """
    
    def __init__(
        self,
        max_concurrent: int = 1,
        max_queue_size: int = 100,
        request_timeout: float = 600.0
    ):
        """
        Initialize the request queue manager.
        
        Args:
            max_concurrent: Maximum concurrent requests (default 1 for single GPU)
            max_queue_size: Maximum number of requests to queue
            request_timeout: Maximum time in seconds a request can wait in queue
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        
        # Semaphore to limit concurrent processing
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Queue for pending requests
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Track active and queued requests
        self.active_count = 0
        self.queued_requests: Dict[str, QueuedRequest] = {}
        
        # Metrics
        self.total_processed = 0
        self.total_rejected = 0
        self.total_timeout = 0
        self.start_time = time.time()
        
        logger.info(
            f"RequestQueueManager initialized: max_concurrent={max_concurrent}, "
            f"max_queue_size={max_queue_size}, timeout={request_timeout}s"
        )
    
    async def process_request(
        self,
        process_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Process a request with queuing support.
        
        Args:
            process_func: The synchronous function to call for processing
            *args: Arguments for the process function
            **kwargs: Keyword arguments for the process function
            
        Returns:
            The result from process_func
            
        Raises:
            asyncio.QueueFull: If the queue is full
            asyncio.TimeoutError: If the request times out while waiting
        """
        request_id = str(uuid.uuid4())
        queue_position = 0
        
        # Check if queue is full
        if self.request_queue.qsize() >= self.max_queue_size:
            self.total_rejected += 1
            logger.warning(f"Request {request_id} rejected: queue full")
            raise asyncio.QueueFull("Request queue is full. Please try again later.")
        
        # Create a future for this request
        future = asyncio.Future()
        queued_request = QueuedRequest(
            request_id=request_id,
            future=future,
            created_at=time.time()
        )
        
        # Add to tracking
        self.queued_requests[request_id] = queued_request
        
        try:
            # Get queue position
            queue_position = self.request_queue.qsize()
            queued_request.position = queue_position
            
            if queue_position > 0:
                logger.info(
                    f"Request {request_id} queued at position {queue_position}. "
                    f"Active: {self.active_count}, Queued: {queue_position}"
                )
            
            # Add to queue
            await self.request_queue.put((queued_request, process_func, args, kwargs))
            
            # Start processor if not running
            asyncio.create_task(self._process_queue())
            
            # Wait for result with timeout
            try:
                result = await asyncio.wait_for(future, timeout=self.request_timeout)
                self.total_processed += 1
                return result
            except asyncio.TimeoutError:
                self.total_timeout += 1
                logger.error(f"Request {request_id} timed out after {self.request_timeout}s")
                # Try to cancel the request if it's still in queue
                future.cancel()
                raise asyncio.TimeoutError(
                    f"Request timed out after waiting {self.request_timeout} seconds"
                )
                
        finally:
            # Clean up tracking
            if request_id in self.queued_requests:
                del self.queued_requests[request_id]
    
    async def _process_queue(self):
        """Process queued requests one by one"""
        try:
            # Only process if we can acquire the semaphore
            async with self.processing_semaphore:
                if self.request_queue.empty():
                    return
                
                # Get next request from queue
                queued_request, process_func, args, kwargs = await self.request_queue.get()
                
                # Check if request has already timed out or been cancelled
                if queued_request.future.cancelled():
                    logger.info(f"Skipping cancelled request {queued_request.request_id}")
                    return
                
                # Check if request has been waiting too long
                wait_time = time.time() - queued_request.created_at
                if wait_time > self.request_timeout:
                    logger.warning(
                        f"Request {queued_request.request_id} expired after waiting {wait_time:.1f}s"
                    )
                    queued_request.future.set_exception(
                        asyncio.TimeoutError(f"Request expired after waiting {wait_time:.1f} seconds")
                    )
                    return
                
                self.active_count += 1
                start_time = time.time()
                
                try:
                    logger.info(
                        f"Processing request {queued_request.request_id} "
                        f"(waited {wait_time:.1f}s in queue)"
                    )
                    
                    # Run the synchronous function in a thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,  # Use default thread pool
                        process_func,
                        *args,
                        **kwargs
                    )
                    
                    # Set the result
                    if not queued_request.future.cancelled():
                        queued_request.future.set_result(result)
                    
                    process_time = time.time() - start_time
                    logger.info(
                        f"Request {queued_request.request_id} completed in {process_time:.1f}s"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Request {queued_request.request_id} failed: {e}"
                    )
                    if not queued_request.future.cancelled():
                        queued_request.future.set_exception(e)
                finally:
                    self.active_count -= 1
                    
        except Exception as e:
            logger.error(f"Error in queue processor: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        uptime = time.time() - self.start_time
        return {
            "active_requests": self.active_count,
            "queued_requests": self.request_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "max_concurrent": self.max_concurrent,
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "total_timeout": self.total_timeout,
            "uptime_seconds": uptime,
            "requests_per_minute": (self.total_processed / uptime) * 60 if uptime > 0 else 0
        }
    
    def get_queue_position(self, request_id: str) -> Optional[int]:
        """Get the current queue position for a request"""
        if request_id in self.queued_requests:
            return self.queued_requests[request_id].position
        return None