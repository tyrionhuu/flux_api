"""
Queue manager for handling concurrent requests in the FLUX API
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from config.settings import DEFAULT_GUIDANCE_SCALE, INFERENCE_STEPS

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueRequest:
    """Represents a request in the queue"""

    id: str
    prompt: str
    status: RequestStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    priority: int = 0  # Higher number = higher priority
    # LoRA configuration
    lora_name: Optional[str] = None  # For backward compatibility
    lora_weight: float = 1.0  # For backward compatibility
    loras: Optional[list[dict]] = None  # New multiple LoRA support
    # Generation parameters
    num_inference_steps: int = INFERENCE_STEPS  # Fixed value
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE  # Default value, can be overridden
    width: int = 512
    height: int = 512
    seed: Optional[int] = None


class QueueManager:
    """Manages request queuing and processing"""

    def __init__(self, max_concurrent: int = 2, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.active_requests: Dict[str, QueueRequest] = {}
        self.completed_requests: Dict[str, QueueRequest] = {}
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the queue processor"""
        if self.is_running:
            return

        self.is_running = True
        self._task = asyncio.create_task(self._process_queue())
        logger.info(
            f"Queue manager started with max {self.max_concurrent} concurrent requests"
        )

    async def stop(self):
        """Stop the queue processor"""
        if not self.is_running:
            return

        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Queue manager stopped")

    async def submit_request(
        self,
        prompt: str,
        lora_name: Optional[str] = None,
        lora_weight: float = 1.0,
        loras: Optional[list[dict]] = None,  # New multiple LoRA support
        priority: int = 0,
        num_inference_steps: int = INFERENCE_STEPS,  # Fixed value
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,  # Default value, can be overridden
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> str:
        """Submit a new request to the queue"""
        if self.queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Queue is full")

        request_id = str(uuid.uuid4())
        request = QueueRequest(
            id=request_id,
            prompt=prompt,
            status=RequestStatus.PENDING,
            created_at=time.time(),
            lora_name=lora_name,
            lora_weight=lora_weight,
            loras=loras,
            priority=priority,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )

        await self.queue.put((priority, request))
        logger.info(f"Request {request_id} submitted to queue (priority: {priority})")
        return request_id

    async def get_request_status(self, request_id: str) -> Optional[QueueRequest]:
        """Get the status of a specific request"""
        # Check active requests first
        if request_id in self.active_requests:
            return self.active_requests[request_id]

        # Check completed requests
        if request_id in self.completed_requests:
            return self.completed_requests[request_id]

        return None

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request"""
        # Find the request in the queue
        queue_items = []
        found = False

        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if item[1].id == request_id and item[1].status == RequestStatus.PENDING:
                    item[1].status = RequestStatus.CANCELLED
                    found = True
                    logger.info(f"Request {request_id} cancelled")
                else:
                    queue_items.append(item)
            except asyncio.QueueEmpty:
                break

        # Put items back in the queue
        for item in queue_items:
            await self.queue.put(item)

        return found

    async def _process_queue(self):
        """Process the queue continuously"""
        while self.is_running:
            try:
                # Get next request from queue
                priority, request = await self.queue.get()

                if request.status == RequestStatus.CANCELLED:
                    continue

                # Process the request
                await self._process_request(request)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")

    async def _process_request(self, request: QueueRequest):
        """Process a single request"""
        async with self.processing_semaphore:
            try:
                request.status = RequestStatus.PROCESSING
                request.started_at = time.time()
                self.active_requests[request.id] = request

                logger.info(f"Processing request {request.id}: {request.prompt}")

                # Here we would call the actual image generation
                # For now, we'll simulate processing
                await asyncio.sleep(1)  # Simulate processing time

                # Mark as completed
                request.status = RequestStatus.COMPLETED
                request.completed_at = time.time()
                request.result = {"message": "Request processed successfully"}

                # Move to completed requests
                self.completed_requests[request.id] = request
                del self.active_requests[request.id]

                logger.info(f"Request {request.id} completed successfully")

            except Exception as e:
                request.status = RequestStatus.FAILED
                request.completed_at = time.time()
                request.error = str(e)

                # Move to completed requests
                self.completed_requests[request.id] = request
                if request.id in self.active_requests:
                    del self.active_requests[request.id]

                logger.error(f"Request {request.id} failed: {e}")

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        return {
            "queue_size": self.queue.qsize(),
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
        }
