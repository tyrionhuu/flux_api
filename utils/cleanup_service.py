"""
Background cleanup service for automatic directory maintenance.
"""

import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import loguru

from config.cleanup_settings import (CLEANUP_INTERVAL_SECONDS,
                                     CLEANUP_ON_GENERATION, CLEANUP_ON_UPLOAD)

from .cleanup_directories import DirectoryCleanup

logger = loguru.logger


class CleanupService:
    """Background service for automatic directory cleanup."""

    def __init__(self, cleanup_manager: Optional[DirectoryCleanup] = None):
        """
        Initialize the cleanup service.

        Args:
            cleanup_manager: Optional DirectoryCleanup instance
        """
        self.cleanup_manager = cleanup_manager or DirectoryCleanup()
        self.running = False
        self.cleanup_thread: Optional[threading.Thread] = None
        self.last_cleanup = None
        self.cleanup_count = 0

        # Ensure logs directory exists
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        logger.info("Cleanup service initialized")

    def start(self):
        """Start the background cleanup service."""
        if self.running:
            logger.warning("Cleanup service is already running")
            return

        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Cleanup service started")

    def stop(self):
        """Stop the background cleanup service."""
        if not self.running:
            logger.warning("Cleanup service is not running")
            return

        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        logger.info("Cleanup service stopped")

    def _cleanup_loop(self):
        """Main cleanup loop that runs in background thread."""
        while self.running:
            try:
                # Wait for next cleanup interval
                time.sleep(CLEANUP_INTERVAL_SECONDS)

                if self.running:
                    logger.info("Running scheduled cleanup...")
                    self._perform_cleanup("scheduled")

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def _perform_cleanup(self, trigger: str):
        """Perform the actual cleanup operation."""
        try:
            start_time = time.time()
            results = self.cleanup_manager.cleanup_all()

            cleanup_time = time.time() - start_time
            self.last_cleanup = datetime.now()
            self.cleanup_count += 1

            logger.info(
                f"Cleanup completed in {cleanup_time:.2f}s (trigger: {trigger}): {results}"
            )

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def trigger_cleanup(self, reason: str = "manual"):
        """Manually trigger a cleanup operation."""
        logger.info(f"Manual cleanup triggered: {reason}")
        self._perform_cleanup(reason)

    def cleanup_after_upload(self):
        """Trigger cleanup after file upload if enabled."""
        if CLEANUP_ON_UPLOAD:
            logger.debug("Triggering cleanup after upload")
            self.trigger_cleanup("upload")

    def cleanup_after_generation(self):
        """Trigger cleanup after image generation if enabled."""
        if CLEANUP_ON_GENERATION:
            logger.debug("Triggering cleanup after generation")
            self.trigger_cleanup("generation")

    def get_status(self) -> dict:
        """Get current service status."""
        return {
            "running": self.running,
            "last_cleanup": (
                self.last_cleanup.isoformat() if self.last_cleanup else None
            ),
            "cleanup_count": self.cleanup_count,
            "next_scheduled": self._get_next_scheduled_cleanup(),
            "directory_status": self.cleanup_manager.get_status(),
        }

    def _get_next_scheduled_cleanup(self) -> Optional[str]:
        """Calculate when the next scheduled cleanup will occur."""
        if not self.last_cleanup:
            return None

        next_cleanup = self.last_cleanup + timedelta(seconds=CLEANUP_INTERVAL_SECONDS)
        return next_cleanup.isoformat()


# Global cleanup service instance
_cleanup_service: Optional[CleanupService] = None


def get_cleanup_service() -> CleanupService:
    """Get or create the global cleanup service instance."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = CleanupService()
    return _cleanup_service


def start_cleanup_service():
    """Start the global cleanup service."""
    service = get_cleanup_service()
    service.start()


def stop_cleanup_service():
    """Stop the global cleanup service."""
    global _cleanup_service
    if _cleanup_service:
        _cleanup_service.stop()
        _cleanup_service = None


def trigger_cleanup(reason: str = "manual"):
    """Trigger cleanup using the global service."""
    service = get_cleanup_service()
    service.trigger_cleanup(reason)


def cleanup_after_upload():
    """Trigger cleanup after upload using the global service."""
    service = get_cleanup_service()
    service.cleanup_after_upload()


def cleanup_after_generation():
    """Trigger cleanup after generation using the global service."""
    service = get_cleanup_service()
    service.cleanup_after_generation()


if __name__ == "__main__":
    service = CleanupService()

    print("Starting cleanup service...")
    service.start()

    try:
        # Run for a few minutes to test
        print("Service running for 5 minutes...")
        time.sleep(300)
    except KeyboardInterrupt:
        print("\nStopping service...")
    finally:
        service.stop()
        print("Service stopped")
