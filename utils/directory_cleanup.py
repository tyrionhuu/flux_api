"""
Directory cleanup utility for maintaining size limits on generated images and uploaded files.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple
import shutil

logger = logging.getLogger(__name__)


class DirectoryCleanup:
    """Manages automatic cleanup of directories to maintain size limits."""

    def __init__(
        self,
        generated_images_dir: str = "generated_images",
        uploads_dir: str = "uploads/lora_files",
        generated_images_limit_gb: float = 1.0,
        uploads_limit_gb: float = 2.0,
    ):
        """
        Initialize the directory cleanup manager.

        Args:
            generated_images_dir: Path to generated images directory
            uploads_dir: Path to uploads directory
            generated_images_limit_gb: Size limit for generated images in GB
            uploads_limit_gb: Size limit for uploads in GB
        """
        self.generated_images_dir = Path(generated_images_dir)
        self.uploads_dir = Path(uploads_dir)
        self.generated_images_limit_bytes = int(
            generated_images_limit_gb * 1024 * 1024 * 1024
        )
        self.uploads_limit_bytes = int(uploads_limit_gb * 1024 * 1024 * 1024)

        # Ensure directories exist
        self.generated_images_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Directory cleanup initialized:")
        logger.info(
            f"  Generated images: {self.generated_images_dir} (limit: {generated_images_limit_gb}GB)"
        )
        logger.info(f"  Uploads: {self.uploads_dir} (limit: {uploads_limit_gb}GB)")

    def get_directory_size(self, directory: Path) -> int:
        """Calculate total size of a directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Error calculating directory size for {directory}: {e}")
        return total_size

    def get_file_info(self, directory: Path) -> List[Tuple[Path, int, float]]:
        """
        Get list of files in directory with their sizes and modification times.

        Returns:
            List of tuples: (file_path, size_bytes, modification_time)
        """
        files = []
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        files.append((file_path, stat.st_size, stat.st_mtime))
                    except OSError as e:
                        logger.warning(f"Could not stat file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")

        return files

    def cleanup_directory(
        self, directory: Path, size_limit_bytes: int, directory_name: str
    ) -> int:
        """
        Clean up directory to stay under size limit by removing oldest files first.

        Args:
            directory: Directory to clean up
            size_limit_bytes: Size limit in bytes
            directory_name: Human-readable name for logging

        Returns:
            Number of files removed
        """
        current_size = self.get_directory_size(directory)
        if current_size <= size_limit_bytes:
            logger.debug(
                f"{directory_name} size ({current_size / (1024**3):.2f}GB) is under limit ({size_limit_bytes / (1024**3):.2f}GB)"
            )
            return 0

        logger.info(
            f"{directory_name} size ({current_size / (1024**3):.2f}GB) exceeds limit ({size_limit_bytes / (1024**3):.2f}GB), cleaning up..."
        )

        files = self.get_file_info(directory)
        if not files:
            logger.warning(f"No files found in {directory_name}")
            return 0

        # Sort files by modification time (oldest first)
        files.sort(key=lambda x: x[2])

        files_removed = 0
        total_removed_size = 0

        for file_path, file_size, _ in files:
            if current_size - total_removed_size <= size_limit_bytes:
                break

            try:
                # Remove the file
                file_path.unlink()
                total_removed_size += file_size
                files_removed += 1
                logger.info(
                    f"Removed old file: {file_path.name} ({file_size / (1024**2):.1f}MB)"
                )
            except OSError as e:
                logger.error(f"Failed to remove file {file_path}: {e}")

        if files_removed > 0:
            final_size = self.get_directory_size(directory)
            logger.info(
                f"{directory_name} cleanup complete: removed {files_removed} files, "
                f"freed {total_removed_size / (1024**3):.2f}GB, "
                f"new size: {final_size / (1024**3):.2f}GB"
            )
        else:
            logger.warning(f"Could not free enough space in {directory_name}")

        return files_removed

    def cleanup_all(self) -> dict:
        """
        Clean up both directories to maintain their size limits.

        Returns:
            Dictionary with cleanup results
        """
        results = {}

        # Clean up generated images
        results["generated_images"] = self.cleanup_directory(
            self.generated_images_dir,
            self.generated_images_limit_bytes,
            "Generated Images",
        )

        # Clean up uploads
        results["uploads"] = self.cleanup_directory(
            self.uploads_dir, self.uploads_limit_bytes, "Uploads"
        )

        return results

    def get_status(self) -> dict:
        """Get current status of both directories."""
        generated_size = self.get_directory_size(self.generated_images_dir)
        uploads_size = self.get_directory_size(self.uploads_dir)

        return {
            "generated_images": {
                "path": str(self.generated_images_dir),
                "current_size_bytes": generated_size,
                "current_size_gb": generated_size / (1024**3),
                "limit_bytes": self.generated_images_limit_bytes,
                "limit_gb": self.generated_images_limit_bytes / (1024**3),
                "usage_percent": (generated_size / self.generated_images_limit_bytes)
                * 100,
            },
            "uploads": {
                "path": str(self.uploads_dir),
                "current_size_bytes": uploads_size,
                "current_size_gb": uploads_size / (1024**3),
                "limit_bytes": self.uploads_limit_bytes,
                "limit_gb": self.uploads_limit_bytes / (1024**3),
                "usage_percent": (uploads_size / self.uploads_limit_bytes) * 100,
            },
        }


def cleanup_directories_auto():
    """Convenience function for automatic cleanup."""
    cleanup_manager = DirectoryCleanup()
    return cleanup_manager.cleanup_all()


if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the cleanup functionality
    cleanup_manager = DirectoryCleanup()

    print("Current directory status:")
    status = cleanup_manager.get_status()
    for dir_name, info in status.items():
        print(
            f"  {dir_name}: {info['current_size_gb']:.2f}GB / {info['limit_gb']:.2f}GB ({info['usage_percent']:.1f}%)"
        )

    print("\nRunning cleanup...")
    results = cleanup_manager.cleanup_all()

    print(f"Cleanup results: {results}")

    print("\nStatus after cleanup:")
    status = cleanup_manager.get_status()
    for dir_name, info in status.items():
        print(
            f"  {dir_name}: {info['current_size_gb']:.2f}GB / {info['limit_gb']:.2f}GB ({info['usage_percent']:.1f}%)"
        )
