#!/usr/bin/env python3
"""
Standalone script for manual directory cleanup.
Run this script to manually clean up directories and maintain size limits.
"""

import argparse
import loguru
import os
import sys
from pathlib import Path
from typing import List, Tuple

from config.cleanup_settings import (GENERATED_IMAGES_SIZE_LIMIT_GB,
                                     UPLOADS_SIZE_LIMIT_GB)

logger = loguru.logger


class DirectoryCleanup:
    """
    Manages automatic cleanup of directories to maintain size limits.
    """

    def __init__(
        self,
        generated_images_dir: str = "generated_images",
        uploads_dir: str = "uploads/lora_files",
        uploads_images_dir: str = "uploads/images",
        generated_images_limit_gb: float = 1.0,
        uploads_limit_gb: float = 2.0,
        uploads_images_limit_gb: float = 1.0,
    ):
        """
        Initialize the directory cleanup manager.

        Args:
            generated_images_dir: Path to generated images directory
            uploads_dir: Path to uploads directory
            uploads_images_dir: Path to uploads images directory
            generated_images_limit_gb: Size limit for generated images in GB
            uploads_limit_gb: Size limit for uploads in GB
            uploads_images_limit_gb: Size limit for uploads images in GB
        """
        self.generated_images_dir = Path(generated_images_dir)
        self.uploads_dir = Path(uploads_dir)
        self.uploads_images_dir = Path(uploads_images_dir)
        self.generated_images_limit_bytes = int(
            generated_images_limit_gb * 1024 * 1024 * 1024
        )
        self.uploads_limit_bytes = int(uploads_limit_gb * 1024 * 1024 * 1024)
        self.uploads_images_limit_bytes = int(uploads_images_limit_gb * 1024 * 1024 * 1024)

        # Ensure directories exist
        self.generated_images_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_images_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Directory cleanup initialized:")
        logger.info(
            f"  Generated images: {self.generated_images_dir} (limit: {generated_images_limit_gb}GB)"
        )
        logger.info(f"  Uploads: {self.uploads_dir} (limit: {uploads_limit_gb}GB)")
        logger.info(f"  Uploads images: {self.uploads_images_dir} (limit: {uploads_images_limit_gb}GB)")

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
        Clean up all directories to maintain their size limits.

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

        # Clean up uploads images
        results["uploads_images"] = self.cleanup_directory(
            self.uploads_images_dir, self.uploads_images_limit_bytes, "Uploads Images"
        )

        return results

    def get_status(self) -> dict:
        """Get current status of all directories."""
        generated_size = self.get_directory_size(self.generated_images_dir)
        uploads_size = self.get_directory_size(self.uploads_dir)
        uploads_images_size = self.get_directory_size(self.uploads_images_dir)

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
            "uploads_images": {
                "path": str(self.uploads_images_dir),
                "current_size_bytes": uploads_images_size,
                "current_size_gb": uploads_images_size / (1024**3),
                "limit_bytes": self.uploads_images_limit_bytes,
                "limit_gb": self.uploads_images_limit_bytes / (1024**3),
                "usage_percent": (uploads_images_size / self.uploads_images_limit_bytes) * 100,
            },
        }


def cleanup_directories_auto():
    """Convenience function for automatic cleanup."""
    cleanup_manager = DirectoryCleanup()
    return cleanup_manager.cleanup_all()


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean up directories to maintain size limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run cleanup with default settings
  python cleanup_directories.py
  
  # Run cleanup with custom size limits
  python cleanup_directories.py --generated-limit 2.0 --uploads-limit 3.0
  
  # Show status only (no cleanup)
  python cleanup_directories.py --status-only
  
  # Verbose output
  python cleanup_directories.py --verbose
        """,
    )

    parser.add_argument(
        "--generated-limit",
        type=float,
        default=GENERATED_IMAGES_SIZE_LIMIT_GB,
        help=f"Size limit for generated images in GB (default: {GENERATED_IMAGES_SIZE_LIMIT_GB})",
    )

    parser.add_argument(
        "--uploads-limit",
        type=float,
        default=UPLOADS_SIZE_LIMIT_GB,
        help=f"Size limit for uploads in GB (default: {UPLOADS_SIZE_LIMIT_GB})",
    )

    parser.add_argument(
        "--uploads-images-limit",
        type=float,
        default=1.0,
        help="Size limit for uploads images in GB (default: 1.0)",
    )

    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Show directory status without performing cleanup",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting files",
    )

    args = parser.parse_args()

    # Setup logging
    logger = loguru.logger

    logger.info("Directory cleanup utility")
    logger.info(f"Generated images limit: {args.generated_limit}GB")
    logger.info(f"Uploads limit: {args.uploads_limit}GB")
    logger.info(f"Uploads images limit: {args.uploads_images_limit}GB")

    try:
        # Initialize cleanup manager
        cleanup_manager = DirectoryCleanup(
            generated_images_limit_gb=args.generated_limit,
            uploads_limit_gb=args.uploads_limit,
            uploads_images_limit_gb=args.uploads_images_limit,
        )

        # Show current status
        status = cleanup_manager.get_status()
        print("\n" + "=" * 60)
        print("CURRENT DIRECTORY STATUS")
        print("=" * 60)

        for dir_name, info in status.items():
            print(f"\n{dir_name.upper().replace('_', ' ')}:")
            print(f"  Path: {info['path']}")
            print(f"  Current size: {info['current_size_gb']:.2f}GB")
            print(f"  Size limit: {info['limit_gb']:.2f}GB")
            print(f"  Usage: {info['usage_percent']:.1f}%")

            if info["usage_percent"] > 100:
                print(f"  ⚠️  OVER LIMIT by {info['usage_percent'] - 100:.1f}%")
            elif info["usage_percent"] > 80:
                print(f"  ⚠️  APPROACHING LIMIT")
            else:
                print(f"  ✅ UNDER LIMIT")

        if args.status_only:
            print("\nStatus check complete. No cleanup performed.")
            return

        # Check if cleanup is needed
        needs_cleanup = any(info["usage_percent"] > 100 for info in status.values())

        if not needs_cleanup:
            print("\n✅ All directories are within size limits. No cleanup needed.")
            return

        print(f"\n⚠️  Cleanup needed for directories exceeding limits.")

        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No files will be deleted")
            print("Files that would be deleted:")

            # Show what would be deleted
            for dir_name, info in status.items():
                if info["usage_percent"] > 100:
                    print(f"\n{dir_name}:")
                    if dir_name == "generated_images":
                        target_dir = cleanup_manager.generated_images_dir
                    elif dir_name == "uploads":
                        target_dir = cleanup_manager.uploads_dir
                    elif dir_name == "uploads_images":
                        target_dir = cleanup_manager.uploads_images_dir
                    else:
                        continue
                    
                    files = cleanup_manager.get_file_info(target_dir)
                    files.sort(key=lambda x: x[2])  # Sort by modification time

                    current_size = info["current_size_bytes"]
                    limit_size = info["limit_bytes"]
                    to_remove_size = current_size - limit_size
                    removed_size = 0

                    for file_path, file_size, mtime in files:
                        if removed_size >= to_remove_size:
                            break
                        print(
                            f"  Would delete: {file_path.name} ({file_size / (1024**2):.1f}MB)"
                        )
                        removed_size += file_size

            print("\nDry run complete. Use --cleanup to actually perform cleanup.")
            return

        # Perform cleanup
        print("\n🧹 Starting cleanup...")
        results = cleanup_manager.cleanup_all()

        print("\n" + "=" * 60)
        print("CLEANUP RESULTS")
        print("=" * 60)

        for dir_name, files_removed in results.items():
            print(f"\n{dir_name.upper().replace('_', ' ')}:")
            if files_removed > 0:
                print(f"  ✅ Removed {files_removed} files")
            else:
                print(f"  ℹ️  No files removed")

        # Show final status
        print("\n" + "=" * 60)
        print("FINAL STATUS")
        print("=" * 60)

        final_status = cleanup_manager.get_status()
        for dir_name, info in final_status.items():
            print(f"\n{dir_name.upper().replace('_', ' ')}:")
            print(f"  Current size: {info['current_size_gb']:.2f}GB")
            print(f"  Size limit: {info['limit_gb']:.2f}GB")
            print(f"  Usage: {info['usage_percent']:.1f}%")

            if info["usage_percent"] <= 100:
                print(f"  ✅ Now within limits")
            else:
                print(f"  ⚠️  Still over limit")

        print("\n✅ Cleanup complete!")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
