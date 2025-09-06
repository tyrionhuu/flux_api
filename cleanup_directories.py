#!/usr/bin/env python3
"""
Standalone script for manual directory cleanup.
Run this script to manually clean up directories and maintain size limits.
"""

import argparse
import sys
from pathlib import Path

import loguru

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.cleanup_settings import (GENERATED_IMAGES_SIZE_LIMIT_GB,
                                     UPLOADS_SIZE_LIMIT_GB)
from utils.directory_cleanup import DirectoryCleanup

logger = loguru.logger


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

    logger.info("Directory cleanup utility")
    logger.info(f"Generated images limit: {args.generated_limit}GB")
    logger.info(f"Uploads limit: {args.uploads_limit}GB")

    try:
        # Initialize cleanup manager
        cleanup_manager = DirectoryCleanup(
            generated_images_limit_gb=args.generated_limit,
            uploads_limit_gb=args.uploads_limit,
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
                    files = cleanup_manager.get_file_info(
                        cleanup_manager.generated_images_dir
                        if dir_name == "generated_images"
                        else cleanup_manager.uploads_dir
                    )
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
