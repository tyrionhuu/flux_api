"""
Configuration settings for automatic directory cleanup.
"""

# Directory cleanup settings
CLEANUP_ENABLED = True

# Size limits in GB
GENERATED_IMAGES_SIZE_LIMIT_GB = 1.0
UPLOADS_SIZE_LIMIT_GB = 2.0

# Cleanup intervals (in seconds)
CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour
CLEANUP_ON_UPLOAD = True  # Clean up after each upload
CLEANUP_ON_GENERATION = True  # Clean up after each image generation

# File retention settings
MIN_FILE_AGE_HOURS = 1  # Don't delete files newer than this
PRIORITY_FILE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
]  # Keep these file types longer

# Logging settings
CLEANUP_LOG_LEVEL = "INFO"
CLEANUP_LOG_FILE = "logs/cleanup.log"
