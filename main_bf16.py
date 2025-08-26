"""
Main FastAPI application for the BF16 FLUX API (Port 8001)
"""

import logging
import os
import torch

# Set PyTorch thread limits BEFORE any other imports or operations
# This prevents thread oversubscription when running multiple instances
num_cores = os.cpu_count() or 8
threads_per_instance = max(1, num_cores // 4)  # Use 1/4 of cores per instance
torch.set_num_threads(threads_per_instance)

# Also set inter-op threads to prevent thread explosion
torch.set_num_interop_threads(max(1, threads_per_instance // 2))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from api.bf16_routes import router
from config.bf16_settings import API_TITLE, API_DESCRIPTION, API_VERSION, BF16_API_PORT
from utils.cleanup_service import start_cleanup_service, stop_cleanup_service

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/flux_api_bf16.log")],
)

# Log PyTorch thread configuration
logger = logging.getLogger(__name__)
logger.info(f"System CPU cores: {num_cores}")
logger.info(f"PyTorch threads per instance: {torch.get_num_threads()}")
logger.info(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")

# Configure specific loggers for better error visibility
logging.getLogger("api.bf16_routes").setLevel(logging.INFO)
logging.getLogger("models.bf16_flux_model").setLevel(logging.INFO)
logging.getLogger("utils.cleanup_service").setLevel(logging.INFO)

# Add a single enhanced console handler for better formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


# Create a custom formatter that adds emojis and better structure
class EnhancedFormatter(logging.Formatter):
    def format(self, record):
        # No emojis - just clean formatting
        return super().format(record)


# Apply the enhanced formatter
enhanced_formatter = EnhancedFormatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(enhanced_formatter)

# Get root logger and replace the default stream handler
root_logger = logging.getLogger()
# Remove existing stream handlers
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, logging.FileHandler
    ):
        root_logger.removeHandler(handler)
# Add our enhanced handler
root_logger.addHandler(console_handler)

# Create FastAPI app with lifespan context manager
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    try:
        start_cleanup_service()
        logging.info("BF16 FLUX API started with cleanup service")
    except Exception as e:
        logging.error(f"Failed to start cleanup service: {e}")

    yield

    # Shutdown
    try:
        stop_cleanup_service()
        logging.info("BF16 FLUX API shutdown, cleanup service stopped")
    except Exception as e:
        logging.error(f"Error stopping cleanup service: {e}")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="")

# Mount static files for frontend
if os.path.exists("frontend/static"):
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/ui", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the ComfyUI-style frontend"""
    frontend_path = "frontend/templates/index.html"
    if os.path.exists(frontend_path):
        with open(frontend_path, "r") as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>BF16 FLUX API</title></head>
            <body>
                <h1>BF16 FLUX API - Frontend Not Available</h1>
                <p>The frontend files are not found. Please check the frontend directory.</p>
                <p><a href="/docs">Visit API Documentation</a></p>
            </body>
        </html>
        """


# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BF16 FLUX API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=BF16_API_PORT)
