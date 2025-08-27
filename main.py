"""
Main FastAPI application for the FP4 FLUX API (Port 8002)
"""

import logging
import os
import time
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import get_model_manager, router
from config.settings import (API_DESCRIPTION, API_TITLE, API_VERSION,
                             FP4_API_PORT)
from utils.cleanup_service import start_cleanup_service, stop_cleanup_service

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)

# Log current working directory and absolute paths for debugging
current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
generated_images_abs = os.path.abspath("generated_images")
logging.info(f"Current working directory: {current_dir}")
logging.info(f"Script directory: {script_dir}")
logging.info(f"Generated images absolute path: {generated_images_abs}")

# Configure logging (base setup; uvicorn log_config below ensures file logging too)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/flux_api.log")],
)

# Configure specific loggers for better error visibility
logging.getLogger("api.routes").setLevel(logging.INFO)
logging.getLogger("models.flux_model").setLevel(logging.INFO)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    try:
        start_cleanup_service()
        logging.info("FP4 FLUX API started with cleanup service")

        # Auto-load the FLUX model
        logging.info("Auto-loading FLUX model...")

        model_manager = get_model_manager()

        if model_manager.load_model():
            logging.info("FLUX model loaded successfully during startup")
        else:
            logging.error("Failed to load FLUX model during startup")

        time.sleep(2)

        # Verify model is ready
        if model_manager.is_loaded():
            logging.info("FLUX model verified and ready for requests")
        else:
            logging.warning(
                "FLUX model may not be fully ready - some requests may fail"
            )

    except Exception as e:
        logging.error(f"Failed to start services: {e}")

    yield

    # Shutdown
    try:
        stop_cleanup_service()
        logging.info("FP4 FLUX API shutdown, cleanup service stopped")
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


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch any unhandled errors"""
    # Log the full error with traceback
    logging.error(f"Unhandled exception in {request.url}: {exc}")
    logging.error(f"Traceback: {traceback.format_exc()}")

    # Return a proper JSON error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url),
        },
    )


# Add request validation middleware to filter malformed requests
@app.middleware("http")
async def validate_requests(request, call_next):
    """Filter out malformed HTTP requests"""
    try:
        # Check if request has valid headers
        if not request.headers.get("host"):
            # Log and reject malformed requests
            logging.warning(
                f"Rejecting malformed request from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request", "detail": "Missing host header"},
            )

        response = await call_next(request)
        return response
    except Exception as e:
        logging.error(f"Error in request validation middleware: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)},
        )


# Include API routes
app.include_router(router, prefix="")

# Mount static files for frontend
if os.path.exists("frontend/static"):
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Mount generated images directory for downloads
if os.path.exists("generated_images"):
    app.mount(
        "/generated_images",
        StaticFiles(directory="generated_images"),
        name="generated_images",
    )
    logging.info("Mounted generated_images directory for static file serving")

    files = os.listdir("generated_images")
    logging.info(f"Generated images directory contains: {files}")
else:
    logging.warning("generated_images directory not found - downloads may not work")


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
            <head><title>FP4 FLUX API</title></head>
            <body>
                <h1>FP4 FLUX API - Frontend Not Available</h1>
                <p>The frontend files are not found. Please check the frontend directory.</p>
                <p><a href="/docs">Visit API Documentation</a></p>
            </body>
        </html>
        """


# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        model_manager = get_model_manager()
        model_loaded = model_manager.is_loaded()

        return {
            "status": "healthy" if model_loaded else "model_loading",
            "service": "FP4 FLUX API",
            "model_loaded": model_loaded,
            "model_ready": model_loaded,
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "FP4 FLUX API",
            "error": str(e),
            "model_loaded": False,
        }


if __name__ == "__main__":
    # Ensure uvicorn also writes to our file
    LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "level": "INFO",
                "filename": "logs/flux_api_fp4.log",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "api.routes": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "models.flux_model": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "utils.cleanup_service": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "": {"handlers": ["console", "file"], "level": "INFO"},
        },
    }

    uvicorn.run(app, host="0.0.0.0", port=FP4_API_PORT, log_config=LOG_CONFIG)
