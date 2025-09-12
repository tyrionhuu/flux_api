"""
Main FastAPI application for the FP4 FLUX API (Port 8002)
"""

import argparse
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager

import loguru
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import get_model_manager, router
from config.settings import API_DESCRIPTION, API_TITLE, API_VERSION
from utils.cleanup_service import start_cleanup_service, stop_cleanup_service

# Hugging Face token setup
if "HUGGINGFACE_HUB_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"]
    try:
        from huggingface_hub import login

        login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
        print("✅ Hugging Face token configured successfully")
    except Exception as e:
        print(f"⚠️  Failed to configure Hugging Face token: {e}")
else:
    print("⚠️  HUGGINGFACE_HUB_TOKEN not set, model downloads may fail")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)

# Log current working directory and absolute paths for debugging
current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
generated_images_abs = os.path.abspath("generated_images")

logger = loguru.logger

logger.info(f"Current working directory: {current_dir}")
logger.info(f"Script directory: {script_dir}")
logger.info(f"Generated images absolute path: {generated_images_abs}")

# Configure logging (base setup; uvicorn log_config below ensures file logging too)
# Remove any existing handlers first
logger.remove()

# Add console output
logger.add(sys.stdout, level="INFO", format="{time} - {name} - {level} - {message}")

# Add main log file for all logs
logger.add(
    "logs/flux_api.log", level="INFO", format="{time} - {name} - {level} - {message}"
)

# Configure specific loggers for better error visibility (optional separate files)
logger.add(
    "logs/api_routes.log",
    level="INFO",
    format="{time} - {name} - {level} - {message}",
    filter=lambda record: "api.routes" in record["name"],
)
logger.add(
    "logs/models_flux_model.log",
    level="INFO",
    format="{time} - {name} - {level} - {message}",
    filter=lambda record: "models.flux_model" in record["name"],
)
logger.add(
    "logs/utils_cleanup_service.log",
    level="INFO",
    format="{time} - {name} - {level} - {message}",
    filter=lambda record: "utils.cleanup_service" in record["name"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    try:
        start_cleanup_service()
        logger.info("FP4 FLUX API started with cleanup service")

        # Auto-load the FLUX model
        logger.info("Auto-loading FLUX model...")

        model_manager = get_model_manager()

        if model_manager.load_model():
            logger.info("FLUX model loaded successfully during startup")
        else:
            logger.error("Failed to load FLUX model during startup")

        time.sleep(2)

    except Exception as e:
        logger.error(f"Failed to start services: {e}")

    yield

    # Shutdown
    try:
        stop_cleanup_service()
        logger.info("FP4 FLUX API shutdown, cleanup service stopped")
    except Exception as e:
        logger.error(f"Error stopping cleanup service: {e}")


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
    logger.error(f"Unhandled exception in {request.url}: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")

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
            logger.warning(
                f"Rejecting malformed request from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request", "detail": "Missing host header"},
            )

        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error in request validation middleware: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)},
        )


# Include API routes
app.include_router(router, prefix="")

# Mount generated images directory for downloads
if os.path.exists("generated_images"):
    app.mount(
        "/generated_images",
        StaticFiles(directory="generated_images"),
        name="generated_images",
    )
    logger.info("Mounted generated_images directory for static file serving")
else:
    logger.warning("generated_images directory not found - downloads may not work")





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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FLUX API Service")
    parser.add_argument(
        "--port", type=int, default=9200, help="API port number (default: 9200)"
    )
    args = parser.parse_args()

    # Store port in app state for access by routes
    app.state.port = args.port

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
                "filename": "logs/flux_api.log",
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

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_config=LOG_CONFIG)
