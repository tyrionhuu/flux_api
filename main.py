"""
Main FastAPI application for the Diffusion API (Backend Only)
"""

import argparse
import os
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import get_model_manager, router
from config.settings import API_DESCRIPTION, API_PORT, API_TITLE, API_VERSION
from utils.cleanup_service import start_cleanup_service, stop_cleanup_service
from utils.lora_fusion import LoRAFusionConfig, apply_startup_lora

# Ensure logs and images directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
)
logger.add(
    "logs/diffusion_api.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    try:
        start_cleanup_service()
        logger.info("Diffusion API started with cleanup service")

        # Parse LoRA fusion configuration
        lora_config = LoRAFusionConfig()
        if not lora_config.parse_from_env():
            logger.error("Failed to parse LoRA fusion configuration")
            raise RuntimeError("Invalid LoRA fusion configuration")

        # Override with command line arguments from app.state if available
        if hasattr(app.state, "fusion_mode") and app.state.fusion_mode:
            logger.info("Fusion mode enabled via command line argument")

            # Set LoRA from command line arguments
            lora_name_provided = hasattr(app.state, "lora_name") and app.state.lora_name
            if lora_name_provided:
                # Strip whitespace from lora_name to handle edge cases
                lora_name_stripped = app.state.lora_name.strip()
                if lora_name_stripped:
                    lora_config.fusion_mode = True
                    lora_config.lora_name = lora_name_stripped
                    lora_config.lora_weight = getattr(app.state, "lora_weight", 1.0)
                    lora_config.loras_config = []  # Clear any env-based config
                    logger.info(
                        f"Using LoRA from command line: {lora_config.lora_name} (weight: {lora_config.lora_weight})"
                    )

                    # Re-validate with new configuration
                    if not lora_config._validate_config():
                        logger.error(
                            "Failed to validate LoRA configuration from command line"
                        )
                        raise RuntimeError(
                            "Invalid LoRA configuration from command line"
                        )
                else:
                    logger.info(
                        "LoRA name is empty after stripping whitespace, fusion mode will not be enabled"
                    )
            else:
                logger.info(
                    "Fusion mode requested but no LoRA name provided, fusion mode will not be enabled"
                )

        # Auto-load the Diffusion model
        model_type = os.environ.get("MODEL_TYPE", "flux").lower()
        logger.info(f"Auto-loading Diffusion model (type: {model_type})...")

        model_manager = get_model_manager()

        if model_manager.load_model(model_type):
            logger.info(
                f"Diffusion model ({model_type}) loaded successfully during startup"
            )

            # Apply LoRA fusion if configured
            if lora_config.is_fusion_mode_enabled():
                logger.info("Applying LoRA fusion...")
                if apply_startup_lora(model_manager, lora_config):
                    logger.info("LoRA fusion completed successfully")
                else:
                    logger.error("LoRA fusion failed")
                    raise RuntimeError("LoRA fusion failed")
            else:
                logger.info("No LoRA fusion configured")
        else:
            logger.error(f"Failed to load Diffusion model ({model_type}) during startup")
            raise RuntimeError("Failed to load Diffusion model")

        time.sleep(2)

        # Verify model is ready
        if model_manager.is_loaded():
            logger.info("Diffusion model verified and ready for requests")
        else:
            logger.warning(
                "Diffusion model may not be fully ready - some requests may fail"
            )

    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        raise

    yield

    # Shutdown
    try:
        stop_cleanup_service()
        logger.info("Diffusion API shutdown, cleanup service stopped")
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="")


# Health check endpoint
@app.get("/health")
def health_check():
    """Production-ready health check endpoint"""
    try:
        model_manager = get_model_manager()
        model_loaded = model_manager.is_loaded()

        # Get fusion mode status
        fusion_mode = getattr(model_manager, "fusion_mode", False)
        lora_info = (
            model_manager.get_lora_info()
            if hasattr(model_manager, "get_lora_info")
            else None
        )

        # Get uptime
        uptime = time.time() - getattr(app.state, "start_time", time.time())

        return {
            "status": "healthy" if model_loaded else "model_loading",
            "service": "Diffusion API",
            "version": API_VERSION,
            "model_loaded": model_loaded,
            "model_ready": model_loaded,
            "fusion_mode": fusion_mode,
            "lora_info": lora_info,
            "timestamp": time.time(),
            "uptime": uptime,
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "Diffusion API",
            "version": API_VERSION,
            "error": str(e),
            "model_loaded": False,
            "fusion_mode": False,
            "lora_info": None,
        }


# Kubernetes readiness probe
@app.get("/ready")
def readiness_check():
    """Kubernetes readiness probe"""
    try:
        model_manager = get_model_manager()
        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail="Model not ready")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")


# Kubernetes liveness probe
@app.get("/live")
def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Diffusion API Service")
    parser.add_argument(
        "--port", type=int, default=API_PORT, help=f"API port number (default: {API_PORT})"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="API host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--lora-name", type=str, help="LoRA file path or HF repo for fusion"
    )
    parser.add_argument(
        "--lora-weight", type=float, default=1.0, help="LoRA weight (default: 1.0)"
    )
    parser.add_argument(
        "--loras-config", type=str, help="JSON config for multiple LoRAs"
    )
    parser.add_argument(
        "--fusion-mode", action="store_true", help="Enable LoRA fusion mode"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Maximum workers (default: 1)"
    )
    args = parser.parse_args()

    # Store configuration in app state for access by routes and startup
    app.state.port = args.port
    app.state.start_time = time.time()
    app.state.lora_name = args.lora_name
    app.state.lora_weight = args.lora_weight
    app.state.loras_config = args.loras_config
    app.state.fusion_mode = args.fusion_mode

    # Configure uvicorn logging
    LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
                "filename": "logs/diffusion_api.log",
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
            "": {"handlers": ["console", "file"], "level": "INFO"},
        },
    }

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=LOG_CONFIG,
        workers=args.max_workers,
    )
