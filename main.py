"""
Main FastAPI application for the FP4 Diffusion API (Port 8002)
"""

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.routes import get_model_manager, router
from config.settings import API_DESCRIPTION, API_PORT, API_TITLE, API_VERSION
from utils.cleanup_service import start_cleanup_service, stop_cleanup_service

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    "logs/diffusion_api.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
)

# Create FastAPI app with lifespan context manager
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    try:
        start_cleanup_service()
        logger.info("FP4 Diffusion API started with cleanup service")

        # Auto-load the Diffusion model
        logger.info("Auto-loading Diffusion model...")

        model_manager = get_model_manager()

        if model_manager.load_model():
            logger.info("Diffusion model loaded successfully during startup")
        else:
            logger.error("Failed to load Diffusion model during startup")

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

    yield

    # Shutdown
    try:
        stop_cleanup_service()
        logger.info("FP4 Diffusion API shutdown, cleanup service stopped")
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
            <head><title>FP4 Diffusion API</title></head>
            <body>
                <h1>FP4 Diffusion API - Frontend Not Available</h1>
                <p>The frontend files are not found. Please check the frontend directory.</p>
                <p><a href="/docs">Visit API Documentation</a></p>
            </body>
        </html>
        """


# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FP4 Diffusion API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
