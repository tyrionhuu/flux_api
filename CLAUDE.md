# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX API is a dual-model AI image generation service using FastAPI, featuring both FP4 (quantized) and BF16 (full-precision) FLUX models with LoRA support. The project provides RESTful APIs and a modern web interface for image generation.

## Common Development Commands

### Starting Services
```bash
# Start FP4 service (Port 8000, ~8GB VRAM)
./start_fp4_api.sh

# Start BF16 service (Port 8001, ~16GB VRAM)  
./start_bf16_api.sh

# Start with specific GPU(s)
./start_fp4_api.sh -g 0      # Use GPU 0
./start_bf16_api.sh -g 1,2   # Use GPUs 1 and 2

# Start frontend (Port 9000)
cd frontend && python -m http.server 9000
```

### Code Quality Tools
```bash
# Format code with black
black .

# Run linter
ruff check .
ruff check --fix .  # Auto-fix issues
```

### Log Management
```bash
# Check log status
./scripts/manage_logs.sh status

# Clean large logs
./scripts/manage_logs.sh clean

# Follow logs in real-time
./scripts/manage_logs.sh follow
```

### Service Management
```bash
# Install cleanup service
sudo ./scripts/manage_services.sh install flux-cleanup

# Check service status
./scripts/manage_services.sh status flux-cleanup

# View service logs
./scripts/manage_services.sh logs flux-cleanup
```

## High-Level Architecture

### Microservice Architecture
The system consists of two independent FastAPI services that can run on different GPUs:

1. **FP4 Service** (`main_fp4.py:` Port 8000)
   - Quantized FLUX.1-schnell model using nunchaku library
   - Optimized for lower VRAM (~8GB)
   - LoRA merging for multiple LoRA support
   - Model manager: `models/fp4_flux_model.py`

2. **BF16 Service** (`main_bf16.py:` Port 8001)
   - Full-precision FLUX.1-schnell model
   - Higher quality output (~16GB VRAM)
   - Native multi-LoRA support via diffusers
   - Model manager: `models/bf16_flux_model.py`

### Request Processing Flow
1. **API Routes** (`api/fp4_routes.py`, `api/bf16_routes.py`)
   - Handle HTTP requests with Pydantic validation
   - Manage LoRA file uploads to `uploads/lora_files/`
   - Queue requests via `QueueManager`

2. **Queue Management** (`utils/queue_manager.py`)
   - FIFO request processing
   - Prevents GPU memory overflow
   - Configurable queue size per service

3. **GPU Management** (`utils/gpu_manager.py`)
   - Automatic GPU detection and allocation
   - VRAM monitoring and load balancing
   - CUDA_VISIBLE_DEVICES environment handling

4. **Model Managers** 
   - Lazy model loading for memory efficiency
   - LoRA weight application and management
   - Image generation with seed control

5. **Cleanup Service** (`utils/cleanup_service.py`)
   - Automatic temporary file cleanup
   - Configurable retention periods
   - Runs as systemd service or thread

### Frontend Architecture
- **Single-page application** (`frontend/templates/index.html`)
- **ComfyUI-style interface** with vanilla JavaScript
- **Direct API integration** with both model endpoints
- **LoRA file upload** with drag-and-drop support

### Configuration System
- **Port configuration** via environment variables or CLI flags
- **Service-specific settings** in `config/` directory
- **Dynamic port allocation** in startup scripts
- **GPU assignment** through startup script flags

## Key Design Patterns

1. **Service Isolation**: Each model runs as independent service with own port, queue, and GPU allocation
2. **Lazy Loading**: Models loaded only when first request arrives to save memory
3. **Resource Management**: Automatic cleanup, queue limits, and GPU monitoring
4. **Graceful Degradation**: Services continue if cleanup fails, ports auto-freed on startup
5. **Configuration Over Code**: Environment variables and CLI flags for all settings

## Development Workflow

When implementing new features:
1. Check existing patterns in similar files (e.g., routes, models)
2. Maintain separation between FP4 and BF16 implementations
3. Use the established logging pattern with named loggers
4. Follow the existing error handling with proper HTTP status codes
5. Test with both model services if feature affects core functionality

## Important Considerations

- **No test suite**: Manual testing via API calls required
- **GPU-dependent**: Features must handle CUDA availability gracefully  
- **Memory constraints**: Monitor VRAM usage, especially for BF16 model
- **Port conflicts**: Startup scripts handle port cleanup automatically
- **File permissions**: Ensure write access to `uploads/`, `generated_images/`, `logs/`