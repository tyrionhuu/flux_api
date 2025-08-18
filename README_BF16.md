# BF16 FLUX API Service

This document describes how to use the BF16 FLUX API service that runs alongside the existing FP4 service.

## Overview

The system now supports two FLUX models running in parallel:

- **FP4 Model (Port 8000)**: The original quantized FLUX.1-dev model using Nunchaku
- **BF16 Model (Port 8001)**: The full-precision FLUX.1-dev model in bfloat16 format

## Architecture

The BF16 service is designed to avoid code duplication by:

1. **Extending the base model manager**: `BF16FluxModelManager` inherits from `FluxModelManager`
2. **Reusing API logic**: The BF16 routes import and adapt existing functionality
3. **Separate configuration**: BF16-specific settings in `config/bf16_settings.py`
4. **Independent services**: Each model runs on its own port with separate processes

## Files Added

### Configuration
- `config/bf16_settings.py` - BF16-specific configuration

### Model Management
- `models/bf16_flux_model.py` - BF16 model manager (extends base)

### API Routes
- `api/bf16_routes.py` - BF16 API endpoints (reuses existing logic)

### Service Files
- `main_bf16.py` - BF16 FastAPI application
- `start_bf16_service.py` - BF16 service starter script
- `start_bf16_api.sh` - BF16 service shell script

### Combined Management
- `start_both_services.py` - Script to start both services simultaneously

## Usage

### Starting Individual Services

#### FP4 Service (Port 8000)
```bash
# Using Python script
python start_service.py

# Using shell script
./start_api.sh
```

#### BF16 Service (Port 8001)
```bash
# Using Python script
python start_bf16_service.py

# Using shell script
./start_bf16_api.sh
```

### Starting Both Services Simultaneously

```bash
python start_both_services.py
```

### Service URLs

- **FP4 API**: http://localhost:8000
- **BF16 API**: http://localhost:8001
- **FP4 Health**: http://localhost:8000/health
- **BF16 Health**: http://localhost:8001/health
- **FP4 Docs**: http://localhost:8000/docs
- **BF16 Docs**: http://localhost:8001/docs

## API Endpoints

Both services provide identical endpoints:

- `GET /` - Service status and available endpoints
- `POST /generate` - Generate images with optional LoRA support
- `POST /load-model` - Load the respective model
- `GET /model-status` - Get model status and GPU information
- `POST /apply-lora` - Apply LoRA to the model
- `POST /remove-lora` - Remove LoRA from the model
- `GET /lora-status` - Get current LoRA status
- `GET /loras` - Get available LoRAs (placeholder)
- `GET /health` - Health check endpoint

## Model Differences

### FP4 Model (Port 8000)
- Uses Nunchaku quantization for memory efficiency
- Lower VRAM usage
- Slightly lower quality due to quantization
- Faster inference

### BF16 Model (Port 8001)
- Full-precision bfloat16 format
- Higher VRAM usage
- Better quality output
- Slower inference

## Memory Requirements

- **FP4 Model**: ~8-12 GB VRAM
- **BF16 Model**: ~16-24 GB VRAM
- **Both Models**: ~24-36 GB VRAM (depending on GPU)

## Example Usage

### Generate Image with FP4 Model
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape painting",
    "num_inference_steps": 25,
    "guidance_scale": 3.5
  }'
```

### Generate Image with BF16 Model
```bash
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape painting",
    "num_inference_steps": 25,
    "guidance_scale": 3.5
  }'
```

### Apply LoRA to BF16 Model
```bash
curl -X POST "http://localhost:8001/apply-lora" \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "username/model-name",
    "lora_weight": 0.8
  }'
```

## Monitoring

### Check Service Status
```bash
# FP4 service
curl http://localhost:8000/model-status

# BF16 service
curl http://localhost:8001/model-status
```

### Health Checks
```bash
# FP4 service
curl http://localhost:8000/health

# BF16 service
curl http://localhost:8001/health
```

## Troubleshooting

### GPU Memory Issues
If you encounter CUDA out of memory errors:

```bash
# Check GPU status and available memory
python check_gpu_status.py

# Check what's using GPU memory
nvidia-smi

# Kill processes using GPU memory
sudo kill -9 <PID>
```

### Port Conflicts
If you encounter port conflicts:

```bash
# Check what's using the ports
lsof -i :8000
lsof -i :8001

# Kill processes using the ports
sudo kill -9 <PID>
```

### Memory Issues
If you run out of VRAM:

1. Stop one of the services
2. Use only the FP4 model for lower memory usage
3. Ensure you have sufficient GPU memory for both models

### Service Won't Start
1. Check if `flux_env` virtual environment exists
2. Verify all dependencies are installed
3. Check logs for specific error messages
4. Ensure ports are not in use by other processes

## Development

### Adding New Features
When adding new features:

1. **Add to base classes first**: Implement in `FluxModelManager` or base routes
2. **Extend for BF16**: Override or extend in BF16-specific classes
3. **Maintain consistency**: Keep both services in sync

### Testing
Test both services independently:

```bash
# Test FP4 service
python -m pytest tests/ -v

# Test BF16 service
python -m pytest tests/ -v --port 8001
```

## Performance Considerations

- **GPU Memory**: Both models load into GPU memory simultaneously
- **Inference Speed**: BF16 model is slower but higher quality
- **Batch Processing**: Consider using one service for batch jobs
- **Load Balancing**: Distribute requests between services based on quality vs. speed requirements

## GPU Selection

The system now automatically selects the best available GPU for each model:

- **FP4 Model**: Requires at least 8GB free GPU memory
- **BF16 Model**: Requires at least 16GB free GPU memory
- **Smart Selection**: Automatically finds GPUs with sufficient memory
- **Conflict Avoidance**: Prevents loading models on already-occupied GPUs

### GPU Assignment Strategy

The system now uses **dynamic GPU selection** to automatically find and assign the best available GPUs:

- **Automatic Selection**: Services automatically find GPUs with sufficient memory
- **Conflict Avoidance**: No two services can use the same GPU
- **Memory Requirements**: 
  - FP4 Model: 8GB+ free memory required
  - BF16 Model: 16GB+ free memory required
- **Optimal Assignment**: System assigns largest models to GPUs with most free memory first

#### How It Works

1. **Service Startup**: When a service starts, it requests a GPU with sufficient memory
2. **GPU Selection**: System finds the best available GPU and assigns it exclusively to that service
3. **Environment Setup**: Sets `CUDA_VISIBLE_DEVICES` to restrict PyTorch to only see the assigned GPU
4. **Conflict Prevention**: Once assigned, a GPU cannot be used by other services until released

#### Example Assignment

```
GPU 0: Reserved for other processes (FP4 model already loaded)
GPU 1: Automatically assigned to FP4 FLUX Service (Port 8000)
GPU 2: Automatically assigned to BF16 FLUX Service (Port 8001)
GPUs 3-7: Available for other tasks
```

The actual GPU numbers may vary depending on system availability and memory requirements.

### Check GPU Status
```bash
# Run the GPU status checker
python check_gpu_status.py

# This will show:
# - Available GPUs and their memory
# - Which models can run on which GPUs
# - Recommendations for optimal setup
```

## Future Enhancements

- **Model switching**: Dynamic model selection based on request parameters
- **Quality presets**: Predefined configurations for different use cases
- **Resource management**: Automatic GPU memory management
- **Service discovery**: Health monitoring and automatic failover
