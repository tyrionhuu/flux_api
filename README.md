# FLUX API - Dual Model Service

AI image generation API with CUDA support, offering both FP4 (quantized) and BF16 (full-precision) FLUX models with automatic multi-GPU load balancing.

## 🚀 Quick Start

### Single Service Startup

#### FP4 Service (Port 8000) - Quantized Model
```bash
# Single GPU
./start_api.sh -g 1

# Multi-GPU balanced mode
./start_api.sh -g 1,2,3

# All available GPUs
./start_api.sh
```

#### BF16 Service (Port 8001) - Full Precision Model
```bash
# Single GPU
./start_bf16_api.sh -g 2

# Multi-GPU balanced mode
./start_bf16_api.sh -g 0,1

# All available GPUs
./start_bf16_api.sh
```

### Alternative Python Startup
```bash
# FP4 Service
source flux_env/bin/activate
python start_service.py

# BF16 Service
source flux_env/bin/activate
python start_bf16_service.py
```

## 🏗️ Architecture Overview

The system supports two FLUX models running in parallel:

- **FP4 Model (Port 8000)**: Quantized FLUX.1-dev model using Nunchaku for memory efficiency
- **BF16 Model (Port 8001)**: Full-precision FLUX.1-dev model in bfloat16 format for maximum quality

### Code Architecture
The BF16 service avoids duplication by:
1. **Extending the base model manager**: `BF16FluxModelManager` inherits from `FluxModelManager`
2. **Reusing API logic**: BF16 routes import and adapt existing functionality
3. **Separate configuration**: BF16-specific settings in `config/bf16_settings.py`
4. **Independent processes**: Each model runs on its own port with separate GPU assignment

## 🎮 GPU Management & Multi-GPU Support

### Automatic Balanced Mode
Both services now support **automatic multi-GPU load balancing**:

- **Single GPU**: Uses traditional `cuda:0` assignment
- **Multiple GPUs**: Uses `device_map="balanced"` for automatic model distribution across all visible GPUs
- **Manual Control**: Use `-g` flag to specify which GPU(s) each service should use

### GPU Selection Examples
```bash
# Manual single GPU assignment
./start_api.sh -g 1              # FP4 on GPU 1
./start_bf16_api.sh -g 2         # BF16 on GPU 2

# Multi-GPU balanced mode
./start_api.sh -g 1,2,3          # FP4 balanced across GPUs 1,2,3
./start_bf16_api.sh -g 0,1       # BF16 balanced across GPUs 0,1

# All available GPUs (no CUDA_VISIBLE_DEVICES restriction)
./start_api.sh                   # FP4 uses all GPUs
./start_bf16_api.sh              # BF16 uses all GPUs
```

### Memory Requirements
- **FP4 Model**: ~8-12 GB VRAM (single GPU) / distributed (multi-GPU)
- **BF16 Model**: ~16-24 GB VRAM (single GPU) / distributed (multi-GPU)
- **Both Models**: GPU memory is automatically distributed when using multi-GPU mode

## 📋 Service Management Scripts

| Script | Purpose | GPU Support | When to Use |
|--------|---------|-------------|-------------|
| **`start_api.sh`** | **FP4 service startup** | `-g` flag for GPU selection | **Primary FP4 service** |
| **`start_bf16_api.sh`** | **BF16 service startup** | `-g` flag for GPU selection | **High-quality generation** |
| `start_service.py` | Python FP4 starter | Manual CUDA_VISIBLE_DEVICES | Advanced users |
| `start_bf16_service.py` | Python BF16 starter | Manual CUDA_VISIBLE_DEVICES | Advanced users |

## 🌐 API Endpoints

Both services provide identical endpoints:

### Service URLs
- **FP4 API**: http://localhost:8000
- **BF16 API**: http://localhost:8001
- **FP4 Health**: http://localhost:8000/health
- **BF16 Health**: http://localhost:8001/health
- **FP4 Docs**: http://localhost:8000/docs
- **BF16 Docs**: http://localhost:8001/docs

### Available Endpoints
- `GET /` - Service status and available endpoints
- `POST /generate` - Generate images with optional LoRA support
- `POST /load-model` - Load the respective model
- `GET /model-status` - Get model status and GPU information
- `POST /apply-lora` - Apply LoRA to the model
- `POST /remove-lora` - Remove LoRA from the model
- `GET /lora-status` - Get current LoRA status
- `GET /loras` - Get available LoRAs (placeholder)
- `GET /health` - Health check endpoint

## ⚙️ Generation Parameters

Both services support configurable sampling parameters:

### **Core Parameters**
- **`num_inference_steps`** (1-100): Number of denoising steps
  - Lower values (10-20): Faster generation, lower quality
  - Higher values (30-50): Slower generation, higher quality
  - Default: 25

- **`guidance_scale`** (0.0-20.0): Classifier-free guidance strength
  - Lower values (1.0-3.0): More creative, less prompt adherence
  - Higher values (5.0-10.0): More prompt adherence, less creative
  - Default: 3.5

### **Image Parameters**
- **`width`** (256-1024): Image width in pixels
- **`height`** (256-1024): Image height in pixels
- **`seed`** (0-4294967295): Random seed for reproducible results

### **Advanced Parameters**
- **`negative_prompt`**: Text to avoid in the generated image
- **`lora_name`**: Hugging Face LoRA repository ID
- **`lora_weight`** (0.0-2.0): LoRA influence strength

## 📝 Usage Examples

### Generate Image with FP4 Model
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "num_inference_steps": 30,
    "guidance_scale": 4.0,
    "width": 768,
    "height": 768,
    "seed": 42,
    "negative_prompt": "blurry, low quality"
  }'
```

### Generate Image with BF16 Model
```bash
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "num_inference_steps": 30,
    "guidance_scale": 4.0,
    "width": 768,
    "height": 768,
    "seed": 42
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

## 🔍 Model Differences

### FP4 Model (Port 8000)
- ✅ Uses Nunchaku quantization for memory efficiency
- ✅ Lower VRAM usage (~8-12 GB)
- ✅ Faster inference
- ⚠️ Slightly lower quality due to quantization
- 🎯 **Best for**: High throughput, memory-constrained setups

### BF16 Model (Port 8001)
- ✅ Full-precision bfloat16 format
- ✅ Better quality output
- ✅ Standard diffusers LoRA support
- ⚠️ Higher VRAM usage (~16-24 GB)
- ⚠️ Slower inference
- 🎯 **Best for**: High-quality output, final production

## 📊 Monitoring & Status

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

### GPU Status
```bash
# Check GPU memory and utilization
nvidia-smi

# Check CUDA environment
echo $CUDA_VISIBLE_DEVICES
```

## 🔧 GPU Requirements

- **NVIDIA CUDA drivers** installed
- **PyTorch with CUDA support** in flux_env
- **Sufficient GPU memory** for model loading
- **Multiple GPUs** recommended for parallel services

### Recommended GPU Configurations

#### Single GPU Setup
```bash
# Use FP4 model only (lower memory)
./start_api.sh -g 0
```

#### Dual GPU Setup
```bash
# Terminal 1: FP4 on GPU 0
./start_api.sh -g 0

# Terminal 2: BF16 on GPU 1
./start_bf16_api.sh -g 1
```

#### Multi-GPU Setup
```bash
# Terminal 1: FP4 balanced across GPUs 0,1
./start_api.sh -g 0,1

# Terminal 2: BF16 balanced across GPUs 2,3
./start_bf16_api.sh -g 2,3
```

## 🆘 Troubleshooting

### GPU Memory Issues
```bash
# Check GPU status and available memory
nvidia-smi

# Kill processes using GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Port Conflicts
```bash
# Check what's using the ports
lsof -i :8000
lsof -i :8001

# Kill processes using the ports
sudo kill -9 <PID>
```

### Service Won't Start
1. **Check virtual environment**: Ensure `flux_env` exists and is activated
2. **Verify dependencies**: All required packages installed in flux_env
3. **Check GPU availability**: `nvidia-smi` shows available GPUs
4. **Check ports**: Ensure 8000/8001 are not in use
5. **Check logs**: Service startup logs show specific errors

### Memory Issues
If you run out of VRAM:
1. **Use single service**: Run only FP4 or BF16, not both
2. **Reduce batch size**: Lower inference steps or image resolution
3. **Use FP4 model**: Lower memory usage than BF16
4. **Multi-GPU mode**: Distribute model across multiple GPUs

### CUDA Errors
```bash
# Check CUDA installation
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Restart services with clean GPU state
./start_api.sh -g 0
```

## 🎯 Performance Optimization

### Single GPU Performance
- **FP4 Model**: Faster inference, use for real-time applications
- **BF16 Model**: Higher quality, use for final production

### Multi-GPU Performance
- **Balanced Mode**: Automatic load distribution across specified GPUs
- **Memory Distribution**: Large models split across multiple GPUs automatically
- **Parallel Inference**: Multiple requests can be processed simultaneously

### Load Balancing Strategies
- **Speed Priority**: Route requests to FP4 service (port 8000)
- **Quality Priority**: Route requests to BF16 service (port 8001)
- **Batch Processing**: Use BF16 for batch jobs, FP4 for interactive
- **A/B Testing**: Compare outputs between services

## 📁 Project Structure

```
flux_api/
├── main.py                    # FP4 FastAPI application
├── main_bf16.py              # BF16 FastAPI application
├── start_api.sh              # FP4 service launcher (with -g flag)
├── start_bf16_api.sh         # BF16 service launcher (with -g flag)
├── start_service.py          # FP4 Python launcher
├── start_bf16_service.py     # BF16 Python launcher
├── config/
│   ├── settings.py           # FP4 configuration
│   └── bf16_settings.py      # BF16 configuration
├── models/
│   ├── flux_model.py         # FP4 model manager (multi-GPU support)
│   └── bf16_flux_model.py    # BF16 model manager (extends base)
├── api/
│   ├── routes.py             # FP4 API endpoints
│   └── bf16_routes.py        # BF16 API endpoints
├── utils/
│   └── gpu_manager.py        # GPU utilities (legacy)
└── flux_env/                 # Python virtual environment
```

## 🚀 Development

### Adding New Features
1. **Implement in base classes**: Add to `FluxModelManager` or base routes first
2. **Extend for BF16**: Override or extend in BF16-specific classes
3. **Maintain API consistency**: Keep both services synchronized
4. **Test both modes**: Verify single and multi-GPU functionality

### Testing Multi-GPU Features
```bash
# Test single GPU mode
CUDA_VISIBLE_DEVICES=0 ./start_api.sh

# Test multi-GPU mode
./start_api.sh -g 0,1,2

# Test both services
./start_api.sh -g 0,1 &
./start_bf16_api.sh -g 2,3 &
```

## 🔮 Future Enhancements

- **Dynamic Model Switching**: Runtime model selection based on request parameters
- **Quality Presets**: Predefined configurations for different use cases
- **Auto-scaling**: Automatic GPU resource management based on load
- **Service Discovery**: Health monitoring and automatic failover
- **Request Routing**: Intelligent load balancing between FP4/BF16 based on requirements

---

## 🎉 Benefits of This Setup

- **Full GPU Access**: Direct access to all available NVIDIA GPUs
- **No Docker Overhead**: Native performance without containerization
- **Flexible GPU Assignment**: Manual control over which GPUs each service uses
- **Multi-GPU Load Balancing**: Automatic model distribution for large memory requirements
- **Dual Quality Options**: Choose between speed (FP4) and quality (BF16)
- **Easy Development**: Direct access to logs and environment for debugging