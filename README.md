# FLUX API - AI Image Generation Service

A high-performance AI image generation API service featuring the FLUX FP4 model with LoRA support.

## Features

- **Model Support**: FP4 (quantized) FLUX model
- **GPU Management**: Automatic GPU selection and load balancing
- **LoRA Support**: Apply custom LoRA weights for style customization
- **LoRA File Upload**: Upload local LoRA files directly through the web interface
- **ComfyUI-style Frontend**: Modern, intuitive web interface
- **RESTful API**: Easy integration with external applications

## Model Service

### FP4 Model (Port 8000)
- **Port**: 8000 (configurable)
- **Model**: FLUX.1-schnell (quantized)
- **Memory**: ~8GB VRAM
- **Speed**: Fast inference with LoRA merging support


## Quick Start

### 1. Start the Service

```bash
# Start FP4 service
./start_fp4_api.sh

# Start frontend (optional)
cd frontend && python -m http.server 9000
```

### 2. Access the Service

- **Frontend**: http://localhost:9000
- **FP4 API**: http://localhost:8000

## LoRA Support

### Using Hugging Face LoRAs

```bash
# Apply a Hugging Face LoRA
curl -X POST "http://localhost:8000/apply-lora" \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "username/model-name", "weight": 1.0}'
```

### Uploading Local LoRA Files

The web interface now supports uploading local LoRA files:

1. **Click "Upload LoRA"** button in the frontend
2. **Select your LoRA file** (.safetensors, .bin, .pt, .pth)
3. **Set the weight** (0.0 - 2.0)
4. **Apply the LoRA** using the "Apply LoRA" button

**Supported Formats**:
- `.safetensors` (recommended)
- `.bin`
- `.pt` / `.pth`

**File Size Limit**: 1GB maximum

### Multiple LoRA Support

The FP4 model supports applying multiple LoRAs simultaneously by merging multiple LoRAs into a single LoRA (maximum 3 layers). Weight combinations are calculated automatically.

## API Usage

### Generate Image

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains and a lake",
    "width": 512,
    "height": 512,
    "seed": 42,
    "loras": [
      {"name": "username/style-lora", "weight": 1.0},
      {"name": "uploads/lora_files/uploaded_lora_123.safetensors", "weight": 0.8}
    ]
  }'
```

### Upload LoRA File

```bash
curl -X POST "http://localhost:8000/upload-lora" \
  -F "file=@/path/to/your/lora.safetensors"
```

## Configuration

### Port Configuration

Set custom ports using environment variables or command-line flags:

```bash
# Environment variables
export FP4_PORT=8000

# Or command-line flags
./start_fp4_api.sh --port 8000
```

### GPU Configuration

The service automatically detects and uses available GPUs:

```bash
# Check GPU status
python -c "from utils.gpu_manager import GPUManager; gm = GPUManager(); print(gm.get_gpu_info())"
```

## File Structure

```
flux_api/
├── api/                    # API routes and models
├── models/                 # FLUX model implementations (FP4)
├── utils/                  # Utility modules
├── config/                 # Configuration files
├── frontend/               # Web interface
├── uploads/                # Uploaded LoRA files
│   └── lora_files/        # LoRA file storage
├── generated_images/       # Generated images
└── start_fp4_api.sh        # Service startup script
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill process using port
   sudo lsof -ti:8000 | xargs kill -9
   ```

2. **GPU Memory Issues**
   - Reduce image dimensions
   - Use FP4 model for lower memory usage
   - Check GPU memory with `nvidia-smi`

3. **LoRA Upload Failures**
   - Check file format (.safetensors recommended)
   - Ensure file size < 1GB
   - Verify file integrity

### Logs

Check service logs for detailed error information:

```bash
# View real-time logs
tail -f logs/flux_api.log
```

## Development

### Adding New Features

1. **Frontend**: Modify `frontend/static/js/app.js`
2. **API**: Add routes in `api/fp4_routes.py`
3. **Models**: Extend `models/fp4_flux_model.py`

### Testing

```bash
# Test API endpoints
python -c "import requests; print(requests.get('http://localhost:8000/').json())"

# Test model imports
python -c "from models.fp4_flux_model import FluxModelManager; print('✅ FP4 model imports successfully')"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.