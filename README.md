# Diffusion API - Text to Image Generation Service

A high-performance AI image generation API service featuring the Diffusion FP4 model with LoRA support and automatic cleanup services.

## Features

- **Model Support**: FP4 (quantized) Diffusion model
- **GPU Management**: Automatic GPU selection and load balancing
- **LoRA Support**: Apply custom LoRA weights for style customization
- **LoRA File Upload**: Upload local LoRA files directly through the web interface
- **ComfyUI-style Frontend**: Modern, intuitive web interface
- **RESTful API**: Easy integration with external applications
- **Automatic Cleanup**: Built-in directory cleanup service for maintenance
- **Systemd Integration**: Run as system services for production deployment

## Model Service

### FP4 Model (Port 8000)
- **Port**: 8000 (configurable via environment variable `API_PORT`)
- **Model**: FLUX.1-dev (quantized)
- **Memory**: ~8GB VRAM
- **Speed**: Fast inference with LoRA merging support

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or create a virtual environment first
python -m venv flux_env
source flux_env/bin/activate  # On Windows: flux_env\Scripts\activate
pip install -r requirements.txt
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0dev20250830/nunchaku-1.0.0.dev20250830+torch2.8-cp312-cp312-linux_x86_64.whl
```

### 2. Start the Service

```bash
# Start the main API service
./start_api.sh

# Or start with specific GPU(s)
./start_api.sh -g 0,1

# Or start with custom port
./start_api.sh -p 8002

# Start frontend (optional)
cd frontend && python -m http.server 9000
```

### 3. Access the Service

- **Frontend**: http://localhost:8000

## Services

### Main API Service

The primary service runs on port 8000 by default and handles:
- Image generation requests
- LoRA management
- File uploads
- GPU load balancing

### Cleanup Service

An automatic cleanup service that maintains directory size limits:

```bash
# Start cleanup service manually
python utils/cleanup_service.py

# Or install as systemd service
sudo cp services/cleanup.service /etc/systemd/system/
sudo systemctl enable cleanup.service
sudo systemctl start cleanup.service
```

For detailed service configuration, see [services/README.md](services/README.md).

## LoRA Support

### Using Hugging Face LoRAs

```bash
# Apply a Hugging Face LoRA
curl -X POST "http://localhost:8000/apply-lora" \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "username/model-name", "weight": 1.0}'
```

### Uploading Local LoRA Files

The web interface supports uploading local LoRA files:

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

Set custom ports using environment variables:

```bash
# Environment variables
export API_PORT=8002

# Or modify config/settings.py
API_PORT = 8002
```

### GPU Configuration

The service automatically detects and uses available GPUs:

```bash
# Check GPU status
python -c "from utils.gpu_manager import GPUManager; gm = GPUManager(); print(gm.get_gpu_info())"

# Start with specific GPUs
./start_api.sh -g 0,1
```

### Cleanup Configuration

Configure automatic cleanup in `config/cleanup_settings.py`:

```python
# Directory size limits
MAX_DIRECTORY_SIZE_GB = 10
MAX_FILE_AGE_DAYS = 30
```

## File Structure

```
flux_api/
├── api/                    # API routes and models
├── models/                 # Diffusion model implementations
├── utils/                  # Utility modules including cleanup service
├── config/                 # Configuration files
├── services/               # Systemd service files
├── frontend/               # Web interface
├── uploads/                # Uploaded LoRA files
│   └── lora_files/        # LoRA file storage
├── generated_images/       # Generated images
├── logs/                   # Application logs
├── main.py                 # Main FastAPI application
├── start_api.sh       # Service startup script
├── start_service.py   # Alternative startup script
├── cleanup_directories.py  # Manual cleanup utility
└── requirements.txt        # Python dependencies
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # The startup script automatically handles port conflicts
   # Or manually kill processes using the port
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

4. **Cleanup Service Issues**
   - Check service status: `sudo systemctl status cleanup.service`
   - View logs: `sudo journalctl -u cleanup.service -f`
   - Verify configuration in `config/cleanup_settings.py`

### Logs

Check service logs for detailed error information:

```bash
# View real-time logs
tail -f logs/flux_api_fp4.log

# Check cleanup service logs
sudo journalctl -u cleanup.service -f
```

## Development

### Adding New Features

1. **Frontend**: Modify `frontend/static/js/app.js`
2. **API**: Add routes in `api/routes.py`
3. **Models**: Extend `models/models.py`
4. **Services**: Add new service files in `services/`

### Testing

```bash
# Test API endpoints
python -c "import requests; print(requests.get('http://localhost:8000/').json())"

# Test model imports
python -c "from models.models import DiffusionModelManager; print('✅ FP4 model imports successfully')"

# Test cleanup service
python utils/cleanup_service.py --dry-run
```

## Production Deployment

### Environment Variables

Set production environment variables:

```bash
export API_PORT=8000
export CUDA_VISIBLE_DEVICES=0,1
export FLUX_LOG_LEVEL=INFO
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.