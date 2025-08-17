# FLUX API

AI image generation API with CUDA support.

## 🚀 Quick Start

### Start the API (Recommended)
```bash
./start_api.sh
```

This will:
- Activate the flux_env virtual environment
- Check GPU and dependency availability
- Start the API service directly
- Show real-time logs

### Alternative: Direct Python
```bash
# Activate virtual environment
source flux_env/bin/activate

# Start the service
python start_service.py
```

### Manual Start (Advanced)
```bash
# Activate virtual environment
source flux_env/bin/activate

# Start the service directly
python main.py
```

## 📋 Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| **`start_api.sh`** | **Start the API service** | **Use this to start the API** |
| `start_service.py` | Python service starter | For advanced users |

## 🔧 GPU Configuration

The FLUX API automatically detects and uses available GPUs from your flux_env environment.

### GPU Requirements

- NVIDIA CUDA drivers installed
- PyTorch with CUDA support in flux_env
- Sufficient GPU memory for model loading

### Check GPU Status

The service starter will automatically show:
- Available GPUs and memory
- CUDA version and compatibility
- PyTorch GPU support status

## 🌐 API Endpoints

- **Health**: `http://localhost:8000/health`
- **API Docs**: `http://localhost:8000/docs`
- **Generate Image**: `POST http://localhost:8000/generate`
- **Queue Management**: `POST http://localhost:8000/submit-request`

## ⚙️ Generation Parameters

The FLUX API now supports configurable sampling parameters for fine-tuned image generation:

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

### **Example Request**
```json
{
  "prompt": "A beautiful sunset over mountains",
  "num_inference_steps": 30,
  "guidance_scale": 4.0,
  "width": 768,
  "height": 768,
  "seed": 42,
  "negative_prompt": "blurry, low quality"
}
```

## 📁 Files

- `start_service.py` - Service starter script
- `main.py` - FastAPI application entry point
- `flux_env/` - Python virtual environment with dependencies
- `requirements.txt` - Python dependencies (in flux_env)

## 🆘 Troubleshooting

- **Virtual environment not found**: Ensure flux_env is properly set up
- **Missing dependencies**: Activate flux_env and install requirements
- **GPU not working**: Check nvidia-smi and PyTorch CUDA installation
- **Port in use**: Change port in main.py if needed

## 🎯 Benefits of Direct Python Setup

- **Full GPU Access**: Direct access to all 8 RTX 5090 GPUs
- **No Docker Overhead**: Faster startup and better performance
- **Easier Debugging**: Direct access to logs and environment
- **Simpler Development**: No container rebuilds needed
