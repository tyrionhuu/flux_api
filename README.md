# FLUX API - Modular Architecture

A high-performance image generation API using the FLUX model with quantized weights support, built with FastAPI and organized in a clean, modular architecture.

## 🏗️ Architecture Overview

The application is organized into logical modules for better maintainability, testability, and scalability:

```
flux_api/
├── main.py               # Main FastAPI application entry point
├── config/               # Configuration and settings
│   ├── __init__.py
│   └── settings.py       # All configuration constants
├── models/               # Model management
│   ├── __init__.py
│   └── flux_model.py     # FLUX model loading and quantization
├── api/                  # API endpoints and routes
│   ├── __init__.py
│   └── routes.py         # All FastAPI route handlers
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── gpu_manager.py    # GPU management and CUDA operations
│   ├── image_utils.py    # Image processing utilities
│   └── system_utils.py   # System monitoring utilities
├── requirements.txt      # Python dependencies
├── start_api.sh         # Startup script
└── README.md            # This file
```

## 🚀 Features

- **Modular Design**: Clean separation of concerns
- **Quantized Model Support**: Automatic integration of Nunchaku quantized weights
- **GPU Optimization**: Smart GPU selection and memory management
- **FastAPI**: Modern, fast web framework with automatic API documentation
- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Comprehensive error handling and fallbacks

## 📦 Module Details

### Config Module (`config/`)
- **`settings.py`**: Centralized configuration constants
- Model IDs, file paths, device settings, API configuration
- Easy to modify without touching business logic

### Models Module (`models/`)
- **`flux_model.py`**: Core model management
- Handles FLUX model loading, quantization integration
- GPU/CPU device management
- Model status and pipeline access

### API Module (`api/`)
- **`routes.py`**: All FastAPI endpoint definitions
- Clean route organization
- Request/response handling
- Business logic coordination

### Utils Module (`utils/`)
- **`gpu_manager.py`**: GPU operations and CUDA management
- **`image_utils.py`**: Image processing and file operations
- **`system_utils.py`**: System monitoring and memory tracking

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd flux_api
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv flux_env
   source flux_env/bin/activate  # On Windows: flux_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API**:
   ```bash
   ./start_api.sh
   ```
   
   Or run directly:
   ```bash
   python main.py
   ```

## 🎯 API Endpoints

- **`GET /`**: Root endpoint with API status
- **`GET /health`**: Health check endpoint
- **`POST /generate`**: Generate images from text prompts
- **`GET /static-image`**: Serve static images
- **`POST /load-model`**: Manually load the FLUX model
- **`GET /model-status`**: Get current model status
- **`GET /gpu-info`**: Detailed GPU information

## 🖼️ Image Generation

### Basic Usage
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"}'
```

### Form Data Support
```bash
curl -X POST "http://localhost:8000/generate" \
     -F "prompt=Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
```

## 🔍 Model Management

### Automatic Loading
The model is automatically loaded when you first call the `/generate` endpoint.

### Manual Loading
```bash
curl -X POST "http://localhost:8000/load-model"
```

### Status Check
```bash
curl "http://localhost:8000/model-status"
```

## 🎮 GPU Management

### Automatic GPU Selection
- Automatically detects available GPUs
- Selects GPU with most free memory
- Falls back to CPU if CUDA issues occur

### GPU Information
```bash
curl "http://localhost:8000/gpu-info"
```

## 🧠 Quantized Model Integration

The API automatically integrates Nunchaku quantized weights when available:

- **FP4 Weights**: Optimized for Blackwell GPUs (RTX 5090)
- **INT4 Weights**: Fallback for older GPUs
- **Automatic Fallback**: Uses standard model if quantization fails

## 🛠️ Development

### Code Structure
- **Separation of Concerns**: Each module has a specific responsibility
- **Dependency Injection**: Clean interfaces between modules
- **Type Hints**: Full type safety throughout
- **Error Handling**: Comprehensive error handling and logging

### Adding New Features
1. **New Endpoints**: Add to `api/routes.py`
2. **New Models**: Add to `models/` directory
3. **New Utilities**: Add to `utils/` directory
4. **Configuration**: Update `config/settings.py`

### Testing
```bash
# Test individual modules
python -c "from models.flux_model import FluxModelManager; print('Models OK')"
python -c "from api.routes import router; print('Routes OK')"
python -c "from utils.gpu_manager import GPUManager; print('GPU Manager OK')"

# Test the main application
python -m py_compile app.py
```

## 📊 Performance

- **Memory Optimization**: Low CPU memory usage, efficient GPU utilization
- **Quantized Weights**: Reduced memory footprint with quantized models
- **Smart Device Selection**: Automatic GPU/CPU selection based on availability
- **Caching**: HuggingFace model caching for faster subsequent loads

## 🔒 Security

- **CORS Configuration**: Configurable CORS middleware
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Safe error responses without information leakage

## 🚀 Production Deployment

### Environment Variables
Set appropriate environment variables for production:
- `CORS_ORIGINS`: Configure allowed origins
- `HOST`: Bind address (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

### Process Management
Use process managers like `systemd`, `supervisor`, or `pm2` for production deployment.

## 📝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests if applicable**
5. **Submit a pull request**

## 📄 License

This project uses the FLUX model which is subject to its own license terms.

## 🤝 Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include system information and error logs

---

**Built with ❤️ using FastAPI and the FLUX model**
