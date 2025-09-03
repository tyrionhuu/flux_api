# FLUX API - Image to Image Generation Service

A dual-model AI image generation API service featuring FLUX models with LoRA support.

## Features

- **FP4 Model Support**: Quantized FLUX model for efficient inference
- **GPU Management**: Automatic GPU selection and load balancing
- **LoRA Support**: Apply custom LoRA weights for style customization
- **LoRA File Upload**: Upload local LoRA files directly through the web interface
- **ComfyUI-style Frontend**: Modern, intuitive web interface
- **RESTful API**: Easy integration with external applications

## Model Services

### FP4 Model (Port 8000)
- **Port**: 8000 (configurable)
- **Model**: FLUX.1-schnell (quantized)
- **Memory**: ~8GB VRAM
- **Speed**: Fast inference with LoRA merging support

## Installation

### Prerequisites

- Python 3.12
- CUDA-compatible GPU with sufficient VRAM
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:tyrionhuu/flux_api.git
   cd flux_api
   git checkout img2img
   ```

2. **Create conda environment**
   ```bash
   conda create -n flux_api python=3.12
   conda activate flux_api
   ```

3. **Install dependencies from requirements.txt**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support**
   ```bash
   pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

5. **Install Nunchaku package**
   ```bash
   pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.2/nunchaku-0.3.2+torch2.8-cp312-cp312-linux_x86_64.whl
   ```


## Quick Start

### 1. Start the Services

```bash
# Start FP4 service
./start_flux_api.sh

# Start frontend (optional)
cd frontend && python -m http.server 9000
```

### 2. Access the Services

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

The FP4 model supports applying multiple LoRAs simultaneously:

- **FP4 Model**: Merges multiple LoRAs into a single LoRA
- **Maximum**: 3 LoRA layers
- **Weight Combination**: Automatic weight calculation

## API Usage

### Text-to-Image (JSON) — `/generate`

Generates an image and returns JSON with file paths and a download URL.

```bash
curl -X POST "http://localhost:9000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains and a lake",
    "seed": 42,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "upscale": false,
    "upscale_factor": 2,
    "remove_background": false,
    "loras": [
      {"name": "username/style-lora", "weight": 1.0},
      {"name": "uploaded_lora_1700000000.safetensors", "weight": 0.8}
    ]
  }'
```

Response (example):

```json
{
  "message": "Generated FLUX image for prompt: ...",
  "image_url": "generated_images/flux_...png",
  "download_url": "/download/flux_...png",
  "filename": "flux_...png",
  "generation_time": "4.92s",
  "lora_applied": "username/style-lora",
  "lora_weight": 1.8,
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

Notes:
- `loras` is optional. If provided as an empty list `[]`, all LoRAs are removed before generation.
- If you prefer a single LoRA, you can also send `lora_name` and `lora_weight` instead of `loras`.
- This endpoint auto-sizes to 1024x1024.

### Text-to-Image (binary image) — `/generate-and-return-image`

Same as `/generate` but returns the PNG image bytes directly.

```bash
curl -X POST "http://localhost:9000/generate-and-return-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A watercolor cityscape at dusk",
    "remove_background": false
  }' \
  -o output.png
```

### Image-to-Image (JSON) — `/generate-with-image`

Send an image plus form fields; returns JSON with saved filename and URL.

```bash
curl -X POST "http://localhost:9000/generate-with-image" \
  -F "prompt=Product photo, soft shadows" \
  -F "image=@/path/to/input.jpg" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=3.0" \
  -F "width=512" \
  -F "height=512" \
  -F "seed=123" \
  -F "remove_background=false"
```

Response fields:
- `image_url` and `download_url` (for this endpoint, download URL is under `/generated_images/...`).
- `filename`, `generation_time`, `width`, `height`, `seed`.
 - `lora_applied`, `lora_weight` if a LoRA is active.

### Image-to-Image (binary image) — `/generate-with-image-and-return`

Same request as above, but returns PNG bytes directly.

```bash
curl -X POST "http://localhost:9000/generate-with-image-and-return" \
  -F "prompt=Portrait photo, soft studio lighting" \
  -F "image=@/path/to/input.png" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=3.0" \
  -F "width=512" \
  -F "height=512" \
  -F "seed=123" \
  -F "negative_prompt=blurry, low-res" \
  -F "prompt_prefix=professional product photo" \
  -F "remove_background=true" \
  -F "bg_strength=0.6" \
  -F 'loras_json=[{"name":"username/model-name","weight":1.0}]' \
  -o output.png
```

Parameters (multipart/form-data):
- `prompt` (string, required): Text prompt.
- `image` (file, required): Reference image. Allowed extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`. Max size: 10MB.
- `num_inference_steps` (int, optional, default from server): Diffusion steps. Example: 24.
- `guidance_scale` (float, optional, default from server): CFG guidance. Example: 3.0.
- `width` (int, optional, default derived from preprocessing): Must be 256–1024.
- `height` (int, optional, default derived from preprocessing): Must be 256–1024.
- `seed` (int, optional): Random seed for reproducibility.
- `negative_prompt` (string, optional): Negative guidance text.
- `prompt_prefix` (string, optional): Prepended to `prompt` if provided.
- `remove_background` (bool, optional, default false): If true, applies background removal to the final image.
- `bg_strength` (float 0..1, optional): Controls background removal aggressiveness when `remove_background=true`.
- `loras_json` (JSON array, optional): Multiple LoRAs, e.g., `[{"name":"user/model","weight":1.0}]`. Use empty array `[]` to remove all LoRAs.
- `lora_name` (string, optional) and `lora_weight` (float 0..2, optional): Single LoRA shorthand.
- `use_default_lora` (bool, optional): Apply server default LoRA.

Behavior and constraints:
- The endpoint validates that the uploaded file is an image (MIME type + extension) and ≤ 10MB; otherwise it returns HTTP 400.
- `width` and `height` must be within 256–1024; out-of-range values return HTTP 400.
- If `prompt_prefix` is supplied, the effective prompt is `"{prompt_prefix}, {prompt}"`.
- If `remove_background=true`, the server post-processes the generated image using alpha matting; `bg_strength` maps 0..1 to more aggressive background removal.
- This endpoint returns binary PNG bytes directly with `Content-Type: image/png`; it does not return JSON.

Example response headers:
```
HTTP/1.1 200 OK
content-type: image/png
content-length: <bytes>
```

Error responses (examples):
- `400`: missing `prompt` or `image`, invalid dimensions, non-image upload, or file too large.
- `500`: model not loaded or internal generation errors.

### Manage LoRAs

- List available LoRAs (uploaded + default):

```bash
curl "http://localhost:9000/loras"
```

- Apply a LoRA to the loaded model (query params):

```bash
curl -X POST "http://localhost:9000/apply-lora?lora_name=username/model-name&weight=1.0"
```

- Remove the currently applied LoRA:

```bash
curl -X POST "http://localhost:9000/remove-lora"
```

- Upload a LoRA file (server stores it under `uploads/lora_files/`):

```bash
curl -X POST "http://localhost:9000/upload-lora" \
  -F "file=@/path/to/your/lora.safetensors"
```

- Remove an uploaded LoRA file and its index entry:

```bash
curl -X DELETE "http://localhost:9000/remove-lora/uploaded_lora_1700000000.safetensors"
```

### Queue APIs

- Submit a generation request to the queue:

```bash
curl -X POST "http://localhost:9000/submit-request" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cozy cabin in the woods at night"}'
```

- Check request status:

```bash
curl "http://localhost:9000/request-status/<request_id>"
```

- Cancel a queued request:

```bash
curl -X DELETE "http://localhost:9000/cancel-request/<request_id>"
```

- Get queue stats:

```bash
curl "http://localhost:9000/queue-stats"
```

### Download Generated Image

Use the `download_url` from `/generate` responses (served by `/download/{filename}`) or the `/generated_images/{filename}` path from image-to-image endpoints.

```bash
curl -L "http://localhost:9000/download/<filename>.png" -o result.png
```

## Configuration

### Port Configuration

Set custom ports using environment variables or command-line flags:

```bash
# Environment variables
export FP4_PORT=8000

# Or command-line flags
./start_flux_api.sh --port 8000
```

## File Structure

```
flux_api/
├── api/                    # API routes and models
├── models/                 # FLUX model implementations
├── utils/                  # Utility modules
├── config/                 # Configuration files
├── frontend/               # Web interface
├── uploads/                # Uploaded LoRA files
│   └── lora_files/        # LoRA file storage
├── generated_images/       # Generated images
└── start_*.sh             # Service startup scripts
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
2. **API**: Add routes in `api/routes.py`
3. **Models**: Extend `models/flux_model.py`

### Testing

```bash
# Test API endpoints
python -c "import requests; print(requests.get('http://localhost:8000/').json())"

# Test model imports
python -c "from models.flux_model import FluxModelManager; print('✅ FP4 model imports successfully')"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.