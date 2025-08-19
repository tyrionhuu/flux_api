# FLUX API - Dual Model Service

AI image generation API with FP4 (quantized) and BF16 (full-precision) FLUX models, featuring automatic multi-GPU load balancing and default LoRA integration.

## 🚀 Quick Start

### FP4 Service (Port 8000) - Fast & Efficient
```bash
# Single GPU
./start_api.sh -g 1

# Multi-GPU balanced
./start_api.sh -g 1,2,3

# All GPUs
./start_api.sh
```

### BF16 Service (Port 8001) - High Quality
```bash
# Single GPU  
./start_bf16_api.sh -g 2

# Multi-GPU balanced
./start_bf16_api.sh -g 0,1

# All GPUs
./start_bf16_api.sh
```

## 🔧 GPU Selection

| Command | GPU Mode | Use Case |
|---------|----------|----------|
| `./start_api.sh -g 1` | Single GPU 1 | Limited VRAM |
| `./start_api.sh -g 1,2,3` | Balanced across GPUs 1,2,3 | High memory models |
| `./start_api.sh` | All available GPUs | Maximum performance |

## 📊 Service Comparison

| Feature | FP4 (Port 8000) | BF16 (Port 8001) |
|---------|------------------|-------------------|
| **Quality** | Good | Excellent |
| **Speed** | Fast | Slower |
| **VRAM** | ~8-12 GB | ~16-24 GB |
| **Best For** | Real-time, batch | Final production |

## 🌐 API Endpoints

Both services available at:
- **FP4**: http://localhost:8000/docs
- **BF16**: http://localhost:8001/docs

## 📝 Usage Examples

### Basic Image Generation
```bash
# FP4 - Fast generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over mountains"}'

# BF16 - High quality  
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over mountains"}'
```

### Advanced Parameters
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape painting",
    "num_inference_steps": 30,
    "guidance_scale": 4.0,
    "width": 768,
    "height": 768,
    "seed": 42,
    "negative_prompt": "blurry, low quality"
  }'
```

### Custom LoRA
```bash
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "lora_name": "another-user/custom-lora",
    "lora_weight": 0.8
  }'
```

### Download to Local Machine
```bash
# Step 1: Generate image and get download URL
response=$(curl -s -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A majestic dragon"}')

# Step 2: Extract download URL and filename
download_url=$(echo $response | jq -r '.download_url')
filename=$(echo $response | jq -r '.filename')

# Step 3: Download to your local machine
curl -o "$filename" "http://localhost:8000$download_url"

# Or download with custom name
curl -o "my_dragon.png" "http://localhost:8000$download_url"
```

### One-liner Download
```bash
# Generate and download in one command
curl -s -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A space station"}' | \
  jq -r '"curl -o \"" + .filename + "\" \"http://localhost:8000" + .download_url + "\""' | \
  bash
```



## 🎨 Default LoRA

Both services automatically load **Ghibli-style illustration LoRA** (`aleksa-codes/flux-ghibsky-illustration`) with weight `1.0`:

- ✅ **No LoRA specified**: Uses default LoRA
- ✅ **Custom LoRA**: Override with `lora_name` parameter  
- ✅ **Weight 0.0**: Disable LoRA influence

## ⚙️ Parameters Reference

### Core Parameters
- `prompt` **(required)**: Text description
- `num_inference_steps` (1-100): Quality vs speed (default: 25)
- `guidance_scale` (0.0-20.0): Prompt adherence (default: 3.5)
- `width`/`height` (256-1024): Image dimensions (default: 512)
- `seed` (optional): Reproducible results
- `negative_prompt` (optional): What to avoid

### LoRA Parameters  
- `lora_name` (optional): HuggingFace repo ID
- `lora_weight` (0.0-2.0): LoRA strength (default: 1.0)



### Response Fields
- `download_url`: API endpoint to download the image to your local machine
- `filename`: Generated filename for the image
- `image_url`: Server path where the image is stored

## 🔍 Status & Monitoring

```bash
# Check service status
curl http://localhost:8000/model-status  # FP4
curl http://localhost:8001/model-status  # BF16

# Health checks
curl http://localhost:8000/health        # FP4  
curl http://localhost:8001/health        # BF16

# GPU status
nvidia-smi
```

## 🆘 Troubleshooting

### GPU Issues
```bash
# Check GPU memory
nvidia-smi

# Kill GPU processes
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>
```

### Port Issues
```bash
# Check ports
lsof -i :8000
lsof -i :8001

# Kill port processes
sudo kill -9 <PID>
```

### Service Won't Start
1. Check virtual environment: `ls flux_env/`
2. Check GPU availability: `nvidia-smi`
3. Check file permissions: `ls -la start_*.sh`
4. Activate environment: `source flux_env/bin/activate`

### Memory Issues
- Use FP4 instead of BF16 for lower VRAM
- Use single GPU mode: `-g 1`
- Reduce image size: `"width": 512, "height": 512`
- Lower inference steps: `"num_inference_steps": 15`

## 📁 File Structure

```
flux_api/
├── start_api.sh              # FP4 launcher
├── start_bf16_api.sh         # BF16 launcher  
├── main_fp4.py               # FP4 app
├── main_bf16.py              # BF16 app
├── config/
│   ├── fp4_settings.py       # FP4 config
│   └── bf16_settings.py      # BF16 config
├── models/
│   ├── fp4_flux_model.py     # FP4 model
│   └── bf16_flux_model.py    # BF16 model
├── api/
│   ├── fp4_routes.py         # FP4 endpoints
│   └── bf16_routes.py        # BF16 endpoints
└── flux_env/                 # Python environment
```

## 💡 Tips

- **Development**: Use FP4 for fast iteration
- **Production**: Use BF16 for final outputs  
- **Batch Jobs**: Use BF16 with high steps
- **Real-time**: Use FP4 with low steps
- **Multi-GPU**: Use balanced mode for large models
- **Memory Saving**: Use single GPU with FP4
- **Direct Downloads**: Use download URLs to get images to your local machine

## 📥 Download Feature

The API now supports downloading generated images directly to your local machine:

### 🔧 How It Works
1. **Generate**: Make a POST request to `/generate`
2. **Get URL**: Response includes `download_url` and `filename`
3. **Download**: Use the download URL to get the file locally

### 🌐 Download Method

```bash
# Generate and get download info
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A fantasy castle"}' > response.json

# Download to local machine
download_url=$(jq -r '.download_url' response.json)
filename=$(jq -r '.filename' response.json)
curl -o "$filename" "http://localhost:8000$download_url"
```

### 🛡️ Security Features
- Download endpoint only serves files from `generated_images/` directory
- Path validation prevents directory traversal attacks
- File type validation for supported image formats
- Error handling with detailed feedback

### 📝 Batch Download Script
```bash
#!/bin/bash
# Generate multiple images and download them
prompts=("Dragon" "Castle" "Forest" "Ocean" "Mountain")

for i in "${!prompts[@]}"; do
  echo "Generating: ${prompts[$i]}"
  
  # Generate image
  response=$(curl -s -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${prompts[$i]}\", \"seed\": $((42 + i))}")
  
  # Download image
  download_url=$(echo $response | jq -r '.download_url')
  filename=$(echo $response | jq -r '.filename')
  
  curl -s -o "${prompts[$i],,}_$filename" "http://localhost:8000$download_url"
  echo "Downloaded: ${prompts[$i],,}_$filename"
done
```

---

🎯 **Ready to generate? Start with `./start_api.sh -g 1` and visit http://localhost:8000/docs**