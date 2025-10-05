# Docker Usage Guide

## Quick Start

### Build the Image
```bash
docker build -t flux-dev-clean:latest .
```

### Run Basic Container (No LoRA)
```bash
docker run -d \
  --name flux-dev-clean \
  --gpus all \
  -p 8000:8000 \
  flux-dev-clean:latest
```
**Description**: Starts backend API on port 8000 with all GPUs, no LoRA fusion.

### Run with LoRA Fusion Mode (Local File)
```bash
docker run -d --name flux-dev-clean-ghibli \
  --gpus '"device=7"' -p 8000:8000 \
  -e MODEL_TYPE=flux -e FUSION_MODE=true \
  -e LORA_NAME="/lora_weights/Studio_Ghibli_Flux.safetensors" \
  -e LORA_WEIGHT=1.0 \
  -e HF_TOKEN=$HUGGINGFACE_HUB_TOKEN \
  -v /data/weights/lora_checkpoints:/lora_weights:ro \
  flux-dev-clean:latest
```
**Description**: Starts API with local LoRA file pre-applied at startup. Mount host directory and use container path. LoRA cannot be changed without restart.

### Run with LoRA Fusion Mode (Hugging Face)
```bash
docker run -d --name flux-dev-clean-ghibli \
  --gpus '"device=7"' -p 8000:8000 \
  -e MODEL_TYPE=flux -e FUSION_MODE=true \
  -e LORA_NAME="username/repo-name" \
  -e LORA_WEIGHT=1.0 \
  -e HF_TOKEN=$HUGGINGFACE_HUB_TOKEN \
  flux-dev-clean:latest
```
**Description**: Downloads and applies LoRA from Hugging Face at startup. No volume mount needed.

### Run with Specific GPU
```bash
docker run -d \
  --name flux-dev-clean \
  --gpus '"device=0"' \
  -p 8000:8000 \
  -e MODEL_TYPE=flux \
  flux-dev-clean:latest
```
**Description**: Runs on GPU 0 only. Useful for multi-GPU systems.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | API server port |
| `MODEL_TYPE` | flux | Model to load (flux/qwen) |
| `FUSION_MODE` | false | Enable LoRA fusion at startup |
| `LORA_NAME` | "" | Single LoRA path or HF repo |
| `LORA_WEIGHT` | 1.0 | LoRA weight (0.0-2.0) |
| `LORAS_CONFIG` | "" | JSON array for multiple LoRAs |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `MAX_WORKERS` | 1 | Uvicorn worker count |

## Health Checks

Container includes automatic health checks on `/health` endpoint every 30 seconds.

---

# API Usage Guide (curl)

## Basic Operations

### Check API Health
```bash
curl http://localhost:8000/health
```
**Description**: Returns service status, model info, and fusion mode status.

### Check Model Status
```bash
curl http://localhost:8000/model-status
```
**Description**: Detailed model status including GPU memory, LoRA info, fusion mode.

### Check Fusion Mode Status
```bash
curl http://localhost:8000/fusion-mode-status
```
**Description**: Check if fusion mode is enabled and what LoRAs are locked.

## Image Generation

### Generate Image (Returns JSON with Download URL)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains",
    "width": 512,
    "height": 512
  }'
```
**Description**: Generate image with default settings. Returns JSON with download URL. Uses currently applied LoRA if any.

### Generate and Return Image (Returns PNG Bytes)
```bash
curl -X POST http://localhost:8000/generate-and-return-image \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A sunset over the ocean","width":1024,"height":768,"seed":42}' \
  --output image.png
```

### Generate Image (Full Options)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A sunset over the ocean",
    "width": 1024,
    "height": 768,
    "seed": 42,
    "num_inference_steps": 20,
    "guidance_scale": 3.5,
    "upscale": false
  }'
```