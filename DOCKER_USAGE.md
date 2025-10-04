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

### Run with LoRA Fusion Mode
```bash
docker run -d --name flux-dev-clean-ghibli \
  --gpus '"device=7"' -p 8000:8000 \
  -e MODEL_TYPE=flux -e FUSION_MODE=true \
  -e LORA_NAME="/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors" \
  -e LORA_WEIGHT=1.0 -e HF_TOKEN=$HUGGINGFACE_HUB_TOKEN\
  flux-dev-clean:latest
```
**Description**: Starts API with LoRA pre-applied at startup. LoRA cannot be changed without restart.

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

### Run with Multiple LoRAs
```bash
docker run -d \
  --name flux-dev-clean \
  --gpus all \
  -p 8000:8000 \
  -e FUSION_MODE=true \
  -e LORAS_CONFIG='[{"name":"lora1","weight":1.0},{"name":"lora2","weight":0.8}]' \
  flux-dev-clean:latest
```
**Description**: Applies multiple LoRAs at startup with different weights.

## Container Management

### View Logs
```bash
docker logs -f flux-dev-clean
```
**Description**: Follow container logs in real-time. Check for model loading status.

### Stop Container
```bash
docker stop flux-dev-clean
```
**Description**: Gracefully stops the container.

### Remove Container
```bash
docker rm -f flux-dev-clean
```
**Description**: Force removes container (stops if running).

### Shell Access
```bash
docker exec -it flux-dev-clean bash
```
**Description**: Opens interactive shell inside running container for debugging.

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

### Generate Image (Simple)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains",
    "width": 512,
    "height": 512
  }'
```
**Description**: Generate image with default settings. Uses currently applied LoRA if any.

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
**Description**: Full control over generation parameters. Seed ensures reproducibility.

## LoRA Management (Runtime - Not in Fusion Mode)

### Apply LoRA from Hugging Face
```bash
curl -X POST "http://localhost:8000/apply-lora?lora_name=username/repo-name&weight=1.0"
```
**Description**: Download and apply LoRA from Hugging Face. Returns 403 if fusion mode enabled.

### Remove LoRA
```bash
curl -X POST http://localhost:8000/remove-lora
```
**Description**: Remove currently applied LoRA. Returns 403 if fusion mode enabled.

### Upload Local LoRA File
```bash
curl -X POST http://localhost:8000/upload-lora \
  -F "file=@/path/to/lora.safetensors"
```
**Description**: Upload local LoRA file to server (max 1GB). Returns filename for later use.

### List Available LoRAs
```bash
curl http://localhost:8000/loras
```
**Description**: List all uploaded LoRA files with metadata.

### Get LoRA Status
```bash
curl http://localhost:8000/lora-status
```
**Description**: Check which LoRA is currently applied and its weight.

## Two-Step Generation with LoRA

### Step 1: Apply LoRA
```bash
curl -X POST "http://localhost:8000/apply-lora?lora_name=Fihade/style-lora&weight=1.0"
```
**Description**: Apply LoRA to model. This modifies the model state.

### Step 2: Generate with Applied LoRA
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A stylized portrait", "width": 512, "height": 512}'
```
**Description**: Generate using the LoRA applied in step 1.

## Queue Operations

### Submit Request to Queue
```bash
curl -X POST http://localhost:8000/submit-request \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cozy cabin in the woods",
    "width": 512,
    "height": 512
  }'
```
**Description**: Submit generation to queue for async processing. Returns request ID.

### Check Request Status
```bash
curl http://localhost:8000/request-status/{request_id}
```
**Description**: Check status of queued request (pending/processing/completed/failed).

### Cancel Request
```bash
curl -X DELETE http://localhost:8000/cancel-request/{request_id}
```
**Description**: Cancel pending request before processing starts.

### Get Queue Stats
```bash
curl http://localhost:8000/queue-stats
```
**Description**: View queue statistics (pending, processing, completed counts).

## Model Management

### Switch Model Type
```bash
curl -X POST http://localhost:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model_type": "qwen"}'
```
**Description**: Switch between flux and qwen models. Unloads current model first.

### Get GPU Info
```bash
curl http://localhost:8000/gpu-info
```
**Description**: Detailed GPU information (memory, temperature, utilization).

## Download Generated Image

### Get Image by Filename
```bash
curl -O http://localhost:8000/download/{filename}.png
```
**Description**: Download generated image using filename from generation response.

### Direct Download
```bash
curl http://localhost:8000/download/flux_20250104_123456.png -o output.png
```
**Description**: Download and save with custom filename.

## Testing Fusion Mode Lock

### Test Fusion Mode Protection (Should Return 403)
```bash
curl -X POST "http://localhost:8000/apply-lora?lora_name=test&weight=1.0"
```
**Description**: If fusion mode enabled, this returns 403 Forbidden with error message.

## Response Examples

### Successful Generation
```json
{
  "message": "Generated FLUX image",
  "image_url": "generated_images/flux_20250104_123456.png",
  "download_url": "/download/flux_20250104_123456.png",
  "filename": "flux_20250104_123456.png",
  "generation_time": "4.2s",
  "lora_applied": "username/style-lora",
  "lora_weight": 1.0,
  "width": 512,
  "height": 512,
  "seed": 42
}
```

### Fusion Mode Active
```json
{
  "fusion_mode": true,
  "description": "Fusion mode prevents runtime LoRA changes. LoRAs were configured at startup.",
  "lora_info": {
    "name": "username/repo-name",
    "weight": 1.0
  }
}
```

### Error (Fusion Mode Lock)
```json
{
  "detail": "Runtime LoRA changes are disabled in fusion mode. LoRAs were configured at startup and cannot be modified."
}
```

## Tips

1. **Fusion Mode**: Use for production when you want consistent style across all generations
2. **Runtime LoRA**: Use for development/testing when you need flexibility
3. **Queue**: Use for batch processing or handling multiple concurrent requests
4. **Seeds**: Use same seed for reproducible results
5. **Health Checks**: Monitor `/health` endpoint for service status
