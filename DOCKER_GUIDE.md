### 1. Build the Docker Image
```bash
docker build -t kontext-api .
```

### 2. Run the Container
```bash
docker run -d \
  --name kontext-api \
  --gpus all \
  -p 9300:9300 \
  -e CUDA_VISIBLE_DEVICES=5 \
  -e HUGGINGFACE_HUB_TOKEN=your_hf_token_here \
  kontext-api
```

### 3. Check Status
```bash
# Check if container is running
docker ps | grep kontext

# Check health
curl http://localhost:9300/health

# View logs
docker logs kontext-api
```

## Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device to use (e.g., `5`)
- `HUGGINGFACE_HUB_TOKEN`: Your Hugging Face token for LoRA downloads
- `LORA_NAME`: LoRA to fuse at startup (default: `Fihade/Apple_Emoji_Style_Kontext_LoRA`)
- `LORA_WEIGHT`: LoRA weight (default: `1.0`)

### Port Configuration
- **Container Port**: 9300
- **Host Port**: 9300 (or any available port)
- **Health Check**: `http://localhost:9300/health`

## Usage Examples

### Generate Image
```bash
curl -X POST "http://localhost:9300/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Turn it to APPLE_EMOJI, white background, a portrait of a person smiling",
    "width": 512,
    "height": 512
  }'
```

### Check Model Status
```bash
curl http://localhost:9300/model-status
```

### View Generation Logs
```bash
# Follow logs in real-time
docker logs -f kontext-api

# View specific log files
docker exec kontext-api tail -f logs/api_routes.log
docker exec kontext-api tail -f logs/models_flux_model.log
```

## Management Commands

### Stop Container
```bash
docker stop kontext-api
```

### Remove Container
```bash
docker rm kontext-api
```

### Restart Container
```bash
docker restart kontext-api
```

### Update Container
```bash
# Stop and remove old container
docker stop kontext-api && docker rm kontext-api

# Rebuild image
docker build -t kontext-api .

# Run new container
docker run -d \
  --name kontext-api \
  --gpus all \
  -p 9300:9300 \
  -e CUDA_VISIBLE_DEVICES=5 \
  -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN \
  kontext-api
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs kontext-api

# Check if port is available
netstat -tulpn | grep 9300
```

### GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA in container
docker exec kontext-api nvidia-smi
```

### LoRA Issues
```bash
# Check LoRA status
curl http://localhost:9300/health | jq '.lora_info'

# Check LoRA logs
docker exec kontext-api tail -f logs/models_flux_model.log
```

## Production Deployment

### Docker Compose (Optional)
```yaml
version: '3.8'
services:
  kontext-api:
    image: kontext-api
    ports:
      - "9300:9300"
    environment:
      - CUDA_VISIBLE_DEVICES=5
      - HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Health Monitoring
```bash
# Kubernetes-style health check
curl -f http://localhost:9300/health || exit 1

# Readiness check
curl -f http://localhost:9300/health | jq '.model_ready'
```

## Key Features

- ✅ **LoRA Fusion**: LoRA automatically fused at startup
- ✅ **GPU Support**: CUDA acceleration with NVIDIA GPUs
- ✅ **Health Checks**: Built-in health monitoring
- ✅ **Logging**: Comprehensive logging to files
- ✅ **Production Ready**: Single entry point, proper error handling

## API Endpoints

- `GET /health` - Health check
- `GET /model-status` - Model status and LoRA info
- `POST /generate` - Generate images
- `GET /download/{filename}` - Download generated images

---

**Note**: This container uses port 9300 and requires a valid Hugging Face token for LoRA downloads.

