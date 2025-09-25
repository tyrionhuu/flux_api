## Quick Start

### 1. Pull the Docker Image from DockerHub
```bash
docker pull eigenai/kontext-clean-0924:latest
```

### 2. Run the Container
```bash
docker run -d \
  --name kontext-api \
  --gpus all \
  -p 9300:9300 \
  -e CUDA_VISIBLE_DEVICES=5 \
  -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN \
  eigenai/kontext-clean-0924:latest \
  --port 9300 \
  --host 0.0.0.0 \
  --log-level INFO
```

### 2a. Run with LoRA Fusion (Optional)
```bash
docker run -d \
  --name kontext-api \
  --gpus all \
  -p 9300:9300 \
  -e CUDA_VISIBLE_DEVICES=5 \
  -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN \
  eigenai/kontext-clean-0924:latest \
  --port 9300 \
  --host 0.0.0.0 \
  --lora-name "Fihade/Apple_Emoji_Style_Kontext_LoRA" \
  --lora-weight 1.0 \
  --fusion-mode \
  --log-level INFO
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

### View Generation Logs
```bash
# Follow logs in real-time
docker logs -f kontext-api

# View specific log files
docker exec kontext-api tail -f logs/api_routes.log
docker exec kontext-api tail -f logs/models_flux_model.log
```
