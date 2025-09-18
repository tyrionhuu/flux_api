## Quick Start

### 1. Pull the Docker Image from DockerHub
```bash
docker pull eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

### 2. Run the Container
```bash
docker run -d \
  --name kontext-api \
  -p 9300:9300 \
  -e CUDA_VISIBLE_DEVICES=5 \
  -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1 \
  --port 9300 \
  --host 0.0.0.0 \
  --lora-name "Fihade/Apple_Emoji_Style_Kontext_LoRA" \
  --lora-weight 1.0 \
  --loras-config "" \
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
docker logs eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

### View Generation Logs
```bash
# Follow logs in real-time
docker logs -f eigenai/kontext-api-20250918:kontext-api-20250918-v1

# View specific log files
docker exec eigenai/kontext-api-20250918:kontext-api-20250918-v1 tail -f logs/api_routes.log
docker exec eigenai/kontext-api-20250918:kontext-api-20250918-v1 tail -f logs/models_flux_model.log
```
