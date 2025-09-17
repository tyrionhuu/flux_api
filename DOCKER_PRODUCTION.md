# Kontext API - Production Docker Setup

This document provides a quick reference for the production-ready Docker implementation of Kontext API with LoRA fusion capabilities.

## Quick Start

### Build and Run
```bash
# Build the image
./quick_start.sh build

# Run basic container
./quick_start.sh run

# Run with LoRA fusion
./quick_start.sh run-lora

# Check health
./quick_start.sh health
```

### Manual Commands
```bash
# Build
docker build -t kontext-api-20250918 .

# Run basic
docker run -d --name kontext-api -p 9200:9200 kontext-api-20250918

# Run with LoRA fusion
docker run -d --name kontext-api -p 9200:9200 \
  -e LORA_NAME="my-lora.safetensors" \
  -e LORA_WEIGHT=1.2 \
  -e FUSION_MODE=true \
  kontext-api-20250918
```

## Key Features

### ✅ Production Ready
- Single entry point (`./start_server.sh`)
- Comprehensive health checks (`/health`, `/ready`, `/live`)
- Proper error handling and logging
- Resource management and limits

### ✅ LoRA Fusion
- Environment variable configuration
- Single LoRA: `LORA_NAME` + `LORA_WEIGHT`
- Multiple LoRAs: `LORAS_CONFIG` (JSON)
- Fusion mode prevents runtime changes
- Support for local files and Hugging Face repos

### ✅ EigenAI Compliance
- Repository: `eigenai/kontext-api-20250918`
- Proper tagging strategy (no `latest`)
- Production labels and metadata
- Comprehensive documentation

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | API port | 9200 |
| `HOST` | API host | 0.0.0.0 |
| `FUSION_MODE` | Enable LoRA fusion | false |
| `LORA_NAME` | Single LoRA path/repo | "" |
| `LORA_WEIGHT` | LoRA weight (0.0-2.0) | 1.0 |
| `LORAS_CONFIG` | Multiple LoRAs (JSON) | "" |
| `LOG_LEVEL` | Logging level | INFO |
| `MAX_WORKERS` | Worker processes | 1 |

## Health Endpoints

- **`/health`**: Comprehensive status with fusion mode info
- **`/ready`**: Kubernetes readiness probe
- **`/live`**: Kubernetes liveness probe

## Testing

```bash
# Run comprehensive test suite
./tests/docker/run_tests.sh

# Quick health check
curl http://localhost:9200/health
```

## Deployment

### For EigenDeploy Team
- **Image**: `eigenai/kontext-api-20250918:kontext-api-20250918-v1`
- **Entry Point**: `./start_server.sh`
- **Port**: 9200
- **Health Check**: `http://localhost:9200/health`

### Production Command
```bash
docker run -d \
  --name kontext-api \
  -p 9200:9200 \
  --memory=16g \
  --cpus=8 \
  --gpus=all \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

## Files Created/Modified

### New Files
- `start_server.sh` - Production startup script
- `utils/lora_fusion.py` - LoRA fusion utilities
- `scripts/build_and_push.sh` - Automated build/push
- `tests/docker/run_tests.sh` - Comprehensive testing
- `DEPLOYMENT.md` - Complete deployment guide
- `quick_start.sh` - Quick start commands

### Modified Files
- `Dockerfile` - Production labels, environment variables, entry point
- `main.py` - LoRA fusion integration, enhanced health checks
- `models/flux_model.py` - Fusion mode support
- `api/routes.py` - Fusion mode blocking for LoRA endpoints

## Next Steps

1. **Build and Test**: Use `./quick_start.sh build` and `./quick_start.sh test`
2. **Push to DockerHub**: Use `./scripts/build_and_push.sh`
3. **Handoff**: Provide image name and entry point to EigenDeploy team
4. **Monitor**: Use health endpoints for production monitoring

---

**Status**: ✅ Production Ready  
**Version**: kontext-api-20250918-v1  
**Last Updated**: 2025-09-18
