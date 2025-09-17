# Kontext API Deployment Guide

This guide provides comprehensive instructions for deploying the Kontext API Docker image in production environments, following EigenAI deployment standards.

## Overview

**Service Name**: Kontext API  
**Version**: kontext-api-20250918-v1  
**Docker Repository**: `eigenai/kontext-api-20250918`  
**Port**: 9200 (default)  
**Base Image**: NVIDIA CUDA 12.8 with Ubuntu 24.04  

## Docker Image Information

### Repository Details
- **Repository**: `eigenai/kontext-api-20250918`
- **Tags**: `kontext-api-20250918-v1`, `kontext-api-20250918-v1-YYYYMMDD-HHMMSS`
- **Type**: Private repository (requires EigenAI DockerHub access)
- **Size**: ~15GB (includes CUDA runtime and dependencies)

### Image Labels
```dockerfile
LABEL maintainer="EigenAI"
LABEL version="kontext-api-20250918-v1"
LABEL description="Kontext API - FLUX Image Generation with LoRA Fusion"
LABEL service="kontext-api"
LABEL date="2025-09-18"
```

## Entry Point

The Docker image uses a single, production-ready entry point:

```dockerfile
ENTRYPOINT ["./start_server.sh"]
```

The startup script handles:
- Environment variable parsing
- LoRA fusion configuration
- Model loading with fusion
- Health check preparation
- Error handling and logging

## Environment Variables

### Core Service Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 9200 | API port number |
| `HOST` | 0.0.0.0 | API host address |
| `SERVICE_NAME` | kontext-api | Service identifier |
| `SERVICE_VERSION` | kontext-api-20250918-v1 | Service version |

### LoRA Fusion Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `FUSION_MODE` | false | Enable LoRA fusion mode |
| `LORA_NAME` | "" | Single LoRA file path or HF repo |
| `LORA_WEIGHT` | 1.0 | LoRA weight (0.0-2.0) |
| `LORAS_CONFIG` | "" | JSON config for multiple LoRAs |

### Production Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | INFO | Logging level |
| `MAX_WORKERS` | 1 | Maximum worker processes |

## Deployment Examples

### Basic Deployment
```bash
docker run -d \
  --name kontext-api \
  -p 9200:9200 \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

### LoRA Fusion Deployment (Single LoRA)
```bash
docker run -d \
  --name kontext-api-with-lora \
  -p 9200:9200 \
  -e LORA_NAME="my-style-lora.safetensors" \
  -e LORA_WEIGHT=1.2 \
  -e FUSION_MODE=true \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

### LoRA Fusion Deployment (Multiple LoRAs)
```bash
docker run -d \
  --name kontext-api-multi-lora \
  -p 9200:9200 \
  -e LORAS_CONFIG='[{"name": "style.safetensors", "weight": 0.8}, {"name": "character.safetensors", "weight": 1.0}]' \
  -e FUSION_MODE=true \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

### Hugging Face LoRA Deployment
```bash
docker run -d \
  --name kontext-api-hf-lora \
  -p 9200:9200 \
  -e LORA_NAME="username/my-lora-repo" \
  -e LORA_WEIGHT=1.0 \
  -e FUSION_MODE=true \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

### Production Deployment with Resource Limits
```bash
docker run -d \
  --name kontext-api-prod \
  -p 9200:9200 \
  --memory=16g \
  --cpus=8 \
  --gpus=all \
  -e LOG_LEVEL=INFO \
  -e MAX_WORKERS=1 \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

## Health Check Endpoints

### `/health` - Comprehensive Health Check
```bash
curl http://localhost:9200/health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "Kontext API",
  "version": "kontext-api-20250918-v1",
  "model_loaded": true,
  "model_ready": true,
  "fusion_mode": false,
  "lora_info": null,
  "timestamp": 1734567890.123,
  "uptime": 120.5
}
```

### `/ready` - Kubernetes Readiness Probe
```bash
curl http://localhost:9200/ready
```

**Response**:
```json
{
  "status": "ready"
}
```

### `/live` - Kubernetes Liveness Probe
```bash
curl http://localhost:9200/live
```

**Response**:
```json
{
  "status": "alive"
}
```

## Docker Health Check

The image includes a built-in health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD conda run -n img2img curl -f http://localhost:9200/health || exit 1
```

## LoRA Fusion Mode

### What is Fusion Mode?
Fusion mode applies LoRA weights to the model during startup and prevents runtime LoRA changes. This ensures:
- Consistent model behavior
- No runtime LoRA switching overhead
- Production-grade stability

### Enabling Fusion Mode
Set `FUSION_MODE=true` and provide either:
- `LORA_NAME` for single LoRA
- `LORAS_CONFIG` for multiple LoRAs

### Fusion Mode Behavior
- LoRA is applied once during model loading
- Runtime LoRA changes are blocked
- `/apply-lora` and `/remove-lora` endpoints return errors
- Model behavior remains consistent throughout container lifetime

## Resource Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 20GB free space

### Recommended Production
- **CPU**: 8+ cores
- **Memory**: 16GB+ RAM
- **GPU**: NVIDIA RTX 4090 or better
- **Storage**: 50GB+ free space

## Security Considerations

### Network Security
- API runs on port 9200 by default
- Use reverse proxy (nginx/traefik) for production
- Implement rate limiting
- Use HTTPS in production

### Container Security
- Run as non-root user (handled by conda environment)
- Use read-only filesystem where possible
- Implement resource limits
- Regular security updates

## Monitoring and Logging

### Logging
- Structured logging to stdout
- Log files in `/app/logs/` directory
- Configurable log levels via `LOG_LEVEL` environment variable

### Monitoring
- Health check endpoints for monitoring systems
- Prometheus metrics (optional)
- Resource usage monitoring
- Model loading status tracking

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check container logs
docker logs kontext-api

# Check resource usage
docker stats kontext-api

# Verify GPU access
docker run --rm --gpus=all nvidia/cuda:12.8.0-runtime-ubuntu24.04 nvidia-smi
```

#### Health Check Failures
```bash
# Check if port is accessible
curl -v http://localhost:9200/health

# Check container status
docker ps -a

# Check resource limits
docker inspect kontext-api | grep -A 10 "Resources"
```

#### LoRA Fusion Issues
```bash
# Check LoRA file exists
docker exec kontext-api ls -la uploads/lora_files/

# Verify LoRA configuration
docker exec kontext-api env | grep LORA

# Check fusion mode status
curl http://localhost:9200/health | jq '.fusion_mode'
```

### Performance Tuning

#### Memory Optimization
```bash
# Increase shared memory for large models
docker run -d \
  --shm-size=8g \
  --name kontext-api \
  -p 9200:9200 \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

#### GPU Optimization
```bash
# Use specific GPU
docker run -d \
  --gpus='"device=0"' \
  --name kontext-api \
  -p 9200:9200 \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

## Building and Pushing

### Local Build
```bash
# Build image
docker build -t kontext-api-20250918 .

# Tag for push
docker tag kontext-api-20250918 eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

### Automated Build and Push
```bash
# Set DockerHub token
export DOCKERHUB_TOKEN=your_token_here

# Run build script
./scripts/build_and_push.sh
```

### Manual Push
```bash
# Login to DockerHub
docker login -u eigenai -p $DOCKERHUB_TOKEN

# Push image
docker push eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

## Testing

### Run Test Suite
```bash
# Run comprehensive tests
./tests/docker/run_tests.sh
```

### Manual Testing
```bash
# Test basic functionality
curl http://localhost:9200/

# Test health check
curl http://localhost:9200/health

# Test model status
curl http://localhost:9200/model-status
```

## Handoff to EigenDeploy Team

### Required Information
1. **Docker Image**: `eigenai/kontext-api-20250918:kontext-api-20250918-v1`
2. **Entry Point**: `./start_server.sh`
3. **Port**: 9200
4. **Health Check**: `http://localhost:9200/health`

### Deployment Command
```bash
docker run -d \
  --name kontext-api \
  -p 9200:9200 \
  eigenai/kontext-api-20250918:kontext-api-20250918-v1
```

### Verification
```bash
curl http://localhost:9200/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Kontext API",
  "version": "kontext-api-20250918-v1",
  "model_loaded": true,
  "model_ready": true,
  "fusion_mode": false,
  "lora_info": null,
  "timestamp": 1734567890.123,
  "uptime": 120.5
}
```

## Support

For issues or questions:
- Check logs: `docker logs kontext-api`
- Run tests: `./tests/docker/run_tests.sh`
- Contact EigenAI deployment team

---

**Last Updated**: 2025-09-18  
**Version**: kontext-api-20250918-v1
