# Docker Usage Guide

This guide explains how to run the FLUX API using Docker.

## Prerequisites

- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)
- Hugging Face token for model access

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t flux-api .
```

### 2. Run the Container

```bash
# Basic usage with Hugging Face token
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  --name flux-api \
  flux-api
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_HUB_TOKEN` | Required | Your Hugging Face token for model access |
| `FP4_API_PORT` | 9200 | Port for the API service |
| `CUDA_VISIBLE_DEVICES` | all | GPU IDs to use (comma-separated) |
| `ENABLE_FRONTEND` | enabled | Enable/disable the web frontend |

### Port Mapping

The container exposes multiple ports:
- `9200` - Main API port (default)

## Usage Examples

### Basic Usage

```bash
# Run with default settings
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  --name flux-api \
  flux-api
```

### Custom Port and GPU

```bash
# Use specific GPU and port
docker run -d \
  --gpus '"device=0"' \
  -p 9002:9002 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -e FP4_API_PORT=9002 \
  -e CUDA_VISIBLE_DEVICES=0 \
  --name flux-api-gpu0 \
  flux-api
```

### Multiple GPUs

```bash
# Use multiple GPUs
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --name flux-api-multi-gpu \
  flux-api
```

### With Volume Mounts

```bash
# Mount directories for persistent storage
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -v $(pwd)/generated_images:/app/generated_images \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/uploads:/app/uploads \
  --name flux-api \
  flux-api
```

## Container Management

### View Logs

```bash
# View container logs
docker logs flux-api

# Follow logs in real-time
docker logs -f flux-api
```

### Stop Container

```bash
# Stop the container
docker stop flux-api

# Stop and remove the container
docker stop flux-api && docker rm flux-api
```

### Restart Container

```bash
# Restart the container
docker restart flux-api
```

### Access Container Shell

```bash
# Access the container shell
docker exec -it flux-api /bin/bash
```