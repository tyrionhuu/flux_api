## Docker Usage Guide

This guide explains how to run the FLUX API using Docker.

### 1. Build the Docker Image

```bash
docker build -t img2img-api .
```

### 2. Run the Container

```bash
# Basic usage with Hugging Face token
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  --name img2img-api \
  img2img-api
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_HUB_TOKEN` | Required | Your Hugging Face token for model access |
| `CUDA_VISIBLE_DEVICES` | all | GPU IDs to use (comma-separated) |

### Command-Line Arguments

The Docker container runs with these default arguments:
- `--port 9200` - API port (can be overridden with custom CMD)
- `--cleanup` - Enable cleanup of existing processes
- `--no-frontend` - Disable frontend (backend-only mode) - frontend is enabled by default

### Port Mapping

The container exposes multiple ports:
- `9200` - Main API port (default)

### Basic Usage

```bash
# Run with default settings
docker run -d \
  --gpus all \
  -p 9200:9200 \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  --name img2img-api \
  img2img-api
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
  --name img2img-api-gpu0 \
  img2img-api
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
  --name img2img-api \
  img2img-api
```