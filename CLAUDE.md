# CLAUDE.md

RULE: ALWAYS CODE IN LINUS TORVALDS STYLE

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is the Sekai API - a multi-GPU FLUX image generation service running 8 instances across 8 GPUs with nginx load balancing.

The service uses:
- **Nunchaku FLUX model** (nunchaku-tech/nunchaku-flux.1-dev) with FP4 quantization
- **LoRA support** with automatic merging for multiple LoRA layers (max 3)
- **Queue management** for concurrent request handling (max 2 concurrent, queue size 100)
- **JPEG output** with 65% quality compression

### Planned Features (see docs/Sekai_API_Update_Plan.md)
- NSFW content detection using Falconsai/nsfw_image_detection model
- S3 upload with pre-signed URLs (direct PUT to provided URLs)
- Enhanced /generate endpoint with response_format, upscale, and enable_nsfw_check parameters

## Key Commands

```bash
# Start multi-GPU service (8 GPUs with load balancing)
./start_multi_gpu.sh -m fp4_sekai

# Monitor running services
./monitor_multi_gpu.sh

# Run pressure testing at 2 RPS
./run_pressure_test.sh 2

# Start single instance for development
python main_fp4_sekai.py  # Runs on port 8000

# Check GPU status and memory
nvidia-smi

# View service logs
tail -f logs/flux_api_fp4.log
tail -f logs/multi_gpu/gpu_*.log
```

## Project Structure

- **api/sekai_routes.py**: Main API endpoints for Sekai service including /generate endpoint
- **models/fp4_flux_model.py**: FLUX model manager with LoRA support and GPU management
- **config/sekai_settings.py**: Configuration for Sekai deployment (ports, LoRA paths, image settings)
- **utils/**: GPU manager, image utilities, queue management, cleanup service
- **start_multi_gpu.sh**: Deployment script for multi-GPU setup with nginx load balancing

## Development Notes

### Sekai API Current Implementation
- Binary image output (JPEG format, 65% quality)
- Default LoRAs at /data/pingzhi/checkpoints/
- Multi-GPU deployment across 8 GPUs
- Load balancing via nginx on port 8080

### Upcoming API Changes (per docs/Sekai_API_Update_Plan.md)
- NSFW detection with 5-second timeout (failure/timeout returns nsfw:true)
- S3 upload using pre-signed URLs (PUT request directly to provided URL)
- Response format will include s3_url and nsfw_score
- New parameters: response_format, upscale, s3_prefix, enable_nsfw_check

### GPU and Thread Management
- Each instance auto-configures PyTorch threads based on NUM_GPU_INSTANCES environment variable
- Uses CUDA_VISIBLE_DEVICES to isolate GPU per instance
- Thread limits prevent oversubscription: threads_per_instance = max(1, num_cores // num_gpu_instances)

### LoRA Configuration
Default LoRAs are configured in config/sekai_settings.py:
- lora_1_weight_1.safetensors (weight: 1.0)
- lora_2_weight_0_7.safetensors (weight: 0.7)  
- lora_3_weight_0_5.safetensors (weight: 0.5)

### Testing
Use run_pressure_test.sh for load testing - sends requests at controlled RPS to nginx load balancer on port 8080.

### Important File Paths
- Generated images: /data/pingzhi/generated_images/
- LoRA checkpoints: /data/pingzhi/checkpoints/
- Uploaded LoRAs: uploads/lora_files/

### Current Branch
Working on prod/sekai branch (main branch is master).