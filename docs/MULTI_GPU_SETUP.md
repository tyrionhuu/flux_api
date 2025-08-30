# Multi-GPU FLUX API Setup Guide

This guide explains how to deploy FLUX API across multiple GPUs for high-concurrency image generation.

## Architecture Overview

```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    Nginx    │ (Port 80)
                    │Load Balancer│
                    └──────┬──────┘
                           │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼───────┐  ┌───────────┐  ┌────────▼──────┐
│  GPU 0:8000   │  │    ...    │  │  GPU 7:8007   │
│  FLUX Model   │  │           │  │  FLUX Model   │
└───────────────┘  └───────────┘  └───────────────┘
```

## Prerequisites

1. **Multiple GPUs**: System with 2-8 NVIDIA GPUs
2. **Nginx**: Install with `sudo apt-get install nginx`
3. **Sufficient VRAM**: 
   - FP4 model: ~8GB per GPU
   - BF16 model: ~16GB per GPU

## Quick Start

### 1. Start Multi-GPU Service

```bash
# Start with FP4 models (default, lower memory)
./start_multi_gpu.sh

# Or start with BF16 models (higher quality, more memory)
./start_multi_gpu.sh -m bf16

# Stop all services
./start_multi_gpu.sh stop
```

### 2. Monitor Services

```bash
# Real-time monitoring
./monitor_multi_gpu.sh

# Single status check
./monitor_multi_gpu.sh --once

# Show with error logs
./monitor_multi_gpu.sh --errors
```

### 3. Access the API

Single endpoint for all GPUs:
```bash
# Generate image (load balanced across all GPUs)
curl -X POST "http://localhost/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful mountain landscape",
    "width": 512,
    "height": 512
  }'
```

## Configuration

### Adjusting Number of GPUs

1. Edit `start_multi_gpu.sh`:
```bash
NUM_GPUS=4  # Change from 8 to your GPU count
```

2. Edit `nginx.conf`:
```nginx
upstream flux_api_backend {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;  # GPU 0
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;  # GPU 1
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;  # GPU 2
    server 127.0.0.1:8003 max_fails=3 fail_timeout=30s;  # GPU 3
    # Remove or add lines as needed
}
```

3. Edit `monitor_multi_gpu.sh`:
```bash
NUM_GPUS=4  # Match your GPU count
```

### Custom Nginx Configuration

To use your own Nginx config:
```bash
./start_multi_gpu.sh -c /path/to/custom/nginx.conf
```

### Port Configuration

Services use ports 8000-8007 by default. To change:

1. Edit `start_multi_gpu.sh`:
```bash
BASE_PORT=9000  # Change base port
```

2. Update `nginx.conf` accordingly

## Systemd Service (Production)

### Install Service

```bash
# Copy service file
sudo cp services/flux-multi-gpu.service /etc/systemd/system/

# Edit service file to set correct user and paths
sudo nano /etc/systemd/system/flux-multi-gpu.service

# Enable and start
sudo systemctl enable flux-multi-gpu
sudo systemctl start flux-multi-gpu
```

### Manage Service

```bash
# Check status
sudo systemctl status flux-multi-gpu

# View logs
sudo journalctl -u flux-multi-gpu -f

# Restart
sudo systemctl restart flux-multi-gpu
```

## Performance Tuning

### 1. Queue Configuration

Each GPU instance has its own queue. Default settings:
- Max concurrent requests per GPU: 2
- Max queue size per GPU: 100

To adjust, modify in `api/fp4_routes.py` or `api/bf16_routes.py`:
```python
queue_manager = QueueManager(max_concurrent=3, max_queue_size=200)
```

### 2. Nginx Optimization

For very high concurrency, tune Nginx:

```nginx
# Add to nginx.conf http block
worker_processes auto;
worker_connections 4096;

# Add to upstream block
keepalive 64;  # Increase from 32
```

### 3. GPU-Specific Tuning

Assign specific models to specific GPUs:
```bash
# Edit start_multi_gpu.sh to use different models per GPU
if [ $gpu_id -lt 4 ]; then
    # Use FP4 for first 4 GPUs
    export FP4_API_PORT=$port
    nohup python main_fp4.py > "$log_file" 2>&1 &
else
    # Use BF16 for remaining GPUs
    export BF16_API_PORT=$port
    nohup python main_bf16.py > "$log_file" 2>&1 &
fi
```

## Monitoring and Debugging

### Check Individual Services

```bash
# Check specific GPU service
curl http://localhost:8003/  # GPU 3

# Check queue status
curl http://localhost:8003/queue-stats
```

### View Logs

```bash
# All logs are in logs/multi_gpu/
tail -f logs/multi_gpu/flux_gpu0_port8000.log

# Check Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### GPU Utilization

```bash
# Watch GPU usage
nvidia-smi -l 1

# Or use the monitor script
./monitor_multi_gpu.sh
```

## Troubleshooting

### Services Won't Start

1. Check GPU availability:
```bash
nvidia-smi
```

2. Check port conflicts:
```bash
lsof -ti:8000-8007
```

3. Check logs:
```bash
tail logs/multi_gpu/*.log
```

### Nginx Issues

1. Test configuration:
```bash
sudo nginx -t -c $PWD/nginx.conf
```

2. Check if Nginx is installed:
```bash
which nginx || sudo apt-get install nginx
```

3. Permission issues:
```bash
# Run without sudo first, it will fallback to direct access
./start_multi_gpu.sh
```

### Uneven Load Distribution

The least_conn algorithm should distribute load evenly. If not:

1. Check if all services are healthy:
```bash
./monitor_multi_gpu.sh --once
```

2. Consider using different algorithms in nginx.conf:
```nginx
# Round-robin (default)
upstream flux_api_backend {
    server 127.0.0.1:8000;
    # ...
}

# IP hash (sticky sessions)
upstream flux_api_backend {
    ip_hash;
    server 127.0.0.1:8000;
    # ...
}
```

## Advanced Configurations

### Mixed Model Deployment

Deploy both FP4 and BF16 models for flexibility:

1. Create `nginx_mixed.conf` with two upstreams
2. Route by URL path:
```nginx
location /generate/fast {
    proxy_pass http://flux_fp4_backend;
}

location /generate/quality {
    proxy_pass http://flux_bf16_backend;
}
```

### Docker Deployment

See `docker-compose.multi-gpu.yml` (if using Docker):
```yaml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    
  flux-gpu0:
    build: .
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - FP4_API_PORT=8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
```

### High Availability

For production HA:
1. Use external load balancer (HAProxy, AWS ALB)
2. Health checks on `/health` endpoint
3. Auto-restart failed services
4. Monitor with Prometheus/Grafana

## Best Practices

1. **Start with FP4**: Lower memory allows more concurrent requests
2. **Monitor regularly**: Use monitoring script during high load
3. **Scale gradually**: Start with fewer GPUs and scale up
4. **Log rotation**: Set up logrotate for production
5. **Resource limits**: Set ulimits for production stability

## Integration Examples

### Python Client

```python
import requests

# Single API endpoint for all GPUs
API_URL = "http://localhost/generate"

response = requests.post(API_URL, json={
    "prompt": "A futuristic city",
    "width": 512,
    "height": 512,
    "seed": 42
})

if response.status_code == 200:
    result = response.json()
    print(f"Image URL: {result['image_url']}")
```

### Load Testing

```bash
# Install wrk
sudo apt-get install wrk

# Test with 100 connections, 8 threads, 30 seconds
wrk -t8 -c100 -d30s -s load_test.lua http://localhost/generate
```

## Conclusion

The multi-GPU setup provides:
- **Linear scaling**: Each GPU adds ~2x throughput
- **High availability**: Failed GPUs don't affect others
- **Simple management**: Single entry point
- **Easy monitoring**: Built-in tools

For questions or issues, check the logs first, then the monitoring tool.