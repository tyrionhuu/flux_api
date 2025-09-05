#!/bin/bash

# Docker Container Startup Script
# This script is designed to run inside Docker containers
# It reads environment variables and starts the FLUX API service

set -e

echo "🐳 Docker Container Startup"
echo "=========================="

# Default values
DEFAULT_PORT=9001
DEFAULT_GPU=""

# Get values from environment variables
PORT=${FP4_API_PORT:-$DEFAULT_PORT}
GPU_ID=${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU}

echo "📋 Configuration:"
echo "   Port: $PORT"
echo "   GPU: ${GPU_ID:-"all"}"
echo "   Frontend: ${ENABLE_FRONTEND:-"enabled"}"
echo "   Token: ${HUGGINGFACE_HUB_TOKEN:0:10}..."
echo ""

# Check if token is provided
if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "❌ Error: HUGGINGFACE_HUB_TOKEN environment variable is required"
    echo "   Set it with: -e HUGGINGFACE_HUB_TOKEN=your_token_here"
    exit 1
fi

# Export the token for the child process
export HUGGINGFACE_HUB_TOKEN

# Start the service with appropriate parameters
if [ -n "$GPU_ID" ]; then
    echo "🚀 Starting with GPU $GPU_ID on port $PORT..."
    ./start_flux_api.sh -g "$GPU_ID" -p "$PORT"
else
    echo "🚀 Starting with all GPUs on port $PORT..."
    ./start_flux_api.sh -p "$PORT"
fi
