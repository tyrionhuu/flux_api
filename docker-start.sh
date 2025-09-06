#!/bin/bash

# Docker Container Startup Script
# This script is designed to run inside Docker containers
# It reads environment variables and starts the FLUX API service

set -e

# Function to handle shutdown signals
cleanup() {
    echo "🛑 Received shutdown signal, cleaning up..."
    if [ ! -z "$SERVICE_PID" ]; then
        echo "⏹️  Stopping service (PID: $SERVICE_PID)..."
        kill -TERM "$SERVICE_PID" 2>/dev/null || true
        wait "$SERVICE_PID" 2>/dev/null || true
        echo "✅ Service stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

echo "🐳 Docker Container Startup"
echo "=========================="

# Default values
DEFAULT_PORT=9001
DEFAULT_GPU=""
DEFAULT_FRONTEND="true"
DEFAULT_MODEL="flux"

# Get values from environment variables
PORT=${FP4_API_PORT:-$DEFAULT_PORT}
GPU_ID=${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU}
FRONTEND_ENABLED=${ENABLE_FRONTEND:-$DEFAULT_FRONTEND}
MODEL_TYPE=${MODEL_TYPE:-$DEFAULT_MODEL}

echo "📋 Configuration:"
echo "   Port: $PORT"
echo "   GPU: ${GPU_ID:-"all"}"
echo "   Frontend: $FRONTEND_ENABLED"
echo "   Model: $MODEL_TYPE"
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
    ./start_api.sh -g "$GPU_ID" -p "$PORT" -f "$FRONTEND_ENABLED" -m "$MODEL_TYPE" &
    SERVICE_PID=$!
else
    echo "🚀 Starting with all GPUs on port $PORT..."
    ./start_api.sh -p "$PORT" -f "$FRONTEND_ENABLED" -m "$MODEL_TYPE" &
    SERVICE_PID=$!
fi

# Wait for the service to complete
wait $SERVICE_PID
