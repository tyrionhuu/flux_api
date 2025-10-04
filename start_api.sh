#!/bin/bash

# Diffusion API Startup Script (Backend Only)
# Supports GPU selection, port configuration, model type, and LoRA fusion

set -e

echo "🚀 Diffusion API Startup Script"
echo "=========================="

# Parse arguments
GPU_ID=""
PORT=8000
MODEL_TYPE="flux"
FUSION_MODE=false
LORA_NAME=""
LORA_WEIGHT=1.0
LORAS_CONFIG=""

usage() {
    echo "Usage: $0 [-g <gpu_ids>] [-p <port>] [-m <model_type>] [--fusion-mode] [--lora-name <name>] [--lora-weight <weight>] [--loras-config <json>]"
    echo "  -g <gpu_ids>         GPU index or comma-separated list (e.g., 1 or 1,2,3)"
    echo "  -p <port>            Port number for the API service (default: 8000)"
    echo "  -m <model_type>      Model type to load (flux/qwen, default: flux)"
    echo "  --fusion-mode        Enable LoRA fusion mode"
    echo "  --lora-name <name>   LoRA file path or HF repo for fusion"
    echo "  --lora-weight <weight> LoRA weight (default: 1.0)"
    echo "  --loras-config <json> JSON config for multiple LoRAs"
}

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g)
            GPU_ID="$2"
            shift 2
            ;;
        -p)
            PORT="$2"
            shift 2
            ;;
        -m)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --fusion-mode)
            FUSION_MODE=true
            shift
            ;;
        --lora-name)
            LORA_NAME="$2"
            shift 2
            ;;
        --lora-weight)
            LORA_WEIGHT="$2"
            shift 2
            ;;
        --loras-config)
            LORAS_CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Function to cleanup port
cleanup_port() {
    local port=$1
    echo "🧹 Checking port $port for conflicts..."

    # Find processes using the port
    local pids=$(lsof -ti:$port 2>/dev/null || echo "")

    if [ -n "$pids" ]; then
        echo "   Found processes using port $port: $pids"

        # Kill processes gracefully first
        for pid in $pids; do
            echo "   🚫 Terminating process $pid..."
            kill -TERM $pid 2>/dev/null || true
        done

        # Wait a bit for graceful shutdown
        sleep 2

        # Check if processes are still running
        local remaining_pids=$(lsof -ti:$port 2>/dev/null || echo "")

        if [ -n "$remaining_pids" ]; then
            echo "   💀 Force killing remaining processes..."
            for pid in $remaining_pids; do
                kill -KILL $pid 2>/dev/null || true
            done
            sleep 1
        fi

        # Final check
        local final_check=$(lsof -ti:$port 2>/dev/null || echo "")
        if [ -n "$final_check" ]; then
            echo "   ⚠️  Port $port still in use after cleanup"
            return 1
        else
            echo "   ✅ Port $port is now free"
            return 0
        fi
    else
        echo "   ✅ Port $port is free"
        return 0
    fi
}

# Function to wait for port to be available
wait_for_port() {
    local port=$1
    local max_wait=${2:-30}
    local wait_time=0

    echo "⏳ Waiting for port $port to become available..."

    while [ $wait_time -lt $max_wait ]; do
        if ! lsof -i:$port >/dev/null 2>&1; then
            echo "   ✅ Port $port is available"
            return 0
        fi
        sleep 1
        wait_time=$((wait_time + 1))
        echo "   ⏳ Still waiting... (${wait_time}s)"
    done

    echo "   ❌ Port $port did not become available within $max_wait seconds"
    return 1
}

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory!"
    echo "   Please run this script from the txt2img directory."
    exit 1
fi

echo "✅ Environment check passed"

# Clean up selected port
if ! cleanup_port "$PORT"; then
    echo "⚠️  Port cleanup incomplete, but continuing..."
fi

# Wait for port to be available
if ! wait_for_port "$PORT" 30; then
    echo "❌ Port $PORT is not available, cannot start service"
    exit 1
fi

echo "🚀 Starting Diffusion API Service..."

# Set GPU visibility if specified
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "🔧 Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    echo "🔧 No -g provided; using all visible GPUs"
fi

# Export configuration environment variables
export API_PORT="$PORT"
export MODEL_TYPE="$MODEL_TYPE"

# Export fusion mode configuration
if [ "$FUSION_MODE" = true ]; then
    export FUSION_MODE="true"
    if [ -n "$LORA_NAME" ]; then
        export LORA_NAME="$LORA_NAME"
        export LORA_WEIGHT="$LORA_WEIGHT"
    fi
    if [ -n "$LORAS_CONFIG" ]; then
        export LORAS_CONFIG="$LORAS_CONFIG"
    fi
fi

echo "🔧 Configuration:"
echo "   API_PORT=${API_PORT}"
echo "   MODEL_TYPE=${MODEL_TYPE}"
if [ "$FUSION_MODE" = true ]; then
    echo "   FUSION_MODE=true"
    [ -n "$LORA_NAME" ] && echo "   LORA_NAME=${LORA_NAME}"
    [ -n "$LORA_WEIGHT" ] && echo "   LORA_WEIGHT=${LORA_WEIGHT}"
    [ -n "$LORAS_CONFIG" ] && echo "   LORAS_CONFIG=${LORAS_CONFIG}"
fi

# Build Python command with arguments
PYTHON_CMD="python main.py --port ${PORT}"

if [ "$FUSION_MODE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --fusion-mode"

    if [ -n "$LORA_NAME" ]; then
        PYTHON_CMD="$PYTHON_CMD --lora-name \"${LORA_NAME}\" --lora-weight ${LORA_WEIGHT}"
    fi

    if [ -n "$LORAS_CONFIG" ]; then
        PYTHON_CMD="$PYTHON_CMD --loras-config '${LORAS_CONFIG}'"
    fi
fi

echo "🚀 Executing: $PYTHON_CMD"
eval $PYTHON_CMD
