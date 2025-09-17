#!/bin/bash
set -e

# Kontext API Production Startup Script
# This script handles all startup logic including LoRA fusion

echo "🚀 Starting Kontext API Server..."
echo "=================================="

# Parse environment variables with defaults
PORT=${PORT:-9200}
HOST=${HOST:-0.0.0.0}
SERVICE_NAME=${SERVICE_NAME:-"kontext-api"}
SERVICE_VERSION=${SERVICE_VERSION:-"kontext-api-20250918-v1"}

# LoRA fusion configuration
LORA_NAME=${LORA_NAME:-""}
LORA_WEIGHT=${LORA_WEIGHT:-1.0}
LORAS_CONFIG=${LORAS_CONFIG:-""}
FUSION_MODE=${FUSION_MODE:-false}

# Production configuration
LOG_LEVEL=${LOG_LEVEL:-INFO}
MAX_WORKERS=${MAX_WORKERS:-1}

# Display configuration
echo "📋 Configuration:"
echo "   Service: $SERVICE_NAME"
echo "   Version: $SERVICE_VERSION"
echo "   Port: $PORT"
echo "   Host: $HOST"
echo "   Fusion Mode: $FUSION_MODE"
echo "   LoRA Name: ${LORA_NAME:-"None"}"
echo "   LoRA Weight: $LORA_WEIGHT"
echo "   LoRAs Config: ${LORAS_CONFIG:-"None"}"
echo "   Log Level: $LOG_LEVEL"
echo "   Max Workers: $MAX_WORKERS"
echo ""

# Validate LoRA configuration if provided
if [ "$FUSION_MODE" = "true" ]; then
    echo "🔧 LoRA Fusion Mode Enabled"
    
    if [ -n "$LORA_NAME" ] && [ -n "$LORAS_CONFIG" ]; then
        echo "❌ Error: Cannot specify both LORA_NAME and LORAS_CONFIG"
        exit 1
    fi
    
    if [ -z "$LORA_NAME" ] && [ -z "$LORAS_CONFIG" ]; then
        echo "❌ Error: Fusion mode requires either LORA_NAME or LORAS_CONFIG"
        exit 1
    fi
    
    echo "✅ LoRA configuration validated"
fi

# Check if conda environment exists
if [ ! -d "/opt/conda/envs/img2img" ]; then
    echo "❌ Error: Conda environment 'img2img' not found"
    exit 1
fi

# Activate conda environment
echo "🐍 Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate img2img

# Verify Python executable
PYTHON_EXEC="/opt/conda/envs/img2img/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "❌ Error: Python executable not found at $PYTHON_EXEC"
    exit 1
fi

echo "✅ Python executable found: $PYTHON_EXEC"

# Build command arguments
CMD_ARGS=(
    "$PYTHON_EXEC"
    "main.py"
    "--port" "$PORT"
    "--host" "$HOST"
)

# Add LoRA fusion arguments if fusion mode is enabled
if [ "$FUSION_MODE" = "true" ]; then
    CMD_ARGS+=("--fusion-mode")
    
    if [ -n "$LORA_NAME" ]; then
        CMD_ARGS+=("--lora-name" "$LORA_NAME")
        CMD_ARGS+=("--lora-weight" "$LORA_WEIGHT")
    fi
    
    if [ -n "$LORAS_CONFIG" ]; then
        CMD_ARGS+=("--loras-config" "$LORAS_CONFIG")
    fi
fi

# Add production configuration
CMD_ARGS+=("--log-level" "$LOG_LEVEL")
CMD_ARGS+=("--max-workers" "$MAX_WORKERS")

echo "🚀 Starting service with command:"
echo "   ${CMD_ARGS[*]}"
echo ""

# Start the service
exec "${CMD_ARGS[@]}"
