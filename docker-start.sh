#!/bin/bash

# Docker startup script for FLUX API
# This script handles GPU initialization and proper startup

set -e

echo "🐳 FLUX API Docker Startup Script"
echo "=================================="

# Default configuration (can be overridden by args or env)
PORT_DEFAULT=9000
GPUS_DEFAULT=all

# Parsed arguments
PORT_ARG=""
GPUS_ARG=""

# Function to print usage
print_usage() {
    echo "Usage: $0 [-p PORT|--port PORT] [-g GPUS|--gpus GPUS]"
    echo "  -p, --port   API port (default: ${PORT_DEFAULT})"
    echo "  -g, --gpus   GPUs to expose, e.g. 'all', '0', '0,1' (default: ${GPUS_DEFAULT})"
}

# Function to parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--port)
                PORT_ARG="$2"
                shift 2
                ;;
            -g|--gpus)
                GPUS_ARG="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unknown argument: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# Function to check if NVIDIA runtime is available
check_nvidia_runtime() {
    echo "Checking NVIDIA runtime availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi found"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    else
        echo "nvidia-smi not found - GPU support may not be available"
        return 1
    fi
    
    # Check if CUDA is available in Python
    python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || {
        echo "PyTorch CUDA check failed"
        return 1
    }
}

# Function to create necessary directories
setup_directories() {
    echo "Setting up directories..."
    
    mkdir -p logs generated_images uploads/lora_files cache/merged_loras cache/nunchaku_loras
    
    # Set proper permissions
    chmod 755 logs generated_images uploads cache
    chmod 755 uploads/lora_files cache/merged_loras cache/nunchaku_loras
    
    echo "Directories created and permissions set"
}

# Function to check port availability
check_port() {
    local port=${1:-9000}
    echo "Checking port $port availability..."
    
    if lsof -i:$port >/dev/null 2>&1; then
        echo "Port $port is already in use"
        return 1
    else
        echo "Port $port is available"
        return 0
    fi
}

# Function to start the API service
start_api() {
    echo "Starting FLUX API service..."
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
    export FP4_API_PORT=${FP4_API_PORT:-9000}
    export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
    export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
    
    echo "Environment variables:"
    echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "   FP4_API_PORT=$FP4_API_PORT"
    echo "   NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
    
    # Start the service
    exec python3 main.py
}

# Main execution
main() {
    echo "Starting FLUX API container..."
    
    # Parse CLI arguments
    parse_args "$@"

    # Apply port and GPU selections (CLI overrides env, then defaults)
    if [[ -n "$PORT_ARG" ]]; then
        export FP4_API_PORT="$PORT_ARG"
    else
        export FP4_API_PORT="${FP4_API_PORT:-${PORT_DEFAULT}}"
    fi

    if [[ -n "$GPUS_ARG" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPUS_ARG"
        export NVIDIA_VISIBLE_DEVICES="$GPUS_ARG"
    else
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS_DEFAULT}}"
        export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-${GPUS_DEFAULT}}"
    fi

    # Setup directories
    setup_directories
    
    # Check NVIDIA runtime
    if ! check_nvidia_runtime; then
        echo "NVIDIA runtime check failed, but continuing..."
    fi
    
    # Check port
    if ! check_port ${FP4_API_PORT:-9000}; then
        echo "Port check failed, cannot start service"
        exit 1
    fi
    
    # Start the API
    start_api
}

# Handle signals for graceful shutdown
trap 'echo "Received shutdown signal, stopping service..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"
