#!/bin/bash

# FLUX API Startup Script with Port Cleanup
# This script ensures clean startup by handling port conflicts

set -e

echo "🚀 FLUX API Startup Script"
echo "=========================="

# Parse args (-g supports single GPU id or comma-separated list, e.g., "1,2,3")
GPU_ID=""
PORT=8000

usage() {
    echo "Usage: $0 [-g <gpu_ids>]"
    echo "  -g <gpu_ids>   GPU index or comma-separated list (e.g., 1 or 1,2,3). If omitted, all GPUs remain visible."
}

while getopts ":g:p:h" opt; do
  case "$opt" in
    g)
      GPU_ID="$OPTARG"
      ;;
    p)
      PORT="$OPTARG"
      ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
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
if [ ! -f "main_fp4.py" ]; then
    echo "❌ main_fp4.py not found in current directory!"
    echo "   Please run this script from the flux_api directory."
    exit 1
fi



# Check if start_flux_service.py exists
if [ ! -f "start_flux_service.py" ]; then
    echo "❌ start_flux_service.py not found!"
    echo "   Please ensure the service starter script exists."
    exit 1
fi

echo "✅ Environment check passed"

# Clean up selected port
if ! cleanup_port "$PORT"; then
    echo "⚠️  Port cleanup incomplete, but continuing..."
fi

# Wait for port to be available
if ! wait_for_port "$PORT" 30; then
    echo "❌ Port 8000 is not available, cannot start service"
    exit 1
fi

echo "🚀 Starting FLUX API Service..."

# Assign GPU visibility only if -g provided; otherwise leave all GPUs visible
if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "🔧 Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} for FP4 service"
else
  echo "🔧 No -g provided; using all visible GPUs"
fi

# Export selected port for the Python service
export FP4_API_PORT="$PORT"
echo "🔧 Using FP4_API_PORT=${FP4_API_PORT}"

python start_flux_service.py
