#!/bin/bash

# Multi-GPU FLUX API Deployment Script
# Starts 8 FLUX API instances on 8 different GPUs with load balancing

set -e

echo "🚀 Multi-GPU FLUX API Deployment"
echo "=========================================="

# Configuration
BASE_PORT=23333
NUM_GPUS=8
LOG_DIR="logs/multi_gpu"
PID_FILE="flux_multi_gpu.pids"
VENV_PATH="${VENV_PATH:-venv}"  # Allow override via environment

# Parse command line arguments
MODEL_TYPE="fp4_sekai"  # Default to fp4 for lower memory usage
NGINX_CONFIG_OVERRIDE=""  # Will use this if provided, otherwise auto-select

usage() {
    echo "Usage: $0 [-m <model_type>] [-c <nginx_config>]"
    echo "  -m <model_type>    Model type: fp4 or bf16 (default: fp4)"
    echo "  -c <nginx_config>  Path to nginx config (default: auto-select based on GPU count)"
    echo "  -h                 Show this help message"
}

while getopts ":m:c:h" opt; do
  case "$opt" in
    m)
      MODEL_TYPE="$OPTARG"
      if [[ "$MODEL_TYPE" != "fp4" && "$MODEL_TYPE" != "bf16" && "$MODEL_TYPE" != "fp4_sekai" ]]; then
        echo "❌ Invalid model type: $MODEL_TYPE (must be fp4 or bf16 or fp4_sekai)"
        exit 1
      fi
      ;;
    c)
      NGINX_CONFIG_OVERRIDE="$OPTARG"
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
  esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Function to check GPU availability
check_gpus() {
    echo "🔍 Checking GPU availability..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    # Check if CUDA_VISIBLE_DEVICES is set
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "   Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        # Parse the GPU IDs from CUDA_VISIBLE_DEVICES
        IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        NUM_GPUS=${#GPU_IDS[@]}
        echo "   Will use $NUM_GPUS GPUs: ${GPU_IDS[*]}"
        echo "   [DEBUG] GPU_IDS array: ${GPU_IDS[@]}"
        echo "   [DEBUG] NUM_GPUS: $NUM_GPUS"
    else
        local gpu_count=$(nvidia-smi -L | wc -l)
        echo "   Found $gpu_count GPUs total"
        
        if [ $gpu_count -lt $NUM_GPUS ]; then
            echo "⚠️  Warning: Only $gpu_count GPUs available, adjusting NUM_GPUS"
            NUM_GPUS=$gpu_count
        fi
        
        # Create GPU_IDS array for all GPUs
        GPU_IDS=()
        for ((i=0; i<$NUM_GPUS; i++)); do
            GPU_IDS+=($i)
        done
    fi
    
    # Show GPU info
    for gpu_id in "${GPU_IDS[@]}"; do
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits -i $gpu_id | \
        while IFS=, read -r index name memory; do
            echo "   GPU $index: $name (${memory}MB)"
        done
    done
    
    echo "✅ GPU check passed"
}

# Function to stop all services
stop_all_services() {
    echo "🛑 Stopping all FLUX services..."
    
    if [ -f "$PID_FILE" ]; then
        while read pid; do
            if kill -0 $pid 2>/dev/null; then
                echo "   Stopping PID $pid..."
                kill -TERM $pid 2>/dev/null || true
            fi
        done < "$PID_FILE"
        
        # Wait for graceful shutdown
        sleep 2
        
        # Force kill if needed
        while read pid; do
            if kill -0 $pid 2>/dev/null; then
                echo "   Force killing PID $pid..."
                kill -KILL $pid 2>/dev/null || true
            fi
        done < "$PID_FILE"
        
        rm -f "$PID_FILE"
    fi
    
    # Also check for any remaining processes on our ports
    for ((i=0; i<$NUM_GPUS; i++)); do
        local port=$((BASE_PORT + i))
        local pids=$(lsof -ti:$port 2>/dev/null || echo "")
        if [ -n "$pids" ]; then
            echo "   Cleaning up port $port..."
            for pid in $pids; do
                kill -KILL $pid 2>/dev/null || true
            done
        fi
    done
    
    echo "✅ All services stopped"
}

# Function to start a single service
start_service() {
    local idx=$1
    local gpu_id=${GPU_IDS[$idx]}
    local port=$((BASE_PORT + idx))
    local log_file="$LOG_DIR/flux_gpu${gpu_id}_port${port}.log"
    local debug_file="$LOG_DIR/debug_gpu${gpu_id}_port${port}.log"
    
    echo "🚀 Starting service on GPU $gpu_id (port $port)..."
    echo "[DEBUG] Index: $idx, GPU ID: $gpu_id, Port: $port" | tee "$debug_file"
    
    # Prepare Python command with virtual environment
    local python_cmd="python"
    if [ -d "$VENV_PATH" ]; then
        python_cmd="$VENV_PATH/bin/python"
    fi
    
    # Calculate threads per GPU instance to prevent CPU oversubscription
    local total_cpus=$(nproc)
    local threads_per_gpu=$((total_cpus / NUM_GPUS / 2))
    # Ensure at least 2 threads per instance
    threads_per_gpu=$(( threads_per_gpu < 2 ? 2 : threads_per_gpu ))
    
    echo "   Configuring with $threads_per_gpu CPU threads (total CPUs: $total_cpus, GPUs: $NUM_GPUS)"
    
    # Debug: Test GPU visibility before launching
    echo "[DEBUG] Testing GPU visibility with CUDA_VISIBLE_DEVICES=$gpu_id..." | tee -a "$debug_file"
    CUDA_VISIBLE_DEVICES=$gpu_id $python_cmd -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPU(s)'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else None}')" >> "$debug_file" 2>&1
    
    # Debug: Show environment variables
    echo "[DEBUG] Environment variables:" >> "$debug_file"
    echo "  CUDA_VISIBLE_DEVICES=$gpu_id" >> "$debug_file"
    echo "  FP4_API_PORT=$port" >> "$debug_file"
    echo "  NUM_GPU_INSTANCES=$NUM_GPUS" >> "$debug_file"
    echo "  Python command: $python_cmd" >> "$debug_file"
    
    # Start the service in background with proper environment and thread limits
    if [ "$MODEL_TYPE" = "fp4" ]; then
        echo "[DEBUG] Launching fp4 model..." >> "$debug_file"
        echo "[DEBUG] Full command: CUDA_VISIBLE_DEVICES=$gpu_id FP4_API_PORT=$port $python_cmd main_fp4.py" >> "$debug_file"
        
        # Export variables properly and launch
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            export FP4_API_PORT=$port
            export NUM_GPU_INSTANCES=$NUM_GPUS
            export OMP_NUM_THREADS=$threads_per_gpu
            export MKL_NUM_THREADS=$threads_per_gpu
            export NUMEXPR_NUM_THREADS=$threads_per_gpu
            export OPENBLAS_NUM_THREADS=$threads_per_gpu
            export VECLIB_MAXIMUM_THREADS=$threads_per_gpu
            export TORCH_NUM_THREADS=$threads_per_gpu
            
            echo "[DEBUG] Environment in subshell:" >> "$debug_file"
            echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$debug_file"
            echo "  FP4_API_PORT=$FP4_API_PORT" >> "$debug_file"
            
            nohup $python_cmd main_fp4.py >> "$log_file" 2>&1 &
            echo $! > "${debug_file}.pid"
        )
    elif [ "$MODEL_TYPE" = "fp4_sekai" ]; then
        echo "[DEBUG] Launching fp4_sekai model..." >> "$debug_file"
        echo "[DEBUG] Full command: CUDA_VISIBLE_DEVICES=$gpu_id FP4_API_PORT=$port $python_cmd main_fp4_sekai.py" >> "$debug_file"
        
        # Export variables properly and launch
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            export FP4_API_PORT=$port
            export NUM_GPU_INSTANCES=$NUM_GPUS
            export FLUX_RETURN_BASE64=true
            export PRELOAD_UPSCALER=true
            export OMP_NUM_THREADS=$threads_per_gpu
            export MKL_NUM_THREADS=$threads_per_gpu
            export NUMEXPR_NUM_THREADS=$threads_per_gpu
            export OPENBLAS_NUM_THREADS=$threads_per_gpu
            export VECLIB_MAXIMUM_THREADS=$threads_per_gpu
            export TORCH_NUM_THREADS=$threads_per_gpu
            
            echo "[DEBUG] Environment in subshell:" >> "$debug_file"
            echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$debug_file"
            echo "  FP4_API_PORT=$FP4_API_PORT" >> "$debug_file"
            echo "  FLUX_RETURN_BASE64=$FLUX_RETURN_BASE64" >> "$debug_file"
            
            nohup $python_cmd main_fp4_sekai.py >> "$log_file" 2>&1 &
            echo $! > "${debug_file}.pid"
        )
    else
        echo "[DEBUG] Launching bf16 model..." >> "$debug_file"
        echo "[DEBUG] Full command: CUDA_VISIBLE_DEVICES=$gpu_id BF16_API_PORT=$port $python_cmd main_bf16.py" >> "$debug_file"
        
        # Export variables properly and launch
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            export BF16_API_PORT=$port
            export NUM_GPU_INSTANCES=$NUM_GPUS
            export OMP_NUM_THREADS=$threads_per_gpu
            export MKL_NUM_THREADS=$threads_per_gpu
            export NUMEXPR_NUM_THREADS=$threads_per_gpu
            export OPENBLAS_NUM_THREADS=$threads_per_gpu
            export VECLIB_MAXIMUM_THREADS=$threads_per_gpu
            export TORCH_NUM_THREADS=$threads_per_gpu
            
            echo "[DEBUG] Environment in subshell:" >> "$debug_file"
            echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$debug_file"
            echo "  BF16_API_PORT=$BF16_API_PORT" >> "$debug_file"
            
            nohup $python_cmd main_bf16.py >> "$log_file" 2>&1 &
            echo $! > "${debug_file}.pid"
        )
    fi
    
    # Read the PID from the debug file
    if [ -f "${debug_file}.pid" ]; then
        local pid=$(cat "${debug_file}.pid")
        rm -f "${debug_file}.pid"
    else
        echo "[ERROR] Failed to get PID!" | tee -a "$debug_file"
        return 1
    fi
    
    echo $pid >> "$PID_FILE"
    
    echo "   Started PID $pid, log: $log_file"
    echo "   Thread limits: OMP=$threads_per_gpu, MKL=$threads_per_gpu"
    
    # Check if process is still running after 2 seconds
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        echo "   ✅ Process $pid is running" | tee -a "$debug_file"
    else
        echo "   ❌ Process $pid died immediately!" | tee -a "$debug_file"
        echo "   Checking error in log file:" | tee -a "$debug_file"
        if [ -f "$log_file" ]; then
            echo "====== Last 20 lines of $log_file ======" | tee -a "$debug_file"
            tail -20 "$log_file" | tee -a "$debug_file"
            echo "========================================" | tee -a "$debug_file"
        else
            echo "   Log file not created!" | tee -a "$debug_file"
        fi
        return 1
    fi
    
    # Brief pause to avoid race conditions
    sleep 1
}

# Function to wait for services to be ready
wait_for_services() {
    echo "⏳ Waiting for all services to be ready..."
    
    local max_wait=60
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        local ready_count=0
        
        for ((i=0; i<$NUM_GPUS; i++)); do
            local port=$((BASE_PORT + i))
            if curl -s -f "http://localhost:$port/" > /dev/null 2>&1; then
                ((ready_count++))
            fi
        done
        
        if [ $ready_count -eq $NUM_GPUS ]; then
            echo "✅ All $NUM_GPUS services are ready!"
            return 0
        fi
        
        echo "   $ready_count/$NUM_GPUS services ready..."
        sleep 2
        ((wait_time+=2))
    done
    
    echo "⚠️  Warning: Only $ready_count/$NUM_GPUS services are ready after ${max_wait}s"
    return 1
}

# Function to start nginx
start_nginx() {
    echo "🔄 Starting Nginx load balancer..."
    
    # Determine which nginx config to use
    if [ -n "$NGINX_CONFIG_OVERRIDE" ]; then
        # Use the override if provided
        NGINX_CONFIG="$NGINX_CONFIG_OVERRIDE"
        echo "   Using specified nginx config: $NGINX_CONFIG"
    else
        # Auto-select based on GPU count
        NGINX_CONFIG="configs/nginx-${NUM_GPUS}.conf"
        echo "   Auto-selected nginx config for $NUM_GPUS GPUs: $NGINX_CONFIG"
    fi
    
    # Check if config file exists
    if [ ! -f "$NGINX_CONFIG" ]; then
        echo "❌ Nginx config file not found: $NGINX_CONFIG"
        if [ -z "$NGINX_CONFIG_OVERRIDE" ]; then
            echo "   Please ensure configs/nginx-${NUM_GPUS}.conf exists"
            echo "   Available configs: $(ls configs/nginx-*.conf 2>/dev/null | xargs basename -a | tr '\n' ' ')"
        fi
        echo "   Services are running on ports ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
        return 1
    fi
    
    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        echo "⚠️  Nginx not installed. Install with: sudo apt-get install nginx"
        echo "   Services are running on ports ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
        return 1
    fi
    
    # Stop the systemd nginx service if it's running
    echo "   Checking for systemd nginx service..."
    if systemctl is-active nginx > /dev/null 2>&1; then
        echo "   Stopping systemd nginx service..."
        sudo systemctl stop nginx
        sleep 2
    fi
    
    # Also stop any other nginx instances
    if pgrep -x nginx > /dev/null; then
        echo "   Stopping remaining nginx instances..."
        sudo nginx -s quit 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if pgrep -x nginx > /dev/null; then
            sudo pkill -9 nginx 2>/dev/null || true
            sleep 1
        fi
    fi
    
    # Test nginx config
    if ! sudo nginx -t -c "$PWD/$NGINX_CONFIG" 2>/dev/null; then
        echo "⚠️  Nginx config test failed. Please check $NGINX_CONFIG"
        echo "   Services are running on ports ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
        return 1
    fi
    
    # Start nginx with our config
    echo "   Starting nginx with config: $NGINX_CONFIG"
    if sudo nginx -c "$PWD/$NGINX_CONFIG"; then
        # Verify nginx started successfully and is listening on port 8080
        sleep 2
        if sudo lsof -i :8080 > /dev/null 2>&1; then
            echo "✅ Nginx started successfully on port 8080"
            return 0
        else
            echo "⚠️  Nginx started but not listening on port 8080"
            echo "   Checking nginx status..."
            sudo nginx -t -c "$PWD/$NGINX_CONFIG"
            return 1
        fi
    else
        echo "❌ Failed to start nginx. Error output above."
        echo "   Try manually: sudo nginx -c $PWD/$NGINX_CONFIG"
        return 1
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "📊 Service Status:"
    echo "=================="
    
    for ((i=0; i<$NUM_GPUS; i++)); do
        local port=$((BASE_PORT + i))
        local status="❌ Down"
        
        if curl -s -f "http://localhost:$port/" > /dev/null 2>&1; then
            status="✅ Running"
        fi
        
        echo "   GPU $i (Port $port): $status"
    done
    
    echo ""
    echo "📝 Logs available in: $LOG_DIR/"
    echo "🌐 Load balancer URL: http://localhost/"
    echo ""
    echo "To stop all services: $0 stop"
}

# Main execution
main() {
    # Handle stop command
    if [ "${1:-}" = "stop" ]; then
        stop_all_services
        # Stop nginx (both custom and systemd)
        sudo nginx -s stop 2>/dev/null || true
        sudo systemctl stop nginx 2>/dev/null || true
        sudo pkill -9 nginx 2>/dev/null || true
        exit 0
    fi
    
    # Check environment
    check_gpus
    
    # Stop any existing services
    stop_all_services
    
    # Start all services
    echo ""
    echo "🚀 Starting $NUM_GPUS FLUX services..."
    rm -f "$PID_FILE"
    
    for ((i=0; i<$NUM_GPUS; i++)); do
        start_service $i
    done
    
    # Wait for services to be ready
    wait_for_services
    
    # Start nginx
    start_nginx
    
    # Show final status
    show_status
}

# Run main function
main "$@"
