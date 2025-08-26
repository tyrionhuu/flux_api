#!/bin/bash

# Multi-GPU FLUX API Monitoring Script
# Monitors health, performance, and resource usage of all GPU services

set -e

# Configuration
BASE_PORT=8000
NUM_GPUS=8
REFRESH_INTERVAL=5  # seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to check service health
check_service_health() {
    local port=$1
    local gpu_id=$2
    
    # Try to reach the service
    if curl -s -f -m 2 "http://localhost:$port/" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ UP${NC}"
    else
        echo -e "${RED}❌ DOWN${NC}"
    fi
}

# Function to get service queue status
get_queue_status() {
    local port=$1
    
    # Try to get queue status from the service
    local response=$(curl -s -m 2 "http://localhost:$port/queue-stats" 2>/dev/null || echo "{}")
    
    if [ -n "$response" ] && [ "$response" != "{}" ]; then
        echo "$response"
    else
        echo '{"queue_size": "N/A", "active_requests": "N/A"}'
    fi
}

# Function to get GPU utilization
get_gpu_stats() {
    local gpu_id=$1
    
    # Get GPU stats using nvidia-smi
    local stats=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
                  --format=csv,noheader,nounits -i $gpu_id 2>/dev/null || echo "$gpu_id,N/A,N/A,N/A,N/A")
    
    echo "$stats"
}

# Function to display header
display_header() {
    clear
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    FLUX Multi-GPU Service Monitor                       ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')    Refresh: ${REFRESH_INTERVAL}s    Press Ctrl+C to exit"
    echo ""
}

# Function to display service status
display_services() {
    echo -e "${YELLOW}Service Status:${NC}"
    echo "─────────────────────────────────────────────────────────────────────────"
    printf "%-6s %-8s %-10s %-15s %-20s\n" "GPU" "Port" "Status" "Queue Status" "Active Requests"
    echo "─────────────────────────────────────────────────────────────────────────"
    
    for ((i=0; i<$NUM_GPUS; i++)); do
        local port=$((BASE_PORT + i))
        local status=$(check_service_health $port $i)
        local queue_info=$(get_queue_status $port)
        
        # Parse queue info (handle both valid JSON and N/A)
        local queue_size="N/A"
        local processing="N/A"
        
        if [[ "$queue_info" != *"N/A"* ]]; then
            queue_size=$(echo "$queue_info" | grep -o '"queue_size":[0-9]*' | cut -d':' -f2 || echo "N/A")
            processing=$(echo "$queue_info" | grep -o '"active_requests":[0-9]*' | cut -d':' -f2 || echo "N/A")
        fi
        
        printf "%-6s %-8s %-10s %-15s %-20s\n" \
            "GPU $i" \
            "$port" \
            "$status" \
            "$queue_size" \
            "$processing"
    done
    echo ""
}

# Function to display GPU stats
display_gpu_stats() {
    echo -e "${YELLOW}GPU Resource Usage:${NC}"
    echo "─────────────────────────────────────────────────────────────────────────"
    printf "%-6s %-15s %-20s %-12s\n" "GPU" "Utilization" "Memory Usage" "Temperature"
    echo "─────────────────────────────────────────────────────────────────────────"
    
    for ((i=0; i<$NUM_GPUS; i++)); do
        local stats=$(get_gpu_stats $i)
        IFS=',' read -r idx util mem_used mem_total temp <<< "$stats"
        
        # Format memory usage
        if [[ "$mem_used" != "N/A" && "$mem_total" != "N/A" ]]; then
            local mem_percent=$(( mem_used * 100 / mem_total ))
            local mem_display="${mem_used}MB/${mem_total}MB (${mem_percent}%)"
        else
            local mem_display="N/A"
        fi
        
        # Format utilization with color
        if [[ "$util" != "N/A" ]]; then
            if [ "$util" -gt 80 ]; then
                util_display="${RED}${util}%${NC}"
            elif [ "$util" -gt 50 ]; then
                util_display="${YELLOW}${util}%${NC}"
            else
                util_display="${GREEN}${util}%${NC}"
            fi
        else
            util_display="N/A"
        fi
        
        # Format temperature with color
        if [[ "$temp" != "N/A" ]]; then
            if [ "$temp" -gt 80 ]; then
                temp_display="${RED}${temp}°C${NC}"
            elif [ "$temp" -gt 70 ]; then
                temp_display="${YELLOW}${temp}°C${NC}"
            else
                temp_display="${GREEN}${temp}°C${NC}"
            fi
        else
            temp_display="N/A"
        fi
        
        printf "%-6s %-25b %-20s %-20b\n" \
            "GPU $i" \
            "$util_display" \
            "$mem_display" \
            "$temp_display"
    done
    echo ""
}

# Function to display nginx status
display_nginx_status() {
    echo -e "${YELLOW}Load Balancer Status:${NC}"
    echo "─────────────────────────────────────────────────────────────────────────"
    
    # Check if nginx is running
    if pgrep -x nginx > /dev/null; then
        echo -e "Nginx: ${GREEN}✅ Running${NC}"
        
        # Try to get nginx status if stub_status is enabled
        local nginx_status=$(curl -s "http://localhost/nginx_status" 2>/dev/null || echo "")
        if [ -n "$nginx_status" ]; then
            echo "$nginx_status" | grep -E "Active connections:|server accepts handled requests|Reading:|Writing:|Waiting:"
        fi
    else
        echo -e "Nginx: ${RED}❌ Not Running${NC}"
    fi
    echo ""
}

# Function to display recent errors
display_recent_errors() {
    echo -e "${YELLOW}Recent Errors (last 5 per service):${NC}"
    echo "─────────────────────────────────────────────────────────────────────────"
    
    local found_errors=false
    
    for ((i=0; i<$NUM_GPUS; i++)); do
        local log_file="logs/multi_gpu/flux_gpu${i}_port$((BASE_PORT + i)).log"
        
        if [ -f "$log_file" ]; then
            local errors=$(grep -i "error\|exception\|failed" "$log_file" | tail -5 2>/dev/null || echo "")
            if [ -n "$errors" ]; then
                echo -e "${CYAN}GPU $i:${NC}"
                echo "$errors" | sed 's/^/  /'
                echo ""
                found_errors=true
            fi
        fi
    done
    
    if [ "$found_errors" = false ]; then
        echo "No recent errors found."
    fi
    echo ""
}

# Function to run monitoring loop
monitor_loop() {
    while true; do
        display_header
        display_services
        display_gpu_stats
        display_nginx_status
        
        # Optional: display errors (can be toggled with a flag)
        if [[ "${SHOW_ERRORS:-}" == "true" ]]; then
            display_recent_errors
        fi
        
        # Show commands
        echo -e "${BLUE}────────────────────────────────────────────────────────────────────────${NC}"
        echo "Commands: [q] Quit  [e] Toggle errors  [r] Refresh now"
        echo ""
        
        # Wait for input or timeout
        read -t $REFRESH_INTERVAL -n 1 key || true
        
        case $key in
            q|Q)
                echo "Exiting monitor..."
                exit 0
                ;;
            e|E)
                if [[ "${SHOW_ERRORS:-}" == "true" ]]; then
                    SHOW_ERRORS=false
                else
                    SHOW_ERRORS=true
                fi
                ;;
            r|R)
                # Immediate refresh
                continue
                ;;
        esac
    done
}

# Function to run single status check
single_check() {
    display_services
    display_gpu_stats
    display_nginx_status
}

# Parse command line arguments
case "${1:-}" in
    --once|-o)
        # Run single check and exit
        single_check
        ;;
    --errors|-e)
        # Start with errors shown
        SHOW_ERRORS=true
        monitor_loop
        ;;
    --help|-h)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --once, -o     Run single check and exit"
        echo "  --errors, -e   Show errors on startup"
        echo "  --help, -h     Show this help message"
        echo ""
        echo "Interactive commands:"
        echo "  q - Quit"
        echo "  e - Toggle error display"
        echo "  r - Refresh immediately"
        ;;
    *)
        # Default: run monitoring loop
        monitor_loop
        ;;
esac