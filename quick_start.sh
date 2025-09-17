#!/bin/bash
set -e

# Kontext API Quick Start Script
# This script provides quick commands for common Docker operations

echo "🚀 Kontext API Quick Start"
echo "=========================="

# Configuration
IMAGE_NAME="eigenai/kontext-api-20250918:kontext-api-20250918-v1"
CONTAINER_NAME="kontext-api"
PORT=9200

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     - Build Docker image locally"
    echo "  run       - Run container (basic)"
    echo "  run-lora  - Run container with LoRA fusion"
    echo "  stop      - Stop running container"
    echo "  logs      - Show container logs"
    echo "  health    - Check health status"
    echo "  test      - Run test suite"
    echo "  clean     - Clean up containers and images"
    echo "  help      - Show this help"
    echo ""
}

build_image() {
    echo -e "${YELLOW}🔨 Building Docker image...${NC}"
    docker build -t kontext-api-20250918 .
    docker tag kontext-api-20250918 $IMAGE_NAME
    echo -e "${GREEN}✅ Image built and tagged as $IMAGE_NAME${NC}"
}

run_basic() {
    echo -e "${YELLOW}🚀 Starting Kontext API (basic)...${NC}"
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:9200 \
        $IMAGE_NAME
    
    echo -e "${GREEN}✅ Container started${NC}"
    echo "🌐 API URL: http://localhost:$PORT"
    echo "🏥 Health check: http://localhost:$PORT/health"
    echo "📚 API docs: http://localhost:$PORT/docs"
}

run_with_lora() {
    echo -e "${YELLOW}🚀 Starting Kontext API with LoRA fusion...${NC}"
    
    read -p "Enter LoRA name/path: " LORA_NAME
    read -p "Enter LoRA weight (default 1.0): " LORA_WEIGHT
    LORA_WEIGHT=${LORA_WEIGHT:-1.0}
    
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:9200 \
        -e LORA_NAME="$LORA_NAME" \
        -e LORA_WEIGHT=$LORA_WEIGHT \
        -e FUSION_MODE=true \
        $IMAGE_NAME
    
    echo -e "${GREEN}✅ Container started with LoRA fusion${NC}"
    echo "🌐 API URL: http://localhost:$PORT"
    echo "🏥 Health check: http://localhost:$PORT/health"
}

stop_container() {
    echo -e "${YELLOW}⏹️  Stopping container...${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || echo "Container not running"
    docker rm $CONTAINER_NAME 2>/dev/null || echo "Container not found"
    echo -e "${GREEN}✅ Container stopped and removed${NC}"
}

show_logs() {
    echo -e "${YELLOW}📋 Container logs:${NC}"
    docker logs -f $CONTAINER_NAME
}

check_health() {
    echo -e "${YELLOW}🏥 Checking health status...${NC}"
    if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is healthy${NC}"
        curl -s http://localhost:$PORT/health | jq . 2>/dev/null || curl -s http://localhost:$PORT/health
    else
        echo -e "${RED}❌ Service is not responding${NC}"
    fi
}

run_tests() {
    echo -e "${YELLOW}🧪 Running test suite...${NC}"
    if [ -f "./tests/docker/run_tests.sh" ]; then
        ./tests/docker/run_tests.sh
    else
        echo -e "${RED}❌ Test script not found${NC}"
    fi
}

cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    
    # Stop and remove containers
    docker ps -a --filter "name=kontext" --format "{{.Names}}" | xargs -r docker rm -f
    
    # Remove images
    docker images --filter "reference=kontext-api*" --format "{{.Repository}}:{{.Tag}}" | xargs -r docker rmi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Main script logic
case "${1:-help}" in
    "build")
        build_image
        ;;
    "run")
        stop_container
        run_basic
        ;;
    "run-lora")
        stop_container
        run_with_lora
        ;;
    "stop")
        stop_container
        ;;
    "logs")
        show_logs
        ;;
    "health")
        check_health
        ;;
    "test")
        run_tests
        ;;
    "clean")
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac
