#!/bin/bash

# FLUX API Backend-Only Runner
# Simplified script that starts only the backend API without frontend

set -e

# Default values
DEFAULT_PORT=9001
DEFAULT_TOKEN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            FP4_API_PORT="$2"
            shift 2
            ;;
        -t|--token)
            HUGGINGFACE_HUB_TOKEN="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_FLAG="--build"
            shift
            ;;
        -d|--detach)
            DETACH_FLAG="-d"
            shift
            ;;
        -h|--help)
            echo "FLUX API Backend-Only Runner"
            echo "============================="
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "This script starts the FLUX API in backend-only mode (no frontend UI)."
            echo ""
            echo "Options:"
            echo "  -p, --port PORT        Set the API port (default: $DEFAULT_PORT)"
            echo "  -t, --token TOKEN      Set the Hugging Face token (required)"
            echo "  -b, --build           Force rebuild the Docker image"
            echo "  -d, --detach          Run in background (default)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -t hf_your_token_here                    # Use default port 9001"
            echo "  $0 -p 9002 -t hf_your_token_here           # Use port 9002"
            echo "  $0 -b -t hf_your_token_here                # Rebuild and start"
            echo "  $0 -p 9003 -t hf_your_token_here --build   # Custom port with rebuild"
            echo ""
            echo "Environment Variables:"
            echo "  FP4_API_PORT          API port (overrides -p option)"
            echo "  HUGGINGFACE_HUB_TOKEN Hugging Face token (overrides -t option)"
            echo ""
            echo "Commands:"
            echo "  $0 logs               # Show container logs"
            echo "  $0 stop               # Stop the container"
            echo "  $0 restart            # Restart the container"
            echo "  $0 status             # Show container status"
            exit 0
            ;;
        logs)
            docker compose logs -f flux-api
            exit 0
            ;;
        stop)
            docker compose down
            exit 0
            ;;
        restart)
            docker compose restart flux-api
            exit 0
            ;;
        status)
            docker compose ps
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set defaults if not provided
export FP4_API_PORT=${FP4_API_PORT:-$DEFAULT_PORT}
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-$DEFAULT_TOKEN}

# Force backend-only mode
export ENABLE_FRONTEND=false
echo "🔧 Backend-only mode enabled - frontend will be disabled"

# Check if token is provided
if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "❌ Error: Hugging Face token is required"
    echo "Use -t or --token to provide your token"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Check if port is available
if lsof -i:$FP4_API_PORT >/dev/null 2>&1; then
    echo "⚠️  Warning: Port $FP4_API_PORT is already in use"
    echo "   The container will still start, but you may need to stop the conflicting service"
fi

echo "🚀 Starting FLUX API (Backend-Only) with:"
echo "   Port: $FP4_API_PORT"
echo "   Token: ${HUGGINGFACE_HUB_TOKEN:0:10}..."
echo "   Build: ${BUILD_FLAG:-"no"}"
echo "   Frontend: disabled"
echo ""

# Run docker compose
docker compose up ${DETACH_FLAG:-"-d"} ${BUILD_FLAG}

# Show status
echo ""
echo "📊 Container Status:"
docker compose ps

echo ""
echo "🔗 Access the API at: http://localhost:$FP4_API_PORT"
echo "📱 UI: Disabled (backend-only mode)"
echo "❤️  Health check: http://localhost:$FP4_API_PORT/health"
echo "📚 API docs: http://localhost:$FP4_API_PORT/docs"
echo ""
echo "📝 Useful commands:"
echo "   $0 logs     # View logs"
echo "   $0 stop     # Stop container"
echo "   $0 restart  # Restart container"
echo "   $0 status   # Show status"
