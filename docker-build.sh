#!/bin/bash

# Docker build script for FLUX API
# This script builds the Docker image with proper configuration

set -e

echo "🔨 FLUX API Docker Build Script"
echo "==============================="

# Configuration
IMAGE_NAME="flux-img2img-api"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Function to check if Docker is running
check_docker() {
    echo "🔍 Checking Docker availability..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker daemon is not running"
        echo "   Please start Docker and try again"
        exit 1
    fi
    
    echo "✅ Docker is available and running"
}

# Function to check NVIDIA Docker runtime
check_nvidia_docker() {
    echo "🔍 Checking NVIDIA Docker runtime..."
    
    if docker info 2>/dev/null | grep -q nvidia; then
        echo "✅ NVIDIA Docker runtime is available"
    else
        echo "⚠️  NVIDIA Docker runtime not detected"
        echo "   GPU support may not work properly"
        echo "   Make sure nvidia-docker2 is installed"
    fi
}

# Function to build the image
build_image() {
    echo "🔨 Building Docker image: $FULL_IMAGE_NAME"
    echo "   This may take several minutes..."
    
    # Build with buildkit for better caching
    DOCKER_BUILDKIT=1 docker build \
        --tag "$FULL_IMAGE_NAME" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        .
    
    if [ $? -eq 0 ]; then
        echo "✅ Docker image built successfully: $FULL_IMAGE_NAME"
    else
        echo "❌ Docker image build failed"
        exit 1
    fi
}

# Function to show image info
show_image_info() {
    echo "📊 Image information:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo ""
    echo "📋 Next steps:"
    echo "   1. Run the container:"
    echo "      docker-compose up -d"
    echo ""
    echo "   2. Or run directly:"
    echo "      docker run --gpus all -p 9000:9000 $FULL_IMAGE_NAME"
    echo ""
    echo "   3. Check logs:"
    echo "      docker-compose logs -f"
    echo ""
    echo "   4. Access the API:"
    echo "      http://localhost:9000/health"
    echo "      http://localhost:9000/docs"
}

# Function to clean up old images (optional)
cleanup_old_images() {
    echo "🧹 Cleaning up old images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Optionally remove old versions of this image
    read -p "Remove old versions of $IMAGE_NAME? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker images "$IMAGE_NAME" --format "{{.ID}}" | tail -n +2 | xargs -r docker rmi
        echo "✅ Old images removed"
    fi
}

# Main execution
main() {
    echo "Starting Docker build process..."
    
    # Check prerequisites
    check_docker
    check_nvidia_docker
    
    # Build the image
    build_image
    
    # Show information
    show_image_info
    
    # Optional cleanup
    cleanup_old_images
    
    echo "🎉 Build process completed successfully!"
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            echo "🔄 Building without cache..."
            DOCKER_BUILDKIT=1 docker build --no-cache --tag "$FULL_IMAGE_NAME" .
            exit 0
            ;;
        --clean)
            cleanup_old_images
            exit 0
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-cache    Build without using cache"
            echo "  --clean       Clean up old images"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    shift
done

# Run main function
main "$@"
