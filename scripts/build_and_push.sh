#!/bin/bash
set -e

# Kontext API Docker Build and Push Script
# This script builds and pushes the Docker image to EigenAI DockerHub

echo "🐳 Kontext API Docker Build and Push Script"
echo "============================================="

# Configuration
REPO_NAME="eigenai/kontext-api-20250918"
VERSION=$(date +%Y%m%d-%H%M%S)
TAG="${REPO_NAME}:kontext-api-20250918-v1-${VERSION}"
LATEST_TAG="${REPO_NAME}:kontext-api-20250918-v1"

echo "📋 Configuration:"
echo "   Repository: $REPO_NAME"
echo "   Version: $VERSION"
echo "   Tag: $TAG"
echo "   Latest Tag: $LATEST_TAG"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

echo "✅ Docker is running"

# Build the image
echo "🔨 Building Docker image..."
docker build -t $TAG .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Tag as latest version
echo "🏷️  Tagging as latest version..."
docker tag $TAG $LATEST_TAG

# Check if DockerHub credentials are available
if [ -z "$DOCKERHUB_TOKEN" ]; then
    echo "⚠️  DOCKERHUB_TOKEN not set. Please set it to push to DockerHub:"
    echo "   export DOCKERHUB_TOKEN=your_token_here"
    echo ""
    echo "📦 Image built locally with tags:"
    echo "   $TAG"
    echo "   $LATEST_TAG"
    echo ""
    echo "🚀 To push manually:"
    echo "   docker login -u eigenai -p \$DOCKERHUB_TOKEN"
    echo "   docker push $TAG"
    echo "   docker push $LATEST_TAG"
    exit 0
fi

# Login to DockerHub
echo "🔐 Logging into DockerHub..."
docker login -u eigenai -p $DOCKERHUB_TOKEN

if [ $? -eq 0 ]; then
    echo "✅ Successfully logged into DockerHub"
else
    echo "❌ Failed to login to DockerHub"
    exit 1
fi

# Push the image
echo "📤 Pushing Docker image..."
docker push $TAG

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed $TAG"
else
    echo "❌ Failed to push $TAG"
    exit 1
fi

# Push the latest tag
echo "📤 Pushing latest tag..."
docker push $LATEST_TAG

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed $LATEST_TAG"
else
    echo "❌ Failed to push $LATEST_TAG"
    exit 1
fi

# Cleanup
echo "🧹 Cleaning up..."
docker logout

echo ""
echo "🎉 Successfully built and pushed Kontext API Docker image!"
echo "📦 Image: $TAG"
echo "📦 Latest: $LATEST_TAG"
echo ""
echo "🚀 Deployment command:"
echo "   docker run -d --name kontext-api -p 9200:9200 $LATEST_TAG"
echo ""
echo "🔧 With LoRA fusion:"
echo "   docker run -d --name kontext-api-with-lora -p 9200:9200 \\"
echo "     -e LORA_NAME=\"my-lora.safetensors\" \\"
echo "     -e LORA_WEIGHT=1.2 \\"
echo "     -e FUSION_MODE=true \\"
echo "     $LATEST_TAG"
echo ""
echo "🏥 Health check:"
echo "   curl http://localhost:9200/health"
