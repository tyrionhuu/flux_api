#!/bin/bash
set -e

# Kontext API Docker Testing Framework
# This script runs comprehensive tests on the Docker image

echo "🧪 Kontext API Docker Testing Framework"
echo "======================================="

# Configuration
IMAGE_NAME="eigenai/kontext-api-20250918:kontext-api-20250918-v1"
TEST_CONTAINER_PREFIX="kontext-test"
TEST_PORT=9201
HEALTH_CHECK_TIMEOUT=120
STARTUP_TIMEOUT=180

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# Helper functions
log_test() {
    echo -e "${YELLOW}🧪 Running test: $1${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

log_success() {
    echo -e "${GREEN}✅ Test passed: $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_failure() {
    echo -e "${RED}❌ Test failed: $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

cleanup_containers() {
    echo "🧹 Cleaning up test containers..."
    docker ps -a --filter "name=${TEST_CONTAINER_PREFIX}" --format "{{.Names}}" | xargs -r docker rm -f
}

wait_for_health() {
    local container_name=$1
    local port=$2
    local timeout=${3:-$HEALTH_CHECK_TIMEOUT}
    
    echo "⏳ Waiting for health check (timeout: ${timeout}s)..."
    local start_time=$(date +%s)
    
    while [ $(($(date +%s) - start_time)) -lt $timeout ]; do
        if curl -f http://localhost:$port/health > /dev/null 2>&1; then
            echo "✅ Health check passed"
            return 0
        fi
        sleep 5
        echo "   Still waiting... ($(($(date +%s) - start_time))s elapsed)"
    done
    
    echo "❌ Health check timeout"
    return 1
}

# Cleanup function
trap cleanup_containers EXIT

echo "📋 Test Configuration:"
echo "   Image: $IMAGE_NAME"
echo "   Test Port: $TEST_PORT"
echo "   Health Check Timeout: ${HEALTH_CHECK_TIMEOUT}s"
echo "   Startup Timeout: ${STARTUP_TIMEOUT}s"
echo ""

# Test 1: Basic Container Startup
log_test "Basic Container Startup"
cleanup_containers

CONTAINER_NAME="${TEST_CONTAINER_PREFIX}-basic"
docker run -d --name $CONTAINER_NAME -p $TEST_PORT:9200 $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "✅ Container started successfully"
    
    if wait_for_health $CONTAINER_NAME $TEST_PORT $STARTUP_TIMEOUT; then
        log_success "Basic Container Startup"
    else
        log_failure "Basic Container Startup - Health check failed"
    fi
else
    log_failure "Basic Container Startup - Container failed to start"
fi

# Test 2: Health Check Endpoints
log_test "Health Check Endpoints"

# Test /health endpoint
if curl -f http://localhost:$TEST_PORT/health > /dev/null 2>&1; then
    echo "✅ /health endpoint working"
    
    # Test /ready endpoint
    if curl -f http://localhost:$TEST_PORT/ready > /dev/null 2>&1; then
        echo "✅ /ready endpoint working"
        
        # Test /live endpoint
        if curl -f http://localhost:$TEST_PORT/live > /dev/null 2>&1; then
            echo "✅ /live endpoint working"
            log_success "Health Check Endpoints"
        else
            log_failure "Health Check Endpoints - /live failed"
        fi
    else
        log_failure "Health Check Endpoints - /ready failed"
    fi
else
    log_failure "Health Check Endpoints - /health failed"
fi

# Test 3: API Functionality
log_test "API Functionality"

# Test root endpoint
if curl -f http://localhost:$TEST_PORT/ > /dev/null 2>&1; then
    echo "✅ Root endpoint working"
    
    # Test model status endpoint
    if curl -f http://localhost:$TEST_PORT/model-status > /dev/null 2>&1; then
        echo "✅ Model status endpoint working"
        log_success "API Functionality"
    else
        log_failure "API Functionality - Model status failed"
    fi
else
    log_failure "API Functionality - Root endpoint failed"
fi

# Test 4: LoRA Fusion Mode (without LoRA)
log_test "LoRA Fusion Mode (No LoRA)"
cleanup_containers

CONTAINER_NAME="${TEST_CONTAINER_PREFIX}-fusion-no-lora"
docker run -d --name $CONTAINER_NAME -p $TEST_PORT:9200 \
    -e FUSION_MODE=true \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    if wait_for_health $CONTAINER_NAME $TEST_PORT $STARTUP_TIMEOUT; then
        # Check if fusion mode is enabled in health response
        HEALTH_RESPONSE=$(curl -s http://localhost:$TEST_PORT/health)
        if echo "$HEALTH_RESPONSE" | grep -q '"fusion_mode":true'; then
            log_success "LoRA Fusion Mode (No LoRA)"
        else
            log_failure "LoRA Fusion Mode (No LoRA) - Fusion mode not enabled"
        fi
    else
        log_failure "LoRA Fusion Mode (No LoRA) - Health check failed"
    fi
else
    log_failure "LoRA Fusion Mode (No LoRA) - Container failed to start"
fi

# Test 5: Environment Variable Configuration
log_test "Environment Variable Configuration"

# Test custom port
cleanup_containers
CONTAINER_NAME="${TEST_CONTAINER_PREFIX}-env-port"
docker run -d --name $CONTAINER_NAME -p $TEST_PORT:8000 \
    -e PORT=8000 \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    if wait_for_health $CONTAINER_NAME $TEST_PORT $STARTUP_TIMEOUT; then
        log_success "Environment Variable Configuration"
    else
        log_failure "Environment Variable Configuration - Health check failed"
    fi
else
    log_failure "Environment Variable Configuration - Container failed to start"
fi

# Test 6: Resource Usage
log_test "Resource Usage"

CONTAINER_NAME="${TEST_CONTAINER_PREFIX}-resource"
docker run -d --name $CONTAINER_NAME -p $TEST_PORT:9200 \
    --memory=8g --cpus=4 \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    if wait_for_health $CONTAINER_NAME $TEST_PORT $STARTUP_TIMEOUT; then
        # Check memory usage
        MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemUsage}}" $CONTAINER_NAME)
        echo "📊 Memory usage: $MEMORY_USAGE"
        log_success "Resource Usage"
    else
        log_failure "Resource Usage - Health check failed"
    fi
else
    log_failure "Resource Usage - Container failed to start"
fi

# Final cleanup
cleanup_containers

# Test Summary
echo ""
echo "📊 Test Summary"
echo "==============="
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
