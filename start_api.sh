#!/bin/bash

# FLUX API Startup Script
# This script starts the FastAPI server with proper environment setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}    FLUX API Startup Script     ${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_status "Python version: $PYTHON_VERSION"
}

# Function to check if virtual environment exists
check_virtual_env() {
    if [ ! -d "flux_env" ]; then
        print_error "Virtual environment 'flux_env' not found!"
        echo ""
        print_status "Creating virtual environment..."
        $PYTHON_CMD -m venv flux_env
        print_status "Virtual environment created successfully!"
    fi
}

# Function to check if requirements are installed
check_requirements() {
    if [ ! -f "flux_env/bin/pip" ]; then
        print_error "Virtual environment is corrupted or incomplete"
        exit 1
    fi
    
    # Check if key packages are installed
    if ! flux_env/bin/pip show fastapi >/dev/null 2>&1; then
        print_warning "Dependencies not installed. Installing now..."
        print_status "Installing requirements from requirements.txt..."
        flux_env/bin/pip install -r requirements.txt
        print_status "Dependencies installed successfully!"
    else
        print_status "Dependencies already installed"
    fi
}

# Function to check CUDA availability
check_cuda() {
    if command_exists nvidia-smi; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
            print_status "  GPU: $line"
        done
    else
        print_warning "NVIDIA GPU not detected. CUDA features may not work."
    fi
}

# Function to start the server
start_server() {
    print_status "Starting FLUX API server..."
    echo ""
    print_status "Server will be available at:"
    echo "  🌐 API: http://127.0.0.1:8000"
    echo "  📚 Docs: http://127.0.0.1:8000/docs"
    echo "  🔍 Status: http://127.0.0.1:8000/"
    echo ""
    print_status "Press Ctrl+C to stop the server"
    echo ""
    
    # Start the server
    flux_env/bin/uvicorn main:app --reload --host 127.0.0.1 --port 8000
}

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down FLUX API server..."
    exit 0
}

# Main execution
main() {
    print_header
    
    # Set up signal handlers
    trap cleanup SIGINT SIGTERM
    
    # Check Python
    print_status "Checking Python installation..."
    check_python_version
    
    # Check virtual environment
    print_status "Checking virtual environment..."
    check_virtual_env
    
    # Check requirements
    print_status "Checking dependencies..."
    check_requirements
    
    # Check CUDA
    print_status "Checking CUDA availability..."
    check_cuda
    
    # Start server
    start_server
}

# Run main function
main "$@"
