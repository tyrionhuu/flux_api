#!/bin/bash

# FLUX API Service Management Script
# This script helps manage systemd services for FLUX API components

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$PROJECT_ROOT/services"

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
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This command requires root privileges. Use 'sudo $0 $1'"
        exit 1
    fi
}

# Function to install service
install_service() {
    local service_name="$1"
    local service_file="$SERVICES_DIR/$service_name.service"
    
    if [ ! -f "$service_file" ]; then
        print_error "Service file not found: $service_file"
        return 1
    fi
    
    print_status "Installing $service_name service..."
    
    # Copy service file
    cp "$service_file" "/etc/systemd/system/"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable "$service_name.service"
    
    print_status "$service_name service installed and enabled"
}

# Function to uninstall service
uninstall_service() {
    local service_name="$1"
    
    print_status "Uninstalling $service_name service..."
    
    # Stop and disable service
    systemctl stop "$service_name.service" 2>/dev/null || true
    systemctl disable "$service_name.service" 2>/dev/null || true
    
    # Remove service file
    rm -f "/etc/systemd/system/$service_name.service"
    
    # Reload systemd
    systemctl daemon-reload
    
    print_status "$service_name service uninstalled"
}

# Function to show service status
show_service_status() {
    local service_name="$1"
    
    if [ -z "$service_name" ]; then
        print_header "All FLUX API Services Status"
        
        # Check for installed services
        local services=("flux-cleanup")
        
        for service in "${services[@]}"; do
            if systemctl list-unit-files | grep -q "$service.service"; then
                echo ""
                echo "Service: $service"
                systemctl status "$service.service" --no-pager -l || true
            else
                echo ""
                echo "Service: $service (not installed)"
            fi
        done
    else
        if systemctl list-unit-files | grep -q "$service_name.service"; then
            print_header "$service_name Service Status"
            systemctl status "$service_name.service" --no-pager -l
        else
            print_error "Service $service_name not found"
            return 1
        fi
    fi
}

# Function to start service
start_service() {
    local service_name="$1"
    
    if [ -z "$service_name" ]; then
        print_error "Please specify a service name"
        return 1
    fi
    
    if ! systemctl list-unit-files | grep -q "$service_name.service"; then
        print_error "Service $service_name not installed"
        return 1
    fi
    
    print_status "Starting $service_name service..."
    systemctl start "$service_name.service"
    print_status "$service_name service started"
}

# Function to stop service
stop_service() {
    local service_name="$1"
    
    if [ -z "$service_name" ]; then
        print_error "Please specify a service name"
        return 1
    fi
    
    if ! systemctl list-unit-files | grep -q "$service_name.service"; then
        print_error "Service $service_name not installed"
        return 1
    fi
    
    print_status "Stopping $service_name service..."
    systemctl stop "$service_name.service"
    print_status "$service_name service stopped"
}

# Function to restart service
restart_service() {
    local service_name="$1"
    
    if [ -z "$service_name" ]; then
        print_error "Please specify a service name"
        return 1
    fi
    
    if ! systemctl list-unit-files | grep -q "$service_name.service"; then
        print_error "Service $service_name not installed"
        return 1
    fi
    
    print_status "Restarting $service_name service..."
    systemctl restart "$service_name.service"
    print_status "$service_name service restarted"
}

# Function to show service logs
show_service_logs() {
    local service_name="$1"
    
    if [ -z "$service_name" ]; then
        print_error "Please specify a service name"
        return 1
    fi
    
    if ! systemctl list-unit-files | grep -q "$service_name.service"; then
        print_error "Service $service_name not installed"
        return 1
    fi
    
    print_status "Showing logs for $service_name service (Ctrl+C to stop)..."
    journalctl -u "$service_name.service" -f
}

# Function to list available services
list_services() {
    print_header "Available Services"
    
    echo "Services in $SERVICES_DIR:"
    echo ""
    
    if [ -d "$SERVICES_DIR" ]; then
        for service_file in "$SERVICES_DIR"/*.service; do
            if [ -f "$service_file" ]; then
                service_name=$(basename "$service_file" .service)
                echo "  $service_name"
            fi
        done
    else
        echo "  No services directory found"
    fi
    
    echo ""
    echo "Installed systemd services:"
    echo ""
    
    local services=("flux-cleanup")
    for service in "${services[@]}"; do
        if systemctl list-unit-files | grep -q "$service.service"; then
            status=$(systemctl is-active "$service.service" 2>/dev/null || echo "unknown")
            echo "  $service: $status"
        else
            echo "  $service: not installed"
        fi
    done
}

# Function to show help
show_help() {
    echo "FLUX API Service Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [SERVICE]"
    echo ""
    echo "Commands:"
    echo "  list                    List available and installed services"
    echo "  install SERVICE         Install a service (requires root)"
    echo "  uninstall SERVICE       Uninstall a service (requires root)"
    echo "  start SERVICE           Start a service"
    echo "  stop SERVICE            Stop a service"
    echo "  restart SERVICE         Restart a service"
    echo "  status [SERVICE]        Show service status"
    echo "  logs SERVICE            Show service logs"
    echo "  help                    Show this help message"
    echo ""
    echo "Available Services:"
    echo "  flux-cleanup            Directory cleanup service"
    echo ""
    echo "Examples:"
    echo "  $0 list                                    # List all services"
    echo "  $0 install flux-cleanup                    # Install cleanup service"
    echo "  $0 start flux-cleanup                      # Start cleanup service"
    echo "  $0 status flux-cleanup                     # Show cleanup service status"
    echo "  $0 logs flux-cleanup                       # Follow cleanup service logs"
    echo ""
    echo "Note: Install/uninstall commands require root privileges"
}

# Main script logic
main() {
    case "${1:-help}" in
        list)
            list_services
            ;;
        install)
            check_root
            if [ -z "$2" ]; then
                print_error "Please specify a service to install"
                exit 1
            fi
            install_service "$2"
            ;;
        uninstall)
            check_root
            if [ -z "$2" ]; then
                print_error "Please specify a service to uninstall"
                exit 1
            fi
            uninstall_service "$2"
            ;;
        start)
            if [ -z "$2" ]; then
                print_error "Please specify a service to start"
                exit 1
            fi
            start_service "$2"
            ;;
        stop)
            if [ -z "$2" ]; then
                print_error "Please specify a service to stop"
                exit 1
            fi
            stop_service "$2"
            ;;
        restart)
            if [ -z "$2" ]; then
                print_error "Please specify a service to restart"
                exit 1
            fi
            restart_service "$2"
            ;;
        status)
            show_service_status "$2"
            ;;
        logs)
            if [ -z "$2" ]; then
                print_error "Please specify a service to show logs for"
                exit 1
            fi
            show_service_logs "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
