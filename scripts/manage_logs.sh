#!/bin/bash

# FLUX API Log Management Script
# This script helps manage log files and directory cleanup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_ROOT/logs"

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

# Function to show log status
show_log_status() {
    print_header "Log File Status"
    
    if [ ! -d "$LOGS_DIR" ]; then
        print_error "Logs directory not found: $LOGS_DIR"
        return 1
    fi
    
    cd "$LOGS_DIR"
    
    echo "Log files in $LOGS_DIR:"
    echo ""
    
    total_size=0
    for log_file in *.log; do
        if [ -f "$log_file" ]; then
            size=$(du -h "$log_file" | cut -f1)
            lines=$(wc -l < "$log_file" 2>/dev/null || echo "0")
            echo "  $log_file: $size ($lines lines)"
            total_size=$((total_size + $(stat -c%s "$log_file" 2>/dev/null || echo 0)))
        fi
    done
    
    total_size_mb=$((total_size / 1024 / 1024))
    echo ""
    echo "Total log size: ${total_size_mb}MB"
}

# Function to clean old logs
clean_old_logs() {
    print_header "Cleaning Old Logs"
    
    if [ ! -d "$LOGS_DIR" ]; then
        print_error "Logs directory not found: $LOGS_DIR"
        return 1
    fi
    
    cd "$LOGS_DIR"
    
    # Create backup directory
    backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Move old logs to backup
    moved_count=0
    for log_file in *.log; do
        if [ -f "$log_file" ]; then
            file_size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
            file_size_mb=$((file_size / 1024 / 1024))
            
            if [ $file_size_mb -gt 100 ]; then
                print_status "Moving large log file: $log_file (${file_size_mb}MB)"
                mv "$log_file" "$backup_dir/"
                moved_count=$((moved_count + 1))
            fi
        fi
    done
    
    if [ $moved_count -eq 0 ]; then
        print_status "No large log files found to clean"
    else
        print_status "Moved $moved_count log files to $backup_dir"
        
        # Compress backup
        tar -czf "${backup_dir}.tar.gz" "$backup_dir"
        rm -rf "$backup_dir"
        print_status "Created compressed backup: ${backup_dir}.tar.gz"
    fi
}

# Function to clear all logs
clear_all_logs() {
    print_header "Clearing All Logs"
    
    if [ ! -d "$LOGS_DIR" ]; then
        print_error "Logs directory not found: $LOGS_DIR"
        return 1
    fi
    
    cd "$LOGS_DIR"
    
    print_warning "This will clear all log files. Are you sure? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        for log_file in *.log; do
            if [ -f "$log_file" ]; then
                print_status "Clearing $log_file"
                > "$log_file"
            fi
        done
        print_status "All log files cleared"
    else
        print_status "Operation cancelled"
    fi
}

# Function to follow logs
follow_logs() {
    print_header "Following Logs"
    
    if [ ! -d "$LOGS_DIR" ]; then
        print_error "Logs directory not found: $LOGS_DIR"
        return 1
    fi
    
    cd "$LOGS_DIR"
    
    echo "Available log files:"
    echo ""
    
    log_files=()
    i=1
    for log_file in *.log; do
        if [ -f "$log_file" ]; then
            echo "  $i) $log_file"
            log_files+=("$log_file")
            i=$((i + 1))
        fi
    done
    
    echo ""
    echo "Enter the number of the log file to follow (or 'all' for all logs):"
    read -r choice
    
    if [ "$choice" = "all" ]; then
        print_status "Following all log files (Ctrl+C to stop)"
        tail -f *.log
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#log_files[@]} ]; then
        selected_file="${log_files[$((choice - 1))]}"
        print_status "Following $selected_file (Ctrl+C to stop)"
        tail -f "$selected_file"
    else
        print_error "Invalid choice"
        return 1
    fi
}

# Function to show help
show_help() {
    echo "FLUX API Log Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  status     Show log file status and sizes"
    echo "  clean      Clean old/large log files"
    echo "  clear      Clear all log files (interactive)"
    echo "  follow     Follow log files in real-time"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status          # Show current log status"
    echo "  $0 clean           # Clean old logs"
    echo "  $0 follow          # Follow logs interactively"
}

# Main script logic
main() {
    case "${1:-help}" in
        status)
            show_log_status
            ;;
        clean)
            clean_old_logs
            ;;
        clear)
            clear_all_logs
            ;;
        follow)
            follow_logs
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
