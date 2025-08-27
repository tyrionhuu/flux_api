# FLUX API Management Scripts

This directory contains utility scripts for managing FLUX API components.

## Available Scripts

### `manage_logs.sh` - Log Management

Manages log files and provides utilities for log monitoring and cleanup.

#### Usage

```bash
# Show log status
./scripts/manage_logs.sh status

# Clean old/large log files
./scripts/manage_logs.sh clean

# Clear all log files (interactive)
./scripts/manage_logs.sh clear

# Follow logs in real-time
./scripts/manage_logs.sh follow

# Show help
./scripts/manage_logs.sh help
```

#### Examples

```bash
# Check current log sizes
./scripts/manage_logs.sh status

# Clean logs larger than 100MB
./scripts/manage_logs.sh clean

# Follow BF16 API logs
./scripts/manage_logs.sh follow
# Then select the log file to follow
```

### `manage_services.sh` - Service Management

Manages systemd services for FLUX API components.

#### Usage

```bash
# List available and installed services
./scripts/manage_services.sh list

# Install a service (requires root)
sudo ./scripts/manage_services.sh install flux-cleanup

# Start a service
./scripts/manage_services.sh start flux-cleanup

# Stop a service
./scripts/manage_services.sh stop flux-cleanup

# Show service status
./scripts/manage_services.sh status flux-cleanup

# Follow service logs
./scripts/manage_services.sh logs flux-cleanup

# Show help
./scripts/manage_services.sh help
```

#### Examples

```bash
# Install the cleanup service
sudo ./scripts/manage_services.sh install flux-cleanup

# Check service status
./scripts/manage_services.sh status flux-cleanup

# Start the service
./scripts/manage_services.sh start flux-cleanup

# Follow logs in real-time
./scripts/manage_services.sh logs flux-cleanup
```

## Script Features

### Log Management Script

- **Status Check**: Shows log file sizes and line counts
- **Smart Cleanup**: Moves large logs to compressed backups
- **Interactive Clear**: Safely clears all logs with confirmation
- **Real-time Following**: Follow specific or all log files
- **Color-coded Output**: Easy-to-read status messages

### Service Management Script

- **Service Installation**: Easy systemd service installation
- **Service Control**: Start, stop, restart services
- **Status Monitoring**: Check service health and status
- **Log Following**: Real-time service log monitoring
- **Root Privilege Handling**: Clear error messages for permission issues

## Prerequisites

- **Bash**: Scripts require bash shell
- **systemctl**: For service management (systemd-based systems)
- **Root Access**: Some commands require sudo privileges
- **Python3**: For running the cleanup service

## Directory Structure

```
scripts/
├── manage_logs.sh          # Log management script
├── manage_services.sh      # Service management script
└── README.md              # This file
```

## Common Use Cases

### Daily Operations

```bash
# Check system health
./scripts/manage_logs.sh status
./scripts/manage_services.sh status

# Monitor logs
./scripts/manage_logs.sh follow
```

### Maintenance

```bash
# Clean up old logs
./scripts/manage_logs.sh clean

# Restart services if needed
./scripts/manage_services.sh restart flux-cleanup
```

### Troubleshooting

```bash
# Check service status
./scripts/manage_services.sh status flux-cleanup

# Follow service logs for errors
./scripts/manage_services.sh logs flux-cleanup

# Check log file sizes
./scripts/manage_logs.sh status
```

## Error Handling

Both scripts include comprehensive error handling:

- **Permission Checks**: Clear messages for permission issues
- **File Validation**: Checks for required files and directories
- **Service Validation**: Verifies services exist before operations
- **Graceful Failures**: Continues operation when possible

## Integration

These scripts work with the main FLUX API components:

- **Main API**: `main_fp4.py`
- **Cleanup Service**: `utils/cleanup_service.py`
- **Systemd Services**: `services/flux-cleanup.service`
- **Log Files**: `logs/` directory

## Support

For issues with these scripts:

1. Check script permissions: `ls -la scripts/`
2. Verify bash availability: `which bash`
3. Check systemd availability: `which systemctl`
4. Review script help: `./scripts/manage_logs.sh help`
