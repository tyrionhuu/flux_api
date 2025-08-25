# FLUX API Services

This directory contains systemd service files for running FLUX API components as system services.

## Available Services

### flux-cleanup.service

The directory cleanup service that automatically maintains size limits on your FLUX API directories.

## Installation

### 1. Copy the service file

```bash
sudo cp flux-cleanup.service /etc/systemd/system/
```

### 2. Update the WorkingDirectory path

Edit the service file to match your actual FLUX API installation path:

```bash
sudo nano /etc/systemd/system/flux-cleanup.service
```

Change this line to your actual path:
```
WorkingDirectory=/home/tianyu/flux_api
```

### 3. Update the ExecStart path

Also update the ExecStart path if needed:
```
ExecStart=/usr/bin/python3 /home/tianyu/flux_api/utils/cleanup_service.py
```

### 4. Reload systemd and enable the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable flux-cleanup.service
sudo systemctl start flux-cleanup.service
```

## Usage

### Check service status
```bash
sudo systemctl status flux-cleanup.service
```

### Start the service
```bash
sudo systemctl start flux-cleanup.service
```

### Stop the service
```bash
sudo systemctl stop flux-cleanup.service
```

### Restart the service
```bash
sudo systemctl restart flux-cleanup.service
```

### View service logs
```bash
sudo journalctl -u flux-cleanup.service -f
```

## Configuration

The service will use the settings from `config/cleanup_settings.py` in your FLUX API directory.

## Troubleshooting

### Service won't start
1. Check the service status: `sudo systemctl status flux-cleanup.service`
2. Check the logs: `sudo journalctl -u flux-cleanup.service`
3. Verify the paths in the service file are correct
4. Ensure the user has proper permissions

### Permission denied errors
1. Check file permissions in your FLUX API directory
2. Ensure the service user can access the required directories
3. Verify the ReadWritePaths in the service file

### Python import errors
1. Check that PYTHONPATH is set correctly
2. Verify all required Python packages are installed
3. Test the Python script manually: `python3 utils/cleanup_service.py`
