# FLUX API Logs

This directory contains all log files for the FLUX API system.

## Log Files

### API Logs

- **`flux_api.log`** - Main FLUX API log (legacy)
- **`flux_api_fp4.log`** - FP4 FLUX API specific logs
- **`flux_api_fp4.log`** - FP4 FLUX API specific logs (created when service starts)

### Service Logs

- **`cleanup.log`** - Directory cleanup service logs

## Log Rotation

Logs are not automatically rotated by default. Consider implementing log rotation to prevent logs from growing too large:

### Using logrotate

Create `/etc/logrotate.d/flux-api`:

```
/home/tianyu/flux_api/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload flux-api-fp4
        systemctl reload flux-api-fp4
    endscript
}
```

### Manual Cleanup

You can also manually clean up old logs:

```bash
# Keep only last 7 days of logs
find logs/ -name "*.log" -mtime +7 -delete

# Compress old logs
find logs/ -name "*.log" -mtime +1 -exec gzip {} \;
```

## Log Levels

The system uses the following log levels:
- **DEBUG** - Detailed information for debugging
- **INFO** - General information about operations
- **WARNING** - Warning messages for potential issues
- **ERROR** - Error messages for failed operations
- **CRITICAL** - Critical errors that may cause service failure

## Configuration

Log levels can be configured in the main API files:
- `main_fp4.py` - FP4 API logging
- `main_fp4.py` - FP4 API logging
- `config/cleanup_settings.py` - Cleanup service logging

## Monitoring

### Real-time Log Monitoring

```bash
# Monitor FP4 API logs
tail -f logs/flux_api_fp4.log

# Monitor FP4 API logs
tail -f logs/flux_api_fp4.log

# Monitor cleanup logs
tail -f logs/cleanup.log

# Monitor all logs
tail -f logs/*.log
```

### Log Analysis

```bash
# Count errors in logs
grep -c "ERROR" logs/*.log

# Find recent errors
grep "ERROR" logs/*.log | tail -20

# Search for specific patterns
grep "upload" logs/*.log
```

## Troubleshooting

### Logs Not Being Written

1. Check file permissions on the logs directory
2. Verify the API services have write access
3. Check disk space availability
4. Ensure the logs directory exists

### Logs Too Large

1. Implement log rotation
2. Reduce log verbosity in configuration
3. Archive old logs to separate storage
4. Use log aggregation tools (ELK stack, etc.)

### Performance Issues

1. Reduce logging frequency
2. Use asynchronous logging
3. Implement log buffering
4. Consider using structured logging
