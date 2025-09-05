#!/bin/bash

# Pressure Test Script for FLUX API
# Sends requests at a controlled rate (RPS - Requests Per Second)

# Default RPS if not provided
RPS=${1:-2}

# Total number of requests to send
TOTAL_REQUESTS=9999999

# Calculate sleep interval between requests (in seconds)
# Using bc for floating point arithmetic
INTERVAL=$(echo "scale=4; 1/$RPS" | bc)

# Counter for request tracking
REQUEST_COUNT=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

# Create a temporary directory for response files
TEMP_DIR=$(mktemp -d)

# Start time for overall test
START_TIME=$(date +%s.%N)

echo "🔥 FLUX Pressure Test @ ${RPS} RPS"

# Function to send a request and track response
send_request() {
    local request_num=$1
    local response_file="$TEMP_DIR/response_${request_num}.txt"
    
    # Send request in background and capture HTTP status code
    {
        http_code=$(curl -X POST http://localhost:8080/generate \
            -H "Content-Type: application/json" \
            -d "{
                \"prompt\": \"test image $request_num\",
                \"width\": 512,
                \"height\": 512
            }" \
            -o "$response_file" \
            -w "%{http_code}" \
            -s 2>&1)
        
        # Save status for real-time statistics
        echo "$http_code" > "$TEMP_DIR/status_${request_num}.txt"
        
        # Update counters in shared files for real-time display
        if [[ "$http_code" == "200" ]]; then
            echo "1" >> "$TEMP_DIR/success_count.txt"
        else
            echo "1" >> "$TEMP_DIR/failure_count.txt"
            # Save error details for display
            echo "$request_num:$http_code" >> "$TEMP_DIR/error_details.txt"
        fi
    } &
}

# Function to display real-time stats
display_stats() {
    local sent=$1
    local success_count=0
    local failure_count=0
    
    # Count completed requests
    if [ -f "$TEMP_DIR/success_count.txt" ]; then
        success_count=$(wc -l < "$TEMP_DIR/success_count.txt" 2>/dev/null || echo 0)
    fi
    if [ -f "$TEMP_DIR/failure_count.txt" ]; then
        failure_count=$(wc -l < "$TEMP_DIR/failure_count.txt" 2>/dev/null || echo 0)
    fi
    
    local completed=$((success_count + failure_count))
    local success_rate=0
    if [ $completed -gt 0 ]; then
        success_rate=$(echo "scale=1; $success_count * 100 / $completed" | bc)
    fi
    
    # Check for new errors and display them
    if [ -f "$TEMP_DIR/error_details.txt" ] && [ -f "$TEMP_DIR/last_error_count.txt" ]; then
        local last_count=$(cat "$TEMP_DIR/last_error_count.txt")
        local current_count=$(wc -l < "$TEMP_DIR/error_details.txt" 2>/dev/null || echo 0)
        if [ $current_count -gt $last_count ]; then
            # Get new errors since last check
            local new_errors=$(tail -n $((current_count - last_count)) "$TEMP_DIR/error_details.txt")
            # Display each new error on a new line in red
            while IFS=: read -r req_num http_code; do
                printf "\n\033[31m✗ Request #%s failed with HTTP %s\033[0m" "$req_num" "$http_code"
            done <<< "$new_errors"
            echo ""  # New line after errors
        fi
        echo "$current_count" > "$TEMP_DIR/last_error_count.txt"
    elif [ -f "$TEMP_DIR/error_details.txt" ]; then
        # Initialize error count tracking
        wc -l < "$TEMP_DIR/error_details.txt" > "$TEMP_DIR/last_error_count.txt" 2>/dev/null
    else
        echo "0" > "$TEMP_DIR/last_error_count.txt"
    fi
    
    # Display stats on current line
    printf "\rRPS: %s | Sent: %d/%d | Done: %d | OK: %d (%.1f%%) | Fail: %d" \
        "$RPS" "$sent" "$TOTAL_REQUESTS" "$completed" "$success_count" "$success_rate" "$failure_count"
}

# Main loop to send requests
for ((i=1; i<=TOTAL_REQUESTS; i++)); do
    send_request $i
    REQUEST_COUNT=$((REQUEST_COUNT + 1))
    
    # Display real-time stats
    display_stats $i
    
    # Sleep for the calculated interval before sending next request
    # Only sleep if not the last request
    if [ $i -lt $TOTAL_REQUESTS ]; then
        sleep $INTERVAL
    fi
done

# Continue showing live stats until all requests complete
while true; do
    success_count=0
    failure_count=0
    
    if [ -f "$TEMP_DIR/success_count.txt" ]; then
        success_count=$(wc -l < "$TEMP_DIR/success_count.txt" 2>/dev/null || echo 0)
    fi
    if [ -f "$TEMP_DIR/failure_count.txt" ]; then
        failure_count=$(wc -l < "$TEMP_DIR/failure_count.txt" 2>/dev/null || echo 0)
    fi
    
    completed=$((success_count + failure_count))
    success_rate=0
    if [ $completed -gt 0 ]; then
        success_rate=$(echo "scale=1; $success_count * 100 / $completed" | bc)
    fi
    
    # Check for new errors and display them
    if [ -f "$TEMP_DIR/error_details.txt" ] && [ -f "$TEMP_DIR/last_error_count.txt" ]; then
        local last_count=$(cat "$TEMP_DIR/last_error_count.txt")
        local current_count=$(wc -l < "$TEMP_DIR/error_details.txt" 2>/dev/null || echo 0)
        if [ $current_count -gt $last_count ]; then
            # Get new errors since last check
            local new_errors=$(tail -n $((current_count - last_count)) "$TEMP_DIR/error_details.txt")
            # Display each new error on a new line in red
            while IFS=: read -r req_num http_code; do
                printf "\n\033[31m✗ Request #%s failed with HTTP %s\033[0m" "$req_num" "$http_code"
            done <<< "$new_errors"
            echo ""  # New line after errors
        fi
        echo "$current_count" > "$TEMP_DIR/last_error_count.txt"
    fi
    
    printf "\rRPS: %s | Sent: %d/%d | Done: %d | OK: %d (%.1f%%) | Fail: %d" \
        "$RPS" "$TOTAL_REQUESTS" "$TOTAL_REQUESTS" "$completed" "$success_count" "$success_rate" "$failure_count"
    
    # Break when all requests are complete
    if [ $completed -eq $TOTAL_REQUESTS ]; then
        break
    fi
    
    sleep 0.5
done

echo ""

# Wait for all background jobs to finish
wait

# Calculate end time and duration
END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)

# Final statistics
echo ""
SUCCESS_COUNT=$(wc -l < "$TEMP_DIR/success_count.txt" 2>/dev/null || echo 0)
FAILURE_COUNT=$(wc -l < "$TEMP_DIR/failure_count.txt" 2>/dev/null || echo 0)
SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_REQUESTS" | bc)
ACTUAL_RPS=$(echo "scale=2; $TOTAL_REQUESTS / $DURATION" | bc)

echo "✅ Test Complete: $SUCCESS_COUNT/$TOTAL_REQUESTS successful (${SUCCESS_RATE}%) | Duration: ${DURATION}s | Actual RPS: $ACTUAL_RPS"

# Cleanup
rm -rf "$TEMP_DIR"
