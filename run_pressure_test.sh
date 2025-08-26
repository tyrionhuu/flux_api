#!/bin/bash

# Pressure Test Script for FLUX API
# Sends requests at a controlled rate (RPS - Requests Per Second)

# Default RPS if not provided
RPS=${1:-2}

# Total number of requests to send
TOTAL_REQUESTS=128

# Calculate sleep interval between requests (in seconds)
# Using bc for floating point arithmetic
INTERVAL=$(echo "scale=4; 1/$RPS" | bc)

echo "==========================================
🔥 FLUX API Pressure Test
==========================================
📊 Configuration:
   - Target: http://localhost:8080/generate
   - Requests per second: $RPS
   - Interval between requests: ${INTERVAL}s
   - Total requests: $TOTAL_REQUESTS
==========================================
"

# Counter for request tracking
REQUEST_COUNT=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

# Create a temporary directory for response files
TEMP_DIR=$(mktemp -d)
echo "📁 Temporary response directory: $TEMP_DIR"
echo ""

# Start time for overall test
START_TIME=$(date +%s.%N)

echo "🚀 Starting pressure test..."
echo "----------------------------------------"

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
        
        # Log result
        if [[ "$http_code" == "200" ]]; then
            echo "✅ Request #$request_num: Success (HTTP $http_code)"
        else
            echo "❌ Request #$request_num: Failed (HTTP $http_code)"
        fi
        
        # Save status for final statistics
        echo "$http_code" > "$TEMP_DIR/status_${request_num}.txt"
    } &
}

# Main loop to send requests
for ((i=1; i<=TOTAL_REQUESTS; i++)); do
    send_request $i
    REQUEST_COUNT=$((REQUEST_COUNT + 1))
    
    # Show progress every 10 requests
    if [ $((i % 10)) -eq 0 ]; then
        echo "📊 Progress: $i/$TOTAL_REQUESTS requests sent..."
    fi
    
    # Sleep for the calculated interval before sending next request
    # Only sleep if not the last request
    if [ $i -lt $TOTAL_REQUESTS ]; then
        sleep $INTERVAL
    fi
done

echo "----------------------------------------"
echo "⏳ Waiting for all requests to complete..."

# Wait for all background jobs to finish
wait

# Calculate end time and duration
END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "==========================================
📈 Test Results
==========================================
"

# Analyze results
for ((i=1; i<=TOTAL_REQUESTS; i++)); do
    if [ -f "$TEMP_DIR/status_${i}.txt" ]; then
        status=$(cat "$TEMP_DIR/status_${i}.txt")
        if [[ "$status" == "200" ]]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
        fi
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi
done

# Calculate statistics
SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_REQUESTS" | bc)
ACTUAL_RPS=$(echo "scale=2; $TOTAL_REQUESTS / $DURATION" | bc)

echo "📊 Summary Statistics:"
echo "   - Total Requests: $TOTAL_REQUESTS"
echo "   - Successful: $SUCCESS_COUNT"
echo "   - Failed: $FAILURE_COUNT"
echo "   - Success Rate: ${SUCCESS_RATE}%"
echo "   - Test Duration: ${DURATION}s"
echo "   - Target RPS: $RPS"
echo "   - Actual RPS: $ACTUAL_RPS"
echo ""

# Show HTTP status code distribution
echo "📋 HTTP Status Code Distribution:"
cat $TEMP_DIR/status_*.txt 2>/dev/null | sort | uniq -c | while read count code; do
    echo "   - HTTP $code: $count requests"
done

echo ""
echo "==========================================
"

# Cleanup
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "✅ Pressure test completed!"
echo ""

# Provide recommendations based on results
if [ $FAILURE_COUNT -gt 0 ]; then
    FAILURE_RATE=$(echo "scale=2; $FAILURE_COUNT * 100 / $TOTAL_REQUESTS" | bc)
    echo "⚠️  Warning: ${FAILURE_RATE}% of requests failed!"
    echo "   Recommendations:"
    echo "   - Consider reducing RPS to improve success rate"
    echo "   - Check server logs for error details"
    echo "   - Verify server resources (CPU, memory, GPU)"
    if [ $RPS -gt 4 ]; then
        echo "   - Try RPS=2 or RPS=1 for baseline testing"
    fi
else
    echo "🎉 All requests completed successfully!"
    echo "   You may try increasing RPS to find the maximum throughput."
fi