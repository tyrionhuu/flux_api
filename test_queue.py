#!/usr/bin/env python3
"""
Test script to verify request queuing works correctly
Sends multiple concurrent requests to test the queue
"""

import asyncio
import aiohttp
import time
import sys
import argparse
from typing import List, Dict, Any


async def send_request(session: aiohttp.ClientSession, url: str, request_id: int) -> Dict[str, Any]:
    """Send a single generation request"""
    start_time = time.time()
    
    payload = {
        "prompt": f"Test prompt {request_id}: A beautiful sunset over mountains",
        "width": 512,
        "height": 512,
        "num_inference_steps": 5,  # Use fewer steps for testing
        "loras": [],  # No LoRA for testing
        "response_format": "json"  # Get JSON response instead of binary
    }
    
    try:
        print(f"[Request {request_id}] Sending request...")
        async with session.post(f"{url}/generate", json=payload) as response:
            elapsed = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                print(f"[Request {request_id}] ✅ Success after {elapsed:.1f}s")
                return {
                    "request_id": request_id,
                    "status": "success",
                    "elapsed": elapsed,
                    "data": data
                }
            elif response.status == 503:
                print(f"[Request {request_id}] ⚠️  Queue full (503) after {elapsed:.1f}s")
                return {
                    "request_id": request_id,
                    "status": "queue_full",
                    "elapsed": elapsed,
                    "error": await response.text()
                }
            elif response.status == 504:
                print(f"[Request {request_id}] ⏱️  Timeout (504) after {elapsed:.1f}s")
                return {
                    "request_id": request_id,
                    "status": "timeout",
                    "elapsed": elapsed,
                    "error": await response.text()
                }
            else:
                error_text = await response.text()
                print(f"[Request {request_id}] ❌ Error {response.status} after {elapsed:.1f}s: {error_text[:100]}")
                return {
                    "request_id": request_id,
                    "status": "error",
                    "status_code": response.status,
                    "elapsed": elapsed,
                    "error": error_text
                }
                
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Request {request_id}] 💥 Exception after {elapsed:.1f}s: {e}")
        return {
            "request_id": request_id,
            "status": "exception",
            "elapsed": elapsed,
            "error": str(e)
        }


async def check_queue_status(session: aiohttp.ClientSession, url: str):
    """Check the current queue status"""
    try:
        async with session.get(f"{url}/request-queue-stats") as response:
            if response.status == 200:
                data = await response.json()
                stats = data["queue_stats"]
                print(f"\n📊 Queue Status:")
                print(f"   Active: {stats['active_requests']}/{stats['max_concurrent']}")
                print(f"   Queued: {stats['queued_requests']}/{stats['max_queue_size']}")
                print(f"   Processed: {stats['total_processed']}")
                print(f"   Rejected: {stats['total_rejected']}")
                print(f"   Timeouts: {stats['total_timeout']}")
                print(f"   RPM: {stats['requests_per_minute']:.1f}\n")
                return data
    except Exception as e:
        print(f"Failed to get queue status: {e}")
        return None


async def run_test(url: str, num_requests: int, delay: float = 0.0):
    """Run the test with multiple concurrent requests"""
    print(f"\n🚀 Testing queue with {num_requests} concurrent requests to {url}")
    print(f"   Delay between requests: {delay}s\n")
    
    async with aiohttp.ClientSession() as session:
        # Check initial queue status
        await check_queue_status(session, url)
        
        # Create tasks for all requests
        tasks = []
        for i in range(num_requests):
            task = send_request(session, url, i + 1)
            tasks.append(task)
            if delay > 0 and i < num_requests - 1:
                await asyncio.sleep(delay)
        
        # Wait for all requests to complete
        print(f"\n⏳ Waiting for all {num_requests} requests to complete...\n")
        results = await asyncio.gather(*tasks)
        
        # Check final queue status
        await check_queue_status(session, url)
        
        # Analyze results
        print("\n📈 Results Summary:")
        success_count = sum(1 for r in results if r["status"] == "success")
        queue_full_count = sum(1 for r in results if r["status"] == "queue_full")
        timeout_count = sum(1 for r in results if r["status"] == "timeout")
        error_count = sum(1 for r in results if r["status"] in ["error", "exception"])
        
        print(f"   ✅ Success: {success_count}/{num_requests}")
        print(f"   ⚠️  Queue Full: {queue_full_count}/{num_requests}")
        print(f"   ⏱️  Timeouts: {timeout_count}/{num_requests}")
        print(f"   ❌ Errors: {error_count}/{num_requests}")
        
        if success_count > 0:
            success_times = [r["elapsed"] for r in results if r["status"] == "success"]
            print(f"\n⏱️  Timing Statistics (successful requests):")
            print(f"   Min: {min(success_times):.1f}s")
            print(f"   Max: {max(success_times):.1f}s")
            print(f"   Avg: {sum(success_times) / len(success_times):.1f}s")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test FLUX API request queuing")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="API URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between sending requests in seconds (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_test(args.url, args.requests, args.delay))
    except KeyboardInterrupt:
        print("\n\n⛔ Test interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()