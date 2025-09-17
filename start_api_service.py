#!/usr/bin/env python3
"""
FLUX API Service Starter
This script starts the FLUX API service directly using the flux_env virtual environment.
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def check_port_available(port: int = 9001) -> bool:
    """Check if a port is available by attempting to bind to it"""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.bind(("localhost", port))
            s.close()
            return True
    except OSError:
        return False


def wait_for_port_free(port: int = 9001, max_wait: int = 30) -> bool:
    """Wait for a port to become free"""
    print(f"⏳ Waiting for port {port} to become free...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_port_available(port):
            print(f"   ✅ Port {port} is available")
            return True
        time.sleep(1)
        print(f"   ⏳ Still waiting... ({int(time.time() - start_time)}s)")

    print(f"   ❌ Port {port} did not become free within {max_wait} seconds")
    return False


def start_service(api_port: int = 9200):
    """Start the FLUX API service"""
    print("\n🚀 Starting FLUX API Service...")
    print("=" * 50)

    # Use provided port
    target_port = api_port
    print(f"🌐 API Port: {target_port}")

    # Check port availability
    print(f"🔍 Checking port {target_port} availability...")
    if not check_port_available(target_port):
        print(f"   ❌ Port {target_port} is not available")
        return False
    else:
        print(f"   ✅ Port {target_port} is available")



    # Resolve Python executable: prefer flux_env if present, else current python
    flux_env_python = Path("flux_env/bin/python")
    python_exec = str(flux_env_python) if flux_env_python.exists() else sys.executable

    # Start the service
    try:
        print("Starting API server...")
        # Manual GPU selection: respect existing CUDA_VISIBLE_DEVICES
        env = os.environ.copy()

        # Debug: Check if HUGGINGFACE_HUB_TOKEN is available
        if env.get("HUGGINGFACE_HUB_TOKEN"):
            print(f"✅ HUGGINGFACE_HUB_TOKEN found in environment")
        else:
            print("⚠️  HUGGINGFACE_HUB_TOKEN not found in environment")

        # Port will be passed as command line argument instead of environment variable

        if env.get("CUDA_VISIBLE_DEVICES"):
            print(
                f"Using CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} for FP4 service"
            )
        else:
            print("CUDA_VISIBLE_DEVICES not set; default visible GPU will be used")

        # Build command with arguments
        cmd = [python_exec, "main.py", "--port", str(target_port)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        print(f"Service started with PID: {process.pid}")
        print(f"API URL: http://localhost:{target_port}")
        print(f"Health check: http://localhost:{target_port}/health")
        print(f"API docs: http://localhost:{target_port}/docs")
        print("\n📋 Service logs:")
        print("-" * 50)

        # Stream the output
        if process.stdout:
            for line in process.stdout:
                print(line.rstrip())

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping service...")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("Service stopped")
    except Exception as e:
        print(f"Failed to start service: {e}")
        return False

    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="FLUX API Service Starter")

    parser.add_argument(
        "--port", type=int, default=9200, help="API port number (default: 9200)"
    )

    args = parser.parse_args()

    print("FLUX API Service Starter")
    print("=" * 30)
    print(f"API Port: {args.port}")

    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found in current directory!")
        print("   Please run this script from the flux_api directory.")
        sys.exit(1)

    # Start the service
    start_service(api_port=args.port)


if __name__ == "__main__":
    main()
