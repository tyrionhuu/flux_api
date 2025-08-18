#!/usr/bin/env python3
"""
BF16 FLUX API Service Starter (Port 8001)
This script starts the BF16 FLUX API service directly using the flux_env virtual environment.
"""

import os
import sys
import subprocess
import time
import signal
import psutil
from pathlib import Path


def check_flux_env():
    """Check if flux_env virtual environment exists and is accessible"""
    flux_env_path = Path("flux_env")
    if not flux_env_path.exists():
        print("❌ flux_env virtual environment not found!")
        print("   Please ensure the virtual environment is set up correctly.")
        return False

    python_path = flux_env_path / "bin" / "python"
    if not python_path.exists():
        print("❌ Python executable not found in flux_env!")
        return False

    print("✅ flux_env virtual environment found")
    return True


def cleanup_port(port: int = 8001):
    """Clean up processes using the specified port"""
    print(f"🧹 Checking port {port} for existing processes...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use lsof to find processes using the port
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                print(
                    f"   Found {len(pids)} process(es) using port {port} (attempt {attempt + 1}/{max_retries})"
                )

                for pid in pids:
                    if pid.strip():
                        try:
                            pid_int = int(pid.strip())
                            print(f"   🚫 Terminating process {pid_int}...")

                            # Try graceful termination first
                            os.kill(pid_int, signal.SIGTERM)
                            time.sleep(2)  # Give more time for graceful shutdown

                            # Check if process is still running
                            try:
                                os.kill(pid_int, 0)  # Check if process exists
                                print(f"   💀 Force killing process {pid_int}...")
                                os.kill(pid_int, signal.SIGKILL)
                                time.sleep(1)
                            except OSError:
                                print(
                                    f"   ✅ Process {pid_int} terminated successfully"
                                )

                        except (ValueError, OSError) as e:
                            print(f"   ⚠️  Could not terminate process {pid}: {e}")

                # Wait longer for port to be released
                time.sleep(3)

                # Verify port is free
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0 and result.stdout.strip():
                    if attempt < max_retries - 1:
                        print(f"   ⚠️  Port {port} still in use, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        print(
                            f"   ❌ Port {port} still in use after {max_retries} attempts"
                        )
                        return False
                else:
                    print(f"   ✅ Port {port} is now free")
                    return True
            else:
                print(f"   ✅ Port {port} is free")
                return True

        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Port cleanup timed out (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return False
        except FileNotFoundError:
            print(f"   ⚠️  lsof not available, trying alternative method...")
            # Try alternative method using netstat
            try:
                result = subprocess.run(
                    ["netstat", "-tlnp"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if f":{port}" in line and "LISTEN" in line:
                            print(
                                f"   🚫 Found process using port {port}, attempting to kill..."
                            )
                            # Extract PID from netstat output
                            parts = line.split()
                            if len(parts) > 6:
                                pid_part = parts[6].split("/")[0]
                                try:
                                    pid_int = int(pid_part)
                                    os.kill(pid_int, signal.SIGKILL)
                                    print(f"   ✅ Killed process {pid_int}")
                                    time.sleep(2)
                                except (ValueError, OSError):
                                    pass
                    return True
                else:
                    print(f"   ⚠️  Alternative cleanup method failed")
                    return False
            except Exception:
                print(f"   ⚠️  Alternative cleanup method not available")
                return False
        except Exception as e:
            print(f"   ⚠️  Port cleanup failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return False

    return False


def check_port_available(port: int = 8001) -> bool:
    """Check if a port is available by attempting to bind to it"""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.bind(("localhost", port))
            s.close()
            return True
    except OSError:
        return False


def wait_for_port_free(port: int = 8001, max_wait: int = 30) -> bool:
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


def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} available")

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")  # type: ignore
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  CUDA not available - GPU acceleration disabled")

        import fastapi

        print(f"✅ FastAPI {fastapi.__version__} available")

        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please activate flux_env and install requirements")
        return False


def start_service():
    """Start the BF16 FLUX API service"""
    print("\n🚀 Starting BF16 FLUX API Service...")
    print("=" * 50)

    # Clean up port before starting
    if not cleanup_port(8001):
        print("   ⚠️  Port cleanup incomplete, but continuing...")

    # Final port verification
    print("🔍 Final port verification...")
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":8001"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            print(f"   ❌ Port 8001 still in use by: {result.stdout.strip()}")
            print("   🚫 Attempting final cleanup...")
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid.strip():
                    try:
                        subprocess.run(["kill", "-9", pid.strip()], timeout=5)
                        print(f"   ✅ Killed process {pid.strip()}")
                    except Exception:
                        pass
            time.sleep(2)
        else:
            print("   ✅ Port 8001 is confirmed free")
    except Exception as e:
        print(f"   ⚠️  Final verification failed: {e}")

    # Wait for port to be truly available
    if not wait_for_port_free(8001, max_wait=30):
        print("   ❌ Port 8001 is not available, cannot start service")
        return False

    # Get the path to the flux_env Python executable
    flux_env_python = Path("flux_env/bin/python")

    # Start the service
    try:
        print("📡 Starting BF16 API server...")

        # Manual GPU selection: respect existing CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        if env.get("CUDA_VISIBLE_DEVICES"):
            print(
                f"🔧 Using CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} for BF16 service"
            )
        else:
            print("⚠️  CUDA_VISIBLE_DEVICES not set; default visible GPU will be used")
        # Reduce fragmentation per PyTorch docs
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        process = subprocess.Popen(
            [str(flux_env_python), "main_bf16.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        print(f"✅ BF16 service started with PID: {process.pid}")
        print("📍 BF16 API URL: http://localhost:8001")
        print("🔍 Health check: http://localhost:8001/health")
        print("📚 API docs: http://localhost:8001/docs")
        print("\n📋 Service logs:")
        print("-" * 50)

        # Stream the output
        if process.stdout:
            for line in process.stdout:
                print(line.rstrip())

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping BF16 service...")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("✅ BF16 service stopped")
    except Exception as e:
        print(f"❌ Failed to start BF16 service: {e}")
        return False

    return True


def main():
    """Main function"""
    print("🐍 BF16 FLUX API Service Starter (Port 8001)")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("main_bf16.py").exists():
        print("❌ main_bf16.py not found in current directory!")
        print("   Please run this script from the flux_api directory.")
        sys.exit(1)

    # Check flux_env
    if not check_flux_env():
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Start the service
    start_service()


if __name__ == "__main__":
    main()
