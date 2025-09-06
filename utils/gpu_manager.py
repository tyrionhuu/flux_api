"""
Comprehensive GPU Management for Diffusion API
Handles GPU selection, service assignment, monitoring, and dynamic adaptation.
"""

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
from loguru import logger


@dataclass
class ServiceInfo:
    """Information about a running service"""

    name: str
    required_memory_gb: float
    current_gpu: Optional[int]
    start_time: float
    last_activity: float
    priority: int = 1  # Higher number = higher priority


@dataclass
class GPUStatus:
    """Real-time GPU status"""

    gpu_id: int
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float
    temperature: float
    utilization: float
    last_update: float
    services: Set[str]  # Services currently using this GPU


class GPUManager:
    """Comprehensive GPU manager with dynamic service assignment and monitoring"""

    def __init__(self, monitoring_interval: float = 10.0):
        self.selected_gpu: Optional[int] = None
        self.device_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )

        # Service tracking
        self.services: Dict[str, ServiceInfo] = {}
        self.gpu_status: Dict[int, GPUStatus] = {}

        # Dynamic assignment state
        self.assignment_lock = threading.Lock()
        self.monitoring_thread = None
        self.running = False

        # Performance thresholds
        self.memory_threshold = 0.8  # 80% memory usage triggers rebalancing
        self.temperature_threshold = 80.0  # 80°C triggers migration
        self.utilization_threshold = 0.9  # 90% utilization triggers rebalancing

        # Initialize GPU status lazily to avoid blocking app startup
        self.status_initialized = False

    def _initialize_gpu_status(self):
        """Initialize GPU status tracking"""
        # Clear and re-populate
        self.gpu_status = {}
        for gpu_id in range(self.device_count):
            try:
                # Get actual GPU properties
                props = torch.cuda.get_device_properties(gpu_id)
                total_memory_gb = props.total_memory / 1024**3

                # Get current memory usage
                torch.cuda.set_device(gpu_id)
                used_memory_gb = torch.cuda.memory_allocated(gpu_id) / 1024**3
                free_memory_gb = total_memory_gb - used_memory_gb

                self.gpu_status[gpu_id] = GPUStatus(
                    gpu_id=gpu_id,
                    total_memory_gb=total_memory_gb,
                    used_memory_gb=used_memory_gb,
                    free_memory_gb=free_memory_gb,
                    temperature=0.0,  # Will be updated by monitoring
                    utilization=0.0,  # Will be updated by monitoring
                    last_update=time.time(),
                    services=set(),
                )

                logger.info(
                    f"Initialized GPU {gpu_id}: {total_memory_gb:.1f}GB total, {free_memory_gb:.1f}GB free"
                )

            except Exception as e:
                logger.warning(f"Failed to initialize GPU {gpu_id}: {e}")
                # Fallback to default values
                self.gpu_status[gpu_id] = GPUStatus(
                    gpu_id=gpu_id,
                    total_memory_gb=31.0,  # Assume RTX 5090 size
                    used_memory_gb=0.0,
                    free_memory_gb=31.0,
                    temperature=0.0,
                    utilization=0.0,
                    last_update=time.time(),
                    services=set(),
                )
        self.status_initialized = True

    def _ensure_status_initialized(self):
        if not self.status_initialized:
            try:
                self._initialize_gpu_status()
            except Exception:
                # Best-effort; leave as is if initialization fails
                pass

    def test_cuda_compatibility(self) -> bool:
        """Test if CUDA is actually working (not just available)"""
        if not torch.cuda.is_available():
            return False

        try:
            # Test if we can actually use CUDA (use first available GPU)
            test_tensor = torch.randn((100, 100), device="cuda:0")
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"CUDA compatibility issue: {e}")
            return False

    def select_best_gpu(self) -> Optional[int]:
        """Select GPU with most free memory, accounting for system-wide usage"""
        if not torch.cuda.is_available():
            return None

        # Ensure we have a status snapshot
        self._ensure_status_initialized()

        gpu_memory = []
        for i in range(self.device_count):
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3

            # Get system-wide memory usage (not just PyTorch)
            try:
                # Use nvidia-smi to get real-time memory info
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(i),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if lines and "," in lines[0]:
                        used_memory, total_memory_smi = lines[0].split(",")
                        used_memory = float(used_memory) / 1024  # Convert MB to GB
                        total_memory_smi = (
                            float(total_memory_smi) / 1024
                        )  # Convert MB to GB
                        free_memory = total_memory_smi - used_memory
                    else:
                        # Fallback to PyTorch method
                        free_memory = (
                            torch.cuda.get_device_properties(i).total_memory
                            - torch.cuda.memory_allocated(i)
                        ) / 1024**3
                else:
                    # Fallback to PyTorch method
                    free_memory = (
                        torch.cuda.get_device_properties(i).total_memory
                        - torch.cuda.memory_allocated(i)
                    ) / 1024**3
            except Exception:
                # Fallback to PyTorch method if nvidia-smi fails
                free_memory = (
                    torch.cuda.get_device_properties(i).total_memory
                    - torch.cuda.memory_allocated(i)
                ) / 1024**3

            gpu_memory.append((i, free_memory, total_memory))
            print(f"GPU {i}: {free_memory:.1f}GB free / {total_memory:.1f}GB total")

        # Filter out GPUs with insufficient memory (less than 8GB free for BF16 model)
        sufficient_memory_gpus = [
            (i, free, total) for i, free, total in gpu_memory if free >= 8.0
        ]

        if not sufficient_memory_gpus:
            print("⚠️  No GPU has sufficient free memory (need at least 8GB)")
            # Return GPU with most memory anyway, but warn
            best_gpu = max(gpu_memory, key=lambda x: x[1])
            print(
                f"⚠️  Selecting GPU {best_gpu[0]} with only {best_gpu[1]:.1f}GB free (may fail)"
            )
            self.selected_gpu = best_gpu[0]
            return self.selected_gpu

        # Select GPU with most free memory among those with sufficient memory
        best_gpu = max(sufficient_memory_gpus, key=lambda x: x[1])
        self.selected_gpu = best_gpu[0]
        print(
            f"✅ Selected GPU {self.selected_gpu} with {best_gpu[1]:.1f}GB free memory"
        )
        return self.selected_gpu

    def set_device(self, gpu_id: int) -> bool:
        """Set the current CUDA device"""
        try:
            if 0 <= gpu_id < self.device_count:
                torch.cuda.set_device(gpu_id)
                self.selected_gpu = gpu_id
                return True
            return False
        except Exception as e:
            print(f"Error setting GPU device {gpu_id}: {e}")
            return False

    def get_device_info(self) -> dict:
        """Get detailed GPU information"""
        gpu_info = []

        if torch.cuda.is_available():
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
                free_memory = total_memory - allocated_memory

                gpu_info.append(
                    {
                        "gpu_id": i,
                        "name": props.name,
                        "total_memory_gb": f"{total_memory:.1f}",
                        "allocated_memory_gb": f"{allocated_memory:.1f}",
                        "free_memory_gb": f"{free_memory:.1f}",
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )

        return {
            "cuda_available": torch.cuda.is_available(),
            "device_count": self.device_count,
            "selected_gpu": self.selected_gpu,
            "gpus": gpu_info,
        }

    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        try:
            if torch.cuda.is_available() and self.selected_gpu is not None:
                return torch.cuda.memory_allocated(self.selected_gpu) / 1024**3
            return 0.0
        except:
            return 0.0

    def check_gpu_memory_sufficient(
        self, gpu_id: int, required_memory_gb: float = 16.0
    ) -> bool:
        """Check if a specific GPU has sufficient free memory"""
        try:
            if not torch.cuda.is_available() or gpu_id >= self.device_count:
                print(f"   ❌ GPU {gpu_id}: CUDA not available or invalid GPU ID")
                return False

            # Use nvidia-smi to get real-time memory info
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(gpu_id),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines and "," in lines[0]:
                    used_memory, total_memory = lines[0].split(",")
                    used_memory = float(used_memory) / 1024  # Convert MB to GB
                    total_memory = float(total_memory) / 1024  # Convert MB to GB
                    free_memory = total_memory - used_memory

                    print(
                        f"   📊 GPU {gpu_id}: {used_memory:.1f}GB used, {total_memory:.1f}GB total, {free_memory:.1f}GB free"
                    )
                    print(
                        f"   {'✅' if free_memory >= required_memory_gb else '❌'} GPU {gpu_id}: {'Sufficient' if free_memory >= required_memory_gb else 'Insufficient'} memory ({free_memory:.1f}GB free, {required_memory_gb}GB required)"
                    )

                    return free_memory >= required_memory_gb

            # Fallback to PyTorch method
            print(f"   ⚠️  GPU {gpu_id}: nvidia-smi failed, using PyTorch fallback")
            torch.cuda.set_device(gpu_id)
            total_memory = (
                torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            )
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3
            free_memory = total_memory - allocated_memory

            print(
                f"   📊 GPU {gpu_id}: {allocated_memory:.1f}GB allocated, {total_memory:.1f}GB total, {free_memory:.1f}GB free (PyTorch)"
            )
            print(
                f"   {'✅' if free_memory >= required_memory_gb else '❌'} GPU {gpu_id}: {'Sufficient' if free_memory >= required_memory_gb else 'Insufficient'} memory ({free_memory:.1f}GB free, {required_memory_gb}GB required)"
            )

            return free_memory >= required_memory_gb

        except Exception as e:
            print(f"   ❌ GPU {gpu_id}: Error checking memory: {e}")
            return False

    def find_available_gpu_for_model(
        self, required_memory_gb: float = 16.0
    ) -> Optional[int]:
        """Find a GPU with sufficient memory for a specific model"""
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return None

        self._ensure_status_initialized()

        print(f"🔍 Looking for GPU with at least {required_memory_gb}GB free memory...")
        print(f"📊 Found {self.device_count} GPU(s)")

        available_gpus = []
        for i in range(self.device_count):
            print(f"   Checking GPU {i}...")
            if self.check_gpu_memory_sufficient(i, required_memory_gb):
                print(f"   ✅ GPU {i} has sufficient memory")
                available_gpus.append(i)
            else:
                print(f"   ❌ GPU {i} insufficient memory")

        print(f"📋 Available GPUs: {available_gpus}")

        if not available_gpus:
            print(f"❌ No GPU has sufficient memory ({required_memory_gb}GB required)")
            return None

        # Return the GPU with most free memory (using nvidia-smi for consistency)
        best_gpu = None
        best_free_memory = 0

        for gpu_id in available_gpus:
            try:
                # Use nvidia-smi for consistent memory measurement
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(gpu_id),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if lines and "," in lines[0]:
                        used_memory, total_memory = lines[0].split(",")
                        used_memory = float(used_memory) / 1024  # Convert MB to GB
                        total_memory = float(total_memory) / 1024  # Convert MB to GB
                        free_memory = total_memory - used_memory

                        if free_memory > best_free_memory:
                            best_free_memory = free_memory
                            best_gpu = gpu_id
                else:
                    # Fallback to PyTorch method
                    torch.cuda.set_device(gpu_id)
                    total_memory = (
                        torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    )
                    allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    free_memory = total_memory - allocated_memory

                    if free_memory > best_free_memory:
                        best_free_memory = free_memory
                        best_gpu = gpu_id
            except Exception:
                continue

        if best_gpu is not None:
            print(
                f"✅ Found available GPU {best_gpu} with {best_free_memory:.1f}GB free memory"
            )

        return best_gpu

    def get_optimal_device(self) -> Tuple[str, Optional[int]]:
        """Get the optimal device (GPU or CPU) for the current setup"""
        if self.test_cuda_compatibility():
            selected_gpu = self.select_best_gpu()
            if selected_gpu is not None:
                self.set_device(selected_gpu)
                return f"cuda:{selected_gpu}", selected_gpu
            else:
                return "cpu", None
        else:
            return "cpu", None

    # ===== DYNAMIC SERVICE MANAGEMENT =====

    def start_monitoring(self):
        """Start the dynamic monitoring thread"""
        if self.monitoring_thread is not None:
            return

        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("🚀 Started dynamic GPU monitoring")

    def stop_monitoring(self):
        """Stop the dynamic monitoring thread"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            self.monitoring_thread = None
        logger.info("🛑 Stopped dynamic GPU monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop that continuously adapts GPU assignments"""
        while self.running:
            try:
                # Update GPU status
                self._update_gpu_status()

                # Check for rebalancing opportunities
                self._check_and_rebalance()

                # Handle failures and migrations
                self._handle_failures()

                # Wait for next monitoring cycle
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)

    def _update_gpu_status(self):
        """Update real-time GPU status"""
        for gpu_id in range(self.device_count):
            try:
                # Get current GPU status using nvidia-smi
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(gpu_id),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if lines and "," in lines[0]:
                        used_memory, total_memory, temperature, utilization = lines[
                            0
                        ].split(",")

                        used_memory_gb = float(used_memory) / 1024
                        total_memory_gb = float(total_memory) / 1024
                        free_memory_gb = total_memory_gb - used_memory_gb
                        temperature_c = float(temperature)
                        utilization_pct = float(utilization) / 100

                        with self.assignment_lock:
                            self.gpu_status[gpu_id].used_memory_gb = used_memory_gb
                            self.gpu_status[gpu_id].total_memory_gb = total_memory_gb
                            self.gpu_status[gpu_id].free_memory_gb = free_memory_gb
                            self.gpu_status[gpu_id].temperature = temperature_c
                            self.gpu_status[gpu_id].utilization = utilization_pct
                            self.gpu_status[gpu_id].last_update = time.time()

            except Exception as e:
                logger.warning(f"Failed to update GPU {gpu_id} status: {e}")

    def _check_and_rebalance(self):
        """Check if rebalancing is needed and perform it"""
        with self.assignment_lock:
            # Find GPUs that need rebalancing
            overloaded_gpus = []
            underutilized_gpus = []

            for gpu_id, status in self.gpu_status.items():
                memory_usage = status.used_memory_gb / status.total_memory_gb

                if (
                    memory_usage > self.memory_threshold
                    or status.temperature > self.temperature_threshold
                    or status.utilization > self.utilization_threshold
                ):
                    overloaded_gpus.append(gpu_id)
                elif (
                    memory_usage < 0.3  # Less than 30% usage
                    and status.temperature < 60.0  # Cool temperature
                    and status.utilization < 0.5
                ):  # Low utilization
                    underutilized_gpus.append(gpu_id)

            # Perform rebalancing if needed
            if overloaded_gpus and underutilized_gpus:
                self._perform_rebalancing(overloaded_gpus, underutilized_gpus)

    def _perform_rebalancing(
        self, overloaded_gpus: List[int], underutilized_gpus: List[int]
    ):
        """Move services from overloaded GPUs to underutilized ones"""
        logger.info(
            f"🔄 Rebalancing: Moving services from {overloaded_gpus} to {underutilized_gpus}"
        )

        for overloaded_gpu in overloaded_gpus:
            if overloaded_gpu not in self.gpu_status:
                continue

            # Find services to move
            services_to_move = list(self.gpu_status[overloaded_gpu].services.copy())

            for service_name in services_to_move:
                if service_name not in self.services:
                    continue

                service = self.services[service_name]

                # Find best underutilized GPU for this service
                best_gpu = self._find_best_gpu_for_service(
                    service, exclude_gpus=[overloaded_gpu]
                )

                if best_gpu is not None and best_gpu in underutilized_gpus:
                    # Migrate service
                    if self._migrate_service(service_name, best_gpu):
                        logger.info(
                            f"✅ Migrated {service_name} from GPU {overloaded_gpu} to GPU {best_gpu}"
                        )
                        break

    def _handle_failures(self):
        """Handle GPU failures and migrate services"""
        current_time = time.time()

        for gpu_id, status in self.gpu_status.items():
            # Check if GPU is unresponsive (no updates in last 30 seconds)
            if current_time - status.last_update > 30:
                logger.warning(
                    f"⚠️  GPU {gpu_id} appears unresponsive, migrating services"
                )
                self._handle_gpu_failure(gpu_id)

    def _handle_gpu_failure(self, failed_gpu: int):
        """Handle failure of a specific GPU"""
        if failed_gpu not in self.gpu_status:
            return

        # Get all services on failed GPU
        services_to_migrate = list(self.gpu_status[failed_gpu].services.copy())

        for service_name in services_to_migrate:
            if service_name not in self.services:
                continue

            # Find alternative GPU
            alternative_gpu = self._find_best_gpu_for_service(
                self.services[service_name], exclude_gpus=[failed_gpu]
            )

            if alternative_gpu is not None:
                if self._migrate_service(service_name, alternative_gpu):
                    logger.info(
                        f"✅ Migrated {service_name} from failed GPU {failed_gpu} to GPU {alternative_gpu}"
                    )
            else:
                logger.error(f"❌ No alternative GPU available for {service_name}")

    def _find_best_gpu_for_service(
        self, service: ServiceInfo, exclude_gpus: Optional[List[int]] = None
    ) -> Optional[int]:
        """Find the best GPU for a service, excluding specified GPUs"""
        if exclude_gpus is None:
            exclude_gpus = []

        best_gpu = None
        best_score = -1

        for gpu_id, status in self.gpu_status.items():
            if gpu_id in exclude_gpus:
                continue

            # Get real-time memory via nvidia-smi for accurate selection
            realtime_used_gb = None
            realtime_total_gb = None
            realtime_free_gb = None
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(gpu_id),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    line = result.stdout.strip().split("\n")[0]
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        used_mb = float(parts[0])
                        total_mb = float(parts[1])
                        realtime_used_gb = used_mb / 1024.0
                        realtime_total_gb = total_mb / 1024.0
                        realtime_free_gb = realtime_total_gb - realtime_used_gb
                        # Update status snapshot used for scoring
                        status.used_memory_gb = realtime_used_gb
                        status.total_memory_gb = realtime_total_gb
                        status.free_memory_gb = realtime_free_gb
                        try:
                            status.temperature = float(parts[2])
                        except Exception:
                            pass
                        try:
                            status.utilization = float(parts[3]) / 100.0
                        except Exception:
                            pass
            except Exception:
                # Ignore realtime failures; we'll fall back to existing status
                pass

            free_gb = (
                realtime_free_gb
                if realtime_free_gb is not None
                else status.free_memory_gb
            )
            total_gb = (
                realtime_total_gb
                if realtime_total_gb is not None
                else status.total_memory_gb
            )

            # Check if GPU has sufficient memory now
            if free_gb < service.required_memory_gb:
                continue

            # Calculate GPU score (higher is better)
            memory_score = 0.0
            if total_gb > 0:
                memory_score = free_gb / total_gb
            temperature_score = 1.0 - (
                status.temperature / 100.0
            )  # Lower temperature = higher score
            utilization_score = (
                1.0 - status.utilization
            )  # Lower utilization = higher score

            # Weighted score
            score = (
                memory_score * 0.5 + temperature_score * 0.3 + utilization_score * 0.2
            )

            if score > best_score:
                best_score = score
                best_gpu = gpu_id

        return best_gpu

    def _migrate_service(self, service_name: str, target_gpu: int) -> bool:
        """Migrate a service to a different GPU"""
        if service_name not in self.services:
            return False

        service = self.services[service_name]
        old_gpu = service.current_gpu

        # Update service tracking
        service.current_gpu = target_gpu
        service.last_activity = time.time()

        # Update GPU tracking
        if old_gpu is not None and old_gpu in self.gpu_status:
            self.gpu_status[old_gpu].services.discard(service_name)

        if target_gpu in self.gpu_status:
            self.gpu_status[target_gpu].services.add(service_name)

        # Set environment variable for the service
        os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)

        logger.info(
            f"🔄 Migrated {service_name} from GPU {old_gpu} to GPU {target_gpu}"
        )
        return True

    # ===== SERVICE REGISTRATION AND MANAGEMENT =====

    def register_service(
        self, service_name: str, required_memory_gb: float, priority: int = 1
    ) -> Optional[int]:
        """Register a new service and assign it to the best available GPU"""
        with self.assignment_lock:
            # Check if service already exists
            if service_name in self.services:
                logger.info(
                    f"✅ Service {service_name} already registered on GPU {self.services[service_name].current_gpu}"
                )
                return self.services[service_name].current_gpu

            # Create service info
            service = ServiceInfo(
                name=service_name,
                required_memory_gb=required_memory_gb,
                current_gpu=None,
                start_time=time.time(),
                last_activity=time.time(),
                priority=priority,
            )

            # Refresh GPU status from nvidia-smi before selecting
            try:
                self._update_gpu_status()
            except Exception:
                pass

            # Try dynamic GPU selection first (uses realtime nvidia-smi data when available)
            best_gpu = self._find_best_gpu_for_service(service)

            # Fallback to original GPU selection if dynamic fails
            if best_gpu is None:
                logger.warning(
                    f"⚠️  Dynamic GPU selection failed for {service_name}, trying fallback method"
                )
                best_gpu = self.find_available_gpu_for_model(required_memory_gb)

                if best_gpu is None:
                    logger.error(f"❌ No suitable GPU available for {service_name}")
                    return None
                else:
                    logger.info(f"✅ Fallback GPU selection successful: GPU {best_gpu}")

            # Assign service to GPU
            service.current_gpu = best_gpu
            self.services[service_name] = service

            if best_gpu in self.gpu_status:
                self.gpu_status[best_gpu].services.add(service_name)

            # Set environment variable
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

            logger.info(f"✅ Registered {service_name} on GPU {best_gpu}")
            return best_gpu

    def unregister_service(self, service_name: str) -> bool:
        """Unregister a service and free its GPU"""
        with self.assignment_lock:
            if service_name not in self.services:
                return False

            service = self.services[service_name]
            gpu_id = service.current_gpu

            # Remove from GPU tracking
            if gpu_id is not None and gpu_id in self.gpu_status:
                self.gpu_status[gpu_id].services.discard(service_name)

            # Remove service
            del self.services[service_name]

            logger.info(f"✅ Unregistered {service_name} from GPU {gpu_id}")
            return True

    def get_service_gpu(self, service_name: str) -> Optional[int]:
        """Get the current GPU for a service"""
        with self.assignment_lock:
            if service_name in self.services:
                return self.services[service_name].current_gpu
            return None

    def get_gpu_status_summary(self) -> Dict:
        """Get comprehensive GPU status summary"""
        with self.assignment_lock:
            summary = {
                "services": {},
                "gpus": {},
                "overall": {
                    "total_gpus": len(self.gpu_status),
                    "active_services": len(self.services),
                    "monitoring_active": self.running,
                },
            }

            # Service summary
            for service_name, service in self.services.items():
                summary["services"][service_name] = {
                    "gpu": service.current_gpu,
                    "memory_required": service.required_memory_gb,
                    "uptime": time.time() - service.start_time,
                    "priority": service.priority,
                }

            # GPU summary
            for gpu_id, status in self.gpu_status.items():
                summary["gpus"][gpu_id] = {
                    "memory_used": status.used_memory_gb,
                    "memory_free": status.free_memory_gb,
                    "memory_total": status.total_memory_gb,
                    "temperature": status.temperature,
                    "utilization": status.utilization,
                    "services": list(status.services),
                    "last_update": status.last_update,
                }

            return summary

    def print_status_summary(self):
        """Print current status summary"""
        summary = self.get_gpu_status_summary()

        print("\n🎯 GPU Manager Status")
        print("=" * 60)

        print(
            f"📊 Monitoring: {'🟢 Active' if summary['overall']['monitoring_active'] else '🔴 Inactive'}"
        )
        print(f"📊 Total GPUs: {summary['overall']['total_gpus']}")
        print(f"📊 Active Services: {summary['overall']['active_services']}")

        print("\n🔧 Service Status:")
        print("-" * 40)
        for service_name, info in summary["services"].items():
            print(
                f"✅ {service_name}: GPU {info['gpu']} (Memory: {info['memory_required']}GB, Priority: {info['priority']})"
            )

        print("\n📱 GPU Status:")
        print("-" * 40)
        for gpu_id, info in summary["gpus"].items():
            memory_pct = (info["memory_used"] / info["memory_total"]) * 100
            status_icon = "🟢" if memory_pct < 80 else "🟡" if memory_pct < 95 else "🔴"
            print(
                f"{status_icon} GPU {gpu_id}: {memory_pct:.1f}% used, {info['temperature']:.1f}°C, {info['utilization']*100:.1f}% util"
            )
            if info["services"]:
                print(f"   Services: {', '.join(info['services'])}")


# ===== CONVENIENCE FUNCTIONS =====


def get_gpu_manager():
    """Get the global GPU manager instance"""
    global _gpu_manager_instance
    if "_gpu_manager_instance" not in globals():
        globals()["_gpu_manager_instance"] = GPUManager()
    return globals()["_gpu_manager_instance"]


def start_dynamic_gpu_monitoring():
    """Start the dynamic GPU monitoring system"""
    manager = get_gpu_manager()
    manager.start_monitoring()
    return manager


def stop_dynamic_gpu_monitoring():
    """Stop the dynamic GPU monitoring system"""
    manager = get_gpu_manager()
    manager.stop_monitoring()


def register_service_dynamically(
    service_name: str, required_memory_gb: float, priority: int = 1
) -> Optional[int]:
    """Register a service with dynamic GPU management"""
    manager = get_gpu_manager()
    return manager.register_service(service_name, required_memory_gb, priority)


def unregister_service_dynamically(service_name: str) -> bool:
    """Unregister a service from dynamic GPU management"""
    manager = get_gpu_manager()
    return manager.unregister_service(service_name)


def set_cuda_visible_devices_for_service(
    service_name: str, required_memory_gb: float
) -> Optional[str]:
    """Set CUDA_VISIBLE_DEVICES for a service dynamically"""
    manager = get_gpu_manager()

    # Register service and get GPU assignment
    gpu_id = manager.register_service(service_name, required_memory_gb)

    if gpu_id is None:
        return None

    # Set environment variable
    cuda_devices = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    logger.info(
        f"🔧 Set CUDA_VISIBLE_DEVICES={cuda_devices} for {service_name} (GPU {gpu_id})"
    )

    return cuda_devices
