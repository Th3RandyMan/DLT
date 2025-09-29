"""
DLT GPU and Performance Management

Features:
- Automatic multi-GPU detection and setup
- DistributedDataParallel for optimal multi-GPU training
- Optional mixed precision training
- Memory optimization and monitoring
- Performance profiling and bottleneck detection
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from contextlib import contextmanager
import psutil
import time
import numpy as np

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Manages GPU resources, multi-GPU training, and performance optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device_config = config.get('hardware', {})
        self.performance_config = config.get('performance', {})
        
        # GPU detection and setup
        self.available_gpus = self._detect_gpus()
        self.target_gpus = self._parse_gpu_config()
        self.world_size = len(self.target_gpus) if self.target_gpus else 1
        self.is_distributed = self.world_size > 1
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Performance optimization flags
        self.mixed_precision_enabled = self._should_use_mixed_precision()
        self.compile_enabled = self._should_compile_model()
        self.memory_optimization = self.performance_config.get('memory_optimization', True)
        
        # Setup distributed training if needed
        if self.is_distributed and not dist.is_initialized():
            self._setup_distributed()
        
        logger.info(f"GPU Manager initialized: {len(self.available_gpus)} GPUs available, "
                   f"using {len(self.target_gpus) if self.target_gpus else 1} GPU(s)")
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs."""
        if not torch.cuda.is_available():
            return []
        
        gpu_count = torch.cuda.device_count()
        available_gpus = []
        
        for i in range(gpu_count):
            try:
                # Test GPU accessibility
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                available_gpus.append(i)
                
                # Log GPU info
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"GPU {i}: {props.name}, {memory_gb:.1f}GB memory")
                
            except Exception as e:
                logger.warning(f"GPU {i} not accessible: {e}")
        
        return available_gpus
    
    def _parse_gpu_config(self) -> List[int]:
        """Parse GPU configuration from config."""
        device_setting = self.device_config.get('device', 'auto')
        gpu_ids = self.device_config.get('gpu_ids', None)
        
        if device_setting == 'cpu':
            return []
        elif device_setting == 'auto':
            # Use all available GPUs
            return self.available_gpus
        elif isinstance(device_setting, str) and device_setting.startswith('cuda'):
            # Single GPU specified (e.g., 'cuda:0')
            try:
                gpu_id = int(device_setting.split(':')[1])
                return [gpu_id] if gpu_id in self.available_gpus else []
            except:
                return self.available_gpus[:1] if self.available_gpus else []
        elif gpu_ids is not None:
            # Explicit GPU list
            if isinstance(gpu_ids, int):
                gpu_ids = [gpu_ids]
            return [gpu for gpu in gpu_ids if gpu in self.available_gpus]
        else:
            return self.available_gpus
    
    def _should_use_mixed_precision(self) -> bool:
        """Determine if mixed precision should be enabled."""
        mp_config = self.performance_config.get('mixed_precision', {})
        
        # Explicit setting
        if 'enabled' in mp_config:
            return mp_config['enabled']
        
        # Auto-enable for compatible GPUs
        if not self.available_gpus:
            return False
        
        # Check GPU compute capability (Tensor Cores available on 7.0+)
        for gpu_id in self.target_gpus:
            props = torch.cuda.get_device_properties(gpu_id)
            major, minor = props.major, props.minor
            compute_capability = major + minor / 10.0
            
            if compute_capability >= 7.0:  # V100, RTX 20/30/40 series, A100, etc.
                return True
        
        return False
    
    def _should_compile_model(self) -> bool:
        """Determine if model compilation should be enabled."""
        compile_config = self.performance_config.get('compile', {})
        
        # Explicit setting
        if 'enabled' in compile_config:
            return compile_config['enabled']
        
        # Auto-enable for PyTorch 2.0+ with compatible setup
        if hasattr(torch, 'compile'):
            # Enable by default for multi-GPU or large models
            return self.is_distributed or len(self.available_gpus) > 0
        
        return False
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        # Use NCCL backend for multi-GPU training
        backend = 'nccl' if self.available_gpus else 'gloo'
        
        # Initialize process group
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
        if 'MASTER_PORT' not in os.environ:
            # Find an available port dynamically
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
            os.environ['MASTER_PORT'] = str(port)
        
        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.local_rank
        )
        
        # Set device for this process
        if self.available_gpus:
            torch.cuda.set_device(self.local_rank)
        
        logger.info(f"Distributed training initialized: rank {self.local_rank}/{self.world_size}")
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for this process."""
        if self.available_gpus:
            if self.is_distributed:
                return torch.device(f'cuda:{self.local_rank}')
            else:
                return torch.device(f'cuda:{self.target_gpus[0]}')
        else:
            return torch.device('cpu')
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap model for optimal training (DDP, compilation, etc.).
        """
        device = self.get_device()
        model = model.to(device)
        
        # Apply model compilation if enabled
        if self.compile_enabled:
            try:
                compile_config = self.performance_config.get('compile', {})
                mode = compile_config.get('mode', 'default')  # default, reduce-overhead, max-autotune
                
                logger.info(f"Compiling model with mode: {mode}")
                model = torch.compile(model, mode=mode)
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Wrap with DistributedDataParallel for multi-GPU
        if self.is_distributed:
            # Find unused parameters for complex models
            find_unused_parameters = self.performance_config.get('find_unused_parameters', False)
            
            model = DDP(
                model,
                device_ids=[self.local_rank] if self.available_gpus else None,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=True,  # Memory optimization
                static_graph=self.performance_config.get('static_graph', False)
            )
            
            logger.info("Model wrapped with DistributedDataParallel")
        elif len(self.target_gpus) > 1 and not self.is_distributed:
            # Use DataParallel as fallback (not recommended but available)
            logger.warning("Using DataParallel instead of DistributedDataParallel. "
                          "For best performance, use distributed training.")
            model = torch.nn.DataParallel(model, device_ids=self.target_gpus)
        
        return model
    
    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """
        Create optimized DataLoader with distributed sampling if needed.
        """
        # Distributed sampler for multi-GPU
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        
        # Performance optimizations
        num_workers = self.device_config.get('num_workers', 4)
        pin_memory = (self.device_config.get('pin_memory', True) and 
                     len(self.available_gpus) > 0)
        
        # Adjust batch size for distributed training
        if self.is_distributed:
            batch_size = batch_size // self.world_size
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            **kwargs
        )
        
        return dataloader
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        info = {}
        
        # System memory
        mem = psutil.virtual_memory()
        info['system_memory'] = {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_percent': mem.percent
        }
        
        # GPU memory
        if self.available_gpus:
            info['gpu_memory'] = {}
            for gpu_id in self.available_gpus:
                try:
                    torch.cuda.set_device(gpu_id)
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    cached = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    
                    info['gpu_memory'][f'gpu_{gpu_id}'] = {
                        'total_gb': total,
                        'allocated_gb': allocated,
                        'cached_gb': cached,
                        'free_gb': total - cached
                    }
                except Exception as e:
                    logger.warning(f"Could not get memory info for GPU {gpu_id}: {e}")
        
        return info
    
    def optimize_memory(self):
        """Apply memory optimizations."""
        if not self.memory_optimization:
            return
        
        # Clear GPU cache
        if self.available_gpus:
            torch.cuda.empty_cache()
        
        # Set memory management options
        if self.available_gpus:
            # Enable memory pool for better allocation
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self.device_config.get('memory_fraction', 0.95)
                )
            except:
                pass
        
        logger.info("Memory optimization applied")
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.mixed_precision_enabled and self.available_gpus:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        if self.available_gpus:
            torch.cuda.empty_cache()


class PerformanceProfiler:
    """
    Performance profiler for identifying training bottlenecks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('profiling', {}).get('enabled', False)
        self.profile_memory = config.get('profiling', {}).get('memory', True)
        self.profile_compute = config.get('profiling', {}).get('compute', True)
        
        self.timings = {}
        self.memory_snapshots = []
        self.step_times = []
    
    @contextmanager
    def profile_step(self, step_name: str):
        """Profile a training step."""
        if not self.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        start_memory = None
        
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        
        try:
            yield
        finally:
            if self.profile_compute:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                if step_name not in self.timings:
                    self.timings[step_name] = []
                self.timings[step_name].append(duration)
            
            if self.profile_memory and start_memory is not None:
                end_memory = torch.cuda.memory_allocated()
                memory_delta = (end_memory - start_memory) / (1024**2)  # MB
                
                self.memory_snapshots.append({
                    'step': step_name,
                    'memory_delta_mb': memory_delta,
                    'total_memory_mb': end_memory / (1024**2)
                })
    
    def add_step_time(self, step_time: float):
        """Add overall step timing."""
        self.step_times.append(step_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analysis summary."""
        if not self.enabled:
            return {'profiling_disabled': True}
        
        summary = {}
        
        # Timing analysis
        if self.timings:
            timing_analysis = {}
            for step_name, times in self.timings.items():
                if times:
                    timing_analysis[step_name] = {
                        'avg_ms': np.mean(times) * 1000,
                        'min_ms': np.min(times) * 1000,
                        'max_ms': np.max(times) * 1000,
                        'std_ms': np.std(times) * 1000,
                        'count': len(times)
                    }
            summary['step_timings'] = timing_analysis
        
        # Overall throughput
        if self.step_times:
            summary['throughput'] = {
                'avg_step_time_ms': np.mean(self.step_times) * 1000,
                'steps_per_second': 1.0 / np.mean(self.step_times),
                'total_steps': len(self.step_times)
            }
        
        # Memory analysis
        if self.memory_snapshots:
            total_memory_usage = [s['total_memory_mb'] for s in self.memory_snapshots]
            memory_deltas = [s['memory_delta_mb'] for s in self.memory_snapshots]
            
            summary['memory_usage'] = {
                'peak_memory_mb': np.max(total_memory_usage),
                'avg_memory_mb': np.mean(total_memory_usage),
                'avg_memory_delta_mb': np.mean(memory_deltas),
                'max_memory_delta_mb': np.max(memory_deltas)
            }
        
        return summary
    
    def identify_bottlenecks(self) -> List[str]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []
        
        if not self.enabled or not self.timings:
            return bottlenecks
        
        # Find slowest operations
        avg_times = {name: np.mean(times) for name, times in self.timings.items()}
        total_time = sum(avg_times.values())
        
        for name, avg_time in avg_times.items():
            if avg_time / total_time > 0.3:  # Taking more than 30% of time
                bottlenecks.append(f"Operation '{name}' is taking {avg_time/total_time*100:.1f}% of time")
        
        # Check memory efficiency
        if self.memory_snapshots:
            large_allocations = [s for s in self.memory_snapshots if s['memory_delta_mb'] > 100]
            if large_allocations:
                bottlenecks.append(f"Found {len(large_allocations)} steps with >100MB memory allocation")
        
        # Check throughput
        if self.step_times and len(self.step_times) > 10:
            recent_times = self.step_times[-10:]
            early_times = self.step_times[:10]
            
            if np.mean(recent_times) > np.mean(early_times) * 1.2:
                bottlenecks.append("Training speed is decreasing over time (possible memory leak)")
        
        return bottlenecks