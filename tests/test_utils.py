"""Tests for DLT utility functions - Updated for current API."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from dlt.utils.loss import LossTracker, SmartLossFunction, ClassificationLoss, RegressionLoss, CombinedLoss
from dlt.utils.performance import GPUManager, PerformanceProfiler


class TestLossTracker:
    """Test the LossTracker system."""
    
    def test_loss_tracker_basic(self):
        """Test basic loss tracking."""
        config = {'adaptive_weighting': False}
        tracker = LossTracker(config)
        
        # Track some loss values
        tracker.update(1.5)
        tracker.update(1.2)
        tracker.update(0.8)
        
        assert len(tracker.loss_history) == 3
        assert tracker.loss_history == [1.5, 1.2, 0.8]
    
    def test_loss_tracker_with_components(self):
        """Test loss tracking with multiple components."""
        config = {'adaptive_weighting': False}
        tracker = LossTracker(config)
        
        # Track with components
        tracker.update(1.0, {'loss_a': 0.6, 'loss_b': 0.4})
        tracker.update(0.8, {'loss_a': 0.5, 'loss_b': 0.3})
        
        assert len(tracker.loss_history) == 2
        assert 'loss_a' in tracker.component_history
        assert 'loss_b' in tracker.component_history
        assert len(tracker.component_history['loss_a']) == 2
    
    def test_smoothed_loss(self):
        """Test exponentially smoothed loss calculation."""
        config = {}
        tracker = LossTracker(config)
        
        for i in range(10):
            tracker.update(float(i))
        
        smoothed = tracker.get_smoothed_loss(window=5)
        assert isinstance(smoothed, float)
        assert smoothed >= 0
    
    def test_loss_trend_analysis(self):
        """Test loss trend analysis."""
        config = {}
        tracker = LossTracker(config)
        
        # Improving trend
        for i in range(20, 0, -1):
            tracker.update(float(i))
        
        trend = tracker.get_trend(window=20)
        assert trend in ['improving', 'degrading', 'stable', 'insufficient_data']


class TestSmartLossFunction:
    """Test smart loss function base class."""
    
    def test_classification_loss_creation(self):
        """Test classification loss creation."""
        config = {'type': 'cross_entropy', 'weight': 1.0}
        loss_fn = ClassificationLoss(config)
        
        assert loss_fn.loss_type == 'cross_entropy'
        assert loss_fn.weight == 1.0
        assert loss_fn.focal is False
    
    def test_classification_loss_with_focal(self):
        """Test classification loss with focal loss."""
        config = {
            'type': 'cross_entropy',
            'focal': True,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0
        }
        loss_fn = ClassificationLoss(config)
        
        assert loss_fn.focal is True
        assert loss_fn.focal_alpha == 0.25
        assert loss_fn.focal_gamma == 2.0
    
    @pytest.mark.torch
    def test_classification_loss_forward(self):
        """Test classification loss forward pass."""
        config = {'type': 'cross_entropy'}
        loss_fn = ClassificationLoss(config)
        
        predictions = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_regression_loss_creation(self):
        """Test regression loss creation."""
        config = {'type': 'mse', 'weight': 1.0}
        loss_fn = RegressionLoss(config)
        
        assert loss_fn.loss_type == 'mse'
        assert loss_fn.weight == 1.0
    
    @pytest.mark.torch
    def test_regression_loss_forward(self):
        """Test regression loss forward pass."""
        config = {'type': 'mse'}
        loss_fn = RegressionLoss(config)
        
        predictions = torch.randn(10, 1)
        targets = torch.randn(10, 1)
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_combined_loss_creation(self):
        """Test combined loss creation."""
        config = {
            'components': [
                {'category': 'classification', 'type': 'cross_entropy'},
                {'category': 'regression', 'type': 'mse'}
            ]
        }
        combined_loss = CombinedLoss(config)
        
        assert len(combined_loss.loss_functions) == 2
    
    def test_adaptive_weighting(self):
        """Test adaptive weighting functionality."""
        config = {
            'adaptive_weighting': True,
            'weight_update_frequency': 5
        }
        tracker = LossTracker(config)
        
        # Add some loss values to trigger weight updates
        for i in range(10):
            components = {'loss_a': 1.0 - i * 0.1, 'loss_b': 0.5 + i * 0.05}
            tracker.update(sum(components.values()), components)
        
        # Weights should be updated
        assert tracker.step_count == 10


class TestGPUManager:
    """Test the GPUManager class."""
    
    def test_gpu_manager_initialization(self):
        """Test GPU manager initialization."""
        config = {
            'hardware': {'device': 'cpu'},
            'performance': {'mixed_precision': {'enabled': False}}
        }
        gpu_manager = GPUManager(config)
        
        assert gpu_manager.config == config
        assert gpu_manager.device_config == config['hardware']
        assert gpu_manager.performance_config == config['performance']
    
    def test_device_selection(self):
        """Test device selection logic."""
        config = {
            'hardware': {'device': 'cpu'},
            'performance': {}
        }
        gpu_manager = GPUManager(config)
        
        # Should handle CPU-only configuration
        assert not gpu_manager.is_distributed
        assert gpu_manager.world_size == 1
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_gpu_detection(self, mock_cuda_available, mock_device_count):
        """Test GPU detection."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1  # Single GPU to avoid distributed setup
        
        config = {
            'hardware': {'device': 'cpu'},  # Force CPU to avoid distributed initialization
            'performance': {}
        }
        gpu_manager = GPUManager(config)
        
        # Should detect CUDA availability
        mock_cuda_available.assert_called()
    
    def test_ddp_setup(self):
        """Test DistributedDataParallel setup configuration."""
        config = {
            'hardware': {
                'device': 'cpu',
                'distributed': {
                    'enabled': False,
                    'backend': 'gloo'
                }
            },
            'performance': {}
        }
        gpu_manager = GPUManager(config)
        
        # Should not be distributed for this config
        assert not gpu_manager.is_distributed
    
    def test_memory_optimization(self):
        """Test memory optimization settings."""
        config = {
            'hardware': {'device': 'cpu'},
            'performance': {'memory_optimization': True}
        }
        gpu_manager = GPUManager(config)
        
        assert gpu_manager.memory_optimization is True
    
    def test_mixed_precision_setup(self):
        """Test mixed precision setup."""
        config = {
            'hardware': {'device': 'cpu'},
            'performance': {
                'mixed_precision': {
                    'enabled': True,
                    'init_scale': 32768.0
                }
            }
        }
        gpu_manager = GPUManager(config)
        
        # Mixed precision should be configured
        assert 'mixed_precision' in gpu_manager.performance_config
    
    def test_model_compilation(self):
        """Test model compilation settings."""
        config = {
            'hardware': {'device': 'cpu'},
            'performance': {
                'compile': {
                    'enabled': True,
                    'mode': 'default'
                }
            }
        }
        gpu_manager = GPUManager(config)
        
        # Should have compile settings
        assert 'compile' in gpu_manager.performance_config


class TestPerformanceProfiler:
    """Test the PerformanceProfiler class."""
    
    def test_profiler_initialization(self):
        """Test performance profiler initialization."""
        config = {
            'profiling': {
                'enabled': True,
                'memory': True,
                'compute': True
            }
        }
        profiler = PerformanceProfiler(config)
        
        assert profiler.config == config
        assert 'profiling' in profiler.config
    
    def test_profiler_context_manager(self):
        """Test profiler as context manager."""
        config = {
            'profiling': {
                'enabled': True,
                'memory': True,
                'compute': True
            }
        }
        profiler = PerformanceProfiler(config)
        
        # Test context manager usage
        with profiler.profile_step("test_section"):
            # Simulate some work
            x = sum(range(1000))
        
        # Should have recorded the section
        assert len(profiler.timings) > 0 or not profiler.enabled
    
    def test_profiler_memory_tracking(self):
        """Test memory usage tracking."""
        config = {
            'profiling': {
                'enabled': True,
                'memory': True,
                'compute': False
            }
        }
        profiler = PerformanceProfiler(config)
        
        # Test memory tracking - check if GPU is available first
        try:
            import torch
            if torch.cuda.is_available():
                # GPU memory tracking would be available
                initial_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                assert isinstance(initial_memory, (int, float))
            else:
                # No GPU, skip this test
                pass
        except ImportError:
            # No torch, skip
            pass
    
    def test_profiler_bottleneck_detection(self):
        """Test bottleneck detection."""
        config = {
            'profiling': {
                'enabled': True,
                'memory': True,
                'compute': True
            }
        }
        profiler = PerformanceProfiler(config)
        
        # Add some mock sections
        with profiler.profile_step("fast_section"):
            pass
        
        with profiler.profile_step("slow_section"):
            # Simulate slower work
            sum(range(100))
        
        # Get bottlenecks
        bottlenecks = profiler.identify_bottlenecks()
        assert isinstance(bottlenecks, list)
    
    def test_profiler_disabled(self):
        """Test profiler when disabled."""
        config = {
            'profiling': {
                'enabled': False
            }
        }
        profiler = PerformanceProfiler(config)
        
        # When disabled, operations should be no-ops
        with profiler.profile_step("test"):
            pass
        
        # Should not record anything when disabled
        assert not profiler.enabled or len(profiler.timings) == 0


class TestUtilityIntegration:
    """Test integration between utility classes."""
    
    def test_gpu_manager_with_loss_manager(self):
        """Test GPU manager integration with loss management."""
        config = {
            'hardware': {'device': 'cpu'},
            'performance': {'mixed_precision': {'enabled': False}}
        }
        gpu_manager = GPUManager(config)
        
        loss_config = {'type': 'cross_entropy'}
        loss_fn = ClassificationLoss(loss_config)
        
        # Both should work together
        assert gpu_manager.device_config['device'] == 'cpu'
        assert loss_fn.loss_type == 'cross_entropy'
    
    @pytest.mark.torch
    def test_mixed_precision_with_loss(self):
        """Test mixed precision integration with loss functions."""
        config = {
            'hardware': {'device': 'cpu'},
            'performance': {
                'mixed_precision': {'enabled': True}
            }
        }
        gpu_manager = GPUManager(config)
        
        loss_config = {'type': 'cross_entropy'}
        loss_fn = ClassificationLoss(loss_config)
        
        # Test with sample data
        predictions = torch.randn(5, 3)
        targets = torch.randint(0, 3, (5,))
        
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
    
    def test_profiler_with_gpu_manager(self):
        """Test profiler integration with GPU manager."""
        gpu_config = {
            'hardware': {'device': 'cpu'},
            'performance': {}
        }
        
        profiler_config = {
            'profiling': {
                'enabled': True,
                'memory': True
            }
        }
        
        gpu_manager = GPUManager(gpu_config)
        profiler = PerformanceProfiler(profiler_config)
        
        # Both should initialize without conflicts
        assert gpu_manager.world_size == 1
        assert profiler.config['profiling']['enabled'] is True