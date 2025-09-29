"""
DLT Smart Loss System - Intelligent Loss Management

Features:
- Automatic loss detection based on task type
- Weighted loss functions with smart balancing
- Multi-objective loss combinations
- Custom loss functions with easy registration
- Automatic gradient scaling and stability checks
"""

from typing import Any, Dict, Optional, Union, List, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class LossTracker:
    """
    Tracks loss values, handles weighting, and provides smart loss management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loss_history = []
        self.component_history = {}  # For multi-component losses
        self.weights = config.get('weights', None)
        self.adaptive_weighting = config.get('adaptive_weighting', False)
        self.smoothing_factor = config.get('smoothing_factor', 0.1)
        
        # Smart weighting parameters
        self.weight_update_frequency = config.get('weight_update_frequency', 10)
        self.step_count = 0
        
    def update(self, loss_value: float, components: Optional[Dict[str, float]] = None):
        """Update loss tracking with current values."""
        self.loss_history.append(loss_value)
        self.step_count += 1
        
        if components:
            for name, value in components.items():
                if name not in self.component_history:
                    self.component_history[name] = []
                self.component_history[name].append(value)
        
        # Update adaptive weights if enabled
        if (self.adaptive_weighting and 
            self.step_count % self.weight_update_frequency == 0 and
            len(self.component_history) > 1):
            self._update_adaptive_weights()
    
    def _update_adaptive_weights(self):
        """Update weights based on loss component magnitudes and trends."""
        if not self.component_history:
            return
            
        # Calculate relative magnitudes and adjust weights
        current_values = {name: history[-1] for name, history in self.component_history.items()}
        total_magnitude = sum(abs(val) for val in current_values.values())
        
        if total_magnitude > 0:
            # Inverse weighting - give more weight to smaller components
            new_weights = {}
            for name, value in current_values.items():
                relative_mag = abs(value) / total_magnitude
                new_weights[name] = 1.0 / (relative_mag + 1e-8)
            
            # Normalize weights
            weight_sum = sum(new_weights.values())
            self.weights = {name: w / weight_sum for name, w in new_weights.items()}
            
            logger.debug(f"Updated adaptive weights: {self.weights}")
    
    def get_smoothed_loss(self, window: int = 10) -> float:
        """Get exponentially smoothed loss value."""
        if not self.loss_history:
            return 0.0
        
        if len(self.loss_history) <= window:
            return np.mean(self.loss_history)
        
        recent_losses = self.loss_history[-window:]
        weights = np.exp(np.linspace(-1, 0, len(recent_losses)))
        weights /= weights.sum()
        return np.average(recent_losses, weights=weights)
    
    def get_trend(self, window: int = 20) -> str:
        """Analyze loss trend: improving, degrading, or stable."""
        if len(self.loss_history) < window:
            return "insufficient_data"
        
        recent = self.loss_history[-window:]
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        relative_change = (second_half - first_half) / (first_half + 1e-8)
        
        if relative_change < -0.05:  # 5% improvement
            return "improving"
        elif relative_change > 0.05:  # 5% degradation
            return "degrading"
        else:
            return "stable"


class SmartLossFunction(ABC):
    """Base class for smart loss functions with automatic weighting and stability."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.weight = config.get('weight', 1.0)
        self.tracker = LossTracker(config.get('tracking', {}))
        
    @abstractmethod
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss value."""
        pass
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with tracking and weighting."""
        loss = self.forward(predictions, targets)
        
        # Apply weighting
        weighted_loss = loss * self.weight
        
        # Track the loss
        self.tracker.update(weighted_loss.item())
        
        return weighted_loss


class ClassificationLoss(SmartLossFunction):
    """Smart classification loss with class balancing and focal loss options."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.loss_type = config.get('type', 'cross_entropy')
        self.class_weights = config.get('class_weights', None)
        self.focal = config.get('focal', False)
        self.focal_alpha = config.get('focal_alpha', 0.25)
        self.focal_gamma = config.get('focal_gamma', 2.0)
        self.label_smoothing = config.get('label_smoothing', 0.0)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'cross_entropy':
            if self.focal:
                return self._focal_loss(predictions, targets)
            else:
                weights = None
                if self.class_weights:
                    weights = torch.tensor(self.class_weights, device=predictions.device)
                
                return F.cross_entropy(
                    predictions, 
                    targets, 
                    weight=weights,
                    label_smoothing=self.label_smoothing
                )
        elif self.loss_type == 'binary_cross_entropy':
            pos_weight = None
            if self.class_weights and len(self.class_weights) == 2:
                pos_weight = torch.tensor([self.class_weights[1] / self.class_weights[0]], 
                                        device=predictions.device)
            return F.binary_cross_entropy_with_logits(predictions, targets.float(), pos_weight=pos_weight)
        else:
            raise ValueError(f"Unsupported classification loss type: {self.loss_type}")
    
    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()


class RegressionLoss(SmartLossFunction):
    """Smart regression loss with robust alternatives and outlier handling."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.loss_type = config.get('type', 'mse')
        self.huber_delta = config.get('huber_delta', 1.0)
        self.quantile_levels = config.get('quantile_levels', None)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            return F.mse_loss(predictions, targets)
        elif self.loss_type == 'mae':
            return F.l1_loss(predictions, targets)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(predictions, targets, beta=self.huber_delta)
        elif self.loss_type == 'quantile':
            if self.quantile_levels is None:
                raise ValueError("quantile_levels must be specified for quantile loss")
            return self._quantile_loss(predictions, targets)
        else:
            raise ValueError(f"Unsupported regression loss type: {self.loss_type}")
    
    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Quantile regression loss."""
        errors = targets.unsqueeze(-1) - predictions
        losses = torch.max(
            self.quantile_levels * errors,
            (self.quantile_levels - 1) * errors
        )
        return losses.mean()


class CombinedLoss(SmartLossFunction):
    """Combine multiple loss functions with smart weighting."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.loss_functions = []
        self.loss_names = []
        
        # Initialize component losses
        for loss_config in config.get('components', []):
            loss_fn = self._create_loss_function(loss_config)
            self.loss_functions.append(loss_fn)
            self.loss_names.append(loss_config.get('name', f'loss_{len(self.loss_functions)}'))
    
    def _create_loss_function(self, config: Dict[str, Any]) -> SmartLossFunction:
        """Factory method to create loss functions."""
        loss_type = config.get('category', 'classification')
        
        if loss_type == 'classification':
            return ClassificationLoss(config)
        elif loss_type == 'regression':
            return RegressionLoss(config)
        else:
            raise ValueError(f"Unknown loss category: {loss_type}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combine multiple losses with smart weighting."""
        total_loss = 0.0
        components = {}
        
        # Handle multiple outputs if predictions is a tuple/list
        if isinstance(predictions, (tuple, list)):
            pred_list = predictions
        else:
            pred_list = [predictions] * len(self.loss_functions)
        
        # Handle multiple targets if targets is a tuple/list
        if isinstance(targets, (tuple, list)):
            target_list = targets
        else:
            target_list = [targets] * len(self.loss_functions)
        
        for i, (loss_fn, name) in enumerate(zip(self.loss_functions, self.loss_names)):
            pred_i = pred_list[min(i, len(pred_list) - 1)]
            target_i = target_list[min(i, len(target_list) - 1)]
            
            component_loss = loss_fn(pred_i, target_i)
            components[name] = component_loss.item()
            total_loss += component_loss
        
        # Update tracking with components
        self.tracker.update(total_loss.item(), components)
        
        return total_loss


class LossManager:
    """
    Central manager for loss functions with automatic selection and optimization.
    """
    
    def __init__(self, config: Dict[str, Any], model_config: Dict[str, Any] = None, num_classes: int = None):
        self.config = config
        self.model_config = model_config or {}
        self.num_classes = num_classes or self.model_config.get('num_classes', 2)
        self.loss_function = self._create_loss_function()
        self.scaler = None
        
        # Setup gradient scaling for mixed precision
        if config.get('mixed_precision', {}).get('enabled', False):
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=config.get('mixed_precision', {}).get('init_scale', 65536.0),
                growth_factor=config.get('mixed_precision', {}).get('growth_factor', 2.0),
                backoff_factor=config.get('mixed_precision', {}).get('backoff_factor', 0.5),
                growth_interval=config.get('mixed_precision', {}).get('growth_interval', 2000)
            )
    
    def _create_loss_function(self) -> SmartLossFunction:
        """Automatically create appropriate loss function."""
        loss_config = self.config.get('loss', {})
        
        # Auto-detection if not specified
        if not loss_config or loss_config == 'auto':
            return self._auto_detect_loss()
        
        # Handle string specification
        if isinstance(loss_config, str):
            return self._create_from_string(loss_config)
        
        # Handle dictionary configuration
        if isinstance(loss_config, dict):
            loss_type = loss_config.get('category', 'auto')
            
            if loss_type == 'auto':
                return self._auto_detect_loss(loss_config)
            elif loss_type == 'classification':
                return ClassificationLoss(loss_config)
            elif loss_type == 'regression':
                return RegressionLoss(loss_config)
            elif loss_type == 'combined':
                return CombinedLoss(loss_config)
            else:
                raise ValueError(f"Unknown loss category: {loss_type}")
        
        raise ValueError(f"Invalid loss configuration: {loss_config}")
    
    def _auto_detect_loss(self, config: Optional[Dict[str, Any]] = None) -> SmartLossFunction:
        """Automatically detect appropriate loss function based on task."""
        config = config or {}
        
        # Try to infer from model output shape and task
        task_type = self.model_config.get('task_type', 'classification')  # Default assumption
        
        if task_type == 'classification':
            config.update({
                'category': 'classification',
                'type': 'cross_entropy',
                'name': 'auto_classification_loss'
            })
            return ClassificationLoss(config)
        elif task_type == 'regression':
            config.update({
                'category': 'regression',
                'type': 'mse',
                'name': 'auto_regression_loss'
            })
            return RegressionLoss(config)
        else:
            # Default to classification
            logger.warning(f"Unknown task type: {task_type}, defaulting to classification")
            config.update({
                'category': 'classification',
                'type': 'cross_entropy',
                'name': 'auto_default_loss'
            })
            return ClassificationLoss(config)
    
    def _create_from_string(self, loss_str: str) -> SmartLossFunction:
        """Create loss function from string specification."""
        loss_map = {
            'cross_entropy': {'category': 'classification', 'type': 'cross_entropy'},
            'bce': {'category': 'classification', 'type': 'binary_cross_entropy'},
            'focal': {'category': 'classification', 'type': 'cross_entropy', 'focal': True},
            'mse': {'category': 'regression', 'type': 'mse'},
            'mae': {'category': 'regression', 'type': 'mae'},
            'huber': {'category': 'regression', 'type': 'huber'},
        }
        
        if loss_str.lower() in loss_map:
            config = loss_map[loss_str.lower()]
            config['name'] = f'auto_{loss_str}_loss'
            
            if config['category'] == 'classification':
                return ClassificationLoss(config)
            else:
                return RegressionLoss(config)
        else:
            raise ValueError(f"Unknown loss string: {loss_str}")
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute loss with optional component breakdown."""
        loss = self.loss_function(predictions, targets)
        
        if return_components and hasattr(self.loss_function, 'tracker'):
            components = self.loss_function.tracker.component_history
            latest_components = {
                name: history[-1] for name, history in components.items()
                if history
            }
            return loss, latest_components
        
        return loss
    
    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Handle backward pass with optional mixed precision."""
        if self.scaler is not None:
            # Mixed precision backward
            self.scaler.scale(loss).backward()
            
            # Gradient clipping if configured
            max_norm = self.config.get('gradient_clipping', {}).get('max_norm')
            if max_norm:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'], 
                    max_norm
                )
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard backward
            loss.backward()
            
            # Gradient clipping if configured
            max_norm = self.config.get('gradient_clipping', {}).get('max_norm')
            if max_norm:
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'], 
                    max_norm
                )
            
            optimizer.step()
    
    def get_loss_info(self) -> Dict[str, Any]:
        """Get information about the current loss function."""
        info = {
            'loss_type': self.loss_function.__class__.__name__,
            'config': self.loss_function.config,
            'mixed_precision': self.scaler is not None,
        }
        
        if hasattr(self.loss_function, 'tracker'):
            info['recent_loss'] = self.loss_function.tracker.get_smoothed_loss()
            info['trend'] = self.loss_function.tracker.get_trend()
        
        return info
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss using the configured loss function."""
        return self.loss_function(predictions, targets)