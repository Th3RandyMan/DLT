"""
DLT Advanced Loss System

Flexible loss system supporting multiple loss functions, dynamic weighting,
permutation invariant training, and domain-specific losses.
Inspired by smart loss architectures for complex ML tasks.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from abc import ABC, abstractmethod


class SmartLoss(ABC):
    """Base class for smart loss functions with advanced features."""
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the loss."""
        pass
    
    def get_loss_info(self) -> Dict[str, Any]:
        """Get information about the loss function."""
        return {
            'name': self.__class__.__name__,
            'requires_logits': getattr(self, 'requires_logits', False),
            'supports_weights': getattr(self, 'supports_weights', True),
        }


class PermutationInvariantLoss(SmartLoss):
    """Base class for permutation invariant training (PIT) losses."""
    
    def __init__(self, base_loss_fn: callable, num_sources: int = 3):
        self.base_loss_fn = base_loss_fn
        self.num_sources = num_sources
        self.supports_weights = True
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute PIT loss by finding best permutation."""
        from itertools import permutations
        
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Generate all permutations
        perms = list(permutations(range(self.num_sources)))
        perm_losses = []
        
        for perm in perms:
            # Reorder predictions according to permutation
            perm_preds = predictions[:, perm]
            # Compute loss for this permutation
            loss = self.base_loss_fn(perm_preds, targets)
            perm_losses.append(loss)
        
        # Stack losses and find minimum for each sample
        perm_losses = torch.stack(perm_losses, dim=0)  # (num_perms, batch_size)
        min_losses, best_perms = torch.min(perm_losses, dim=0)
        
        return min_losses.mean()


class AdaptiveWeightedLoss(SmartLoss):
    """Loss with adaptive weighting inspired by Dynamic Weight Average."""
    
    def __init__(self, loss_functions: List[_Loss], initial_weights: Optional[List[float]] = None, 
                 temperature: float = 2.0, update_freq: int = 1):
        self.loss_functions = loss_functions
        self.temperature = temperature
        self.update_freq = update_freq
        self.step_count = 0
        
        # Initialize weights
        n_losses = len(loss_functions)
        if initial_weights is None:
            self.weights = torch.ones(n_losses) / n_losses
        else:
            self.weights = torch.tensor(initial_weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()
        
        # Track loss history for adaptive weighting
        self.loss_history = []
        self.supports_weights = True
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute weighted loss with adaptive weights."""
        losses = []
        
        # Compute individual losses
        for loss_fn in self.loss_functions:
            loss = loss_fn(predictions, targets)
            losses.append(loss)
        
        losses = torch.stack(losses)
        self.loss_history.append(losses.detach().cpu().numpy())
        
        # Update weights if needed
        if self.step_count % self.update_freq == 0 and len(self.loss_history) > 1:
            self._update_weights()
        
        self.step_count += 1
        
        # Compute weighted loss
        weighted_loss = torch.sum(self.weights.to(losses.device) * losses)
        return weighted_loss
    
    def _update_weights(self):
        """Update weights based on loss history using DWA-style approach."""
        if len(self.loss_history) < 2:
            return
        
        # Calculate relative decrease rate
        current_losses = self.loss_history[-1]
        previous_losses = self.loss_history[-2]
        
        # Avoid division by zero
        previous_losses = np.maximum(previous_losses, 1e-8)
        relative_rates = current_losses / previous_losses
        
        # Apply softmax with temperature
        exp_rates = np.exp(relative_rates / self.temperature)
        new_weights = len(self.loss_functions) * exp_rates / np.sum(exp_rates)
        
        self.weights = torch.tensor(new_weights, dtype=torch.float32)


class FocalLoss(SmartLoss):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.requires_logits = True
        self.supports_weights = True
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiScaleLoss(SmartLoss):
    """Multi-scale loss for different frequency/resolution components."""
    
    def __init__(self, scales: List[int] = [1, 2, 4], weights: Optional[List[float]] = None):
        self.scales = scales
        self.weights = weights if weights else [1.0] * len(scales)
        self.supports_weights = True
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute multi-scale loss."""
        total_loss = 0.0
        
        for i, scale in enumerate(self.scales):
            # Downsample both predictions and targets
            if scale > 1:
                pred_scaled = F.avg_pool1d(predictions.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)
                target_scaled = F.avg_pool1d(targets.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)
            else:
                pred_scaled = predictions
                target_scaled = targets
            
            # Compute MSE loss at this scale
            scale_loss = F.mse_loss(pred_scaled, target_scaled)
            total_loss += self.weights[i] * scale_loss
        
        return total_loss


class SpectralLoss(SmartLoss):
    """Spectral domain loss for signal processing tasks."""
    
    def __init__(self, n_fft: int = 1024, hop_length: int = 512, 
                 magnitude_weight: float = 1.0, phase_weight: float = 0.1):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.supports_weights = True
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute spectral loss."""
        # Compute STFT
        window = torch.hann_window(self.n_fft, device=predictions.device)
        
        pred_stft = torch.stft(predictions, n_fft=self.n_fft, hop_length=self.hop_length, 
                              window=window, return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length, 
                                window=window, return_complex=True)
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        magnitude_loss = F.l1_loss(pred_mag, target_mag)
        
        # Phase loss (using instantaneous frequency)
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        phase_loss = F.l1_loss(pred_phase, target_phase)
        
        return self.magnitude_weight * magnitude_loss + self.phase_weight * phase_loss


class ContrastiveLoss(SmartLoss):
    """Contrastive loss for representation learning."""
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        self.temperature = temperature
        self.margin = margin
        self.supports_weights = True
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute contrastive loss."""
        # Normalize embeddings
        pred_norm = F.normalize(predictions, dim=1)
        target_norm = F.normalize(targets, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(pred_norm, target_norm.T) / self.temperature
        
        # Create labels (positive pairs are on diagonal)
        batch_size = predictions.shape[0]
        labels = torch.arange(batch_size, device=predictions.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class VariationalLoss(SmartLoss):
    """Variational loss for VAE-style models."""
    
    def __init__(self, beta: float = 1.0, reconstruction_loss: str = 'mse'):
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.supports_weights = True
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute variational loss (reconstruction + KLD)."""
        # Reconstruction loss
        if self.reconstruction_loss == 'mse':
            recon_loss = F.mse_loss(predictions['reconstruction'], targets)
        elif self.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy_with_logits(predictions['reconstruction'], targets)
        else:
            raise ValueError(f"Unsupported reconstruction loss: {self.reconstruction_loss}")
        
        # KL divergence
        mu = predictions['mu']
        logvar = predictions['logvar']
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        return recon_loss + self.beta * kld


class LossAgent:
    """Advanced loss agent supporting multiple loss functions and smart weighting."""
    
    def __init__(self, loss_functions: Union[List[Union[_Loss, SmartLoss]], _Loss, SmartLoss],
                 weights: Optional[List[float]] = None, 
                 adaptive_weighting: bool = False,
                 names: Optional[List[str]] = None):
        
        # Normalize loss functions to list
        if not isinstance(loss_functions, list):
            loss_functions = [loss_functions]
        
        self.loss_functions = loss_functions
        self.adaptive_weighting = adaptive_weighting
        self.names = names or [f"loss_{i}" for i in range(len(loss_functions))]
        
        # Initialize weights
        if weights is None:
            self.weights = torch.ones(len(loss_functions)) / len(loss_functions)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()
        
        # Setup adaptive weighting if requested
        if adaptive_weighting and len(loss_functions) > 1:
            self.adaptive_loss = AdaptiveWeightedLoss(loss_functions)
        else:
            self.adaptive_loss = None
        
        # Tracking
        self.loss_history = []
        self.current_losses = {}
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute weighted loss."""
        if self.adaptive_loss:
            return self.adaptive_loss.compute_loss(predictions, targets, **kwargs)
        
        total_loss = 0.0
        self.current_losses = {}
        
        for i, loss_fn in enumerate(self.loss_functions):
            if isinstance(loss_fn, SmartLoss):
                loss = loss_fn.compute_loss(predictions, targets, **kwargs)
            else:
                loss = loss_fn(predictions, targets)
            
            self.current_losses[self.names[i]] = loss.item()
            total_loss += self.weights[i] * loss
        
        self.loss_history.append(self.current_losses.copy())
        return total_loss
    
    def get_loss_info(self) -> Dict[str, Any]:
        """Get information about all loss functions."""
        info = {}
        for i, loss_fn in enumerate(self.loss_functions):
            if isinstance(loss_fn, SmartLoss):
                info[self.names[i]] = loss_fn.get_loss_info()
            else:
                info[self.names[i]] = {
                    'name': loss_fn.__class__.__name__,
                    'requires_logits': False,
                    'supports_weights': True,
                }
        return info
    
    def get_current_losses(self) -> Dict[str, float]:
        """Get current individual loss values."""
        return self.current_losses.copy()
    
    def reset_history(self):
        """Reset loss history."""
        self.loss_history = []
        self.current_losses = {}


# Registry of advanced loss functions
ADVANCED_LOSS_REGISTRY = {
    'focal': FocalLoss,
    'pit': PermutationInvariantLoss,
    'adaptive_weighted': AdaptiveWeightedLoss,
    'multi_scale': MultiScaleLoss,
    'spectral': SpectralLoss,
    'contrastive': ContrastiveLoss,
    'variational': VariationalLoss,
    'vae': VariationalLoss,
}


def create_loss_agent(loss_config: Dict[str, Any]) -> LossAgent:
    """Create loss agent from configuration."""
    loss_type = loss_config.get('type', 'mse')
    
    # Handle single loss function
    if isinstance(loss_type, str):
        if loss_type in ADVANCED_LOSS_REGISTRY:
            loss_fn = ADVANCED_LOSS_REGISTRY[loss_type](**loss_config.get('params', {}))
        else:
            # Standard PyTorch losses
            loss_map = {
                'mse': nn.MSELoss,
                'mae': nn.L1Loss,
                'cross_entropy': nn.CrossEntropyLoss,
                'bce': nn.BCEWithLogitsLoss,
                'huber': nn.HuberLoss,
            }
            loss_fn = loss_map.get(loss_type, nn.MSELoss)()
        
        return LossAgent([loss_fn])
    
    # Handle multiple loss functions
    elif isinstance(loss_type, list):
        loss_functions = []
        weights = loss_config.get('weights', None)
        names = loss_config.get('names', None)
        
        for i, loss_spec in enumerate(loss_type):
            if isinstance(loss_spec, str):
                if loss_spec in ADVANCED_LOSS_REGISTRY:
                    loss_fn = ADVANCED_LOSS_REGISTRY[loss_spec]()
                else:
                    loss_map = {
                        'mse': nn.MSELoss,
                        'mae': nn.L1Loss,
                        'cross_entropy': nn.CrossEntropyLoss,
                        'bce': nn.BCEWithLogitsLoss,
                    }
                    loss_fn = loss_map.get(loss_spec, nn.MSELoss)()
            elif isinstance(loss_spec, dict):
                loss_name = loss_spec['type']
                loss_params = loss_spec.get('params', {})
                if loss_name in ADVANCED_LOSS_REGISTRY:
                    loss_fn = ADVANCED_LOSS_REGISTRY[loss_name](**loss_params)
                else:
                    loss_map = {
                        'mse': nn.MSELoss,
                        'mae': nn.L1Loss,
                        'cross_entropy': nn.CrossEntropyLoss,
                        'bce': nn.BCEWithLogitsLoss,
                    }
                    loss_fn = loss_map.get(loss_name, nn.MSELoss)(**loss_params)
            
            loss_functions.append(loss_fn)
        
        adaptive = loss_config.get('adaptive_weighting', False)
        return LossAgent(loss_functions, weights=weights, adaptive_weighting=adaptive, names=names)
    
    else:
        raise ValueError(f"Unsupported loss configuration: {loss_config}")


def detect_task_and_recommend_loss(output_shape: Tuple, target_characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Automatically detect task type and recommend appropriate loss configuration."""
    num_classes = output_shape[-1] if len(output_shape) > 1 else 1
    target_type = target_characteristics.get('dtype', 'float')
    target_range = target_characteristics.get('range', (0, 1))
    is_multi_class = num_classes > 1
    
    # Classification tasks
    if target_type in ['int', 'long'] or (is_multi_class and target_range[1] < 100):
        if num_classes == 2:
            return {
                'type': 'bce',
                'task_type': 'binary_classification',
                'recommendation': 'Binary classification detected'
            }
        else:
            return {
                'type': 'cross_entropy',
                'task_type': 'multi_class_classification',
                'recommendation': f'Multi-class classification detected ({num_classes} classes)'
            }
    
    # Regression tasks
    elif target_type in ['float', 'double']:
        if target_range[1] - target_range[0] > 10:
            return {
                'type': 'huber',
                'task_type': 'regression',
                'recommendation': 'Robust regression (large range) detected - using Huber loss'
            }
        else:
            return {
                'type': 'mse',
                'task_type': 'regression',
                'recommendation': 'Standard regression detected'
            }
    
    # Default fallback
    return {
        'type': 'mse',
        'task_type': 'unknown',
        'recommendation': 'Could not auto-detect task type, defaulting to MSE'
    }