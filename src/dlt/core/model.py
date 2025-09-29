"""
DLT Model System - Universal ML/DL Model Wrapper

Supports any model architecture with consistent API:
- Scikit-learn models (RandomForest, SVM, etc.)
- PyTorch models (CNN, RNN, Transformer, custom architectures)
- TensorFlow/Keras models
- JAX models
- Custom models with fit/predict interface
"""

from typing import Any, Dict, Optional, Union, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from sklearn.base import BaseEstimator
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DLTModel(ABC):
    """
    Universal model wrapper that provides consistent API for any ML/DL model.
    
    Automatically handles:
    - Model creation from config
    - Framework-specific optimizations
    - Device management (CPU/GPU/MPS)
    - Memory efficiency
    - Serialization/deserialization
    
    Examples:
        # Scikit-learn model
        model = DLTModel.from_config(DLTConfig(
            model_type="sklearn.ensemble.RandomForestClassifier",
            model_params={"n_estimators": 100}
        ))
        
        # PyTorch CNN
        model = DLTModel.from_config(DLTConfig(
            model_type="torch.nn.Sequential",
            model_params={
                "layers": [
                    {"type": "Conv2d", "in_channels": 3, "out_channels": 64, "kernel_size": 3},
                    {"type": "ReLU"},
                    {"type": "MaxPool2d", "kernel_size": 2},
                    {"type": "Flatten"},
                    {"type": "Linear", "in_features": 64*112*112, "out_features": 10}
                ]
            }
        ))
        
        # Custom model
        class MyModel(DLTModel):
            def fit(self, X, y): ...
            def predict(self, X): ...
    """
    
    def __init__(self, config: "DLTConfig"):
        self.config = config
        self._model = None
        self._framework = config.get_framework()
        self._device = self._get_device()
        
    @classmethod
    def from_config(cls, config: "DLTConfig") -> "DLTModel":
        """Factory method to create appropriate model wrapper based on framework."""
        framework = config.get_framework()
        
        if framework == 'sklearn':
            return SklearnModelWrapper(config)
        elif framework == 'torch':
            return TorchModelWrapper(config)
        elif framework == 'tensorflow':
            return TensorFlowModelWrapper(config)
        else:
            return CustomModelWrapper(config)
    
    def _get_device(self) -> str:
        """Auto-detect optimal device."""
        if hasattr(self.config, 'hardware') and isinstance(self.config.hardware, dict):
            device_config = self.config.hardware.get('device', 'auto')
        else:
            device_config = 'auto'
        
        if device_config != 'auto':
            return device_config
            
        # Auto-detection
        if HAS_TORCH and torch.cuda.is_available():
            return 'cuda'
        elif HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    @abstractmethod
    def build_model(self) -> Any:
        """Build the actual model from config."""
        pass
    
    @abstractmethod 
    def fit(self, X: Any, y: Any, **kwargs) -> "DLTModel":
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions."""
        pass
    
    def predict_proba(self, X: Any, **kwargs) -> Optional[Any]:
        """Predict class probabilities (if supported)."""
        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(X, **kwargs)
        return None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._framework == 'sklearn':
            import joblib
            joblib.dump(self._model, path)
        elif self._framework == 'torch':
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'config': self.config.to_dict()
            }, path)
        elif self._framework == 'tensorflow':
            self._model.save(path)
        else:
            # Custom serialization
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self._model, f)
    
    @classmethod
    def load(cls, path: Union[str, Path], config: Optional["DLTConfig"] = None) -> "DLTModel":
        """Load model from disk."""
        path = Path(path)
        
        # Try to infer framework from file
        if path.suffix == '.pkl':
            framework = 'sklearn'
        elif path.suffix in ['.pt', '.pth']:
            framework = 'torch'
        elif path.is_dir():  # TensorFlow SavedModel
            framework = 'tensorflow'
        else:
            framework = 'custom'
        
        if framework == 'sklearn':
            import joblib
            model_obj = joblib.load(path)
            wrapper = SklearnModelWrapper(config)
            wrapper._model = model_obj
            return wrapper
        elif framework == 'torch':
            checkpoint = torch.load(path, map_location='cpu')
            if config is None:
                from .config import DLTConfig
                config = DLTConfig(**checkpoint['config'])
            wrapper = TorchModelWrapper(config)
            wrapper.build_model()
            wrapper._model.load_state_dict(checkpoint['model_state_dict'])
            return wrapper
        # Add other frameworks as needed
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        info = {
            'framework': self._framework,
            'device': self._device,
            'model_type': self.config.model_type,
            'parameters': self._get_param_count() if hasattr(self, '_get_param_count') else None
        }
        return info


class SklearnModelWrapper(DLTModel):
    """Wrapper for scikit-learn models."""
    
    def __init__(self, config: "DLTConfig"):
        super().__init__(config)
        # Build the model immediately for sklearn
        self.build_model()
    
    def build_model(self) -> Any:
        """Build sklearn model from config."""
        model_class = self.config.get_model_class()
        self._model = model_class(**self.config.model_params)
        return self._model
    
    def fit(self, X: Any, y: Any, **kwargs) -> "DLTModel":
        """Train sklearn model."""
        if self._model is None:
            self.build_model()
        
        self._model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions with sklearn model."""
        return self._model.predict(X, **kwargs)


class TorchModelWrapper(DLTModel):
    """Wrapper for PyTorch models with optimization."""
    
    def __init__(self, config: "DLTConfig"):
        super().__init__(config)
        self.model = self.build_model()  # Build the model immediately
    
    def build_model(self) -> Any:
        """Build PyTorch model from config."""
        model_params = self.config.model_params.copy()
        
        if self.config.model_type == "torch.nn.Sequential":
            # Special handling for Sequential models defined by layers
            layers = []
            for layer_config in model_params.get('layers', []):
                # Create a copy to avoid modifying original config
                layer_config_copy = layer_config.copy()
                layer_type = layer_config_copy.pop('type')
                if hasattr(nn, layer_type):
                    layer_class = getattr(nn, layer_type)
                    layers.append(layer_class(**layer_config_copy))
            self._model = nn.Sequential(*layers)
        else:
            # Standard model creation
            model_class = self.config.get_model_class()
            self._model = model_class(**model_params)
        
        # Move to device and apply optimizations
        self._model = self._model.to(self._device)
        
        # PyTorch 2.0+ compilation for efficiency
        compile_enabled = False
        if isinstance(self.config.hardware, dict):
            compile_enabled = self.config.hardware.get('compile', False)
            
        if compile_enabled and hasattr(torch, 'compile'):
            self._model = torch.compile(self._model)
            
        return self._model
    
    def fit(self, X: Any, y: Any, **kwargs) -> "DLTModel":
        """Train PyTorch model (this would be handled by DLTTrainer typically)."""
        if self._model is None:
            self.build_model()
        
        # Basic training loop - in practice, use DLTTrainer for full functionality
        self._model.train()
        # Training logic would go here...
        return self
    
    def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions with PyTorch model."""
        self._model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float().to(self._device)
            elif torch.is_tensor(X):
                X = X.to(self._device)
            
            outputs = self._model(X)
            
            # Return numpy array for consistency
            if torch.is_tensor(outputs):
                return outputs.cpu().numpy()
            return outputs
    
    def _get_param_count(self) -> int:
        """Get number of trainable parameters."""
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)


class TensorFlowModelWrapper(DLTModel):
    """Wrapper for TensorFlow/Keras models."""
    
    def build_model(self) -> Any:
        """Build TensorFlow model from config."""
        model_class = self.config.get_model_class()
        self._model = model_class(**self.config.model_params)
        return self._model
    
    def fit(self, X: Any, y: Any, **kwargs) -> "DLTModel":
        """Train TensorFlow model."""
        if self._model is None:
            self.build_model()
        
        self._model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions with TensorFlow model."""
        return self._model.predict(X, **kwargs)


class CustomModelWrapper(DLTModel):
    """Wrapper for custom models with fit/predict interface."""
    
    def build_model(self) -> Any:
        """Build custom model from config."""
        model_class = self.config.get_model_class()
        self._model = model_class(**self.config.model_params)
        return self._model
    
    def fit(self, X: Any, y: Any, **kwargs) -> "DLTModel":
        """Train custom model."""
        if self._model is None:
            self.build_model()
        
        if hasattr(self._model, 'fit'):
            self._model.fit(X, y, **kwargs)
        else:
            raise NotImplementedError("Custom model must implement 'fit' method")
        
        return self
    
    def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions with custom model."""
        if hasattr(self._model, 'predict'):
            return self._model.predict(X, **kwargs)
        else:
            raise NotImplementedError("Custom model must implement 'predict' method")