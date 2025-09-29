"""
DLT Configuration System

Supports any ML/DL model architecture with hierarchical configuration and validation.
Compatible with: scikit-learn, PyTorch, TensorFlow, JAX, and any custom models.
"""

from typing import Any, Dict, Optional, Union, Type, List
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, validator, ConfigDict
import yaml
import json


class DLTConfig(BaseModel):
    """
    Universal configuration class for any ML/DL model architecture.
    
    Supports everything from simple linear regression to complex transformers,
    with automatic validation and smart defaults.
    
    Examples:
        # Basic ML model
        config = DLTConfig(
            model_type="sklearn.RandomForestClassifier",
            model_params={"n_estimators": 100, "random_state": 42}
        )
        
        # Deep learning model
        config = DLTConfig(
            model_type="torch.nn.Sequential",
            model_params={
                "layers": [
                    {"type": "Linear", "in_features": 784, "out_features": 128},
                    {"type": "ReLU"},
                    {"type": "Linear", "in_features": 128, "out_features": 10}
                ]
            }
        )
        
        # Transformer model
        config = DLTConfig(
            model_type="transformers.AutoModel",
            model_params={"model_name": "bert-base-uncased"}
        )
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='allow'  # Allow any additional fields for flexibility
    )
    
    # Core model configuration
    model_type: str = Field(
        description="Model class path (e.g., 'sklearn.ensemble.RandomForestClassifier', 'torch.nn.Sequential')"
    )
    model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters passed to model constructor"
    )
    
    # Training configuration (universal)
    training: Dict[str, Any] = Field(
        default_factory=lambda: {
            "optimizer": {"type": "auto", "lr": 1e-3},
            "loss": {
                "type": "auto",  # Auto-detect based on task
                "weights": None,  # Class weights for imbalanced data
                "adaptive_weighting": False,  # Dynamic loss reweighting
                "focal": False,  # Use focal loss for classification
                "label_smoothing": 0.0  # Label smoothing factor
            },
            "metrics": ["auto"],
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "early_stopping": {"patience": 10, "min_delta": 1e-4},
            "device": "auto",  # auto-detect GPU/CPU
            "scheduler": {
                "type": None,  # None, 'step', 'cosine', 'plateau'
                "step_size": 30,
                "gamma": 0.1,
                "patience": 10
            },
            "gradient_clipping": {
                "enabled": False,
                "max_norm": 1.0
            }
        },
        description="Training configuration for any framework"
    )
    
    # Data configuration (universal)
    data: Dict[str, Any] = Field(
        default_factory=lambda: {
            "input_shape": None,  # Auto-inferred if not provided
            "output_shape": None,  # Auto-inferred if not provided
            "preprocessing": [],  # List of preprocessing steps
            "augmentation": [],  # List of augmentation steps
            "validation_strategy": "random_split"
        },
        description="Data configuration"
    )
    
    # Experiment configuration
    experiment: Dict[str, Any] = Field(
        default_factory=lambda: {
            "name": "dlt_experiment",
            "tags": [],
            "notes": "",
            "seed": 42,
            "logging": {"level": "INFO", "wandb": False, "tensorboard": True}
        },
        description="Experiment tracking and reproducibility"
    )
    
    # Hardware and performance configuration
    hardware: Dict[str, Any] = Field(
        default_factory=lambda: {
            "device": "auto",  # auto, cpu, cuda, cuda:0, mps
            "gpu_ids": None,  # Specific GPU IDs to use [0, 1, 2]
            "num_workers": 4,
            "pin_memory": True,
            "memory_fraction": 0.95,  # GPU memory fraction to use
            "distributed": {  # Multi-GPU training
                "enabled": "auto",  # auto-enable for multiple GPUs
                "backend": "nccl",  # nccl for GPU, gloo for CPU
                "find_unused_parameters": False,
                "static_graph": False
            }
        },
        description="Hardware and device settings"
    )
    
    # Performance optimization configuration  
    performance: Dict[str, Any] = Field(
        default_factory=lambda: {
            "mixed_precision": {
                "enabled": "auto",  # auto-enable for compatible GPUs
                "init_scale": 65536.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000
            },
            "compile": {
                "enabled": "auto",  # PyTorch 2.0+ model compilation
                "mode": "default"  # default, reduce-overhead, max-autotune
            },
            "memory_optimization": True,
            "profiling": {
                "enabled": False,
                "memory": True,
                "compute": True
            }
        },
        description="Performance optimization settings"
    )
    
    # Hyperparameter optimization (Optuna integration)
    optimization: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hyperparameter optimization settings with Optuna"
    )
    
    @validator('model_type')
    def validate_model_type(cls, v):
        """Validate that model_type is a valid import path."""
        if not isinstance(v, str) or '.' not in v:
            raise ValueError("model_type must be a valid import path (e.g., 'sklearn.ensemble.RandomForestClassifier')")
        return v
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "DLTConfig":
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls(**data)
    
    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "DLTConfig":
        """Create from Hydra configuration."""
        return cls(**OmegaConf.to_container(cfg, resolve=True))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def get_model_class(self) -> Type:
        """Dynamically import and return the model class."""
        module_path, class_name = self.model_type.rsplit('.', 1)
        
        # Handle special cases for common frameworks
        if module_path.startswith('sklearn'):
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        elif module_path.startswith('torch'):
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        elif module_path.startswith('tensorflow') or module_path.startswith('tf'):
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        else:
            # Generic import
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
    
    def is_sklearn_model(self) -> bool:
        """Check if this is a scikit-learn model."""
        return self.model_type.startswith('sklearn')
    
    def is_torch_model(self) -> bool:
        """Check if this is a PyTorch model."""
        return self.model_type.startswith('torch') or 'torch' in self.model_type.lower()
    
    def is_tensorflow_model(self) -> bool:
        """Check if this is a TensorFlow model."""
        return self.model_type.startswith('tensorflow') or self.model_type.startswith('tf')
    
    def get_framework(self) -> str:
        """Detect the ML framework being used."""
        if self.is_sklearn_model():
            return 'sklearn'
        elif self.is_torch_model():
            return 'torch'
        elif self.is_tensorflow_model():
            return 'tensorflow'
        else:
            return 'custom'
    
    def model_dump(self) -> Dict[str, Any]:
        """Override model_dump to handle Field objects properly."""
        # Use pydantic's built-in serialization with proper field handling
        return super().model_dump(exclude_defaults=False, exclude_none=False)