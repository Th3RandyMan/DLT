"""
DLT - Deep Learning Toolkit

A modern, extensible framework for machine learning and deep learning projects.

Design Philosophy:
    - 3 Core Classes: Configuration, Model, Trainer (inspired by Transformers)
    - Hydra-powered hierarchical configuration  
    - Optional PyTorch Lightning integration
    - Rich CLI experience with intuitive commands
    - Built-in hyperparameter optimization with Optuna
    - Easy PyPI distribution and packaging

Usage:
    from dlt import DLTConfig, DLTModel, DLTTrainer
    
    # Or use high-level APIs
    from dlt import train, evaluate, predict
"""

__version__ = "0.1.0"
__author__ = "Randall Fowler"
__email__ = "rlfowler@ucdavis.edu"

# Core classes (Transformers-style minimal API)
from .dlt.core.config import DLTConfig
from .dlt.core.model import DLTModel  
from .dlt.core.trainer import DLTTrainer

# High-level convenience functions
from .dlt.core.pipeline import train, evaluate, predict, tune

# Model registry
from .dlt.models import ModelRegistry

__all__ = [
    # Core classes
    "DLTConfig", 
    "DLTModel", 
    "DLTTrainer",
    # High-level functions
    "train",
    "evaluate", 
    "predict",
    "tune",
    # Registry
    "ModelRegistry",
]