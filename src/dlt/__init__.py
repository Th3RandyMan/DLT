"""DLT - Deep Learning Toolkit

A modern, extensible framework for machine learning and deep learning projects.
"""

from .core.config import DLTConfig
from .core.model import DLTModel
from .core.trainer import DLTTrainer
from .core.pipeline import train, evaluate, predict, tune

__version__ = "0.1.0"
__all__ = ["DLTConfig", "DLTModel", "DLTTrainer", "train", "evaluate", "predict", "tune"]
