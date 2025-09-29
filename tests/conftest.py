"""Test configuration and shared fixtures for DLT framework."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to Python path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def classification_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=3,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'features': X.shape[1],
        'classes': len(np.unique(y))
    }


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'features': X.shape[1]
    }


@pytest.fixture
def basic_sklearn_config():
    """Basic sklearn configuration for testing."""
    return {
        'model_type': 'sklearn.ensemble.RandomForestClassifier',
        'model_params': {
            'n_estimators': 10,  # Small for fast testing
            'random_state': 42
        },
        'experiment': {
            'name': 'test_experiment'
        }
    }


@pytest.fixture
def basic_pytorch_config():
    """Basic PyTorch configuration for testing."""
    return {
        'model_type': 'torch.nn.Sequential',
        'model_params': {
            'layers': [
                {'type': 'Linear', 'in_features': 10, 'out_features': 32},
                {'type': 'ReLU'},
                {'type': 'Linear', 'in_features': 32, 'out_features': 3}
            ]
        },
        'training': {
            'optimizer': {'type': 'adam', 'lr': 0.01},
            'epochs': 3,  # Small for fast testing
            'batch_size': 16
        },
        'experiment': {
            'name': 'test_pytorch_experiment'
        }
    }


@pytest.fixture
def advanced_config():
    """Advanced configuration with optional features."""
    return {
        'model_type': 'torch.nn.Sequential',
        'model_params': {
            'layers': [
                {'type': 'Linear', 'in_features': 10, 'out_features': 64},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': 0.2},
                {'type': 'Linear', 'in_features': 64, 'out_features': 3}
            ]
        },
        'training': {
            'optimizer': {'type': 'adamw', 'lr': 0.001},
            'epochs': 5,
            'batch_size': 32,
            'loss': {
                'type': 'classification',
                'focal': True,
                'weights': [1.0, 2.0, 1.5]
            }
        },
        'performance': {
            'mixed_precision': {'enabled': 'auto'},
            'compile': {'enabled': False},  # Disable for testing
            'memory_optimization': True
        },
        'hardware': {
            'device': 'auto'
        },
        'experiment': {
            'name': 'test_advanced_experiment',
            'tags': ['test', 'advanced']
        }
    }


@pytest.fixture
def yaml_config_file(temp_dir, basic_sklearn_config):
    """Create a YAML config file for testing."""
    import yaml
    
    config_file = temp_dir / 'test_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(basic_sklearn_config, f)
    
    return config_file


@pytest.fixture
def json_config_file(temp_dir, basic_pytorch_config):
    """Create a JSON config file for testing."""
    import json
    
    config_file = temp_dir / 'test_config.json'
    with open(config_file, 'w') as f:
        json.dump(basic_pytorch_config, f, indent=2)
    
    return config_file


# Skip tests if optional dependencies are not available
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "torch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "tensorflow: mark test as requiring TensorFlow"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle optional dependencies."""
    torch_available = True
    tensorflow_available = True
    
    try:
        import torch
    except ImportError:
        torch_available = False
    
    try:
        import tensorflow as tf
    except ImportError:
        tensorflow_available = False
    
    skip_torch = pytest.mark.skip(reason="PyTorch not available")
    skip_tensorflow = pytest.mark.skip(reason="TensorFlow not available")
    
    for item in items:
        if "torch" in item.keywords and not torch_available:
            item.add_marker(skip_torch)
        if "tensorflow" in item.keywords and not tensorflow_available:
            item.add_marker(skip_tensorflow)