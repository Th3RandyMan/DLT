# DLT - Deep Learning Toolkit

**A modern, extensible framework for machine learning and deep learning projects**

DLT provides a unified interface for training any ML/DL model - from simple scikit-learn models to complex neural networks - with automatic optimization, multi-GPU support, and advanced features like mixed precision training.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Th3RandyMan/DLT.git
cd DLT

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Train Your First Model (2 lines!)

```python
from dlt import train, DLTConfig

# Train a Random Forest
results = train(
    config={'model_type': 'sklearn.ensemble.RandomForestClassifier'},
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)
```

That's it! DLT automatically handles everything else.

## 🎯 Why DLT?

- **Universal**: Works with any ML/DL framework (scikit-learn, PyTorch, TensorFlow, Transformers)
- **Simple**: Minimal learning curve - only 3 core classes to understand
- **Powerful**: Advanced features like multi-GPU training, mixed precision, smart loss functions
- **Flexible**: From basic linear regression to cutting-edge transformers
- **Efficient**: Automatic performance optimizations for your hardware

## 📚 Core Concepts

DLT follows a simple 3-class architecture inspired by Hugging Face Transformers:

1. **`DLTConfig`** - Configuration management with validation
2. **`DLTModel`** - Universal model wrapper for any framework  
3. **`DLTTrainer`** - Intelligent training with automatic optimizations

## 🔥 Examples

### 1. Basic Machine Learning

```python
from dlt import train

# Random Forest for classification
results = train(
    config={
        'model_type': 'sklearn.ensemble.RandomForestClassifier',
        'model_params': {'n_estimators': 100, 'random_state': 42}
    },
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)

print(f"Accuracy: {results['test_results']['accuracy']:.4f}")
```

### 2. Deep Learning with PyTorch

```python
# Simple Neural Network
results = train(
    config={
        'model_type': 'torch.nn.Sequential',
        'model_params': {
            'layers': [
                {'type': 'Linear', 'in_features': 784, 'out_features': 128},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': 0.2},
                {'type': 'Linear', 'in_features': 128, 'out_features': 10}
            ]
        },
        'training': {
            'optimizer': {'type': 'adam', 'lr': 0.001},
            'epochs': 50,
            'batch_size': 64
        }
    },
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)
```

### 3. Convolutional Neural Network

```python
# CNN for image classification
cnn_config = {
    'model_type': 'torch.nn.Sequential',
    'model_params': {
        'layers': [
            {'type': 'Conv2d', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3},
            {'type': 'ReLU'},
            {'type': 'MaxPool2d', 'kernel_size': 2},
            {'type': 'Conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3},
            {'type': 'ReLU'},
            {'type': 'AdaptiveAvgPool2d', 'output_size': (1, 1)},
            {'type': 'Flatten'},
            {'type': 'Linear', 'in_features': 64, 'out_features': 10}
        ]
    },
    'training': {
        'optimizer': {'type': 'adamw', 'lr': 0.001},
        'epochs': 100,
        'batch_size': 32
    },
    # Enable performance optimizations
    'performance': {
        'mixed_precision': {'enabled': True},
        'compile': {'enabled': True}
    }
}

results = train(config=cnn_config, train_data=train_data, val_data=val_data)
```

### 4. Multi-GPU Training

```python
# Automatically use all available GPUs
multi_gpu_config = {
    'model_type': 'torch.nn.Sequential',
    'model_params': {...},
    'hardware': {
        'device': 'auto',  # Uses all GPUs
        'gpu_ids': [0, 1, 2, 3]  # Or specify which GPUs
    },
    'performance': {
        'mixed_precision': {'enabled': True}
    }
}

# DLT automatically sets up DistributedDataParallel
results = train(config=multi_gpu_config, train_data=train_data)
```

### 5. Handling Imbalanced Data

```python
# Smart loss functions for imbalanced datasets
imbalanced_config = {
    'model_type': 'torch.nn.Sequential',
    'model_params': {...},
    'training': {
        'loss': {
            'type': 'classification',
            'focal': True,  # Focal loss for hard examples
            'weights': [1.0, 5.0, 2.0],  # Class weights
            'adaptive_weighting': True  # Dynamic rebalancing
        }
    }
}

results = train(config=imbalanced_config, train_data=train_data)
```

## 🛠 Configuration Files

### Using YAML Configs

Create `my_config.yaml`:

```yaml
# Basic CNN configuration
model_type: "torch.nn.Sequential"
model_params:
  layers:
    - type: "Conv2d"
      in_channels: 3
      out_channels: 64
      kernel_size: 3
    - type: "ReLU"
    - type: "AdaptiveAvgPool2d"
      output_size: [1, 1]
    - type: "Flatten"
    - type: "Linear"
      in_features: 64
      out_features: 10

training:
  optimizer:
    type: "adamw"
    lr: 0.001
  epochs: 50
  batch_size: 32

# Optional performance optimizations
performance:
  mixed_precision:
    enabled: "auto"  # Auto-enable on compatible GPUs
  compile:
    enabled: "auto"  # Auto-enable on PyTorch 2.0+

experiment:
  name: "my_cnn_experiment"
  tags: ["cnn", "computer_vision"]
```

Then train:

```python
from dlt import train
results = train(config="my_config.yaml", train_data=data, val_data=val_data)
```

## 🎛 Advanced Features

### Hyperparameter Optimization

```python
from dlt import tune

# Automatic hyperparameter search with Optuna
best_results = tune(
    base_config={'model_type': 'sklearn.ensemble.RandomForestClassifier'},
    config_space={
        'model_params.n_estimators': (50, 500),
        'model_params.max_depth': (3, 20),
        'model_params.min_samples_split': (2, 20)
    },
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    n_trials=100
)

print(f"Best accuracy: {best_results['best_value']:.4f}")
print("Best parameters:", best_results['best_params'])
```

### Performance Profiling

```python
# Enable detailed performance profiling
config = {
    'model_type': 'torch.nn.Sequential',
    'model_params': {...},
    'performance': {
        'profiling': {
            'enabled': True,
            'memory': True,
            'compute': True
        }
    }
}

results = train(config=config, train_data=data)

# Get performance insights
profile = results['trainer'].profiler.get_performance_summary()
bottlenecks = results['trainer'].profiler.identify_bottlenecks()
```

## 📁 File Organization

```
your_project/
├── configs/
│   ├── base_config.yaml          # Base configuration
│   ├── cnn_config.yaml          # CNN-specific config
│   └── hyperopt_config.yaml     # Hyperparameter optimization
├── data/
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Preprocessed data
│   └── external/                # External datasets
├── models/                      # Saved models
├── notebooks/                   # Jupyter notebooks
├── scripts/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
└── results/                     # Training results and logs
```

## 🚀 Getting Started Guide

### Step 1: Install DLT

```bash
# From source (recommended for development)
git clone https://github.com/Th3RandyMan/DLT.git
cd DLT
uv sync  # or pip install -e .
```

### Step 2: Prepare Your Data

```python
import numpy as np
from sklearn.datasets import make_classification

# Example: Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)

# Split into train/val/test
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_data = (X_train, y_train)
val_data = (X_val, y_val)
test_data = (X_test, y_test)
```

### Step 3: Choose Your Model

```python
from dlt import train

# Start simple - Random Forest
results = train(
    config={'model_type': 'sklearn.ensemble.RandomForestClassifier'},
    train_data=train_data,
    val_data=val_data,
    test_data=test_data
)

print(f"Random Forest Accuracy: {results['test_results']['accuracy']:.4f}")
```

### Step 4: Try Deep Learning

```python
# Simple neural network
nn_config = {
    'model_type': 'torch.nn.Sequential',
    'model_params': {
        'layers': [
            {'type': 'Linear', 'in_features': 20, 'out_features': 64},
            {'type': 'ReLU'},
            {'type': 'Dropout', 'p': 0.2},
            {'type': 'Linear', 'in_features': 64, 'out_features': 32},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in_features': 32, 'out_features': 3}
        ]
    },
    'training': {
        'optimizer': {'type': 'adam', 'lr': 0.001},
        'epochs': 50,
        'batch_size': 32,
        'early_stopping': {'patience': 10}
    }
}

nn_results = train(config=nn_config, train_data=train_data, val_data=val_data, test_data=test_data)
print(f"Neural Network Accuracy: {nn_results['test_results']['accuracy']:.4f}")
```

### Step 5: Optimize Performance

```python
# Enable all performance optimizations
optimized_config = {
    **nn_config,  # Use the same neural network
    'hardware': {
        'device': 'auto',  # Auto-detect best device (GPU if available)
    },
    'performance': {
        'mixed_precision': {'enabled': 'auto'},  # Faster training on modern GPUs
        'compile': {'enabled': 'auto'},          # PyTorch 2.0+ compilation
        'memory_optimization': True
    }
}

optimized_results = train(config=optimized_config, train_data=train_data, val_data=val_data)
```

### Step 6: Hyperparameter Tuning

```python
from dlt import tune

# Find the best hyperparameters
best_model = tune(
    base_config={'model_type': 'sklearn.ensemble.RandomForestClassifier'},
    config_space={
        'model_params.n_estimators': (10, 200),
        'model_params.max_depth': (3, 15),
        'model_params.min_samples_split': (2, 10)
    },
    train_data=train_data,
    val_data=val_data,
    n_trials=50
)

# Test the best model
from dlt import evaluate
final_results = evaluate(best_model['best_model'], test_data)
print(f"Optimized Model Accuracy: {final_results['accuracy']:.4f}")
```

## 🔧 Command Line Interface

DLT also provides a CLI for quick experimentation:

```bash
# Create a configuration template
dlt create-config my_config.yaml --model sklearn.ensemble.RandomForestClassifier --template basic

# Train a model
dlt train my_config.yaml --train-data data/train.csv --val-data data/val.csv --epochs 50

# Hyperparameter optimization
dlt tune base_config.yaml search_space.yaml --trials 100 --output results/

# Get system info
dlt info
```

## 📊 Model Registry

DLT includes a model registry for managing different architectures:

```python
from dlt.models import ModelRegistry

# Register custom models
@ModelRegistry.register("my_custom_model")
class MyCustomModel:
    def __init__(self, input_dim, output_dim):
        # Your model implementation
        pass

# Use registered models
config = {
    'model_type': 'my_custom_model',
    'model_params': {'input_dim': 784, 'output_dim': 10}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Getting Help

- **Documentation**: Check the `docs/` folder for detailed guides
- **Examples**: See `examples/` for complete project examples
- **Issues**: Open an issue on GitHub for bugs or feature requests

## 🎉 What's Next?

1. **Try the examples** in the `examples/` directory
2. **Read the documentation** for advanced features
3. **Join our community** for discussions and support

Happy training! 🚀
Deep Learning Tools (DLT) for most projects
