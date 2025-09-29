"""
DLT Command Line Interface

Simple CLI for training models with DLT framework.
"""

import typer
from typing import Optional, List
from pathlib import Path
import yaml
import json

app = typer.Typer(help="DLT - Deep Learning Toolkit CLI")


@app.command()
def train(
    config: Path = typer.Argument(..., help="Path to configuration file"),
    data_train: Optional[Path] = typer.Option(None, "--train-data", help="Training data file"),
    data_val: Optional[Path] = typer.Option(None, "--val-data", help="Validation data file"),
    data_test: Optional[Path] = typer.Option(None, "--test-data", help="Test data file"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output directory for model and logs"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of training epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size"),
    lr: Optional[float] = typer.Option(None, "--learning-rate", help="Learning rate"),
    device: Optional[str] = typer.Option(None, "--device", help="Device to use (cpu, cuda, auto)"),
    mixed_precision: Optional[bool] = typer.Option(None, "--mixed-precision", help="Enable mixed precision"),
    compile_model: Optional[bool] = typer.Option(None, "--compile", help="Compile model (PyTorch 2.0+)"),
    profile: bool = typer.Option(False, "--profile", help="Enable performance profiling"),
    verbose: bool = typer.Option(True, "--verbose", help="Verbose output"),
):
    """Train a model using DLT framework."""
    
    # Import here to avoid issues with optional dependencies
    try:
        from dlt import train as dlt_train, DLTConfig
    except ImportError as e:
        typer.echo(f"Error importing DLT: {e}")
        typer.echo("Please ensure all dependencies are installed.")
        raise typer.Exit(1)
    
    # Load config
    try:
        config_obj = DLTConfig.from_file(config)
    except Exception as e:
        typer.echo(f"Error loading config from {config}: {e}")
        raise typer.Exit(1)
    
    # Override config with CLI arguments
    if epochs is not None:
        config_obj.training['epochs'] = epochs
    if batch_size is not None:
        config_obj.training['batch_size'] = batch_size
    if lr is not None:
        config_obj.training['optimizer']['lr'] = lr
    if device is not None:
        config_obj.hardware['device'] = device
    if mixed_precision is not None:
        config_obj.performance['mixed_precision']['enabled'] = mixed_precision
    if compile_model is not None:
        config_obj.performance['compile']['enabled'] = compile_model
    if profile:
        config_obj.performance['profiling']['enabled'] = True
    
    # Load data (placeholder - would need actual data loading logic)
    if not data_train:
        typer.echo("Training data not specified. Please provide --train-data")
        raise typer.Exit(1)
    
    # TODO: Implement actual data loading
    typer.echo(f"Loading training data from {data_train}")
    typer.echo("Note: Data loading not implemented in this demo")
    
    # For demo purposes, create dummy data
    import numpy as np
    X_train = np.random.randn(1000, 784)
    y_train = np.random.randint(0, 10, 1000)
    train_data = (X_train, y_train)
    
    val_data = None
    if data_val:
        X_val = np.random.randn(200, 784)
        y_val = np.random.randint(0, 10, 200)
        val_data = (X_val, y_val)
    
    # Train model
    if verbose:
        typer.echo(f"Starting training with config: {config}")
        typer.echo(f"Model type: {config_obj.model_type}")
        typer.echo(f"Framework: {config_obj.get_framework()}")
    
    try:
        results = dlt_train(
            config=config_obj,
            train_data=train_data,
            val_data=val_data,
            save_model=output / "model.pt" if output else None,
            verbose=verbose
        )
        
        if verbose:
            typer.echo("Training completed successfully!")
            if 'training_time' in results:
                typer.echo(f"Training time: {results['training_time']:.2f} seconds")
            if 'test_results' in results:
                for metric, value in results['test_results'].items():
                    typer.echo(f"Test {metric}: {value:.4f}")
                    
    except Exception as e:
        typer.echo(f"Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def tune(
    config: Path = typer.Argument(..., help="Base configuration file"),
    search_space: Path = typer.Argument(..., help="Hyperparameter search space file"),
    data_train: Optional[Path] = typer.Option(None, "--train-data", help="Training data file"),
    data_val: Optional[Path] = typer.Option(None, "--val-data", help="Validation data file"),
    n_trials: int = typer.Option(100, "--trials", help="Number of optimization trials"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Timeout in seconds"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output directory"),
    verbose: bool = typer.Option(True, "--verbose", help="Verbose output"),
):
    """Hyperparameter optimization using Optuna."""
    
    try:
        from dlt import tune as dlt_tune, DLTConfig
    except ImportError as e:
        typer.echo(f"Error importing DLT: {e}")
        raise typer.Exit(1)
    
    # Load configs
    try:
        base_config = DLTConfig.from_file(config)
        with open(search_space) as f:
            if search_space.suffix.lower() in ['.yaml', '.yml']:
                space_config = yaml.safe_load(f)
            else:
                space_config = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading configurations: {e}")
        raise typer.Exit(1)
    
    # TODO: Load actual data
    import numpy as np
    X_train = np.random.randn(1000, 784)
    y_train = np.random.randint(0, 10, 1000)
    train_data = (X_train, y_train)
    
    val_data = None
    if data_val:
        X_val = np.random.randn(200, 784)
        y_val = np.random.randint(0, 10, 200)
        val_data = (X_val, y_val)
    
    if verbose:
        typer.echo(f"Starting hyperparameter optimization with {n_trials} trials")
    
    try:
        best_results = dlt_tune(
            base_config=base_config,
            config_space=space_config,
            train_data=train_data,
            val_data=val_data,
            n_trials=n_trials,
            timeout=timeout,
            verbose=verbose
        )
        
        if verbose:
            typer.echo("Optimization completed!")
            typer.echo(f"Best value: {best_results['best_value']}")
            typer.echo("Best parameters:")
            for param, value in best_results['best_params'].items():
                typer.echo(f"  {param}: {value}")
        
        # Save best model
        if output:
            output.mkdir(parents=True, exist_ok=True)
            best_results['best_model'].save(output / "best_model.pt")
            best_results['best_config'].save(output / "best_config.yaml")
            typer.echo(f"Best model and config saved to {output}")
            
    except Exception as e:
        typer.echo(f"Optimization failed: {e}")
        raise typer.Exit(1)


@app.command()
def info():
    """Show information about DLT installation and capabilities."""
    
    typer.echo("DLT - Deep Learning Toolkit")
    typer.echo("==========================")
    
    # Check dependencies
    deps = {
        'PyTorch': 'torch',
        'TensorFlow': 'tensorflow',
        'Scikit-learn': 'sklearn',
        'Optuna': 'optuna',
        'Weights & Biases': 'wandb',
        'Hydra': 'hydra',
    }
    
    typer.echo("\nDependency Status:")
    for name, module in deps.items():
        try:
            __import__(module)
            typer.echo(f"  ✓ {name}")
        except ImportError:
            typer.echo(f"  ✗ {name} (not installed)")
    
    # Check GPU availability
    typer.echo("\nHardware:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            typer.echo(f"  ✓ CUDA available ({gpu_count} GPU(s))")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                typer.echo(f"    GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        else:
            typer.echo("  ✗ CUDA not available")
            
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            typer.echo("  ✓ Apple Metal Performance Shaders available")
    except ImportError:
        typer.echo("  ? PyTorch not available - cannot check GPU status")


@app.command()
def create_config(
    output: Path = typer.Argument(..., help="Output configuration file path"),
    model_type: str = typer.Option("sklearn.ensemble.RandomForestClassifier", "--model", help="Model type"),
    template: str = typer.Option("basic", "--template", help="Config template (basic, cnn, transformer)"),
):
    """Create a configuration file template."""
    
    templates = {
        'basic': {
            'model_type': model_type,
            'model_params': {},
            'training': {
                'validation_split': 0.2,
                'metrics': ['accuracy']
            },
            'hardware': {
                'device': 'auto'
            },
            'experiment': {
                'name': 'dlt_experiment',
                'tags': ['basic']
            }
        },
        'cnn': {
            'model_type': 'torch.nn.Sequential',
            'model_params': {
                'layers': [
                    {'type': 'Conv2d', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3},
                    {'type': 'ReLU'},
                    {'type': 'MaxPool2d', 'kernel_size': 2},
                    {'type': 'Flatten'},
                    {'type': 'Linear', 'in_features': 32*14*14, 'out_features': 10}
                ]
            },
            'training': {
                'optimizer': {'type': 'adam', 'lr': 0.001},
                'loss': {'type': 'cross_entropy'},
                'epochs': 50,
                'batch_size': 32
            },
            'hardware': {
                'device': 'auto'
            },
            'performance': {
                'mixed_precision': {'enabled': 'auto'},
                'compile': {'enabled': 'auto'}
            }
        }
    }
    
    if template not in templates:
        typer.echo(f"Unknown template: {template}")
        typer.echo(f"Available templates: {list(templates.keys())}")
        raise typer.Exit(1)
    
    config = templates[template]
    
    # Create output directory if needed
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save config
    if output.suffix.lower() in ['.yaml', '.yml']:
        with open(output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    else:
        with open(output, 'w') as f:
            json.dump(config, f, indent=2)
    
    typer.echo(f"Configuration template created: {output}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()