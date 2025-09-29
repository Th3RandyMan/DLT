"""
DLT High-Level Pipeline - Simple API for Common ML/DL Tasks

Provides convenience functions for:
- Quick model training with minimal code
- Automated hyperparameter tuning
- Model evaluation and comparison
- Easy experimentation

Examples:
    # Train any model in 2 lines
    from dlt import train, DLTConfig
    
    results = train(
        config=DLTConfig(model_type="sklearn.ensemble.RandomForestClassifier"),
        train_data=(X_train, y_train),
        val_data=(X_val, y_val)
    )
    
    # Automatic hyperparameter tuning
    best_model = tune(
        config_space={
            'model_params.n_estimators': (50, 500),
            'model_params.max_depth': (3, 20)
        },
        train_data=(X_train, y_train)
    )
"""

from typing import Any, Dict, Optional, Union, Tuple, Callable, List
from pathlib import Path
import time

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def train(
    config: Union["DLTConfig", Dict[str, Any], str, Path],
    train_data: Tuple[Any, Any],
    val_data: Optional[Tuple[Any, Any]] = None,
    test_data: Optional[Tuple[Any, Any]] = None,
    save_model: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Universal training function for any ML/DL model.
    
    Args:
        config: Model configuration (DLTConfig, dict, or path to config file)
        train_data: Tuple of (X, y) training data
        val_data: Optional validation data
        test_data: Optional test data for final evaluation
        save_model: Path to save trained model
        verbose: Whether to show training progress
        
    Returns:
        Dictionary containing training results, metrics, and model
        
    Examples:
        # Train a Random Forest
        results = train(
            config={'model_type': 'sklearn.ensemble.RandomForestClassifier'},
            train_data=(X_train, y_train),
            val_data=(X_val, y_val)
        )
        
        # Train a PyTorch CNN
        results = train(
            config={
                'model_type': 'torch.nn.Sequential',
                'model_params': {
                    'layers': [
                        {'type': 'Conv2d', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3},
                        {'type': 'ReLU'},
                        {'type': 'AdaptiveAvgPool2d', 'output_size': (1, 1)},
                        {'type': 'Flatten'},
                        {'type': 'Linear', 'in_features': 64, 'out_features': 10}
                    ]
                }
            },
            train_data=(X_train, y_train)
        )
    """
    from .config import DLTConfig
    from .model import DLTModel
    from .trainer import DLTTrainer
    
    # Handle different config formats
    if isinstance(config, (str, Path)):
        config = DLTConfig.from_file(config)
    elif isinstance(config, dict):
        config = DLTConfig(**config)
    elif not isinstance(config, DLTConfig):
        raise ValueError("Config must be DLTConfig, dict, or path to config file")
    
    # Create model and trainer
    model = DLTModel.from_config(config)
    trainer = DLTTrainer(config, model)
    
    # Training
    start_time = time.time()
    if verbose:
        print(f"Starting training with {config.model_type}")
        print(f"Framework: {config.get_framework()}")
        print(f"Device: {trainer.device}")
    
    training_results = trainer.train(train_data, val_data, verbose=verbose)
    training_time = time.time() - start_time
    
    # Results dictionary
    results = {
        'model': model,
        'trainer': trainer,
        'config': config,
        'training_time': training_time,
        'training_results': training_results,
        'history': training_results  # For backwards compatibility
    }
    
    # Test evaluation
    if test_data is not None:
        if verbose:
            print("Evaluating on test data...")
        test_results = trainer.evaluate(test_data)
        results['test_results'] = test_results
        
        if verbose:
            for metric, value in test_results.items():
                print(f"Test {metric}: {value:.4f}")
    
    # Save model
    if save_model is not None:
        if verbose:
            print(f"Saving model to {save_model}")
        model.save(save_model)
        results['saved_path'] = save_model
    
    if verbose:
        print(f"Training completed in {training_time:.2f} seconds")
    
    return results


def evaluate(
    model: Union["DLTModel", str, Path],
    test_data: Tuple[Any, Any],
    metrics: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model or path to saved model
        test_data: Test data as (X, y) tuple
        metrics: List of metrics to compute
        verbose: Whether to print results
        
    Returns:
        Dictionary of evaluation metrics
    """
    from .model import DLTModel
    
    # Load model if path provided
    if isinstance(model, (str, Path)):
        model = DLTModel.load(model)
    
    # Create basic trainer for evaluation
    from .trainer import DLTTrainer
    trainer = DLTTrainer(model.config, model)
    
    results = trainer.evaluate(test_data)
    
    if verbose:
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
    
    return results


def predict(
    model: Union["DLTModel", str, Path],
    data: Any,
    batch_size: Optional[int] = None,
    verbose: bool = False
) -> Any:
    """
    Make predictions with a trained model.
    
    Args:
        model: Trained model or path to saved model
        data: Input data for prediction
        batch_size: Batch size for prediction (for large datasets)
        verbose: Whether to show progress
        
    Returns:
        Model predictions
    """
    from .model import DLTModel
    
    # Load model if path provided
    if isinstance(model, (str, Path)):
        model = DLTModel.load(model)
    
    return model.predict(data)


def tune(
    base_config: Union[Dict[str, Any], "DLTConfig"],
    config_space: Dict[str, Any],
    train_data: Tuple[Any, Any],
    val_data: Optional[Tuple[Any, Any]] = None,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    metric: str = 'val_loss',
    direction: str = 'minimize',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Automated hyperparameter optimization using Optuna.
    
    Args:
        base_config: Base configuration to optimize
        config_space: Dictionary defining hyperparameter search space
        train_data: Training data
        val_data: Validation data
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
        metric: Metric to optimize
        direction: 'minimize' or 'maximize'
        verbose: Whether to show optimization progress
        
    Returns:
        Dictionary with best parameters, model, and optimization results
        
    Examples:
        best = tune(
            base_config={'model_type': 'sklearn.ensemble.RandomForestClassifier'},
            config_space={
                'model_params.n_estimators': (50, 500),
                'model_params.max_depth': (3, 20),
                'model_params.min_samples_split': (2, 20)
            },
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            n_trials=50
        )
    """
    
    if not HAS_OPTUNA:
        raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")
    
    from .config import DLTConfig
    
    # Convert to DLTConfig if needed
    if isinstance(base_config, dict):
        base_config = DLTConfig(**base_config)
    
    def objective(trial):
        # Create trial config by sampling from config_space
        trial_config = base_config.model_copy()
        
        for param_path, space_def in config_space.items():
            if isinstance(space_def, tuple) and len(space_def) == 2:
                # Range specification
                low, high = space_def
                if isinstance(low, int) and isinstance(high, int):
                    value = trial.suggest_int(param_path, low, high)
                else:
                    value = trial.suggest_float(param_path, low, high)
            elif isinstance(space_def, list):
                # Categorical choice
                value = trial.suggest_categorical(param_path, space_def)
            else:
                raise ValueError(f"Unsupported config space format for {param_path}: {space_def}")
            
            # Set the parameter using dot notation
            _set_nested_param(trial_config, param_path, value)
        
        # Train model with trial config
        try:
            results = train(
                config=trial_config,
                train_data=train_data,
                val_data=val_data,
                verbose=False
            )
            
            # Extract metric value
            if metric in results.get('test_results', {}):
                return results['test_results'][metric]
            elif metric in results.get('training_results', {}).get('history', {}):
                history = results['training_results']['history'][metric]
                return history[-1] if history else float('inf')
            else:
                # Default to validation loss
                val_loss = results['training_results'].get('history', {}).get('val_loss', [float('inf')])
                return val_loss[-1] if val_loss else float('inf')
                
        except Exception as e:
            if verbose:
                print(f"Trial failed: {e}")
            return float('inf') if direction == 'minimize' else float('-inf')
    
    # Create study
    study = optuna.create_study(direction=direction)
    
    if verbose:
        print(f"Starting hyperparameter optimization with {n_trials} trials...")
        print(f"Optimizing {metric} ({direction})")
    
    # Optimize
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout,
        show_progress_bar=verbose
    )
    
    # Train best model
    best_config = base_config.model_copy()
    for param_path, value in study.best_params.items():
        _set_nested_param(best_config, param_path, value)
    
    if verbose:
        print(f"Best trial value: {study.best_value}")
        print("Best parameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
    
    # Train final model with best parameters
    best_results = train(
        config=best_config,
        train_data=train_data,
        val_data=val_data,
        verbose=verbose
    )
    
    return {
        'best_model': best_results['model'],
        'best_config': best_config,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study,
        'training_results': best_results
    }


def _set_nested_param(config: "DLTConfig", param_path: str, value: Any) -> None:
    """Set a nested parameter using dot notation."""
    parts = param_path.split('.')
    current = config
    
    # Navigate to the parent of the target parameter
    for part in parts[:-1]:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        else:
            raise ValueError(f"Cannot set parameter {param_path}")
    
    # Set the final parameter
    final_key = parts[-1]
    if isinstance(current, dict):
        current[final_key] = value
    else:
        setattr(current, final_key, value)


# Convenience aliases for common tasks
quick_train = train  # Alias for backwards compatibility
auto_tune = tune     # Alias for hyperparameter tuning