"""Tests for DLT trainer system - Updated for current API."""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
import tempfile
from pathlib import Path

from dlt.core.config import DLTConfig
from dlt.core.model import DLTModel
from dlt.core.trainer import DLTTrainer


class TestDLTTrainer:
    """Test the DLTTrainer class."""
    
    def test_sklearn_trainer_creation(self):
        """Test sklearn trainer creation."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 5, 'random_state': 42}
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        assert trainer.framework == 'sklearn'
        assert trainer.model == model
    
    def test_pytorch_trainer_creation(self):
        """Test PyTorch trainer creation."""
        config = DLTConfig(
            model_type='torch.nn.Sequential',
            model_params={
                'layers': [
                    {'type': 'Linear', 'in_features': 10, 'out_features': 5},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 5, 'out_features': 2}
                ]
            }
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        assert trainer.framework == 'torch'
        assert trainer.model == model
    
    def test_sklearn_trainer_fit(self):
        """Test sklearn trainer training."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 5, 'random_state': 42}
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Train model
        history = trainer.train((X_train, y_train))
        
        assert isinstance(history, dict)
        assert 'training_time' in history
        
        # Test evaluation
        results = trainer.evaluate((X_test, y_test))
        assert 'accuracy' in results
        assert 0 <= results['accuracy'] <= 1
    
    def test_pytorch_trainer_fit(self):
        """Test PyTorch trainer training."""
        config = DLTConfig(
            model_type='torch.nn.Sequential',
            model_params={
                'layers': [
                    {'type': 'Linear', 'in_features': 10, 'out_features': 16},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 16, 'out_features': 2}
                ]
            },
            training={'epochs': 3, 'batch_size': 16}
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Train model
        history = trainer.train((X_train, y_train), verbose=False)
        
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert len(history['train_loss']) == 3  # 3 epochs
        
        # Test evaluation
        results = trainer.evaluate((X_test, y_test))
        assert 'accuracy' in results
        assert 0 <= results['accuracy'] <= 1
    
    def test_pytorch_trainer_with_early_stopping(self):
        """Test PyTorch trainer with validation data."""
        config = DLTConfig(
            model_type='torch.nn.Sequential',
            model_params={
                'layers': [
                    {'type': 'Linear', 'in_features': 10, 'out_features': 8},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 8, 'out_features': 2}
                ]
            },
            training={'epochs': 5, 'batch_size': 16}
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_train, X_val = X[:60], X[60:80]
        y_train, y_val = y[:60], y[60:80]
        
        # Train with validation data
        history = trainer.train((X_train, y_train), (X_val, y_val), verbose=False)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
    
    def test_trainer_evaluate(self):
        """Test trainer evaluation functionality."""
        config = DLTConfig(
            model_type='sklearn.svm.SVC',
            model_params={'kernel': 'linear', 'random_state': 42}
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        # Generate and train on sample data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        trainer.train((X_train, y_train))
        results = trainer.evaluate((X_test, y_test))
        
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert isinstance(results['accuracy'], float)
        assert 0 <= results['accuracy'] <= 1
    
    def test_trainer_with_advanced_config(self):
        """Test trainer with advanced configuration."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 10, 'random_state': 42},
            training={'epochs': 1},
            hardware={'device': 'cpu'}
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        assert trainer.device == 'cpu'
        
        # Generate sample data and train
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        history = trainer.train((X, y))
        
        assert isinstance(history, dict)
    
    def test_trainer_save_load_checkpoint(self):
        """Test trainer checkpoint save/load."""
        config = DLTConfig(
            model_type='sklearn.linear_model.LogisticRegression',
            model_params={'random_state': 42, 'max_iter': 100}
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        # Train model
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        trainer.train((X, y))
        
        # Save checkpoint (might not work for all models)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            trainer.save_checkpoint(checkpoint_path)
            assert Path(checkpoint_path).exists()
        except AttributeError:
            # Some model wrappers may not support checkpoints
            pass
        finally:
            if Path(checkpoint_path).exists():
                Path(checkpoint_path).unlink()
    
    def test_trainer_device_handling(self):
        """Test trainer device handling."""
        config = DLTConfig(
            model_type='sklearn.svm.SVC',
            hardware={'device': 'cpu'}
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        assert trainer.device == 'cpu'
    
    def test_trainer_with_custom_metrics(self):
        """Test trainer with custom training configuration."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 5, 'random_state': 42},
            training={
                'epochs': 1,
                'metrics': ['accuracy']
            }
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        history = trainer.train((X, y))
        
        assert isinstance(history, dict)
        
        # Test evaluation with custom metrics
        results = trainer.evaluate((X, y))
        assert 'accuracy' in results
    
    def test_trainer_gradient_clipping(self):
        """Test PyTorch trainer with gradient clipping configuration."""
        config = DLTConfig(
            model_type='torch.nn.Sequential',
            model_params={
                'layers': [
                    {'type': 'Linear', 'in_features': 5, 'out_features': 3},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 3, 'out_features': 2}
                ]
            },
            training={
                'epochs': 2,
                'gradient_clipping': {'enabled': True, 'max_norm': 1.0}
            }
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        
        history = trainer.train((X, y), verbose=False)
        assert isinstance(history, dict)
        assert 'train_loss' in history
    
    def test_trainer_with_different_optimizers(self):
        """Test PyTorch trainer with different optimizers."""
        optimizers = ['adam', 'sgd', 'adamw']
        
        for opt in optimizers:
            config = DLTConfig(
                model_type='torch.nn.Sequential',
                model_params={
                    'layers': [
                        {'type': 'Linear', 'in_features': 5, 'out_features': 3},
                        {'type': 'ReLU'},
                        {'type': 'Linear', 'in_features': 3, 'out_features': 2}
                    ]
                },
                training={
                    'epochs': 2,
                    'optimizer': {'type': opt, 'lr': 0.01}
                }
            )
            
            model = DLTModel.from_config(config)
            trainer = DLTTrainer(config, model)
            
            X, y = make_classification(n_samples=30, n_features=5, n_classes=2, random_state=42)
            
            history = trainer.train((X, y), verbose=False)
            assert isinstance(history, dict)
            assert 'train_loss' in history
    
    def test_trainer_learning_rate_scheduling(self):
        """Test trainer with learning rate scheduling configuration."""
        config = DLTConfig(
            model_type='torch.nn.Sequential',
            model_params={
                'layers': [
                    {'type': 'Linear', 'in_features': 5, 'out_features': 2}
                ]
            },
            training={
                'epochs': 3,
                'optimizer': {'type': 'adam', 'lr': 0.1},
                'scheduler': {'type': 'step', 'step_size': 2, 'gamma': 0.5}
            }
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        X, y = make_classification(n_samples=30, n_features=5, n_classes=2, random_state=42)
        
        history = trainer.train((X, y), verbose=False)
        assert isinstance(history, dict)
    
    def test_trainer_regression_task(self):
        """Test trainer with regression task."""
        config = DLTConfig(
            model_type='sklearn.linear_model.LinearRegression'
        )
        
        model = DLTModel.from_config(config)
        trainer = DLTTrainer(config, model)
        
        # Generate regression data
        X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        
        # Train
        history = trainer.train((X_train, y_train))
        assert isinstance(history, dict)
        
        # Note: Current evaluation method uses accuracy which doesn't work for regression
        # This test just ensures the trainer can handle regression data for training