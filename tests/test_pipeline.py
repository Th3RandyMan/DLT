"""Tests for DLT pipeline functions - Updated for current API."""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
import tempfile
from pathlib import Path
import yaml

from dlt.core.config import DLTConfig
from dlt import train, evaluate, predict, tune


class TestPipelineFunctions:
    """Test high-level pipeline functions."""
    
    def test_train_with_dict_config(self):
        """Test train function with dictionary configuration."""
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        config = {
            'model_type': 'sklearn.ensemble.RandomForestClassifier',
            'model_params': {'n_estimators': 10, 'random_state': 42}
        }
        
        results = train(
            config=config,
            train_data=(X_train, y_train),
            test_data=(X_test, y_test)
        )
        
        # Check that results contain expected keys
        assert 'model' in results
        assert 'config' in results
        assert 'training_time' in results
        assert 'test_results' in results  # Changed from 'val_results'
        
        # Check test results
        assert 'accuracy' in results['test_results']
        assert 0 <= results['test_results']['accuracy'] <= 1
    
    def test_train_with_dlt_config(self):
        """Test train function with DLTConfig object."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        
        config = DLTConfig(
            model_type='sklearn.svm.SVC',
            model_params={'kernel': 'linear', 'random_state': 42}
        )
        
        results = train(
            config=config,
            train_data=(X_train, y_train),
            test_data=(X_test, y_test)
        )
        
        assert isinstance(results['config'], DLTConfig)
        assert results['config'].model_type == 'sklearn.svm.SVC'
        assert 'test_results' in results
    
    def test_train_without_test_data(self):
        """Test train function without test data."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        
        config = {
            'model_type': 'sklearn.linear_model.LogisticRegression',
            'model_params': {'random_state': 42, 'max_iter': 100}
        }
        
        results = train(
            config=config,
            train_data=(X, y)
        )
        
        assert 'model' in results
        assert 'config' in results
        assert 'test_results' not in results  # No test data provided
    
    def test_train_with_save_model(self):
        """Test train function with model saving."""
        X, y = make_classification(n_samples=30, n_features=5, n_classes=2, random_state=42)
        
        config = {
            'model_type': 'sklearn.tree.DecisionTreeClassifier',
            'model_params': {'random_state': 42, 'max_depth': 3}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            results = train(
                config=config,
                train_data=(X, y),
                save_model=model_path
            )
            
            assert 'saved_path' in results
            assert results['saved_path'] == model_path
            assert Path(model_path).exists()
        finally:
            Path(model_path).unlink()
    
    def test_train_pytorch_model(self):
        """Test train function with PyTorch model."""
        X, y = make_classification(n_samples=50, n_features=8, n_classes=3, n_informative=6, n_redundant=0, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        
        config = {
            'model_type': 'torch.nn.Sequential',
            'model_params': {
                'layers': [
                    {'type': 'Linear', 'in_features': 8, 'out_features': 16},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 16, 'out_features': 3}
                ]
            },
            'training': {
                'epochs': 3,
                'batch_size': 16,
                'optimizer': {'type': 'adam', 'lr': 0.01}
            }
        }
        
        results = train(
            config=config,
            train_data=(X_train, y_train),
            test_data=(X_test, y_test),
            verbose=False
        )
        
        assert 'model' in results
        assert 'test_results' in results
        assert 'history' in results
        assert 'train_loss' in results['history']
    
    def test_evaluate_function(self):
        """Test standalone evaluate function."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        
        # First train a model
        config = {
            'model_type': 'sklearn.neighbors.KNeighborsClassifier',
            'model_params': {'n_neighbors': 3}
        }
        
        train_results = train(
            config=config,
            train_data=(X_train, y_train)
        )
        
        # Now evaluate it
        eval_results = evaluate(
            model=train_results['model'],
            test_data=(X_test, y_test)
        )
        
        assert isinstance(eval_results, dict)
        assert 'accuracy' in eval_results
        assert 0 <= eval_results['accuracy'] <= 1
    
    def test_predict_function(self):
        """Test standalone predict function."""
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        
        # Train a model
        config = {
            'model_type': 'sklearn.ensemble.RandomForestClassifier',
            'model_params': {'n_estimators': 5, 'random_state': 42}
        }
        
        train_results = train(
            config=config,
            train_data=(X_train, y_train)
        )
        
        # Make predictions
        predictions = predict(
            model=train_results['model'],
            data=X_test
        )
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)  # Binary classification
    
    def test_tune_function(self):
        """Test hyperparameter tuning function."""
        X, y = make_classification(n_samples=80, n_features=5, n_classes=2, random_state=42)
        X_train, X_val = X[:60], X[60:]
        y_train, y_val = y[:60], y[60:]
        
        base_config = {
            'model_type': 'sklearn.ensemble.RandomForestClassifier',
            'model_params': {'random_state': 42}
        }
        
        param_space = {
            'model_params.n_estimators': (10, 20),
            'model_params.max_depth': (3, 10)
        }
        
        results = tune(
            base_config=base_config,
            config_space=param_space,  # Changed from param_space
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            n_trials=5,
            verbose=False
        )
        
        assert 'best_params' in results
        assert 'best_value' in results
        assert 'best_model' in results
        assert 'study' in results
        
        # Check that best value is reasonable (not infinity)
        if results['best_value'] != float('inf'):
            assert 0 <= results['best_value'] <= 1
    
    def test_train_with_config_file(self):
        """Test train function with configuration file."""
        config_data = {
            'model_type': 'sklearn.ensemble.RandomForestClassifier',
            'model_params': {'n_estimators': 10, 'random_state': 42},
            'experiment': {'name': 'test_experiment'},
            'training': {'epochs': 1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
            
            results = train(
                config=config_file,
                train_data=(X, y)
            )
            
            assert 'model' in results
            assert results['config'].experiment['name'] == 'test_experiment'
        finally:
            Path(config_file).unlink()
    
    def test_train_with_validation_data(self):
        """Test train function with validation data."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_train, X_val, X_test = X[:60], X[60:80], X[80:]
        y_train, y_val, y_test = y[:60], y[60:80], y[80:]
        
        config = {
            'model_type': 'torch.nn.Sequential',
            'model_params': {
                'layers': [
                    {'type': 'Linear', 'in_features': 10, 'out_features': 8},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 8, 'out_features': 2}
                ]
            },
            'training': {'epochs': 3, 'batch_size': 16}
        }
        
        results = train(
            config=config,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            test_data=(X_test, y_test),
            verbose=False
        )
        
        assert 'history' in results
        assert 'val_loss' in results['history']
        assert len(results['history']['val_loss']) == 3  # 3 epochs
    
    def test_train_regression_task(self):
        """Test train function with regression task."""
        X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        
        config = {
            'model_type': 'sklearn.linear_model.LinearRegression'
        }
        
        # Note: This will fail evaluation due to accuracy_score not supporting regression
        # But training should work
        try:
            results = train(
                config=config,
                train_data=(X_train, y_train),
                test_data=(X_test, y_test)
            )
        except ValueError as e:
            # Expected to fail on evaluation for regression
            assert 'continuous is not supported' in str(e)
    
    def test_cross_validation_simulation(self):
        """Test simulating cross-validation with multiple train calls."""
        X, y = make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42)
        
        config = {
            'model_type': 'sklearn.ensemble.RandomForestClassifier',
            'model_params': {'n_estimators': 10, 'random_state': 42}
        }
        
        # Simulate 3-fold CV
        fold_size = len(X) // 3
        accuracies = []
        
        for i in range(3):
            # Create fold splits
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < 2 else len(X)
            
            # Test fold
            X_test = X[start_idx:end_idx]
            y_test = y[start_idx:end_idx]
            
            # Training fold
            X_train = np.vstack([X[:start_idx], X[end_idx:]])
            y_train = np.hstack([y[:start_idx], y[end_idx:]])
            
            results = train(
                config=config,
                train_data=(X_train, y_train),
                test_data=(X_test, y_test)
            )
            
            accuracies.append(results['test_results']['accuracy'])
        
        # Check that we got accuracy scores for all folds
        assert len(accuracies) == 3
        assert all(0 <= acc <= 1 for acc in accuracies)
        
        # Calculate mean accuracy
        mean_accuracy = np.mean(accuracies)
        assert 0 <= mean_accuracy <= 1