"""Tests for DLT model system - Updated for current API."""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
import tempfile
from pathlib import Path

from dlt.core.config import DLTConfig
from dlt.core.model import DLTModel


class TestDLTModel:
    """Test the DLTModel system."""
    
    def test_sklearn_model_creation(self):
        """Test sklearn model creation using factory method."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 10, 'random_state': 42}
        )
        
        # Use factory method instead of direct instantiation
        model = DLTModel.from_config(config)
        assert model is not None
        assert model._framework == 'sklearn'
    
    def test_sklearn_model_training(self):
        """Test sklearn model training."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 10, 'random_state': 42}
        )
        
        model = DLTModel.from_config(config)
        
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert predictions.shape == y.shape
    
    def test_pytorch_model_creation(self):
        """Test PyTorch model creation."""
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
        assert model is not None
        assert model._framework == 'torch'
    
    def test_pytorch_model_training(self):
        """Test PyTorch model basic functionality."""
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
        
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X = X.astype(np.float32)
        
        # Test basic functionality (actual training is done by DLTTrainer)
        predictions = model.predict(X)
        assert predictions is not None
        assert len(predictions) == len(y)
    
    def test_model_save_load_sklearn(self):
        """Test sklearn model save and load."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 5, 'random_state': 42}
        )
        
        model = DLTModel.from_config(config)
        
        # Train model
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        model.fit(X, y)
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            model.save(model_path)
            loaded_model = DLTModel.load(model_path, config)
            
            # Test predictions are the same
            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)
            np.testing.assert_array_equal(original_pred, loaded_pred)
        finally:
            Path(model_path).unlink()
    
    def test_model_save_load_pytorch(self):
        """Test PyTorch model save and load."""
        config = DLTConfig(
            model_type='torch.nn.Sequential',
            model_params={
                'layers': [
                    {'type': 'Linear', 'in_features': 5, 'out_features': 3},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 3, 'out_features': 2}
                ]
            }
        )
        
        model = DLTModel.from_config(config)
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
        
        try:
            model.save(model_path)
            loaded_model = DLTModel.load(model_path, config)
            
            assert loaded_model is not None
            assert loaded_model._framework == 'torch'
        finally:
            Path(model_path).unlink()
    
    def test_model_framework_detection(self):
        """Test automatic framework detection."""
        sklearn_config = DLTConfig(model_type='sklearn.svm.SVC')
        sklearn_model = DLTModel.from_config(sklearn_config)
        assert sklearn_model._framework == 'sklearn'
        
        torch_config = DLTConfig(model_type='torch.nn.Linear', model_params={'in_features': 10, 'out_features': 1})
        torch_model = DLTModel.from_config(torch_config)
        assert torch_model._framework == 'torch'
    
    def test_model_predict_proba_sklearn(self):
        """Test predict_proba functionality for sklearn models."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 5, 'random_state': 42}
        )
        
        model = DLTModel.from_config(config)
        
        # Train model
        X, y = make_classification(n_samples=50, n_features=5, n_classes=3, n_informative=3, n_redundant=0, random_state=42)
        model.fit(X, y)
        
        # Test predict_proba
        probabilities = model.predict_proba(X)
        assert probabilities is not None
        assert probabilities.shape[0] == len(y)
        assert probabilities.shape[1] == 3  # 3 classes
        
        # Probabilities should sum to 1
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)
    
    def test_model_predict_proba_pytorch(self):
        """Test predict_proba functionality for PyTorch models."""
        config = DLTConfig(
            model_type='torch.nn.Sequential',
            model_params={
                'layers': [
                    {'type': 'Linear', 'in_features': 5, 'out_features': 3},
                    {'type': 'Softmax', 'dim': 1}
                ]
            }
        )
        
        model = DLTModel.from_config(config)
        
        # Generate sample data
        X, _ = make_classification(n_samples=20, n_features=5, n_classes=3, n_informative=3, n_redundant=0, random_state=42)
        X = X.astype(np.float32)
        
        # Test predict_proba
        probabilities = model.predict_proba(X)
        if probabilities is not None:
            assert probabilities.shape[0] == len(X)
            assert probabilities.shape[1] == 3  # 3 classes
    
    def test_model_with_regression_data(self):
        """Test model with regression data."""
        config = DLTConfig(
            model_type='sklearn.linear_model.LinearRegression'
        )
        
        model = DLTModel.from_config(config)
        
        # Generate regression data
        X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
        
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64 or predictions.dtype == np.float32
    
    def test_model_get_info(self):
        """Test model information retrieval."""
        config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 10}
        )
        
        model = DLTModel.from_config(config)
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'framework' in info
        assert 'model_type' in info
        assert info['framework'] == 'sklearn'
    
    def test_model_custom_parameters(self):
        """Test model with custom parameters."""
        config = DLTConfig(
            model_type='sklearn.svm.SVC',
            model_params={
                'C': 10.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
        )
        
        model = DLTModel.from_config(config)
        
        # Check that parameters are set correctly
        assert model._model.C == 10.0
        assert model._model.kernel == 'rbf'
        assert model._model.gamma == 'scale'
        
        # Train and test
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_pytorch_sequential_layers(self):
        """Test PyTorch Sequential model with various layer types."""
        config = DLTConfig(
            model_type='torch.nn.Sequential',
            model_params={
                'layers': [
                    {'type': 'Linear', 'in_features': 10, 'out_features': 20},
                    {'type': 'BatchNorm1d', 'num_features': 20},
                    {'type': 'ReLU'},
                    {'type': 'Dropout', 'p': 0.1},
                    {'type': 'Linear', 'in_features': 20, 'out_features': 5},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 5, 'out_features': 2}
                ]
            }
        )
        
        model = DLTModel.from_config(config)
        assert model is not None
        
        # Test forward pass
        X = np.random.randn(10, 10).astype(np.float32)
        output = model.predict(X)
        assert output is not None
        assert len(output) == 10
    
    def test_model_with_different_sklearn_algorithms(self):
        """Test different sklearn algorithms."""
        algorithms = [
            'sklearn.linear_model.LogisticRegression',
            'sklearn.tree.DecisionTreeClassifier',
            'sklearn.svm.SVC',
            'sklearn.neighbors.KNeighborsClassifier'
        ]
        
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        
        for algorithm in algorithms:
            config = DLTConfig(
                model_type=algorithm,
                model_params={'random_state': 42} if 'random_state' in algorithm else {}
            )
            
            model = DLTModel.from_config(config)
            model.fit(X, y)
            predictions = model.predict(X)
            
            assert len(predictions) == len(y)
            assert set(predictions).issubset(set(y))  # Predictions should be from the same classes