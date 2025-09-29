"""Integration tests for the complete DLT framework - Updated for current API."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import yaml
import json
from sklearn.datasets import make_classification, make_regression

from dlt import train, evaluate, predict, tune
from dlt.core.config import DLTConfig
from dlt.core.model import DLTModel
from dlt.core.trainer import DLTTrainer


@pytest.fixture
def classification_data():
    """Generate classification dataset for testing."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, 
        n_informative=8, n_redundant=2, random_state=42
    )
    
    # Split into train/val/test
    train_size = 120
    val_size = 40
    
    return {
        'train': (X[:train_size], y[:train_size]),
        'val': (X[train_size:train_size+val_size], y[train_size:train_size+val_size]),
        'test': (X[train_size+val_size:], y[train_size+val_size:])
    }

@pytest.fixture
def regression_data():
    """Generate regression dataset for testing."""
    X, y = make_regression(
        n_samples=200, n_features=10, noise=0.1, random_state=42
    )
    
    train_size = 120
    val_size = 40
    
    return {
        'train': (X[:train_size], y[:train_size]),
        'val': (X[train_size:train_size+val_size], y[train_size:train_size+val_size]),
        'test': (X[train_size+val_size:], y[train_size+val_size:])
    }

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestFrameworkIntegration:
    """Test complete framework integration scenarios."""
    
    def test_end_to_end_sklearn_workflow(self, classification_data, temp_dir):
        """Test complete sklearn workflow from config to prediction."""
        # 1. Create configuration
        config_dict = {
            'model_type': 'sklearn.ensemble.RandomForestClassifier',
            'model_params': {
                'n_estimators': 20,
                'max_depth': 5,
                'random_state': 42
            },
            'experiment': {
                'name': 'integration_test_sklearn',
                'description': 'End-to-end integration test',
                'tags': ['test', 'integration', 'sklearn']
            }
        }
        
        # 2. Save config to file
        config_file = temp_dir / 'integration_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # 3. Train model
        results = train(
            config=config_file,
            train_data=classification_data['train'],
            val_data=classification_data['val'],
            test_data=classification_data['test']
        )
        
        # 4. Verify training results
        assert 'model' in results
        assert 'config' in results
        assert 'test_results' in results
        assert 'accuracy' in results['test_results']
        assert 0 <= results['test_results']['accuracy'] <= 1
        
        # 5. Test standalone evaluation
        eval_results = evaluate(
            model=results['model'],
            test_data=classification_data['test']
        )
        assert 'accuracy' in eval_results
        
        # 6. Test standalone prediction
        X_test, _ = classification_data['test']
        predictions = predict(
            model=results['model'],
            data=X_test
        )
        assert len(predictions) == len(X_test)
        
        # 7. Save and load model
        model_path = temp_dir / 'model.pkl'
        results['model'].save(model_path)
        loaded_model = DLTModel.load(model_path, results['config'])
        
        # Test loaded model predictions match
        loaded_predictions = predict(model=loaded_model, data=X_test)
        np.testing.assert_array_equal(predictions, loaded_predictions)
    
    def test_end_to_end_pytorch_workflow(self, classification_data):
        """Test complete PyTorch workflow."""
        config = {
            'model_type': 'torch.nn.Sequential',
            'model_params': {
                'layers': [
                    {'type': 'Linear', 'in_features': 10, 'out_features': 16},
                    {'type': 'ReLU'},
                    {'type': 'Dropout', 'p': 0.2},
                    {'type': 'Linear', 'in_features': 16, 'out_features': 8},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 8, 'out_features': 3}
                ]
            },
            'training': {
                'epochs': 5,
                'batch_size': 32,
                'optimizer': {'type': 'adam', 'lr': 0.01}
            },
            'experiment': {
                'name': 'integration_test_pytorch'
            }
        }
        
        results = train(
            config=config,
            train_data=classification_data['train'],
            val_data=classification_data['val'],
            test_data=classification_data['test'],
            verbose=False
        )
        
        # Verify PyTorch training results
        assert 'model' in results
        assert 'history' in results
        assert 'train_loss' in results['history']
        assert 'val_loss' in results['history']
        assert len(results['history']['train_loss']) == 5  # 5 epochs
        
        # Test evaluation
        eval_results = evaluate(
            model=results['model'],
            test_data=classification_data['test']
        )
        assert 'accuracy' in eval_results
        
        # Test prediction
        X_test, _ = classification_data['test']
        predictions = predict(model=results['model'], data=X_test)
        assert len(predictions) == len(X_test)
    
    def test_hyperparameter_optimization_integration(self, classification_data):
        """Test hyperparameter optimization integration."""
        base_config = {
            'model_type': 'sklearn.svm.SVC',
            'model_params': {'random_state': 42}
        }
        
        param_space = {
            'model_params.C': (0.1, 100.0),
            'model_params.gamma': (0.001, 1.0),
            'model_params.kernel': ['linear', 'rbf', 'poly']
        }
        
        results = tune(
            base_config=base_config,
            config_space=param_space,  # Changed from param_space to config_space
            train_data=classification_data['train'],
            val_data=classification_data['val'],
            n_trials=8,
            verbose=False
        )
        
        assert 'best_params' in results
        assert 'best_value' in results
        assert 'best_model' in results
        
        # Test that best model can make predictions
        X_test, _ = classification_data['test']
        predictions = predict(model=results['best_model'], data=X_test)
        assert len(predictions) == len(X_test)
    
    def test_configuration_serialization_roundtrip(self, temp_dir):
        """Test configuration save/load roundtrip."""
        original_config = DLTConfig(
            model_type='sklearn.ensemble.RandomForestClassifier',
            model_params={'n_estimators': 50, 'random_state': 42},
            training={'epochs': 25},
            experiment={'name': 'serialization_test', 'seed': 123}
        )
        
        # Save to YAML
        yaml_path = temp_dir / 'config_test.yaml'
        original_config.save(yaml_path)
        
        # Load from YAML
        loaded_config = DLTConfig.from_file(yaml_path)
        
        # Compare key fields
        assert loaded_config.model_type == original_config.model_type
        assert loaded_config.model_params == original_config.model_params
        assert loaded_config.training['epochs'] == original_config.training['epochs']
        assert loaded_config.experiment['name'] == original_config.experiment['name']
    
    def test_multi_framework_comparison(self, classification_data):
        """Test training same task with different frameworks."""
        X_train, y_train = classification_data['train']
        X_test, y_test = classification_data['test']
        
        configs = [
            # Sklearn
            {
                'model_type': 'sklearn.ensemble.RandomForestClassifier',
                'model_params': {'n_estimators': 10, 'random_state': 42}
            },
            # PyTorch
            {
                'model_type': 'torch.nn.Sequential',
                'model_params': {
                    'layers': [
                        {'type': 'Linear', 'in_features': 10, 'out_features': 8},
                        {'type': 'ReLU'},
                        {'type': 'Linear', 'in_features': 8, 'out_features': 3}
                    ]
                },
                'training': {'epochs': 5, 'batch_size': 16}
            }
        ]
        
        results = []
        for config in configs:
            result = train(
                config=config,
                train_data=(X_train, y_train),
                test_data=(X_test, y_test),
                verbose=False
            )
            results.append(result)
        
        # Both frameworks should produce valid results
        assert len(results) == 2
        for result in results:
            assert 'model' in result
            assert 'test_results' in result
            assert 'accuracy' in result['test_results']
    
    def test_memory_efficiency_large_dataset(self):
        """Test framework with larger dataset."""
        # Generate larger dataset
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, 
            random_state=42
        )
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        config = {
            'model_type': 'sklearn.linear_model.SGDClassifier',
            'model_params': {'random_state': 42, 'max_iter': 100}
        }
        
        results = train(
            config=config,
            train_data=(X_train, y_train),
            test_data=(X_test, y_test)
        )
        
        assert 'model' in results
        assert 'test_results' in results
        assert results['test_results']['accuracy'] > 0.5  # Should be better than random
    
    def test_pytorch_with_advanced_features(self, classification_data):
        """Test PyTorch with advanced configuration options."""
        config = {
            'model_type': 'torch.nn.Sequential',
            'model_params': {
                'layers': [
                    {'type': 'Linear', 'in_features': 10, 'out_features': 16},
                    {'type': 'BatchNorm1d', 'num_features': 16},
                    {'type': 'ReLU'},
                    {'type': 'Dropout', 'p': 0.3},
                    {'type': 'Linear', 'in_features': 16, 'out_features': 8},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 8, 'out_features': 3}
                ]
            },
            'training': {
                'epochs': 4,
                'batch_size': 16,
                'optimizer': {'type': 'adamw', 'lr': 0.001},
                'gradient_clipping': {'enabled': True, 'max_norm': 1.0}
            },
            'hardware': {
                'device': 'cpu'  # Force CPU for testing
            }
        }
        
        results = train(
            config=config,
            train_data=classification_data['train'],
            val_data=classification_data['val'],
            test_data=classification_data['test'],
            verbose=False
        )
        
        assert 'model' in results
        assert 'history' in results
        assert len(results['history']['train_loss']) == 4
    
    def test_error_handling_invalid_config(self):
        """Test error handling with invalid configuration."""
        with pytest.raises((ValueError, ImportError, AttributeError)):
            config = {
                'model_type': 'nonexistent.InvalidModel',
                'model_params': {}
            }
            
            X, y = make_classification(n_samples=50, n_features=5, random_state=42)
            train(config=config, train_data=(X, y))
    
    def test_reproducibility(self, classification_data):
        """Test that results are reproducible with same seed."""
        config = {
            'model_type': 'sklearn.ensemble.RandomForestClassifier',
            'model_params': {'n_estimators': 10, 'random_state': 42},
            'experiment': {'seed': 42}
        }
        
        # Train twice with same config
        results1 = train(
            config=config,
            train_data=classification_data['train'],
            test_data=classification_data['test']
        )
        
        results2 = train(
            config=config,
            train_data=classification_data['train'],
            test_data=classification_data['test']
        )
        
        # Results should be identical (or very close)
        assert results1['test_results']['accuracy'] == results2['test_results']['accuracy']
    
    def test_config_inheritance_and_overrides(self):
        """Test configuration inheritance and parameter overrides."""
        base_config = DLTConfig(
            model_type='sklearn.svm.SVC',
            model_params={'kernel': 'rbf', 'random_state': 42}
        )
        
        # Create modified version
        modified_config = DLTConfig(
            model_type=base_config.model_type,
            model_params={**base_config.model_params, 'C': 10.0},
            training={'epochs': 50}
        )
        
        assert modified_config.model_params['kernel'] == 'rbf'  # Inherited
        assert modified_config.model_params['C'] == 10.0  # Overridden
        assert modified_config.training['epochs'] == 50  # New parameter"