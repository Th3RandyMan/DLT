#!/usr/bin/env python3
"""
Quick Start Example for DLT Framework

This example demonstrates the basic usage of DLT for both traditional ML
and deep learning models with minimal setup.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import DLT components
from dlt import train, evaluate, tune
from dlt.core.config import DLTConfig


def generate_sample_data():
    """Generate sample datasets for demonstration."""
    print("🔢 Generating sample datasets...")
    
    # Classification data
    X_clf, y_clf = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_classes=3, 
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Regression data  
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=20, 
        noise=0.1,
        random_state=42
    )
    
    # Split classification data
    X_train_clf, X_temp_clf, y_train_clf, y_temp_clf = train_test_split(
        X_clf, y_clf, test_size=0.4, random_state=42
    )
    X_val_clf, X_test_clf, y_val_clf, y_test_clf = train_test_split(
        X_temp_clf, y_temp_clf, test_size=0.5, random_state=42
    )
    
    # Split regression data
    X_train_reg, X_temp_reg, y_train_reg, y_temp_reg = train_test_split(
        X_reg, y_reg, test_size=0.4, random_state=42
    )
    X_val_reg, X_test_reg, y_val_reg, y_test_reg = train_test_split(
        X_temp_reg, y_temp_reg, test_size=0.5, random_state=42
    )
    
    return {
        'classification': {
            'train': (X_train_clf, y_train_clf),
            'val': (X_val_clf, y_val_clf),
            'test': (X_test_clf, y_test_clf)
        },
        'regression': {
            'train': (X_train_reg, y_train_reg),
            'val': (X_val_reg, y_val_reg),
            'test': (X_test_reg, y_test_reg)
        }
    }


def example_1_simple_sklearn():
    """Example 1: Simple scikit-learn model (2 lines!)"""
    print("\n🌳 Example 1: Simple Random Forest")
    print("=" * 50)
    
    data = generate_sample_data()['classification']
    
    # This is literally all you need!
    results = train(
        config={'model_type': 'sklearn.ensemble.RandomForestClassifier'},
        train_data=data['train'],
        val_data=data['val']
    )
    
    # Test the model
    test_results = evaluate(results['model'], data['test'])
    
    print(f"✅ Validation Accuracy: {results['val_results']['accuracy']:.4f}")
    print(f"✅ Test Accuracy: {test_results['accuracy']:.4f}")
    
    return results


def example_2_neural_network():
    """Example 2: Simple neural network with PyTorch"""
    print("\n🧠 Example 2: Neural Network")
    print("=" * 50)
    
    data = generate_sample_data()['classification']
    
    config = {
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
            'epochs': 20,  # Quick demo
            'batch_size': 32,
            'early_stopping': {'patience': 5}
        },
        'experiment': {
            'name': 'simple_nn_demo'
        }
    }
    
    results = train(config=config, train_data=data['train'], val_data=data['val'])
    test_results = evaluate(results['model'], data['test'])
    
    print(f"✅ Final Validation Accuracy: {results['val_results']['accuracy']:.4f}")
    print(f"✅ Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"🔥 Trained for {len(results['history']['train_loss'])} epochs")
    
    return results


def example_3_optimized_training():
    """Example 3: Training with performance optimizations"""
    print("\n⚡ Example 3: Optimized Training")
    print("=" * 50)
    
    data = generate_sample_data()['classification']
    
    config = {
        'model_type': 'torch.nn.Sequential',
        'model_params': {
            'layers': [
                {'type': 'Linear', 'in_features': 20, 'out_features': 128},
                {'type': 'ReLU'},
                {'type': 'BatchNorm1d', 'num_features': 128},
                {'type': 'Dropout', 'p': 0.3},
                {'type': 'Linear', 'in_features': 128, 'out_features': 64},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': 0.3},
                {'type': 'Linear', 'in_features': 64, 'out_features': 3}
            ]
        },
        'training': {
            'optimizer': {'type': 'adamw', 'lr': 0.001, 'weight_decay': 0.01},
            'epochs': 30,
            'batch_size': 64
        },
        'hardware': {
            'device': 'auto'  # Auto-detect best device
        },
        'performance': {
            'mixed_precision': {'enabled': 'auto'},  # Auto-enable if supported
            'compile': {'enabled': 'auto'},          # PyTorch 2.0+ optimization
            'memory_optimization': True
        },
        'experiment': {
            'name': 'optimized_nn_demo',
            'tags': ['demo', 'optimized']
        }
    }
    
    print("🔧 Enabled optimizations:")
    print("   - Mixed precision training (if supported)")
    print("   - Model compilation (PyTorch 2.0+)")
    print("   - Memory optimization")
    
    results = train(config=config, train_data=data['train'], val_data=data['val'])
    test_results = evaluate(results['model'], data['test'])
    
    print(f"✅ Final Validation Accuracy: {results['val_results']['accuracy']:.4f}")
    print(f"✅ Test Accuracy: {test_results['accuracy']:.4f}")
    
    return results


def example_4_hyperparameter_tuning():
    """Example 4: Automatic hyperparameter optimization"""
    print("\n🎯 Example 4: Hyperparameter Tuning")
    print("=" * 50)
    
    data = generate_sample_data()['classification']
    
    base_config = {
        'model_type': 'sklearn.ensemble.RandomForestClassifier',
        'model_params': {
            'random_state': 42  # Keep this fixed
        }
    }
    
    # Define search space
    search_space = {
        'model_params.n_estimators': (50, 200),
        'model_params.max_depth': (3, 15),
        'model_params.min_samples_split': (2, 10),
        'model_params.min_samples_leaf': (1, 5)
    }
    
    print("🔍 Searching for optimal hyperparameters...")
    print(f"   Search space: {len(search_space)} parameters")
    print("   Trials: 20")
    
    # Run optimization
    best_results = tune(
        base_config=base_config,
        config_space=search_space,
        train_data=data['train'],
        val_data=data['val'],
        n_trials=20  # Quick demo
    )
    
    # Test best model
    test_results = evaluate(best_results['best_model'], data['test'])
    
    print(f"✅ Best Validation Score: {best_results['best_value']:.4f}")
    print(f"✅ Test Accuracy: {test_results['accuracy']:.4f}")
    print("🏆 Best Parameters:")
    for param, value in best_results['best_params'].items():
        print(f"   {param}: {value}")
    
    return best_results


def example_5_config_file():
    """Example 5: Using configuration files"""
    print("\n📄 Example 5: Configuration Files")
    print("=" * 50)
    
    # Create a configuration
    config = DLTConfig(
        model_type='sklearn.svm.SVC',
        model_params={
            'kernel': 'rbf',
            'C': 1.0,
            'random_state': 42
        },
        experiment={
            'name': 'svm_demo',
            'description': 'Support Vector Machine example'
        }
    )
    
    print("📋 Configuration created:")
    print(f"   Model: {config.model_type}")
    print(f"   Parameters: {config.model_params}")
    
    data = generate_sample_data()['classification']
    
    # Train with config object
    results = train(config=config, train_data=data['train'], val_data=data['val'])
    test_results = evaluate(results['model'], data['test'])
    
    print(f"✅ Validation Accuracy: {results['val_results']['accuracy']:.4f}")
    print(f"✅ Test Accuracy: {test_results['accuracy']:.4f}")
    
    return results


def main():
    """Run all examples."""
    print("🚀 DLT Framework - Quick Start Examples")
    print("=" * 60)
    print()
    print("This script demonstrates various features of DLT:")
    print("1. Simple scikit-learn models (2 lines!)")
    print("2. Neural networks with PyTorch")
    print("3. Performance-optimized training")
    print("4. Automatic hyperparameter tuning")
    print("5. Configuration management")
    
    try:
        # Run examples
        example_1_simple_sklearn()
        example_2_neural_network()
        example_3_optimized_training()
        example_4_hyperparameter_tuning()
        example_5_config_file()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("- Check the generated models in the 'models/' directory")
        print("- Look at training logs and results")
        print("- Try modifying the configurations")
        print("- Explore the config examples in 'config/' directory")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("\nTips:")
        print("- Make sure DLT is properly installed")
        print("- Check that all dependencies are available")
        print("- Try running individual examples separately")
        

if __name__ == "__main__":
    main()