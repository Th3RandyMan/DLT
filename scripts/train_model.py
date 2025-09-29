#!/usr/bin/env python3
"""
Simple training script for DLT Framework

Usage:
    python scripts/train_model.py --config config/basic_config.yaml
    python scripts/train_model.py --config config/cnn_config.yaml --data-path data/
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path so we can import dlt
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dlt import train, evaluate
from dlt.core.config import DLTConfig
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_sample_data():
    """Generate sample data for demonstration."""
    print("📊 Generating sample dataset...")
    
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    parser = argparse.ArgumentParser(description='Train a model using DLT Framework')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory (optional, will generate sample data)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--test', action='store_true',
                       help='Run evaluation on test set after training')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1
    
    print(f"⚙️  Loading configuration from: {config_path}")
    
    try:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = DLTConfig(**config_dict)
        elif config_path.suffix == '.json':
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = DLTConfig(**config_dict)
        else:
            print("❌ Configuration file must be .yaml, .yml, or .json")
            return 1
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    print(f"✅ Configuration loaded successfully")
    print(f"   Model type: {config.model_type}")
    print(f"   Experiment: {config.experiment.name}")
    
    # Load data
    if args.data_path:
        # TODO: Implement data loading from files
        print(f"📂 Loading data from: {args.data_path}")
        print("⚠️  Custom data loading not implemented in this demo script")
        print("   Using sample data instead...")
    
    train_data, val_data, test_data = load_sample_data()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"   Output directory: {output_dir}")
    
    try:
        results = train(
            config=config,
            train_data=train_data,
            val_data=val_data,
            output_dir=str(output_dir)
        )
        
        print(f"\n✅ Training completed!")
        print(f"   Final validation accuracy: {results['val_results']['accuracy']:.4f}")
        
        # Save results
        results_file = output_dir / f"{config.experiment.name}_results.json"
        import json
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {
                'experiment': config.experiment.name,
                'model_type': config.model_type,
                'val_accuracy': float(results['val_results']['accuracy']),
                'config': config_dict
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"   Results saved to: {results_file}")
        
        # Test evaluation
        if args.test:
            print(f"\n🧪 Running test evaluation...")
            test_results = evaluate(results['model'], test_data)
            print(f"   Test accuracy: {test_results['accuracy']:.4f}")
            
            # Update results file with test scores
            serializable_results['test_accuracy'] = float(test_results['accuracy'])
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        print(f"\n🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())