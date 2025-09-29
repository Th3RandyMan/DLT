"""
DLT Trainer System - Universal Training Orchestration

Simple trainer that works with any model type.
"""

from typing import Any, Dict, Optional
import time
import numpy as np

# Optional imports with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DLTTrainer:
    """Universal trainer for any ML/DL model."""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.framework = config.get_framework()
        
        # Get device safely
        self.device = 'cpu'
        if hasattr(config, 'hardware') and isinstance(config.hardware, dict):
            device_config = config.hardware.get('device', 'auto')
            if device_config == 'auto':
                if HAS_TORCH and torch.cuda.is_available():
                    self.device = 'cuda:0'
                else:
                    self.device = 'cpu'
            else:
                self.device = device_config
        
        # Setup framework-specific components
        if self.framework == 'torch' and HAS_TORCH:
            self._setup_torch_components()
    
    def _setup_torch_components(self):
        """Setup PyTorch optimizer and loss function."""
        if hasattr(self.config, 'training') and isinstance(self.config.training, dict):
            optimizer_config = self.config.training.get('optimizer', {'type': 'adam', 'lr': 0.001})
            lr = optimizer_config.get('lr', 0.001)
            opt_type = optimizer_config.get('type', 'adam').lower()
            
            model_params = self.model._model.parameters() if hasattr(self.model, '_model') else self.model.parameters()
            
            if opt_type == 'adam':
                self.optimizer = torch.optim.Adam(model_params, lr=lr)
            elif opt_type == 'sgd':
                self.optimizer = torch.optim.SGD(model_params, lr=lr)
            elif opt_type == 'adamw':
                self.optimizer = torch.optim.AdamW(model_params, lr=lr)
            else:
                self.optimizer = torch.optim.Adam(model_params, lr=lr)
            
            self.criterion = torch.nn.CrossEntropyLoss()
    
    def train(self, train_data, val_data=None, verbose=False):
        """Train the model."""
        if verbose:
            print(f"Training {self.config.model_type}...")
        
        start_time = time.time()
        
        if self.framework == 'sklearn':
            history = self._train_sklearn(train_data)
        elif self.framework == 'torch':
            history = self._train_torch(train_data, val_data, verbose)
        else:
            history = self._train_generic(train_data)
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"Training completed in {training_time:.2f}s")
        
        return history
    
    def _train_sklearn(self, train_data):
        """Train sklearn model."""
        X_train, y_train = train_data
        model = self.model._model if hasattr(self.model, '_model') else self.model.model
        model.fit(X_train, y_train)
        return {'training_time': time.time()}
    
    def _train_torch(self, train_data, val_data=None, verbose=False):
        """Train PyTorch model."""
        X_train, y_train = train_data
        
        # Get training parameters
        epochs = 10
        batch_size = 32
        if hasattr(self.config, 'training') and isinstance(self.config.training, dict):
            epochs = self.config.training.get('epochs', 10)
            batch_size = self.config.training.get('batch_size', 32)
        
        # Convert to tensors
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train)
            y_train = torch.LongTensor(y_train)
        
        # Move to same device as model
        model_device = next(self.model._model.parameters()).device
        X_train = X_train.to(model_device)
        y_train = y_train.to(model_device)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        self.model._model.train()
        for epoch in range(epochs):
            # Simple batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model._model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            # Record loss
            with torch.no_grad():
                outputs = self.model._model(X_train)
                train_loss = self.criterion(outputs, y_train).item()
                history['train_loss'].append(train_loss)
                
                if val_data is not None:
                    X_val, y_val = val_data
                    if not isinstance(X_val, torch.Tensor):
                        X_val = torch.FloatTensor(X_val)
                        y_val = torch.LongTensor(y_val)
                    X_val = X_val.to(model_device)
                    y_val = y_val.to(model_device)
                    
                    val_outputs = self.model._model(X_val)
                    val_loss = self.criterion(val_outputs, y_val).item()
                    history['val_loss'].append(val_loss)
                else:
                    history['val_loss'].append(train_loss)
            
            if verbose and epoch % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
        
        return history
    
    def _train_generic(self, train_data):
        """Train generic model."""
        X_train, y_train = train_data
        if hasattr(self.model, 'fit'):
            self.model.fit(X_train, y_train)
        return {'training_time': time.time()}
    
    def evaluate(self, test_data):
        """Evaluate the model."""
        X_test, y_test = test_data
        
        if self.framework == 'sklearn':
            model = self.model._model if hasattr(self.model, '_model') else self.model.model
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            return {'accuracy': accuracy}
        elif self.framework == 'torch':
            self.model._model.eval()
            with torch.no_grad():
                if not isinstance(X_test, torch.Tensor):
                    X_test = torch.FloatTensor(X_test)
                
                # Move to same device as model
                model_device = next(self.model._model.parameters()).device
                X_test = X_test.to(model_device)
                
                outputs = self.model._model(X_test)
                _, predictions = torch.max(outputs, 1)
                predictions = predictions.cpu().numpy()
                accuracy = accuracy_score(y_test, predictions)
                return {'accuracy': accuracy}
        else:
            # Generic evaluation
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            return {'accuracy': accuracy}
    
    def save_checkpoint(self, path):
        """Save training checkpoint."""
        checkpoint = {
            'model_state': self.model.model.state_dict() if hasattr(self.model.model, 'state_dict') else None,
            'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'config': self.config
        }
        
        if self.framework == 'torch':
            torch.save(checkpoint, path)
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path):
        """Load training checkpoint."""
        if self.framework == 'torch':
            checkpoint = torch.load(path)
        else:
            import pickle
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
        
        if checkpoint.get('model_state') and hasattr(self.model.model, 'load_state_dict'):
            self.model.model.load_state_dict(checkpoint['model_state'])
        
        if checkpoint.get('optimizer_state') and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])