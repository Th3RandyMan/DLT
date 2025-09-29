"""
DLT Model Templates

Provides pre-built model architectures with smart defaults for common ML/DL tasks.
Inspired by flexible model creation patterns.
"""

from typing import Dict, Any, Optional, List, Union, Type
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ModelTemplate(ABC):
    """Base class for model templates."""
    
    SUPPORTED_LOSS_FUNCTIONS = {}
    DEFAULT_LOSS_CONFIG = {}
    
    @abstractmethod
    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build the model from configuration."""
        pass
    
    @classmethod
    def get_default_config(cls, task_type: str = 'auto', **kwargs) -> Dict[str, Any]:
        """Get default configuration for this template."""
        return {
            'template': cls.__name__,
            'task_type': task_type,
            'loss': cls.DEFAULT_LOSS_CONFIG.copy(),
            **kwargs
        }


class MLPTemplate(ModelTemplate):
    """Multi-Layer Perceptron template with flexible architecture."""
    
    SUPPORTED_LOSS_FUNCTIONS = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'cross_entropy': nn.CrossEntropyLoss,
        'bce': nn.BCEWithLogitsLoss,
        'huber': nn.HuberLoss,
    }
    
    DEFAULT_LOSS_CONFIG = {
        'type': 'auto',  # Auto-detect based on task
        'weights': None,
        'adaptive_weighting': False,
        'label_smoothing': 0.0
    }
    
    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build MLP from configuration."""
        input_size = config.get('input_size', config.get('in_features', 128))
        output_size = config.get('output_size', config.get('out_features', 10))
        hidden_sizes = config.get('hidden_sizes', [256, 128])
        dropout = config.get('dropout', 0.1)
        activation = config.get('activation', 'relu')
        batch_norm = config.get('batch_norm', True)
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
        }
        return activations.get(activation.lower(), nn.ReLU())


class ConvNetTemplate(ModelTemplate):
    """Convolutional Neural Network template."""
    
    SUPPORTED_LOSS_FUNCTIONS = {
        'cross_entropy': nn.CrossEntropyLoss,
        'focal': 'FocalLoss',  # Custom implementation
        'bce': nn.BCEWithLogitsLoss,
        'mse': nn.MSELoss,
    }
    
    DEFAULT_LOSS_CONFIG = {
        'type': 'cross_entropy',
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'label_smoothing': 0.0
    }
    
    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build ConvNet from configuration."""
        in_channels = config.get('in_channels', 3)
        num_classes = config.get('num_classes', 10)
        channels = config.get('channels', [32, 64, 128])
        kernel_sizes = config.get('kernel_sizes', [3, 3, 3])
        pool_sizes = config.get('pool_sizes', [2, 2, 2])
        dropout = config.get('dropout', 0.1)
        
        layers = []
        prev_channels = in_channels
        
        # Convolutional layers
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(channels, kernel_sizes, pool_sizes)):
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(pool_size)
            ])
            prev_channels = out_channels
        
        # Global average pooling and classifier
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev_channels, num_classes)
        ])
        
        return nn.Sequential(*layers)


class TransformerTemplate(ModelTemplate):
    """Transformer template for sequence tasks."""
    
    SUPPORTED_LOSS_FUNCTIONS = {
        'cross_entropy': nn.CrossEntropyLoss,
        'mse': nn.MSELoss,
        'label_smoothing_ce': 'LabelSmoothingCrossEntropy',
    }
    
    DEFAULT_LOSS_CONFIG = {
        'type': 'cross_entropy',
        'label_smoothing': 0.1,
        'ignore_index': -100
    }
    
    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build Transformer from configuration."""
        vocab_size = config.get('vocab_size', 10000)
        d_model = config.get('d_model', 512)
        nhead = config.get('nhead', 8)
        num_layers = config.get('num_layers', 6)
        dim_feedforward = config.get('dim_feedforward', 2048)
        dropout = config.get('dropout', 0.1)
        max_seq_len = config.get('max_seq_len', 512)
        
        class TransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.classifier = nn.Linear(d_model, vocab_size)
                
            def forward(self, x):
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:seq_len]
                x = self.transformer(x)
                return self.classifier(x)
        
        return TransformerModel()


class VAETemplate(ModelTemplate):
    """Variational Autoencoder template."""
    
    SUPPORTED_LOSS_FUNCTIONS = {
        'vae': 'VAELoss',  # Custom implementation
        'reconstruction': nn.MSELoss,
        'kld': 'KLDivergenceLoss',
    }
    
    DEFAULT_LOSS_CONFIG = {
        'type': 'vae',
        'beta': 1.0,  # β-VAE parameter
        'reconstruction_weight': 1.0,
        'kld_weight': 1.0,
        'adaptive_weighting': True
    }
    
    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build VAE from configuration."""
        input_dim = config.get('input_dim', 784)
        latent_dim = config.get('latent_dim', 64)
        hidden_dims = config.get('hidden_dims', [512, 256])
        
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                
                self.encoder = nn.Sequential(*encoder_layers)
                self.mu_layer = nn.Linear(prev_dim, latent_dim)
                self.logvar_layer = nn.Linear(prev_dim, latent_dim)
                
                # Decoder
                decoder_layers = []
                prev_dim = latent_dim
                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
                
            def encode(self, x):
                h = self.encoder(x)
                return self.mu_layer(h), self.logvar_layer(h)
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return {
                    'reconstruction': self.decode(z),
                    'mu': mu,
                    'logvar': logvar,
                    'z': z
                }
        
        return VAE()


class GANTemplate(ModelTemplate):
    """Generative Adversarial Network template."""
    
    SUPPORTED_LOSS_FUNCTIONS = {
        'adversarial': 'AdversarialLoss',
        'wgan': 'WGANLoss',
        'lsgan': 'LSGANLoss',
        'hinge': 'HingeLoss',
    }
    
    DEFAULT_LOSS_CONFIG = {
        'type': 'adversarial',
        'generator_weight': 1.0,
        'discriminator_weight': 1.0,
        'gradient_penalty': 10.0,  # For WGAN-GP
    }
    
    def build_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build GAN from configuration."""
        latent_dim = config.get('latent_dim', 100)
        output_dim = config.get('output_dim', 784)
        hidden_dims = config.get('hidden_dims', [256, 512])
        
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev_dim = latent_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU()
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, output_dim))
                layers.append(nn.Tanh())
                self.model = nn.Sequential(*layers)
            
            def forward(self, z):
                return self.model(z)
        
        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev_dim = output_dim
                
                for hidden_dim in reversed(hidden_dims):
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.LeakyReLU(0.2)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        class GAN(nn.Module):
            def __init__(self):
                super().__init__()
                self.generator = Generator()
                self.discriminator = Discriminator()
            
            def forward(self, z):
                return self.generator(z)
        
        return GAN()


# Registry of available templates
MODEL_TEMPLATES = {
    'mlp': MLPTemplate,
    'multi_layer_perceptron': MLPTemplate,
    'convnet': ConvNetTemplate,
    'cnn': ConvNetTemplate,
    'transformer': TransformerTemplate,
    'vae': VAETemplate,
    'variational_autoencoder': VAETemplate,
    'gan': GANTemplate,
    'generative_adversarial_network': GANTemplate,
}


def get_template(template_name: str) -> Type[ModelTemplate]:
    """Get model template by name."""
    if template_name.lower() not in MODEL_TEMPLATES:
        available = ', '.join(MODEL_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    return MODEL_TEMPLATES[template_name.lower()]


def create_model_from_template(template_name: str, config: Dict[str, Any]) -> nn.Module:
    """Create model from template and configuration."""
    template_class = get_template(template_name)
    template = template_class()
    return template.build_model(config)


def get_template_config(template_name: str, **kwargs) -> Dict[str, Any]:
    """Get default configuration for a template."""
    template_class = get_template(template_name)
    return template_class.get_default_config(**kwargs)