"""Implement the glam model glam baseline module."""
import torch
import torch.nn as nn
import random
from typing import Dict, List, Tuple, Optional
from .architectures import SingleGraphArchitecture, PairGraphArchitecture, TripleGraphArchitecture


class ConfigurationSpace:
    """Represent the ConfigurationSpace baseline component."""
    def __init__(self):
        # Baseline workflow step.
        self.norm_types = ['batch', 'layer', 'instance', None]
        self.activations = ['relu', 'gelu', 'celu', 'tanh']
        self.mp_types = ['gcn', 'gat', 'mpn', 'tri_mpn', 'light_tri_mpn']
        self.pool_types = ['mean', 'max', 'sum']
        self.fusion_types = ['concat', 'dot', 'add', 'multiply']
        self.dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.num_mp_layers_range = [1, 2, 3, 4, 5, 6]
        self.hidden_dims = [64, 128, 256, 512]
        
        # Run the training step.
        self.batch_sizes = [16, 32, 64, 128]
        self.learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
        self.optimizers = ['adam', 'sgd', 'adamw']
        self.loss_functions = ['mse', 'mae', 'bce', 'ce']
    
    def sample_config(self, task_type='property'):
        """Run the sample config baseline operation."""
        config = {
            'norm_type': random.choice(self.norm_types),
            'activation': random.choice(self.activations),
            'mp_type': random.choice(self.mp_types),
            'pool_type': random.choice(self.pool_types),
            'dropout': random.choice(self.dropout_values),
            'num_mp_layers': random.choice(self.num_mp_layers_range),
            'hidden_dim': random.choice(self.hidden_dims),
            'batch_size': random.choice(self.batch_sizes),
            'learning_rate': random.choice(self.learning_rates),
            'optimizer': random.choice(self.optimizers),
            'loss_function': random.choice(self.loss_functions),
        }
        
        if task_type == 'interaction':
            config['fusion_type'] = random.choice(self.fusion_types)
        
        return config
    
    def sample_configs(self, n, task_type='property'):
        """Run the sample configs baseline operation."""
        return [self.sample_config(task_type) for _ in range(n)]


class GLAM(nn.Module):
    """Represent the GLAM baseline component."""
    def __init__(self, node_dim, out_dim, task_type='property', 
                 config=None, ensemble_size=3):
        """Run the init baseline operation."""
        super(GLAM, self).__init__()
        self.task_type = task_type
        self.ensemble_size = ensemble_size
        
        # Baseline workflow step.
        if config is None:
            config = {
                'norm_type': 'batch',
                'activation': 'relu',
                'mp_type': 'gcn',
                'pool_type': 'mean',
                'dropout': 0.2,
                'num_mp_layers': 3,
                'hidden_dim': 128,
                'fusion_type': 'concat' if task_type == 'interaction' else None
            }
        
        self.config = config
        
        # Baseline workflow step.
        if task_type == 'property':
            self.model = SingleGraphArchitecture(
                node_dim=node_dim,
                hidden_dim=config['hidden_dim'],
                out_dim=out_dim,
                num_mp_layers=config['num_mp_layers'],
                mp_type=config['mp_type'],
                norm_type=config['norm_type'],
                dropout=config['dropout'],
                activation=config['activation'],
                pool_type=config['pool_type']
            )
        elif task_type == 'interaction':
            self.model = PairGraphArchitecture(
                node_dim=node_dim,
                hidden_dim=config['hidden_dim'],
                out_dim=out_dim,
                num_mp_layers=config['num_mp_layers'],
                mp_type=config['mp_type'],
                norm_type=config['norm_type'],
                dropout=config['dropout'],
                activation=config['activation'],
                pool_type=config['pool_type'],
                fusion_type=config['fusion_type']
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(self, *args, **kwargs):
        """Run the forward baseline operation."""
        return self.model(*args, **kwargs)


class GLAMEnsemble(nn.Module):
    """Represent the GLAMEnsemble baseline component."""
    def __init__(self, node_dim, out_dim, task_type='property',
                 configs=None, ensemble_size=3):
        """Run the init baseline operation."""
        super(GLAMEnsemble, self).__init__()
        self.task_type = task_type
        self.ensemble_size = ensemble_size
        
        # Baseline workflow step.
        if configs is None:
            config_space = ConfigurationSpace()
            configs = config_space.sample_configs(ensemble_size, task_type)
        
        # Configure the baseline model.
        self.models = nn.ModuleList([
            GLAM(node_dim, out_dim, task_type, config, ensemble_size=1)
            for config in configs
        ])
    
    def forward(self, *args, **kwargs):
        """Run the forward baseline operation."""
        outputs = []
        for model in self.models:
            output = model(*args, **kwargs)
            outputs.append(output)
        
        # Baseline workflow step.
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output
    
    def predict_with_weights(self, *args, weights=None, **kwargs):
        """Run the predict with weights baseline operation."""
        outputs = []
        for model in self.models:
            output = model(*args, **kwargs)
            outputs.append(output)
        
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        
        # Baseline workflow step.
        ensemble_output = sum(w * out for w, out in zip(weights, outputs))
        return ensemble_output


class GLAM_LLE(nn.Module):
    """Represent the GLAM_LLE baseline component."""
    def __init__(self, node_dim, out_dim, config=None):
        """Run the init baseline operation."""
        super(GLAM_LLE, self).__init__()
        
        # Baseline workflow step.
        if config is None:
            config = {
                'norm_type': 'batch',
                'activation': 'relu',
                'mp_type': 'gcn',
                'pool_type': 'mean',
                'dropout': 0.2,
                'num_mp_layers': 3,
                'hidden_dim': 128,
                'fusion_type': 'concat'
            }
        
        self.config = config
        
        # Baseline workflow step.
        self.model = TripleGraphArchitecture(
            node_dim=node_dim,
            hidden_dim=config['hidden_dim'],
            out_dim=out_dim,
            num_mp_layers=config['num_mp_layers'],
            mp_type=config['mp_type'],
            norm_type=config['norm_type'],
            dropout=config['dropout'],
            activation=config['activation'],
            pool_type=config['pool_type'],
            fusion_type=config['fusion_type']
        )
    
    def forward(self, il_graph, comp2_graph, comp3_graph, temperature=None):
        """Run the forward baseline operation."""
        return self.model(
            il_graph.x, il_graph.edge_index,
            comp2_graph.x, comp2_graph.edge_index,
            comp3_graph.x, comp3_graph.edge_index,
            edge_attr1=il_graph.edge_attr if hasattr(il_graph, 'edge_attr') else None,
            edge_attr2=comp2_graph.edge_attr if hasattr(comp2_graph, 'edge_attr') else None,
            edge_attr3=comp3_graph.edge_attr if hasattr(comp3_graph, 'edge_attr') else None,
            batch1=il_graph.batch if hasattr(il_graph, 'batch') else None,
            batch2=comp2_graph.batch if hasattr(comp2_graph, 'batch') else None,
            batch3=comp3_graph.batch if hasattr(comp3_graph, 'batch') else None,
            temperature=temperature
        )

