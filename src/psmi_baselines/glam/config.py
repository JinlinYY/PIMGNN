"""Implement the glam config baseline module."""
import os
from dataclasses import dataclass
from typing import Optional

from psmi_baselines.paths import EXPERIMENT_ROOT, MODEL_ROOT, TOTAL_CSV


@dataclass
class DataConfig:
    """Represent the DataConfig baseline component."""
    csv_path: str = str(TOTAL_CSV)
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    batch_size: int = 32
    num_workers: int = 0  # Baseline workflow step.


@dataclass
class ModelConfig:
    """Represent the ModelConfig baseline component."""
    node_dim: int = 8  # Process the experiment data.
    out_dim: int = 6   # Configure the output artifacts.
    hidden_dim: int = 128
    num_mp_layers: int = 3
    mp_type: str = 'gcn'  # 'gcn', 'gat', 'mpn', 'tri_mpn', 'light_tri_mpn'
    norm_type: str = 'batch'  # 'batch', 'layer', 'instance', None
    dropout: float = 0.2
    activation: str = 'relu'  # 'relu', 'gelu', 'celu', 'tanh'
    pool_type: str = 'mean'  # 'mean', 'max', 'sum'
    fusion_type: str = 'concat'  # 'concat', 'add', 'multiply', 'attention'
    use_temperature: bool = True


@dataclass
class TrainingConfig:
    """Represent the TrainingConfig baseline component."""
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    scheduler: str = 'plateau'  # 'plateau', 'step', 'cosine'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stop_patience: int = 100  # Baseline workflow step.
    early_stop_min_delta: float = 0.0  # Apply early stopping.
    checkpoint_save_freq: int = 10  # Save the generated artifacts.
    rest_interval_hours: float = 3.0  # Baseline workflow step.
    rest_duration: int = 600  # Baseline workflow step.
    gradient_clip: Optional[float] = None  # Update model gradients.


@dataclass
class Config:
    """Represent the Config baseline component."""
    # Configure the runtime device.
    device: str = 'cuda'  # Baseline workflow step.
    seed: int = 2024  # Baseline workflow step.
    
    # Configure repository paths.
    model_save_dir: str = str(MODEL_ROOT / "glam")
    log_dir: str = str(EXPERIMENT_ROOT / "runs" / "glam" / "logs")
    result_dir: str = str(EXPERIMENT_ROOT / "runs" / "glam" / "results")
    
    # Baseline workflow step.
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        """Run the post init baseline operation."""
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        
        # Configure repository paths.
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
    
    def to_dict(self):
        """Run the to dict baseline operation."""
        return {
            'device': self.device,
            'seed': self.seed,
            'data': {
                'csv_path': self.data.csv_path,
                'test_size': self.data.test_size,
                'val_size': self.data.val_size,
                'batch_size': self.data.batch_size,
            },
            'model': {
                'hidden_dim': self.model.hidden_dim,
                'num_mp_layers': self.model.num_mp_layers,
                'mp_type': self.model.mp_type,
                'dropout': self.model.dropout,
                'fusion_type': self.model.fusion_type,
            },
            'training': {
                'num_epochs': self.training.num_epochs,
                'learning_rate': self.training.learning_rate,
                'optimizer': self.training.optimizer,
            }
        }


# Baseline workflow step.
default_config = Config()

