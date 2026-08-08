"""Implement the solvbert config baseline module."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Represent the ModelConfig baseline component."""
    vocab_size: int = 1000
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    hidden_dropout_rate: float = 0.4  # Baseline workflow step.


@dataclass
class PretrainConfig:
    """Represent the PretrainConfig baseline component."""
    # Process the experiment data.
    train_data: str = "data/train.csv"
    val_data: Optional[str] = None
    
    # Configure the baseline model.
    vocab_size: int = 1000
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 1024
    
    # Run the training step.
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 1000
    mlm_probability: float = 0.15
    max_length: int = 512
    
    # Baseline workflow step.
    output_dir: str = "./checkpoints"
    tokenizer_name: str = "bert-base-uncased"
    device: str = "cuda"
    save_steps: int = 1000


@dataclass
class FinetuneConfig:
    """Represent the FinetuneConfig baseline component."""
    # Process the experiment data.
    train_data: str = "data/train.csv"
    val_data: Optional[str] = None
    smiles_col: str = "smiles"
    label_col: str = "label"
    
    # Configure the baseline model.
    pretrained_model: Optional[str] = None
    vocab_size: int = 1000
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 1024
    hidden_dropout_rate: float = 0.4
    
    # Run the training step.
    batch_size: int = 16
    learning_rate: float = 8e-5
    num_epochs: int = 20
    warmup_steps: int = 500
    max_length: int = 512
    
    # Baseline workflow step.
    output_dir: str = "./checkpoints_finetune"
    tokenizer_path: Optional[str] = None
    tokenizer_name: str = "bert-base-uncased"
    device: str = "cuda"


# Configure experiment parameters.
PAPER_CONFIG = {
    'pretrain': {
        'batch_size': 16,
        'learning_rate': 2e-5,
        'mlm_probability': 0.15,
        'num_epochs': 10,
    },
    'finetune': {
        'batch_size': 16,
        'learning_rate': 8e-5,
        'num_epochs': 20,
        'hidden_dropout_rate': 0.4,
    }
}

