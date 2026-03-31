from dataclasses import dataclass, asdict, field
from typing import Optional, List
import yaml

@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str = "data/raw/"
    processed_path: str = "data/processed/"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Physics-specific
    index_pool_size: int = 100
    normalize_indices: bool = True
    normalize_momentum: bool = True
    physics_model: str = "QED"  # or "QCD"

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = "baseline"  # "baseline" or "jepa"
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 256

@dataclass
class JEPAConfig(ModelConfig):
    """JEPA-specific configuration"""
    context_encoder_layers: int = 6
    target_encoder_layers: int = 6
    share_encoders: bool = False
    temperature: float = 0.1
    use_projection_head: bool = True
    projection_dim: Optional[int] = None

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 32
    val_batch_size: int = 64
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Optimizer & scheduler
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"  # "cosine" or "linear"
    
    # Checkpointing
    save_freq: int = 5  # Save every N epochs
    val_freq: int = 1   # Validate every N epochs
    
    # W&B logging
    use_wandb: bool = True
    project_name: str = "symba-hep-lm-jepa"
    run_name: str = "baseline-qed"
    
    # Hardware
    device: str = "cuda"
    distributed: bool = False
    num_workers: int = 4

@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)