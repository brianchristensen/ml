# config.py
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

@dataclass
class ModelConfig:
    num_nodes: int = 5
    som_dim: int = 10
    latent_dim: int = 256
    input_channels: int = 3
    output_dim: int = 10
    temperature: float = 0.3
    num_heads: int = 4
    decoder_output_shape: Tuple[int, int] = (32, 32)
    task_type: str = "classification"  # or "regression"
    model_save_path: str = "models/model_hitop.pth"

@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    # Loss weights
    node_loss_λ: float = 0.3
    node_div_λ: float = 0.04
    graph_div_λ: float = 0.04
    proto_usage_λ: float = 0.625
    proto_recon_λ: float = 0.5

    label_smoothing: float = 0.1

@dataclass
class DatasetConfig:
    name: str = "cifar10"                     # Dataset name (used in switch)
    input_shape: Tuple[int, int, int] = (3, 32, 32)
    num_classes: int = 10
    task_type: str = "classification"         # or "regression"
    root: str = "data"
    transform_train: Optional[Callable] = None
    transform_test: Optional[Callable] = None

def autofill_model_config(model_cfg: ModelConfig, dataset_cfg: DatasetConfig) -> ModelConfig:
    model_cfg.input_channels = dataset_cfg.input_shape[0]
    model_cfg.output_dim = dataset_cfg.num_classes
    model_cfg.task_type = dataset_cfg.task_type
    model_cfg.decoder_output_shape = dataset_cfg.input_shape[1:]  # height, width
    return model_cfg
