from dataclasses import dataclass, field
from uuid import uuid4

import yaml
from dacite import from_dict


@dataclass
class General:
    output_dir: str
    log_to_wandb: bool = False
    run_id: str = str(uuid4())
    checkpoint_freq: int = 10


@dataclass
class Data:
    dataset: str
    dataset_root: str
    train_aug: str = 'mocov2'
    val_aug: str = 'mocov2'
    image_size: int = 224
    max_images: int = None


@dataclass
class Optimisation:
    device: str = 'cpu'
    lr: float = 3e-4
    weight_decay: float = 0.0
    workers: int = 4
    batch_size: int = 4
    epochs: int = 1
    warmup_epochs: int = 1
    optimiser: str = 'adam'


@dataclass
class Backbone:
    name: str = 'resnet50'


@dataclass
class Model:
    backbone: Backbone = field(default_factory=lambda: Backbone())
    hidden_dim: int = 2048
    proj_dim: int = 128
    memory_bank_size: int = 4096


@dataclass
class Config:
    general: General
    data: Data
    model: Model = field(default_factory=lambda: Model())
    optim: Optimisation = field(default_factory=lambda: Optimisation())


def load_config(config_path: str) -> Config:
    with open(config_path) as file:
        data = yaml.safe_load(file)
    return from_dict(Config, data)
