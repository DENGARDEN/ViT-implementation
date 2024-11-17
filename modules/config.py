from dataclasses import dataclass

from torch.optim.lr_scheduler import LinearLR, LRScheduler


@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 10  # CIFAR10
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @classmethod
    def base(cls):
        return cls()  # 86M params

    @classmethod
    def large(cls):
        return cls(embed_dim=1024, depth=24, num_heads=16)  # 307M params

    @classmethod
    def huge(cls):
        return cls(embed_dim=1280, depth=32, num_heads=16)  # 632M params


@dataclass
class TrainingConfig:
    epochs: int = 7
    lr: float = 8e-4
    lr_scheduler: LRScheduler = LinearLR
    weight_decay: float = 0.1
    seed: int = 42

    @classmethod
    def base(cls):
        return cls()

    @classmethod
    def large(cls):
        return cls(epochs=7, lr=4e-4)

    @classmethod
    def huge(cls):
        return cls(epochs=14, lr=3e-4)

    @classmethod
    def resnet152(cls):
        return cls(epochs=7, lr=6e-4)


@dataclass
class DataConfig:
    batch_size: int = 4096
    num_workers: int = 4
    img_size: int = 224
    num_classes: int = 10  # CIFAR10

    @classmethod
    def base(cls):
        return cls()

    @classmethod
    def large(cls):
        # TODO
        return cls()

    @classmethod
    def huge(cls):
        # TODO
        return cls()
