from dataclasses import dataclass, field

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
    epochs: int = 200  # Original: 7
    lr: float = 8e-4
    lr_scheduler: LRScheduler = LinearLR
    weight_decay: float = 0.1
    seed: int = 42
    model_name: str = "ViT"
    patience: int = 5

    @classmethod
    def vit_base(cls):
        return cls()

    @classmethod
    def vit_large(cls):
        return cls(lr=4e-4)

    @classmethod
    def vit_huge(cls):
        return cls(lr=3e-4)

    @classmethod
    def resnet152(cls):
        return cls(lr=6e-4, model_name="ResNet152")


@dataclass
class DataConfig:
    batch_size: int = 256  # Original: 4096
    num_workers: int = 4
    img_size: int = 224
    num_classes: int = 10  # CIFAR10
    pin_memory: bool = True
    debug: bool = field(default=False)

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
