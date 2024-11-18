from dataclasses import dataclass, field


@dataclass
class ViTConfig:

    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10  # CIFAR10
    embed_dim: int = 192
    depth: int = 9
    num_heads: int = 12
    mlp_ratio: float = 2.0
    dropout: float = 0.0

    @classmethod
    def base(cls):
        return cls()

    # TODO: Add more configurations
    # @classmethod
    # def base(cls):
    #     return cls(embed_dim=768, depth=24, num_heads=12)  # 86M params

    # @classmethod
    # def large(cls):
    #     return cls(embed_dim=1024, depth=24, num_heads=16)  # 307M params

    # @classmethod
    # def huge(cls):
    #     return cls(embed_dim=1280, depth=32, num_heads=16)  # 632M params


@dataclass
class TrainingConfig:
    epochs: int = 1000  # Original: 7
    lr: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 42
    model_name: str = "ViT"
    warmup_epochs: int = 20
    patience: int = 20

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
        return cls(lr=1e-3, model_name="ResNet152")

    @classmethod
    def resnet50(cls):
        return cls(lr=1e-3, model_name="ResNet50")


@dataclass
class DataConfig:
    batch_size: int = 4096  # Original: 4096
    num_workers: int = 4
    img_size: int = 32
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
