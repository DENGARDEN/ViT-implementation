# Description: Vision Transformer (ViT) model implementation.
#   - Original paper: https://arxiv.org/abs/2010.11929
# Specific requirements:
#   - without relying on pre-implemented transformer or self-attention modules such as `torch.nn.Transformer`` or `torch.nn.MultiheadAttention`.
#   - That is, you should implement the self-attention mechanism and the transformer block using lower-level `torch.nn`` modules.


import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used in the Vision Transformer (ViT) model.
    "The MLP contains two layers with a GELU non-linearity." [Dosovitskiy et al., 2021, p. 4]

    Args:
        embed_dim (int): The input embedding dimension.
        mlp_ratio (float, optional): The ratio of the hidden dimension to the input embedding
            dimension. Default is 4.0. (Base: 768 -> 3072, Large: 1024 -> 4096,
            Huge: 1280 -> 5120)
        dropout (float, optional): The dropout probability. Default is 0.0.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        act (nn.GELU): The activation function (GELU).
        dropout (nn.Dropout): The dropout layer.

    Methods:
        _init_weights: Initializes the weights of the fully connected layers.
        forward: Performs the forward pass of the MLP block.
    """

    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the fully connected layers using Xavier initialization for weights
        and normal distribution with a small standard deviation for biases.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        """
        Performs the forward pass of the MLP block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):
    """
    Multi-head Self Attention mechanism.

    Mathematical formulation:
    - Query, Key, Value: Q, K, V = xU_q, xU_k, xU_v
    - Attention weights: A = softmax(QK^T/√d_k)
    - Output: O = AV

    Shape transformations:
    - input: (B, N, D)  # batch, sequence length, embedding dim
    - q,k,v: (B, k, N, D/k)  # k: num_heads
    - attn: (B, k, N, N)  # attention weights
    - out: (B, N, D)  # final output

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.0.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        scale (float): The scaling factor for the attention scores.
        qkv (nn.Linear): Linear layer for computing the query, key, and value projections.
        proj (nn.Linear): Linear layer for projecting the combined attention heads.
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(x): Performs the forward pass of the attention block.

    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # D_h = D/k
        self.scale = self.head_dim**-0.5  # D_h^-0.5

        # Parmeter sharing: U_q, U_k, U_v share the same weight matrix
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # D -> 3 * D; U_qkv = D x 3D; D = D_h * k
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)

        # Bias initialization
        nn.init.normal_(self.qkv.bias, std=1e-6)
        nn.init.normal_(self.proj.bias, std=1e-6)

    def forward(self, x):
        """
        Performs the forward pass of the attention block.

        Args:
            x (torch.Tensor): The input tensor of shape (B, N, C), where B is the batch size,
                N is the sequence length, and C is the input dimension.

        Returns:
            torch.Tensor: The output tensor of shape (B, N, C), where B is the batch size,
                N is the sequence length, and C is the output dimension.
                The representation of each token in the sequence is updated based on the attention scores.

        """

        # Eq. 5 - 8: Mutlihead self-attention (MSA)
        B, N, C = x.shape  # B: batch size, N: sequence length, C: input dimension (D)

        # Eq. 5: Generate q, k, v
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )  # 3 x (B x k x N x D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Eq. 6: Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B x k x N x N)
        attn = attn.softmax(dim=-1)  # (B x k x N x N)
        attn = self.dropout(attn)

        # Eq. 7 - 8: Weight the values and project concatenated attention heads
        x = (
            (attn @ v).transpose(1, 2).reshape(B, N, C)
        )  # (B x k x N x D_h) -> (B x N x k x D_h) -> (B x N x D); k x D_h = D
        x = self.proj(x)
        x = self.dropout(x)
        return x  # Multihead self-attention (MSA)


class TransformerEncoderBlock(nn.Module):
    """
    TransformerEncoderBlock class represents a single block in the Transformer model.

    Args:
        embed_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        mlp_ratio (float, optional): The ratio of the hidden size of the MLP layer to the input embeddings dimension. Defaults to 4.0.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = AttentionBlock(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLPBlock(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        """
        Forward pass of the TransformerEncoderBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x + self.attn(self.ln1(x))  # Eq. 2: x + MSA, (B x N x D)
        x = x + self.mlp(self.ln2(x))  # Eq. 3: x + MLP,
        return x


class HybridEmbed(nn.Module):
    # TODO: Use this class if PatchEmbed is not meeting the expectations

    """
    Hybrid architecture using CNN feature map as patch embedding

    Usage:
        ```
        backbone = torchvision.models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        embed_layer = HybridEmbedding(backbone)
        ```
    """

    def __init__(self, backbone, img_size=256, feature_size=2048, embed_dim=768):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(feature_size, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """
    PatchEmbed (Linear Projection) module for Vision Transformer (ViT) model.

    Args:
        img_size (int): The size of the input image (default: 256).
        patch_size (int): The size of each patch (default: 16).
        in_channels (int): The number of input channels (default: 3).
        embed_dim (int): The dimension (*D*) of the embedded patches (default: 768).
    """

    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size  # *H* or *W*
        self.patch_size = patch_size  # *P*
        self.num_patches = (img_size // patch_size) ** 2  # *N*

        # Linear projection
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)  # P^2 * C -> D

        # Initialize weights for linear projection
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights and biases for the linear projection layer.

        Weights:
            Uses Xavier/Glorot uniform initialization (nn.init.xavier_uniform_) which:
            - Maintains consistent variance of activations across layers
            - Prevents vanishing/exploding gradients
            - Works well with tanh/sigmoid activations
            - Enables faster training convergence

        Biases:
            Uses zero initialization (nn.init.zeros_) which:
            - Provides a neutral starting point
            - Lets weights learn the main transformations
            - Keeps initial network behavior simple
        """
        nn.init.xavier_uniform_(self.proj.weight)  # (Glorot and Bengio, 2010)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size and W == self.img_size  # Ensure input image size square (H = W)

        # Flatten patches
        x = x.reshape(
            B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size
        )  # B, C, H/P, P, W/P, P
        x = x.permute(
            0, 2, 4, 1, 3, 5
        ).contiguous()  # B, H, W, C, P, P, contiguous() to ensure memory is contiguous
        x = x.reshape(B, self.num_patches, -1)  # B, N, P^2 * C

        # Project patches
        x = self.proj(x)  # B, N, D
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model for image classification.
    Default configuration: ViT-Base

    Args:
        img_size (int): The size of the input image (default: 256).
        patch_size (int): The size of each patch (default: 16).
        in_channels (int): The number of input channels (default: 3).
        num_classes (int): The number of output classes (default: 10).
        embed_dim (int): The dimensionality of the patch embeddings (default: 768).
        depth (int): The number of transformer encoder blocks (default: 12).
        num_heads (int): The number of attention heads in each transformer encoder block (default: 12).
        mlp_ratio (float): The ratio of the hidden dimension size of the feed-forward network in each transformer encoder block (default: 4.0).
        dropout (float): The dropout probability (default: 0.0).
        embed_layer (nn.Module): The patch embedding layer (default: PatchEmbed).

    Attributes:
        patch_embed (nn.Module): The patch embedding layer.
        cls_token (nn.Parameter): The learnable class token embedding.
        pos_embed (nn.Parameter): The learnable positional embeddings.
        pos_dropout (nn.Dropout): The dropout layer for positional embeddings.
        transformer_encoder_blocks (nn.Sequential): The sequence of transformer encoder blocks.
        ln (nn.LayerNorm): The layer normalization layer.
        head (nn.Linear): The classification MLP head.

    Methods:
        forward_features(x): Computes the image representation without the classification head.
        forward(x): Computes the output logits for the input image.

    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        embed_layer=PatchEmbed,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Learnable class embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )  # Learnable positional embedding
        self.pos_dropout = nn.Dropout(dropout)  # Dropout

        # Transformer blocks
        self.transformer_encoder_blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # Final normalization and classification head
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)  # MLP head

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings and class token
        # The bounded range helps prevent extreme values that could cause unstable gradients early in training
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        """
        Computes the image representation without the classification head.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The image representation tensor.

        """
        B = x.shape[0]

        # Eq. 1: Flatten the patches and project them to *D* dimensions
        x = self.patch_embed(x)

        # Eq. 1: Prepend class token and add position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  # B, 1, D
        x = torch.cat((cls_tokens, x), dim=1)  # B, N + 1, D
        x = x + self.pos_embed
        x = self.pos_dropout(x)  # For regularization

        # Apply transformer blocks
        x = self.transformer_encoder_blocks(x)
        x = self.ln(x)  # Eq. 4: image representation

        return x[:, 0]  # Return only the class tokens

    def forward(self, x):
        """
        Computes the output logits for the input image.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output logits tensor.

        """
        x = self.forward_features(x)
        x = self.head(x)
        return x
