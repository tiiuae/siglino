# Model configurations for Falcon Vision
# Pre-defined architectures for different model sizes

from dataclasses import dataclass, field
from .moe import MoEArgs


@dataclass
class AMOEArgs:
    """Configuration for Falcon Vision Encoder."""
    dim: int = 768
    n_layers: int = 18
    n_heads: int = 12
    head_dim: int | None = 128
    n_kv_heads: int | None = 4
    
    # MoE configuration
    moe_dim: int = 768
    moe_args: MoEArgs = field(default_factory=lambda: MoEArgs(
        num_experts=16,
        num_shared_experts=1,
        top_k=3,
        score_before_experts=False,
        route_norm=True,
        route_scale=0.8633,
        activation="relu2",
    ))

    # Dense FFN configuration
    first_n_layers_dense: int = 0  # number of initial layers using dense FFN instead of MoE
    ffn_dim: int | None = None  # hidden dim for dense FFN layers (None = use moe_dim)
    activation: str = "silu"  # activation for dense FFN layers
    
    # Vision settings
    channel_size: int = 3
    spatial_patch_size: int = 16
    temporal_patch_size: int = 1
    
    # RoPE settings
    enable_3d_rope: bool = True
    rope_theta: float = 100000.0
    rope_min_freqs: float = 1.0
    rope_max_freqs: float = 20.0
    max_seq_len: int = 8192
    
    # Normalization
    norm_eps: float = 1e-5
    use_qk_norm: bool = True
    use_tok_norm: bool = True
    parameterized_norm: bool = True  # True = nn.RMSNorm (learnable), False = F.rms_norm (no params)
    
    # Distillation settings
    n_storage_tokens: int = 4  # number of register tokens
    teachers: tuple[str, ...] = ("siglip2", "dinov3")
    teachers_dim: tuple[int, ...] = (1152, 1024)
    
    # FlexAttention
    use_flex_attn: bool = True


# Pre-defined configurations
amoe_configs = {
    "18-layers-distillation": AMOEArgs(
        n_layers=18,
        n_heads=8,
        head_dim=128,
        n_kv_heads=8,
        dim=768,
        moe_dim=384,
        moe_args=MoEArgs(
            num_experts=28,
            num_shared_experts=0,
            top_k=6,
            score_before_experts=False,
        ),
        spatial_patch_size=16,
        enable_3d_rope=True,
        use_qk_norm=True,
    ),

    "ultrasparse": AMOEArgs(
        n_layers=18,
        n_heads=8,
        head_dim=128,
        n_kv_heads=8,
        dim=768,
        moe_dim=384,
        moe_args=MoEArgs(
            num_experts=28,
            num_shared_experts=0,
            top_k=2,
            score_before_experts=False,
        ),
        spatial_patch_size=16,
        enable_3d_rope=True,
        use_qk_norm=True,
    ),

    "dense-L": AMOEArgs(
        n_layers=18,
        n_heads=16,
        head_dim=80,
        n_kv_heads=16,
        dim=1280,
        moe_dim=384,
        moe_args=MoEArgs(
            num_experts=28,
            num_shared_experts=0,
            top_k=6,
            score_before_experts=False,
        ),
        first_n_layers_dense=18,
        ffn_dim=5120,
        spatial_patch_size=16,
        enable_3d_rope=True,
        use_qk_norm=True,
    ),

    "dense-XS": AMOEArgs(
        n_layers=12,
        n_heads=6,
        head_dim=64,
        n_kv_heads=6,
        dim=384,
        moe_dim=0,
        moe_args=MoEArgs(
            num_experts=1,
            num_shared_experts=0,
            top_k=1,
            score_before_experts=False,
        ),
        first_n_layers_dense=12,
        ffn_dim=1536,
        activation="silu",
        spatial_patch_size=16,
        enable_3d_rope=True,
        use_qk_norm=True,
        parameterized_norm=False,
    ),

    "dense-S": AMOEArgs(
        n_layers=12,
        n_heads=8,
        head_dim=64,
        n_kv_heads=8,
        dim=512,
        moe_dim=0,
        moe_args=MoEArgs(
            num_experts=1,
            num_shared_experts=0,
            top_k=1,
            score_before_experts=False,
        ),
        first_n_layers_dense=12,
        ffn_dim=2048,
        activation="silu",
        spatial_patch_size=16,
        enable_3d_rope=True,
        use_qk_norm=True,
        parameterized_norm=False,
    ),

}

