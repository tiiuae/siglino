# Attention module for Falcon Vision
# Supports FlexAttention for efficient vision-only attention patterns

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    AuxRequest,
    create_block_mask,
    flex_attention,
)
import einops as E

from .rope import apply_rotary_emb, apply_3d_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match query heads."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x.unsqueeze(3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class FlexAttentionWrapper(nn.Module):
    """Wrapper for flex_attention with optional compilation and aux outputs."""

    _compiled = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
        compile: bool = True,
        return_aux: bool = False,
    ):
        # Choose compiled or eager function
        fn = flex_attention
        if compile:
            if FlexAttentionWrapper._compiled is None:
                FlexAttentionWrapper._compiled = torch.compile(
                    flex_attention,
                    mode="max-autotune-no-cudagraphs",
                )
            fn = FlexAttentionWrapper._compiled

        if return_aux:
            # Request log-sum-exp aux for sink attention
            return fn(q, k, v, block_mask=block_mask, return_aux=AuxRequest(lse=True))
        else:
            return fn(q, k, v, block_mask=block_mask)


class SDPAttentionWrapper(nn.Module):
    """Fallback SDPA attention when flex_attention is not available."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        use_qk_norm: bool = False,
        enable_3d_rope: bool = False,
        use_flex_attn: bool = True,
        use_sink_attn: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim or dim // n_heads
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.wq = nn.Linear(dim, self.q_dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.kv_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.use_qk_norm = use_qk_norm
        self.enable_3d_rope = enable_3d_rope
        self.use_flex_attn = use_flex_attn

        self.sink_attn = use_sink_attn
        if self.sink_attn:
            self.sinks = nn.Parameter(torch.empty(n_heads))

        self.inner_attention = FlexAttentionWrapper()

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.wo.weight)
        if self.sink_attn:
            nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_2d: torch.Tensor | None = None,
        pos_thw: torch.Tensor | None = None,
        attention_masks: BlockMask | torch.Tensor | None = None,
        compile: bool = True,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.use_qk_norm:
            xq = F.rms_norm(xq, (xq.size(-1),))
            xk = F.rms_norm(xk, (xk.size(-1),))

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_thw)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output, aux = self.inner_attention(
            xq,
            xk,
            xv,
            block_mask=attention_masks,
            compile=compile,
            return_aux=True,
        )
        # aux.lse: (B, H, S) log-sum-exp per head & position
        sinks_BHL = E.rearrange(self.sinks, "h -> 1 h 1")
        sink_scale = torch.sigmoid(aux.lse - sinks_BHL)
        output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)


        output = E.rearrange(output, "b h s d -> b s (h d)").contiguous()
        return self.wo(output)


def create_attention_mask(
    mask_mod,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    BLOCK_SIZE: tuple[int, int] = (64, 64),
) -> BlockMask:
    """Create a BlockMask for flex_attention."""
    return create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        BLOCK_SIZE=BLOCK_SIZE,
    )