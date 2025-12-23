# RoPE (Rotary Position Embedding) implementation for Falcon Vision
# Includes 1D RoPE and 2D Golden RoPE for vision tokens

import torch
import einops as E


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute frequency tensor for 1D rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[:seqlen]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    pos_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    if freqs_cis.ndim == 3:
        freqs_cis = freqs_cis.unsqueeze(-2)
    elif pos_t is not None:
        freqs_cis = freqs_cis[pos_t.long()].unsqueeze(-2)
    else:
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_3d_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    freqs_cis_2d: torch.Tensor | None,
    pos_hw: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D rotary embeddings (1D temporal + 2D spatial)."""
    xq_t, xq_hw = xq.chunk(chunks=2, dim=-1)
    xk_t, xk_hw = xk.chunk(chunks=2, dim=-1)

    pos_t = pos_hw[:, :, 0].clone() if pos_hw is not None else None
    xq_t, xk_t = apply_rotary_emb(xq_t, xk_t, freqs_cis, pos_t=pos_t)
    
    if freqs_cis_2d is not None and pos_hw is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, freqs_cis_2d, pos_hw[..., 1:])
        xk_hw = apply_golden_rotary_emb(xk_hw, freqs_cis_2d, pos_hw[..., 1:])

    xq_out = torch.concat([xq_t, xq_hw], dim=-1).type_as(xq)
    xk_out = torch.concat([xk_t, xk_hw], dim=-1).type_as(xk)
    return xq_out, xk_out


def _phi(m: int) -> float:
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x


def make_directions(n: int, d: int) -> torch.Tensor:
    g = _phi(d)
    alpha = (1.0 / g) ** torch.arange(1, d + 1, dtype=torch.float64)
    i = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1)
    z = torch.fmod(i * alpha, 1.0)
    directions = torch.erfinv(2.0 * z - 1.0)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return directions.float()


def precompute_golden_freqs_cis(
    n_heads: int,
    head_dim: int,
    min_freq: float,
    max_freq: float,
    pos_dim: int = 2,
    p_zero_freqs: float = 0.0,
) -> torch.Tensor:
    """Precompute golden ratio based 2D frequencies for vision tokens."""
    n_freqs = head_dim // 2
    n_zero_freqs = round(p_zero_freqs * n_freqs)
    omega_F = torch.cat((
        torch.zeros(n_zero_freqs),
        min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
    ))
    directions_hFP = make_directions(n_heads * n_freqs, pos_dim).reshape(n_heads, n_freqs, pos_dim)
    return directions_hFP * omega_F.reshape(n_freqs, 1)


def apply_golden_freqs_cis_to_visual_pos(freqs_hFP: torch.Tensor, pos_BSP: torch.Tensor) -> torch.Tensor:
    """Apply golden frequencies to visual positions."""
    img_mask_BS = E.reduce(~torch.isnan(pos_BSP), 'b s p -> b s', reduction='all')
    pos_tP = pos_BSP[img_mask_BS].float()
    theta_thF = torch.einsum('tp,hfp->thf', pos_tP, freqs_hFP)
    return torch.polar(torch.ones_like(theta_thF), theta_thF)


def apply_golden_rotary_emb(
    input_BShd: torch.Tensor,
    freqs_cis_thF: torch.Tensor,
    pos_BSP: torch.Tensor,
) -> torch.Tensor:
    """Apply golden rotary embedding to image tokens only."""
    img_mask_BS = E.reduce(~torch.isnan(pos_BSP), 'b s p -> b s', reduction='all')
    input_thd = input_BShd[img_mask_BS]

    input_thd = torch.view_as_complex(
        E.rearrange(input_thd.float(), "t h (d two) -> t h d two", two=2)
    )
    output_thd = input_thd * freqs_cis_thF
    output_thd = torch.view_as_real(output_thd).flatten(-2).type_as(input_BShd)

    img_mask_BS11 = E.rearrange(img_mask_BS, 'b s -> b s 1 1')
    return input_BShd.masked_scatter(img_mask_BS11, output_thd)

