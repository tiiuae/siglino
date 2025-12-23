# MoE (Mixture of Experts) implementation for Falcon Vision
# Simplified from torchtitan's MoE for standalone use

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True
    top_k: int = 1
    use_grouped_mm: bool = False  # disabled by default for compatibility
    activation: Literal["silu", "relu2"] = "silu"


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "silu"):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "relu2":
            return self.w2(2 * F.relu(self.w1(x)).square() * self.w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.w2.weight)


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    act: str = "silu",
) -> torch.Tensor:
    num_tokens_list = num_tokens_per_expert.to(torch.int32).tolist()
    total_tokens = sum(num_tokens_list)    
    num_padding = x.shape[0] - total_tokens
    x_splits = torch.split(x[:total_tokens], split_size_or_sections=num_tokens_list, dim=0)
    out_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        if act == "relu2":
            h = 2 * F.relu(torch.matmul(x_expert, w1[expert_idx].T)).square()
        else:
            h = F.silu(torch.matmul(x_expert, w1[expert_idx].T))
        h = h * torch.matmul(x_expert, w3[expert_idx].T)
        h = torch.matmul(h, w2[expert_idx].T)
        out_splits.append(h)
    
    out = torch.cat(out_splits, dim=0)
    if num_padding > 0:
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


class GroupedExperts(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int, activation: str = "silu"):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.activation = activation

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        return _run_experts_for_loop(
            self.w1, self.w2, self.w3, x, num_tokens_per_expert, self.activation
        )

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.zeros_(self.w2)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: str = "sigmoid",
        route_norm: bool = False,
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

    def forward(self, x: torch.Tensor, expert_bias: torch.Tensor | None = None):
        scores = self.gate(x)
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.float())
        else:
            scores = F.softmax(scores.float(), dim=1)

        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(scores + expert_bias, k=self.top_k, dim=1)
        else:
            _, selected_experts_indices = torch.topk(scores, k=self.top_k, dim=1)

        top_scores = scores.gather(dim=1, index=selected_experts_indices)
        if self.route_norm:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()
        num_experts = moe_args.num_experts

        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            activation=moe_args.activation,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts, activation=moe_args.activation)
            if moe_args.num_shared_experts > 0
            else None
        )
        self.score_before_experts = moe_args.score_before_experts
        self.top_k = moe_args.top_k
        
        # Register buffer for load balancing (matches torchtitan checkpoint)
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x, expert_bias=self.expert_bias)

        # Reorder tokens by expert
        token_indices_sorted = torch.argsort(selected_experts_indices.view(-1), stable=True)
        top_scores_sorted = top_scores.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.top_k

        token_indices_expanded = token_indices_sorted.view(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_expanded)

        if self.score_before_experts:
            routed_input = (routed_input.float() * top_scores_sorted.view(-1, 1)).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        routed_output = (
            routed_output.to(torch.float32)
            * top_scores_sorted.view(-1, 1)
        ).to(x.dtype)

        out = out.scatter_add(dim=0, index=token_indices_expanded, src=routed_output)
        return out.view(bs, slen, dim)

    def init_weights(self, init_std: float, buffer_device: torch.device = None):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

