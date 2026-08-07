"""Neural modules for the TIAE (Topological Intent-Action Embedding) framework."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # Compute the statistic in fp32 so bf16 training remains numerically valid.
        proj32 = proj.float()
        directions = torch.randn(
            proj32.size(-1), self.num_proj, device=proj.device, dtype=torch.float32
        )
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-8)
        x_t = (proj32 @ directions).unsqueeze(-1) * self.t
        error = (
            (x_t.cos().mean(-3) - self.phi).square()
            + x_t.sin().mean(-3).square()
        )
        statistic = (error @ self.weights) * proj32.size(-2)
        return statistic.mean().to(dtype=proj.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if inner_dim != dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(value, "b t (h d) -> b h t d", h=self.heads) for value in qkv
        )
        dropout = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout, is_causal=causal
        )
        return self.to_out(rearrange(output, "b h t d -> b t (h d)"))


class ConditionalBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            condition
        ).chunk(6, dim=-1)
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.condition_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        condition = self.condition_proj(condition)
        for layer in self.layers:
            x = layer(x, condition)
        return self.output_proj(self.norm(x))


class Embedder(nn.Module):
    """Embed one flattened action block at each latent time step."""

    def __init__(
        self,
        input_dim: int = 10,
        smoothed_dim: int | None = None,
        emb_dim: int = 192,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        smoothed_dim = smoothed_dim or input_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x.float().permute(0, 2, 1)).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ) -> None:
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntentActionActor(nn.Module):
    """Shared local/goal action law with an explicit TIAE feature grammar.

    ``five_slot`` uses ``[z_t, m_t, 0, z_t*m_t, A(a_{t-1})]``. ``four_slot``
    removes only the constant-zero slot. Both local and goal intents use this
    same module and differ only in how ``m_t`` is constructed — the topology
    of the intent manifold is preserved in the induced action laws.
    """

    VALID_FEATURE_LAYOUTS = {"five_slot", "four_slot"}

    def __init__(
        self,
        *,
        embed_dim: int,
        action_emb_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        depth: int = 3,
        dropout: float = 0.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        feature_layout: str = "four_slot",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("IntentActionActor depth must be positive")
        self.embed_dim = embed_dim
        self.action_emb_dim = action_emb_dim
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        if feature_layout not in self.VALID_FEATURE_LAYOUTS:
            raise ValueError(
                f"feature_layout must be one of {sorted(self.VALID_FEATURE_LAYOUTS)}, "
                f"got {feature_layout}"
            )
        self.feature_layout = feature_layout
        latent_slots = 4 if feature_layout == "five_slot" else 3

        layers: list[nn.Module] = [
            nn.Linear(latent_slots * embed_dim + action_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 2 * action_dim))
        self.net = nn.Sequential(*layers)

    def actor_features(
        self, z: torch.Tensor, intent: torch.Tensor, prev_act_emb: torch.Tensor
    ) -> torch.Tensor:
        if z.shape != intent.shape:
            raise ValueError(
                f"z and intent must have the same shape, got {z.shape} and {intent.shape}"
            )
        if self.feature_layout == "five_slot":
            return torch.cat(
                [z, intent, torch.zeros_like(z), z * intent, prev_act_emb], dim=-1
            )
        return torch.cat([z, intent, z * intent, prev_act_emb], dim=-1)

    def forward(
        self, z: torch.Tensor, intent: torch.Tensor, prev_act_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(self.actor_features(z, intent, prev_act_emb))
        mean, log_std = params.chunk(2, dim=-1)
        return mean, log_std.clamp(self.min_log_std, self.max_log_std)

    def action_mean(
        self, z: torch.Tensor, intent: torch.Tensor, prev_act_emb: torch.Tensor
    ) -> torch.Tensor:
        return self(z, intent, prev_act_emb)[0]


class ARPredictor(nn.Module):
    """Autoregressive action-conditioned latent predictor from LeWM."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ConditionalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.pos_embedding.size(1):
            raise ValueError(
                f"Predictor received {length} frames, max is {self.pos_embedding.size(1)}"
            )
        return self.transformer(
            self.dropout(x + self.pos_embedding[:, :length]), condition
        )


# =============================================================================
# TIAE-specific modules: Topology Preservation and Dual-Channel Encoding
# =============================================================================


class TopologyPreservingLoss(nn.Module):
    """Preserve the distance structure between intent manifold and action manifold.

    Derived from the theory of manifold learning (ITT, Section 3), this loss
    ensures that pairs of intents that are close in the latent space induce
    action laws that are close in the action space — and conversely, distant
    intents map to distinct action distributions.

    L_topo =  Σ_{i,j}  ( d_intent(i,j) — d_action(μ_i, μ_j) )²
    """

    def __init__(
        self,
        *,
        distance_metric: str = "cosine",
        margin: float = 0.0,
        subset_size: int = 256,
    ) -> None:
        super().__init__()
        if distance_metric not in {"cosine", "euclidean", "mahalanobis"}:
            raise ValueError(
                f"distance_metric must be one of cosine/euclidean/mahalanobis, "
                f"got {distance_metric}"
            )
        self.distance_metric = distance_metric
        self.margin = float(margin)
        self.subset_size = int(subset_size)

    @staticmethod
    def _pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        return 1.0 - x_norm @ x_norm.T

    @staticmethod
    def _pairwise_euclidean(x: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, x, p=2)

    def _pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "cosine":
            return self._pairwise_cosine(x)
        if self.distance_metric == "euclidean":
            return self._pairwise_euclidean(x)
        return self._pairwise_euclidean(x)

    def forward(
        self,
        intent_latents: torch.Tensor,
        action_means: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the topology-preserving loss.

        Args:
            intent_latents: (B, D) — latent intent representations
            action_means:  (B, A) — predicted action means for each intent

        Returns:
            scalar topology preservation loss
        """
        B = intent_latents.size(0)
        if B < 2:
            return intent_latents.new_zeros(())

        # Use a random subset for computational efficiency at scale
        if B > self.subset_size:
            indices = torch.randperm(B, device=intent_latents.device)[: self.subset_size]
            intent_latents = intent_latents[indices]
            action_means = action_means[indices]
            B = self.subset_size

        D_intent = self._pairwise_distance(intent_latents)
        D_action = self._pairwise_distance(action_means)

        # Normalize to [0, 1] so the loss is scale-invariant
        intent_max = D_intent.max().detach().clamp_min(1e-8)
        action_max = D_action.max().detach().clamp_min(1e-8)

        D_intent_norm = D_intent / intent_max
        D_action_norm = D_action / action_max

        # Mask diagonal (self-distance)
        mask = 1.0 - torch.eye(B, device=intent_latents.device, dtype=D_intent.dtype)
        diff = (D_intent_norm - D_action_norm) * mask

        loss = (diff.square()).sum() / max(mask.sum(), 1.0)
        return loss


class DualChannelEncoder(nn.Module):
    """Dual-channel encoding: state channel and intent channel with cross-attention.

    Separates the state representation from the intent representation, then fuses
    them via a lightweight cross-attention module. This ensures the topology of the
    intent space is preserved independently before being combined with state context.
    """

    def __init__(
        self,
        state_dim: int,
        intent_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.intent_proj = nn.Sequential(
            nn.Linear(intent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        state_features: torch.Tensor,
        intent_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse state and intent features with topology-aware dual-channel encoding.

        Args:
            state_features: (B, D_s) — state representation
            intent_features: (B, D_i) — intent representation

        Returns:
            fused: (B, hidden_dim) — fused state-intent representation
        """
        state = self.state_proj(state_features)
        intent = self.intent_proj(intent_features)

        # Unsqueeze for attention: (B, 1, D)
        state_attn = state.unsqueeze(1)
        intent_attn = intent.unsqueeze(1)

        # Intent attends to state context
        fused_attn, _ = self.cross_attn(
            query=intent_attn,
            key=state_attn,
            value=state_attn,
        )
        fused_attn = fused_attn.squeeze(1)

        # Concatenate and project
        fused = self.fusion(torch.cat([state, fused_attn], dim=-1))
        return fused


class IntrinsicDimensionEstimator(nn.Module):
    """Estimate intrinsic dimensionality of learned manifolds.

    Used as a diagnostic to monitor whether the topology-preserving loss
    effectively captures the manifold structure of the intent-action mapping.
    """

    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> float:
        """Estimate intrinsic dimension via the MLE method.

        Args:
            x: (B, D) — batch of representations

        Returns:
            estimated intrinsic dimension
        """
        if x.size(0) < self.k + 1:
            return float(x.size(-1))
        distances = torch.cdist(x, x, p=2)
        distances.fill_diagonal_(float("inf"))
        # (B, k) — distances to k nearest neighbors for each point
        k_nearest = distances.topk(self.k, largest=False, dim=-1).values
        # MLE intrinsic dimension estimator (Levina & Bickel, 2004)
        # dim = 1 / (mean over i of [1/(k-1) Σ_j log(T_{k}(i) / T_j(i))])
        with torch.no_grad():
            T_k = k_nearest[:, -1:]  # (B, 1) — distance to k-th neighbor
            T_j = k_nearest[:, :-1]  # (B, k-1) — distances to 1..(k-1) neighbors
            ratios = T_k / T_j.clamp_min(1e-10)  # (B, k-1)
            dim_est = 1.0 / (
                torch.log(ratios).mean() / max(self.k - 1, 1)
                + 1e-8
            )
        return float(dim_est.item())
