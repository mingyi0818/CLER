"""ARFusion-Rec v3: dual-score fusion + cross-view CL + CLER warm-start."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2,
             pos_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """InfoNCE with optional multi-positive mask.

    When ``pos_mask`` is ``None``, falls back to the diagonal-label variant
    (each position i in ``z1`` is positive with position i in ``z2``).
    When ``pos_mask`` is provided, it must be a ``[B, B]`` float tensor where
    ``pos_mask[i, j] = 1`` indicates that ``(z1[i], z2[j])`` is a positive
    pair. This correctly handles duplicate entities in a batch (same user or
    same item appearing multiple times) by treating them as additional
    positives instead of false negatives. Implementation follows the
    supervised contrastive loss (Khosla et al., NeurIPS 2020).
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    if pos_mask is None:
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(logits, labels)
    # Multi-positive: average -log(p) over each anchor's positives
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
    loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
    return loss.mean()


def embedding_dropout(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    if not x.requires_grad:
        return x
    mask = (torch.rand_like(x) > drop_prob).float()
    return x * mask / (1.0 - drop_prob)


class ARFusionModel(nn.Module):
    """
    Adaptive Reliability Fusion Recommender (ARFusion-Rec).

    Novelty vs CV-CLER / Multimodal CF:
    - Dual-stream *score* fusion with sparsity-aware reliability λ_u.
    - Neural-prior blended gate: λ = η·σ(MLP) + (1-η)·σ(α·log(1+c)−β).
    - Profile-anchored distillation on sparse users.
    - Optional LightGCN refinement of collaborative stream.
    - Cross-view contrastive alignment (profile ↔ behavior) during training.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        feat_dim: int,
        dim: int = 64,
        n_gnn_layers: int = 2,
        ui_temperature: float = 0.2,
        cv_temperature: float = 0.2,
        graph_mix_init: float = 0.2,
        use_graph: bool = True,
        score_mode: str = "dual",
        gate_mode: str = "full",
    ):
        super().__init__()
        self.dim = dim
        self.n_gnn_layers = n_gnn_layers
        self.ui_temperature = ui_temperature
        self.cv_temperature = cv_temperature
        self.use_graph = use_graph and n_gnn_layers > 0
        self.score_mode = score_mode
        self.gate_mode = gate_mode

        self.user_beh = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

        self.profile_encoder = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

        self.reliability_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

        self.alpha_prior = nn.Parameter(torch.tensor(2.0))
        self.beta_prior = nn.Parameter(torch.tensor(1.6))
        self.eta_blend = nn.Parameter(torch.tensor(0.4))

        if self.use_graph:
            self.graph_mix = nn.Parameter(torch.tensor(graph_mix_init))
        else:
            self.graph_mix = None

        self.profile_projector = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.behavior_projector = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.ui_projector = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        nn.init.normal_(self.user_beh.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.register_buffer("user_features", torch.zeros(n_users, feat_dim))
        self.register_buffer("user_log_counts", torch.zeros(n_users, 1))

        self.adj: Optional[torch.Tensor] = None
        self._cached_user_graph: Optional[torch.Tensor] = None
        self._cached_item_graph: Optional[torch.Tensor] = None

    def set_user_features(self, features: np.ndarray) -> None:
        device = self.user_beh.weight.device
        self.user_features = torch.tensor(features, dtype=torch.float32, device=device)

    def set_user_log_counts(self, counts: np.ndarray) -> None:
        device = self.user_beh.weight.device
        log_c = np.log1p(np.asarray(counts, dtype=np.float32)).reshape(-1, 1)
        self.user_log_counts = torch.tensor(log_c, dtype=torch.float32, device=device)

    def set_adj(self, adj: torch.Tensor) -> None:
        self.adj = adj if self.use_graph else None

    def clear_cache(self) -> None:
        self._cached_user_graph = None
        self._cached_item_graph = None

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synchronous bipartite graph propagation (standard LightGCN).

        Both user and item updates at layer k+1 use embeddings from layer k:
            u^{(k+1)} = A @ v^{(k)}
            v^{(k+1)} = A^T @ u^{(k)}   (NOT u^{(k+1)})
        Then assign simultaneously. Previous sequential update incorrectly
        produced v^{(k+1)} = A^T @ A @ v^{(k)}, over-smoothing item side.
        """
        assert self.adj is not None
        user_layers = [self.user_beh.weight]
        item_layers = [self.item_emb.weight]
        u, v = self.user_beh.weight, self.item_emb.weight
        for _ in range(self.n_gnn_layers):
            u_new = torch.sparse.mm(self.adj, v)
            v_new = torch.sparse.mm(self.adj.transpose(0, 1), u)
            u, v = u_new, v_new
            user_layers.append(u)
            item_layers.append(v)
        user_out = torch.stack(user_layers, dim=0).mean(dim=0)
        item_out = torch.stack(item_layers, dim=0).mean(dim=0)
        self._cached_user_graph = user_out
        self._cached_item_graph = item_out
        return user_out, item_out

    def _graph_user(self, users: torch.Tensor) -> torch.Tensor:
        if self._cached_user_graph is None:
            self.propagate()
        return self._cached_user_graph[users]

    def _graph_item(self, items: torch.Tensor) -> torch.Tensor:
        if self._cached_item_graph is None:
            self.propagate()
        return self._cached_item_graph[items]

    def reliability(self, users: torch.Tensor) -> torch.Tensor:
        lam_prior = torch.sigmoid(self.alpha_prior * self.user_log_counts[users] - self.beta_prior)
        if self.gate_mode == "prior_only":
            return lam_prior
        h_beh = self.user_beh(users)
        h_prof = self.profile_encoder(self.user_features[users])
        gate_in = torch.cat([h_beh, h_prof, self.user_log_counts[users]], dim=-1)
        lam_nn = torch.sigmoid(self.reliability_mlp(gate_in))
        if self.gate_mode == "mlp_only":
            return lam_nn
        eta = torch.sigmoid(self.eta_blend)
        return eta * lam_nn + (1.0 - eta) * lam_prior

    def behavior_repr(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_beh(users)

    def collaborative_repr(self, users: torch.Tensor) -> torch.Tensor:
        h_beh = self.user_beh(users)
        if not self.use_graph or self.adj is None:
            return h_beh
        h_graph = self._graph_user(users)
        mix = torch.sigmoid(self.graph_mix)
        return (1.0 - mix) * h_beh + mix * h_graph

    def profile_repr(self, users: torch.Tensor) -> torch.Tensor:
        return self.profile_encoder(self.user_features[users])

    def item_collab(self, items: torch.Tensor) -> torch.Tensor:
        v = self.item_emb(items)
        if not self.use_graph or self.adj is None:
            return v
        mix = torch.sigmoid(self.graph_mix)
        return (1.0 - mix) * v + mix * self._graph_item(items)

    def _score_collab(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.collaborative_repr(users) * self.item_collab(items)).sum(dim=1)

    def _score_profile(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.profile_repr(users) * self.item_emb(items)).sum(dim=1)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if self.score_mode == "collab":
            return self._score_collab(users, items)
        if self.score_mode == "additive":
            lam = self.reliability(users).squeeze(-1)
            u = self.collaborative_repr(users) + (1.0 - lam).unsqueeze(-1) * self.profile_repr(users)
            return (u * self.item_emb(items)).sum(dim=1)
        lam = self.reliability(users).squeeze(-1)
        return lam * self._score_collab(users, items) + (1.0 - lam) * self._score_profile(users, items)

    def all_scores(self, user_id: int) -> torch.Tensor:
        device = self.user_beh.weight.device
        users = torch.tensor([user_id], device=device)
        items = torch.arange(self.item_emb.num_embeddings, device=device)
        if self.score_mode == "collab":
            u = self.collaborative_repr(users)[0]
            v = self.item_collab(items)
            return v @ u
        if self.score_mode == "additive":
            lam = self.reliability(users).squeeze(-1)
            u = self.collaborative_repr(users)[0] + (1.0 - lam) * self.profile_repr(users)[0]
            return self.item_emb.weight @ u
        lam = self.reliability(users).squeeze(-1)
        s_c = self._score_collab(users, items)
        s_p = self._score_profile(users, items)
        return lam * s_c + (1.0 - lam) * s_p

    def cross_view_loss(self, users: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        z_form = self.profile_projector(embedding_dropout(self.profile_repr(users), drop_prob))
        z_beh = self.behavior_projector(embedding_dropout(self.behavior_repr(users), drop_prob))
        # Multi-positive mask: same user at different batch positions is a positive
        # pair (not a false negative). Includes the diagonal (user with itself).
        pos_mask = (users.unsqueeze(0) == users.unsqueeze(1)).float()
        return 0.5 * (
            info_nce(z_form, z_beh, self.cv_temperature, pos_mask=pos_mask)
            + info_nce(z_beh, z_form, self.cv_temperature, pos_mask=pos_mask.T)
        )

    def profile_distillation_loss(self, users: torch.Tensor) -> torch.Tensor:
        lam = self.reliability(users).detach().squeeze(-1)
        h_collab = self.collaborative_repr(users)
        h_prof = self.profile_repr(users)
        cos = F.cosine_similarity(h_collab, h_prof, dim=1)
        weight = (1.0 - lam).clamp(min=0.0)
        if weight.sum() < 1e-6:
            return torch.tensor(0.0, device=users.device)
        return (weight * (1.0 - cos)).sum() / weight.sum()

    def fusion_loss(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> torch.Tensor:
        """BPR-style loss on fused scores so gate params receive gradients."""
        lam = self.reliability(users).squeeze(-1)
        s_c_pos = self._score_collab(users, pos_items)
        s_c_neg = self._score_collab(users, neg_items)
        s_p_pos = self._score_profile(users, pos_items)
        s_p_neg = self._score_profile(users, neg_items)
        fused_pos = lam * s_c_pos + (1.0 - lam) * s_p_pos
        fused_neg = lam * s_c_neg + (1.0 - lam) * s_p_neg
        return -torch.log(torch.sigmoid(fused_pos - fused_neg) + 1e-8).mean()

    def ui_contrastive_loss(self, users: torch.Tensor, items: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        u = self.collaborative_repr(users)
        v = self.item_collab(items)
        z1 = self.ui_projector(embedding_dropout(u, drop_prob))
        z2 = self.ui_projector(embedding_dropout(v, drop_prob))
        # Multi-positive mask: if user[i] == user[j], then item[j] is also a
        # positive for user[i] (same user, another observed positive item).
        pos_mask = (users.unsqueeze(0) == users.unsqueeze(1)).float()
        return info_nce(z1, z2, self.ui_temperature, pos_mask=pos_mask)


def warm_start_from_mmcf(arf: ARFusionModel, mmc: nn.Module) -> None:
    """Warm-start behavior/item branches from trained Multimodal CF."""
    arf.user_beh.load_state_dict(mmc.user_cf.state_dict())
    arf.item_emb.load_state_dict(mmc.item_emb.state_dict())


def warm_start_from_cler(arf: ARFusionModel, cler: nn.Module) -> None:
    arf.user_beh.load_state_dict(cler.user_emb.state_dict())
    arf.item_emb.load_state_dict(cler.item_emb.state_dict())
    arf.ui_projector.load_state_dict(cler.projection.state_dict())


def warm_start_from_mf(arf: ARFusionModel, mf: nn.Module) -> None:
    arf.user_beh.load_state_dict(mf.user_emb.state_dict())
    arf.item_emb.load_state_dict(mf.item_emb.state_dict())
