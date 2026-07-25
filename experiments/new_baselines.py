"""2023-2025 强基线模型实现：XSimGCL / BM3 / DiffRec。

本文件仅实现模型类（含前向、打分、辅助损失接口），不包含训练循环。
训练逻辑由 pipeline.train_model 或外部自定义循环调用以下接口完成：
    - score(users, items)              BPR 打分 [B]
    - all_scores_batch(user_ids)       批量评估 [B, M]
    - contrastive_loss / diffusion_loss 等辅助损失
    - set_adj(adj) / set_user_features(features) 资源注入

接口与 arfusion_recommender.pipeline 保持兼容：
    - evaluate_topk 优先使用 all_scores_batch
    - _prepare_model 会自动调用 set_user_features（若存在）
    - 图模型需手动调用 set_adj 注入邻接矩阵

参考文献：
    [1] XSimGCL: Yu et al., "XSimGCL: Towards Extremely Simple Graph
        Contrastive Learning for Recommendation", SIGIR 2023.
    [2] BM3: Zhou et al., "Bootstrap Latent Representations for Multi-modal
        Recommendation", WWW 2023.
    [3] DiffRec: Wang et al., "Diffusion Recommender Model", SIGIR 2023.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from arfusion_recommender.pipeline import build_bipartite_adj, bpr_loss  # noqa: E402
from arfusion_recommender.arfusion_model import info_nce, embedding_dropout  # noqa: E402


__all__ = ["XSimGCLModel", "BM3Model", "DiffRecModel"]


# =============================================================================
# 1. XSimGCL (SIGIR 2023)
# =============================================================================
class XSimGCLModel(nn.Module):
    """XSimGCL：极简图对比学习推荐模型。

    在 LightGCN 基础上仅对 item 嵌入施加随机高斯扰动，构造正样本对，
    用 InfoNCE 对比损失替代复杂的图 dropout/边扰动方案。
    - 无额外投影头、无 dropout，仅增加一个对比损失项。
    - 图传播采用同步二部更新（与 ARFusionModel 一致，避免 item 侧 over-smoothing）。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        n_layers: int = 2,
        cl_weight: float = 0.1,
        noise_scale: float = 0.1,
        ui_temperature: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.cl_weight = cl_weight
        self.noise_scale = noise_scale
        self.ui_temperature = ui_temperature

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # 邻接矩阵通过 set_adj 注入；非 buffer（与 ARFusionModel 保持一致），
        # set_adj 内部会同步到模型所在设备。
        self.adj: Optional[torch.Tensor] = None
        self._cached_user: Optional[torch.Tensor] = None
        self._cached_item: Optional[torch.Tensor] = None

    # ---- 资源注入 ----
    def set_adj(self, adj: torch.Tensor) -> None:
        """注入归一化二部邻接矩阵 [n_users, n_items]（稀疏）。"""
        device = self.user_emb.weight.device
        self.adj = adj.to(device)
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cached_user = None
        self._cached_item = None

    # ---- 图传播 ----
    def propagate(self, adj: Optional[torch.Tensor] = None) -> tuple:
        """LightGCN 风格的多层图传播，返回 (user_repr, item_repr)。

        采用同步二部更新：第 k+1 层的 user/item 都基于第 k 层嵌入计算，
        然后同时赋值，避免 v^{(k+1)} = A^T A v^{(k)} 的过度平滑问题。
        """
        if adj is None:
            adj = self.adj
        assert adj is not None, "XSimGCL: 邻接矩阵未设置，请先调用 set_adj(adj)。"

        user_layers = [self.user_emb.weight]
        item_layers = [self.item_emb.weight]
        u, v = self.user_emb.weight, self.item_emb.weight
        for _ in range(self.n_layers):
            u_new = torch.sparse.mm(adj, v)
            v_new = torch.sparse.mm(adj.transpose(0, 1), u)
            u, v = u_new, v_new
            user_layers.append(u)
            item_layers.append(v)
        # LightGCN 层均值聚合
        user_out = torch.stack(user_layers, dim=0).mean(dim=0)
        item_out = torch.stack(item_layers, dim=0).mean(dim=0)
        self._cached_user = user_out
        self._cached_item = item_out
        return user_out, item_out

    def _ensure_propagated(self) -> None:
        if self._cached_user is None or self._cached_item is None:
            self.propagate()

    def _user_repr(self, users: torch.Tensor) -> torch.Tensor:
        self._ensure_propagated()
        return self._cached_user[users]

    def _item_repr(self, items: torch.Tensor) -> torch.Tensor:
        self._ensure_propagated()
        return self._cached_item[items]

    # ---- 打分 ----
    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """BPR 打分：user 与 item 传播后嵌入的点积。"""
        u = self._user_repr(users)
        v = self._item_repr(items)
        return (u * v).sum(dim=1)

    def all_scores_batch(self, user_ids: List[int]) -> torch.Tensor:
        """批量评估：返回 [B, M] 的全物品打分矩阵。"""
        device = self.user_emb.weight.device
        users = torch.tensor(user_ids, dtype=torch.long, device=device)
        self._ensure_propagated()
        u = self._cached_user[users]          # [B, D]
        return u @ self._cached_item.T        # [B, M]

    # ---- 对比损失 ----
    def contrastive_loss(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """对 item 传播后嵌入加高斯扰动，构造正样本对做 InfoNCE。

        v       = 传播后的 item 嵌入（锚点）
        v_noisy = v + noise_scale * randn(v)  （正样本）
        批内其他 v_noisy 作为负样本。
        对批次内重复 item 使用多重正样本 mask，避免假负样本。
        """
        v = self._item_repr(items)
        noise = self.noise_scale * torch.randn_like(v)
        v_noisy = v + noise
        # 重复 item 互为正样本（含自身）
        pos_mask = (items.unsqueeze(0) == items.unsqueeze(1)).float()
        return info_nce(v, v_noisy, self.ui_temperature, pos_mask=pos_mask)


# =============================================================================
# 2. BM3 (WWW 2023)
# =============================================================================
class BM3Model(nn.Module):
    """BM3：极简多模态推荐模型。

    核心思想：不做重型模态编码，仅用线性投影 + dropout 处理 profile 特征，
    通过三重损失联合训练：
        L = L_BPR + cl_weight * L_modal_cl + align_weight * L_align
    - L_BPR:        协同过滤 BPR 排序损失
    - L_modal_cl:   对 user_emb 做 dropout 后 InfoNCE（模态内对比）
    - L_align:      user_emb 与 profile 投影的 cosine 对齐（跨模态对齐）
    user 表示 = user_emb + profile_proj(features)，与 item_emb 点积打分。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        feat_dim: int,
        dim: int = 64,
        cl_weight: float = 0.1,
        align_weight: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.cl_weight = cl_weight
        self.align_weight = align_weight
        self.dropout = dropout
        # 模态内对比损失温度（BM3 原文未显式参数化，使用常用默认值 0.2）
        self.cl_temperature = 0.2

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        # 极简 profile 投影：单层线性 + dropout（BM3 原文风格）
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, dim),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.register_buffer("user_features", torch.zeros(n_users, feat_dim))

    # ---- 资源注入 ----
    def set_user_features(self, features) -> None:
        device = self.user_emb.weight.device
        self.user_features = torch.tensor(
            np.asarray(features, dtype=np.float32), dtype=torch.float32, device=device
        )

    def set_adj(self, adj: torch.Tensor) -> None:
        """BM3 不依赖图传播，提供空实现以保持接口统一。"""
        return None

    # ---- 表示与打分 ----
    def user_repr(self, users: torch.Tensor) -> torch.Tensor:
        """user 表示 = user_emb + profile 投影。"""
        return self.user_emb(users) + self.feat_proj(self.user_features[users])

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_repr(users)
        v = self.item_emb(items)
        return (u * v).sum(dim=1)

    def all_scores_batch(self, user_ids: List[int]) -> torch.Tensor:
        device = self.user_emb.weight.device
        users = torch.tensor(user_ids, dtype=torch.long, device=device)
        u = self.user_repr(users)              # [B, D]
        return u @ self.item_emb.weight.T      # [B, M]

    # ---- 辅助损失 ----
    def modality_contrastive_loss(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """模态内对比：对 user_emb 做 dropout 生成增强视图，InfoNCE。

        锚点 = user_emb(users)
        正样本 = embedding_dropout(user_emb(users))
        重复 user 互为正样本（多重正样本 mask）。
        """
        z1 = self.user_emb(users)
        z2 = embedding_dropout(z1, self.dropout)
        pos_mask = (users.unsqueeze(0) == users.unsqueeze(1)).float()
        return info_nce(z1, z2, self.cl_temperature, pos_mask=pos_mask)

    def cross_modal_align_loss(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """跨模态对齐：user_emb 与 profile 投影的 cosine 相似度。

        损失 = mean(1 - cos(user_emb, profile_proj))，越接近 0 表示越对齐。
        """
        u = self.user_emb(users)
        p = self.feat_proj(self.user_features[users])
        cos = F.cosine_similarity(u, p, dim=1)
        return (1.0 - cos).mean()


# =============================================================================
# 3. DiffRec (SIGIR 2023)
# =============================================================================
class DiffRecModel(nn.Module):
    """DiffRec：基于扩散模型的推荐模型。

    对 user 嵌入做前向加噪 + 反向去噪，将扩散过程作为 user 表示的正则化。
    - 前向：q(x_t | x_0) = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) ε
    - 反向：MLP 去噪器（替代 U-Net，推荐场景下足够）预测噪声 ε
    - 训练：L = L_BPR(score 用原始 user_emb) + diff_weight * L_diff(MSE)
    - 推理：默认用原始 user_emb 打分；可选 use_denoise=True 用采样去噪后的表示

    说明：去噪器输入为 [x_t, time_emb]（不含 user_emb 条件），因为 user 身份
    已编码在被扩散的 x_t 中；若额外以 user_emb 为条件会泄漏干净目标 x_0。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        n_diffusion_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        diff_weight: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_diffusion_steps = n_diffusion_steps
        self.diff_weight = diff_weight
        # 推理时是否使用去噪后的 user 表示（默认 False，用原始嵌入打分）
        self.use_denoise = False

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # 时间嵌入维度（与 dim 一致，便于拼接）
        self.time_dim = dim
        # 去噪 MLP：[x_t (dim) + time_emb (time_dim)] -> 预测噪声 (dim)
        hidden = dim * 2
        self.denoise_mlp = nn.Sequential(
            nn.Linear(dim + self.time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

        # 线性噪声调度（DDPM）
        betas = torch.linspace(beta_start, beta_end, n_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    # ---- 资源注入（接口兼容）----
    def set_user_features(self, features) -> None:
        """DiffRec 不使用 profile 特征，提供空实现以保持接口统一。"""
        return None

    def set_adj(self, adj: torch.Tensor) -> None:
        """DiffRec 不依赖图传播，提供空实现以保持接口统一。"""
        return None

    # ---- 时间嵌入 ----
    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """正弦/余弦时间嵌入 [B] -> [B, time_dim]。"""
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, time_dim]
        if emb.size(1) < self.time_dim:  # 奇数维度补零
            emb = F.pad(emb, (0, self.time_dim - emb.size(1)))
        return emb

    # ---- 前向加噪 ----
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """q(x_t | x_0) = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) ε。

        返回 (x_t, ε)，ε 为采样的标准高斯噪声。
        """
        noise = torch.randn_like(x0)
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1)          # [B, 1]
        sqrt_1mab = (1.0 - self.alpha_bars[t]).sqrt().view(-1, 1)
        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    # ---- 单步去噪 ----
    def denoise_step(self, xt: torch.Tensor, t: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        """单步去噪：预测噪声 ε_θ(x_t, t)。

        users 参数接受以保持接口一致；去噪器仅以 x_t + time_emb 为输入
        （user 身份已编码在 x_t 中，额外以 user_emb 为条件会泄漏 x_0）。
        """
        time_emb = self._time_embedding(t)              # [B, time_dim]
        inp = torch.cat([xt, time_emb], dim=1)          # [B, dim + time_dim]
        return self.denoise_mlp(inp)                    # [B, dim]

    # ---- 打分 ----
    def user_repr(self, users: torch.Tensor) -> torch.Tensor:
        """返回 user 表示：默认原始 user_emb；use_denoise=True 时用去噪采样。"""
        if self.use_denoise and not self.training:
            return self._denoised_user_repr(users)
        return self.user_emb(users)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_repr(users)
        v = self.item_emb(items)
        return (u * v).sum(dim=1)

    def all_scores_batch(self, user_ids: List[int]) -> torch.Tensor:
        device = self.user_emb.weight.device
        users = torch.tensor(user_ids, dtype=torch.long, device=device)
        u = self.user_repr(users)                       # [B, D]
        return u @ self.item_emb.weight.T               # [B, M]

    # ---- 扩散训练损失 ----
    def diffusion_loss(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """扩散模型训练损失：MSE(预测噪声, 真实噪声)。

        x_0 = user_emb(users)（被扩散的干净目标）
        随机采样时间步 t，前向加噪得到 x_t，去噪器预测噪声，与真实噪声做 MSE。
        """
        x0 = self.user_emb(users)
        bsz = x0.size(0)
        t = torch.randint(0, self.n_diffusion_steps, (bsz,), device=x0.device)
        xt, noise = self.forward_diffusion(x0, t)
        pred_noise = self.denoise_step(xt, t, users)
        return F.mse_loss(pred_noise, noise)

    # ---- 反向采样（推理可选）----
    @torch.no_grad()
    def _denoised_user_repr(self, users: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """从高斯噪声出发，按 DDPM 反向过程逐步去噪，得到 user 表示。

        n_steps 默认使用 n_diffusion_steps；可设较小值加速。
        """
        n_steps = n_steps or self.n_diffusion_steps
        x = torch.randn_like(self.user_emb(users))  # x_T ~ N(0, I)
        for i in reversed(range(n_steps)):
            t = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
            pred_noise = self.denoise_step(x, t, users)
            beta_t = self.betas[t].view(-1, 1)
            alpha_t = self.alphas[t].view(-1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1)
            mean = (1.0 / alpha_t.sqrt()) * (x - (beta_t / (1.0 - alpha_bar_t).sqrt()) * pred_noise)
            if i > 0:
                # DDPM 采样噪声
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean
        return x


# =============================================================================
# 模块自检（仅验证可导入与基本前向形状，不产生任何硬编码指标）
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    n_users, n_items, dim = 50, 40, 16
    feat_dim = 8
    users = torch.randint(0, n_users, (4,))
    items = torch.randint(0, n_items, (4,))

    # XSimGCL：构造归一化稀疏二部邻接矩阵
    xsim = XSimGCLModel(n_users, n_items, dim=dim, n_layers=2)
    rows = torch.randint(0, n_users, (20,))
    cols = torch.randint(0, n_items, (20,))
    idx = torch.stack([rows, cols])
    vals = torch.ones(20)
    raw = torch.sparse_coo_tensor(idx, vals, (n_users, n_items)).coalesce()
    udeg = torch.sparse.sum(raw, dim=1).to_dense().clamp(min=1.0)
    ideg = torch.sparse.sum(raw, dim=0).to_dense().clamp(min=1.0)
    norm_vals = raw.values() * udeg[rows].pow(-0.5) * ideg[cols].pow(-0.5)
    adj = torch.sparse_coo_tensor(raw.indices(), norm_vals, raw.shape).coalesce()
    xsim.set_adj(adj)
    print("XSimGCL score shape:", xsim.score(users, items).shape)
    print("XSimGCL cl_loss:", xsim.contrastive_loss(users, items).item())

    # BM3
    bm3 = BM3Model(n_users, n_items, feat_dim, dim=dim)
    bm3.set_user_features(np.random.rand(n_users, feat_dim).astype(np.float32))
    print("BM3 score shape:", bm3.score(users, items).shape)
    print("BM3 align_loss:", bm3.cross_modal_align_loss(users, items).item())

    # DiffRec
    diff = DiffRecModel(n_users, n_items, dim=dim, n_diffusion_steps=10)
    print("DiffRec score shape:", diff.score(users, items).shape)
    print("DiffRec diffusion_loss:", diff.diffusion_loss(users, items).item())
    print("All models OK.")
