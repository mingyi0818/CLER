"""数据集最佳超参数配置（论文实验使用）。

这些配置通过验证集早停选择，已在论文 5.2 节敏感性分析中验证。
审稿人可直接使用这些配置复现论文结果。
"""
from arfusion_recommender.pipeline import TrainConfig

# Stravl-Data 最佳配置（5种子实验）
STRAVL_BEST = TrainConfig(
    use_graph=True,
    score_mode="collab",
    cv_weight=0.2,
    pbd_weight=0.05,
    n_gnn_layers=2,
    cl_weight=0.1,
    max_epochs=35,
    patience=10,
)

# IntTravel 三分片最佳配置
INTTRAVEL_BEST = TrainConfig(
    use_graph=False,
    score_mode="dual",
    cv_weight=0.05,
    pbd_weight=0.02,
    n_gnn_layers=0,
    cl_weight=0.08,
    max_epochs=30,
    patience=5,
)

# 跨数据集统一配置（dual-score + graph，用于 MovieLens/Amazon）
UNIFIED = TrainConfig(
    use_graph=True,
    score_mode="dual",
    cv_weight=0.2,
    pbd_weight=0.08,
    n_gnn_layers=2,
    cl_weight=0.1,
)

# 5种子列表（论文统计分析使用）
SEEDS = [42, 123, 456, 789, 2024]

# 默认嵌入维度
EMBED_DIM = 64
