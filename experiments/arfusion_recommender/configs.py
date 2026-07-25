"""Dataset-specific best configs discovered during tuning."""

from arfusion_recommender.pipeline import TrainConfig

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

# Unified config for cross-dataset generalization narrative (dual-score + graph)
UNIFIED = TrainConfig(
    use_graph=True,
    score_mode="dual",
    cv_weight=0.2,
    pbd_weight=0.08,
    n_gnn_layers=2,
    cl_weight=0.1,
)
