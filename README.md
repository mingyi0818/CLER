# CLER: 面向声明与行为弱相关的跨视图对比推荐方法

> **审稿人复现指南** — 本仓库提供论文《面向声明与行为弱相关的跨视图对比推荐方法》的全部实验源代码、结果数据与复现流程，用于验证论文实验数据的真实性。

## 论文信息

- **标题**：面向声明与行为弱相关的跨视图对比推荐方法（Cross-View Contrastive Recommendation under Weak Declarative–Behavioral Correlation）
- **作者**：曾镜源，郭江鸿，姜传贤，冯亚芬
- **单位**：嘉应学院计算机学院 / 地理科学与旅游学院
- **基金项目**：广东省本科高校高等教育教学改革项目（粤教高函〔2024〕9-989）
- **拟投期刊**：计算机工程与应用（CEA）

## 方法概要

本文针对旅游与内容平台中注册表单（profile）与卡片式隐式反馈弱相关、交互长尾并存的问题，实现递进框架：

```
BPR ──(+L_UI)──> CLER ──(+L_CV)──> CV-CLER ──(+λ_u, L_PBD)──> ARFusion-Rec
```

- **CLER**：在BPR成对排序上叠加用户-物品对比学习（UI-CL）
- **CV-CLER**：引入声明-行为对称跨视图对比（L_CV），推理期保持评分解耦
- **ARFusion-Rec**：稀疏感知可靠性门控融合，按用户正样本密度自适应加权

核心发现：在profile与行为弱相关时，固定早期融合（Multimodal CF）劣于纯BPR；评分解耦（CV-CLER）在弱相关域一致稳健；门控融合（ARFusion-Rec）仅在profile字段语义清晰且极稀疏时才优于解耦。

## 仓库结构

```
CLER/
├── arfusion_recommender/          # 核心算法包（ARFusion-Rec模型与训练管线）
│   ├── __init__.py
│   ├── arfusion_model.py          # ARFusion-Rec模型实现
│   ├── configs.py                 # 数据集最佳超参数
│   └── pipeline.py                # 数据加载、训练、评估管线
├── travel_recommender/            # 跨数据集加载与基线模型包
│   ├── __init__.py
│   ├── paper_pipeline.py          # 统一模型定义（BPR/MMSSL/CLER等）
│   ├── inttravel_loader.py        # IntTravel数据集加载器
│   └── public_dataset_loaders.py  # MovieLens/Amazon加载器
├── experiments/
│   ├── results/                   # 38个实验结果JSON/CSV文件（数据溯源依据）
│   ├── run_stravl_5seed.py        # Stravl 5种子主实验（表4-5）
│   ├── run_cvcler_5seed.py        # CV-CLER 5种子实验（表4-5）
│   ├── run_cross_dataset.py       # 跨数据集验证（表20-21）
│   ├── run_inttravel_cvcler_mmssl.py  # IntTravel三分片实验（表19,22-23）
│   ├── run_xsimgcl_inttravel.py   # XSimGCL基线对比（表19）
│   ├── run_new_baselines.py       # 新基线（XSimGCL/BM3/DiffRec）（表4-5）
│   ├── run_beliefs_5seed.py       # MovieLens Beliefs 5种子实验
│   ├── run_profile_noise_ablation.py  # Profile噪声消融（表15）
│   ├── run_gate_ablation.py       # 门控消融实验（表13）
│   ├── run_cvcler_5seed.py        # CV-CLER消融
│   ├── run_lambda_cv_sensitivity.py   # λ_cv敏感性分析（表18）
│   ├── run_static_lambda_ablation.py  # 静态λ消融
│   ├── run_equal_budget_ablation.py   # 等预算消融
│   ├── run_cu_bucket_and_wilcoxon.py  # c_u分桶与Wilcoxon检验（表8,12）
│   ├── run_dropoutnet_baseline.py # DropoutNet基线
│   ├── run_lightgcl_baseline.py   # LightGCL基线
│   ├── run_mmssl_decoupled_stravl.py  # MMSSL-decoupled变体
│   ├── run_rho_beliefs.py         # 相关性proxy计算
│   ├── analyze_holm_correction.py # Holm校正统计分析（表8）
│   ├── aggregate_results.py       # 结果汇总
│   ├── load_movielens_beliefs.py  # MovieLens Beliefs加载
│   └── new_baselines.py           # 新基线模型实现
├── figures/                       # 论文图片（12幅PNG）
├── configs/
│   └── best_configs.py            # 数据集最佳超参数配置
├── data_scripts/                  # 数据集下载脚本
│   ├── download_all.py            # 一键下载全部数据集
│   ├── download_stravl.py         # 下载Stravl-Data
│   ├── download_movielens.py      # 下载MovieLens-1M
│   ├── download_amazon.py         # 下载Amazon-Electronics
│   └── download_inttravel.py      # 下载IntTravel
├── paths.py                       # 路径配置（审稿人可修改）
├── verify_results.py              # 数据真实性验证脚本（审稿人专用）
├── manuscript.md                  # 论文稿（中文）
├── memo.md                        # 修订记录
├── requirements.txt               # Python依赖
├── .gitignore
└── README.md                      # 本文件
```

## 环境配置

### 硬件要求

- GPU：NVIDIA GPU（≥8GB显存，推荐≥16GB）
- CPU：多核处理器（实验支持CPU回退，但速度较慢）
- 内存：≥16GB（IntTravel实验需≥32GB）

### 软件环境

- Python 3.10+
- CUDA 11.8+（GPU训练）

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/mingyi0818/CLER.git
cd CLER

# 安装依赖
pip install -r requirements.txt
```

依赖列表：
- torch>=2.0.0
- numpy>=1.24.0
- pandas>=2.0.0
- scipy>=1.10.0
- scikit-learn>=1.2.0
- tqdm>=4.65.0
- matplotlib>=3.7.0

## 数据集下载

### 一键下载（推荐）

```bash
python data_scripts/download_all.py
```

数据集将下载至 `data/` 目录。

### 单独下载

```bash
# Stravl-Data（主实验数据集）
python data_scripts/download_stravl.py

# MovieLens-1M（跨数据集验证）
python data_scripts/download_movielens.py

# Amazon-Electronics（跨数据集验证）
python data_scripts/download_amazon.py

# IntTravel（极稀疏POI验证）
python data_scripts/download_inttravel.py
```

### 数据集说明

| 数据集 | 用户数 | 物品数 | 密度 | 论文用途 |
|--------|--------|--------|------|----------|
| Stravl-Data | 80,301 | 1,452 | 0.73% | 主实验、5种子、消融、噪声 |
| MovieLens-1M | 6,038 | 3,533 | 2.70% | 跨数据集验证 |
| Amazon-Electronics | 25,000 | 56,725 | 0.033% | 跨数据集验证 |
| IntTravel (3-shard) | 25,000 | ~87k | 0.0059% | 极稀疏POI验证 |

## 快速验证数据真实性（审稿人首选）

无需运行实验，直接验证论文数字与结果文件的一致性：

```bash
python verify_results.py
```

该脚本将：
1. 检查 `experiments/results/` 下38个结果文件是否完整
2. 验证论文中报告的关键NDCG@10数值是否与JSON文件精确对应（误差<0.001）
3. 验证统计检验p值、消融实验数据的合理性
4. 输出逐项验证结果与总评分（100分制）

**预期输出**：通过41/41项验证，数据真实性评分100.0/100。

## 复现实验

### 1. Stravl主实验（论文表4-5，5种子）

```bash
cd experiments

# 运行BPR/Multimodal CF/CLER/ARFusion-Rec 5种子实验
python run_stravl_5seed.py

# 运行CV-CLER 5种子实验
python run_cvcler_5seed.py
```

结果保存至 `experiments/results/stravl_5seed_results.json` 和 `cvcler_5seed_results.json`。

### 2. 跨数据集验证（论文表20-21）

```bash
cd experiments
python run_cross_dataset.py
```

在MovieLens-1M和Amazon-Electronics上运行BPR/MMSSL/CLER/CV-CLER（seed=42）。

### 3. IntTravel极稀疏验证（论文表19, 22-23）

```bash
cd experiments
python run_inttravel_cvcler_mmssl.py
python run_xsimgcl_inttravel.py
```

### 4. 统计分析（论文表8）

```bash
cd experiments
# Holm校正与Wilcoxon检验
python analyze_holm_correction.py
# c_u分桶与逐用户Wilcoxon
python run_cu_bucket_and_wilcoxon.py
```

### 5. 消融实验

```bash
cd experiments
# Profile噪声消融（表15）
python run_profile_noise_ablation.py
# 门控消融（表13）
python run_gate_ablation.py
# λ_cv敏感性分析（表18）
python run_lambda_cv_sensitivity.py
# 静态λ消融
python run_static_lambda_ablation.py
# 等预算消融
python run_equal_budget_ablation.py
# CV-CLER组件消融
python run_cvcler_5seed.py  # 已包含消融配置
```

### 6. 基线方法

```bash
cd experiments
# DropoutNet基线
python run_dropoutnet_baseline.py
# LightGCL基线
python run_lightgcl_baseline.py
# MMSSL-decoupled变体
python run_mmssl_decoupled_stravl.py
# 新基线（XSimGCL/BM3/DiffRec）
python run_new_baselines.py
```

### 7. 结果汇总

```bash
cd experiments
python aggregate_results.py
```

## 论文表格与结果文件对照

| 论文表格 | 内容 | 结果文件 |
|----------|------|----------|
| 表4-5 | Stravl主实验5种子均值 | stravl_5seed_results.json, cvcler_5seed_results.json |
| 表8 | 统计显著性检验 | holm_correction_analysis.json, per_user_wilcoxon_complete_results.json |
| 表12 | c_u分桶分析 | cu_bucket_results.json |
| 表13 | 门控消融 | gate_ablation_results.json |
| 表15 | Profile噪声鲁棒性 | profile_noise_results.json |
| 表18 | λ_cv敏感性 | cv_lambda_sensitivity_results.json |
| 表19 | IntTravel对比 | inttravel_cvcler_mmssl_results.json, inttravel_xsimgcl_5seed_results.json |
| 表20-21 | 跨数据集验证 | cross_dataset_results.json |
| 表22-23 | IntTravel详细指标 | inttravel_tune_v2_results.json, inttravel_shard3_results.json |

## 论文图片与文件对照

| 图片 | 内容 | 文件 |
|------|------|------|
| 图1 | CLER框架图 | figures/fig1_cler_framework.png |
| 图2 | 结果对比 | figures/fig2_results_comparison.png |
| 图3 | 消融实验 | figures/fig3_ablation.png |
| 图4 | λ_cv敏感性 | figures/fig4_lambda_cv_sensitivity.png |
| 图5 | Profile噪声 | figures/fig4_profile_noise.png |
| 图6 | IntTravel对比 | figures/fig7_inttravel_comparison.png |

## 路径配置

所有脚本使用相对路径，默认布局为：
- 数据集：`data/`（仓库根目录下）
- 结果：`experiments/results/`
- 缓存：`experiments/cache/`

如需自定义数据集路径，修改 `paths.py` 中的 `DATA_ROOT`，或设置环境变量：

```bash
# Linux/Mac
export CLER_DATA_ROOT=/path/to/your/datasets

# Windows PowerShell
$env:CLER_DATA_ROOT = "D:\my\datasets"
```

## 实验结果

运行上方命令将在本地 `results/` 目录下重新生成全部指标、消融实验与论文图表（本仓库**不**存储 `results/`）。为避免提前公开未发表结果，本文档**不**预先刊登具体数值；审稿人可通过运行代码自行复现。

## 常见问题

### Q: 运行实验时提示找不到 `arfusion_recommender` 模块？

A: 确保在 `experiments/` 目录下运行脚本，脚本已配置自动路径。或手动设置：
```bash
export PYTHONPATH=/path/to/CLER/experiments:$PYTHONPATH
```

### Q: IntTravel实验内存不足？

A: IntTravel有25,000用户×87,000物品，全量评分矩阵较大。脚本已实现分批评估（`evaluate_topk_batched`），如仍不足，减小 `batch_size` 参数。

### Q: GPU不可用怎么办？

A: 修改 `configs/best_configs.py` 中的 `device` 参数为 `"cpu"`。CPU运行较慢但结果一致。

### Q: 如何只验证数据而不运行实验？

A: 直接运行 `python verify_results.py`，无需GPU或数据集。

## 引用

如使用本代码，请引用：

```bibtex
@article{zeng2026cler,
  title={面向声明与行为弱相关的跨视图对比推荐方法},
  author={曾镜源 and 郭江鸿 and 姜传贤 and 冯亚芬},
  journal={计算机工程与应用},
  year={2026},
  note={Submitted}
}
```

## 许可证

MIT License

## 联系方式

- 曾镜源（第一作者）：zjy@jyu.edu.cn
- 冯亚芬（通讯作者）：fyf81@163.com
- 嘉应学院计算机学院
