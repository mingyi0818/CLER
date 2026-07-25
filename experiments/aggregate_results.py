"""Aggregate all experiment result files into summary tables and statistics.

Reads every JSON result file under experiments/results/ listed in FILE_MAP,
normalizes their (differing) schemas into a common {method: {seed: metrics}}
layout, then produces:
  - main results table (mean +/- std, 95% CI for NDCG@10 / P@10 / R@10 / F1@10)
  - paired statistical significance tests (paired t-test, Wilcoxon, Cohen's d)
  - RC-SAFR ablation analysis (full vs no-mcdropout / no-router / no-maml)
  - MAML cross-domain analysis (zero_shot / finetune_all / maml_router)
  - LaTeX tables and CSV summaries for direct inclusion in the paper.

All metric numbers come from the JSON result files; no metric values are
hardcoded.  Missing files are skipped with a [MISSING] notice so partial runs
still aggregate.

Outputs (written to experiments/results/):
  - aggregated_summary.json
  - main_results_table.csv
  - significance_tests.csv
  - ablation_results.csv
  - maml_results.csv
  - latex_tables.tex
"""
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.stats import t as t_dist, wilcoxon


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS_DEFAULT = [42, 123, 456, 789, 2024]

# Canonical method ordering for the main results table
METHOD_ORDER_MAIN: List[str] = [
    "BPR",
    "Multimodal CF",
    "CLER",
    "CV-CLER",
    "ARFusion-Rec",
    "XSimGCL",
    "BM3",
    "DiffRec",
    "RC-SAFR",
]

METRICS: List[str] = ["ndcg@10", "precision@10", "recall@10", "f1@10"]

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

FILE_MAP: Dict[str, str] = {
    "stravl_main": "stravl_5seed_results.json",
    "stravl_cvcler": "cvcler_5seed_results.json",
    "beliefs_main": "beliefs_5seed_results.json",
    "inttravel_main": "inttravel_cvcler_mmssl_results.json",
    "stravl_baselines": "new_baselines_stravl_results.json",
    "beliefs_baselines": "new_baselines_beliefs_results.json",
    "rcsafr_stravl": "rcsafr_stravl_5seed_results.json",
    "rcsafr_beliefs": "rcsafr_beliefs_5seed_results.json",
    "rcsafr_ablation": "rcsafr_ablation_stravl_results.json",
    "maml_beliefs": "rcsafr_maml_stravl_to_beliefs_results.json",
    "maml_inttravel": "rcsafr_maml_stravl_to_inttravel_results.json",
}

# File keys that contribute methods to each dataset's main table
DATASET_GROUPS: List[Tuple[str, List[str]]] = [
    ("Stravl", ["stravl_main", "stravl_cvcler", "stravl_baselines", "rcsafr_stravl"]),
    ("Beliefs", ["beliefs_main", "beliefs_baselines", "rcsafr_beliefs"]),
    ("IntTravel", ["inttravel_main"]),
]

# For single-method result files whose per_seed_results is keyed directly by
# seed (no method level), assign this method name.
SINGLE_METHOD_KEY: Dict[str, str] = {
    "rcsafr_stravl": "RC-SAFR",
    "rcsafr_beliefs": "RC-SAFR",
}

MAML_METHOD_ORDER: List[str] = ["zero_shot", "finetune_all", "maml_router"]


# ---------------------------------------------------------------------------
# 1. Load all result files
# ---------------------------------------------------------------------------
def load_results() -> Dict[str, Any]:
    """Load every JSON file listed in FILE_MAP.  Missing/corrupt files skipped."""
    results: Dict[str, Any] = {}
    print("=" * 60)
    print("Loading result files")
    print("=" * 60)
    for key, filename in FILE_MAP.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"  [MISSING] {filename}")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                results[key] = json.load(f)
            print(f"  [OK] {filename}")
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")
    print()
    return results


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------
def normalize_method_seed(
    data: Dict[str, Any], file_key: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Normalize a result file's schema to {method: {seed_str: metrics_dict}}.

    Recognized schemas (checked in order):
      A. data["summary"][method]["seeds"][seed] = metrics
         (stravl_5seed_results.json)
      B. data["per_seed_results"][seed][method] = metrics
         (beliefs_5seed_results.json, new_baselines_*_results.json)
      C. data["per_seed_results"][seed] = metrics  (single-method file)
         (rcsafr_*_5seed_results.json -> method name from SINGLE_METHOD_KEY)
      D. data["results"][method]["metrics"], single seed in data["seed"]
         (inttravel_cvcler_mmssl_results.json)

    Ablation and MAML files are handled by their own dedicated routines.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Schema A: summary[method]["seeds"][seed] = metrics
    if isinstance(data.get("summary"), dict):
        for method, payload in data["summary"].items():
            if not isinstance(payload, dict):
                continue
            seeds_dict = payload.get("seeds")
            if isinstance(seeds_dict, dict) and seeds_dict:
                method_out: Dict[str, Dict[str, float]] = {}
                for seed, m in seeds_dict.items():
                    if isinstance(m, dict) and "ndcg@10" in m:
                        method_out[str(seed)] = m
                if method_out:
                    out[method] = method_out
        if out:
            return out

    # Schema B / C: per_seed_results
    per_seed = data.get("per_seed_results")
    if isinstance(per_seed, dict) and per_seed:
        sample = next(iter(per_seed.values()))
        if isinstance(sample, dict):
            looks_like_metrics = any(k in sample for k in METRICS)
            if looks_like_metrics:
                # Schema C: single-method file
                method_name = SINGLE_METHOD_KEY.get(file_key, "Method")
                method_out: Dict[str, Dict[str, float]] = {}
                for seed, m in per_seed.items():
                    if isinstance(m, dict) and "ndcg@10" in m:
                        method_out[str(seed)] = m
                if method_out:
                    out[method_name] = method_out
                    return out
            else:
                # Schema B: per_seed_results[seed][method] = metrics
                for seed, methods_dict in per_seed.items():
                    if not isinstance(methods_dict, dict):
                        continue
                    for method, m in methods_dict.items():
                        if isinstance(m, dict) and "ndcg@10" in m:
                            out.setdefault(method, {})[str(seed)] = m
                if out:
                    return out

    # Schema D: results[method]["metrics"], single seed
    if isinstance(data.get("results"), dict):
        seed_val = data.get("seed", 42)
        for method, payload in data["results"].items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics", payload)
            if isinstance(metrics, dict) and "ndcg@10" in metrics:
                out.setdefault(method, {})[str(seed_val)] = metrics
        if out:
            return out

    return out


def merge_dataset_methods(
    results: Dict[str, Any], dataset_keys: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Merge normalized {method: {seed: metrics}} from multiple file keys."""
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key in dataset_keys:
        if key not in results:
            continue
        per_method = normalize_method_seed(results[key], key)
        for method, seed_map in per_method.items():
            merged.setdefault(method, {})
            merged[method].update(seed_map)
    return merged


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def compute_stats(values: List[float]) -> Dict[str, float]:
    """Mean, std (ddof=0, matching existing scripts), 95% CI (t-dist), n."""
    n = len(values)
    if n == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0,
        }
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))  # ddof=0 (matches existing compute_summary)
    if n >= 2:
        sem = stats.sem(arr)  # ddof=1
        if sem > 0:
            lo, hi = t_dist.interval(0.95, df=n - 1, loc=mean, scale=sem)
        else:
            lo = hi = mean
    else:
        lo = hi = float("nan")
    return {
        "mean": mean, "std": std,
        "ci_lo": float(lo), "ci_hi": float(hi), "n": n,
    }


def cohen_d_pooled(a: List[float], b: List[float]) -> float:
    """Cohen's d with pooled std (independent-samples form)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    sa = float(np.var(a_arr, ddof=1))
    sb = float(np.var(b_arr, ddof=1))
    pooled = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled)


def effect_size_label(d: float) -> str:
    """Cohen's d magnitude label."""
    if np.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def paired_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired t-test.  Returns (t_stat, p_value); nan if not applicable."""
    if len(a) < 2 or len(a) != len(b):
        return float("nan"), float("nan")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.all(diff == 0):
        return float("nan"), float("nan")
    try:
        t_stat, p_val = stats.ttest_rel(a, b)
        return float(t_stat), float(p_val)
    except Exception:
        return float("nan"), float("nan")


def wilcoxon_signed_rank(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test.  Returns (stat, p_value); nan if not applicable."""
    if len(a) < 2 or len(a) != len(b):
        return float("nan"), float("nan")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.all(diff == 0):
        return float("nan"), float("nan")
    try:
        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float("nan"), float("nan")


def _seed_sort_key(s: str) -> Tuple[int, str]:
    """Sort seed strings numerically when possible."""
    try:
        return (0, str(int(s)))
    except (TypeError, ValueError):
        return (1, str(s))


# ---------------------------------------------------------------------------
# 2. Main results table
# ---------------------------------------------------------------------------
def generate_main_results_table(
    results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """One row per (dataset, method) with mean/std/CI for each metric."""
    rows: List[Dict[str, Any]] = []
    for dataset, keys in DATASET_GROUPS:
        per_method = merge_dataset_methods(results, keys)
        if not per_method:
            print(f"  [WARN] No data for dataset {dataset}")
            continue
        ordered = [m for m in METHOD_ORDER_MAIN if m in per_method]
        extras = [m for m in per_method if m not in ordered]
        ordered += extras
        for method in ordered:
            seed_map = per_method[method]
            seeds_sorted = sorted(seed_map.keys(), key=_seed_sort_key)
            row: Dict[str, Any] = {
                "dataset": dataset,
                "method": method,
                "n_seeds": len(seeds_sorted),
                "seeds": ",".join(seeds_sorted),
            }
            for metric in METRICS:
                values = [
                    seed_map[s].get(metric, float("nan"))
                    for s in seeds_sorted
                ]
                values = [v for v in values if not np.isnan(v)]
                st = compute_stats(values)
                row[f"{metric}_mean"] = st["mean"]
                row[f"{metric}_std"] = st["std"]
                row[f"{metric}_ci_lo"] = st["ci_lo"]
                row[f"{metric}_ci_hi"] = st["ci_hi"]
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 3. Statistical significance tests
# ---------------------------------------------------------------------------
def statistical_significance_tests(
    results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Paired t-test + Wilcoxon + Cohen's d for each method pair per dataset.

    Only method pairs sharing >= 2 common seeds are tested (paired tests need
    at least 2 observations).  IntTravel (single seed) is skipped.
    """
    rows: List[Dict[str, Any]] = []
    for dataset, keys in DATASET_GROUPS:
        if dataset == "IntTravel":
            # Single-seed IntTravel cannot support paired tests
            continue
        per_method = merge_dataset_methods(results, keys)
        methods = [m for m in METHOD_ORDER_MAIN if m in per_method]
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                ma, mb = methods[i], methods[j]
                sa, sb = per_method[ma], per_method[mb]
                common = sorted(set(sa.keys()) & set(sb.keys()),
                                key=_seed_sort_key)
                if len(common) < 2:
                    continue
                for metric in METRICS:
                    a = [sa[s].get(metric, float("nan")) for s in common]
                    b = [sb[s].get(metric, float("nan")) for s in common]
                    if any(np.isnan(x) for x in a) or any(np.isnan(x) for x in b):
                        continue
                    t_stat, t_p = paired_t_test(a, b)
                    w_stat, w_p = wilcoxon_signed_rank(a, b)
                    d = cohen_d_pooled(a, b)
                    rows.append({
                        "dataset": dataset,
                        "method_a": ma,
                        "method_b": mb,
                        "metric": metric,
                        "n_seeds": len(common),
                        "seeds": ",".join(common),
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        "mean_diff": float(np.mean(a) - np.mean(b)),
                        "t_stat": t_stat,
                        "t_p_value": t_p,
                        "wilcoxon_stat": w_stat,
                        "wilcoxon_p_value": w_p,
                        "cohen_d": d,
                        "effect_size": effect_size_label(d),
                    })
    return rows


# ---------------------------------------------------------------------------
# 4. Ablation analysis
# ---------------------------------------------------------------------------
def ablation_analysis(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """RC-SAFR ablation: per-variant mean/std + relative drop vs 'full'."""
    rows: List[Dict[str, Any]] = []
    data = results.get("rcsafr_ablation")
    if not data:
        print("  [SKIP] rcsafr_ablation not available")
        return rows
    per_variant = data.get("per_variant_results", {})
    if not per_variant or not isinstance(per_variant, dict):
        print("  [SKIP] rcsafr_ablation has no per_variant_results")
        return rows

    variant_stats: Dict[str, Dict[str, Any]] = {}
    for variant, seed_map in per_variant.items():
        if not isinstance(seed_map, dict):
            continue
        seeds_sorted = sorted(seed_map.keys(), key=_seed_sort_key)
        row: Dict[str, Any] = {
            "variant": variant,
            "n_seeds": len(seeds_sorted),
            "seeds": ",".join(seeds_sorted),
        }
        for metric in METRICS:
            values = []
            for s in seeds_sorted:
                m = seed_map[s]
                if isinstance(m, dict) and metric in m:
                    values.append(float(m[metric]))
            st = compute_stats(values)
            row[f"{metric}_mean"] = st["mean"]
            row[f"{metric}_std"] = st["std"]
            row[f"{metric}_ci_lo"] = st["ci_lo"]
            row[f"{metric}_ci_hi"] = st["ci_hi"]
        variant_stats[variant] = row

    full = variant_stats.get("full")
    if not full:
        print("  [WARN] Ablation missing 'full' variant; cannot compute drops")
    for variant, row in variant_stats.items():
        if full and variant != "full":
            for metric in METRICS:
                full_mean = full.get(f"{metric}_mean", float("nan"))
                var_mean = row.get(f"{metric}_mean", float("nan"))
                if (not np.isnan(full_mean) and full_mean != 0
                        and not np.isnan(var_mean)):
                    drop_pct = (var_mean - full_mean) / full_mean * 100.0
                else:
                    drop_pct = float("nan")
                row[f"{metric}_drop_vs_full_pct"] = drop_pct
        elif variant == "full":
            for metric in METRICS:
                row[f"{metric}_drop_vs_full_pct"] = 0.0
        else:
            for metric in METRICS:
                row[f"{metric}_drop_vs_full_pct"] = float("nan")
        rows.append(row)

    # Order: full first, then others alphabetically
    rows.sort(key=lambda r: (r["variant"] != "full", r["variant"]))
    return rows


# ---------------------------------------------------------------------------
# 5. MAML cross-domain analysis
# ---------------------------------------------------------------------------
def maml_cross_domain_analysis(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """MAML: zero_shot / finetune_all / maml_router per target domain.

    Computes relative improvement of each non-zero_shot method vs zero_shot.
    """
    rows: List[Dict[str, Any]] = []
    targets = [
        ("Beliefs", "maml_beliefs"),
        ("IntTravel", "maml_inttravel"),
    ]
    for target_name, key in targets:
        data = results.get(key)
        if not data:
            print(f"  [SKIP] {key} not available")
            continue
        per_method = data.get("per_method_per_seed_results", {})
        if not per_method or not isinstance(per_method, dict):
            print(f"  [SKIP] {key} has no per_method_per_seed_results")
            continue

        method_stats: Dict[str, Dict[str, Any]] = {}
        for method, seed_map in per_method.items():
            if not isinstance(seed_map, dict):
                continue
            seeds_sorted = sorted(seed_map.keys(), key=_seed_sort_key)
            row: Dict[str, Any] = {
                "target": target_name,
                "method": method,
                "n_seeds": len(seeds_sorted),
                "seeds": ",".join(seeds_sorted),
            }
            for metric in METRICS:
                values = []
                for s in seeds_sorted:
                    m = seed_map[s]
                    if isinstance(m, dict) and metric in m:
                        values.append(float(m[metric]))
                st = compute_stats(values)
                row[f"{metric}_mean"] = st["mean"]
                row[f"{metric}_std"] = st["std"]
                row[f"{metric}_ci_lo"] = st["ci_lo"]
                row[f"{metric}_ci_hi"] = st["ci_hi"]
            method_stats[method] = row

        zero_shot = method_stats.get("zero_shot")
        for method, row in method_stats.items():
            if zero_shot and method != "zero_shot":
                for metric in METRICS:
                    zs_mean = zero_shot.get(f"{metric}_mean", float("nan"))
                    m_mean = row.get(f"{metric}_mean", float("nan"))
                    if (not np.isnan(zs_mean) and zs_mean != 0
                            and not np.isnan(m_mean)):
                        imp_pct = (m_mean - zs_mean) / zs_mean * 100.0
                    else:
                        imp_pct = float("nan")
                    row[f"{metric}_improvement_vs_zero_shot_pct"] = imp_pct
            elif method == "zero_shot":
                for metric in METRICS:
                    row[f"{metric}_improvement_vs_zero_shot_pct"] = 0.0
            else:
                for metric in METRICS:
                    row[f"{metric}_improvement_vs_zero_shot_pct"] = float("nan")
            rows.append(row)

    # Order by target then by MAML_METHOD_ORDER
    rows.sort(key=lambda r: (
        r["target"],
        MAML_METHOD_ORDER.index(r["method"]) if r["method"] in MAML_METHOD_ORDER else 99,
        r["method"],
    ))
    return rows


# ---------------------------------------------------------------------------
# 6. LaTeX tables
# ---------------------------------------------------------------------------
def _fmt_pm(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "--"
    if np.isnan(std):
        return f"{mean:.4f}"
    return f"{mean:.4f}$\\pm${std:.4f}"


def generate_latex_tables(
    main_rows: List[Dict[str, Any]],
    sig_rows: List[Dict[str, Any]],
    ablation_rows: List[Dict[str, Any]],
    maml_rows: List[Dict[str, Any]],
) -> str:
    """Generate LaTeX tabular environments for the paper."""
    lines: List[str] = []
    lines.append("% Auto-generated by aggregate_results.py")
    lines.append("% All numbers originate from experiments/results/*.json files.")
    lines.append("")

    # --- Main results table ---
    lines.append("% ----- Main results: NDCG@10 / P@10 / R@10 / F1@10 (mean +/- std) -----")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Recommendation performance on three datasets (mean$\pm$std over seeds).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{l|cccc|cccc|cccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c|}{Stravl} & \multicolumn{4}{c|}{Beliefs} & \multicolumn{4}{c}{IntTravel} \\")
    lines.append(r"Method & NDCG@10 & P@10 & R@10 & F1@10 & NDCG@10 & P@10 & R@10 & F1@10 & NDCG@10 & P@10 & R@10 & F1@10 \\")
    lines.append(r"\midrule")
    by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in main_rows:
        by_method.setdefault(r["method"], {})[r["dataset"]] = r
    for method in METHOD_ORDER_MAIN:
        if method not in by_method:
            continue
        cells = [method.replace("_", r"\_")]
        for dataset in ["Stravl", "Beliefs", "IntTravel"]:
            r = by_method[method].get(dataset)
            if r is None:
                cells.extend(["--", "--", "--", "--"])
            else:
                for metric in METRICS:
                    cells.append(_fmt_pm(
                        r.get(f"{metric}_mean", float("nan")),
                        r.get(f"{metric}_std", float("nan")),
                    ))
        lines.append(" & ".join(cells) + r" \\")
    # Append any extras not in METHOD_ORDER_MAIN
    for method in by_method:
        if method in METHOD_ORDER_MAIN:
            continue
        cells = [method.replace("_", r"\_")]
        for dataset in ["Stravl", "Beliefs", "IntTravel"]:
            r = by_method[method].get(dataset)
            if r is None:
                cells.extend(["--", "--", "--", "--"])
            else:
                for metric in METRICS:
                    cells.append(_fmt_pm(
                        r.get(f"{metric}_mean", float("nan")),
                        r.get(f"{metric}_std", float("nan")),
                    ))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    # --- Significance tests (vs RC-SAFR, NDCG@10) ---
    sig_ndcg = [r for r in sig_rows
                if r["metric"] == "ndcg@10" and r["method_b"] == "RC-SAFR"]
    if sig_ndcg:
        lines.append("% ----- Significance tests vs RC-SAFR (NDCG@10, paired, 5 seeds) -----")
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Paired t-test and Wilcoxon signed-rank test vs.\ RC-SAFR on NDCG@10.}")
        lines.append(r"\label{tab:significance}")
        lines.append(r"\begin{tabular}{llrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Dataset & Baseline & $\Delta$ & $t$ & $p$ ($t$) & $p$ (Wil.) & Cohen's $d$ \\")
        lines.append(r"\midrule")
        for r in sig_ndcg:
            cells = [
                r["dataset"],
                r["method_a"].replace("_", r"\_"),
                f"{r['mean_diff']:.4f}",
                f"{r['t_stat']:.3f}" if not np.isnan(r["t_stat"]) else "--",
                f"{r['t_p_value']:.4f}" if not np.isnan(r["t_p_value"]) else "--",
                f"{r['wilcoxon_p_value']:.4f}" if not np.isnan(r["wilcoxon_p_value"]) else "--",
                f"{r['cohen_d']:.3f}" if not np.isnan(r["cohen_d"]) else "--",
            ]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- Ablation table ---
    if ablation_rows:
        lines.append("% ----- RC-SAFR ablation on Stravl -----")
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{RC-SAFR ablation on Stravl (mean$\pm$std over seeds). $\Delta$\% is relative to \texttt{full}.}")
        lines.append(r"\label{tab:ablation}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"Variant & NDCG@10 & P@10 & R@10 & F1@10 \\")
        lines.append(r"\midrule")
        for r in ablation_rows:
            cells = [r["variant"].replace("_", r"\_")]
            for metric in METRICS:
                cells.append(_fmt_pm(
                    r.get(f"{metric}_mean", float("nan")),
                    r.get(f"{metric}_std", float("nan")),
                ))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- MAML cross-domain table ---
    if maml_rows:
        lines.append("% ----- MAML cross-domain: Stravl -> Beliefs / IntTravel -----")
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Cross-domain adaptation (Stravl$\to$target), mean$\pm$std over seeds.}")
        lines.append(r"\label{tab:maml}")
        lines.append(r"\begin{tabular}{l|cccc|cccc}")
        lines.append(r"\toprule")
        lines.append(r" & \multicolumn{4}{c|}{Beliefs} & \multicolumn{4}{c}{IntTravel} \\")
        lines.append(r"Method & NDCG@10 & P@10 & R@10 & F1@10 & NDCG@10 & P@10 & R@10 & F1@10 \\")
        lines.append(r"\midrule")
        by_method_maml: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in maml_rows:
            by_method_maml.setdefault(r["method"], {})[r["target"]] = r
        for method in MAML_METHOD_ORDER:
            if method not in by_method_maml:
                continue
            cells = [method.replace("_", r"\_")]
            for target in ["Beliefs", "IntTravel"]:
                r = by_method_maml[method].get(target)
                if r is None:
                    cells.extend(["--", "--", "--", "--"])
                else:
                    for metric in METRICS:
                        cells.append(_fmt_pm(
                            r.get(f"{metric}_mean", float("nan")),
                            r.get(f"{metric}_std", float("nan")),
                        ))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. CSV writers + JSON sanitizer
# ---------------------------------------------------------------------------
def _sanitize_json(obj: Any) -> Any:
    """Convert NaN/Inf floats to None so JSON output is valid."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to CSV using the first row's key order as fieldnames."""
    if not rows:
        print(f"  [SKIP] No rows for {path.name}")
        return
    # Use union of keys preserving first-seen order so extra columns aren't lost
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  [SAVED] {path.name} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    results = load_results()
    if not results:
        print("No result files found.  Nothing to aggregate.")
        sys.exit(0)

    print("=" * 60)
    print("Generating main results table")
    print("=" * 60)
    main_rows = generate_main_results_table(results)
    for r in main_rows:
        ndcg_m = r.get("ndcg@10_mean", float("nan"))
        ndcg_s = r.get("ndcg@10_std", float("nan"))
        ndcg_str = (
            f"{ndcg_m:.4f}+/-{ndcg_s:.4f}"
            if not np.isnan(ndcg_m) else "n/a"
        )
        print(f"  {r['dataset']:10s} {r['method']:18s} "
              f"NDCG@10={ndcg_str} (n={r['n_seeds']})")

    print()
    print("=" * 60)
    print("Running statistical significance tests")
    print("=" * 60)
    sig_rows = statistical_significance_tests(results)
    print(f"  {len(sig_rows)} method-pair/metric tests performed")
    # Highlight significant comparisons (p < 0.05) on NDCG@10
    for r in sig_rows:
        if r["metric"] != "ndcg@10":
            continue
        p = r["t_p_value"]
        star = "*" if (not np.isnan(p) and p < 0.05) else " "
        print(f"  {star} {r['dataset']:8s} {r['method_a']:18s} vs "
              f"{r['method_b']:18s} d={r['mean_diff']:+.4f} "
              f"t_p={p if not np.isnan(p) else 'n/a'}")

    print()
    print("=" * 60)
    print("Ablation analysis (RC-SAFR)")
    print("=" * 60)
    ablation_rows = ablation_analysis(results)
    for r in ablation_rows:
        ndcg_m = r.get("ndcg@10_mean", float("nan"))
        ndcg_s = r.get("ndcg@10_std", float("nan"))
        drop = r.get("ndcg@10_drop_vs_full_pct", float("nan"))
        ndcg_str = (
            f"{ndcg_m:.4f}+/-{ndcg_s:.4f}"
            if not np.isnan(ndcg_m) else "n/a"
        )
        drop_str = f"{drop:+.2f}%" if not np.isnan(drop) else "n/a"
        print(f"  {r['variant']:18s} NDCG@10={ndcg_str} drop_vs_full={drop_str}")

    print()
    print("=" * 60)
    print("MAML cross-domain analysis")
    print("=" * 60)
    maml_rows = maml_cross_domain_analysis(results)
    for r in maml_rows:
        ndcg_m = r.get("ndcg@10_mean", float("nan"))
        ndcg_s = r.get("ndcg@10_std", float("nan"))
        imp = r.get("ndcg@10_improvement_vs_zero_shot_pct", float("nan"))
        ndcg_str = (
            f"{ndcg_m:.4f}+/-{ndcg_s:.4f}"
            if not np.isnan(ndcg_m) else "n/a"
        )
        imp_str = f"{imp:+.2f}%" if not np.isnan(imp) else "n/a"
        print(f"  {r['target']:12s} {r['method']:18s} "
              f"NDCG@10={ndcg_str} imp_vs_zs={imp_str}")

    print()
    print("=" * 60)
    print("Writing output files")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "meta": {
            "n_files_loaded": len(results),
            "files_loaded": sorted(results.keys()),
            "metrics": METRICS,
        },
        "main_results": main_rows,
        "significance_tests": sig_rows,
        "ablation_results": ablation_rows,
        "maml_results": maml_rows,
    }
    with open(RESULTS_DIR / "aggregated_summary.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary), f, indent=2, ensure_ascii=False)
    print("  [SAVED] aggregated_summary.json")

    write_csv(RESULTS_DIR / "main_results_table.csv", main_rows)
    write_csv(RESULTS_DIR / "significance_tests.csv", sig_rows)
    write_csv(RESULTS_DIR / "ablation_results.csv", ablation_rows)
    write_csv(RESULTS_DIR / "maml_results.csv", maml_rows)

    latex = generate_latex_tables(main_rows, sig_rows, ablation_rows, maml_rows)
    with open(RESULTS_DIR / "latex_tables.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    print("  [SAVED] latex_tables.tex")

    print()
    print("=" * 60)
    print("Aggregation complete.")
    print(f"  Main results:      {len(main_rows)} rows")
    print(f"  Significance:      {len(sig_rows)} rows")
    print(f"  Ablation:          {len(ablation_rows)} rows")
    print(f"  MAML cross-domain: {len(maml_rows)} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
