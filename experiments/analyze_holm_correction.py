"""Comprehensive statistical analysis: Holm correction + Wilcoxon + per-user effective sample size.

Reads 5-seed results and computes:
1. Per-seed paired t-test + Wilcoxon signed-rank test (two-sided)
2. Holm-Bonferroni correction for multiple comparisons
3. Cohen's d effect size (paired)
4. Non-zero paired difference count (effective n)
5. Win/tie/loss at seed level
6. 95% bootstrap CI for mean difference

Also merges CV-CLER 5-seed results from cvcler_5seed_results.json.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

RESULTS_PATH = ROOT / "results" / "stravl_5seed_results.json"
CVCLER_PATH = ROOT / "results" / "cvcler_5seed_results.json"
OUT_PATH = ROOT / "results" / "holm_correction_analysis.json"


def cohens_d_paired(x, y):
    """Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    if len(diff) < 2 or diff.std(ddof=1) < 1e-12:
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def bootstrap_ci(diffs, n_boot=10000, ci=0.95):
    """Bootstrap CI for mean of paired differences."""
    diffs = np.asarray(diffs)
    n = len(diffs)
    if n < 2:
        return 0.0, 0.0
    rng = np.random.RandomState(42)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def main():
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run run_stravl_5seed_v2.py first.")
        return

    data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    summary = data["summary"]
    seeds_completed = data.get("seeds_completed", data["seeds"])

    # Merge CV-CLER 5-seed results if available
    if CVCLER_PATH.exists():
        cvcler_data = json.load(open(CVCLER_PATH, encoding="utf-8"))
        cvcler_summary = cvcler_data.get("summary", {})
        cvcler_seeds = cvcler_data.get("seeds_completed", [])
        if "CV-CLER" in cvcler_summary:
            summary["CV-CLER"] = cvcler_summary["CV-CLER"]
            print(f"Merged CV-CLER results: {cvcler_seeds}")

    methods = list(summary.keys())
    print(f"Methods: {methods}")
    print(f"Seeds completed: {seeds_completed}")

    # Collect per-seed NDCG@10 for each method (only completed seeds)
    per_seed_ndcg = {}
    for m in methods:
        vals = []
        for s in seeds_completed:
            seed_key = str(s)
            if seed_key in summary[m].get("seeds", {}):
                vals.append(summary[m]["seeds"][seed_key]["ndcg@10"])
        per_seed_ndcg[m] = vals

    n = len(seeds_completed)
    print(f"n_seeds = {n}")

    # Pairwise comparisons: each method vs each baseline
    baselines = ["BPR", "Multimodal CF", "CLER", "CV-CLER"]
    comparisons = []
    for baseline in baselines:
        if baseline not in per_seed_ndcg:
            continue
        for method in methods:
            if method == baseline or method not in per_seed_ndcg:
                continue
            x = np.array(per_seed_ndcg[method])
            y = np.array(per_seed_ndcg[baseline])
            diffs = x - y

            # Paired t-test (two-sided)
            if len(diffs) >= 2 and diffs.std(ddof=1) > 1e-12:
                t_stat, p_ttest = stats.ttest_rel(x, y)
            else:
                t_stat, p_ttest = float("nan"), 1.0

            # Wilcoxon signed-rank test (two-sided)
            if len(diffs) >= 5 and np.any(diffs != 0):
                try:
                    w_stat, p_wilcox = stats.wilcoxon(x, y, alternative="two-sided")
                except ValueError:
                    w_stat, p_wilcox = float("nan"), 1.0
            else:
                w_stat, p_wilcox = float("nan"), 1.0

            # Cohen's d
            d = cohens_d_paired(x, y)

            # Bootstrap CI
            ci_lo, ci_hi = bootstrap_ci(diffs.tolist())

            # Non-zero paired differences (effective sample size)
            nonzero_count = int(np.sum(np.abs(diffs) > 1e-8))
            wins = int(np.sum(diffs > 1e-8))
            losses = int(np.sum(diffs < -1e-8))
            ties = n - wins - losses

            comparisons.append({
                "method": method,
                "baseline": baseline,
                "mean_diff": float(diffs.mean()),
                "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                "t_stat": float(t_stat) if not np.isnan(t_stat) else None,
                "p_value_ttest": float(p_ttest) if not np.isnan(p_ttest) else 1.0,
                "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
                "p_value_wilcoxon": float(p_wilcox) if not np.isnan(p_wilcox) else 1.0,
                "cohens_d": d,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
                "n_seeds": n,
                "n_nonzero_diff": nonzero_count,
                "n_wins": wins,
                "n_losses": losses,
                "n_ties": ties,
            })

    # Holm-Bonferroni correction on t-test p-values
    comparisons.sort(key=lambda x: x["p_value_ttest"])
    m_cmp = len(comparisons)
    for i, comp in enumerate(comparisons):
        comp["holm_adjusted_p_ttest"] = min(comp["p_value_ttest"] * (m_cmp - i), 1.0)
        comp["significant_ttest_005"] = comp["holm_adjusted_p_ttest"] < 0.05

    # Also Holm on Wilcoxon p-values
    comparisons.sort(key=lambda x: x["p_value_wilcoxon"])
    for i, comp in enumerate(comparisons):
        comp["holm_adjusted_p_wilcoxon"] = min(comp["p_value_wilcoxon"] * (m_cmp - i), 1.0)
        comp["significant_wilcoxon_005"] = comp["holm_adjusted_p_wilcoxon"] < 0.05

    # Sort for display
    comparisons.sort(key=lambda x: (x["baseline"], x["method"]))

    # Output
    output = {
        "experiment": "Holm-Bonferroni + Wilcoxon on 5-seed Stravl (fixed code)",
        "date": "2026-07-24",
        "n_comparisons": m_cmp,
        "alpha": 0.05,
        "n_seeds": n,
        "seeds": seeds_completed,
        "mean_std_summary": {
            mth: {
                "ndcg@10": f"{summary[mth]['ndcg@10_mean']:.4f}±{summary[mth]['ndcg@10_std']:.4f}",
                "precision@10": f"{summary[mth]['precision@10_mean']:.4f}±{summary[mth]['precision@10_std']:.4f}",
                "recall@10": f"{summary[mth]['recall@10_mean']:.4f}±{summary[mth]['recall@10_std']:.4f}",
            } for mth in methods
        },
        "comparisons": comparisons,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"Statistical Analysis: {m_cmp} comparisons, n={n} seeds, alpha=0.05")
    print(f"{'='*100}")
    print(f"{'Method':<16}{'vs':<6}{'Baseline':<16}{'dNDCG@10':<11}{'p_ttest':<12}{'p_holm':<12}"
          f"{'p_wilcox':<12}{'Cohen_d':<10}{'W/T/L':<10}{'Sig?'}")
    for c in comparisons:
        sig = "YES" if c["significant_ttest_005"] else "no"
        wtl = f"{c['n_wins']}/{c['n_ties']}/{c['n_losses']}"
        print(f"{c['method']:<16}{'vs':<6}{c['baseline']:<16}"
              f"{c['mean_diff']:+.4f}    "
              f"{c['p_value_ttest']:.2e}  "
              f"{c['holm_adjusted_p_ttest']:.2e}  "
              f"{c['p_value_wilcoxon']:.2e}  "
              f"{c['cohens_d']:+.3f}    "
              f"{wtl:<10}{sig}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
