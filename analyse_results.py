"""
Produces all tables and figures for the paper 
from scored_results.json.

Outputs (written to results/figures/ and results/tables/):

    Tables:
        table1_aggregate.csv     - Table 1: aggregate comparison
        table2_by_complexity.csv - Table 2: results by complexity tier
        table3_theta_sweep.csv   - Table 3: θ sensitivity analysis

    Figures:
        fig1_iteration_dist.png       - Figure 1: iteration distribution
        fig2_sret_convergence.png     - Figure 2: S_ret convergence curves
        fig3_faithfulness_scatter.png - Figure 3: faithfulness vs iterations
        fig4_theta_tradeoff.png       - Figure 4: θ efficiency-faithfulness frontier

Usage:
    python analyse_results.py \
        --input results/scored_results.json
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats

# Config
FIGURES_DIR = Path("results/figures")
TABLES_DIR  = Path("results/tables")

COMPLEXITY_ORDER = ["simple", "moderate", "complex"]
THETA_VALUES     = [0.6, 0.7, 0.8]

COLOURS = {
    "FSCL":           "#E05C5C",   # Red
    "GSAL_theta0.6":  "#4CAF7D",   # Green light
    "GSAL_theta0.7":  "#2E7D52",   # Green mid
    "GSAL_theta0.8":  "#1B4F35",   # Green dark
}

LABELS = {
    "FSCL":           "FSCL (N=2)",
    "GSAL_theta0.6":  "GSAL θ=0.6",
    "GSAL_theta0.7":  "GSAL θ=0.7",
    "GSAL_theta0.8":  "GSAL θ=0.8",
}


# Data Loader
def load_data(input_path: str) -> tuple[dict, pd.DataFrame]:
    """
    Load stored results and flattern into a single DataFrame.

    Returns:
        raw : full JSON dict (for S_ret history access)
        df  : flat DataFrame with one row per question per system
    """
    with open(input_path, "r") as f:
        raw = json.load(f)

    rows = []
    for system_key, records in raw["results"].items():
        for r in records:
            # Skip failed records
            if r.get("error", "") or r.get("faithfulness", -1) < 0:
                continue
            rows.append({
                "system":        system_key,
                "question_id":   r["question_id"],
                "complexity":    r["complexity"],
                "question_type": r["question_type"],
                "iterations":    r["iterations_performed"],
                "faithfulness":  r["faithfulness"],
                "latency_ms":    r["latency_total_ms"],
                "tokens":        r["tokens_total"],
                "stopping_reason": r.get("stopping_reason", ""),
                "theta":         r.get("theta", None),
                "n_sret_checks": len(r.get("sret_history", [])),
            })

    df = pd.DataFrame(rows)
    print(f"[ANALYSE] Loaded {len(df)} valid records across "
          f"{df['system'].nunique()} systems.")
    print(f"[ANALYSE] Systems: {df['system'].unique().tolist()}")
    print(f"[ANALYSE] Complexity Distribution:\n"
          f"{df[df['system']=='FSCL']['complexity'].value_counts()}")
    
    return raw, df


# Statistical Tests
def wilcoxon_test(
    df: pd.DataFrame,
    system_a: str,
    system_b: str,
    metric: str
) -> dict:
    """
    Paired Wilcoxon signed-rank test between two systems on a metric.

    Paired on question_id - same questions, different systems.
    Non-parameteric: appropriate for faithfulness scores which 
    are bounded [0, 1] and unlikely to be normally distributed.

    Reports:
        statistic   : Wilcoxon W statistic
        p_value     : two-tailed p-value
        effect_r    : effect size r = Z / sqrt(N)
        significant : p < 0.05

    Args:
        df       : full results DataFrame
        system_a : system key for group A (e.g., 'FSCL')
        system_b : system key for group B (e.g., 'GSAL_theta0.7')
        metric   : column name ('faithfulness', 'iterations', etc.)

    Returns:
        Dict of test results
    """
    a = df[df["system"] == system_a].sort_values("question_id")[metric].values
    b = df[df["system"] == system_b].sort_values("question_id")[metric].values

    # Align lengths - only test on shared question_ids
    ids_a = set(df[df["system"] == system_a]["question_id"])
    ids_b = set(df[df["system"] == system_b]["question_id"])
    shared = sorted(ids_a & ids_b)

    a = df[(df["system"] == system_a) &
           (df["question_id"].isin(shared))].sort_values("question_id")[metric].values
    b = df[(df["system"] == system_b) &
           (df["question_id"].isin(shared))].sort_values("question_id")[metric].values
    
    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError as e:
        # All differences 0 - systems are identical on this metric
        return {
            "system_a"    : system_a,
            "system_b"    : system_b,
            "metric"      : metric,
            "n_pairs"     : n,
            "W"           : 0,
            "p_value"     : 1.0,
            "effect_r"    : 0.0,
            "significant" : False,
            "note"        : f"All differences zero: {e}"
        }

    n       = len(shared)
    z       = stats.norm.ppf(1 - max(p, 1e-15) / 2)
    r       = min(z / np.sqrt(n), 1.0)       # r is bounded [-1, 1] by definition

    return {
        "system_a"    : system_a,
        "system_b"    : system_b,
        "metric"      : metric,
        "n_pairs"     : n,
        "W"           : stat,
        "p_value"     : round(p, 4),
        "effect_r"    : round(r, 3),
        "significant" : p < 0.05,
    }


# Table 1 - Aggregate Comparison
def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: Aggregate comparison across all 150 questions.

    Columns: System | Avg Iterations | Avg Faithfulness |
             Avg Latency (ms) | Avg Tokens | % Converged Early
    """
    print("\n[ANALYSE] Building Table 1 - Aggregate Comparison...")

    rows = []
    systems = ["FSCL"] + [f"GSAL_theta{t}" for t in THETA_VALUES]

    for sys in systems: 
        sdf = df[df["system"] == sys]
        if sdf.empty:
            continue

        pct_early = 0.0
        if sys != "FSCL":
            converged = sdf["stopping_reason"].str.contains(
                "converged", na=False
            ).sum()
            pct_early = 100 * converged / len(sdf)

        # Iteration reduction vs FSCL
        fscl_avg_iter = df[df["system"] == "FSCL"]["iterations"].mean()
        iter_reduction = (
            0.0 if sys == "FSCL"
            else 100 * (fscl_avg_iter - sdf["iterations"].mean()) / fscl_avg_iter
        )

        # Faithfulness delta vs FSCL
        fscl_avg_faith = df[df["system"] == "FSCL"]["faithfulness"].mean()
        faith_delta = (
            0.0 if sys == "FSCL"
            else sdf["faithfulness"].mean() - fscl_avg_faith
        )

        rows.append({
            "System":                 LABELS.get(sys, sys),
            "Avg Iterations":         round(sdf["iterations"].mean(), 2),
            "Iter Reduction (%)":     round(iter_reduction, 1),
            "Avg Faithfulness":       round(sdf["faithfulness"].mean(), 4),
            "Faithfulness Δ vs FSCL": round(faith_delta, 4),
            "Avg Latency (ms)":       round(sdf["latency_ms"].mean(), 0),
            "Avg Tokens":             round(sdf["tokens"].mean(), 0),
            "% Converged Early":      round(pct_early, 1),
        })

    table1 = pd.DataFrame(rows)

    # Statistical significance vs FSCL
    print("\n[ANALYSE] Wilcoxon Tests vs FSCL:")
    for sys in [f"GSAL_theta{t}" for t in THETA_VALUES]:
        if sys not in df["system"].values:
            continue
        for metric in ["iterations", "faithfulness"]:
            result = wilcoxon_test(df, "FSCL", sys, metric)
            sig = "CONFIRMED p<0.05" if result["significant"] else "REJECTED n.s."
            print(f"  FSCL vs {sys} | {metric:15s} | "
                  f"p={result['p_value']:.4f} | "
                  f"r={result['effect_r']:.3f} | {sig}")
            
    path = TABLES_DIR / "table1_aggregate.csv"
    table1.to_csv(path, index=False)
    print(f"\n[ANALYSE] Table 1 saved -> {path}")
    print(table1.to_string(index=False))

    return table1


# Table 2 - By Complexity Tier
def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: Results broken down by complexity tier.
    This is the key table for the paper - shows GSAL's
    efficiency gains are largest on simple/moderate questions.
    """
    print("\n[ANALYSE] Building Table 2 - By Complexity Tier...")

    rows = []
    systems = ["FSCL"] + [f"GSAL_theta{t}" for t in THETA_VALUES]

    for tier in COMPLEXITY_ORDER:
        for sys in systems:
            sdf = df[(df["system"] == sys) & (df["complexity"] == tier)]
            if sdf.empty:
                continue

            rows.append({
                "Complexity": tier.capitalize(),
                "System":     LABELS.get(sys, sys),
                "N":          len(sdf),
                "Avg Iter":   round(sdf["iterations"].mean(), 2),
                "Std Iter":   round(sdf["iterations"].std(), 2),
                "Avg Faith":  round(sdf["faithfulness"].mean(), 4),
                "Std Faith":  round(sdf["faithfulness"].std(), 4),
                "Avg Latency":round(sdf["latency_ms"].mean(), 0),
                "Avg Tokens": round(sdf["tokens"].mean(), 0),
            })

    table2 = pd.DataFrame(rows)

    path = TABLES_DIR / "table2_by_complexity.csv"
    table2.to_csv(path, index=False)
    print(f"[ANALYSE] Table 2 saved -> {path}")
    print(table2.to_string(index=False))

    return table2


# Table 3 - θ Sensitivity 
def build_table3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: θ sweep - faithfulness vs efficiency trade-off.
    """
    print("\n[ANALYSE] Building Table 3 - θ sensitivity...")

    fscl = df[df["system"] == "FSCL"]
    rows = []

    for t in THETA_VALUES:
        sys = f"GSAL_theta{t}"
        sdf = df[df["system"] == sys]
        if sdf.empty:
            continue

        converged = sdf["stopping_reason"].str.contains(
            "converged", na=False
        ).sum()

        rows.append({
            "θ":                t,
            "Avg Iterations":   round(sdf["iterations"].mean(), 2),
            "vs FSCL (%)":      round(
                100 * (fscl["iterations"].mean() - sdf["iterations"].mean())
                / fscl["iterations"].mean(), 1
            ),
            "Avg Faithfulness": round(sdf["faithfulness"].mean(), 4),
            "vs FSCL (Δ)":      round(
                sdf["faithfulness"].mean() - fscl["faithfulness"].mean(), 4
            ),
            "% Converged":      round(100 * converged / len(sdf), 1),
            "Avg Tokens":       round(sdf["tokens"].mean(), 0),
        })

    table3 = pd.DataFrame(rows)

    path = TABLES_DIR / "table3_theta_sweep.csv"
    table3.to_csv(path, index=False)
    print(f"[ANALYSE] Table 3 saved -> {path}")
    print(table3.to_string(index=False))

    return table3


# Figure 1 - Iteration Distribution
def build_fig1(df: pd.DataFrame):
    """
    Figure 1: Iteration count distribution.
    FSCL is always 2, while GSAL shows a spread.
    """
    print("\n[ANALYSE] Building Figure 1 - Iteration distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overall distribution
    ax = axes[0]
    systems = ["FSCL"] + [f"GSAL_theta{t}" for t in THETA_VALUES]
    iter_data   = [df[df["system"] == s]["iterations"].values for s in systems]
    iter_labels = [LABELS[s] for s in systems]
    colours     = [COLOURS[s] for s in systems]

    ax.boxplot(
        iter_data,
        tick_labels=iter_labels,
        patch_artist=True,
        boxprops=dict(facecolor="white"),
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, colour in zip(ax.patches, colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)

    ax.set_ylabel("Iterations Performed", fontsize=11)
    ax.set_title("Iteration Count Distribution", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=15)

    # Right: by complexity tier for GSAL θ=0.7 vs FSCL
    ax2 = axes[1]
    x   = np.arange(len(COMPLEXITY_ORDER))
    w   = 0.35

    fscl_means = [
        df[(df["system"] == "FSCL") &
           (df["complexity"] == t)]["iterations"].mean()
        for t in COMPLEXITY_ORDER
    ]
    gsal_means = [
        df[(df["system"] == "GSAL_theta0.6") &
           (df["complexity"] == t)]["iterations"].mean()
        for t in COMPLEXITY_ORDER
    ]

    bars1 = ax2.bar(x - w/2, fscl_means, w,
                    label="FSCL (N=2)",
                    color=COLOURS["FSCL"], alpha=0.8)
    bars2 = ax2.bar(x + w/2, gsal_means, w,
                    label="GSAL θ=0.6",
                    color=COLOURS["GSAL_theta0.6"], alpha=0.8)
    
    ax2.set_ylabel("Avg Iterations", fontsize=11)
    ax2.set_title("Avg Iterations by Complexity (GSAL θ=0.6)",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.capitalize() for t in COMPLEXITY_ORDER])
    ax2.set_ylim(0, 3.5)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Value labels on bars
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}",
                 ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}",
                 ha="center", va="bottom", fontsize=9)
        
    plt.tight_layout()
    path = FIGURES_DIR / "fig1_iteration_dist.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ANALYSE] Figure 1 saved -> {path}")


# Figure 2 - S_ret Convergence Curves
def build_fig2(raw: dict):
    """
    Figure 2: S_ret convergence curves across evaluation turns.
    Shows how grounding score evolves: the actual signal driving 
    GSAL's stopping decision. One curve per complexity tier.
    """
    print("\n[ANALYSE] Building Figure 2 - S_ret convergence curves...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, tier in zip(axes, COMPLEXITY_ORDER):
        records_06 = [
            r for r in raw["results"].get("GSAL_theta0.6", [])
            if r.get("complexity") == tier
            and r.get("faithfulness", -1) >= 0
            and r.get("sret_history")
        ]

        if not records_06:
            ax.set_title(f"{tier.capitalize()} - no data")
            continue

        # Plot individual trajectories (light, thin)
        max_turns = max(len(r["sret_history"]) for r in records_06)
        for r in records_06:
            turns = list(range(len(r["sret_history"])))
            ax.plot(turns, r["sret_history"],
                    color=COLOURS["GSAL_theta0.6"],
                    alpha=0.15, linewidth=0.8)
            
        # Plot mean trajectory (bold)
        mean_by_turn = []
        for t in range(max_turns):
            vals = [
                r["sret_history"][t]
                for r in records_06
                if t < len(r["sret_history"])
            ]
            mean_by_turn.append(np.mean(vals))

        ax.plot(range(len(mean_by_turn)), mean_by_turn,
                color=COLOURS["GSAL_theta0.6"],
                linewidth=2.5, label="Mean S_ret", zorder=5)
        
        # θ threshold lines
        for t_val, style in [(0.6, ":"), (0.7, "--"), (0.8, "-.")]:
            ax.axhline(t_val, linestyle=style,
                       color="gray", linewidth=1,
                       label=f"θ={t_val}", alpha=0.7)
            
        ax.set_title(f"{tier.capitalize()} Questions",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Evaluation Turn", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.1, max_turns - 0.9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(linestyle="--", alpha=0.3)

    axes[0].set_ylabel("S_ret (Grounding Score)", fontsize=11)
    axes[2].legend(loc="lower right", fontsize=8)

    plt.suptitle("S_ret Convergence by Complexity Tier (GSAL θ=0.6)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = FIGURES_DIR / "fig2_sret_convergence.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ANALYSE] Figure 2 saved -> {path}")


# Figure 3 - Faithfulness vs. Iterations Scatter
def build_fig3(df: pd.DataFrame):
    """
    Figure 3: Faithfulness vs Iteration count scatter.
    The key claim: fewer iterations ≠ lower faithfulness.
    Colour-coded by complexity tier.
    """
    print("\n[ANALYSE] Building Figure 3 - Faithfulness scatter...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    tier_colours = {
        "simple"   :   "#4CAF7D",
        "moderate" : "#FF9800",
        "complex"  :  "#E05C5C",
    }

    for ax, sys in zip(axes, ["FSCL", "GSAL_theta0.6"]):
        sdf = df[df["system"] == sys]

        for tier in COMPLEXITY_ORDER:
            tdf = sdf[sdf["complexity"] == tier]
            ax.scatter(
                tdf["iterations"],
                tdf["faithfulness"],
                c=tier_colours[tier],
                label=tier.capitalize(),
                alpha=0.6,
                s=40,
                edgecolors="white",
                linewidths=0.5,
            )

        # Trend line
        if sys != "FSCL" and len(sdf) > 2:
            z = np.polyfit(sdf["iterations"], sdf["faithfulness"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sdf["iterations"].min(),
                                 sdf["iterations"].max(), 100)
            ax.plot(x_line, p(x_line),
                    "k--", linewidth=1.2, alpha=0.5, label="Trend")
            
        ax.set_xlabel("Iterations Performed", fontsize=11)
        ax.set_ylabel("RAGAS Faithfulness", fontsize=11)
        ax.set_title(LABELS.get(sys, sys),
                     fontsize=12, fontweight="bold")
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.05, 1.1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_faithfulness_scatter.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ANALYSE] Figure 3 saved -> {path}")


# Figure 4 - θ Trade-Off Frontier
def build_fig4(df: pd.DataFrame):
    """
    Figure 4: θ efficiency-faithfulness frontier.
    Shows the trade-off as θ increases - the core of Table 3 
    visualised. Shows the contribution of threshold sweep.
    """
    print("\n[ANALYSE] Building Figure 4 - θ trade-off frontier...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # FSCL as anchor point
    fscl_iter  = df[df["system"] == "FSCL"]["iterations"].mean()
    fscl_faith = df[df["system"] == "FSCL"]["faithfulness"].mean()

    ax.scatter(fscl_iter, fscl_faith,
               color=COLOURS["FSCL"],
               s=160, zorder=5,
               label="FSCL (N=2)", marker="s")    
    ax.axvline(x=fscl_iter, color=COLOURS["FSCL"], 
            linestyle="--", linewidth=1, alpha=0.5)
    ax.axvspan(0, fscl_iter, alpha=0.05, color="green",
            label="Efficiency gain zone")
    ax.axvspan(fscl_iter, 4, alpha=0.05, color="red",
            label="Efficiency loss zone")
    ax.annotate("FSCL (N=2)",
                (fscl_iter, fscl_faith),
                textcoords="offset points",
                xytext=(10, -12), fontsize=9)
    
    # GSAL θ points
    gsal_iters  = []
    gsal_faiths = []

    for t in THETA_VALUES:
        sys = f"GSAL_theta{t}"
        sdf = df[df["system"] == sys]
        if sdf.empty:
            continue
        avg_iter  = sdf["iterations"].mean()
        avg_faith = sdf["faithfulness"].mean()
        gsal_iters.append(avg_iter)
        gsal_faiths.append(avg_faith)

        ax.scatter(avg_iter, avg_faith,
                   color=COLOURS[sys],
                   s=160, zorder=5,
                   label=f"GSAL θ={t}")
        ax.annotate(f"θ={t}",
                    (avg_iter, avg_faith),
                    textcoords="offset points",
                    xytext=(8, 5), fontsize=9)
        
    # Connect GSAL points with frontier line
    if len(gsal_iters) > 1:
        sorted_pairs = sorted(zip(gsal_iters, gsal_faiths))
        xi, yi = zip(*sorted_pairs)
        ax.plot(xi, yi,
                color="gray", linestyle="--",
                linewidth=1.2, alpha=0.6,
                label="GSAL Trade-off Curve")

    ax.set_xlabel("Avg Iterations (lower = more efficient)", fontsize=11)
    ax.set_ylabel("Avg RAGAS Faithfulness (higher = better)", fontsize=11)
    ax.set_title("Efficiency-Faithfulness Trade-off (θ Sensitivity)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_theta_tradeoff.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ANALYSE] Figure 4 saved -> {path}")


# Main
def main(input_path: str):
    print("\n" + "=" * 60)
    print("PATH A - RESULTS ANALYSIS")
    print("=" * 60)

    # Setup output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    raw, df = load_data(input_path)

    # Tables
    build_table1(df)
    build_table2(df)
    build_table3(df)

    # Figures
    build_fig1(df)
    build_fig2(raw)
    build_fig3(df)
    build_fig4(df)

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"    Tables -> {TABLES_DIR}")
    print(f"    Figures -> {FIGURES_DIR}")
    print(f"{'=' * 60}\n")


# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse experiment results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/scored_results.json",
        help="Path to scored results JSON from ragas_scorer.py"
    )
    args = parser.parse_args()
    main(args.input)