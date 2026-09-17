"""
S_ret Distribution Analysis

Computes the natural grounding-score distribution of the HotpotQA gold-context
corpus across all GSAL evaluation turns, to empirically characterise the
natural S_ret ceiling referenced in S5-D and S6-B.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = "results/raw_results.json"
OUTPUT_DIR   = Path("results/figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_gsal_records(path: str) -> dict:
    """Load all GSAL records from the results JSON, keyed by theta value."""
    with open(path, "r") as f:
        data = json.load(f)

    records_by_theta = defaultdict(list)
    results = data.get("results", data)  # handle both wrapped and flat formats

    for key, records in results.items():
        if not key.startswith("GSAL"):
            continue
        for rec in records:
            theta = rec.get("theta")
            sret  = rec.get("sret_history", [])
            if theta is not None and sret:
                records_by_theta[theta].append(rec)

    return dict(records_by_theta)


def collect_all_sret_scores(records: list) -> list:
    """Flatten all sret_history values from a list of records into one list."""
    return [s for rec in records for s in rec.get("sret_history", [])]


def print_distribution_stats(scores: list, label: str):
    arr = np.array(scores)
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  N scores       : {len(arr)}")
    print(f"  Mean           : {arr.mean():.4f}")
    print(f"  Median         : {np.median(arr):.4f}")
    print(f"  Std dev        : {arr.std():.4f}")
    print(f"  Min            : {arr.min():.4f}")
    print(f"  Max            : {arr.max():.4f}")
    print(f"  25th pct       : {np.percentile(arr, 25):.4f}")
    print(f"  50th pct       : {np.percentile(arr, 50):.4f}")
    print(f"  75th pct       : {np.percentile(arr, 75):.4f}")
    print(f"  90th pct       : {np.percentile(arr, 90):.4f}")
    print(f"  95th pct       : {np.percentile(arr, 95):.4f}")
    print(f"  % above 0.60   : {(arr > 0.60).mean()*100:.1f}%")
    print(f"  % above 0.70   : {(arr > 0.70).mean()*100:.1f}%")
    print(f"  % above 0.80   : {(arr > 0.80).mean()*100:.1f}%")


def analyse_by_turn(records: list, theta: float):
    """
    Break down mean S_ret by evaluation turn index (turn 0 = y0, turn 1 = y1...).
    Useful for showing where scores peak and plateau.
    """
    turn_scores = defaultdict(list)
    for rec in records:
        for t, s in enumerate(rec.get("sret_history", [])):
            turn_scores[t].append(s)

    print(f"\n  S_ret by evaluation turn (theta={theta}):")
    print(f"  {'Turn':<8} {'N':<6} {'Mean':<8} {'Median':<8} {'Std':<8}")
    print(f"  {'-'*40}")
    for turn in sorted(turn_scores):
        arr = np.array(turn_scores[turn])
        print(f"  {turn:<8} {len(arr):<6} {arr.mean():<8.4f} "
              f"{np.median(arr):<8.4f} {arr.std():<8.4f}")
    return turn_scores


def plot_sret_distribution(all_scores_by_theta: dict):
    """
    Plot histogram of all S_ret scores pooled across all turns,
    with vertical lines at theta=0.6, 0.7, 0.8 for visual reference.
    """
    # Pool all scores across all theta conditions (they share same corpus)
    # Use theta=0.6 as canonical since it has most turn-0 and turn-1 coverage
    scores_06 = np.array(all_scores_by_theta.get(0.6, []))
    if len(scores_06) == 0:
        print("No theta=0.6 scores found, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores_06, bins=40, color="#4C72B0", alpha=0.75, edgecolor="white",
            label="S_ret scores (θ=0.6 runs, all turns)")

    for theta, color, ls in [(0.6, "#2ca02c", "--"),
                              (0.7, "#ff7f0e", "-."),
                              (0.8, "#d62728", ":")]:
        ax.axvline(theta, color=color, linestyle=ls, linewidth=1.8,
                   label=f"θ = {theta}")

    # Mark mean and 90th pct
    ax.axvline(scores_06.mean(), color="black", linestyle="-", linewidth=1.2,
               label=f"Mean = {scores_06.mean():.3f}")
    p90 = np.percentile(scores_06, 90)
    ax.axvline(p90, color="grey", linestyle="-", linewidth=1.2,
               label=f"90th pct = {p90:.3f}")

    ax.set_xlabel("S_ret (Grounding Score)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("S_ret Score Distribution Across All Evaluation Turns\n"
                 "(HotpotQA Gold-Context Corpus, GSAL θ=0.6)", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "sret_distribution.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved to: {out_path}")
    plt.close()


def main():
    print(f"\nLoading results from: {RESULTS_PATH}")
    records_by_theta = load_gsal_records(RESULTS_PATH)

    if not records_by_theta:
        print("ERROR: No GSAL records found. Check RESULTS_PATH.")
        return

    print(f"Theta values found: {sorted(records_by_theta.keys())}")

    all_scores_by_theta = {}

    for theta in sorted(records_by_theta.keys()):
        records = records_by_theta[theta]
        all_scores = collect_all_sret_scores(records)
        all_scores_by_theta[theta] = all_scores

        print_distribution_stats(
            all_scores,
            f"All S_ret scores — GSAL θ={theta} ({len(records)} questions)"
        )
        analyse_by_turn(records, theta)

    # Pooled analysis across all theta (same corpus, more data points)
    pooled = []
    seen_qids = set()
    # Use theta=0.6 records as canonical to avoid triple-counting same questions
    for rec in records_by_theta.get(0.6, []):
        qid = rec["question_id"]
        if qid not in seen_qids:
            pooled.extend(rec.get("sret_history", []))
            seen_qids.add(qid)

    print_distribution_stats(
        pooled,
        "Pooled S_ret scores — θ=0.6 records (canonical corpus view)"
    )

    plot_sret_distribution(all_scores_by_theta)
    print("\nDone.")


if __name__ == "__main__":
    main()