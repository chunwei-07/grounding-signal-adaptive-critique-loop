"""
Per-Tier Wilcoxon Signed-Rank Tests
Compares FSCL vs GSAL θ=0.6 RAGAS Faithfulness scores
within each complexity tier (simple, moderate, complex).
n=50 per tier, paired test (same questions run under both conditions).
"""

import json
import numpy as np
from scipy import stats
from collections import defaultdict

RESULTS_PATH = "results/scored_results.json"
THETA        = 0.6

def load_records(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    results = data.get("results", data)
    return results


def extract_faithfulness_by_tier(records: list) -> dict:
    """Returns {tier: {question_id: faithfulness}} for a list of records."""
    tier_map = defaultdict(dict)
    for rec in records:
        qid        = rec["question_id"]
        tier       = rec.get("complexity", "unknown")
        faith      = rec.get("faithfulness")
        if faith is not None:
            tier_map[tier][qid] = faith
    return dict(tier_map)


def wilcoxon_effect_size(z_stat: float, n: int) -> float:
    return abs(z_stat) / np.sqrt(n)


def cohen_label(r: float) -> str:
    if r >= 0.5:
        return "large"
    elif r >= 0.3:
        return "medium"
    elif r >= 0.1:
        return "small"
    else:
        return "negligible"


def run_per_tier_tests(results: dict):
    # Extract FSCL and GSAL θ=0.6 records
    fscl_key  = "FSCL"
    gsal_key  = f"GSAL_theta{THETA}"

    # Handle both key formats (GSAL_theta0.6 or GSAL_θ=0.6 etc.)
    gsal_key_actual = None
    for k in results.keys():
        if "GSAL" in k and str(THETA) in k:
            gsal_key_actual = k
            break

    if fscl_key not in results:
        print(f"ERROR: '{fscl_key}' not found in results. Keys: {list(results.keys())}")
        return

    if gsal_key_actual is None:
        print(f"ERROR: No GSAL θ={THETA} key found. Keys: {list(results.keys())}")
        return

    print(f"Using keys: '{fscl_key}' vs '{gsal_key_actual}'")

    fscl_by_tier = extract_faithfulness_by_tier(results[fscl_key])
    gsal_by_tier = extract_faithfulness_by_tier(results[gsal_key_actual])

    tiers = ["simple", "moderate", "complex"]

    print(f"\n{'='*65}")
    print(f"  Per-Tier Wilcoxon Signed-Rank: FSCL vs GSAL θ={THETA}")
    print(f"{'='*65}")

    all_results = {}

    for tier in tiers:
        fscl_tier = fscl_by_tier.get(tier, {})
        gsal_tier = gsal_by_tier.get(tier, {})

        # Paired: only questions present in BOTH conditions
        common_qids = sorted(set(fscl_tier.keys()) & set(gsal_tier.keys()))
        n = len(common_qids)

        if n == 0:
            print(f"\n  {tier.upper()}: No paired questions found.")
            continue

        fscl_scores = np.array([fscl_tier[q] for q in common_qids])
        gsal_scores = np.array([gsal_tier[q] for q in common_qids])
        deltas      = gsal_scores - fscl_scores

        mean_fscl = fscl_scores.mean()
        mean_gsal = gsal_scores.mean()
        mean_delta = deltas.mean()

        # Wilcoxon signed-rank test
        # Check if all differences are zero (degenerate case)
        if np.all(deltas == 0):
            print(f"\n  {tier.upper()} (n={n}):")
            print(f"  All differences are zero — test not applicable.")
            continue

        stat, p_value = stats.wilcoxon(gsal_scores, fscl_scores,
                                        alternative="two-sided")

        # Z approximation for effect size: scipy returns W statistic
        # Use normal approximation Z = (W - mu_W) / sigma_W
        # mu_W = n(n+1)/4, sigma_W = sqrt(n(n+1)(2n+1)/24)
        mu_w    = n * (n + 1) / 4
        sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z_stat  = (stat - mu_w) / sigma_w
        r       = wilcoxon_effect_size(z_stat, n)
        label   = cohen_label(r)

        # Significance label
        if p_value < 0.001:
            sig = "p<0.001 ***"
        elif p_value < 0.01:
            sig = f"p={p_value:.4f} **"
        elif p_value < 0.05:
            sig = f"p={p_value:.4f} *"
        else:
            sig = f"p={p_value:.4f} (n.s.)"

        all_results[tier] = {
            "n": n,
            "mean_fscl": mean_fscl,
            "mean_gsal": mean_gsal,
            "mean_delta": mean_delta,
            "W": stat,
            "Z": z_stat,
            "p": p_value,
            "r": r,
            "label": label
        }

        direction = "↑" if mean_delta > 0 else "↓"

        print(f"\n  {tier.upper()} (n={n})")
        print(f"  {'FSCL mean':<20}: {mean_fscl:.4f}")
        print(f"  {'GSAL mean':<20}: {mean_gsal:.4f}")
        print(f"  {'Delta':<20}: {mean_delta:+.4f} ({mean_delta*100:+.2f}pp) {direction}")
        print(f"  {'W statistic':<20}: {stat:.1f}")
        print(f"  {'Z (approx)':<20}: {z_stat:.3f}")
        print(f"  {'p-value':<20}: {sig}")
        print(f"  {'Effect size r':<20}: {r:.3f} ({label})")

    # Paper-ready summary
    print(f"\n{'='*65}")
    print("  PAPER-READY SUMMARY")
    print(f"{'='*65}")
    for tier in tiers:
        if tier not in all_results:
            continue
        res = all_results[tier]
        print(f"\n  {tier.capitalize()}: FSCL={res['mean_fscl']:.4f}, "
              f"GSAL={res['mean_gsal']:.4f}, "
              f"Δ={res['mean_delta']*100:+.2f}pp, "
              f"W={res['W']:.0f}, Z={res['Z']:.3f}, "
              f"p={res['p']:.4f}, r={res['r']:.3f} ({res['label']})")


if __name__ == "__main__":
    print(f"Loading: {RESULTS_PATH}")
    results = load_records(RESULTS_PATH)
    print(f"Keys found: {list(results.keys())}")
    run_per_tier_tests(results)