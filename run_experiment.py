"""
Master orchestrator for the experiment.

Runs both systems (FSCL and GSAL) on all 150 HotpotQA questions 
and writes raw results to JSON for downstream RAGAS scoring and 
analysis.

Execution order:
    1. Load hotpotqa_150.json
    2. Run FSCL on all 150 questions
    3. Run GSAL at θ=0.6 on all 150 questions
    4. Run GSAL at θ=0.7 on all 150 questions
    5. Run GSAL at θ=0.8 on all 150 questions
    6. Write results/raw_results.json

Total LLM calls (estimate):
    FSCL  : 150 x (1 gen + 2 critique + 2 refine) = 750 calls
    GSAL  : 150 x ~2.5 avg iterations x 3 θ values ≈ 1,125 calls
    Total : ~1875 calls at ~500 tokens avg ≈ ~$3.00

Always run a dry run (--dry-run flag) on 5 questions first to 
verify the pipeline end-to-end before committing to full 150.

Usage:
    # 1. Dry run first
    python run_experiment.py --dry-run

    # 2. Run FSCL first and check results
    python run_experiment.py --system fscl

    # 3. Then run GSAL one theta at a time
    python run_experiment.py --system gsal --theta 0.7

    # 4. If results looks good, run remaining thetas
    python run_experiment.py --system gsal --theta 0.6
    python run_experiment.py --system gsal --theta 0.8
"""

import json
import time
import argparse
from pathlib import Path
from systems.fscl import run_fscl, fscl_result_to_dict
from systems.gsal import run_gsal, gsal_result_to_dict, THETA_VALUES

# Config
DATA_FILE   = Path("data/hotpotqa_150.json")
RESULTS_DIR = Path("results")
RAW_OUTPUT  = RESULTS_DIR / "raw_results.json"

INTER_QUESTION_DELAY_S = 2.0   # Seconds between questions to prevent rate limit burst
DRY_RUN_N = 5


# Data Loader
def load_questions(dry_run: bool = False) -> list[dict]:
    """
    Load questions from hotpotqa_150.json.

    Args:
        dry_run: if True, return only first DRY_RUN_N questions

    Returns:
        List of question dicts
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}\n"
            f"Run load_hotpotqa.py first."
        )
    
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    questions = data["questions"]

    if dry_run:
        # Take first DRY_RUN_N questions, grab at least one of
        # each complexity tier if possible
        sampled = []
        for tier in ["simple", "moderate", "complex"]:
            tier_qs = [q for q in questions if q["complexity"] == tier]
            if tier_qs:
                sampled.append(tier_qs[0])
        # Fill remaining slots from the front
        for q in questions:
            if q not in sampled and len(sampled) < DRY_RUN_N:
                sampled.append(q)
        questions = sampled[:DRY_RUN_N]
        print(f"[RUN] DRY RUN - using {len(questions)} questions")
        for q in questions:
            print(f"    [{q['complexity']:8s}] {q['question'][:60]}...")
    else:
        print(f"[RUN] Full run - {len(questions)} questions loaded")

    return questions


# System Runners
def run_fscl_batch(questions: list[dict]) -> list[dict]:
    """
    Run FSCL on all questions. Returns list of result dicts.
    """
    print(f"\n{'=' * 60}")
    print(f"RUNNING FSCL - {len(questions)} questions")
    print(f"{'=' * 60}")

    results = []
    errors  = 0

    for i, q in enumerate(questions):
        print(f"\n[FSCL] Question {i+1}/{len(questions)}")
        print(f"       ID         : {q['id']}")
        print(f"       Complexity : {q['complexity']}")
        print(f"       Question   : {q['question'][:70]}...")

        result = run_fscl(q)
        record = fscl_result_to_dict(result)
        results.append(record)

        if record["error"]:
            errors += 1

        # Process Log
        print(f"\n[FSCL] - Progress: {i+1}/{len(questions)} | "
              f"Errors: {errors} | "
              f"Tokens this question: {record['tokens_total']}")
        
        # Inter-question delay
        if i < len(questions) - 1:
            time.sleep(INTER_QUESTION_DELAY_S)

    # Batch Summary
    valid = [r for r in results if not r["error"]]
    print(f"\n[FSCL] Batch complete:")
    print(f"       Questions : {len(results)}")
    print(f"       Errors    : {errors}")
    if valid:
        avg_iter    = sum(r["iterations_performed"] for r in valid) / len(valid)
        avg_tokens  = sum(r["tokens_total"] for r in valid) / len(valid)
        avg_latency = sum(r["latency_total_ms"] for r in valid) / len(valid)
        print(f"       Avg iter  : {avg_iter:.2f}")
        print(f"       Avg tokens: {avg_tokens:.0f}")
        print(f"       Avg latency: {avg_latency:.0f}ms")

    return results


def run_gsal_batch(questions: list[dict], theta: float) -> list[dict]:
    """
    Run GSAL at a specific θ on all questions.
    Returns list of result dicts.
    """
    print(f"\n{'=' * 60}")
    print(f"RUNNING GSAL θ={theta} — {len(questions)} questions")
    print(f"{'=' * 60}")

    results = []
    errors  = 0

    for i, q in enumerate(questions):
        print(f"\n[GSAL θ={theta}] Question {i+1}/{len(questions)}")
        print(f"       ID         : {q['id']}")
        print(f"       Complexity : {q['complexity']}")
        print(f"       Question   : {q['question'][:70]}...")

        result = run_gsal(q, theta=theta)
        record = gsal_result_to_dict(result)
        results.append(record)

        if record["error"]:
            errors += 1

        print(f"\n[GSAL θ={theta}] ── Progress: {i+1}/{len(questions)} | "
              f"Errors: {errors} | "
              f"Iterations: {record['iterations_performed']} | "
              f"S_ret history: {[round(s,3) for s in record['sret_history']]}")
        
        if i < len(questions) - 1:
            time.sleep(INTER_QUESTION_DELAY_S)

    # Batch Summary
    valid = [r for r in results if not r["error"]]
    print(f"\n[GSAL θ={theta}] Batch complete:")
    print(f"       Questions : {len(results)}")
    print(f"       Errors    : {errors}")
    if valid:
        avg_iter    = sum(r["iterations_performed"] for r in valid) / len(valid)
        avg_tokens  = sum(r["tokens_total"] for r in valid) / len(valid)
        avg_latency = sum(r["latency_total_ms"] for r in valid) / len(valid)
        stopped_early = sum(
            1 for r in valid
            if "converged" in r["stopping_reason"]
        )
        print(f"       Avg iter      : {avg_iter:.2f}")
        print(f"       Avg tokens    : {avg_tokens:.0f}")
        print(f"       Avg latency   : {avg_latency:.0f}ms")
        print(f"       Converged     : {stopped_early}/{len(valid)} "
              f"({100*stopped_early/len(valid):.1f}%)")
    return results


# Checkpoint Save
def save_checkpoint(data: dict, path: Path):
    """
    Save current results to disk immediately after each system.
    If the process crashes mid-run, you don't lose completed work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[RUN] Checkpoint saved -> {path}")


# Main
def main(dry_run: bool, system: str, theta: float | None):
    """
    Full experiment orchestration.

    Args:
        dry_run : run on DRY_RUN_N questions only
        system  : 'all' | 'fscl' | 'gsal' - which system to run
        theta   : specific θ for GSAL (None = run all three)
    """
    print("\n" + "=" * 60)
    print("RUN ORCHESTRATOR")
    if dry_run:
        print("*** DRY RUN MODE ***")
    print("=" * 60)

    # Load questions
    questions = load_questions(dry_run=dry_run)

    # Initialise output structure
    if RAW_OUTPUT.exists():
        with open(RAW_OUTPUT, "r") as f:
            output = json.load(f)
        print(f"[RUN] Loaded existing checkpoint - "
              f"systems present: {list(output['results'].keys())}")
        output["meta"]["n_questions"] = len(questions)
        output["meta"]["dry_run"]     = dry_run
    else:
        output = {
            "meta": {
                "experiment":   "FSCL vs. GSAL",
                "dataset":      "HotpotQA training split (Stratified)",
                "n_questions":  len(questions),
                "dry_run":      dry_run,
                "started_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
                "theta_values": THETA_VALUES,
            },
            "results": {}
        }

    run_start = time.perf_counter()

    # FSCL
    if system in ("all", "fscl"):
        fscl_results = run_fscl_batch(questions)
        output["results"]["FSCL"] = fscl_results
        save_checkpoint(output, RAW_OUTPUT)     # Save after FSCL completes

    # GSAL (θ sweep)
    if system in ("all", "gsal"):
        thetas_to_run = [theta] if theta is not None else THETA_VALUES

        for t in thetas_to_run:
            gsal_results = run_gsal_batch(questions, theta=t)
            system_key   = f"GSAL_theta{t}"
            output["results"][system_key] = gsal_results
            save_checkpoint(output, RAW_OUTPUT)     # Save after each θ

    # Final output
    total_time = (time.perf_counter() - run_start) / 60
    output["meta"]["completed_at"]    = time.strftime("%Y-%m-%d %H:%M:%S")
    output["meta"]["total_runtime_m"] = round(total_time, 2)
    save_checkpoint(output, RAW_OUTPUT)

    # Final Summary
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Questions   : {len(questions)}")
    print(f"  Systems run : {list(output['results'].keys())}")
    print(f"  Runtime     : {total_time:.1f} minutes")
    print(f"  Output      : {RAW_OUTPUT}")
    print(f"\n Next Step  : python ragas_scorer.py "
          f"--input {RAW_OUTPUT} "
          f"--output results/scored_results.json")
    print(f"{'=' * 60}\n")


# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Experiment - FSCL vs. GSAL."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Run on {DRY_RUN_N} questions only to verify pipeline."
    )
    parser.add_argument(
        "--system",
        type=str,
        choices=["all", "fscl", "gsal"],
        default="all",
        help="Which system(s) to run (default: all)."
    )
    parser.add_argument(
        "--theta",
        type=float,
        choices=THETA_VALUES,
        default=None,
        help="Specific θ for GSAL (default: run all three)."
    )
    args = parser.parse_args()
    main(
        dry_run = args.dry_run,
        system  = args.system,
        theta   = args.theta,
    )