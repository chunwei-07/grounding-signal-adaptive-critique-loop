"""
RAGAS Faithfulness scoring pipeline for this experiment.

Runs as a SEPARATE pass after all LLM generation is complete.
Reads raw result JSON (from run_experiment.py), computes 
RAGAS Faithfulness for every question in every system, and 
writes an enriched JSON with faithfulness scores populated.

Reasons for separate pass:
    1. RAGAS failures never corrupt raw generation results.
    2. Can re-score without re-running expensive LLM calls.
    3. Clean separation between generation and evaluation.

RAGAS Faithfulness definition:
    Measures whether every claim in the generated answer is
    directly supported by the retrieved context. Score in [0, 1].
    1.0 = fully grounded, 0.0 = no grounding.

    In this experiment:
        answer   = final_answer from FSCL or GSAL
        context  = gold supporting facts from HotpotQA
        question = the HotpotQA question (for NLI context)

    This is standard RAGAS usage for open-domain QA.

Reference:
    Es et al. (2023). RAGAs: Automated Evaluation of
    Retrieval Augmented Generation. 10.18653/v1/2024.eacl-demo.16

Usage:
    python ragas_scorer.py \
        --input  results/raw_results.json \
        --output results/scored_results.json
"""

import json
import time
import argparse
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

# Config
RAGAS_LLM_MODEL       = "gpt-4o-mini"
BATCH_SIZE            = 5


# RAGAS Client Setup
def build_ragas_config() -> Faithfulness:
    """
    Build RAGAS evaluator with GPT-4o-mini as the judge LLM.

    RAGAS uses an LLM internally to decompose the answer into 
    atomic claims and verify each claim against the context.
    Using the same model to keep the evaluation self-consistent.
    Uses llm_factory with AsyncOpenAI client - required by new API.

    Return:
        Configured faithfulness metric object
    """
    client = AsyncOpenAI()
    llm    = llm_factory(RAGAS_LLM_MODEL, client=client)
    return Faithfulness(llm=llm)


# Single Question Scorer
def score_one(
    question:     str,
    answer:       str,
    context:      str,
    ragas_metric: Faithfulness,
) -> tuple[float, str]:
    """
    Compute RAGAS Faithfulness for a single question-answer pair.

    RAGAS expects a HuggingFace Dataset with these columns:
        question:  str
        answer:    str (the generated answer to evaluate)
        context:   list[str]  (retrieved chunks as a list)
        reference: str (gold answer, used by other metrics, not faithfulness)

    Args:
        question:     the question text
        answer:       the generated final answer to evaluate
        context:      concatenated context string (will be
                      wrapped in a list for RAGAS schema)
        ragas_metric: configured faithfulness metric

    scorer.score() is the synchronous wrapper around ascore().
    Result score is accessed via result.value.

    Returns:
        (faithfulness_score, error_string)
        faithfulness_score = -1.0 on error
    """
    try:
        result = ragas_metric.score(
            user_input         = question,
            response           = answer,
            retrieved_contexts = [context], 
        )
        return float(result.value), ""
    except Exception as e:
        return -1.0, str(e)
    

# Batch Scorer
def score_results(
    records:      list[dict],
    ragas_metric: Faithfulness,
    system_label: str,
) -> list[dict]:
    """
    Score all records for one system, adding faithfulness field.

    Processes in batches of BATCH_SIZE to manage rate limits.
    Records with errors in generation (error field non-empty)
    are skipped and assigned faithfulness = -1.0.

    Args:
        records:      list of result dicts from run_experiment.py
        ragas_metric: configured faithfulness metric
        system_label: 'FSCL' or 'GSAL_theta0.X' for logging

    Returns:
        Records with faithfulness field populated
    """
    total   = len(records)
    scored  = 0
    errors  = 0
    skipped = 0

    print(f"\n[RAGAS] Scoring {total} records for {system_label}...")

    for i, record in enumerate(records):
        # Skip records that failed during generation
        if record.get("error", ""):
            print(f"  [{i+1}/{total}] Skipping - generation error: "
                  f"{record['error'][:60]}")
            record["faithfulness"] = -1.0
            skipped += 1
            continue

        # Skip records with empty final answer
        if not record.get("final_answer", "").strip():
            print(f"  [{i+1}/{total}] Skipping - empty final answer")
            record["faithfulness"] = -1.0
            skipped += 1
            continue

        # Score
        score, error = score_one(
            question=record["question"],
            answer=record["final_answer"],
            context=record["context"],
            ragas_metric=ragas_metric,
        )

        record["faithfulness"] = score

        if error:
            print(f"  [{i+1}/{total}] RAGAS Error: {error[:80]}")
            errors += 1
        else:
            print(f"  [{i+1}/{total}] {record['question_id'][:12]}... | "
                  f"complexity={record['complexity']:8s} | "
                  f"iterations={record['iterations_performed']} | "
                  f"faithfulness={score:.4f}")
            scored += 1

        # Rate limit buffer between questions
        # RAGAS makes multiple LLM calls per question internally
        if (i + 1) % BATCH_SIZE == 0 and (i + 1) < total:
            print(f"\n  [RAGAS] Batch {(i+1)//BATCH_SIZE} complete - "
                  f"pausing 5s for rate limit buffer...")
            time.sleep(10)

    print(f"\n[RAGAS] {system_label} scoring complete:")
    print(f"        Scored  : {scored}")
    print(f"        Errors  : {errors}")
    print(f"        Skipped : {skipped}")

    return records


# Aggregate States (for quick check)
def print_aggregate_stats(records: list[dict], system_label: str):
    """
    Print quick aggregate faithfulness stats after scoring.
    Full analysis in analyse_results.py.
    """
    valid = [r["faithfulness"] for r in records if r["faithfulness"] >= 0]

    if not valid:
        print(f"[RAGAS] {system_label}: No valid scores to aggregate.")
        return
    
    avg = sum(valid) / len(valid)
    mn  = min(valid)
    mx  = max(valid)

    # Per-complexity breakdown
    for tier in ["simple", "moderate", "complex"]:
        tier_scores = [
            r["faithfulness"] for r in records
            if r["complexity"] == tier and r["faithfulness"] >= 0
        ]
        if tier_scores:
            tier_avg = sum(tier_scores) / len(tier_scores)
            print(f"        {tier:8s}: n={len(tier_scores):3d} | "
                  f"avg={tier_avg:.4f}")
            
    print(f"        Overall : n={len(valid):3d} | "
          f"avg={avg:.4f} | min={mn:.4f} | max={mx:.4f}")
    

# Main Pipeline
def main(input_path: str, output_path: str):
    """
    Full scoring pipeline:
        1. Load raw results JSON
        2. Build RAGAS config
        3. Score each system's records
        4. Write enriched JSON with faithfulness populated
    """
    print("\n" + "=" * 60)
    print("RAGAS FAITHFULNESS SCORER")
    print("=" * 60)

    # Load raw results
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"[RAGAS] Loaded {input_path}")
    print(f"[RAGAS] Systems found: {list(data['results'].keys())}")

    # RAGAS Evaluator
    print(f"\n[RAGAS] Initialising RAGAS with {RAGAS_LLM_MODEL}...")
    ragas_metric = build_ragas_config()
    print(f"[RAGAS] Ready.")

    # Score each system
    total_scored = 0
    for system_label, records in data["results"].items():
        print(f"\n{'=' * 60}")
        print(f"Scoring system: {system_label}")
        print(f"{'=' * 60}")

        scored_records = score_results(records, ragas_metric, system_label)
        data["results"][system_label] = scored_records

        print_aggregate_stats(scored_records, system_label)
        total_scored += len(scored_records)

    # Write enriched results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data["ragas_meta"] = {
        "scorer_model": RAGAS_LLM_MODEL,
        "metric":       "faithfulness",
        "citation":     "Es et al., 10.18653/v1/2024.eacl-demo.16",
        "scored_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Scored {total_scored} records total.")
    print(f"    Output: {output_path}")
    print(f"{'=' * 60}\n")


# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score experiment results with RAGAS Faithfulness."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/raw_results.json",
        help="Path to raw results JSON from run_experiment.py"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scored_results.json",
        help="Path to write enriched results JSON"
    )
    args = parser.parse_args()
    main(args.input, args.output)