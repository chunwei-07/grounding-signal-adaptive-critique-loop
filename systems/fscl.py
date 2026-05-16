"""
Fixed-Step Critique Loop (FSCL) - Baseline System

Performs exactly N=2 critique-refine iterations on every question,
regardless of query complexity or answer quality.

Inspired by the fixed-step verification design in CDA frameworks
of AlignRAG (Wei et al., 2025), FSCL serves as the controlled baseline
against which GSAL's adaptive stopping is evaluated.

The stopping condition is the ONLY difference between FSCL and GSAL.
Both systems use:
    - Identical model       : GPT-4o-mini
    - Identical prompts     : defined in llm_client.py
    - Identical RAG context : oracle supporting facts from HotpotQA
    - Identical max tokens  : defined in llm_client.py

This mechanistic isolation ensures that any observed difference in
iteration count, faithfulness, or latency is attributable solely 
to the stopping condition - not to any other system variable.

Usage:
    from systems.fscl import run_fscl
    result = run_fscl(question_dict)
"""

import time
from dataclasses import dataclass, field
from systems.llm_client import generate_answer, generate_critique, generate_refinement

# Config
FIXED_ITERATIONS = 2

# Result Schema
@dataclass
class FSCLResult:
    """
    Complete result record for one FSCL run on one question.

    All fields are logged to JSON for analysis and RAGAS scoring. 
    Field name match GSALResult exactly so analysis code handles 
    both systems identically.
    """
    # Question metadata (passed through from hotpotqa_150.json)
    question_id:   str
    question:      str
    gold_answer:   str
    context:       str
    complexity:    str      # simple/moderate/complex
    question_type: str      # bridge/comparison

    # Core outputs
    final_answer:   str = ""
    answer_history: list[str] = field(default_factory=list)

    # Iteration tracking
    iterations_performed: int = 0
    critiques_generated:  list[str] = field(default_factory=list)
    stopping_reason:      str = "fixed_n2"    # always same for FSCL

    # Latency breakdown (ms)
    latency_generation_ms: float = 0.0      # initial answer generation
    latency_critique_ms:   list[float] = field(default_factory=list)
    latency_refinement_ms: list[float] = field(default_factory=list)
    latency_total_ms:      float = 0.0      # wall-clock end-to-end

    # Token tracking
    tokens_generation: int = 0
    tokens_critique:   list[int] = field(default_factory=list)
    tokens_refinement: list[int] = field(default_factory=list)
    tokens_total:      int = 0

    # Error handling
    error: str = ""     # non-empty if something failed mid-run


# FSCL Algorithm
def run_fscl(question_dict: dict) -> FSCLResult:
    """
    Execute Fixed-Step Critique Loop on a single question.

    Algorithm: 
        1. Generate initial answer from oracle context
        2. FOR i in range(2):
            a. Generate critique of current answer
            b. Refine answer based on critique
        3. Return final answer after exactly 2 iterations

    Args:
        question_dict: one question from hotpotqa_150.json
                       keys: id, question, answer, context,
                             complexity, type

    Returns:
        FSCLResult with all metrics populated
    """
    wall_start = time.perf_counter()

    q_id        = question_dict["id"]
    question    = question_dict["question"]
    gold_answer = question_dict["answer"]
    context     = question_dict["context"]
    complexity  = question_dict["complexity"]
    q_type      = question_dict["type"]

    result = FSCLResult(
        question_id   = q_id,
        question      = question,
        gold_answer   = gold_answer,
        context       = context,
        complexity    = complexity,
        question_type = q_type,
    )

    try:
        # Step 1: Generate initial answer
        print(f"    [FSCL] Generating initial answer...")
        answer, tokens, latency = generate_answer(question, context)

        result.answer_history.append(answer)
        result.tokens_generation     = tokens
        result.latency_generation_ms = latency

        print(f"           Tokens: {tokens} | Latency: {latency:.0f}ms")
        print(f"           Answer: {answer[:80]}...")

        # Step 2: Fixed N=2 critique-refine iterations
        for i in range(FIXED_ITERATIONS):
            print(f"\n    [FSCL] Iteration {i + 1}/{FIXED_ITERATIONS}")

            # 2a. Critique
            print(f"           Generating critique...")
            critique, c_tokens, c_latency = generate_critique(
                question, context, answer
            )
            result.critiques_generated.append(critique)
            result.tokens_critique.append(c_tokens)
            result.latency_critique_ms.append(c_latency)
            print(f"           Critique tokens: {c_tokens} | Latency: {c_latency:.0f}ms")
            print(f"           Critique: {critique[:80]}...")

            # 2b. Refine
            print(f"           Refining answer...")
            answer, r_tokens, r_latency = generate_refinement(
                question, context, answer, critique
            )
            result.answer_history.append(answer)
            result.tokens_refinement.append(r_tokens)
            result.latency_refinement_ms.append(r_latency)
            print(f"           Refinement tokens: {r_tokens} | Latency: {r_latency:.0f}ms")
            print(f"           Refined: {answer[:80]}...")

        # Step 3: Finalise
        result.final_answer         = answer
        result.iterations_performed = FIXED_ITERATIONS
        result.tokens_total         = (
            result.tokens_generation
            + sum(result.tokens_critique)
            + sum(result.tokens_refinement)
        )
        result.latency_total_ms = (time.perf_counter() - wall_start) * 1000

        print(f"\n    [FSCL] RUN COMPLETED")
        print(f"           Iterations : {result.iterations_performed}")
        print(f"           Tokens     : {result.tokens_total}")
        print(f"           Latency    : {result.latency_total_ms:.0f}ms")

    except Exception as e:
        # Log the error but don't crash the batch run
        result.error            = str(e)
        result.final_answer     = result.answer_history[-1] if result.answer_history else ""
        result.latency_total_ms = (time.perf_counter() - wall_start) * 1000
        print(f"\n    [FSCL] Error on {q_id}: {e}")

    return result


# Result Serialiser
def fscl_result_to_dict(r: FSCLResult) -> dict:
    """
    Serialise an FSCLResult to a JSON-compatible dict.

    This is the canonical record written to results JSON and
    consumed by ragas_scorer.py and analyse_results.py
    """
    return {
        # Metadata
        "system"       : "FSCL",
        "question_id"  : r.question_id,
        "question"     : r.question,
        "gold_answer"  : r.gold_answer,
        "context"      : r.context,
        "complexity"   : r.complexity,
        "question_type": r.question_type,

        # Core outputs
        "final_answer"  : r.final_answer,
        "answer_history": r.answer_history,

        # Iteration metrics (primary for Table 1 & 2)
        "iterations_performed": r.iterations_performed,
        "stopping_reason"     : r.stopping_reason,

        # Latency metrics (supporting)
        "latency_total_ms"     : r.latency_total_ms,
        "latency_generation_ms": r.latency_generation_ms,
        "latency_critique_ms"  : r.latency_critique_ms,
        "latency_refinement_ms": r.latency_refinement_ms,

        # Token metrics (supporting)
        "tokens_total"     : r.tokens_total,
        "tokens_generation": r.tokens_generation,
        "tokens_critique"  : r.tokens_critique,
        "tokens_refinement": r.tokens_refinement,

        # Critique history (for qualitative analysis)
        "critiques_generated": r.critiques_generated,

        # RAGAS fields (populated by ragas_scorer.py)
        "faithfulness": None,

        # Error flag
        "error": r.error,
    }
