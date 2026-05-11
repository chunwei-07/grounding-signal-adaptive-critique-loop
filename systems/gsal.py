"""
Grounding-Signal Adaptive Critique Loop (GSAL) - Proposed System

Adaptively stops critique-refine iterations when the current answer
demonstrates sufficient grounding in the retrieved context, measured 
by retrieval-answer cosine similarity (S_ret).

Stopping condition:
    Stop when S_ret(y_t, D_t) > theta for 2 consecutive turns.

Where:
    y_t   = current answer at evaluation turn t
    D_t   = retrieved context at turn t
            In this experiment: fixed gold supporting facts from HotpotQA (HotpotQA supporting facts)
            In future work    : re-retrieved per iteration based on y_t
    S_ret = CosSim(embed(y_t), embed(D_t)) via text-embedding-3-small

The consecutive confirmation requirement (2 turns above theta) prevents 
premature stopping on a single high-scoring outlier.

Three theta values are evaluated: {0.6, 0.7, 0.8}.
Max iterations is capped at MAX_ITERATIONS = 4 to bound cost.

Mechanistic isolation principle:
    GSAL and FSCL share identical model, prompts, and context.
    The stopping condition is the sole variable under evaluation.

Usage:
    from systems.gsal import run_gsal
    result = run_gsal(question_dict, theta=0.7)
"""

import time
from dataclasses import dataclass, field
from systems.llm_client import generate_answer, generate_critique, generate_refinement
from systems.embedder import compute_sret, ConvergenceTracker

# Config
MAX_ITERATIONS = 4   # Hard cap to prevent indefinite loop
                    # FSCL does N=2, GSAL can do at most 2
                    # extra iterations.
THETA_VALUES  = [0.6, 0.7, 0.8]     # theta sweep


# Result Schema
@dataclass
class GSALResult:
    """
    Complete result record for one GSAL run on one question.

    Field names match FSCLResult exactly so analysis handles 
    both systems with identical code.

    Additional GSAL-specific fields:
        theta                   : threshold used in this run
        sret_history            : S_ret score at each iteration
        consecutive_above_theta : final consecutive count at stop
    """
    # Question metadata
    question_id:    str
    question:       str
    gold_answer:    str
    context:        str
    complexity:     str
    question_type:  str

    # GSAL-specific config
    theta: float = 0.7

    # Core outputs
    final_answer:   str = ""
    answer_history: list[str] = field(default_factory=list)

    # Iteration tracking
    iterations_performed: int = 0
    critiques_generated:  list[str] = field(default_factory=list)
    stopping_reason:      str = ""

    # Sᵣₑₜ tracking — key signal for paper figures
    sret_history:             list[float] = field(default_factory=list)
    consecutive_above_theta:  int = 0

    # Latency breakdown (ms)
    latency_generation_ms:     float = 0.0
    latency_critique_ms:       list[float] = field(default_factory=list)
    latency_refinement_ms:     list[float] = field(default_factory=list)
    latency_sret_ms:           list[float] = field(default_factory=list)
    latency_total_ms:          float = 0.0

    # Token tracking
    tokens_generation:  int = 0
    tokens_critique:    list[int] = field(default_factory=list)
    tokens_refinement:  list[int] = field(default_factory=list)
    tokens_total:       int = 0

    # Error handling
    error: str = ""


# GSAL Algorithm
def run_gsal(question_dict: dict, theta: float = 0.7) -> GSALResult:
    """
    Execute Grounding-Signal Adaptive Critique Loop on one question.

    Algorithm:
        1. Generate initial answer y_0 from context D
        2. Check S_ret(y_0) with tracker.update()
        3. WHILE iterations < MAX_ITERATIONS:
            a. IF tracker.should_stop -> BREAK
            b. Generate critique
            c. Generate refinement -> y_1
            d. Check S_ret(y_1) -> tracker.update()
            e. iterations += 1
        4. Return final answer with full metrics

    Args:
        question_dict : one question from hotpotqa_150.json
        theta         : stopping threshold theta in {0.6, 0.7, 0.8}

    Returns:
        GSALResult with all metrics populated
    """
    if theta not in THETA_VALUES:
        raise ValueError(
            f"theta must be one of {THETA_VALUES}, got {theta}. "
            f"Unlisted θ values are not reported in the paper."
        )
    
    wall_start = time.perf_counter()

    q_id        = question_dict["id"]
    question    = question_dict["question"]
    gold_answer = question_dict["answer"]
    context     = question_dict["context"]
    complexity  = question_dict["complexity"]
    q_type      = question_dict["type"]

    result = GSALResult(
        question_id   = q_id,
        question      = question,
        gold_answer   = gold_answer,
        context       = context,
        complexity    = complexity,
        question_type = q_type,
        theta         = theta,
    )

    tracker = ConvergenceTracker(theta=theta)

    try:
        # Step 1: Generate initial answer
        print(f"    [GSAL θ={theta} Generating initial answer...]")
        answer, tokens, latency = generate_answer(question, context)

        result.answer_history.append(answer)
        result.tokens_generation     = tokens
        result.latency_generation_ms = latency

        print(f"           Tokens: {tokens} | Latency: {latency:.0f}ms")
        print(f"           Answer: {answer[:80]}...")

        # Step 2: Pre-loop S_ret check on initial answer y_0
        # Gives tracker its first data point before any 
        # critique-refine cost in incurred. If y_0 already exceeds
        # theta, only one critique-refine cycle is needed to confirm.
        print(f"\n    [GSAL θ={theta}] Pre-loop S_ret check on y_0...")
        sret, sret_latency = compute_sret(answer, context)
        result.sret_history.append(sret)
        result.latency_sret_ms.append(sret_latency)
        tracker.update(sret)
        print(f"           S_ret(y_0) = {sret:.4f} (θ = {theta})")

        # Step 3: Adaptive WHILE Loop
        # Checks convergence BEFORE spending on critique each turn.
        # Consecutive confirmation requires 2 checks above theta.
        # With pre-loop check, simple questions need only 1
        # critique-refine cycle to achieve the second confirmation.
        iteration = 0
        while iteration < MAX_ITERATIONS:
            print(f"\n    [GSAL θ={theta}] Iteration {iteration + 1}/{MAX_ITERATIONS}")

            # Check BEFORE generating critique
            # If confirmed convergence, break
            if tracker.consecutive_above >= 2:
                result.stopping_reason = tracker.get_stopping_reason()
                print(f"           CONVERGED — {result.stopping_reason}")
                break

            # 3a. Generate Critique
            print(f"           Generating critique...")
            critique, c_tokens, c_latency = generate_critique(
                question, context, answer
            )
            result.critiques_generated.append(critique)
            result.tokens_critique.append(c_tokens)
            result.latency_critique_ms.append(c_latency)
            print(f"           Critique tokens: {c_tokens} | "
                  f"Latency: {c_latency:.0f}ms")
            print(f"           Critique: {critique[:80]}...")

            # 3b. Refine answer
            print(f"           Refining answer...")
            answer, r_tokens, r_latency = generate_refinement(
                question, context, answer, critique
            )
            result.answer_history.append(answer)
            result.tokens_refinement.append(r_tokens)
            result.latency_refinement_ms.append(r_latency)
            print(f"           Refinement tokens: {r_tokens} | "
                  f"Latency: {r_latency:.0f}ms")
            print(f"           Refined: {answer[:80]}...")

            # 3c. Compute S_ret on refined answer y_t+1
            print(f"             Computing S_ret...")
            sret, sret_latency = compute_sret(answer, context)
            result.sret_history.append(sret)
            result.latency_sret_ms.append(sret_latency)
            should_stop = tracker.update(sret)
            result.consecutive_above_theta = tracker.consecutive_above
            print(f"           Sᵣₑₜ(y{iteration+1}) = {sret:.4f} | "
                  f"Consecutive above θ: {tracker.consecutive_above}")
            
            iteration += 1

            # Post-refinement convergence check
            if should_stop:
                result.stopping_reason = tracker.get_stopping_reason()
                print(f"           CONVERGED — {result.stopping_reason}")
                break

        # Step 4: Handle max iterations reached
        if not result.stopping_reason:
            result.stopping_reason = tracker.get_stopping_reason()
            print(f"\n    [GSAL θ={theta}] MAX ITERATION REACHED - "
                  f"{result.stopping_reason}")
            
        # Step 5: Finalise
        result.final_answer         = answer
        result.iterations_performed = iteration

        result.tokens_total = (
            result.tokens_generation
            + sum(result.tokens_critique)
            + sum(result.tokens_refinement)
        )
        result.latency_total_ms = (time.perf_counter() - wall_start) * 1000

        print(f"\n    [GSAL θ={theta}] Complete")
        print(f"           Iterations  : {result.iterations_performed}")
        print(f"           S_ret history: {[round(s, 3) for s in result.sret_history]}")
        print(f"           Stopping    : {result.stopping_reason}")
        print(f"           Tokens      : {result.tokens_total}")
        print(f"           Latency     : {result.latency_total_ms:.0f}ms")

    except Exception as e:
        result.error            = str(e)
        result.final_answer     = result.answer_history[-1] if result.answer_history else ""
        result.latency_total_ms = (time.perf_counter() - wall_start) * 1000
        print(f"\n    [GSAL θ={theta}] ERROR on {q_id}: {e}")

    return result


# Result Serialiser
def gsal_result_to_dict(r: GSALResult) -> dict:
    """
    Serialise a GSALResult to a JSON-compatible dict.

    Matches fscl_result_to_dict schema exactly, with GSAL-specific
    fields appended. Both dicts are consumed identically by 
    ragas_scorer.py and analyse_results.py
    """
    return {
        # Metadata
        "system":        f"GSAL_theta{r.theta}",
        "question_id":   r.question_id,
        "question":      r.question,
        "gold_answer":   r.gold_answer,
        "context":       r.context,
        "complexity":    r.complexity,
        "question_type": r.question_type,

        # Core outputs
        "final_answer":   r.final_answer,
        "answer_history": r.answer_history,

        # Iteration metrics (primary for table 1 & 2)
        "iterations_performed": r.iterations_performed,
        "stopping_reason":      r.stopping_reason,

        # S_ret signal — Figure 2 (convergence curve)
        "theta":                    r.theta,
        "sret_history":             r.sret_history,
        "consecutive_above_theta":  r.consecutive_above_theta,

        # Latency metrics (supporting)
        "latency_total_ms":      r.latency_total_ms,
        "latency_generation_ms": r.latency_generation_ms,
        "latency_critique_ms":   r.latency_critique_ms,
        "latency_refinement_ms": r.latency_refinement_ms,
        "latency_sret_ms":       r.latency_sret_ms,

        # Token metrics (supporting)
        "tokens_total":      r.tokens_total,
        "tokens_generation": r.tokens_generation,
        "tokens_critique":   r.tokens_critique,
        "tokens_refinement": r.tokens_refinement,

        # Critique history (for qualitative analysis)
        "critiques_generated": r.critiques_generated,

        # RAGAS fields (populated by ragas_scorer.py)
        "faithfulness": None,

        # Error flag
        "error": r.error,
    }


