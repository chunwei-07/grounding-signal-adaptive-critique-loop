"""
Computes S_ret: The retriveal-answer grounding signal used as
the adaptive stopping criterion in GSAL.

S_ret is defined as:
    S_ret(y_t, D_t) = CosSim(embed(y_t), embed(D_t))

where:
    y_t   = the current answer at iteration t
    D_t   = the concatenated retrieved context at iteration t
    embed = text-embedding-3-small via OpenAI API

Stopping condition in GSAL:
    Stop when S_ret > θ for 2 consecutive iterations.

This signals that the answer has converged to be well-grounded 
in the retrieved context and further refinement is unlikely to
add any meaningful faithfulness gains.
"""

import os
import time
import numpy as np
from openai import OpenAI

# Client Setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"


# Embedding Call
def embed(text: str) -> np.ndarray:
    """
    Embed a text string using text-embedding-3-small.

    Returns a normalised numpy array (unit vector).
    Normalising at source means cosine similarity reduces to
    a simple dot product (faster and numerically stable).

    Args:
        text: string to embed (will be truncated to 8191 tokens
              by the API if longer)

    Returns:
        numpy array of shape (1536,), unit-normalised
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)

    # Normalise to unit vector so cosine sim = dot product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


# S_ret computation
def compute_sret(answer: str, context: str) -> tuple[float, float]:
    """
    Compute S_ret(y_t, D_t): Cosine similarity between the
    current answer embedding and the context embedding.

    Args:
        answer  : current answer text y_t
        context : concatenated retrieved context chunks D_t

    Returns:
        (sret_score, embedding_latency_ms)
        sret_score is in [-1, 1] but practically in [0, 1]
        for natural language text.
    """
    start = time.perf_counter()

    vec_answer  = embed(answer)
    vec_context = embed(context)

    # Cosine similarity = dot product (both vectors are unit normalised)
    sret = float(np.dot(vec_answer, vec_context))
    latency_ms = (time.perf_counter() - start) * 1000
    return sret, latency_ms


# Convergence Checker Class
class ConvergenceTracker:
    """
    Tracks S_ret across iterations and checks the consecutive
    confirmation requirement: S_ret > θ for 2 consecutive
    iterations before stopping.

    Prevents premature stopping on a single lucky spike.

    Usage:
        tracker = ConvergenceTracker(theta=0.7)
        should_stop = tracker.update(sret_score)
    """

    def __init__(self, theta: float):
        """
        Args:
            theta: stopping threshold θ ∈ {0.6, 0.7, 0.8}
        """
        self.theta = theta
        self.history: list[float] = []      # All S_ret values
        self.consecutive_above: int = 0     # Count of consecutive > θ

    def update(self, sret: float) -> bool:
        """
        Record a new S_ret value and return True if stopping
        condition is met.

        Args:
            s_ret: S_ret score at current iteration

        Returns:
            True  = stop (S_ret > θ for 2 consecutive iters)
            False = continue refining
        """
        self.history.append(sret)
        if sret > self.theta:
            self.consecutive_above += 1
        else:
            self.consecutive_above = 0   # Reset on any dip below θ
        return self.consecutive_above >= 2
    
    def get_history(self) -> list[float]:
        """
        Return full S_ret history for logging and figures.
        """
        return self.history.copy()
    
    def get_stopping_reason(self) -> str:
        """
        Return a human-readable stopping reason for result logging.
        Useful for the 'stopping_reason' field in result JSON.
        """
        if self.consecutive_above >= 2:
            return f"converged (Sret > {self.theta} for 2 consecutive iterations)"
        return "max_iterations_reached"