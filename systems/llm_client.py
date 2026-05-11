"""
A Centralised GPT-4o-mini wrapper

All LLM calls (generation, critique, refinement) route through here.
Easily swap model in one place if needed.
"""

import os
import time
import random
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

# Client Setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
MAX_TOKENS_GENERATION  = 300    # Initial answer + refinement
MAX_TOKENS_CRITIQUE    = 200    
TEMPERATURE_GENERATION = 0.7    # Allow creativity for answer generation
TEMPERATURE_CRITIQUE   = 0.3    # Deterministic for critique

# Retry Config
MAX_RETRIES  = 5        # Max retry attempts
BASE_DELAY_S = 2.0      # Initial wait before first retry
MAX_DELAY_S  = 60.0     # Cap on wait time
JITTER       = True     # Add randomness to avoid thundering herd


# Prompt Templates
# Used by both FSCL and GSAL
GENERATION_PROMPT = """You are a knowledgeable assistant. Answer the following question clearly and concisely, grounded strictly in the provided context.

Context:
{context}

Question:
{question}

Provide a direct, factual answer based only on the context above."""


CRITIQUE_PROMPT = """You are a critical reviewer. Evaluate the following answer against the provided context.

Context:
{context}

Question:
{question}

Current Answer:
{answer}

Identify specific gaps, inaccuracies, or missing information. Be concise and actionable."""


REFINEMENT_PROMPT = """You are a knowledgeable assistant. Improve the following answer based on the critique provided, staying strictly gronded in the context.

Context:
{context}

Question:
{question}

Current Answer:
{answer}

Critique:
{critique}

Provide an improved answer that addresses the critique while remaining faithful to the context."""


# Core LLM Call
def call_llm(prompt: str, max_tokens: int, temperature: float) -> tuple[str, int, float]:
    """
    Make a single LLM call and return (response_text, tokens_used, latency_ms).

    Returns:
        response_text : the generated string
        tokens_used   : total tokens (prompt + completion)
        latency_ms    : wall-clock latency in ms
    """
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    response_text = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens

    return latency_ms, response_text, tokens_used

def call_llm_with_retry(prompt: str, max_tokens: int, temperature: float) -> tuple[str, int, float]:
    """
    LLM call with exponential backoff retry on rate limit errors.

    Retry behaviour:
        Attempt 1: immediate
        Attempt 2: wait 2s + jitter
        Attempt 3: wait 4s + jitter
        Attempt 4: wait 8s + jitter
        Attempt 5: wait 16s + jitter
        After 5 failures: raise exception (logged by caller)

    Args:
        prompt      : full prompt string
        max_tokens  : max completion tokens
        temperature : sampling temperature
    
    Returns:
        (response_text, tokens_used, latency_ms)
    """
    delay = BASE_DELAY_S

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_llm(prompt, max_tokens, temperature)
        
        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                print(f"    [LLM] RATE LIMIT - max retries exceeded: {e}")
                raise
            wait = delay + (random.uniform(0, 1) if JITTER else 0)
            print(f"    [LLM] RATE LIMIT HIT (attempt {attempt}/{MAX_RETRIES}) "
                  f"- waiting {wait:.1f}s...")
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY_S)     # exponential backoff

        except (APITimeoutError, APIConnectionError) as e:
            if attempt == MAX_RETRIES:
                print(f"    [LLM] API ERROR - max retries exceeded: {e}")
                raise
            wait = delay + (random.uniform(0, 1) if JITTER else 0)
            print(f"    [LLM] API ERROR (attempt {attempt}/{MAX_RETRIES}) "
                  f"- waiting {wait:.1f}s...")
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY_S)


# Generate initial answer
def generate_answer(question: str, context: str) -> tuple[str, int, float]:
    """
    Generate initial answer grounded in retrieved context.

    Args:
        question : HotpotQA question string
        context  : concatenated retrieved context chunks

    Returns:
        (answer_text, tokens_used, latency_ms)
    """
    prompt = GENERATION_PROMPT.format(context=context, question=question)
    return call_llm_with_retry(prompt, MAX_TOKENS_GENERATION, TEMPERATURE_GENERATION)

# Generate critique
def generate_critique(question: str, context: str, answer: str) -> tuple[str, int, float]:
    """
    Generate critique of the current answer against context.

    Args:
        question : HotpotQA question string
        context  : retrieved context chunks
        answer   : the current answer to critique

    Returns:
        (critique_text, tokens_used, latency_ms)
    """
    prompt = CRITIQUE_PROMPT.format(
        context=context, question=question, answer=answer
    )
    return call_llm_with_retry(prompt, MAX_TOKENS_CRITIQUE, TEMPERATURE_CRITIQUE)

# Generate refinement
def generate_refinement(question: str, context: str, answer: str, critique: str,) -> tuple[str, int, float]:
    """
    Refine the current answer based on the critique.

    Args:
        question : HotpotQA question string
        context  : retrived context chunks
        answer   : the current answer to improve
        critique : the critique to address

    Returns:
        (refined_answer_text, tokens_used, latency_ms)
    """
    prompt = REFINEMENT_PROMPT.format(
        context=context, question=question, answer=answer, critique=critique
    )
    return call_llm_with_retry(prompt, MAX_TOKENS_GENERATION, TEMPERATURE_GENERATION)