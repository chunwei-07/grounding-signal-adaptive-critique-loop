"""
A Centralised GPT-4o-mini wrapper

All LLM calls (generation, critique, refinement) route through here.
Easily swap model in one place if needed.
"""

import os
import time
from openai import OpenAI

# Client Setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
MAX_TOKENS_GENERATION  = 300    # Initial answer + refinement
MAX_TOKENS_CRITIQUE    = 200    
TEMPERATURE_GENERATION = 0.7    # Allow creativity for answer generation
TEMPERATURE_CRITIQUE   = 0.3    # Deterministic for critique


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
    return call_llm(prompt, MAX_TOKENS_GENERATION, TEMPERATURE_GENERATION)

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
    return call_llm(prompt, MAX_TOKENS_CRITIQUE, TEMPERATURE_CRITIQUE)

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
    return call_llm(prompt, MAX_TOKENS_GENERATION, TEMPERATURE_GENERATION)