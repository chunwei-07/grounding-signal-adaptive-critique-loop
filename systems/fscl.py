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
    
    """