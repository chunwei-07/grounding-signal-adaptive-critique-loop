"""
Loads the HotpotQA training split and stratifies 150 questions
by native difficulty level for the experiment.

Why training split:
    The HotpotQA distractor dev set contains only 'hard'
    multi-hop questions by construction (Yang et al., 2018,
    Table 1). Difficulty-stratified sampling therefore requires
    the training split, which contains all three native levels:
        easy   — single-hop (17,972 examples)
        medium — multi-hop (56,814 examples)
        hard   — hard multi-hop (15,661 examples)

No model in this study is fine-tuned on HotpotQA.
All sampled questions are used purely for inference evaluation.

Stratification:
    Simple   (50): level == 'easy'
    Moderate (50): level == 'medium'
    Complex  (50): level == 'hard'

Output: data/hotpotqa_150.json

Run once before any experiment:
    python load_hotpotqa.py
"""

import json
import random
import urllib.request
from pathlib import Path

# Config
HOTPOTQA_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/"
    "hotpot_train_v1.1.json"
)

DATA_DIR = Path(__file__).resolve().parent
OUT_FILE = DATA_DIR / "hotpotqa_150.json"

N_SIMPLE   = 50
N_MODERATE = 50
N_COMPLEX  = 50
TOTAL      = N_SIMPLE + N_MODERATE + N_COMPLEX    # 150

RANDOM_SEED = 42     # Reproducibility


# Download
def download_hotpotqa() -> list[dict]:
    """
    Download HotPotQA training set (~540MB).
    Cached locally after first download.

    Returns:
        List of raw HotpotQA questions dicts
    """
    raw_path = DATA_DIR / "hotpotqa_train_raw.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        print(f"[load] Cached file already exists at {raw_path}, skipping download.")
        with open(raw_path, "r") as f:
            return json.load(f)
        
    print(f"[load] Downloading HotpotQA validation set (~540MB)...")
    print(f"       This may take a few minutes.")
    urllib.request.urlretrieve(HOTPOTQA_URL, raw_path)
    print(f"[load] Downloaded to {raw_path}")

    with open(raw_path, "r") as f:
        return json.load(f)
    

# Context Builder
def build_context(item: dict) -> str:
    """
    Build a single context string from HotpotQA supporting facts.

    HotpotQA provides 'supporting_facts' as a list of 
    [title, sentence_index] pairs, and 'context' as a list of 
    [title, [sentence0, sentence1, ...]] pairs.

    Only the gold supporting sentences will be extracted as the 
    retrieved context. In the paper this is framed as 'oracle 
    retrieval', a controlled simplification that isolates the 
    stopping condition as the sole variable under evaluation.

    Args:
        item : raw HotpotQA question dict

    Returns:
        Concatenated supporting sentences as a single string
    """
    # Build lookup: title -> list of sentences
    context_lookup: dict[str, list[str]] = {}
    for title, sentences in item["context"]:
        context_lookup[title] = sentences

    # Extract supporting sentences in declared order, deduplicated
    supporting_sentences = []
    seen = set()

    for title, sent_idx in item["supporting_facts"]:
        key = (title, sent_idx)
        if key in seen:
            continue
        seen.add(key)

        if title in context_lookup:
            sentences = context_lookup[title]
            if sent_idx < len(sentences):
                sentence = sentences[sent_idx].strip()
                if sentence:
                    supporting_sentences.append(sentence)

    return " ".join(supporting_sentences)


# Complexity Classification
def classify_complexity(item: dict) -> str:
    """
    Classify a HotpotQA question using its native level field.

    Mapping (Yang et al., 2018):
        Mapping (Yang et al., 2018):
        'easy'   -> 'simple'    single-hop
        'medium' -> 'moderate'  multi-hop, solvable by 2018 QA models
        'hard'   -> 'complex'   multi-hop, NOT solvable by 2018 QA models

    The native label is used directly.
    This is the correct and citable approach.

    Args:
        item : raw HotpotQA question dict

    Returns:
        'simple' | 'moderate' | 'complex'
    """
    level = item.get("level", "medium")

    mapping = {
        "easy":   "simple",
        "medium": "moderate",
        "hard":   "complex",
    }

    return mapping.get(level, "moderate")   # Fallback to moderate if unknown


# Stratified Sampling
def stratify_and_sample(raw_data: list[dict]) -> list[dict]:
    """
    Stratify raw HotpotQA questions by native level and sample
    N_SIMPLE + N_MODERATE + N_COMPLEX = 150 questions.

    Args:
        raw_data: full HotpotQA validation set

    Returns:
        List of 150 processed question dicts, shuffled
    """
    random.seed(RANDOM_SEED)

    buckets: dict[str, list[dict]] = {
        "simple":   [],
        "moderate": [],
        "complex":  [],
    }

    print(f"[load] Classifying {len(raw_data):,} questions by native level...")

    for item in raw_data:
        tier = classify_complexity(item)
        buckets[tier].append(item)

    print(f"[load] Native level distribution in training set:")
    for tier, items in buckets.items():
        print(f"       {tier:10s}: {len(items):,} questions")

    # Verify sufficient questions in each bucket
    targets = [("simple", N_SIMPLE), ("moderate", N_MODERATE), ("complex", N_COMPLEX)]
    for tier, needed in targets:
        available = len(buckets[tier])
        if available < needed:
            raise ValueError(
                f"Insufficient {tier} questions: need {needed}, "
                f"have {available}. The dev set may differ from expected."
            )
        
    # Sample from each bucket
    sampled_simple   = random.sample(buckets["simple"], N_SIMPLE)
    sampled_moderate = random.sample(buckets["moderate"], N_MODERATE) 
    sampled_complex  = random.sample(buckets["complex"], N_COMPLEX)

    all_sampled = sampled_simple + sampled_moderate + sampled_complex
    random.shuffle(all_sampled)
    return all_sampled


# Formatter
def format_question(item: dict, index: int) -> dict:
    """
    Format a raw HotpotQA into the clean schema used by 
    both FSCL and GSAL throughout the experiment.

    Output schema:
    {
        "id"         : str  — original HotpotQA _id
        "index"      : int  — position in our 150-question set
        "question"   : str  — the question text
        "answer"     : str  — gold answer (RAGAS reference)
        "context"    : str  — concatenated supporting sentences
        "complexity" : str  — 'simple' | 'moderate' | 'complex'
        "level"      : str  — original HotpotQA level label
        "type"       : str  — 'comparison' | 'bridge'
        "n_facts"    : int  — number of supporting facts
    }
    """
    return {
        "id":         item["_id"],
        "index":      index,
        "question":   item["question"],
        "answer":     item["answer"],
        "context":    build_context(item),
        "complexity": classify_complexity(item),
        "level":      item.get("level", "unknown"),
        "type":       item.get("type", "unknown"),
        "n_facts":    len(item.get("supporting_facts", [])),
    }


# Main
def main():
    print("\n" + "=" * 60)
    print("HOTPOTQA LOADER")
    print("=" * 60)

    # Step 1: Download
    raw_data = download_hotpotqa()
    print(f"[load] Total questions in training set: {len(raw_data):,}")

    # Step 2: Stratify and sample
    sampled = stratify_and_sample(raw_data)

    # Step 3: Format
    processed = []
    for i, item in enumerate(sampled):
        formatted = format_question(item, index=i)
        processed.append(formatted)

    # Step 4: Sanity Check - flag empty contexts
    empty_context = [q for q in processed if not q["context"].strip()]
    if empty_context:
        print(f"[warn] {len(empty_context)} questions have empty context.")
        for q in empty_context:
            print(f"       ID: {q['id']} | level: {q['level']}")

    # Step 5: Save
    output = {
        "meta": {
            "source":      "HotpotQA training set",
            "citation":    "Yang et al., EMNLP 2018",
            "url":         HOTPOTQA_URL,
            "note":        "Training split used for stratification; no model fine-tuned on this data.",
            "random_seed": RANDOM_SEED,
            "total":       len(processed),
            "n_simple":    sum(1 for q in processed if q["complexity"] == "simple"),
            "n_moderate":  sum(1 for q in processed if q["complexity"] == "moderate"),
            "n_complex":   sum(1 for q in processed if q["complexity"] == "complex"),
        },
        "questions": processed,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Step 6: Report
    print(f"\n[load] Saved {len(processed)} questions to {OUT_FILE}")
    print(f"\n[load] Final stratification:")
    print(f"       Simple   (easy)  : {output['meta']['n_simple']}")
    print(f"       Moderate (medium): {output['meta']['n_moderate']}")
    print(f"       Complex  (hard)  : {output['meta']['n_complex']}")

    sample = processed[0]
    print(f"\n[load] Sample question:")
    print(f"       ID         : {sample['id']}")
    print(f"       Level      : {sample['level']} -> {sample['complexity']}")
    print(f"       Type       : {sample['type']}")
    print(f"       n_facts    : {sample['n_facts']}")
    print(f"       Question   : {sample['question']}")
    print(f"       Answer     : {sample['answer']}")
    print(f"       Context    : {sample['context'][:120]}...")

    print(f"\n{'=' * 60}")
    print("COMPLETED. Run once before any experiment.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
