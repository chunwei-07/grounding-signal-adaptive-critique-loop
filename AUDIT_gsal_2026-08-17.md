# Audit: gsal → ECHO reusability (2026-08-17)

Read-only audit. No files in this repo were modified, executed, or committed
other than this report.

## 1. Extend or rewrite

**Rewrite the stopping logic and loop driver; lift the peripheral components.**
GSAL's `ConvergenceTracker` (systems/embedder.py) already implements a
consecutive-confirmation counter and is a clean, separable class — that pattern
generalises well. But it is single-signal (`update(sret: float)`), and the loop
in `run_gsal()` (systems/gsal.py) interleaves the stop-check with LLM calls,
history logging, and print statements inline in one `while` block, with no
existing multi-signal composition, no entropy sampling, and no `S_delta`
(previous-grounding-value) computation. ECHO needs a different tracker class,
a k-sampling generation step that doesn't exist here at all, and a loop body
restructured to compute and pass three signals instead of one. The LLM client,
embedder's `embed()`/cosine-similarity primitive, RAGAS scorer, and result-dict
pattern are reusable close to as-is; the loop and stopping condition are not.

## 2. Reusability table

| Component | Verdict | Notes |
|---|---|---|
| LLM client / API wrapper | ADAPT | `systems/llm_client.py` — `call_llm_with_retry` (exponential backoff, jitter) is a solid provider-specific pattern, but it's hardcoded to `OpenAI` client and `gpt-4o-mini`, with three fixed prompt templates (generate/critique/refine) baked in as module-level strings. Reusable as a retry/call-wrapper shape; needs re-pointing at ECHO's provider/model and prompts rewritten for the interview-coaching domain. Not provider-agnostic — no abstraction layer over the OpenAI SDK. |
| Embedding computation | LIFT | `systems/embedder.py::embed()` — generic: takes text, returns unit-normalised vector via `text-embedding-3-small`. No HotpotQA-specific logic. Model name is the only thing to potentially change. |
| `S_ret` computation | LIFT | `compute_sret()` in the same file — pure function of (answer, context) → cosine similarity. Directly reusable as ECHO's `S_ret` term. |
| Critique loop skeleton | REWRITE | The `while` loop in `run_gsal()` (systems/gsal.py:178-229) hardcodes: single-answer generation (no k-sampling), a two-step critique→refine call sequence, and inline calls to `tracker.update()`/`tracker.consecutive_above` mixed with token/latency bookkeeping and `print()` calls. There is no seam where a multi-signal check could be substituted without touching the loop body itself — see Q3. |
| Stopping decision | ADAPT | `ConvergenceTracker` (embedder.py:88-142) is already a separable class with its own `update()`/`get_stopping_reason()` — the *shape* (stateful tracker, consecutive-count field) is directly reusable as a design pattern. Its `update(sret: float)` signature and its `sret > theta` single-signal comparison are not; ECHO needs `update(C_t: float)` fed by three pre-computed inputs and would need a second class or a subclass for entropy-collapse guardrail logic. |
| RAGAS scoring | LIFT | `evaluation/ragas_scorer.py` is fully decoupled — reads any result JSON with `question`/`final_answer`/`context` fields, runs Faithfulness async, writes back. Domain-agnostic; works unchanged against ECHO's schema if the same field names are kept. |
| Retrieval layer | REWRITE (doesn't exist) | There is no vector-store interface anywhere in this repo. `context` is a fixed string pulled directly from HotpotQA's gold supporting facts in `question_dict["context"]` (see run_experiment.py / data/hotpotqa_150.json) — no retrieval call, no re-retrieval per iteration (the gsal.py docstring explicitly flags this as "future work"). ECHO must build a retrieval layer from scratch. |
| Experiment runner | ADAPT | `run_experiment.py` — the batch-loop/checkpoint-save/dry-run pattern (`load_questions`, `save_checkpoint`, per-system batch runners, inter-question rate-limit delay) is reusable structure. It's wired to HotpotQA-specific `data/hotpotqa_150.json`, the `theta` sweep, and FSCL/GSAL system names throughout — needs rewriting for ECHO's config surface (weights, tau, collapse_rate, k) and corpus. |
| Logging / results schema | LIFT (as a pattern) | `gsal_result_to_dict()` / `fscl_result_to_dict()` show a clean flat-dict-per-run schema with parallel-list history fields (`sret_history`, `latency_critique_ms`, etc.) explicitly designed so downstream analysis code is system-agnostic. ECHO can copy this shape directly, adding fields for `H_t`, `S_delta`, `C_t` history and entropy-collapse flags. |
| Config / threshold declaration | ADAPT | `THETA_VALUES = [0.6, 0.7, 0.8]` and `MAX_ITERATIONS = 4` are plain module-level constants in systems/gsal.py (not CLI args, not env, not a config file) validated with a `raise ValueError` in `run_gsal()`. Pattern is reusable but ECHO has more parameters (`w1,w2,w3`, `tau`, `collapse_rate`) that don't fit a single flat list-sweep constant — needs restructuring into a config object. |

## Q1 — File inventory

- **systems/gsal.py** (319 lines) — GSAL algorithm. `GSALResult` dataclass, `run_gsal()`, `gsal_result_to_dict()`. Depends on `systems.llm_client` (generate_answer/critique/refinement) and `systems.embedder` (compute_sret, ConvergenceTracker).
- **systems/fscl.py** (224 lines) — Baseline fixed-N=2 critique loop. `FSCLResult` dataclass, `run_fscl()`, `fscl_result_to_dict()`. Depends only on `systems.llm_client`. Structurally near-identical to gsal.py by design (documented as intentional "mechanistic isolation").
- **systems/llm_client.py** (194 lines) — Centralised OpenAI wrapper (`gpt-4o-mini`). Holds all three prompt templates, `call_llm`/`call_llm_with_retry` with exponential backoff, and the three generation functions (`generate_answer`, `generate_critique`, `generate_refinement`). Depends on the `openai` SDK and `OPENAI_API_KEY` env var.
- **systems/embedder.py** (141 lines) — `embed()` (OpenAI `text-embedding-3-small`, unit-normalised), `compute_sret()` (cosine sim as dot product), and `ConvergenceTracker` (stateful consecutive-above-threshold counter). Depends on `openai`, `numpy`.
- **run_experiment.py** (321 lines) — CLI orchestrator. Loads `data/hotpotqa_150.json`, runs FSCL then GSAL θ-sweep in batches, checkpoints to `results/raw_results.json` after each system/theta. Depends on systems.fscl, systems.gsal.
- **evaluation/ragas_scorer.py** (333 lines) — Separate post-hoc pass. Reads raw results JSON, computes RAGAS Faithfulness per record via `ragas.metrics.collections.Faithfulness` and an `AsyncOpenAI` client, writes `results/scored_results.json`. Depends on `ragas`, `openai`.
- **data/load_hotpotqa.py** (302 lines) — One-time downloader/stratifier: pulls HotpotQA train split from a CMU URL, stratifies 150 questions (50 easy/medium/hard) into `data/hotpotqa_150.json`.
- **analyse_results.py** (662 lines, root) — Produces paper tables/figures from `results/scored_results.json` (not part of the five named files; found during inventory). Depends on numpy, pandas, matplotlib, scipy.
- **iteration_distribution.py** (17 lines, root, untracked-adjacent but present) — small helper script.
- **per_tier_wilcoxon.py** (173 lines, root) — **untracked**, not in git history. Wilcoxon signed-rank analysis per complexity tier.
- **sret_distribution.py** (170 lines, root) — **untracked**, not in git history. Produces the S_ret distribution figure.

Data / results / other, not analysed further per instructions:
- `data/hotpotqa_150.json`, `data/hotpotqa_train_raw.json` — data files (gitignored per `.gitignore`, but `hotpotqa_150.json` is present on disk; `hotpotqa_train_raw.json` is explicitly gitignored by name).
- `results/raw_results.json`, `results/scored_results.json`, `results/batch_summary.txt` — result logs (batch_summary.txt shows as modified in git status).
- `results/figures/*.png` (5 files, one untracked: `sret_distribution.png`) — figures.
- `results/tables/*.csv` (3 files) — tables.
- `.venv/`, `uv.lock`, `.python-version` — environment/toolchain, not source.

## Q3 — Coupling assessment of the stopping logic

**1. Separable function or inline logic?**
Partially separable. The *decision primitive* lives in `ConvergenceTracker.update()` (systems/embedder.py:110-127), which is a genuine method: it takes a float and returns a bool, with no side effects beyond mutating its own `self.history`/`self.consecutive_above`. That part is clean. But the *use* of the decision is inline in `run_gsal()`'s while-loop: `if tracker.consecutive_above >= 2:` (gsal.py:184) is a bare `if` embedded directly in the loop body, duplicated as a second check (`if should_stop:` at gsal.py:226) after the post-refinement update. There are two separate inline stop-check sites per loop, not one call.

**2. Does the stopping code do anything besides decide?**
The `ConvergenceTracker` itself is decision-only — `update()` only appends to `self.history` and increments/resets its own counter. But the *code around* the stop-checks is heavily coupled to loop mechanics: quoted evidence —
```python
if tracker.consecutive_above >= 2:
    result.stopping_reason = tracker.get_stopping_reason()
    print(f"           CONVERGED — {result.stopping_reason}")
    break
```
(gsal.py:184-187) — this single "decision" site also writes to the result object (`result.stopping_reason`), prints, and controls loop flow (`break`). It doesn't itself trigger retrieval or LLM calls, but it directly mutates result-object state and console output as part of the branch, and `break`/`iteration += 1` (loop counters) are entangled with it inline (gsal.py:223, 229).

**3. How many places would need to change for a two-consecutive multi-signal rule?**
At minimum: (a) `ConvergenceTracker.__init__`/`update` in embedder.py — replace `theta: float` and `sret: float` with a multi-signal input and weighted composite; (b) the pre-loop check at gsal.py:166-171 (`compute_sret` call + `tracker.update`) — needs to become a multi-signal pre-loop computation (entropy over k samples, S_delta placeholder for iteration 0); (c) the first inline stop-check at gsal.py:184-187; (d) the critique/refine block itself, since ECHO's `S_delta` requires diffing grounding scores across iterations — no such diff exists today; (e) the second inline stop-check at gsal.py:226-229; (f) `GSALResult` dataclass fields (gsal.py:46-98) — `sret_history`/`consecutive_above_theta` would need `entropy_history`, `sdelta_history`, `composite_history`, collapse-guardrail state; (g) `gsal_result_to_dict()` serialisation (gsal.py:265-317) — same fields mirrored. That's at least 5 code sites plus 2 schema sites — the two-consecutive-turns mechanism already exists as a pattern (`consecutive_above`) but is wired specifically to a single float comparison, not a pluggable predicate.

**4. Does the loop retain per-iteration history?**
Yes, partially. `result.answer_history` (all past answers) and `result.sret_history` (all past S_ret values) are retained as lists (gsal.py:72, 80), and `ConvergenceTracker.history` duplicates the S_ret list independently. This is exactly what ECHO's `S_delta` (previous iteration's grounding value) needs — `sret_history[-2]` vs `sret_history[-1]` would give it directly, though no such delta is currently computed anywhere. The "previous decision" needed for the two-consecutive-turns rule already exists as `tracker.consecutive_above`, so that piece of memory is present and reusable in spirit.

**5. Does the loop generate multiple samples per iteration (k-sampling)?**
No. Confirmed by grep across `systems/` — the only samples-related code is `MAX_RETRIES`/`random.uniform` jitter in the retry backoff (llm_client.py:23-26, 128), which is unrelated to sampling multiple answers. Each iteration calls `generate_answer`/`generate_refinement` exactly once (temperature 0.7, single completion — `call_llm` requests one `chat.completions.create` call with no `n=` parameter). k-sampling for semantic entropy is entirely absent and would need to be built new — this is the single largest gap for ECHO.

**Verdict: extend or rewrite?**
Rewrite the loop and stopping predicate, extend everything around them. The `ConvergenceTracker` pattern (stateful class, `update()`→bool, consecutive-count memory) and the history-retention idea are worth carrying forward conceptually, but the actual stop-checks are inlined twice in the loop body with result-mutation and printing fused in, there is no k-sampling infrastructure for entropy at all, and no S_delta computation exists. Given GSAL's loop is only ~230 lines and mechanistically simple, and ECHO needs a materially different generation step (k samples), a new signal (entropy via entailment clustering), a delta signal, and a guardrail, treating the loop as "extend GSAL" would mean touching most of its lines anyway — better characterised as a rewrite that reuses GSAL's LLM client, embedder primitive, RAGAS scorer, and result-schema shape as building blocks.

## Q4 — Threshold and configuration handling

`theta` is declared as a module-level constant `THETA_VALUES = [0.6, 0.7, 0.8]` in `systems/gsal.py:42`, validated inside `run_gsal()` via `if theta not in THETA_VALUES: raise ValueError(...)` (gsal.py:123-127). It is passed into `run_gsal(question_dict, theta=0.7)` as a function argument, and exposed as a `--theta` CLI flag in `run_experiment.py` (choices restricted to `THETA_VALUES`, argparse). `MAX_ITERATIONS = 4` is a similar hardcoded module constant (gsal.py:39). No `.env`/environment-variable threshold reads.

The θ **sweep is the sensitivity analysis itself** — `run_experiment.py` runs all three theta values in its main loop (`thetas_to_run = [theta] if theta is not None else THETA_VALUES`, run_experiment.py:265) and `analyse_results.py` produces `table3_theta_sweep.csv` and `fig4_theta_tradeoff.png` explicitly for this. This entire sweep mechanism is legitimate for GSAL's paper but per the task brief must **not** be carried into ECHO — ECHO's `w1/w2/w3`, `tau`, and `collapse_rate` are a materially different, higher-dimensional config surface and would need its own (probably non-exhaustive) tuning approach, not a copy of this three-value grid.

## Q5 — Interface and data contracts

`run_gsal(question_dict: dict, theta: float = 0.7) -> GSALResult` and `run_fscl(question_dict: dict) -> FSCLResult` (gsal.py:101, fscl.py:78). Input `question_dict` shape (from `data/hotpotqa_150.json`, consumed at gsal.py:131-136 / fscl.py:99-104): `{"id": str, "question": str, "answer": str, "context": str, "complexity": str, "type": str}`.

Output dataclasses (`GSALResult`, `FSCLResult`) are near-identical by explicit design ("field names match ... exactly so analysis handles both systems with identical code" — gsal.py:51-52) and are serialised via `gsal_result_to_dict()`/`fscl_result_to_dict()` into flat dicts with these field groups: metadata (`system`, `question_id`, `question`, `gold_answer`, `context`, `complexity`, `question_type`), core outputs (`final_answer`, `answer_history`), iteration metrics (`iterations_performed`, `stopping_reason`), GSAL-only signal fields (`theta`, `sret_history`, `consecutive_above_theta`), latency breakdown (5 fields, several as per-iteration lists), token breakdown (4 fields), `critiques_generated` list, a `faithfulness` field (populated later, initially `None`), and `error`. Per-run records are collected into a top-level dict written by `run_experiment.py`: `{"meta": {...}, "results": {"FSCL": [...], "GSAL_theta0.6": [...], ...}}`. `ragas_scorer.py` mutates records in place, only adding/overwriting the `faithfulness` field and appending a top-level `ragas_meta` block — it makes no other assumptions about record shape beyond `question`/`final_answer`/`context`/`question_id`/`complexity`/`iterations_performed`/`error`.

## Q6 — Dependencies and stack

From `pyproject.toml`: Python `>=3.13` (`.python-version` pins `3.13`); dependencies `matplotlib>=3.10.9`, `numpy>=2.4.4`, `openai>=2.35.1`, `pandas>=3.0.2`, `ragas>=0.4.3`, `scipy>=1.17.1`. Exact resolved versions are locked in `uv.lock` (not inspected in detail — large lockfile, versions above are the pyproject floors). LLM provider: OpenAI, model `gpt-4o-mini` (systems/llm_client.py:16, evaluation/ragas_scorer.py:46). Embedding model: OpenAI `text-embedding-3-small` (systems/embedder.py:28). Vector store: none present (see Q2 retrieval-layer row). RAGAS: `ragas>=0.4.3`, used via `ragas.llms.llm_factory` and `ragas.metrics.collections.Faithfulness` with an `AsyncOpenAI` client (evaluation/ragas_scorer.py:42-43, 63-65) — the code comment notes this `llm_factory`/`AsyncOpenAI` pattern is "required by new API," implying a prior RAGAS API this code has already migrated away from. `.env` present at repo root; key names only: `OPENAI_API_KEY` (values not printed).

Constraints on reuse: hardcoded OpenAI SDK client construction in two separate files (llm_client.py, embedder.py, ragas_scorer.py) rather than a shared/injectable client — provider swap would touch three files. `openai>=2.35.1` and `ragas>=0.4.3` are both recent floors, not obviously stale. Python 3.13 floor is fairly new and could constrain environments that need to pin older Python.

## Q7 — Freeze state

- **Current branch:** `main` (tracking `origin/main`, up to date).
- **All branches:** `main` (local), `remotes/origin/main`. No other branches.
- **Tags:** none present.
- **Total commit count:** 5.
- **First commit:** 2026-05-06 (`283291f`, "Initial commit").
- **Most recent commit:** 2026-05-16 (`c6d469f`, "Done experiment").
- **Last 10 commits** (all commits, repo has only 5):
  - `c6d469f` 2026-05-16 — Done experiment
  - `bfeca5e` 2026-05-11 — Created RAGAS Scorer and added required packages
  - `6b2d3a5` 2026-05-09 — Created FSCL system
  - `89dbe12` 2026-05-09 — Created LLM Client, Embedder, and HotpotQA downloader
  - `283291f` 2026-05-06 — Initial commit

**`git status` — NOT clean.** ⚠
- Modified, not staged: `results/batch_summary.txt`
- Untracked: `per_tier_wilcoxon.py`, `results/figures/sret_distribution.png`, `sret_distribution.py`

This repo is described as a frozen published artefact but currently has one modified tracked file and three untracked files sitting in the working tree, unrelated to this audit (none were touched during this audit). Flagging per instructions only — not resolved or cleaned.

## Q8 — Observations

- The three untracked files (`per_tier_wilcoxon.py`, `sret_distribution.py`, `results/figures/sret_distribution.png`) and the modified `results/batch_summary.txt` represent post-hoc analysis added after the last commit — worth deciding whether they belong in the frozen history before the October 2026 camera-ready deadline.
- `data/hotpotqa_train_raw.json` is gitignored by name but `data/hotpotqa_150.json` (the derived stratified sample actually used by experiments) is not gitignored and its presence/absence in git history was not further checked beyond the file listing.
- The RAGAS scorer comment flags a prior API migration ("required by new API") — noted per instructions, not investigated further.

## Cannot determine

- Exact resolved dependency versions beyond the `pyproject.toml` floors (would require parsing the full `uv.lock`, not done).
- Whether `data/hotpotqa_150.json` is tracked in git history or gitignored/untracked — not explicitly checked with `git ls-files`.
- Whether the analysis in `results/*` (tables, figures, batch_summary.txt) is internally consistent with the current code — explicitly out of scope per instructions.
