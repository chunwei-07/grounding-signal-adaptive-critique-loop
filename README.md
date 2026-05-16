# Install dependencies
pip install openai ragas numpy scipy pandas matplotlib

# Step 1 — download and prepare data (run once, ~563MB download)
python data/load_hotpotqa.py

# Step 2 — ALWAYS dry run first
python run_experiment.py --dry-run

# Step 3 — full experiment (run system by system)
python run_experiment.py --system fscl
python run_experiment.py --system gsal --theta 0.7
python run_experiment.py --system gsal --theta 0.6
python run_experiment.py --system gsal --theta 0.8

# Step 4 — RAGAS scoring
python evaluation/ragas_scorer.py \
    --input  results/raw_results.json \
    --output results/scored_results.json

# Step 5 — analysis and figures
python analyse_results.py \
    --input results/scored_results.json