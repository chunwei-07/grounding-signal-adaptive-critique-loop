import json
from collections import Counter

def iteration_distribution():
    data = json.load(open("results/scored_results.json"))
    records = data["results"]["GSAL_theta0.8"]

    dist = Counter(r["iterations_performed"] for r in records)
    print("Iteration distribution for GSAL θ=0.8:")
    for k in sorted(dist):
        pct = 100 * dist[k] / len(records)
        bar = "█" * int(pct)
        print(f"  iter={k}: {dist[k]:3d} questions ({pct:.1f}%) {bar}")

if __name__ == "__main__":
    iteration_distribution()

