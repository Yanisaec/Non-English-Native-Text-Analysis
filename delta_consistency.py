import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import Counter

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSVS = [
    "results/AI_generated/typographical/transformed_texts_detector_scores_typographical.csv",
    "results/AI_generated/rewrite_benchmark/transformed_texts_detector_scores_rewrite_benchmark.csv",
    "results/AI_generated/stylistic/transformed_texts_with_detector_scores_full.csv",
    "results/AI_generated/genre_based_perturbations/transformed_texts_detector_scores_genre_based_perturbations.csv",
    "results/AI_generated/syntax/transformed_texts_syntax_with_detector_scores.csv",
]

OUT_DIR = "results/AI_generated/delta_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# LOAD AND MERGE
# -----------------------------
dfs = []
for path in INPUT_CSVS:
    df = pd.read_csv(path)
    df["source_file"] = os.path.basename(path)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("Loaded rows:", len(df))
print("Unique text_ids:", df.text_id.nunique())
print("Unique transformations:", df.transformation_name.nunique())

# -----------------------------
# METRIC 1: Delta Consistency Per text_id
# -----------------------------
def analyze_text_id(text_id_df):
    deltas = text_id_df.delta.values

    # Consistency: all deltas identical?
    is_constant = np.all(deltas == deltas[0])

    # Variance
    variance = np.var(deltas)

    # Entropy
    delta_dist = Counter(deltas)
    probs = np.array(list(delta_dist.values())) / len(deltas)
    ent = entropy(probs)

    return is_constant, variance, ent

records = []
for tid, group in df.groupby("text_id"):
    const, var, ent = analyze_text_id(group)

    records.append({
        "text_id": tid,
        "constant_delta": const,
        "delta_variance": var,
        "delta_entropy": ent,
        "num_transformations": group.transformation_name.nunique()
    })

per_text = pd.DataFrame(records)
per_text.to_csv(os.path.join(OUT_DIR, "delta_statistics_per_text.csv"), index=False)

# -----------------------------
# GLOBAL TEXT METRICS
# -----------------------------
constant_ratio = (per_text.constant_delta).mean()
mean_entropy = per_text.delta_entropy.mean()
mean_variance = per_text.delta_variance.mean()

# -----------------------------
# METRIC 2: Transformation sensitivity
# -----------------------------
transform_stats = []
for tname, g in df.groupby("transformation_name"):
    transform_stats.append({
        "transformation": tname,
        "mean_delta": g.delta.mean(),
        "delta_std": g.delta.std(),
        "count": len(g)
    })

df_transform = pd.DataFrame(transform_stats)
df_transform.to_csv(os.path.join(OUT_DIR, "delta_statistics_per_transformation.csv"), index=False)

# -----------------------------
# VISUALIZATIONS
# -----------------------------

# 1. Entropy distribution
plt.figure(figsize=(8,5))
plt.hist(per_text.delta_entropy, bins=40)
plt.title("Entropy of Delta per text_id")
plt.xlabel("Entropy (low = deterministic)")
plt.ylabel("Count")
plt.savefig(os.path.join(OUT_DIR, "entropy_distribution.png"))
plt.close()

# 2. Delta variance
plt.figure(figsize=(8,5))
plt.hist(per_text.delta_variance, bins=40)
plt.title("Delta variance per text_id")
plt.xlabel("Variance")
plt.ylabel("Count")
plt.savefig(os.path.join(OUT_DIR, "variance_distribution.png"))
plt.close()

# 3. Mean delta per transformation
plt.figure(figsize=(10,5))
df_transform.set_index("transformation")["mean_delta"].sort_values().plot(kind="barh")
plt.title("Mean delta per transformation")
plt.xlabel("Mean delta")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mean_delta_per_transformation.png"))
plt.close()

# 4. Variance by transformation
plt.figure(figsize=(10,5))
df_transform.set_index("transformation")["delta_std"].sort_values().plot(kind="barh")
plt.title("Delta volatility per transformation")
plt.xlabel("Std of delta")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "std_delta_per_transformation.png"))
plt.close()


# -----------------------------
# GLOBAL DECISION METRICS
# -----------------------------
total_texts = len(per_text)
nonconstant = (~per_text.constant_delta).sum()
sensitive_ratio = nonconstant / total_texts

# -----------------------------
# REPORT
# -----------------------------
report = f"""
DELTA CONSISTENCY ANALYSIS REPORT
================================

Total CSV files: {len(INPUT_CSVS)}
Total rows: {len(df)}
Unique text_ids: {df.text_id.nunique()}
Unique transformations: {df.transformation_name.nunique()}

GLOBAL BEHAVIOR
---------------
Ratio of texts with constant delta: {constant_ratio:.3f}
Ratio of texts where transformation changes delta: {sensitive_ratio:.3f}

Mean entropy per text: {mean_entropy:.4f}
Mean variance per text: {mean_variance:.4f}

INTERPRETATION
--------------
If constant_ratio is near 1.0:
- delta is a function of text only.

If constant_ratio is near 0:
- transformations significantly affect delta.

High entropy / variance:
- transformation-dependent behavior.

Low entropy / variance:
- text-dominant behavior.
"""

with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(report)

print(report)
print("Analysis saved in:", OUT_DIR)
