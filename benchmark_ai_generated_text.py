import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from ai_detectors.Binoculars.binoculars.binoculars_tinylama_4bit import BinocularsTinyLlama
import os

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
INPUT_CSV = "datasets/AI_generated/ai_generated_texts_indexed.csv"
OUTPUT_CSV = "datasets/AI_generated/ai_generated_texts_with_detector_scores.csv"
OUTPUT_DETECTOR_ONLY = "datasets/AI_generated/ai_detector_outputs.csv"
OUTPUT_PLOT = "results/AI_generated/detector_distribution.png"
OUTPUT_REPORT = "results/AI_generated/evaluation_report.csv"

BATCH_SIZE = 32

os.makedirs("results/AI_generated", exist_ok=True)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
df = pd.read_csv(INPUT_CSV)
texts = df["generated_text"].tolist()

print(f"Loaded {len(texts)} texts for benchmarking.")

# ----------------------------------------------------
# INITIALIZE DETECTOR
# ----------------------------------------------------
bino = BinocularsTinyLlama(vocab_chunk_size=32000)

# ----------------------------------------------------
# RUN DETECTOR
# ----------------------------------------------------
all_preds = []

print("Running AI detector...")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i + BATCH_SIZE]
    preds = bino.predict(batch)       # binary predictions: 1 = AI, 0 = human
    all_preds.extend(preds)

# Save predictions inside main dataframe
df["detector_prediction"] = all_preds
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions → {OUTPUT_CSV}")
print(df["detector_prediction"].value_counts())

# ----------------------------------------------------
# SAVE DETECTOR OUTPUT AS SEPARATE CSV (with IDs)
# ----------------------------------------------------
detector_df = df[["text_id", "prompt_id", "detector_prediction"]]
detector_df.to_csv(OUTPUT_DETECTOR_ONLY, index=False)

print(f"Saved detector-only outputs → {OUTPUT_DETECTOR_ONLY}")

# ----------------------------------------------------
# DISTRIBUTION PLOT
# ----------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(df["detector_prediction"], bins=[-0.5, 0.5, 1.5], edgecolor="black")
plt.xticks([0, 1], labels=["Not AI", "AI"])
plt.title("Detector Prediction Distribution")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.savefig(OUTPUT_PLOT)
plt.close()

print(f"Saved distribution plot → {OUTPUT_PLOT}")

# ----------------------------------------------------
# GROUP-LEVEL SCORES PER PROMPT ID
# ----------------------------------------------------
group_stats = df.groupby("prompt_id")["detector_prediction"].agg(
    mean_detection_rate="mean",
    num_samples="count",
    num_ai="sum"
)

group_stats["num_not_ai"] = group_stats["num_samples"] - group_stats["num_ai"]

print("\nGroup-level scores per prompt_id:")
print(group_stats)

# ----------------------------------------------------
# FULL EVALUATION REPORT
# ----------------------------------------------------
overall_ai_rate = df["detector_prediction"].mean()

report_df = group_stats.copy()
report_df["overall_ai_rate"] = overall_ai_rate

report_df.to_csv(OUTPUT_REPORT)
print(f"\nSaved full evaluation report → {OUTPUT_REPORT}")

print("\n=== OVERALL DETECTION RATE ===")
print(f"{overall_ai_rate:.4f}")

print("\n=== PER-PROMPT DETECTION RATES ===")
print(group_stats["mean_detection_rate"])
