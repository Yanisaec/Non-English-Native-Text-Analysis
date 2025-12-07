import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ai_detectors.Binoculars.binoculars.binoculars_tinylama_4bit import BinocularsTinyLlama
import os

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
TRANSFOS_CLASS = 'genre_based_perturbations'
TRANSFORMED_CSV = f"datasets/AI_generated/transformed_texts_{TRANSFOS_CLASS}.csv"
BASELINE_CSV = "datasets/AI_generated/ai_detector_outputs.csv"
OUTPUT_CSV = f"results/AI_generated/{TRANSFOS_CLASS}/transformed_texts_detector_scores_{TRANSFOS_CLASS}.csv"
OUTPUT_REPORT = f"results/AI_generated/{TRANSFOS_CLASS}/transformation_impact_report_{TRANSFOS_CLASS}.csv"
SWITCH_COUNTS_CSV = f"results/AI_generated/{TRANSFOS_CLASS}/detection_switch_counts_{TRANSFOS_CLASS}.csv"
OUTPUT_PLOT = f"results/AI_generated/{TRANSFOS_CLASS}/transformation_effect_distribution_{TRANSFOS_CLASS}.png"
BATCH_SIZE = 32

os.makedirs(f"results/AI_generated/{TRANSFOS_CLASS}", exist_ok=True)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
trans_df = pd.read_csv(TRANSFORMED_CSV)
baseline_df = pd.read_csv(BASELINE_CSV)

print(f"Loaded {len(trans_df)} transformed texts.")

# Merge baseline detector scores
merged_df = trans_df.merge(baseline_df, on=["text_id", "prompt_id"], how="left")
merged_df = merged_df.rename(columns={"detector_prediction": "baseline_score"})

print("Merged with baseline detector scores.")

# ----------------------------------------------------
# INITIALIZE DETECTOR
# ----------------------------------------------------
bino = BinocularsTinyLlama(vocab_chunk_size=32000)

# ----------------------------------------------------
# RUN DETECTOR ON TRANSFORMED TEXTS
# ----------------------------------------------------
texts = merged_df["transformed_text"].tolist()
all_preds = []

print("Running AI detector on transformed texts...")

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i + BATCH_SIZE]
    preds = bino.predict(batch)
    all_preds.extend(preds)

merged_df["transformed_score"] = all_preds

# ----------------------------------------------------
# COMPUTE DETECTION CHANGE PER TEXT
# ----------------------------------------------------
merged_df["delta"] = merged_df["transformed_score"] - merged_df["baseline_score"]

# ----------------------------------------------------
# SAVE SCORE-ONLY CSV (NO TEXT)
# ----------------------------------------------------
score_df = merged_df[[
    "text_id",
    "prompt_id",
    "transformation_name",
    "baseline_score",
    "transformed_score",
    "delta"
]]

score_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved score-only dataset → {OUTPUT_CSV}")

# =====================================================================
# PRINT + SAVE SWITCH COUNTS: 1→0, 1→1, 0→0, 0→1
# =====================================================================

switch_counts = {
    "1_to_0": len(merged_df[(merged_df["baseline_score"] == 1) & (merged_df["transformed_score"] == 0)]),
    "1_to_1": len(merged_df[(merged_df["baseline_score"] == 1) & (merged_df["transformed_score"] == 1)]),
    "0_to_0": len(merged_df[(merged_df["baseline_score"] == 0) & (merged_df["transformed_score"] == 0)]),
    "0_to_1": len(merged_df[(merged_df["baseline_score"] == 0) & (merged_df["transformed_score"] == 1)])
}

# Print results
print("\n=== DETECTION SWITCH COUNTS ===")
for k, v in switch_counts.items():
    print(f"{k}: {v}")

# Save results
pd.DataFrame.from_dict(switch_counts, orient="index", columns=["count"]).to_csv(SWITCH_COUNTS_CSV)
print(f"\nSaved detection switch counts → {SWITCH_COUNTS_CSV}")

# =====================================================================
# SWITCH COUNTS PER TRANSFORMATION NAME
# =====================================================================

per_transformation_switch = []

for tname, group in merged_df.groupby("transformation_name"):

    counts = {
        "transformation_name": tname,
        "1_to_0": len(group[(group["baseline_score"] == 1) & (group["transformed_score"] == 0)]),
        "1_to_1": len(group[(group["baseline_score"] == 1) & (group["transformed_score"] == 1)]),
        "0_to_0": len(group[(group["baseline_score"] == 0) & (group["transformed_score"] == 0)]),
        "0_to_1": len(group[(group["baseline_score"] == 0) & (group["transformed_score"] == 1)]),
    }

    per_transformation_switch.append(counts)

# Convert to DataFrame
per_trans_df = pd.DataFrame(per_transformation_switch)

# Print
print("\n=== SWITCH COUNTS PER TRANSFORMATION NAME ===")
print(per_trans_df)

# Save to CSV
PER_TRANS_SWITCH_CSV = "results/AI_generated/detection_switch_counts_per_transformation.csv"
per_trans_df.to_csv(PER_TRANS_SWITCH_CSV, index=False)

print(f"Saved per-transformation switch counts → {PER_TRANS_SWITCH_CSV}")

# =====================================================================
# OVERALL RESULT PER TRANSFORMATION NAME
# =====================================================================
transformation_summary = merged_df.groupby("transformation_name").agg(
    total_texts=("text_id", "count"),
    total_detected=("transformed_score", "sum"),
)

transformation_summary["detection_rate"] = (
    transformation_summary["total_detected"] / transformation_summary["total_texts"]
)

delta_stats = merged_df.groupby("transformation_name")["delta"].value_counts().unstack(fill_value=0)
transformation_summary = transformation_summary.join(delta_stats, how="left")

print("\n=== OVERALL RESULT PER TRANSFORMATION ===")
print(transformation_summary)

# =====================================================================
# OVERALL RESULT PER ORIGINAL PROMPT
# =====================================================================
prompt_summary = merged_df.groupby("prompt_id").agg(
    total_transformed=("text_id", "count"),
    total_detected=("transformed_score", "sum")
)

prompt_summary["detection_rate"] = (
    prompt_summary["total_detected"] / prompt_summary["total_transformed"]
)

prompt_delta_stats = merged_df.groupby("prompt_id")["delta"].value_counts().unstack(fill_value=0)
prompt_summary = prompt_summary.join(prompt_delta_stats, how="left")

print("\n=== OVERALL RESULT PER ORIGINAL PROMPT ===")
print(prompt_summary)

# ----------------------------------------------------
# SAVE BOTH SUMMARIES INTO REPORT (as Excel)
# ----------------------------------------------------
with pd.ExcelWriter(OUTPUT_REPORT.replace(".csv", ".xlsx")) as writer:
    transformation_summary.to_excel(writer, sheet_name="By_Transformation")
    prompt_summary.to_excel(writer, sheet_name="By_Prompt")

print(f"\nSaved full evaluation report → {OUTPUT_REPORT.replace('.csv', '.xlsx')}")

# ----------------------------------------------------
# PLOT: EFFECT DISTRIBUTION
# ----------------------------------------------------
plt.figure(figsize=(8, 5))
merged_df["delta"].hist(bins=15, edgecolor="black")
plt.title("Change in Detector Score After Transformation (delta)")
plt.xlabel("delta (transformed_score - baseline_score)")
plt.ylabel("Number of texts")
plt.savefig(OUTPUT_PLOT)
plt.close()

print(f"Saved transformation-effect plot → {OUTPUT_PLOT}")
