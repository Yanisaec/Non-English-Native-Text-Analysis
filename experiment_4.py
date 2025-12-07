import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt

# from ai_detectors.Binoculars.binoculars.test import Binoculars
# from ai_detectors.Binoculars.binoculars.optimized_binoculars import Binoculars
from ai_detectors.Binoculars.binoculars.binoculars_tinylama_4bit import BinocularsTinyLlama

BATCH_SIZE = 32
# BINO = Binoculars()
BINO = BinocularsTinyLlama(vocab_chunk_size=32000)
TOKENIZER = BINO.tokenizer
MINIMUM_TOKENS = 8
NB_TXT_PER_DATASET = 100

def count_tokens(text):
    return len(TOKENIZER(text).input_ids)

def run_detector(str_list):
    return BINO.predict(str_list)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

DATASETS = [
    ("asap2", "datasets/asap2.pkl", 1),      # native
    ("persuade", "datasets/persuade.pkl", 1),# native
    ("toefl11", "datasets/toefl11.pkl", 0),  # non-native
    ("cell", "datasets/cell.pkl", 0)         # non-native, no prompts
]

OUTPUT_DIR = "results/experiment_4_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Counters
ai_counts = {"native": 0, "non_native": 0}

for dataset_name, path, is_native in DATASETS:
    print(f"\nProcessing {dataset_name}...")

    data = load_pickle(path)

    # Extract texts with their original index
    indexed_texts = [(i, d["text"]) for i, d in enumerate(data)]

    # Filter by minimum token length
    filtered = [(i, txt) for i, txt in indexed_texts if count_tokens(txt) >= MINIMUM_TOKENS]

    if len(filtered) < NB_TXT_PER_DATASET:
        raise ValueError(f"Dataset {dataset_name} has fewer than {NB_TXT_PER_DATASET} valid texts (after token filtering).")

    # Sample NB_TXT_PER_DATASET valid texts
    sampled = np.random.choice(len(filtered), NB_TXT_PER_DATASET, replace=False)
    sampled_entries = [filtered[i] for i in sampled]

    all_indices = [idx for idx, _ in sampled_entries]
    all_texts = [txt for _, txt in sampled_entries]

    # Run detector in batches
    predictions = []
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE)):
        batch = all_texts[i:i+BATCH_SIZE]
        preds = run_detector(batch)
        predictions.extend(preds)

    # Count AI predictions
    num_ai = sum(predictions)
    if is_native:
        ai_counts["native"] += num_ai
    else:
        ai_counts["non_native"] += num_ai

    # Build dataframe
    df = pd.DataFrame({
        "index": all_indices,
        "prediction": predictions,
        "text": all_texts,
    })

    out_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_binoculars.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved → {out_path}")
    print(f"AI-flagged in {dataset_name}: {num_ai}/{NB_TXT_PER_DATASET}")

# ----------------------------
# Plot AI counts
# ----------------------------
categories = list(ai_counts.keys())
counts = [ai_counts[c] for c in categories]

plt.figure(figsize=(6,4))
plt.bar(categories, counts, color=["skyblue", "salmon"])
plt.ylabel("Number of AI-flagged texts")
plt.title("AI-flagged counts by dataset type")
plt.savefig(os.path.join(OUTPUT_DIR, "ai_counts_plot.png"))
plt.show()

# ----------------------------
# Save counts to TXT
# ----------------------------
txt_path = os.path.join(OUTPUT_DIR, "ai_counts.txt")
with open(txt_path, "w") as f:
    for cat, count in ai_counts.items():
        f.write(f"{cat}: {count} AI-flagged texts out of {NB_TXT_PER_DATASET*2} sampled texts\n")
    f.write(f"Total AI-flagged: {sum(ai_counts.values())}\n")

print(f"Saved counts → {txt_path}")
