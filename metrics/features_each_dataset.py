import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os

from all_features import FEATURE_NAMES

# === CONFIGURE FILE PATHS ===
ORIGINAL_DATASET_PATHS = [
    "datasets/cell.pkl",  # "prompt" absent
    "datasets/toefl11.pkl",  # "prompt" present
    "datasets/asap2.pkl",  # "prompt" present
    "datasets/persuade.pkl",  # "prompt" present
]
MERGED_TRAIN_PATH = "datasets/merged_train.pkl"
MERGED_TEST_PATH = "datasets/merged_test.pkl"
FEATURES_TRAIN_PATH = "datasets/precompute/features_train.pkl"
FEATURES_TEST_PATH = "datasets/precompute/features_test.pkl"
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === LOAD ORIGINAL DATASETS AND PROMPTS ===
unique_prompts_per_dataset = []

for path in ORIGINAL_DATASET_PATHS:
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    if any("prompt" in sample for sample in dataset):
        prompts = set(sample["prompt"] for sample in dataset if "prompt" in sample)
        unique_prompts_per_dataset.append(prompts)
    else:
        unique_prompts_per_dataset.append(None)

# === MAP FEATURE ARRAYS TO ORIGINAL DATASET ===

def load_list_from_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

merged_train = load_list_from_pkl(MERGED_TRAIN_PATH)
merged_test  = load_list_from_pkl(MERGED_TEST_PATH)
features_train = load_list_from_pkl(FEATURES_TRAIN_PATH)
features_test  = load_list_from_pkl(FEATURES_TEST_PATH)

assert len(merged_train) == len(features_train)
assert len(merged_test) == len(features_test)

# Features, by original dataset index
features_by_original = defaultdict(list)

def source_dataset_idx_for_sample(sample):
    # If no "prompt" key, belongs to original1
    if "prompt" not in sample:
        return 0
    prompt = sample["prompt"]
    # Find which of original datasets (1,2,3) has this prompt
    for idx in range(1, 4):
        if prompt in unique_prompts_per_dataset[idx]:
            return idx
    raise ValueError(f"Prompt not found: {prompt}")

# === Associate train and test features with originals ===
for sample, feature in tqdm(zip(merged_train, features_train), total=len(features_train), desc="Assigning train features"):
    idx = source_dataset_idx_for_sample(sample)
    features_by_original[idx].append(feature)
for sample, feature in tqdm(zip(merged_test, features_test), total=len(features_test), desc="Assigning test features"):
    idx = source_dataset_idx_for_sample(sample)
    features_by_original[idx].append(feature)

for idx in range(4):
    features_array = np.array(features_by_original[idx])
    avg_features = np.mean(features_array, axis=0)
    try:
        df = pd.DataFrame([avg_features], columns=FEATURE_NAMES)
    except:
        breakpoint()
    df.to_csv(os.path.join(OUTPUT_FOLDER, f"original_dataset{idx+1}_avg_features.csv"), index=False)

    print(f"Dataset {idx+1}: {len(features_array)} feature rows, saved results.")

