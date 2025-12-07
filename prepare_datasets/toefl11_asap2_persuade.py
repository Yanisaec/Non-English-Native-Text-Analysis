import os
import glob
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import re


# ---------- UTILS ----------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def clean_text(text: str) -> str:
    """Remove number references, normalize spaces, and fix punctuation spacing."""
    text = re.sub(r"\(\d+\)", "", text)                   # remove (12), (8), etc.
    text = re.sub(r"\s+", " ", text)                      # normalize spaces
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)          # remove space before punctuation
    return text.strip()


# ---------- LOAD TOEFL11 (non-native) ----------
def load_toefl11(toefl_root: str):
    """
    Args:
        toefl_root: path to TOEFL11 folder containing 'index.csv' and subfolder with *.txt responses.
    Returns:
        List of dicts with 'text', 'label', and 'prompt'.
    """
    index_path = os.path.join(toefl_root, "index.csv")
    df = pd.read_csv(index_path)
    data = []

    for _, row in df.iterrows():
        filename = str(row["Filename"]).strip()
        prompt = str(row["Prompt"]).strip()
        txt_path = os.path.join(toefl_root, "responses/original", filename)
        if not os.path.exists(txt_path):
            txt_path = os.path.join(toefl_root, filename)
        if not os.path.exists(txt_path):
            continue

        try:
            with open(txt_path, encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception:
            continue

        if len(text) > 0:
            data.append({
                "text": text,
                "label": 0,  # non-native
                "prompt": prompt
            })

    return data


# ---------- LOAD ASAP2 (native) ----------
def load_asap2(asap_path: str):
    df = pd.read_csv(asap_path)
    asap_data = []

    for _, row in df.iterrows():
        text = clean_text(str(row["full_text"]))
        prompt = str(row["prompt_name"]).strip()
        if text:
            asap_data.append({
                "text": text,
                "label": 1,  # native
                "prompt": prompt
            })
    return asap_data


# ---------- LOAD PERSUADE (native) ----------
def load_persuade(persuade_path: str):
    df = pd.read_csv(persuade_path)
    persuade_data = []

    for _, row in df.iterrows():
        text = clean_text(str(row["full_text"]))
        prompt = str(row["prompt_name"]).strip()
        if text:
            persuade_data.append({
                "text": text,
                "label": 1,  # native
                "prompt": prompt
            })
    return persuade_data


# ---------- SPLIT FUNCTIONS ----------
def split_by_prompt(data, test_size=0.2, seed=42):
    prompts = sorted(list(set(d["prompt"] for d in data)))
    train_prompts, test_prompts = train_test_split(prompts, test_size=test_size, random_state=seed)
    train = [d for d in data if d["prompt"] in train_prompts]
    test = [d for d in data if d["prompt"] in test_prompts]
    return train, test


# ---------- MERGE ----------
def merge_splits(non_native_train, non_native_test, native_train, native_test):
    train = non_native_train + native_train
    test = non_native_test + native_test
    return train, test


# ---------- MAIN ----------
if __name__ == "__main__":
    # Adjust paths as needed
    TOEFL11_ROOT = "datasets/Non-Native/TOEFL11/data/text"
    ASAP2_CSV = "datasets/Native/ASAP2/ASAP2_train_sourcetexts.csv"
    PERSUADE_CSV = "datasets/Native/PERSUADE/persuade_2.0_human_scores_demo_id_github/persuade_2.0_human_scores_demo_id_github.csv"

    print("Loading TOEFL11 (non-native)...")
    toefl11 = load_toefl11(TOEFL11_ROOT)
    save_pickle(toefl11, "datasets/toefl11.pkl")
    print(f"Loaded {len(toefl11)} TOEFL11 essays.")

    print("Loading ASAP2 (native)...")
    asap2 = load_asap2(ASAP2_CSV)
    save_pickle(asap2, "datasets/asap2.pkl")
    print(f"Loaded {len(asap2)} ASAP2 essays.")

    print("Loading PERSUADE (native)...")
    persuade = load_persuade(PERSUADE_CSV)
    save_pickle(persuade, "datasets/persuade.pkl")
    print(f"Loaded {len(persuade)} PERSUADE essays.")

    # Combine native datasets
    native_data = asap2 + persuade
    print(f"Total native essays: {len(native_data)}")

    # Split train/test by prompt
    print("Splitting TOEFL11 by prompt...")
    toefl_train, toefl_test = split_by_prompt(toefl11, test_size=0.25)
    print(f"TOEFL11 -> Train: {len(toefl_train)}, Test: {len(toefl_test)}")

    print("Splitting native (ASAP2 + PERSUADE) by prompt...")
    native_train, native_test = split_by_prompt(native_data, test_size=0.2)
    print(f"Native -> Train: {len(native_train)}, Test: {len(native_test)}")

    # Merge and save
    train, test = merge_splits(toefl_train, toefl_test, native_train, native_test)
    save_pickle(train, "datasets/toefl_asap2_persuade_train.pkl")
    save_pickle(test, "datasets/toefl_asap2_persuade_test.pkl")

    print(f"\n✅ Saved processed data:")
    print(f"Train: {len(train)} examples")
    print(f"Test:  {len(test)} examples")