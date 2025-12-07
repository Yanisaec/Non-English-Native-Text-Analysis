import pandas as pd
import glob, os, re
from sklearn.model_selection import train_test_split
import pickle

# ---------- LOAD CELL ----------
def load_cell(cell_root: str):
    cell_data = []

    # loop over top-level essay folders
    for folder in os.listdir(cell_root):
        folder_path = os.path.join(cell_root, folder)
        if not os.path.isdir(folder_path):
            continue

        # find the subfolder that is NOT "_tag"
        for sub in os.listdir(folder_path):
            if "_tag" in sub:
                continue
            sub_path = os.path.join(folder_path, sub)
            if not os.path.isdir(sub_path):
                continue

            # collect all txt files inside this subfolder
            for txt_file in glob.glob(os.path.join(sub_path, "*.txt")):
                with open(txt_file, encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                # Extract Intro, Body, Conclusion
                intro_match = re.search(r"<Intro>(.*?)</Intro>", text, re.DOTALL | re.IGNORECASE)
                body_match = re.search(r"<Body>(.*?)</Body>", text, re.DOTALL | re.IGNORECASE)
                concl_match = re.search(r"<Conclusion>(.*?)</Conclusion>", text, re.DOTALL | re.IGNORECASE)

                parts = []
                if intro_match: parts.append(intro_match.group(1).strip())
                if body_match: parts.append(body_match.group(1).strip())
                if concl_match: parts.append(concl_match.group(1).strip())

                essay_text = " ".join(parts)

                if essay_text.strip():  # only keep non-empty
                    cell_data.append({
                        "text": essay_text,
                        "label": 0  # non-native
                    })

    return cell_data

# ---------- LOAD BAWE ----------
def clean_bawe_text(text):
    # Remove LaTeX-like parentheses
    text = re.sub(r"-lrb-|-rrb-|-LSB-|-LRB-|-RRB-|-RSB-", "", text)

    # Remove backticks around phrases
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Remove inline citations (e.g., Valenzuela 1992:73, Moss 1987)
    text = re.sub(r"\b[A-Z][a-z]+(?: et al)? \d{4}(:\d+)?\b", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def fix_spacing(text):
    # Remove space before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # Ensure space after punctuation if missing
    text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)
    return text

def load_bawe(bawe_root: str):
    bawe_data = []
    
    # Iterate over all CSV files in subfolders
    for csv_file in glob.glob(os.path.join(bawe_root, "*", "*", "*.csv")):
        df = pd.read_csv(csv_file)
        essay_text = " ".join(df["word"].astype(str).tolist())
        
        # Apply cleaning
        essay_text = clean_bawe_text(essay_text)

        essay_text = fix_spacing(essay_text)
        
        if essay_text.strip():  # keep only non-empty
            bawe_data.append({
                "text": essay_text,
                "label": 1  # native
            })
    
    return bawe_data

# ---------- SPLITS ----------
def split_cell(data, test_size=0.2, seed=42):
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    return train, test

def split_bawe(data, test_size=0.2, seed=42):
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    return train, test

# ---------- MERGE ----------
def merge_splits(cell_train, cell_test, bawe_train, bawe_test):
    train = cell_train + bawe_train
    test  = cell_test  + bawe_test
    return train, test

# ---------- SAVE ----------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# ---------- MAIN ----------
cell = load_cell("datasets/Non-Native/CELL/All")
save_pickle(cell, "datasets/cell.pkl")
# bawe = load_bawe("datasets/Native/BAWE")

# cell_train, cell_test = split_cell(cell, test_size=0.2)
# bawe_train, bawe_test = split_bawe(bawe, test_size=0.2)

# train, test = merge_splits(cell_train, cell_test, bawe_train, bawe_test)

# save_pickle(train, "datasets/cell_bawe_train.pkl")
# save_pickle(test, "datasets/cell_bawe_test.pkl")

# print(f"Loaded CELL: {len(cell)} essays, BAWE: {len(bawe)} essays")
# print(f"Train: {len(train)} examples, Test: {len(test)} examples")
