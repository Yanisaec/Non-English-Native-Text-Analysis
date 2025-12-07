import pandas as pd
import glob, os
from sklearn.model_selection import train_test_split
import pickle

def load_pelic_with_prompts(answer_path: str):
    df = pd.read_csv(answer_path)
    data = []
    for _, row in df.iterrows():
        if isinstance(row['text'], str):
            data.append({
                "text": row["text"],
                "label": 0,  # non-native
                "prompt": row["question_id"],  # use question_id to group
                "anon_id": row["anon_id"]
            })
    return data

def load_bawe(bawe_root: str):
    bawe_data = []
    for csv_file in glob.glob(os.path.join(bawe_root, "*", "*", "*.csv")):
        df = pd.read_csv(csv_file)
        essay_text = " ".join(df["word"].astype(str).tolist())
        bawe_data.append({
            "text": essay_text,
            "label": 1  # native
        })
    return bawe_data

def split_pelic_prompt(data, test_size=0.2, seed=42):
    prompts = list(set(d["prompt"] for d in data))
    train_prompts, test_prompts = train_test_split(prompts, test_size=test_size, random_state=seed)
    train = [d for d in data if d["prompt"] in train_prompts]
    test = [d for d in data if d["prompt"] in test_prompts]
    return train, test

def split_bawe(data, test_size=0.2, seed=42):
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    return train, test

def merge_splits(pelic_train, pelic_test, bawe_train, bawe_test):
    train = pelic_train + bawe_train
    test  = pelic_test  + bawe_test
    return train, test

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

pelic = load_pelic_with_prompts("datasets/Non-Native/PELIC-dataset/corpus_files/answer.csv")
bawe  = load_bawe("datasets/Native/BAWE")

save_pickle(pelic, "datasets/pelic.pkl")
save_pickle(bawe, "datasets/bawe.pkl")

# pelic_train, pelic_test = split_pelic_prompt(pelic, test_size=0.2)
# bawe_train, bawe_test   = split_bawe(bawe, test_size=0.2)

# train, test = merge_splits(pelic_train, pelic_test, bawe_train, bawe_test)

# save_pickle(train, "datasets/pelic_bawe_train.pkl")
# save_pickle(test, "datasets/pelic_bawe_test.pkl")

# print(f"Train: {len(train)} examples, Test: {len(test)} examples")