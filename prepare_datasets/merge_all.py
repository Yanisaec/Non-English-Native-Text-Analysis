import pickle
import random
from sklearn.model_selection import train_test_split

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def split_by_prompt(dataset, test_size=0.25, seed=42):
    """Split dataset by unique 'prompt' values to avoid leakage."""
    # Group samples by prompt
    prompt_to_samples = {}
    for sample in dataset:
        prompt = sample.get("prompt", None)
        if prompt not in prompt_to_samples:
            prompt_to_samples[prompt] = []
        prompt_to_samples[prompt].append(sample)

    # Split prompts
    prompts = list(prompt_to_samples.keys())
    random.seed(seed)
    random.shuffle(prompts)
    split_idx = int(len(prompts) * (1 - test_size))
    train_prompts, test_prompts = prompts[:split_idx], prompts[split_idx:]

    # Build datasets
    train_data = [s for p in train_prompts for s in prompt_to_samples[p]]
    test_data = [s for p in test_prompts for s in prompt_to_samples[p]]
    return train_data, test_data

def split_random(dataset, test_size=0.25, seed=42):
    """Standard random split for datasets without prompt key."""
    train_data, test_data = train_test_split(
        dataset, test_size=test_size, random_state=seed, shuffle=True
    )
    return train_data, test_data

def merge_and_split_datasets(dataset_paths, output_train="datasets/merged_train.pkl", output_test="datasets/merged_test.pkl"):
    """
    Load multiple datasets, split by prompt when applicable, and merge into train/test pickles.
    """
    train_all, test_all = [], []

    for path in dataset_paths:
        print(f"Processing {path} ...")
        data = load_pickle(path)
        if len(data) == 0:
            print(f"⚠️ Skipping empty dataset: {path}")
            continue

        # Check if the dataset has prompts
        has_prompt = any("prompt" in d for d in data)
        if has_prompt:
            train_data, test_data = split_by_prompt(data)
            print(f" → Split by prompt ({len(train_data)} train / {len(test_data)} test)")
        else:
            train_data, test_data = split_random(data)
            print(f" → Random split ({len(train_data)} train / {len(test_data)} test)")

        train_all.extend(train_data)
        test_all.extend(test_data)

    print(f"\n✅ Total merged: {len(train_all)} train / {len(test_all)} test")
    save_pickle(train_all, output_train)
    save_pickle(test_all, output_test)
    print(f"💾 Saved as {output_train} and {output_test}")

    return train_all, test_all

dataset_paths = [
    "datasets/prepare_datasets/asap2.pkl",
    "datasets/prepare_datasets/persuade.pkl",
    "datasets/prepare_datasets/toefl11.pkl",
    "datasets/prepare_datasets/cell.pkl"
]

train_set, test_set = merge_and_split_datasets(dataset_paths)