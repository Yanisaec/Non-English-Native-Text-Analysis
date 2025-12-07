import os
import pickle
import numpy as np

def analyze_dataset(path):
    # Load dataset
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if not data or not isinstance(data, list):
        print(f"{path}: invalid format")
        return

    # Extract texts
    texts = [d["text"] for d in data if "text" in d]
    lengths = [len(t.split()) for t in texts]
    char_counts = [len(t) for t in texts]

    # Compute statistics
    n_texts = len(texts)
    avg_len = np.mean(lengths)
    med_len = np.median(lengths)
    avg_char = np.mean(char_counts)
    med_char = np.median(char_counts)
    total_words = np.sum(lengths)

    # Prompts (optional)
    prompts = [d.get("prompt") for d in data if "prompt" in d]
    n_prompts = len(set(prompts)) if prompts else 0

    # Print results
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"Number of texts            : {n_texts}")
    print(f"Average length (words)     : {avg_len:.2f}")
    print(f"Median length (words)      : {med_len:.2f}")
    print(f"Average length (chars)     : {avg_char:.2f}")
    print(f"Median length (chars)      : {med_char:.2f}")
    print(f"Total number of words      : {total_words:,}")
    print(f"Number of different prompts: {n_prompts}")


dataset_paths = [
    "./datasets/asap2.pkl",
    "./datasets/persuade.pkl",
    "./datasets/toefl11.pkl",
    "./datasets/cell.pkl",
]

for path in dataset_paths:
    analyze_dataset(path)
