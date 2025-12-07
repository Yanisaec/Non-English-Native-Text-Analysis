import pickle
import numpy as np

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def average_text_length(dataset, by="words"):
    """
    Compute average text length for a dataset.
    
    Args:
        dataset (list[dict]): List of samples with key 'text'.
        by (str): 'words' or 'chars' to compute length by tokens or characters.
    
    Returns:
        float: Average text length.
    """
    if by == "words":
        lengths = [len(sample["text"].split()) for sample in dataset if sample.get("text")]
    elif by == "chars":
        lengths = [len(sample["text"]) for sample in dataset if sample.get("text")]
    else:
        raise ValueError("by must be either 'words' or 'chars'")
    
    return np.mean(lengths), np.std(lengths)

def print_average_lengths(dataset_paths, by="words"):
    """
    Print average text lengths for multiple datasets.
    
    Args:
        dataset_paths (list[str]): Paths to pickle files.
        by (str): 'words' or 'chars'.
    """
    print(f"\nAverage text length per dataset ({by}):\n")
    for path in dataset_paths:
        data = load_pickle(path)
        avg_len, std_len = average_text_length(data, by=by)
        print(f"{path:<15}  mean = {avg_len:.2f},  std = {std_len:.2f},  n = {len(data)}")

dataset_paths = [
    "datasets/asap2.pkl",
    "datasets/persuade.pkl",
    "datasets/bawe.pkl",
    "datasets/toefl11.pkl",
    "datasets/cell.pkl",
    "datasets/pelic.pkl"
]

print_average_lengths(dataset_paths)