import pickle
import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag
from wordfreq import word_frequency
from math import log
from tqdm import tqdm

# Ensure NLTK components are downloaded
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")
nltk.download("stopwords")

# ---------- UTILS ----------
def load_pickle_dataset(path):
    """Load a dataset (list of dicts) from pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def tokenize(text):
    """Lowercase + tokenize text into alphabetic tokens only."""
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    return tokens


# ---------- METRICS ----------

def ttr(tokens):
    """Type–Token Ratio."""
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)


def msttr(tokens, window_size=50):
    """Mean Segmental Type–Token Ratio (MSTTR)."""
    if len(tokens) < window_size:
        return ttr(tokens)
    ttrs = []
    for i in range(0, len(tokens) - window_size + 1, window_size):
        segment = tokens[i:i+window_size]
        ttrs.append(ttr(segment))
    return np.mean(ttrs)


def mtld(tokens, ttr_threshold=0.72):
    """
    Measure of Textual Lexical Diversity (MTLD)
    Based on McCarthy & Jarvis (2010)
    """
    if len(tokens) == 0:
        return 0

    def calc_mtld_direction(tokens):
        factors = 0
        token_count = 0
        types = set()
        ttr_val = 1.0

        for word in tokens:
            token_count += 1
            types.add(word)
            ttr_val = len(types) / token_count
            if ttr_val < ttr_threshold:
                factors += 1
                token_count = 0
                types = set()
        excess = 1 - ((ttr_threshold - ttr_val) / ttr_threshold)
        return (factors + excess) if token_count > 0 else factors

    forward = calc_mtld_direction(tokens)
    backward = calc_mtld_direction(list(reversed(tokens)))
    return (len(tokens) / ((forward + backward) / 2)) if (forward + backward) > 0 else 0


def lexical_density(tokens):
    """Proportion of content words (N, V, Adj, Adv)."""
    if len(tokens) == 0:
        return 0
    pos_tags = pos_tag(tokens)
    content_tags = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"}
    content_words = [w for w, tag in pos_tags if tag in content_tags]
    return len(content_words) / len(tokens)


def average_log_freq(tokens):
    """Mean log word frequency using SUBTLEX-based `wordfreq`."""
    if len(tokens) == 0:
        return 0
    freqs = [word_frequency(w, "en") for w in tokens]
    freqs = [max(f, 1e-9) for f in freqs]  # avoid log(0)
    return np.mean([log(f) for f in freqs])


# ---------- ANALYSIS ----------
def analyze_dataset(pickle_path, sample_size=None):
    """
    Load dataset and compute lexical metrics (averaged across texts).

    Args:
        pickle_path (str): path to .pkl dataset
        sample_size (int): optionally limit number of essays for speed
    """
    data = load_pickle_dataset(pickle_path)
    if sample_size:
        data = data[:sample_size]

    results = {
        "TTR": [],
        "MSTTR": [],
        "MTLD": [],
        "LexicalDensity": [],
        "AvgLogFreq": []
    }

    print(f"🔍 Analyzing {len(data)} essays...")

    for d in tqdm(data):
        try:
            tokens = tokenize(d["text"])
            results["TTR"].append(ttr(tokens))
            results["MSTTR"].append(msttr(tokens))
            results["MTLD"].append(mtld(tokens))
            results["LexicalDensity"].append(lexical_density(tokens))
            results["AvgLogFreq"].append(average_log_freq(tokens))
        except:
            continue
    summary = {metric: np.mean(vals) for metric, vals in results.items()}
    print("\n📊 Dataset lexical statistics:")
    for k, v in summary.items():
        print(f"{k:20s}: {v:.4f}")

    return summary, results


# ---------- EXAMPLE ----------
if __name__ == "__main__":
    # Example usage — adjust to your actual file paths:
    DATA_PATH = "datasets/pelic.pkl"

    summary, _ = analyze_dataset(DATA_PATH, sample_size=None)  # you can remove sample_size later
