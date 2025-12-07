import re
import numpy as np
import nltk
import spacy
from collections import Counter
from wordfreq import zipf_frequency
import language_tool_python

# Make sure these resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# Grammar checker
tool = language_tool_python.LanguageTool('en-US')


# ============================================================
# ============  FEATURE GROUP A: LEXICAL RICHNESS  ============
# ============================================================

def compute_TTR(tokens):
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)

def compute_MSTTR(tokens, window=50):
    """Mean segmental type-token ratio."""
    if len(tokens) < window:
        return compute_TTR(tokens)
    ttrs = []
    for i in range(0, len(tokens) - window + 1, window):
        segment = tokens[i:i+window]
        ttrs.append(compute_TTR(segment))
    return np.mean(ttrs)

def compute_LexicalDensity(tokens, pos_tags):
    """Content words / total tokens."""
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    if len(tokens) == 0:
        return 0
    content_count = sum(1 for pos in pos_tags if pos in content_pos)
    return content_count / len(tokens)

def compute_AvgLogFreq(tokens):
    """Average log frequency using wordfreq (higher = more common)."""
    if len(tokens) == 0:
        return 0
    freqs = [zipf_frequency(t.lower(), 'en') for t in tokens if t.isalpha()]
    return np.mean(freqs) if freqs else 0


# ============================================================
# ===  FEATURE GROUP B: FUNCTION WORDS & CLOSED CLASSES   ====
# ============================================================

def compute_function_word_ratios(tokens):
    func_words = {
        'articles': {"a", "an", "the"},
        'prepositions': {"in", "on", "at", "of", "to", "for", "from", "by", "with"},
        'pronouns': {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"},
        'auxiliaries': {"am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did"}
    }

    token_count = len(tokens) or 1
    ratios = {}
    for k, words in func_words.items():
        ratios[f"{k}_rate"] = sum(t.lower() in words for t in tokens) / token_count
    return ratios


# ============================================================
# =======  FEATURE GROUP C: SYNTACTIC COMPLEXITY  ============
# ============================================================

def compute_syntactic_features(doc):
    """Compute mean sentence length, subordination index, dependency depth."""
    sentences = list(doc.sents)
    if not sentences:
        return {"mean_sent_len": 0, "subordination_index": 0, "avg_dep_depth": 0}

    sent_lengths = [len([t for t in s if not t.is_punct]) for s in sentences]
    mean_sent_len = np.mean(sent_lengths)

    sub_conj = sum(1 for t in doc if t.dep_ == "mark")
    subordination_index = sub_conj / len(sentences)

    avg_dep_depth = np.mean([len(list(t.ancestors)) for t in doc])
    return {
        "mean_sent_len": mean_sent_len,
        "subordination_index": subordination_index,
        "avg_dep_depth": avg_dep_depth,
    }


# ============================================================
# =======  FEATURE GROUP D: ERROR-BASED FEATURES  ============
# ============================================================

def compute_error_features(text):
    """Grammar + spelling errors per 1000 tokens."""
    matches = tool.check(text)
    error_count = len(matches)
    token_count = len(nltk.word_tokenize(text)) or 1
    return {"error_rate_per_1k": error_count / token_count * 1000}


# ============================================================
# =======  FEATURE GROUP E: SURFACE / ORTHOGRAPHIC  ==========
# ============================================================

def compute_surface_features(text):
    """Count punctuation and capitalization irregularities."""
    num_chars = len(text) or 1
    commas = text.count(',')
    semicolons = text.count(';')
    uppercase_ratio = sum(1 for c in text if c.isupper()) / num_chars
    return {
        "comma_rate": commas / num_chars,
        "semicolon_rate": semicolons / num_chars,
        "uppercase_ratio": uppercase_ratio,
    }


# ============================================================
# =======  FEATURE GROUP F: DISCOURSE & COHESION  ============
# ============================================================

def compute_discourse_features(sentences):
    """Simple connective and cohesion metrics."""
    connectives = {"however", "therefore", "moreover", "because", "although", "thus"}
    if not sentences:
        return {"connective_rate": 0, "lexical_overlap": 0}

    tokens = [nltk.word_tokenize(s.lower()) for s in sentences]
    connective_count = sum(sum(w in connectives for w in t) for t in tokens)
    connective_rate = connective_count / len(sentences)

    # lexical overlap between adjacent sentences
    overlaps = []
    for i in range(len(tokens) - 1):
        overlap = len(set(tokens[i]) & set(tokens[i + 1]))
        overlaps.append(overlap)
    lexical_overlap = np.mean(overlaps) if overlaps else 0
    return {"connective_rate": connective_rate, "lexical_overlap": lexical_overlap}


# ============================================================
# ==================  MAIN FEATURE FUNCTION  =================
# ============================================================

def extract_features_A_to_F(text: str) -> np.ndarray:
    """
    Extracts interpretable linguistic features (A–F) from a text.
    Returns a flat numeric numpy array suitable for ML models.
    """

    if not text or not isinstance(text, str):
        return np.zeros(25)  # adjust if feature count changes

    # Tokenize & POS
    tokens = [t.text for t in nlp(text) if not t.is_space]
    pos_tags = [t.pos_ for t in nlp(text)]
    doc = nlp(text)
    sentences = [s.text for s in doc.sents]

    # A: Lexical richness
    A = {
        "TTR": compute_TTR(tokens),
        "MSTTR": compute_MSTTR(tokens),
        "LexicalDensity": compute_LexicalDensity(tokens, pos_tags),
        "AvgLogFreq": compute_AvgLogFreq(tokens),
    }

    # B: Function words
    B = compute_function_word_ratios(tokens)

    # C: Syntax
    C = compute_syntactic_features(doc)

    # D: Errors
    D = compute_error_features(text)


    # E: Surface
    E = compute_surface_features(text)

    # F: Discourse
    F = compute_discourse_features(sentences)

    # Merge all
    all_features = {**A, **B, **C, **D, **E, **F}

    # Return vector
    feature_vec = np.nan_to_num(np.array(list(all_features.values()), dtype=float))

    return feature_vec


def generate_feature_names():
    # A: Lexical richness keys
    A_keys = ["TTR", "MSTTR", "MTLD", "LexicalDensity", "AvgLogFreq"]

    # B-F: Keys from other groups
    other_groups_keys = list({
        **compute_function_word_ratios(["a"]),
        **compute_syntactic_features(nlp("Example sentence.")),
        **compute_error_features("Example."),
        **compute_surface_features("Example."),
        **compute_discourse_features(["Example.", "Another example."]),
    }.keys())

    # Combine them in the order they appear in extract_features_A_to_G
    return A_keys + other_groups_keys

FEATURE_NAMES = generate_feature_names()

FEATURE_GROUPS = {}
for feat in FEATURE_NAMES:
    name = feat.lower()
    if any(x in name for x in ["ttr", "mtld", "msttr", "lexicaldensity", "wordfreq", "avglogfreq"]):
        FEATURE_GROUPS[feat] = "A"  # Lexical richness & vocabulary
    elif any(x in name for x in ["articles", "prepositions", "pronouns", "determiners", "auxiliaries", "functionword"]):
        FEATURE_GROUPS[feat] = "B"  # Function-word & closed-class profiles
    elif any(x in name for x in ["pos_", "subordination_index", "avg_dep_depth", "mean_sent_len", "dependency"]):
        FEATURE_GROUPS[feat] = "C"  # Morpho-syntactic & syntactic complexity
    elif any(x in name for x in ["error", "grammar", "spelling", "nonstandard", "collocation", "idiomaticity"]):
        FEATURE_GROUPS[feat] = "D"  # Error-based features
    elif any(x in name for x in ["char_ngram", "punctuation", "capitalization", "uppercase", "comma_rate", "semicolon_rate"]):
        FEATURE_GROUPS[feat] = "E"  # Surface / orthographic
    elif any(x in name for x in ["connective", "lexical_overlap", "cohesion"]):
        FEATURE_GROUPS[feat] = "F"  # Discourse & cohesion
    else:
        FEATURE_GROUPS[feat] = "Other"