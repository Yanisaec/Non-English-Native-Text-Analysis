import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from experiment_0 import load_pickle

DATASETS = [
    ("asap2", "datasets/asap2.pkl", 1),      # native
    ("persuade", "datasets/persuade.pkl", 1),# native
    ("toefl11", "datasets/toefl11.pkl", 0),  # non-native
    ("cell", "datasets/cell.pkl", 0)         # non-native, no prompts
]
N_TOP_CONTENT = 100

def prompt_split(data, frac=0.3):
    prompts = sorted(set(d['prompt'] for d in data))
    np.random.seed(42)
    np.random.shuffle(prompts)
    split = int((1-frac)*len(prompts))
    train_prompts = set(prompts[:split])
    test_prompts = set(prompts[split:])
    train = [d for d in data if d['prompt'] in train_prompts]
    test  = [d for d in data if d['prompt'] in test_prompts]
    return train, test

def random_split(data, frac=0.3):
    idx = np.arange(len(data))
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int((1-frac)*len(idx))
    train = [data[i] for i in idx[:split]]
    test  = [data[i] for i in idx[split:]]
    return train, test

def eval_model(Xtr, Xte, ytr, yte, clf):
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, y_pred, average="macro")
    return acc, prec, rec, f1, clf

def get_top_word_indices(X, vectorizer, n):
    sums = np.array(X.sum(axis=0)).flatten()
    top_idx = np.argsort(-sums)[:n]
    return top_idx, [vectorizer.get_feature_names_out()[i] for i in top_idx]

def remove_tfidf_words(X, top_idx):
    mask = np.ones(X.shape[1], dtype=bool)
    mask[top_idx] = False
    return X[:, mask], mask

def merge_splits(splits, label):
    merged_train = []
    merged_test  = []
    for train, test in splits:
        for entry in train:
            merged_train.append({"text": entry["text"], "label": label})
        for entry in test:
            merged_test.append({"text": entry["text"], "label": label})
    return merged_train, merged_test

def remove_words_from_texts(texts, words):
    # Remove each top word from texts via word-boundary regex
    pat = re.compile(r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b', flags=re.IGNORECASE)
    return [' '.join(pat.sub('', t).split()) for t in texts]

results_overall = []
features_overall = []

for split_kind in ["PromptHeldout", "RandomSplit"]:
    split_trains = []
    split_tests  = []
    # Process all datasets
    for name, path, label in DATASETS:
        data = load_pickle(path)
        if name == "cell":
            train, test = random_split(data)   # Only random split for cell
        else:
            if split_kind == "PromptHeldout":
                train, test = prompt_split(data)
            else:
                train, test = random_split(data)
        split_trains.append((train, label))
        split_tests.append((test, label))
    # Merge splits from all datasets
    train_merged = []
    test_merged  = []
    for train, label in split_trains:
        for entry in train:
            train_merged.append({"text": entry["text"], "label": label})
    for test, label in split_tests:
        for entry in test:
            test_merged.append({"text": entry["text"], "label": label})
    X_train = [d["text"] for d in train_merged]
    y_train = [d["label"] for d in train_merged]
    X_test  = [d["text"] for d in test_merged]
    y_test  = [d["label"] for d in test_merged]

    # Char n-grams
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=10000)
    Xtr_char = char_vec.fit_transform(X_train)
    Xte_char = char_vec.transform(X_test)

    # Word n-grams
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=5000, stop_words="english")
    Xtr_word = word_vec.fit_transform(X_train)
    Xte_word = word_vec.transform(X_test)

    # Train models
    for name, clf in [("LR_CharNgram", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
                      ("LR_WordNgram", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
                      ("SVM_WordNgram", LinearSVC(C=1.0, class_weight="balanced"))]:
        if "CharNgram" in name:
            Xtr, Xte = Xtr_char, Xte_char
        else:
            Xtr, Xte = Xtr_word, Xte_word
        acc, prec, rec, f1, clf_fitted = eval_model(Xtr, Xte, y_train, y_test, clf)
        coef = clf_fitted.coef_[0] if hasattr(clf_fitted, "coef_") else None
        vocab = char_vec.get_feature_names_out() if "CharNgram" in name else word_vec.get_feature_names_out()
        top_feats = None
        if coef is not None:
            idxs = np.argsort(np.abs(coef))[::-1][:20]
            top_feats = [(vocab[i], coef[i]) for i in idxs]
        results_overall.append(dict(split=split_kind, model=name,
                                    accuracy=acc, precision=prec, recall=rec, f1=f1,
                                    top_features=top_feats))
        features_overall.append(dict(split=split_kind, model=name, top_features=top_feats))

    # Topic word control: remove top-N content words FROM THE TEXTS
    top_idx, top_words = get_top_word_indices(Xtr_word, word_vec, N_TOP_CONTENT)
    # Remove top words from texts, then recompute char/word ngrams
    X_train_no_words = remove_words_from_texts(X_train, top_words)
    X_test_no_words  = remove_words_from_texts(X_test, top_words)
    # New features
    char_vec_no_words = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=10000)
    word_vec_no_words = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=5000, stop_words="english")
    Xtr_char_no_words = char_vec_no_words.fit_transform(X_train_no_words)
    Xte_char_no_words = char_vec_no_words.transform(X_test_no_words)
    Xtr_word_no_words = word_vec_no_words.fit_transform(X_train_no_words)
    Xte_word_no_words = word_vec_no_words.transform(X_test_no_words)

    # LR Char ngram (no top words)
    acc, prec, rec, f1, clf_fitted = eval_model(Xtr_char_no_words, Xte_char_no_words, y_train, y_test,
                            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    coef = clf_fitted.coef_[0] if hasattr(clf_fitted, "coef_") else None
    vocab = char_vec_no_words.get_feature_names_out()
    top_feats = None
    if coef is not None:
        idxs = np.argsort(np.abs(coef))[::-1][:20]
        top_feats = [(vocab[i], coef[i]) for i in idxs]
    results_overall.append(dict(split=split_kind, model="LR_CharNgram_noTopWords",
                                accuracy=acc, precision=prec, recall=rec, f1=f1,
                                top_features=top_feats, removed_words=top_words))
    features_overall.append(dict(split=split_kind, model="LR_CharNgram_noTopWords",
                                top_features=top_feats, removed_words=top_words))

    # LR/SVM Word ngram (no top words)
    for name, clf in [("LR_WordNgram_noTopWords", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
                      ("SVM_WordNgram_noTopWords", LinearSVC(C=1.0, class_weight="balanced"))]:
        acc, prec, rec, f1, clf_fitted = eval_model(Xtr_word_no_words, Xte_word_no_words, y_train, y_test, clf)
        coef = clf_fitted.coef_[0] if hasattr(clf_fitted, "coef_") else None
        vocab = word_vec_no_words.get_feature_names_out()
        top_feats = None
        if coef is not None:
            idxs = np.argsort(np.abs(coef))[::-1][:20]
            top_feats = [(vocab[i], coef[i]) for i in idxs]
        results_overall.append(dict(split=split_kind, model=name,
                                    accuracy=acc, precision=prec, recall=rec, f1=f1,
                                    top_features=top_feats, removed_words=top_words))
        features_overall.append(dict(split=split_kind, model=name,
                                    top_features=top_feats, removed_words=top_words))

# Save results
output_dir = "results/experiment_3"
os.makedirs(output_dir, exist_ok=True)
df = pd.DataFrame(results_overall)
df.to_csv(os.path.join(output_dir, "exp3_results.csv"), index=False)
feat_df = pd.DataFrame(features_overall)
feat_df.to_csv(os.path.join(output_dir, "exp3_topfeatures.csv"), index=False)

print(df.groupby(["split", "model"])[["accuracy", "precision", "recall", "f1"]].mean())
print("\nSample of top features before/after topic word removal:")
for r in features_overall:
    if "noTopWords" in r["model"]:
        print(f"{r['split']} | {r['model']}")
        print("Top coefficients:", r["top_features"])
        print("Removed: ", r["removed_words"][:10])

print(f"\nResults saved in {output_dir}")
