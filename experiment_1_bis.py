import os
import pickle
import numpy as np
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from experiment_0 import load_pickle, save_pickle
from metrics.all_features import FEATURE_NAMES

# ==========================
# Load datasets and interpretable features
# ==========================
print("Loading datasets and interpretable features...")
train = load_pickle("datasets/merged_train.pkl")
test  = load_pickle("datasets/merged_test.pkl")

X_train_raw = [d["text"] for d in train]
y_train = [d["label"] for d in train]
X_test_raw = [d["text"] for d in test]
y_test = [d["label"] for d in test]

X_train_feat = load_pickle("datasets/precompute/features_train.pkl")
X_test_feat  = load_pickle("datasets/precompute/features_test.pkl")

print(f"Original train size: {len(X_train_raw)}, test size: {len(X_test_raw)}")

# ==========================
# Remove commas and semicolons from texts
# ==========================
def remove_commas_semicolons(texts):
    return [text.replace(",", "").replace(";", "") for text in texts]

X_train = remove_commas_semicolons(X_train_raw)
X_test = remove_commas_semicolons(X_test_raw)

print("Removed commas and semicolons from all texts.")

# ==========================
# Vectorize character n-grams
# ==========================
print("\nExtracting character n-grams (TF-IDF) on modified texts...")
char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=10000)
X_train_char = char_vectorizer.fit_transform(tqdm(X_train, desc="Fitting char TF-IDF"))
X_test_char = char_vectorizer.transform(tqdm(X_test, desc="Transforming char TF-IDF"))

# ==========================
# Vectorize word n-grams
# ==========================
print("\nExtracting word n-grams (TF-IDF) on modified texts...")
word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000)
X_train_word = word_vectorizer.fit_transform(tqdm(X_train, desc="Fitting word TF-IDF"))
X_test_word = word_vectorizer.transform(tqdm(X_test, desc="Transforming word TF-IDF"))

# ==========================
# Zero out the semicolon_rate and comma_rate features in interpretable features
# ==========================
print("\nAdjusting interpretable features by removing semicolon_rate and comma_rate effects...")

# Find indices of the features to zero out
feat_names_lower = [name.lower() for name in FEATURE_NAMES]
semicolon_idx = feat_names_lower.index("semicolon_rate") if "semicolon_rate" in feat_names_lower else None
comma_idx = feat_names_lower.index("comma_rate") if "comma_rate" in feat_names_lower else None

if semicolon_idx is None or comma_idx is None:
    print("Warning: Could not find semicolon_rate or comma_rate feature indices. Skipping zeroing.")
else:
    # Zero out these columns in train and test features
    X_train_feat[:, semicolon_idx] = 0
    X_train_feat[:, comma_idx] = 0
    X_test_feat[:, semicolon_idx] = 0
    X_test_feat[:, comma_idx] = 0
    print(f"Zeroed columns: semicolon_rate (idx {semicolon_idx}), comma_rate (idx {comma_idx})")

# ==========================
# Train & evaluate logistic regression classifiers
# ==========================
def evaluate_clf(X_tr, y_tr, X_te, y_te, name):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average='macro')
    print(f"{name:30s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

print("\nTraining and evaluating logistic regression models on modified text and interpretable features...")

results = []
results.append(evaluate_clf(X_train_char, y_train, X_test_char, y_test, "LR - Char ngrams (no commas/semicolons)"))
results.append(evaluate_clf(X_train_word, y_train, X_test_word, y_test, "LR - Word ngrams (no commas/semicolons)"))
results.append(evaluate_clf(X_train_feat, y_train, X_test_feat, y_test, "LR - Interpretable Features (no ;, rates)"))

# ==========================
# Save results
# ==========================
output_path = "results/experiment_1_bis_results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Experiment 1 bis - Logistic Regression on texts WITHOUT commas and semicolons,\n")
    f.write("with semicolon_rate and comma_rate zeroed out in interpretable features\n\n")
    for r in results:
        f.write(f"{r['model']:30s} | Acc: {r['accuracy']:.4f} | Prec: {r['precision']:.4f} | "
                f"Rec: {r['recall']:.4f} | F1: {r['f1']:.4f}\n")

print(f"\n Results saved to {output_path}")
