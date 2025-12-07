import os
import pickle
import numpy as np
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.sparse import hstack, csr_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from datetime import datetime
from metrics.all_features import extract_features_A_to_F
from transformers import __version__ as hf_version
from packaging import version
from sklearn.metrics import confusion_matrix, classification_report

# ======================================================
# Helper functions
# ======================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# ======================================================
# Evaluation Function
# ======================================================
def evaluate_baseline(X_train, y_train, X_test, y_test, name, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Global metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    # Per-class metrics
    per_class_metrics = precision_recall_fscore_support(y_test, y_pred, average=None)
    class_acc = []
    labels = sorted(list(set(y_test)))
    for c in labels:
        mask = np.array(y_test) == c
        class_acc.append(accuracy_score(np.array(y_test)[mask], np.array(y_pred)[mask]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name:25s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_class": {
            "labels": labels,
            "accuracy": class_acc,
            "precision": per_class_metrics[0].tolist(),
            "recall": per_class_metrics[1].tolist(),
            "f1": per_class_metrics[2].tolist(),
        },
        "confusion_matrix": cm.tolist(),
    }

# ======================================================
# Enhanced Results Saving
# ======================================================
def save_results_to_txt(results, file_path, transformer_results=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Experiment Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of train samples: {len(X_train)}, test samples: {len(X_test)}\n")
        f.write(f"{'-'*70}\n")

        for r in results:
            f.write(f"{r['model']:25s} | Acc: {r['accuracy']:.4f} | "
                    f"Prec: {r['precision']:.4f} | Rec: {r['recall']:.4f} | F1: {r['f1']:.4f}\n")
            f.write(f"  → Per-class metrics:\n")
            for i, label in enumerate(r['per_class']['labels']):
                f.write(f"     Class {label}: "
                        f"Acc={r['per_class']['accuracy'][i]:.4f}, "
                        f"Prec={r['per_class']['precision'][i]:.4f}, "
                        f"Rec={r['per_class']['recall'][i]:.4f}, "
                        f"F1={r['per_class']['f1'][i]:.4f}\n")
            f.write(f"  → Confusion Matrix:\n")
            for row in r['confusion_matrix']:
                f.write("     " + " ".join(f"{x:5d}" for x in row) + "\n")
            f.write(f"{'-'*70}\n")

        if transformer_results is not None:
            f.write(f"Transformer (DistilBERT) Results:\n")
            for k, v in transformer_results.items():
                if isinstance(v, (int, float)):
                    f.write(f"  {k:15s}: {v:.4f}\n")
                else:
                    f.write(f"  {k:15s}: {v}\n")
        f.write(f"{'='*70}\n")

def experiment_0():
    # ======================================================
    # Load data
    # ======================================================
    print("Loading datasets...")
    train = load_pickle("datasets/merged_train.pkl")
    test  = load_pickle("datasets/merged_test.pkl")

    X_train = [d["text"] for d in train]
    y_train = [d["label"] for d in train]
    X_test = [d["text"] for d in test]
    y_test = [d["label"] for d in test]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ======================================================
    # Baseline 1: Character n-grams
    # ======================================================
    print("\n[1/5] Extracting Character n-grams (TF-IDF)...")
    vectorized_char_train_path = "datasets/precompute/vectorized_char_train.pkl"
    vectorized_char_test_path  = "datasets/precompute/vectorized_char_test.pkl"

    if os.path.exists(vectorized_char_train_path) and os.path.exists(vectorized_char_test_path):
        print("Found precomputed char vectorization, loading them...")
        X_train_char = load_pickle(vectorized_char_train_path)
        X_test_char  = load_pickle(vectorized_char_test_path)
    else:
        print("No precomputed char vectorization found, extracting them...")
        char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=10000)
        X_train_char = char_vectorizer.fit_transform(tqdm(X_train, desc="Fitting TF-IDF (char)"))
        X_test_char  = char_vectorizer.transform(tqdm(X_test, desc="Transforming TF-IDF (char)"))
        print("Saving char vectorization for later use...")
        save_pickle(X_train_char, vectorized_char_train_path)
        save_pickle(X_test_char, vectorized_char_test_path)

    # ======================================================
    # Baseline 2: Word n-grams
    # ======================================================
    print("\n[2/5] Extracting Word n-grams (TF-IDF)...")
    vectorized_word_train_path = "datasets/precompute/vectorized_word_train.pkl"
    vectorized_word_test_path  = "datasets/precompute/vectorized_word_test.pkl"

    if os.path.exists(vectorized_word_train_path) and os.path.exists(vectorized_word_test_path):
        print("Found precomputed word vectorization, loading them...")
        X_train_word = load_pickle(vectorized_word_train_path)
        X_test_word  = load_pickle(vectorized_word_test_path)
    else:
        print("No precomputed word vectorization found, extracting them...")
        word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000)
        X_train_word = word_vectorizer.fit_transform(tqdm(X_train, desc="Fitting TF-IDF (word)"))
        X_test_word  = word_vectorizer.transform(tqdm(X_test, desc="Transforming TF-IDF (word)"))
        print("Saving word vectorization for later use...")
        save_pickle(X_train_word, vectorized_word_train_path)
        save_pickle(X_test_word, vectorized_word_test_path)

    # ======================================================
    # Baseline 3: Interpretable features (A–F)
    # ======================================================
    print("\n[3/5] Extracting interpretable features A–F...")
    feat_train_path = "datasets/precompute/features_train.pkl"
    feat_test_path  = "datasets/precompute/features_test.pkl"

    if os.path.exists(feat_train_path) and os.path.exists(feat_test_path):
        print("Found precomputed features, loading them...")
        X_train_feat = load_pickle(feat_train_path)
        X_test_feat  = load_pickle(feat_test_path)
    else:
        print("No precomputed features found, extracting them...")
        X_train_feat = np.array([extract_features_A_to_F(t) for t in tqdm(X_train, desc="Train features")])
        X_test_feat  = np.array([extract_features_A_to_F(t) for t in tqdm(X_test, desc="Test features")])
        print("Saving features for later use...")
        save_pickle(X_train_feat, feat_train_path)
        save_pickle(X_test_feat, feat_test_path)
    
    # ======================================================
    # Combined representation
    # ======================================================
    print("\n[4/5] Combining word n-grams and interpretable features...")
    X_train_comb = hstack([X_train_word, csr_matrix(X_train_feat)])
    X_test_comb  = hstack([X_test_word, csr_matrix(X_test_feat)])

    # ======================================================
    # Classical Models: Logistic Regression + SVM
    # ======================================================
    print("\nEvaluating classical models...\n")

    results = []
    classical_models = [
        ("LR - Char ngrams", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"), X_train_char, X_test_char),
        ("LR - Word ngrams", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"), X_train_word, X_test_word),
        ("LR - Interpretable", LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"), X_train_feat, X_test_feat),
        ("LR - Combined", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"), X_train_comb, X_test_comb),
        ("SVM - Word ngrams", LinearSVC(C=1.0, class_weight="balanced"), X_train_word, X_test_word),
        ("SVM - Interpretable", LinearSVC(C=1.0, class_weight="balanced"), X_train_feat, X_test_feat),
        ("SVM - Combined", LinearSVC(C=1.0, class_weight="balanced"), X_train_comb, X_test_comb),
    ]

    for name, clf, Xtr, Xte in tqdm(classical_models, desc="Running classical models"):
        results.append(evaluate_baseline(Xtr, y_train, Xte, y_test, name, clf))

    save_results_to_txt(results, "results/experiment_0_results.txt")


    # ======================================================
    # Transformer (DistilBERT)
    # ======================================================
    print("\n[5/5] Evaluating Transformer (DistilBERT)...")

    model_name = "distilbert-base-uncased"
    save_dir = "./bert_saved"
    tokenizer_path = os.path.join(save_dir)  # Folder, not file

    # Prepare Datasets as before
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    test_ds  = Dataset.from_dict({"text": X_test, "label": y_test})

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
    test_ds  = test_ds.map(tokenize, batched=True, desc="Tokenizing test")

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### --- MAIN CHANGE STARTS HERE---
    load_existing = False
    if (
        os.path.exists(save_dir) and
        os.path.isfile(os.path.join(save_dir, "pytorch_model.bin")) and
        os.path.isfile(os.path.join(save_dir, "tokenizer_config.json"))
    ):
        print("Found existing saved model and tokenizer. Loading them...")
        load_existing = True
    else:
        print("No saved model or tokenizer found. Training a new model...")

    if load_existing:
        # Load saved model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
    else:
        # Train new model as usual
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y_train))).to(device)

        if version.parse(hf_version) >= version.parse("4.8.0"):
            training_args = TrainingArguments(
                output_dir="./bert_results",
                logging_dir="./bert_logs",
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=2,
                fp16=True,
                save_strategy="no",
                logging_steps=100,
                report_to="none"
            )
        else:
            training_args = TrainingArguments(
                output_dir="./bert_results",
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=2,
                fp16=True,
                logging_dir="./bert_logs",
                logging_steps=100
            )

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            acc = accuracy_score(labels, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        # Save newly trained model & tokenizer
        trainer.save_model(save_dir)        # saves the model weights, config, and tokenizer
        tokenizer.save_pretrained(save_dir) # saves vocab + tokenizer config

    ### --- MAIN CHANGE ENDS HERE---

    # For both cases, set up Trainer for evaluation
    if not load_existing:
        transformer_results = trainer.evaluate()
    else:
        # Fast evaluation using loaded model and Trainer
        eval_training_args = TrainingArguments(
            output_dir="./bert_results",
            per_device_eval_batch_size=16,
            do_train=False,
            do_eval=True,
            report_to="none"
        )
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            acc = accuracy_score(labels, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        
        trainer = Trainer(
            model=model,
            args=eval_training_args,
            train_dataset=None,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        transformer_results = trainer.evaluate()

    save_results_to_txt(results, "results/experiment_0_results.txt", transformer_results)
    print("\n All results saved to: results/experiment_0_results.txt")
    breakpoint()

if __name__ == "__main__":
    experiment_0()