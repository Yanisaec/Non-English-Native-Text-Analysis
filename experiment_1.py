import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, mannwhitneyu, normaltest
from datetime import datetime
from experiment_0 import load_pickle, save_pickle

from metrics.all_features import FEATURE_NAMES  # List of feature names; must align with features loaded

def experiment_1():
    # ==========================
    # Load data
    # ==========================
    print("Loading precomputed features and labels...")

    X_train_feat = load_pickle("datasets/precompute/features_train.pkl")
    X_test_feat  = load_pickle("datasets/precompute/features_test.pkl")
    train_data   = load_pickle("datasets/merged_train.pkl")
    test_data    = load_pickle("datasets/merged_test.pkl")

    y_train = [d["label"] for d in train_data]
    y_test  = [d["label"] for d in test_data]
    labels  = sorted(list(set(y_train)))

    assert X_train_feat.shape[1] == len(FEATURE_NAMES), "Number of features and FEATURE_NAMES mismatch"

    # ==========================
    # Train sparse linear model (L1 regularized Logistic Regression)
    # ==========================
    print("Training L1-regularized Logistic Regression on interpretable features...")

    clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000, class_weight="balanced")
    clf.fit(X_train_feat, y_train)
    y_pred = clf.predict(X_test_feat)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print(f"Test Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    # ==========================
    # 1. Feature coefficients
    # ==========================
    coefs = clf.coef_[0] if clf.coef_.ndim == 2 else clf.coef_
    coef_ranking = np.argsort(np.abs(coefs))[::-1]  # Descending by absolute value

    # ==========================
    # 2. Permutation importance
    # ==========================
    print("Computing permutation importances...")
    perm = permutation_importance(clf, X_test_feat, y_test, n_repeats=10, random_state=42)
    perm_ranking = np.argsort(perm.importances_mean)[::-1]

    # ==========================
    # 3. SHAP values
    # ==========================
    print("Computing SHAP values (may take a while)...")
    explainer = shap.LinearExplainer(clf, X_train_feat, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_feat)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_ranking = np.argsort(mean_abs_shap)[::-1]

    # ==========================
    # Get top-N features (by coefficient)
    # ==========================
    TOP_N = 20
    top_n_idx = coef_ranking[:TOP_N]
    top_features = [FEATURE_NAMES[i] for i in top_n_idx]

    print("\nTop features by model coefficient:")
    for i, idx in enumerate(top_n_idx):
        print(f"{i+1:2d}. {FEATURE_NAMES[idx]:30s} | Coef: {coefs[idx]:+.4f} | PermImp: {perm.importances_mean[idx]:+.4f} | SHAP(abs mean): {mean_abs_shap[idx]:.4f}")

    # ==========================
    # Plot feature distributions (native vs non-native)
    # ==========================
    output_dir = "results/experiment_1"
    os.makedirs(output_dir, exist_ok=True)

    native_mask = np.array(y_test) == labels[0]
    nonnative_mask = np.array(y_test) == labels[1]

    effect_sizes = []
    p_values = []

    eps = 1e-10  # Small epsilon to prevent division by zero

    for idx in top_n_idx:
        native_vals = X_test_feat[native_mask, idx]
        nonnative_vals = X_test_feat[nonnative_mask, idx]

        # Test for normality
        _, p_native = normaltest(native_vals)
        _, p_nonnative = normaltest(nonnative_vals)

        if p_native > 0.05 and p_nonnative > 0.05:
            # Use t-test
            stat, p = ttest_ind(native_vals, nonnative_vals, equal_var=False)
        else:
            # Use Mann-Whitney U
            stat, p = mannwhitneyu(native_vals, nonnative_vals, alternative='two-sided')

        # Cohen's d (effect size), safe against zero division
        pooled_std = np.sqrt((native_vals.std() ** 2 + nonnative_vals.std() ** 2) / 2)
        if pooled_std < eps:
            d = 0.0
            print(f"Warning: pooled std for feature '{FEATURE_NAMES[idx]}' is near zero; set effect size to 0.")
        else:
            d = (native_vals.mean() - nonnative_vals.mean()) / pooled_std

        effect_sizes.append(d)
        p_values.append(p)

        # Plot
        plt.figure(figsize=(6, 4))
        plt.hist(native_vals, alpha=0.5, label=f"Native ({labels[0]})")
        plt.hist(nonnative_vals, alpha=0.5, label=f"Non-native ({labels[1]})")
        plt.title(f"Feature: {FEATURE_NAMES[idx]}\nCohen's d = {d:.2f}, p = {p:.2e}")
        plt.xlabel(FEATURE_NAMES[idx])
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_dist_{FEATURE_NAMES[idx]}.png"))
        plt.close()

    # ==========================
    # Multiple testing correction
    # ==========================
    reject, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh")

    # ==========================
    # Save results
    # ==========================
    result_txt = os.path.join(output_dir, "feature_stats.txt")
    with open(result_txt, "w") as f:
        f.write(f"Experiment 1 -- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Top features by coefficient (with stats):\n")
        f.write("Idx | Name | Coef | PermImp | SHAP(abs mean) | EffectSize_d | p_uncorr | p_FDR | Reject_H0\n")
        for i, idx in enumerate(top_n_idx):
            f.write(f"{i+1:2d} | {FEATURE_NAMES[idx]:30s} | {coefs[idx]:+.4f} | {perm.importances_mean[idx]:+.4f} | ")
            f.write(f"{mean_abs_shap[idx]:.4f} | {effect_sizes[i]:+.3f} | {p_values[i]:.2e} | {pvals_corrected[i]:.2e} | {reject[i]}\n")

    print("\n Experiment 1 complete. Stats and plots saved to", output_dir)

if __name__ == "__main__":
    experiment_1()