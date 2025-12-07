import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.inspection import permutation_importance
from sklearn.utils import shuffle

from experiment_0 import load_pickle, save_pickle
from metrics.all_features import FEATURE_NAMES, FEATURE_GROUPS

def experiment_2():
    # Select groups relevant to your features only
    groups = sorted(set(FEATURE_GROUPS.values()))
    groups = [g for g in groups if g != "Other"]

    print(f"Feature groups: {groups}")

    # -------------------
    # Load features & labels
    X_train = load_pickle("datasets/precompute/features_train.pkl")
    X_test = load_pickle("datasets/precompute/features_test.pkl")
    train = load_pickle("datasets/merged_train.pkl")
    test = load_pickle("datasets/merged_test.pkl")

    y_train = [d["label"] for d in train]
    y_test = [d["label"] for d in test]

    # -------------------
    # Helper functions
    def evaluate_model(Xtr, ytr, Xte, yte):
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        _, _, f1, _ = precision_recall_fscore_support(yte, ypred, average='macro')
        return f1

    def ablate_group(X, group, feature_indices):
        # Return X with columns for this group zeroed out
        X_ablate = X.copy()
        X_ablate[:, feature_indices] = 0
        return X_ablate

    def permute_group(X, group_idx_cols, random_state=42):
        # Permute columns of group independently per column
        X_perm = X.copy()
        rng = np.random.RandomState(random_state)
        for col in group_idx_cols:
            X_perm[:, col] = shuffle(X_perm[:, col], random_state=rng)
        return X_perm

    # -------------------
    # Prepare feature indices by group
    group_to_indices = {g: [] for g in groups}
    for i, feat in enumerate(FEATURE_NAMES):
        grp = FEATURE_GROUPS.get(feat, "Other")
        if grp in groups:
            group_to_indices[grp].append(i)

    # -------------------
    # Baseline performance on full features
    print("Evaluating baseline performance on full feature set...")
    baseline_f1 = evaluate_model(X_train, y_train, X_test, y_test)
    print(f"Baseline macro F1: {baseline_f1:.4f}")

    # -------------------
    # Ablation experiment
    print("Running ablation experiments by feature group...")
    ablation_results = {}
    for grp in groups:
        print(f"Ablating group {grp} with {len(group_to_indices[grp])} features")
        X_train_ablate = ablate_group(X_train, grp, group_to_indices[grp])
        X_test_ablate = ablate_group(X_test, grp, group_to_indices[grp])
        f1_ablate = evaluate_model(X_train_ablate, y_train, X_test_ablate, y_test)
        ablation_results[grp] = baseline_f1 - f1_ablate
        print(f"ΔF1 after ablating group {grp}: {ablation_results[grp]:.4f}")

    # -------------------
    # Permutation importance experiment
    print("Running permutation importance experiments by feature group...")
    permutation_results = {}
    for grp in groups:
        print(f"Permuting group {grp} with {len(group_to_indices[grp])} features")
        X_test_perm = X_test.copy()
        # Permute columns of this group in test set only
        for col in group_to_indices[grp]:
            X_test_perm[:, col] = shuffle(X_test_perm[:, col], random_state=42)
        f1_perm = evaluate_model(X_train, y_train, X_test_perm, y_test)
        permutation_results[grp] = baseline_f1 - f1_perm
        print(f"ΔF1 after permuting group {grp}: {permutation_results[grp]:.4f}")

    # -------------------
    # Plot heatmap of ΔF1 for ablation and permutation
    import matplotlib.pyplot as plt
    import seaborn as sns

    result_matrix = np.vstack([
        [ablation_results[g] for g in groups],
        [permutation_results[g] for g in groups]
    ])

    plt.figure(figsize=(10, 4))
    sns.heatmap(result_matrix, annot=True, xticklabels=groups, yticklabels=["Ablation", "Permutation"], cmap="Reds", cbar_kws={"label": "ΔF1"})
    plt.title("Experiment 2: Feature Group Ablation and Permutation Importance (ΔF1)")
    plt.tight_layout()

    output_dir = "results/experiment_2"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "feature_group_ablation_permutation_heatmap.png"))
    plt.close()

    # -------------------
    # Save results to file
    with open(os.path.join(output_dir, "experiment_2_results.txt"), "w") as f:
        f.write(f"Baseline macro F1: {baseline_f1:.4f}\n\n")
        f.write("Feature Group Ablation (delta F1):\n")
        for g in groups:
            f.write(f"{g}: {ablation_results[g]:.4f}\n")
        f.write("\nFeature Group Permutation Importance (delta F1):\n")
        for g in groups:
            f.write(f"{g}: {permutation_results[g]:.4f}\n")

    print(f"Experiment 2 complete. Results and heatmap saved in {output_dir}")

if __name__ == '__main__':
    experiment_2()