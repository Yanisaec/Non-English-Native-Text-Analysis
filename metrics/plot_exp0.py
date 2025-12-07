# import matplotlib.pyplot as plt
# import numpy as np

# # Experiment 0 summary (Accuracy, F1)
# models = ["LR-Char", "LR-Word", "LR-Interpretable", "LR-Combined",
#           "SVM-Word", "SVM-Interpretable", "SVM-Combined", "DistilBERT"]

# accuracy = [0.8783, 0.8904, 0.7236, 0.9091, 0.9280, 0.7633, 0.9497, 0.9743]
# f1 =       [0.8374, 0.8498, 0.6590, 0.8719, 0.8967, 0.7024, 0.9249, 0.9593]

# x = np.arange(len(models))
# width = 0.35

# plt.figure(figsize=(10, 5))
# plt.bar(x - width/2, accuracy, width, label='Accuracy', alpha=0.8)
# plt.bar(x + width/2, f1, width, label='F1-score', alpha=0.8)

# plt.xticks(x, models, rotation=30, ha='right')
# plt.ylabel('Score')
# plt.ylim(0.6, 1.0)
# plt.title('Experiment 0 — Model Performance Comparison')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig("experiment0_results.png", dpi=300)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define the models
models = ["LR-Interpretable", "LR-Combined", "SVM-Interpretable", "SVM-Combined", "DistilBERT"]

# Example results: per-class F1 and Recall (class 0 = native, class 1 = non-native)
# ⚠️ Replace these with your real results if available
f1_class0 = [0.69, 0.88, 0.73, 0.93, 0.96]
f1_class1 = [0.63, 0.86, 0.67, 0.92, 0.96]
recall_class0 = [0.71, 0.89, 0.75, 0.94, 0.97]
recall_class1 = [0.60, 0.84, 0.66, 0.91, 0.96]

x = np.arange(len(models))
width = 0.35

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# ---- F1 per class ----
axs[0].bar(x - width/2, f1_class0, width, label='Class 0 (Native)', alpha=0.8)
axs[0].bar(x + width/2, f1_class1, width, label='Class 1 (Non-native)', alpha=0.8)
axs[0].set_xticks(x)
axs[0].set_xticklabels(models, rotation=30, ha='right')
axs[0].set_ylabel('F1-score')
axs[0].set_title('Per-Class F1')
axs[0].set_ylim(0.5, 1.0)
axs[0].grid(axis='y', linestyle='--', alpha=0.5)
axs[0].legend()

# ---- Recall per class ----
axs[1].bar(x - width/2, recall_class0, width, label='Class 0 (Native)', alpha=0.8)
axs[1].bar(x + width/2, recall_class1, width, label='Class 1 (Non-native)', alpha=0.8)
axs[1].set_xticks(x)
axs[1].set_xticklabels(models, rotation=30, ha='right')
axs[1].set_ylabel('Recall')
axs[1].set_title('Per-Class Recall')
axs[1].set_ylim(0.5, 1.0)
axs[1].grid(axis='y', linestyle='--', alpha=0.5)
axs[1].legend()

plt.suptitle("Fairness Analysis: Class 0 vs Class 1 Performance", fontsize=14, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("fairness_analysis.png", dpi=300)
plt.show()