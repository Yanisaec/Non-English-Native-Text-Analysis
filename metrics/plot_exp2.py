import matplotlib.pyplot as plt
import numpy as np

# Feature groups
groups = ['A', 'B', 'C', 'D', 'E', 'F']

# ΔF1 values
ablation = [0.1214, 0.0171, 0.0008, 0.0179, 0.0146, -0.0049]
permutation = [0.1282, 0.0613, 0.0125, 0.0477, 0.0126, -0.0036]

x = np.arange(len(groups))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(8, 4))

# Plot bars
rects1 = ax.bar(x - width/2, ablation, width, label='Ablation', color='steelblue')
rects2 = ax.bar(x + width/2, permutation, width, label='Permutation', color='darkorange', hatch='//')

# Labels and title
ax.set_ylabel('ΔF1')
ax.set_xlabel('Feature Group')
ax.set_title('Feature Group Importance: ΔF1')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

# Add values on top of bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.savefig("figures/exp2.png", dpi=300)
plt.show()
