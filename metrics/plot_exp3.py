import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read experiment results
df = pd.read_csv('results/experiment_3/exp3_results.csv')

# Drop any non-result or header rows
df = df.dropna(subset=['accuracy','precision','recall','f1'])

# Melt for grouped barplot (model-wise metrics)
df_plot = df.melt(id_vars=['split', 'model'], value_vars=['accuracy','precision','recall','f1'],
                 var_name='Metric', value_name='Score')

# Plot metrics grouped by model
plt.figure(figsize=(14,6))
sns.barplot(data=df_plot, x='model', y='Score', hue='Metric', errorbar=None)
plt.title('Model Performance by Metric (Experiment 3)')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0.7,1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('exp3_metrics_bar.png')

# Plot metrics grouped by data split
plt.figure(figsize=(14,6))
sns.barplot(data=df_plot, x='split', y='Score', hue='Metric', errorbar=None)
plt.title('Performance by Data Split (Experiment 3)')
plt.xlabel('Data Split')
plt.ylabel('Score')
plt.ylim(0.7,1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('exp3_split_bar.png')
