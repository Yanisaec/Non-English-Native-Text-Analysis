# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# data = {
#     "Name": [
#         "semicolon_rate","comma_rate","LexicalDensity","prepositions_rate",
#         "MTLD","articles_rate","MSTTR","AvgLogFreq","TTR","pronouns_rate",
#         "uppercase_ratio","subordination_index","lexical_overlap","avg_dep_depth",
#         "connective_rate","error_rate_per_1k","mean_sent_len","auxiliaries_rate"
#     ],
#     "Coef": [
#         -1020.6780,-272.4102,-33.6867,-32.6807,25.0265,-10.0696,5.4803,
#         -5.1735,-3.9657,-3.6199,-1.4306,-0.8781,0.2090,-0.0440,0.0433,
#         -0.0302,0.0089,0.0000
#     ],
#     "SHAP": [
#         0.4853,0.7713,1.0413,0.5072,0.0,0.1864,0.1398,0.7086,0.2411,
#         0.0782,0.0091,0.2332,0.2753,0.0162,0.0044,0.8007,0.0473,0.0
#     ],
# }

# df = pd.DataFrame(data)

# # --- Modification for Log Scale ---
# # Handle the zero coefficient: Log of zero is undefined.
# # We replace 0.0 with a small positive number (1e-4) to allow it to be plotted on the log scale.
# df['Coef'] = df['Coef'].replace(0.0, 1e-4)

# # Calculate the absolute value of the coefficient (magnitude)
# df["Abs_Coef"] = df["Coef"].abs()

# # Sort by the absolute value for clearer visualization of feature importance magnitude
# df = df.sort_values("Abs_Coef", ascending=True)

# # Plot the absolute value of the coefficient and set x-axis to log scale
# plt.figure(figsize=(8,6))
# plt.barh(df["Name"], df["Abs_Coef"], color=["#d62728" if c<0 else "#2ca02c" for c in df["Coef"]])
# plt.xscale('log') # The logscale application

# plt.xlabel("Absolute Coefficient Magnitude (Log Scale)")
# plt.title("Top Features by Absolute Coefficient Magnitude (LR + L1)")
# plt.grid(axis="x", linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.savefig("exp1_topcoef_logscale.png", dpi=300)
# # plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data (same as before)
data = {
    "Name": [
        "semicolon_rate","comma_rate","LexicalDensity","prepositions_rate",
        "MTLD","articles_rate","MSTTR","AvgLogFreq","TTR","pronouns_rate",
        "uppercase_ratio","subordination_index","lexical_overlap","avg_dep_depth",
        "connective_rate","error_rate_per_1k","mean_sent_len","auxiliaries_rate"
    ],
    "Coef": [
        -1020.6780,-272.4102,-33.6867,-32.6807,25.0265,-10.0696,5.4803,
        -5.1735,-3.9657,-3.6199,-1.4306,-0.8781,0.2090,-0.0440,0.0433,
        -0.0302,0.0089,0.0000
    ],
    "SHAP": [
        0.4853,0.7713,1.0413,0.5072,0.0,0.1864,0.1398,0.7086,0.2411,
        0.0782,0.0091,0.2332,0.2753,0.0162,0.0044,0.8007,0.0473,0.0
    ],
}

df = pd.DataFrame(data)

# Replace zero SHAP values with a small epsilon to enable log scale
df["SHAP"] = df["SHAP"].replace(0, 1e-4)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    df["Coef"], df["SHAP"], 
    c=df["Coef"], cmap="coolwarm", s=80, edgecolors="k", alpha=0.8
)
plt.xlim(df["Coef"].min()*1.1, df["Coef"].max()*1.1)
plt.yscale("log")
plt.xlabel("Coefficient")
plt.ylabel("Mean |SHAP| Value (log scale)")
plt.title("Feature Importance: Coefficient vs SHAP (log scale)")
plt.colorbar(scatter, label="Coefficient value")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("exp1_coef_shap_log.png", dpi=300)
plt.show()
