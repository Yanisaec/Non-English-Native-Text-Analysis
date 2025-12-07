import pandas as pd

INPUT_CSV = "datasets/AI_generated/ai_generated_texts_cleaned.csv"
OUTPUT_CSV = "datasets/AI_generated/ai_generated_texts_indexed.csv"

# Load the cleaned dataset
df = pd.read_csv(INPUT_CSV)

# Add a unique index column
df["text_id"] = range(len(df))

# Reorder columns: text_id first
cols = ["text_id"] + [col for col in df.columns if col != "text_id"]
df = df[cols]

# Save updated CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Indexed CSV saved → {OUTPUT_CSV}")
print(df.head())