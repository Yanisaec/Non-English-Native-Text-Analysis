import pandas as pd
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
INPUT_CSV = "datasets/AI_generated/ai_generated_texts_indexed.csv"
OUTPUT_CSV = "datasets/AI_generated/transformed_texts_genre_based_perturbations.csv"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

NUM_SAMPLES_PER_PROMPT = 20
MAX_NEW_TOKENS = 250
TEMPERATURE = 0.9

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# Sample 5 per prompt_id
sampled_df = df.groupby("prompt_id").head(NUM_SAMPLES_PER_PROMPT).copy()

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_4bit=True
).eval()

# ----------------------------------------------------
# HELPER: RUN CHAT MODEL
# ----------------------------------------------------
def apply_transformation(text, instruction):
    system_prompt = (
        "You rewrite text according to the user instruction. "
        "Only output the rewritten text."
        "Do not repeat the task you've been given to do at the beginning of your answer."
        "Do not repeat the original text in your answer."
    )
        # "Do not add any acknowledgement, introduction, explanation, or polite phrasing. "
        # "Do not add any acknowledgement. "
        # "Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. "

    chat_prompt = (
        # f"<|system|>{system_prompt}</s>"
        f"<|user|>Original text:\n{text}\n\n{instruction}</s>"
        "<|assistant|>"
    )

    input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        top_p=0.9
    )

    # Extract model continuation (remove prompt)
    gen_ids = output[0][len(input_ids[0]):]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return generated

# ----------------------------------------------------
# TRANSFORMATION INSTRUCTIONS
# ----------------------------------------------------
# transformations = {
#     "less_formal": "Rewrite the text in a more casual and less formal tone while keeping the meaning unchanged. Don't return ANYTHING other than the modified text.",
#     "noisy_human": "Rewrite the text with a few natural human-like imperfections, such as minor typos, contractions, or slight inconsistencies, but keep it readable. Don't return ANYTHING other than the modified text.",
#     "more_human_flow": "Rewrite the text to sound more naturally human, adding discourse markers, varied sentence length, and a more conversational flow. Don't return ANYTHING other than the modified text."
# }
# transformations = {
#     "lexical_density": "Rewrite the text with less lexical density while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
#     "articles_rate": "Rewrite the text with a higher articles rate while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
#     "prepositions_rate": "Rewrite the text with less prepositions while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
#     "pronouns_rate": "Rewrite the text with a higher pronouns rate while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
#     "auxiliaries_rate": "Rewrite the text with a higher auxiliaries rate while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
# }
# transformations = {
#     "punctuation_variation": "Rewrite the text with human-like punctuation variation (Use dashes, semicolons, ellipses, parentheses, etc.) while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
#     "comma_irregularity": "Rewrite the text with slight comma irregularities while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
#     "non_intrusive_typos": "Rewrite the text with non intrusive typos (e.g., “enviroment” → “environment”) while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
# }
# transformations = {
#     "rewrite_benchmark": "Rewrite the text while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. "
# }
transformations = {
    "journalistic_style": "Rewrite the text in a journalistic style (more concise, fact-focused) while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
    "blog_style": "Rewrite the text in a blog style (more informal, with personal commentary) while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. ",
    "academic_style": "Rewrite the text in an academic-like but messy style while keeping the meaning unchanged. Do not say things like 'Sure', 'Certainly', 'Here is the rewritten text', etc. Do not add any acknowledgement. "
}

# ----------------------------------------------------
# INITIALIZE CSV WITH HEADER (OVERWRITES OLD FILE)
# ----------------------------------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text_id", "prompt_id", "transformation_name", "transformed_text"])
    writer.writeheader()

# ----------------------------------------------------
# PROCESS TEXTS AND WRITE INCREMENTALLY
# ----------------------------------------------------
print("Applying transformations (incremental saving)...")

with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text_id", "prompt_id", "transformation_name", "transformed_text"])

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        text_id = row["text_id"]
        prompt_id = row["prompt_id"]
        original = row["generated_text"]

        for name, instr in transformations.items():
            transformed = apply_transformation(original, instr)

            writer.writerow({
                "text_id": text_id,
                "prompt_id": prompt_id,
                "transformation_name": name,
                "transformed_text": transformed
            })
