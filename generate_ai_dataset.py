import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm

# ----------------------------
# SETTINGS
# ----------------------------
AI_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_PROMPTS = 10
NUM_GENERATIONS_PER_PROMPT = 20
MAX_NEW_TOKENS = 250
TEMPERATURE = 0.8

OUTPUT_FOLDER = "datasets/AI_generated/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_PROMPTS_CSV = f"{OUTPUT_FOLDER}/generation_prompts.csv"
OUTPUT_TEXTS_CSV = f"{OUTPUT_FOLDER}/ai_generated_texts.csv"

# ----------------------------
# DEFINE GENERATION PROMPTS
# ----------------------------
generation_prompts = [
    "Write a detailed explanation of how renewable energy technologies work.",
    "Describe the social impact of artificial intelligence in modern society.",
    "Explain the process of natural selection using simple language.",
    "Write a short essay about the challenges of urban transportation.",
    "Describe how machine learning models are trained and evaluated.",
    "Write a narrative paragraph describing a futuristic city.",
    "Explain why biodiversity is important for ecological stability.",
    "Write an argument supporting the use of open-source software in education.",
    "Summarize the main challenges in global climate policy.",
    "Describe the daily life of a scientist working in a research laboratory."
]

# ----------------------------
# SAVE PROMPTS TO CSV
# ----------------------------
with open(OUTPUT_PROMPTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_id", "prompt"])
    for pid, prompt in enumerate(generation_prompts):
        writer.writerow([pid, prompt])

print(f"Saved prompts -> {OUTPUT_PROMPTS_CSV}")

# ----------------------------
# LOAD MODEL
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    AI_MODEL_NAME,
    device_map="auto",
    load_in_4bit=True
).eval()

# ----------------------------
# GENERATE TEXT DATASET
# ----------------------------
rows = []

print("Generating AI texts...")
for pid, prompt in enumerate(tqdm(generation_prompts)):
    for _ in tqdm(range(NUM_GENERATIONS_PER_PROMPT)):
        # input_ids = tokenizer(
        #     prompt,
        #     return_tensors="pt"
        # ).input_ids.to(device)
        chat_prompt = f"<|system|>You are a helpful writing assistant.</s><|user|>{prompt}</s><|assistant|>"
        input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(device)

        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
        )

        # generated_ids = output[0][len(input_ids[0]):]   # remove prompt
        generated_ids = output[0]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        rows.append([pid, prompt, text])

# ----------------------------
# SAVE GENERATED TEXTS
# ----------------------------
with open(OUTPUT_TEXTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_id", "prompt", "generated_text"])
    writer.writerows(rows)

print(f"Saved generated texts -> {OUTPUT_TEXTS_CSV}")
print("Dataset generation complete.")
