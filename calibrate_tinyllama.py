import os
import pickle
import random
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import torch
import pandas as pd

from ai_detectors.Binoculars.binoculars.binoculars_tinylama_4bit import BinocularsTinyLlama

# ----------------------------
# SETTINGS
# ----------------------------
BATCH_SIZE = 16
NUM_HUMAN = 200
NUM_AI = 200
MAX_TOKENS = 512
TEMPERATURE = 1.0

DATASETS = [
    ("asap2", "datasets/asap2.pkl"),
    ("persuade", "datasets/persuade.pkl"),
    ("toefl11", "datasets/toefl11.pkl")
]

AI_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# FUNCTIONS
# ----------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def sample_human_texts(datasets, num_samples):
    all_texts = []
    for name, path in datasets:
        data = load_pickle(path)
        all_texts.extend([d["text"] for d in data])
    random.shuffle(all_texts)
    return all_texts[:num_samples]

def extract_prompts_from_datasets(datasets):
    prompts = []
    for name, path in datasets:
        data = load_pickle(path)
        for d in data:
            if "prompt" in d:
                prompts.append(d["prompt"])
    return prompts

def generate_ai_texts(model, tokenizer, prompts, num_texts, target_lengths):
    """
    Generates AI texts using prompts, trying to match the length distribution of target_lengths (in tokens).
    """
    generated_texts = []
    prompt_idx = 0
    while len(generated_texts) < num_texts:
        # Cycle through prompts
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1

        # Target length in tokens
        target_len = target_lengths[len(generated_texts) % len(target_lengths)]

        # Generate text
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).input_ids.to(DEVICE)
        output = model.generate(input_ids, max_new_tokens=target_len)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_texts.append(text)

    return generated_texts[:num_texts]

def compute_scores_in_batches(bino, texts, batch_size=BATCH_SIZE):
    scores = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_scores = bino.compute_score(batch, temperature=TEMPERATURE)
        scores.extend(batch_scores)
    return scores

def calibrate_threshold(scores, labels):
    best_thr, best_f1 = None, 0
    for thr in np.linspace(min(scores), max(scores), 200):
        preds = [1 if s < thr else 0 for s in scores]
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1

# ----------------------------
# MAIN
# ----------------------------
def main():
    # Initialize TinyLlama binoculars
    BINO = BinocularsTinyLlama(max_token_observed=MAX_TOKENS, vocab_chunk_size=32000)
    tokenizer = BINO.tokenizer

    # Load human texts
    print("Sampling human-written texts...")
    human_texts = sample_human_texts(DATASETS, NUM_HUMAN)

    # Compute token lengths for human texts
    human_token_lengths = [len(tokenizer(txt).input_ids) for txt in human_texts]

    # Extract prompts from datasets
    prompts = extract_prompts_from_datasets(DATASETS)
    if len(prompts) == 0:
        # Fallback prompts
        prompts = [
            "Write a short essay about climate change.",
            "Explain the process of photosynthesis.",
            "Describe the steps of federated learning in simple terms.",
            "Write a persuasive paragraph on the importance of AI in education.",
            "Summarize the main ideas of a research paper about computer vision."
        ]

    # Generate AI texts matching human token lengths
    print("Generating AI texts with length matching...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ai_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_NAME, use_fast=True)
    ai_model = AutoModelForCausalLM.from_pretrained(
        AI_MODEL_NAME,
        device_map="auto",
        load_in_4bit=True
    ).eval()

    ai_texts = generate_ai_texts(ai_model, ai_tokenizer, prompts, NUM_AI, human_token_lengths)

    # Combine for calibration
    all_texts = human_texts + ai_texts
    labels = [0]*NUM_HUMAN + [1]*NUM_AI

    print("Computing binoculars scores...")
    scores = compute_scores_in_batches(BINO, all_texts, BATCH_SIZE)

    print("Calibrating threshold...")
    best_thr, best_f1 = calibrate_threshold(scores, labels)
    BINO.threshold = best_thr

    print(f"Calibration completed. Best threshold = {best_thr:.8f}, F1 = {best_f1:.8f}")

    # Save calibration results
    os.makedirs("calibration", exist_ok=True)
    df = pd.DataFrame({"text": all_texts, "label": labels, "score": scores})
    df.to_csv("calibration/tinylama_threshold_calibration.csv", index=False)

if __name__ == "__main__":
    main()
