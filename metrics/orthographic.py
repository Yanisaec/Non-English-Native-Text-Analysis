import pandas as pd
import numpy as np
import pickle
import re
from typing import Dict, Union, List
from collections import Counter
import nltk

def analyze_surface_features(file_path: str) -> Dict[str, Union[float, int]]:
    """
    Calculates surface and orthographic features (character n-grams, punctuation rates, 
    and capitalization irregularities) from a text corpus.

    Args:
        file_path (str): The path to the pickle file containing the dataset.
                         Assumes a list of samples, each with a 'text' key.

    Returns:
        Dict[str, Union[float, int]]: A dictionary of the calculated linguistic features.
    """
    # 1. Data Loading (Assuming list of samples structure)
    try:
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
            
        # 2. Wrap the sample extraction with tqdm
        # This will show progress as the 'text' is extracted and converted to string
        print(f"Processing {len(data_list)} samples from {file_path}...")
        text_samples = [
            str(sample['text']) for sample in data_list
        ]
        
        corpus = ' '.join(text_samples)
        
    except Exception as e:
        return {"Error": f"File loading failed: {e}"}

    if not corpus.strip():
        return {"Error": "The corpus is empty."}

    # --- 2. Initial Counts and Tokenization ---
    
    # Total characters (excluding spaces) is a good normalization baseline
    corpus_no_space = re.sub(r'\s+', '', corpus)
    total_chars = len(corpus_no_space)
    
    # Use tokens for word-level normalization (for capitalization)
    # NOTE: Tokenization can be slow for very large corpora, but the progress bar is better placed on the
    # list comprehension above as it's an explicit loop over data items.
    tokens = nltk.word_tokenize(corpus)
    total_tokens = len(tokens)
    
    if total_chars == 0 or total_tokens == 0:
        return {"Error": "Corpus has no content for analysis."}
    
    results = {}
    
    # --- A. Character N-grams (1, 2, 3 grams) ---
    
    char_ngrams = Counter()
    
    # Calculate n-grams on the text without spaces for style-based analysis
    # NOTE: You can also choose to calculate these ON THE FULL CORPUS (including spaces) 
    # if you want to capture spacing patterns, but typically they are calculated on chars only.
    
    # Helper function for generating n-grams
    def get_char_ngrams(text: str, n: int):
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    # 1-grams (Monograms)
    char_ngrams.update(get_char_ngrams(corpus_no_space, 1))
    
    # 2-grams (Bigrams)
    char_ngrams.update(get_char_ngrams(corpus_no_space, 2))

    # 3-grams (Trigrams)
    char_ngrams.update(get_char_ngrams(corpus_no_space, 3))
    
    # Store top 10 most frequent N-grams (normalized by total characters)
    for n_size in [1, 2, 3]:
        # Filter for n-grams of the current size
        n_grams_filtered = {k: v for k, v in char_ngrams.items() if len(k) == n_size}
        
        # Get top 5 of each type for a concise report
        for (ngram, count) in Counter(n_grams_filtered).most_common(10):
            # Store as frequency per 1000 characters
            key = f'char_ngram_{n_size}_freq_{ngram}'
            results[key] = (count / total_chars) * 1000

    # --- B. Punctuation Use (Rates per 1000 tokens) ---
    
    corpus_lower = corpus.lower()
    tokens_per_thousand = total_tokens / 1000
    
    # The count includes all occurrences in the raw corpus
    punctuation_counts = {
        'comma_rate': corpus.count(','),
        'semicolon_rate': corpus.count(';'),
        'quotation_marks_rate': corpus.count('"') + corpus.count("'"), # Both single and double quotes
        'parentheses_rate': corpus.count('(') + corpus.count(')'),
    }
    
    for key, count in punctuation_counts.items():
        results[key.replace('_rate', '_per_1000')] = count / tokens_per_thousand

    # --- C. Capitalization Irregularities ---
    
    capital_irregularities_count = 0
    # The first token is often capitalized for sentence start, so we start counting from the second token.
    # However, a simpler, more robust metric is the rate of words that are ALL CAPS.
    
    # 1. Total tokens in all caps (strong indicator of shouting/emphasis irregularity)
    all_caps_count = sum(1 for token in tokens if token.isupper() and token.isalpha())
    
    # 2. Capitalization of non-initial words (indicator of improper noun/mid-sentence capitalization)
    # This checks words that are NOT the start of a sentence and start with a capital letter.
    # A precise check requires sentence segmentation, but we'll use a pragmatic approximation.
    
    # Simple Heuristic: Count internal words that are capitalized.
    # This regex finds a capital letter that is preceded by a lowercase letter and a space (i.e., mid-sentence)
    mid_sentence_cap_regex = re.compile(r'([a-z])\s+([A-Z])', re.DOTALL)
    mid_sentence_cap_count = len(mid_sentence_cap_regex.findall(corpus))

    results['irregularity_all_caps_per_1000'] = all_caps_count / tokens_per_thousand
    results['irregularity_mid_sentence_cap_per_1000'] = mid_sentence_cap_count / tokens_per_thousand
    
    return results

results = analyze_surface_features('datasets/pelic.pkl') 
print(results)