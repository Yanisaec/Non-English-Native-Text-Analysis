import pandas as pd
import numpy as np
import pickle
import re
from typing import Dict, Union, List
from collections import Counter
import nltk
from tqdm import tqdm
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('wordnet') 

def analyze_discourse_features(file_path: str) -> Dict[str, Union[float, int]]:
    """
    Calculates discourse and cohesion features (connective frequency and 
    lexical overlap) from a text corpus.

    Args:
        file_path (str): The path to the pickle file containing the dataset.
                         Assumes a list of samples, each with a 'text' key.

    Returns:
        Dict[str, Union[float, int]]: A dictionary of the calculated linguistic features.
    """
    # 1. Data Loading
    try:
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
            
        print(f"Processing {len(data_list)} samples from {file_path}...")
        # Use tqdm for progress during text extraction
        text_samples = [
            str(sample['text']) for sample in tqdm(data_list, desc="Extracting Text")
        ]
        
        corpus = ' '.join(text_samples)
        
    except Exception as e:
        return {"Error": f"File loading failed: {e}"}

    if not corpus.strip():
        return {"Error": "The corpus is empty."}
    
    # --- Initial Tokenization ---
    
    # 1. Sentence Tokenization (for Cohesion)
    sentences = sent_tokenize(corpus)
    total_sentences = len(sentences)
    
    # 2. Word Tokenization (for Connective and general metrics)
    tokens = word_tokenize(corpus)
    total_tokens = len(tokens)
    
    if total_sentences == 0 or total_tokens == 0:
        return {"Error": "Corpus has no content for analysis."}
    
    tokens_per_thousand = total_tokens / 1000
    
    results = {}
    
    # ====================================================================
    #           A. Discourse & Cohesion Features
    # ====================================================================
    
    ## 1. Connective Frequency (per 1000 tokens)
    
    # List of common formal connectives (can be extended)
    CONNECTIVES = ['however', 'therefore', 'moreover', 'thus', 'consequently', 'furthermore', 'nevertheless']
    
    corpus_lower = corpus.lower()
    connective_counts = Counter()
    
    for word in CONNECTIVES:
        # Use regex with word boundaries to avoid matching substrings (e.g., 'however' in 'whatever')
        count = len(re.findall(r'\b' + re.escape(word) + r'\b', corpus_lower))
        connective_counts[word] = count
        results[f'connective_freq_{word}_per_1000'] = count / tokens_per_thousand

    # Total connective usage
    total_connectives_count = sum(connective_counts.values())
    results['total_connectives_per_1000'] = total_connectives_count / tokens_per_thousand
    
    # --------------------------------------------------------------------
    
    ## 2. Lexical Overlap (Cohesion)
    
    # Function to check if a word is a content word (simplified)
    # A simple way is to check if it's not a stop word and not punctuation,
    # or by checking if it has a WordNet entry (as below)
    
    def is_content_word(token: str) -> bool:
        # Checks if the token is primarily alphabetic and has a WordNet sense (a good proxy for content)
        return token.isalpha() and any(wordnet.synsets(token.lower()))

    # Calculate content word tokens for each sentence
    sentence_content_words = [
        set(w.lower() for w in word_tokenize(s) if is_content_word(w)) 
        for s in sentences
    ]
    
    overlap_ratios = []
    
    # Iterate over adjacent sentence pairs
    # Use tqdm to show progress for the cohesion calculation loop
    for i in tqdm(range(total_sentences - 1), desc="Calculating Cohesion"):
        current_words = sentence_content_words[i]
        next_words = sentence_content_words[i+1]
        
        # Jaccard Index (Intersection over Union) is a common overlap metric
        intersection = current_words.intersection(next_words)
        union = current_words.union(next_words)
        
        if union:
            overlap_ratio = len(intersection) / len(union)
            overlap_ratios.append(overlap_ratio)
        else:
            # If both sentences have no content words (e.g., just punctuation), treat as 0 overlap
            overlap_ratios.append(0.0) 

    # The final metric is the average Jaccard Overlap across all adjacent sentence pairs
    avg_lexical_overlap = np.mean(overlap_ratios) if overlap_ratios else 0.0
    
    results['avg_lexical_overlap_jaccard'] = avg_lexical_overlap
    return results

results = analyze_discourse_features('datasets/bawe.pkl') 
print(results)
results = analyze_discourse_features('datasets/toefl11.pkl') 
print(results)
results = analyze_discourse_features('datasets/cell.pkl') 
print(results)
results = analyze_discourse_features('datasets/pelic.pkl') 
print(results)