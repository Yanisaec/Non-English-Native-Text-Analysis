import pandas as pd
import spacy
from collections import Counter
import numpy as np
import pickle
from typing import Dict, Union, List
import tqdm 

# Load the spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    # Exit or handle the error gracefully
    raise

def analyze_syntactic_complexity(file_path: str) -> Dict[str, Union[float, int]]:
    # --- 1. Data Loading (List of individual text samples) ---
    try:
        # ... (Same loading logic as before to get text_samples: List[str]) ...
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
            
        if not isinstance(data_list, list) or not data_list:
             return {"Error": "Loaded data is not a non-empty list."}
        
        # This list holds your individual documents/samples
        text_samples = [str(sample['text']) for sample in data_list] 
    except Exception as e:
        # ... (error handling) ...
        return {"Error": f"File loading failed: {e}"}

    # --- 2. Initialization for Aggregates ---
    pos_tags = []
    sentence_lengths = []
    subordinating_conjunctions_count = 0
    dependency_lengths = []
    parse_tree_depths = []
    
    # Specific Dependency Relation Counts
    dep_counts = Counter({
        'nsubj_count': 0,
        'obj_omission_count': 0,
        'aux_usage_count': 0,
    })

    # --- 3. Process Text in Batches using nlp.pipe() ---
    
    # Process the list of texts. Use a small batch size (e.g., 500)
    # The disable argument ensures only the necessary components (tagger, parser) run.
    docs_iterator = nlp.pipe(
        text_samples,
        batch_size=1000,
        disable=["ner", "textcat"] 
    )

    for doc in tqdm.tqdm(docs_iterator):
        # Loop through sentences and tokens *within this doc*
        for sent in doc.sents:
            sentence_lengths.append(len(sent))
            
            # POS Tag sequence for n-grams
            sent_pos_tags = [token.pos_ for token in sent]
            pos_tags.extend(sent_pos_tags)
            
            # Subordination Index: Count SCONJ
            subordinating_conjunctions_count += sum(1 for token in sent if token.pos_ == 'SCONJ')

            # Dependency and Tree Metrics
            sent_dependency_lengths = []
            sent_depths = []
            
            for token in sent:
                # Dependency Length
                if token.dep_ != 'ROOT':
                    dep_len = abs(token.i - token.head.i)
                    sent_dependency_lengths.append(dep_len)
                
                # Dependency Relations Counts
                if token.dep_ == 'nsubj':
                    dep_counts['nsubj_count'] += 1
                if token.dep_ == 'aux':
                    dep_counts['aux_usage_count'] += 1
                
                # Tree Depth Calculation
                depth = 0
                curr = token
                while curr.head != curr:
                    curr = curr.head
                    depth += 1
                sent_depths.append(depth)
                
            dependency_lengths.extend(sent_dependency_lengths)
            if sent_depths:
                parse_tree_depths.append(max(sent_depths))
            
            # Object Omission (Heuristic check)
            for token in sent:
                if token.pos_ == 'VERB' and token.head.pos_ != 'AUX':
                    has_dobj = any(child.dep_ == 'dobj' for child in token.children)
                    if not has_dobj:
                        dep_counts['obj_omission_count'] += 0.5 

    # --- 5. Calculation of Final Metrics ---
    results = {}
    
    num_sentences = len(sentence_lengths)
    
    # 5.1 Clause/Sentence Length Stats
    results['mean_sentence_length'] = np.mean(sentence_lengths) if num_sentences else 0.0
    results['variance_sentence_length'] = np.var(sentence_lengths) if num_sentences else 0.0

    # 5.2 POS N-gram Frequencies (1-3 grams)
    pos_ngram_counts = Counter()
    
    # Function to create n-grams
    def get_ngrams(tags: List[str], n: int):
        return [tuple(tags[i:i+n]) for i in range(len(tags) - n + 1)]

    # 1-grams (Unigrams)
    pos_ngram_counts.update(get_ngrams(pos_tags, 1))
    
    # 2-grams (Bigrams)
    pos_ngram_counts.update(get_ngrams(pos_tags, 2))

    # 3-grams (Trigrams)
    pos_ngram_counts.update(get_ngrams(pos_tags, 3))
    
    total_tokens = len(pos_tags)
    
    if total_tokens > 0:
        # Get the top 10 most frequent 1, 2, and 3-grams
        for n_size in [1, 2, 3]:
            # Filter for n-grams of the current size
            n_grams_filtered = {k: v for k, v in pos_ngram_counts.items() if len(k) == n_size}
            
            # Get top 5 of each type for a concise report
            for (ngram, count) in Counter(n_grams_filtered).most_common(5):
                # Store as frequency per 1000 tokens
                key = f'pos_ngram_{n_size}_freq_{"_".join(ngram)}'
                results[key] = (count / total_tokens) * 1000
    
    # 5.3 Subordination Index
    if num_sentences > 0:
        results['subordination_index'] = subordinating_conjunctions_count / num_sentences
    else:
        results['subordination_index'] = 0.0
        
    # 5.4 Parse Tree Depth and Average Dependency Length
    results['mean_parse_tree_depth'] = np.mean(parse_tree_depths) if parse_tree_depths else 0.0
    results['average_dependency_length'] = np.mean(dependency_lengths) if dependency_lengths else 0.0

    # 5.5 Dependency Relation Counts (per 1000 tokens)
    if total_tokens > 0:
        tokens_per_thousand = total_tokens / 1000
        for key, count in dep_counts.items():
            results[key.replace('_count', '_per_1000')] = count / tokens_per_thousand

    return results

results = analyze_syntactic_complexity('datasets/pelic.pkl')
print(results)