import pandas as pd
import nltk
from collections import Counter
import re
from typing import Dict, Union, List
import pickle 
import tqdm
import math

def analyze_corpus_features(file_path: str) -> Dict[str, Union[float, int]]:
    """
    Calculates relative frequencies (per 1000 tokens) for specific word categories
    and certain ratios from a text corpus loaded from a pickle file.

    Args:
        file_path (str): The path to the pickle file containing the dataset.
                         It's assumed the dataset has a column with the text data
                         (e.g., named 'text').

    Returns:
        Dict[str, Union[float, int]]: A dictionary of the calculated linguistic features.
    """
    try:
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
            
        # 2. Verify the structure and extract the 'text' field
        # We assume data_list is a list, and each item is a dict/object with a 'text' key/attribute.
        
        # Ensure it's a list and not empty
        if not isinstance(data_list, list) or not data_list:
             return {"Error": "Loaded data is not a non-empty list."}
        
        # Extract all 'text' values from the list of samples
        text_samples = []
        for sample in data_list:
            if isinstance(sample['text'], str):
                text_samples.append(sample['text'])

        # 3. Combine all text into a single corpus string
        corpus = ' '.join(text_samples).lower() # Lowercasing here is fine too

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return {}
    except KeyError:
        print("Error: 'text' column not found in the DataFrame. Please check the column name.")
        return {}
    # except Exception as e:
    #     print(f"An unexpected error occurred during file loading: {e}")
    #     return {}

    # 2. Tokenize and Tag the Corpus
    tokens = nltk.word_tokenize(corpus.lower())
    tagged_tokens = nltk.pos_tag(tokens)

    total_tokens = len(tokens)
    if total_tokens == 0:
        return {"Error": "The corpus is empty after loading and tokenization."}

    # 3. Define Word Categories for Direct Counting (Specific Word Lists)
    articles_set = {'a', 'an', 'the'}
    prepositions_set = {'in', 'at', 'on', 'of', 'to', 'for', 'with', 'by', 'from', 'up', 'down', 'out', 'off'} # Extended set for better variety measure

    # 4. Define POS Tags for General Categories (NLTK POS Tags)
    # The tags are based on the standard Penn Treebank tag set:
    # - PRP, PRP$: Pronouns (personal and possessive)
    # - DT: Determiners (includes articles, demonstratives, and quantifiers)
    # - MD, VB, VBD, VBG, VBN, VBP, VBZ (subset): Auxiliaries often fall under various verb/modal tags,
    #   but we'll use a pragmatic approach for Auxiliaries based on specific words and then use
    #   POS tags for Determiners and Pronouns.

    # Specific words for the required categories:
    target_articles = {'a', 'an', 'the'}
    target_prepositions = {'in', 'at', 'on', 'of'} # Specific set for frequency calculation
    definite_article = 'the'
    indefinite_articles = {'a', 'an'}

    # NLTK Tags for broader categories:
    pronoun_tags = {'PRP', 'PRP$'} # Personal and Possessive Pronouns
    determiner_tags = {'DT'}       # Determiners (includes articles, but we'll subtract to avoid double-counting)
    # Common Auxiliary verbs (a more reliable way than broad POS tags)
    auxiliaries_set = {'be', 'is', 'am', 'are', 'was', 'were', 'been', 'being',
                       'have', 'has', 'had', 'having',
                       'do', 'does', 'did',
                       'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'}


    # 5. Perform Counting
    counts = {
        'articles': 0,
        'prepositions': 0,
        'pronouns': 0,
        'determiners': 0,
        'auxiliaries': 0,
        'definite_article': 0,
        'indefinite_articles': 0,
        'unique_prepositions': set(), # For Preposition Variety
    }

    # Use a Counter for easy lookup
    token_counts = Counter(tokens)

    # Count Articles (a, an, the)
    counts['articles'] = sum(token_counts[word] for word in target_articles)
    counts['definite_article'] = token_counts[definite_article]
    counts['indefinite_articles'] = sum(token_counts[word] for word in indefinite_articles)

    # Count Target Prepositions (in, at, on, of)
    counts['prepositions'] = sum(token_counts[word] for word in target_prepositions)

    # Count Auxiliaries
    counts['auxiliaries'] = sum(token_counts[word] for word in auxiliaries_set)

    # Iterate through tagged tokens for POS-based categories and Preposition Variety
    token_set_for_variety = set()

    for word, tag in tagged_tokens:
        word = word.lower() # Already done, but for safety

        # Count Pronouns (using POS tag)
        if tag in pronoun_tags:
            counts['pronouns'] += 1

        # Count Determiners (using POS tag)
        # We include all DTs, even the ones counted as 'articles' above,
        # as 'determiners' is a broader category often requested for feature analysis.
        if tag in determiner_tags:
            counts['determiners'] += 1

        # Track unique prepositions for variety ratio
        if tag == 'IN': # NLTK tag for Prepositions/Subordinating Conjunctions
            counts['unique_prepositions'].add(word)


    # 6. Calculate Features
    results = {}
    tokens_per_thousand = total_tokens / 1000

    # Relative Frequencies per 1000 tokens
    results['freq_articles_per_1000'] = counts['articles'] / tokens_per_thousand
    results['freq_prepositions_per_1000'] = counts['prepositions'] / tokens_per_thousand
    results['freq_pronouns_per_1000'] = counts['pronouns'] / tokens_per_thousand
    results['freq_determiners_per_1000'] = counts['determiners'] / tokens_per_thousand
    results['freq_auxiliaries_per_1000'] = counts['auxiliaries'] / tokens_per_thousand

    # Ratios
    # Definite / Indefinite Article Ratio (a, an / the)
    if counts['indefinite_articles'] > 0:
        results['ratio_definite_indefinite_article'] = counts['definite_article'] / counts['indefinite_articles']
    else:
        results['ratio_definite_indefinite_article'] = 0.0 # Avoid division by zero

    # Preposition Variety (Unique Prepositions / Total Prepositions)
    total_prepositions_in_corpus = sum(1 for _, tag in tagged_tokens if tag == 'IN')
    num_unique_prepositions = len(counts['unique_prepositions'])

    if total_prepositions_in_corpus > 0:
        results['ratio_preposition_variety'] = num_unique_prepositions / total_prepositions_in_corpus
    else:
        results['ratio_preposition_variety'] = 0.0

    return results


print("\n--- Running the analysis ---")
file_path = 'datasets/pelic.pkl'
results = analyze_corpus_features(file_path)

import json
print(json.dumps(results, indent=4))