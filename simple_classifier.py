import pickle
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.utils import shuffle
import random 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

train = load_pickle("datasets/merged_train.pkl")
test  = load_pickle("datasets/merged_test.pkl")

X_train_raw = [d["text"] for d in train]
y_train = [d["label"] for d in train]

X_test_raw = [d["text"] for d in test]
y_test = [d["label"] for d in test]

# print(f"Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")

def preprocess_text_safe(text):
    # Skip NaNs
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None  # or return "missing_text" if you prefer
    text = str(text).lower()          # lowercase
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()

# Apply safe preprocessing
X_train = [preprocess_text_safe(t) for t in X_train_raw]
X_test  = [preprocess_text_safe(t) for t in X_test_raw]

# Remove None entries (optional)
filtered_train = [(x,y) for x,y in zip(X_train, y_train) if x is not None]
X_train, y_train = zip(*filtered_train)

filtered_test = [(x,y) for x,y in zip(X_test, y_test) if x is not None]
X_test, y_test = zip(*filtered_test)

def chunk_by_sentence(text, max_words=100):
    # Split text into sentences using period, question mark, exclamation mark
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        word_count = len(sent.split())
        # Start a new chunk if adding this sentence exceeds max_words
        if current_len + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent)
        current_len += word_count
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# # Apply to class 1 texts
# class1_texts = [t for t,l in zip(X_train, y_train) if l == 1]
# class1_chunks = []
# for t in class1_texts:
#     class1_chunks.extend(chunk_by_sentence(t, max_words=40))

# # Labels for chunks
# class1_labels = [1] * len(class1_chunks)
# class0_texts = [t for t,l in zip(X_train, y_train) if l == 0]
# class0_labels = [0] * len(class0_texts)

# X_train_new = class0_texts + class1_chunks
# y_train_new = class0_labels + class1_labels

# for label in [0,1]:
#     texts = [x for x,l in zip(X_train_new, y_train_new) if l==label]
#     lengths = [len(t.split()) for t in texts]
#     print(f"New class {label} avg length: {np.mean(lengths):.2f} words")

# Convert back to lists
# X_train = list(X_train_new)
# y_train = list(y_train_new)
X_train = list(X_train)
y_train = list(y_train)
X_test  = list(X_test)
y_test  = list(y_test)

for label in [0,1]:
    texts = [x for x,l in zip(X_train, y_train) if l==label]
    lengths = [len(t.split()) for t in texts]
    print(f"Class {label} avg length: {np.mean(lengths):.2f} words")

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# print("\nSample training texts:\n")
# for label in [0, 1]:
#     texts = [x for x, l in zip(X_train, y_train) if l == label]
#     print(f"Class {label} ({'non-native' if label==0 else 'native'}):")
#     for ex in random.sample(texts, min(1, len(texts))):  # up to 3 examples
#         # print("  ", ex[:200].replace("\n", " "), "...")  # show first 200 chars
#         print(ex)
#         print('---------------------------------------------------------------')
#     print()

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),  # unigrams + bigrams
    max_features=10000  # limit size for speed
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, solver="liblinear", class_weight='balanced')
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["non-native", "native"]))

from sklearn.svm import LinearSVC

svm_clf = LinearSVC(max_iter=2000, class_weight='balanced')
svm_clf.fit(X_train_vec, y_train)
y_pred_svm = svm_clf.predict(X_test_vec)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nDetailed report:")
print(classification_report(y_test, y_pred_svm, target_names=["non-native", "native"]))

cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["non-native", "native"],
            yticklabels=["non-native", "native"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("SVM Confusion Matrix")
plt.show()

# cell_raw = load_pickle("datasets/cell.pkl")
# pelic_raw = load_pickle("datasets/pelic.pkl")
# bawe_raw = load_pickle("datasets/bawe.pkl")

# cell = [d["text"] for d in cell_raw]
# cell = [preprocess_text_safe(t) for t in cell]
# cell_label = [d["label"] for d in cell_raw]

# pelic = [d["text"] for d in pelic_raw]
# pelic  = [preprocess_text_safe(t) for t in pelic]
# pelic_label = [d["label"] for d in pelic_raw]

# bawe = [d["text"] for d in bawe_raw]
# bawe  = [preprocess_text_safe(t) for t in bawe]
# bawe_label = [d["label"] for d in bawe_raw]

# filtered_cell = [(x,y) for x,y in zip(cell, cell_label) if x is not None]
# cell, cell_label = zip(*filtered_cell)

# filtered_pelic = [(x,y) for x,y in zip(pelic, pelic_label) if x is not None]
# pelic, pelic_label = zip(*filtered_pelic)

# filtered_bawe = [(x,y) for x,y in zip(bawe, bawe_label) if x is not None]
# bawe, bawe_label = zip(*filtered_bawe)

# cell_vec = vectorizer.fit_transform(cell)
# pelic_vec = vectorizer.fit_transform(pelic)
# bawe_vec = vectorizer.fit_transform(bawe)

# cell_pred = svm_clf.predict(cell_vec)
# print("CELL Accuracy:", accuracy_score(cell_label, cell_pred))

# pelic_pred = svm_clf.predict(pelic_vec)
# print("PELIC Accuracy:", accuracy_score(pelic_label, pelic_pred))

# bawe_pred = svm_clf.predict(bawe_vec)
# print("BAWE Accuracy:", accuracy_score(bawe_label, bawe_pred))