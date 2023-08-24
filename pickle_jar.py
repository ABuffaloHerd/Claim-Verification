"""
Aidan Chan 20114106
COMP702 RND - Claim Verification

This file takes in a dataset and pickles it into a jar. Just kidding.
This file takes in a dataset, tokenizes it for BERT and pickles it into a jar (pkl file).
"""

import pickle
import pandas as pd
import transformers
import torch
import os

from transformers import BertTokenizer, BertModel, BertForSequenceClassification

"""
Headers in the dataset:
Claim_id, Claim, Label (TFN) Evidence_id, Evidence, Article_id, Reason
"""

frames = []

# Load the dataset
for filename in os.listdir("data"):
    if filename.endswith(".xlsx"):
        df = pd.read_excel("data/" + filename, header=0)
        print("Loaded dataset: " + filename)
        frames.append(df)

# Compile the dataset into three arrays (claim, evidence, label)
claim = []
evidence = []
label = []

for frame in frames:
    claim.extend(frame['Claim'].tolist())
    evidence.extend(frame['Evidence'].tolist())
    label.extend(frame['Label (TFN)'].tolist())

# Convert the label to a binary value
label_map = {'T': 1, 'F': 0, 'N': 2}
for i in range(len(label)):
    label[i] = str(label_map[label[i]])

# Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the claim and evidence
print("Tokenizing the dataset...")

# For some reason, one of these isn't a string. Convert them all to strings.
claim = [str(c) for c in claim]
evidence = [str(e) for e in evidence]

# Combine the claim and evidence into one string, tokenize it with its label
combined = [" [SEP] ".join([c, e]) for c, e in zip(claim, evidence)]

encodings = tokenizer(combined, label, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Save the tokenized dataset to a pickle file
with open('data.pkl', 'wb') as f:
    pickle.dump(encodings, f)