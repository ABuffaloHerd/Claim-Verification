"""
Aidan Chan 20114106
COMP702 RND - Claim Verification

This script trains the BART and saves it.
"""

import pickle
import torch
import transformers
import os
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

device = torch.device("cuda:0")


# Initialize the model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)

class FactCheckingDataset(Dataset):

    def __init__(self, tokenizer, claims, evidences, labels, reasons, max_length=512):
        self.tokenizer = tokenizer
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.max_length = max_length
        self.reasons = reasons

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        input_text = self.claims[idx] + " [SEP] " + self.evidences[idx]

        # Combine label and reason
        label_str = {1: 'T', 0: 'F', 2: 'N'}[self.labels[idx]]  # Convert label number back to T/F/N
        target_text = label_str + " [SEP] " + self.reasons[idx]

        input_encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        target_encoding = self.tokenizer(target_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# Load the dataset
with open("claims_1000_v2_800.xlsx",'rb') as f:
    df = pd.read_excel(f)

# Compile the dataset into four arrays (claim, evidence, label, reasons)
claim = df['Claim'].tolist()
evidence = df['Evidence'].tolist()

label_map = {'T': 1, 'F': 0, 'N': 2}
label = [label_map[l] for l in df['Label (TFN)'].tolist()]
reasons = df['Reason'].tolist()

# Force the thing into strings not float
claim = [str(x) for x in claim]
evidence = [str(x) for x in evidence]
reasons = [str(x) for x in reasons]

# Test
print(df.head(10))

# Set up training args
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    save_total_limit=3,
    output_dir='./results',
)

# Split the data into training and evaluation sets (e.g., 80% train, 20% evaluation)
claim_train, claim_eval, evidence_train, evidence_eval, label_train, label_eval, reason_train, reason_eval = train_test_split(
    claim, evidence, label, reasons, test_size=0.20, random_state=42
)

# Create the datasets
train_dataset = FactCheckingDataset(tokenizer, claim_train, evidence_train, label_train, reason_train)
eval_dataset = FactCheckingDataset(tokenizer, claim_eval, evidence_eval, label_eval, reason_eval)

# SEND IT
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Results
with open("results.txt", "w") as f:
    f.write(str(trainer.evaluate()))

# Save the model
model.save_pretrained("./model")