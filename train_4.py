"""
Aidan Chan 20114106
COMP702 RND - Claim Verification

This script trains the BART and saves it.
Version 4 uses custom metrics to evaluate the model immediately after training.
"""

import pickle
import torch
import transformers
import os
import pandas as pd
import json

import numpy as np

from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load confusion matrix class
from cm import ConfusionMatrix

# Set device to GPU
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
with open("claims_1000_v3.xlsx",'rb') as f:
    df = pd.read_excel(f)

# Compile the dataset into four arrays (claim, evidence, label, reasons)
claim = df['Claim'].tolist()
evidence = df['Evidence'].tolist()

label_map = {'T': 1, 'F': 0, 'N': 2}
label = [label_map[l] for l in df['Label (TFN)'].tolist()]
reasons = df['Reason'].tolist()

# Force the thing to strings not float
claim = [str(x) for x in claim]
evidence = [str(x) for x in evidence]
reasons = [str(x) for x in reasons]

# Test
print(df.head(10))

# Set up training args
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1, # 8 is too much for my measly 12GB VRAM apparently. Offloading it to the CPU should help
    logging_dir='./logs',
    logging_steps=100,
    save_steps=100,
    eval_steps=10,
    save_total_limit=3,
    evaluation_strategy="steps",
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
)

# Train the model
print("Training mode")
trainer.train()

# Save the model first (just in case)
model.save_pretrained("./model")

# Evaluate the model
print("Evaluation mode")

# Create the confusion matrix
cm = ConfusionMatrix()

# Set model to evaluation mode
model.eval()

# Use the eval versions of the datasets and encode the inputs
trainlen = len(claim_eval)

for i in range(trainlen):
    # Encode the inputs
    text = claim_eval[i] + " [SEP] " + evidence_eval[i]
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    # Run the model to generate output
    output_ids = model.generate(encoding['input_ids'].to(device), attention_mask=encoding['attention_mask'].to(device), max_length=50, num_beams=5, early_stopping=True)

    # Decode
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Take the first character, (T/F/N) and compare it to the ground truth, update the confusion matrix
    cm.update(label_eval[i], decoded_output[0])

    # Sanity check output
    print(f"Pass {i} of {trainlen} complete.")

    # Log event to file for later analysis
    with open("log.txt", "a") as f:
        f.write("-"*20)
        f.write(f"Pass {i} of {trainlen}")
        f.write("Claim " + claim_eval[i])
        f.write("Evidence " + evidence_eval[i])
        f.write("Ground truth " + {1: 'T', 0: 'F', 2: 'N'}[label_eval[i]])
        f.write("Predicted " + decoded_output)

# Print the confusion matrix
cm.table()
print(cm.summary())

# Dump the confusion matrix to a file
with open("confusion_matrix.txt", "w") as f:
    f.write(cm.summary() + '\n')
    f.write(cm.table())