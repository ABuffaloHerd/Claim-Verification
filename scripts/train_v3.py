"""
Aidan Chan 20114106
COMP702 RND - Claim Verification

This script trains the BART and saves it.
Version 3 uses the complete 1000 data point set
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, classification_report, precision_recall_fscore_support

MODE = "train"  # train or eval

if MODE == "train":
    device = torch.device("cuda:0")
else: # eval
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256" # Looks like not even this is enough
    device = torch.device("cpu")


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

# use only 100 examples for evaluation as part of even more desperate optimizations
# Since training takes even more memory let's cut that down to 25 in training mode
eval_sample_size = 25 if MODE == "train" else 100

# Create the datasets
train_dataset = FactCheckingDataset(tokenizer, claim_train, evidence_train, label_train, reason_train)
eval_dataset_full = FactCheckingDataset(tokenizer, claim_eval, evidence_eval, label_eval, reason_eval)

# Take a subset of the evaluation dataset
eval_dataset = torch.utils.data.Subset(eval_dataset_full, indices=range(eval_sample_size))

# Define metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = logits[0]
    # Sanity check: This tells us that the predictions are in the correct format
    print(predictions.shape)

    predictions = np.argmax(logits, axis=1)

    # Print the classification report to the console
    print(classification_report(labels, predictions, target_names=['F', 'T', 'N']))
    
    # Compute metrics for logging and return
    report = classification_report(labels, predictions, output_dict=True, labels=[0,1,2], target_names=['F', 'T', 'N'])

    with open("report.txt", "w") as f:
        f.write(json.dumps(report))

    # Extract the required metrics
    precision = report['micro avg']['precision']
    recall = report['micro avg']['recall']
    f1 = report['micro avg']['f1-score']

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# SEND IT
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics # so, as i've learned, this is not the correct way to do it. This is meant to be used in classification tasks, not seq2seq
)

if MODE == "train":
    # Train the model
    print("Training mode")
    trainer.train()

    # Save the model first (just in case)
    model.save_pretrained("./model")


else:
    # Evaluate the model straight away
    # Load the model from disk
    print("Evaluation mode")

    # More desperate optimizations
    torch.no_grad()

    device_cpu = torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("./model").to(device_cpu) # As a last resort, use CPU

# Clear cuda cache
torch.cuda.empty_cache()

# Evaluate the model
try:
    eval = trainer.evaluate()
except ValueError:
    # do nothing
    print("ValueError most likey to do with inhomogenous shape of the arrays. Ignoring... ")
    

# Results
with open("results.txt", "w") as f:
    f.write(str(eval))