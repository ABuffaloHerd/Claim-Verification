"""
Aidan Chan 20114106
COMP702 RND - Claim Verification

This script trains the model and saves it.
"""

import pickle
import torch
import transformers
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Define the dataset class???
class FactCheckingDataset(Dataset):
    def __init__(self, tokenizer, claims, evidences, labels, max_length=512):
        self.tokenizer = tokenizer
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        text = self.claims[idx] + " [SEP] " + self.evidences[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': self.labels[idx]
        }

# Load the dataset
frames = []
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
    label[i] = label_map[label[i]]

# Force claim and evidence to be strings
claim = [str(x) for x in claim]
evidence = [str(x) for x in evidence]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Split the data into training and evaluation sets (80% train, 20% evaluation)
claim_train, claim_eval, evidence_train, evidence_eval, label_train, label_eval = train_test_split(
    claim, evidence, label, test_size=0.20, random_state=42
)

dataset = FactCheckingDataset(tokenizer, claim, evidence, label)
eval_set = FactCheckingDataset(tokenizer, claim_eval, evidence_eval, label_eval)

# Initialize the model
# Use a sequence classification model for fact checking
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training args
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=eval_set
)

# GPU is automatically detected and used if available
trainer.train()

results = trainer.evaluate()

# Pickle the results in case it's not serializable
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(results)

try:
    with open('results.txt', 'w') as f:
        f.write(str(results))
except:
    print("Failed to write results to file")

try:
    trainer.save_model("./model")
    dataset.tokenizer.save_pretrained("./tokenizer")
except:
    print("Failed to save model and tokenizer")