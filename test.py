import pickle
import torch
import transformers
import os
import pandas as pd
import json

"""
Tests the model on a given dataset

Aidan Chan 20114106
"""

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

# Load the model
path = "./model"
model = BertForSequenceClassification.from_pretrained(path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# EVALUATION mode
model.eval()

# Define inputs
claim = "Ford is a good company"
evidence = "Ford uses child labour"
label = 0

# Encode the inputs
text = claim + " [SEP] " + evidence
encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Run the model
outputs = model(encoding['input_ids'], encoding['attention_mask'])

# Get the predicted label
_, predicted = torch.max(outputs[0], 1)

# Print the results
print("Claim: " + claim)
print("Evidence: " + evidence)
print("Label: " + str(label))

print("Predicted: " + str(predicted.item()))

