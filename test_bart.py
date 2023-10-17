import pickle
import torch
import transformers
import os
import pandas as pd
import json

"""
Tests BART with user input

This file works with BART only. For BERT, use test_bert.py

Aidan Chan 20114106
"""

from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

# Load the model
path = "./model"
model = BartForConditionalGeneration.from_pretrained(path)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# EVALUATION mode
model.eval()

# Define inputs
claim = "FDA staff flags uncertainties on Veru's COVID drug"
evidence = "Veru's shares were up at $15.7 in afternoon trading. They fell nearly 9%% since mid-September, when the FDA rescheduled the advisory committee meeting."
label = 1

# Encode the inputs
text = claim + " [SEP] " + evidence
encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Run the model to generate output
output_ids = model.generate(encoding['input_ids'], attention_mask=encoding['attention_mask'], max_length=50, num_beams=5, early_stopping=True)

# Decode the generated output
decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the results
print("Claim: " + claim)
print("Evidence: " + evidence)
print("Sanity check label: " + {1: 'T', 0: 'F', 2: 'N'}[label])  # Convert label number back to T/F/N
print("Predicted: " + decoded_output)