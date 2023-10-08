import pickle
import torch
import transformers
import os
import pandas as pd
import json

"""
Tests BART with user input

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
claim = "Ford is a good company"
evidence = "Ford uses child labour"
label = 0  # for BART, this might represent a sequence, like "F [SEP] Because of evidence mismatch."

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
print("Label: " + str(label))
print("Predicted: " + decoded_output)