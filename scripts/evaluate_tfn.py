"""
Aidan Chan 20114106
COMP702 RND - Claim Verification

This script evaluates BART's classification performance via a confusion matrix.
"""

"""
This is a three way confusion matrix, where the rows are the actual labels and the columns are the predicted labels.
The three labels are T, F, and N, which stand for True, False, and Neutral respectively.

This will be used to create the confusion matrix class.

           | Predicted T | Predicted F | Predicted N |
-----------------------------------------------------|
Actual T   |    TT       |    TF       |    TN       |
-----------------------------------------------------|
Actual F   |    FT       |    FF       |    FN       |
-----------------------------------------------------|
Actual N   |    NT       |    NF       |    NN       |
-----------------------------------------------------|
"""

import numpy as np
import pandas as pd

from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

from cm import ConfusionMatrix


cm = ConfusionMatrix()

# Create the confusion matrix
cm = ConfusionMatrix()

# Load the model
path = "./model"
model = BartForConditionalGeneration.from_pretrained(path).eval()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Load the dataset
with open("claims_1000_v3.xlsx",'rb') as f:
    df = pd.read_excel(f)

# Compile the dataset into four arrays (claim, evidence, label, reasons)
claim = df['Claim'].tolist()
evidence = df['Evidence'].tolist()
label = df['Label (TFN)'].tolist()


# For each entry in x randomly selected from the dataset, run the model to generate output
for i in range(20):
    # Pick a random index
    idx = np.random.randint(0, len(claim))

    # Encode the inputs
    text = claim[idx] + " [SEP] " + evidence[idx]
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    # Run the model to generate output
    output_ids = model.generate(encoding['input_ids'], attention_mask=encoding['attention_mask'], max_length=50, num_beams=5, early_stopping=True)

    # Decode the generated output
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Take the first character, (T/F/N) and compare it to the ground truth
    cm.update(label[idx], decoded_output[0])

    print("Claim: " + claim[idx])
    print("Evidence: " + evidence[idx])
    print("ground truth sanity check label: " + label[idx])  # Convert label number back to T/F/N
    print("Predicted: " + decoded_output)

    print("Pass " + str(i) + " complete.")

# Print the confusion matrix
print(cm.summary())
cm.table()

print("\nOverall specs:")
print("Micro precision: " + str(cm.micro_precision()))
print("Micro recall: " + str(cm.micro_recall()))
print("Micro F1 score: " + str(cm.micro_f1_score()))