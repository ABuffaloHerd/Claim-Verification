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

class ConfusionMatrix:
    def __init__(self):
        self.classes = [1, 0, 2]  # [True, False, Neutral]
        self.confusion_matrix = np.zeros((3, 3))
        
    def update(self, y_true, y_pred):
        # Convert string labels to numbers
        y_true = { 'T': 1, 'F': 0, 'N': 2 }[y_true]
        y_pred = { 'T': 1, 'F': 0, 'N': 2 }[y_pred]

        idx_true = self.classes.index(y_true)
        idx_pred = self.classes.index(y_pred)
        self.confusion_matrix[idx_true][idx_pred] += 1
    
    def precision(self, label):
        idx = self.classes.index(label)
        column_sum = np.sum(self.confusion_matrix[:, idx])
        if column_sum == 0:
            return 0
        return self.confusion_matrix[idx, idx] / column_sum
    
    def recall(self, label):
        idx = self.classes.index(label)
        row_sum = np.sum(self.confusion_matrix[idx, :])
        if row_sum == 0:
            return 0
        return self.confusion_matrix[idx, idx] / row_sum
    
    def f1_score(self, label):
        p = self.precision(label)
        r = self.recall(label)
        if p + r == 0:
            return 0
        return 2 * (p * r) / (p + r)
    
    def micro_precision(self):
        total_tp = np.trace(self.confusion_matrix)
        total_fp = np.sum(np.sum(self.confusion_matrix, axis=0) - np.diagonal(self.confusion_matrix))
        return total_tp / (total_tp + total_fp)

    def micro_recall(self):
        total_tp = np.trace(self.confusion_matrix)
        total_fn = np.sum(np.sum(self.confusion_matrix, axis=1) - np.diagonal(self.confusion_matrix))
        return total_tp / (total_tp + total_fn)

    def micro_f1_score(self):
        micro_prec = self.micro_precision()
        micro_rec = self.micro_recall()
        if micro_prec + micro_rec == 0:  # Avoid dividing by zero
            return 0
        return 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec)

    def summary(self):
        metrics = {}
        for label in self.classes:
            label_str = str(label)
            metrics[f"precision_{label_str}"] = self.precision(label)
            metrics[f"recall_{label_str}"] = self.recall(label)
            metrics[f"f1_{label_str}"] = self.f1_score(label)

        return metrics
    
    def table(self):
        # Print the column header
        print("Predicted -->".rjust(12), end='')
        for cls in self.classes:
            print(f'{cls}'.center(8), end='')
        print("\n" + "-"*38)
        
        # Print the rows
        for idx, cls in enumerate(self.classes):
            print(f"Actual {cls} |".rjust(12), end='')
            for value in self.confusion_matrix[idx]:
                print(f'{int(value)}'.center(8), end='')
            print()


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