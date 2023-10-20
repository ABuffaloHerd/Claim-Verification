# Claim Verification Phase 2

This repo contains the code used to fine tune BERT and BART for the purposes of text classification and seq2seq respectively.

## Usage

Clone the repo and then `pip install -r requirements.txt`
When it fails to install torch, modify the requirements file to remove +cu117 or install torch manually.

You'll only be interested in `train_v4.py`
Make sure CUDA is installed otherwise modify the device so that it uses your CPU.

Set the `DATASET` parameter to the path to your dataset.xlsx file. It must have the following columns:
- `Claim`: The claim text
- `Label (TFN)`: The TFN label of the claim
- `Evidence`: The evidence text
- `Reason` : The reason for the TFN label

Run `python train_v4.py` to start training. The model will be saved in the `models` folder. It should also save after every epoch. The script also automatically splits the dataset into train, validation and test sets. The split is 80/20. Good luck!