"""Tokenize data/finance/input.txt en train.bin / val.bin (BPE GPT-2).
Placer d'abord votre corpus texte dans data/finance/input.txt"""
import os
import numpy as np
import tiktoken

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
assert os.path.exists(input_file_path), \
    "Placez votre corpus dans data/finance/input.txt (rapports annuels, circulaires BAM, etc.)"

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
enc = tiktoken.get_encoding("gpt2")
train_ids = np.array(enc.encode_ordinary(data[:int(n*0.9)]), dtype=np.uint16)
val_ids = np.array(enc.encode_ordinary(data[int(n*0.9):]), dtype=np.uint16)
print(f"train : {len(train_ids):,} tokens | val : {len(val_ids):,} tokens")
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
