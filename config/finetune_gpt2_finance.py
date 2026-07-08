# Fine-tuning depuis GPT-2 pré-entraîné (architecture originale, obligatoire pour
# charger les poids OpenAI) sur un corpus finance — voir data/finance/README.md
import time
out_dir = 'out-finance'
eval_interval = 50
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

dataset = 'finance'
init_from = 'gpt2'       # force automatiquement arch='gpt2'

batch_size = 2
gradient_accumulation_steps = 16
block_size = 512

max_iters = 300
learning_rate = 3e-5
decay_lr = False
dropout = 0.1
compile = False
