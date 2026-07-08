# Entraînement from scratch de l'architecture MODERNE sur shakespeare_char
# Sur GPU : ~3 min. Sur CPU : ajouter --device=cpu --compile=False et réduire max_iters.
out_dir = 'out-shakespeare-char-modern'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'modern-gpt'

dataset = 'shakespeare_char'
arch = 'modern'          # RMSNorm + RoPE + SwiGLU + QK-norm
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_kv_head = 2            # GQA : 2 têtes kv pour 6 têtes q (cache 3x plus petit)
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
