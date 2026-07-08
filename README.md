# nanoGPT-plus 🚀

Fork modernisé de [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy) intégrant les améliorations d'architecture adoptées par Llama, Mistral et Qwen — tout en restant aussi simple et lisible que l'original.

## Améliorations intégrées

| # | Amélioration | Remplace | Bénéfice | Fichier |
|---|--------------|----------|----------|---------|
| 1 | **RMSNorm** | LayerNorm | Plus simple, plus rapide, aussi stable | `model.py` |
| 2 | **RoPE** (rotary embeddings) | Table de positions apprise (`wpe`) | Meilleure généralisation, zéro paramètre | `model.py` |
| 3 | **SwiGLU** | MLP GELU | Meilleure qualité à budget de paramètres égal | `model.py` |
| 4 | **GQA** (grouped-query attention) | Attention multi-têtes classique | KV cache jusqu'à n_head/n_kv_head fois plus petit | `model.py` |
| 5 | **QK-norm** | — | Stabilise l'entraînement (Qwen3, OLMo 2) | `model.py` |
| 6 | **KV cache** | Recalcul complet à chaque token | Génération O(1)/token au lieu de O(T) | `model.py` (`generate`) |
| 7 | **Double architecture** | — | `arch='modern'` (from scratch) ou `arch='gpt2'` (fine-tuning depuis les poids OpenAI) | `train.py`, `sample.py` |

Déjà présents dans nanoGPT et conservés : flash attention (SDPA), `torch.compile`, bf16/fp16 automatique, DDP multi-GPU, gradient accumulation, weight tying, schedule cosine avec warmup.

## Benchmark mesuré (mêmes données, mêmes batches, ~800K paramètres, 300 itérations)

| Itération | Baseline GPT-2 | Architecture moderne | Gain |
|-----------|---------------|---------------------|------|
| 100 | 2.546 | 2.348 | +0.20 |
| 200 | 2.452 | 2.143 | +0.31 |
| **300** | **2.395** | **2.054** | **+0.34** |

(val loss sur shakespeare_char ; reproduire avec `python bench_compare.py`)

## Installation

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Démarrage rapide

**Entraîner le modèle moderne from scratch** (GPU, ~3 min) :
```sh
python data/shakespeare_char/prepare.py
python train.py config/train_modern_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char-modern
```

**Fine-tuner GPT-2 pré-entraîné sur votre corpus** (ex. finance) :
```sh
# placer votre corpus dans data/finance/input.txt puis :
python data/finance/prepare.py
python train.py config/finetune_gpt2_finance.py
python sample.py --out_dir=out-finance --start="Le risque de crédit"
```

**Sur CPU / petite machine** : ajouter `--device=cpu --compile=False` et réduire les tailles.

## ⚠️ Limitation importante à comprendre

L'architecture moderne est **incompatible avec les poids GPT-2 d'OpenAI** (les matrices n'ont ni la même forme ni le même rôle). C'est le compromis fondamental :

- `arch='modern'` → meilleure architecture, mais entraînement **from scratch** uniquement
- `arch='gpt2'` (`model_gpt2.py`, l'original) → architecture 2019, mais accès aux poids **pré-entraînés** (`init_from='gpt2'`)

Le sélecteur est automatique : `init_from='gpt2*'` force l'original, et `sample.py`/`resume` lisent l'architecture stockée dans le checkpoint.

## Structure

```
model.py          architecture moderne (RMSNorm+RoPE+SwiGLU+GQA+QK-norm+KV cache)
model_gpt2.py     architecture GPT-2 originale de Karpathy (fine-tuning)
train.py          entraînement (les deux architectures, sélecteur --arch)
sample.py         génération (détection auto de l'architecture)
bench_compare.py  benchmark baseline vs moderne, conditions identiques
config/           configs prêtes à l'emploi
data/finance/     template pour votre corpus métier
```

## Pistes suivantes (roadmap)

- Optimiseur **Muon** (cf. [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), la référence des speedruns GPT-2)
- Gradient checkpointing pour entraîner plus gros à mémoire égale
- Sliding window attention (Mistral) pour les longs contextes
- Export / quantization pour l'inférence (GGUF, int8)

## Licence

MIT — basé sur nanoGPT © Andrej Karpathy.
