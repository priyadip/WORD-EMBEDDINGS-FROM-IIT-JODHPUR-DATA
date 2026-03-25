# Character-Level Name Generation using Sequence Models

Implementation and comparison of three recurrent neural architectures for Indian name generation, built from scratch using NumPy.

---

## File Structure

```
NAME GENERATION using Sequence Model
/
├── char_rnn_names.py        # Main script — all models, training, evaluation, plots
├── TrainingName.txt         # 1000 Indian names (training dataset)
├── best_configs.json        # Best hyperparameters found by search
├── terminal.log             # Training output log
│
├── models/                  # Saved model weights (generated after training)
│   ├── vanilla_rnn.pkl
│   ├── blstm.pkl
│   └── attn_rnn.pkl
│
└── results/                 # Plots and metrics (generated after training)
    ├── results.json
    ├── model_size.png
    ├── novelty_diversity.png
    ├── failure_modes.png
    ├── length_distribution.png
    ├── loss_VanillaRNN.png
    ├── loss_BLSTM.png
    ├── loss_RNNWithAttn.png
    ├── train_loss_comparison.png
    └── val_loss_comparison.png
```

---

## Requirements

```bash
pip install numpy matplotlib
```

No deep learning framework required — all models are implemented using NumPy only.

---

## How to Run

### 1. Train all models + full evaluation

```bash
python char_rnn_names.py
```

This will:
- Load `TrainingName.txt`
- Train Vanilla RNN, BLSTM, and RNN+Attention
- Save each model to `models/` after its training completes
- Generate 200 names per model
- Compute novelty rate and diversity
- Run failure mode analysis
- Save all plots to `results/`
- Save all metrics to `results/results.json`

---


Generate Name:

```python
import pickle

with open("models/blstm.pkl", "rb") as f:
    model = pickle.load(f)

# Generate a name
print(model.generate())
```

---

## Models

| Model | Hidden Size | Parameters |
|---|---|---|
| Vanilla RNN | 128 | 91,230 |
| BLSTM | 64 per direction | 279,420 |
| RNN + Attention | 128 | 127,966 |

---

## Results

| Model | Val Loss | Novelty | Diversity |
|---|---|---|---|
| Vanilla RNN | 1.5979 | 98% | 100% |
| BLSTM | 1.5366 | 100% | 100% |
| RNN + Attention | 1.6012 | 99% | 100% |
