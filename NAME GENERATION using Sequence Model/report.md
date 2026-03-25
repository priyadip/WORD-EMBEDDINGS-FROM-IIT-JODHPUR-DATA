# Assignment 2 – Character-Level Name Generation Using Recurrent Neural Architectures

---

## TASK-0: Dataset

### Dataset Construction

A dataset of **1,000 Indian names** was generated using an LLM and stored in `TrainingNames.txt`. Each entry is a full name (first name + surname) representative of Indian naming conventions, covering a diverse range of regional origins (Bengali, South Indian, North Indian, etc.).

**Sample entries:**
```
Aahana Chanda
Aaradhya Haldar
Aarohi Mitra
Aasha Paul
Abha Saha
Aparajita Mandal
```

**Vocabulary Statistics:**

| Property | Value |
|---|---|
| Total training names | 1,000 |
| Vocabulary size | 30 characters |
| Characters | space + a–z (lowercase) |
| Maximum sequence length | 29 characters |
| Average name length | 13.9 characters |

---

## TASK-1: Model Implementation

All three models are implemented **from scratch using NumPy** (no deep learning frameworks). Each model uses a shared learned character embedding matrix instead of one-hot vectors.

---

### 1. Vanilla Recurrent Neural Network (RNN)

**Architecture:**

```
Input(char_idx) → Embedding(30 → 32) → RNN Layer 1(128, tanh)
                → RNN Layer 2(128, tanh) → RNN Layer 3(128, tanh)
                → Linear(128 → 30) → Softmax
```

- 3 stacked RNN layers with tanh activation
- At each timestep t, hidden state update:
  `h_t = tanh(x_t · W_xh + h_{t-1} · W_hh + b_h)`
- Output: `y_t = softmax(h_t · W_hy + b_y)`
- Character generation is autoregressive (feeds previous predicted char as next input)

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Hidden size | 128 |
| Embedding dim | 32 |
| Layers | 3 |
| Learning rate | 0.005 |
| Optimizer | Adam |

**Trainable Parameters: 91,230**

---

### 2. Bidirectional LSTM (BLSTM)

**Architecture:**

```
Input(char_idx) → Embedding(30 → 32)
               → Forward LSTM x3 (hidden=64) ──┐
               → Backward LSTM x3 (hidden=64) ──┤→ Concat(128) → Linear(128 → 30) → Softmax

Generation: Forward-only LSTM x3 (hidden=64) → Linear(64 → 30) → Softmax
```

- 3 stacked forward + 3 stacked backward LSTM layers trained jointly
- Standard LSTM gate equations at each layer:
  - `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`  (forget gate)
  - `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`  (input gate)
  - `c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)` (candidate cell)
  - `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`      (cell state)
  - `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`  (output gate)
  - `h_t = o_t ⊙ tanh(c_t)`
- Top-layer hidden states from both directions are concatenated for training
- A separate 3-layer **forward-only** LSTM (sharing the embedding) is used at inference (generation)

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Hidden size (per direction) | 64 (128 combined) |
| Embedding dim | 32 |
| Layers | 3 (fwd) + 3 (bwd) + 3 (gen) |
| Learning rate | 0.005 |
| Optimizer | Adam |

**Trainable Parameters: 279,420**

---

### 3. RNN with Basic Attention Mechanism

**Architecture:**

```
Input(char_idx) → Embedding(30 → 32) → RNN x3(128, tanh)
               → Bahdanau Self-Attention → Context vector
               → Linear(128 → 30) → Softmax
```

- 3 stacked RNN layers (same as Vanilla RNN)
- After the RNN layers, a **Bahdanau-style additive attention** computes a context vector over all hidden states:
  - Score: `e_t = v^T · tanh(W_a · h_t + U_a · h_query)`
  - Weights: `α_t = softmax(e_t)`
  - Context: `c = Σ α_t · h_t`
- The context vector is used for output projection

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Hidden size | 128 |
| Embedding dim | 32 |
| Attention dim | 64 |
| Layers | 3 |
| Learning rate | 0.005 |
| Optimizer | Adam |

**Trainable Parameters: 127,966**

---

### Model Size Summary

| Model | Parameters |
|---|---|
| Vanilla RNN | 91,230 |
| BLSTM | 279,420 |
| RNN + Attention | 127,966 |

---

## TASK-2: Quantitative Evaluation

### Training Details

All models trained with:
- **Early stopping** (patience-based)
- **ReduceLROnPlateau** (factor=0.3, patience=3, floor=1e-5)
- 500 samples per epoch, max 200 epochs
- 200 names generated per model for evaluation

### Loss Results

| Model | Epochs Run | Best Train Loss | Final Train Loss | Best Val Loss | Final Val Loss |
|---|---|---|---|---|---|
| Vanilla RNN | 90 | 0.8719 | 0.8871 | 1.5979 | 1.6414 |
| BLSTM | 67 | 0.5775 | 0.5992 | 1.5366 | 1.5787 |
| RNN + Attention | 62 | 1.0249 | 1.0483 | 1.6012 | 1.6703 |

### Novelty Rate and Diversity

> **Novelty Rate** = % of generated names not present in training set
> **Diversity** = unique generated names / total generated names

| Model | Generated Names | Novelty Rate | Diversity |
|---|---|---|---|
| Vanilla RNN | 200 | **98.0%** | **100.0%** |
| BLSTM | 200 | **100.0%** | **100.0%** |
| RNN + Attention | 200 | **99.0%** | **100.0%** |

**Key observations:**
- All three models achieve **100% diversity** — no repeated names generated.
- **BLSTM** achieves the highest novelty (100%), meaning it never simply memorised training names.
- Vanilla RNN had the lowest novelty (98%), suggesting slight memorisation at 90 epochs.
- BLSTM achieves the **lowest validation loss (1.5366)**, indicating the best fit to the character distribution.

---

## TASK-3: Qualitative Analysis

### Representative Generated Samples

**Vanilla RNN (91,230 params):**
```
1. sharmina goswami      11. rinati saha
2. ashi saha             12. priya biswas
3. subha das             13. anita roy
4. sanjib mondal         14. sumon chakraborty
5. puja ghosh            15. rima sen
```

**BLSTM (279,420 params):**
```
1. siddhi mitra          11. ratani sanyal
2. priti das             12. supriya bose
3. ananya roy            13. nandita ghosh
4. krishna dutta         14. tapas biswas
5. rima chatterjee       15. dipali mondal
```

**RNN + Attention (127,966 params):**
```
1. krushna biswas        11. karri datta
2. agita biswas          12. priya roy
3. supriya ghosh         13. anita das
4. tapas mondal          14. rima paul
5. nandini sen           15. sanjay mitra
```

---

### Realism of Generated Names

All three models produce names that are **linguistically plausible** Indian names:
- Correct first name + surname structure (two words separated by space)
- Names reflect real Indian regional patterns (Bengali surnames: Chatterjee, Biswas, Ghosh, Mondal; South Indian: Krishnan, Datta)
- Phonotactics consistent with Indian naming conventions (consonant clusters, vowel patterns)
- **BLSTM** generates the most varied and realistic names due to its ability to model long-range dependencies in both directions during training
- **Vanilla RNN** sometimes produces names closer to training examples (lower novelty), but still realistic
- **RNN+Attention** produces fluent names; attention helps maintain coherence across longer sequences

---

### Failure Mode Analysis

| Failure Mode | Vanilla RNN | BLSTM | RNN+Attention |
|---|---|---|---|
| Too short (< 3 chars) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Too long (> 25 chars) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| No space (single word) | 1 (0.5%) | 0 (0.0%) | 0 (0.0%) |
| Repeated chars (3+) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Average length** | **13.6 chars** | **13.7 chars** | **13.6 chars** |
| **Training avg length** | **13.9 chars** | **13.9 chars** | **13.9 chars** |

**Discussion of failure modes:**

- **Vanilla RNN** occasionally generates single-word names (no space), indicating the model sometimes fails to produce the first/last name boundary. This is attributable to the vanishing gradient problem — the plain RNN struggles to maintain the "space character" signal over longer sequences.
- **BLSTM** had zero failures across all categories. The gating mechanism (forget/input/output gates) effectively retains structural information, ensuring the space delimiter is always generated.
- **RNN+Attention** also had zero failures. The attention context helps the model remain aware of earlier characters when predicting later ones, preventing structural errors.
- All models generate names close to the training average length (13.6–13.7 vs 13.9), confirming they learned the length distribution well.
- Occasional phonetically awkward names appear (e.g., "kipas", "agita") — these are minor artefacts of character-level generation without word-level constraints.

---

## Summary Comparison

| Metric | Vanilla RNN | BLSTM | RNN+Attention |
|---|---|---|---|
| Parameters | 91,230 | 279,420 | 127,966 |
| Best Val Loss | 1.5979 | **1.5366** | 1.6012 |
| Novelty Rate | 98.0% | **100.0%** | 99.0% |
| Diversity | 100.0% | 100.0% | 100.0% |
| Failure Rate | 0.5% | **0.0%** | **0.0%** |
| Avg Name Length | 13.6 | 13.7 | 13.6 |

**Conclusion:**
The **BLSTM** is the best-performing model across all metrics — lowest validation loss, highest novelty, and zero failure modes — at the cost of being the largest model (~3× more parameters than Vanilla RNN). The **RNN+Attention** offers a strong middle ground: better than Vanilla RNN in novelty and failures, with only 40% more parameters. The **Vanilla RNN**, while smallest and simplest, still performs competitively and is the fastest to train.

---

## Deliverables

| Deliverable | File |
|---|---|
| Source code (all models) | `char_rnn_names.py` |
| Training dataset | `TrainingNames.txt` |
| Generated name samples | `results/results.json` |
| Evaluation metrics | `results/results.json` |
| Loss curves | `results/loss_VanillaRNN.png`, `loss_BLSTM.png`, `loss_RNNWithAttn.png` |
| Comparative plots | `results/train_loss_comparison.png`, `val_loss_comparison.png` |
| Novelty/Diversity plot | `results/novelty_diversity.png` |
| Model size comparison | `results/model_size.png` |
| Failure mode analysis | `results/failure_modes.png` |
| Length distribution | `results/length_distribution.png` |
