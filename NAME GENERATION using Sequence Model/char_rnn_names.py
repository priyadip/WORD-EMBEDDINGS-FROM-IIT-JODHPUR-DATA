"""
CHARACTER LEVEL NAME GENERATION USING RNN VARIANTS

Model  (Vanilla RNN, BLSTM, RNN+Attention)
Quantitative Evaluation (Novelty Rate, Diversity)
Qualitative Analysis (Realism, Failure Modes, Samples)

"""

import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt

# Fix random seeds for reproducibility
np.random.seed(42)
random.seed(42)

#LOAD DATASET

print("DATASET")


# Load training names
with open("TrainingName.txt", "r") as f:
    training_names = [line.strip().lower() for line in f if line.strip()]

print(f"Loaded {len(training_names)} training names.")
print(f"Sample names: {training_names[:5]}")

# CHARACTER VOCABULARY & ENCODING
# Build character vocabulary from training data
PAD = '\0'    # Padding
SOS = '\1'    # Start of sequence
EOS = '\2'    # End of sequence

all_chars = set()
for name in training_names:
    all_chars.update(name)
all_chars = sorted(list(all_chars))

vocab = [PAD, SOS, EOS] + all_chars
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {all_chars}")

# Encode names as sequences
def encode_name(name):
    """Encode a name as a list of integer indices: SOS + chars + EOS."""
    return [char_to_idx[SOS]] + [char_to_idx[ch] for ch in name] + [char_to_idx[EOS]]

def decode_indices(indices):
    """Decode integer indices back to a string."""
    chars = []
    for idx in indices:
        ch = idx_to_char[idx]
        if ch == EOS:
            break
        if ch not in (PAD, SOS):
            chars.append(ch)
    return ''.join(chars)

encoded_names = [encode_name(name) for name in training_names]
max_len = max(len(seq) for seq in encoded_names)
print(f"Max sequence length: {max_len}")

# Pad sequences
def pad_sequences(sequences, maxlen, pad_value=0):
    """Pad sequences to the same length."""
    result = np.full((len(sequences), maxlen), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        result[i, :len(seq)] = seq
    return result

padded = pad_sequences(encoded_names, max_len)

# NUMPY NEURAL NETWORK BUILDING BLOCKS (From Scratch)

def softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def tanh(x):
    return np.tanh(x)

def cross_entropy_loss(probs, targets):
    """Compute cross-entropy loss."""
    n = len(targets)
    loss = 0.0
    for t in range(n):
        loss -= np.log(probs[t][targets[t]] + 1e-12)
    return loss / n

def clip_gradients(grads, max_norm=5.0):
    """Clip gradients by global norm."""
    total_norm = 0.0
    for g in grads:
        total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in grads]
    return grads

def init_weight(rows, cols):
    """Xavier initialization."""
    return np.random.randn(rows, cols) * np.sqrt(2.0 / (rows + cols))

def one_hot(idx, size):
    """Create one-hot vector."""
    vec = np.zeros(size)
    vec[idx] = 1.0
    return vec

# MODEL 1: VANILLA RNN

class VanillaRNN:
    """
    Vanilla Recurrent Neural Network.

    Architecture:
    - Embedding: Learned dense lookup E (vocab_size -> embed_dim).
      e_t = E[char_idx]   (shape: 1 x embed_dim)
    - Hidden: 3 stacked RNN layers with tanh activation
      h1_t = tanh(W_xh1 * e_t  + W_hh1 * h1_{t-1} + b_h1)
      h2_t = tanh(W_xh2 * h1_t + W_hh2 * h2_{t-1} + b_h2)
      h3_t = tanh(W_xh3 * h2_t + W_hh3 * h3_{t-1} + b_h3)
    - Output: Linear layer + softmax
      y_t = softmax(W_hy * h3_t + b_y)

    Why embedding instead of one-hot:
      one_hot(idx) @ W  ==  W[idx]  (row lookup) — the matrix multiply is redundant.
      Using a separate E with embed_dim < vocab_size gives a compact learned
      representation where similar characters can cluster together.
    """

    def __init__(self, vocab_size, hidden_size=128, embed_dim=16, learning_rate=0.005):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.lr = learning_rate

        # Learned character embedding matrix: vocab_size rows, each embed_dim-wide
        # E[idx] gives the embedding for character idx  (no one-hot )
        self.E = init_weight(vocab_size, embed_dim)

        # Layer 1: embed_dim -> hidden
        self.W_xh1 = init_weight(embed_dim, hidden_size)
        self.W_hh1 = init_weight(hidden_size, hidden_size)
        self.b_h1  = np.zeros((1, hidden_size))
        # Layer 2: hidden -> hidden
        self.W_xh2 = init_weight(hidden_size, hidden_size)
        self.W_hh2 = init_weight(hidden_size, hidden_size)
        self.b_h2  = np.zeros((1, hidden_size))
        # Layer 3: hidden -> hidden
        self.W_xh3 = init_weight(hidden_size, hidden_size)
        self.W_hh3 = init_weight(hidden_size, hidden_size)
        self.b_h3  = np.zeros((1, hidden_size))
        # Output
        self.W_hy = init_weight(hidden_size, vocab_size)
        self.b_y  = np.zeros((1, vocab_size))

        self._init_adam()

    def _init_adam(self):
        self.m = {}
        self.v = {}
        self.t = 0
        for name in ['E',
                     'W_xh1', 'W_hh1', 'b_h1',
                     'W_xh2', 'W_hh2', 'b_h2',
                     'W_xh3', 'W_hh3', 'b_h3',
                     'W_hy', 'b_y']:
            self.m[name] = np.zeros_like(getattr(self, name))
            self.v[name] = np.zeros_like(getattr(self, name))

    def count_parameters(self):
        total = 0
        for name in ['E',
                     'W_xh1', 'W_hh1', 'b_h1',
                     'W_xh2', 'W_hh2', 'b_h2',
                     'W_xh3', 'W_hh3', 'b_h3',
                     'W_hy', 'b_y']:
            total += getattr(self, name).size
        return total

    def forward(self, inputs):
        """Forward pass. No one-hot: uses embedding lookup E[idx]."""
        T = len(inputs)
        h1 = np.zeros((1, self.hidden_size))
        h2 = np.zeros((1, self.hidden_size))
        h3 = np.zeros((1, self.hidden_size))
        hiddens = [(h1.copy(), h2.copy(), h3.copy())]
        probs = []

        for t in range(T):
            e  = self.E[inputs[t]].reshape(1, -1)          # embedding lookup
            h1 = tanh(e  @ self.W_xh1 + h1 @ self.W_hh1 + self.b_h1)
            h2 = tanh(h1 @ self.W_xh2 + h2 @ self.W_hh2 + self.b_h2)
            h3 = tanh(h2 @ self.W_xh3 + h3 @ self.W_hh3 + self.b_h3)
            hiddens.append((h1.copy(), h2.copy(), h3.copy()))
            p = softmax(h3 @ self.W_hy + self.b_y).flatten()
            probs.append(p)

        return probs, hiddens

    def compute_grad_and_loss(self, inputs, targets):
        """Forward + backward with embedding gradients. Returns (loss, grads_dict)."""
        T = len(inputs)
        embs = {}                               # cached embeddings per timestep
        hs1, hs2, hs3 = {}, {}, {}
        hs1[-1] = np.zeros((1, self.hidden_size))
        hs2[-1] = np.zeros((1, self.hidden_size))
        hs3[-1] = np.zeros((1, self.hidden_size))
        probs_list = []

        for t in range(T):
            embs[t] = self.E[inputs[t]].reshape(1, -1)     # embedding lookup, no one-hot
            hs1[t]  = tanh(embs[t] @ self.W_xh1 + hs1[t-1] @ self.W_hh1 + self.b_h1)
            hs2[t]  = tanh(hs1[t]  @ self.W_xh2 + hs2[t-1] @ self.W_hh2 + self.b_h2)
            hs3[t]  = tanh(hs2[t]  @ self.W_xh3 + hs3[t-1] @ self.W_hh3 + self.b_h3)
            probs_list.append(softmax(hs3[t] @ self.W_hy + self.b_y).flatten())

        loss = cross_entropy_loss(probs_list, targets)

        # Gradient buffers
        dE     = np.zeros_like(self.E)          # sparse: only touched rows get gradients
        dW_xh1 = np.zeros_like(self.W_xh1); dW_hh1 = np.zeros_like(self.W_hh1); db_h1 = np.zeros_like(self.b_h1)
        dW_xh2 = np.zeros_like(self.W_xh2); dW_hh2 = np.zeros_like(self.W_hh2); db_h2 = np.zeros_like(self.b_h2)
        dW_xh3 = np.zeros_like(self.W_xh3); dW_hh3 = np.zeros_like(self.W_hh3); db_h3 = np.zeros_like(self.b_h3)
        dW_hy  = np.zeros_like(self.W_hy);  db_y   = np.zeros_like(self.b_y)
        dh1_next = np.zeros((1, self.hidden_size))
        dh2_next = np.zeros((1, self.hidden_size))
        dh3_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(T)):
            dy = probs_list[t].reshape(1, -1).copy()
            dy[0, targets[t]] -= 1.0
            dW_hy += hs3[t].T @ dy;  db_y += dy

            # Layer 3
            dh3     = dy @ self.W_hy.T + dh3_next
            dh3_raw = dh3 * (1 - hs3[t] ** 2)
            dW_xh3 += hs2[t].T @ dh3_raw; dW_hh3 += hs3[t-1].T @ dh3_raw; db_h3 += dh3_raw
            dh3_next = dh3_raw @ self.W_hh3.T

            # Layer 2
            dh2     = dh3_raw @ self.W_xh3.T + dh2_next
            dh2_raw = dh2 * (1 - hs2[t] ** 2)
            dW_xh2 += hs1[t].T @ dh2_raw; dW_hh2 += hs2[t-1].T @ dh2_raw; db_h2 += dh2_raw
            dh2_next = dh2_raw @ self.W_hh2.T

            # Layer 1
            dh1     = dh2_raw @ self.W_xh2.T + dh1_next
            dh1_raw = dh1 * (1 - hs1[t] ** 2)
            dW_xh1 += embs[t].T @ dh1_raw;  dW_hh1 += hs1[t-1].T @ dh1_raw; db_h1 += dh1_raw
            dh1_next = dh1_raw @ self.W_hh1.T

            # Embedding gradient: only update the row for this character
            # de = dh1_raw @ W_xh1.T  (gradient w.r.t. embedding vector)
            dE[inputs[t]] += (dh1_raw @ self.W_xh1.T).flatten()

        _names = ['E','W_xh1','W_hh1','b_h1','W_xh2','W_hh2','b_h2','W_xh3','W_hh3','b_h3','W_hy','b_y']
        _raw   = [dE, dW_xh1, dW_hh1, db_h1, dW_xh2, dW_hh2, db_h2, dW_xh3, dW_hh3, db_h3, dW_hy, db_y]
        return loss, dict(zip(_names, _raw))

    def compute_loss(self, inputs, targets):
        """Forward-only pass for validation. Returns loss without gradients."""
        probs, _ = self.forward(inputs)
        return cross_entropy_loss(probs, targets)

    def _adam_step(self, grads):
        """Clip gradients and apply one Adam update step."""
        names = ['E','W_xh1','W_hh1','b_h1','W_xh2','W_hh2','b_h2','W_xh3','W_hh3','b_h3','W_hy','b_y']
        grad_list = clip_gradients([grads[n] for n in names])
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, grad in zip(names, grad_list):
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * grad ** 2
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, param)

    def train_step(self, inputs, targets):
        """Single-sample forward + backward + Adam update."""
        loss, grads = self.compute_grad_and_loss(inputs, targets)
        self._adam_step(grads)
        return loss

    def generate(self, max_len=50, temperature=0.8):
        """Generate a name using embedding lookup (no one-hot)."""
        h1 = np.zeros((1, self.hidden_size))
        h2 = np.zeros((1, self.hidden_size))
        h3 = np.zeros((1, self.hidden_size))
        idx = char_to_idx[SOS]
        name = []

        for _ in range(max_len):
            e  = self.E[idx].reshape(1, -1)             # embedding lookup
            h1 = tanh(e  @ self.W_xh1 + h1 @ self.W_hh1 + self.b_h1)
            h2 = tanh(h1 @ self.W_xh2 + h2 @ self.W_hh2 + self.b_h2)
            h3 = tanh(h2 @ self.W_xh3 + h3 @ self.W_hh3 + self.b_h3)
            y  = (h3 @ self.W_hy + self.b_y) / temperature
            p  = softmax(y).flatten()
            idx = np.random.choice(len(p), p=p)
            ch = idx_to_char[idx]
            if ch == EOS:
                break
            if ch not in (PAD, SOS):
                name.append(ch)

        return ''.join(name)


# MODEL 2: BIDIRECTIONAL LSTM (BLSTM)

class LSTMCell:
    """Single LSTM cell implementation."""

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        # Gates: forget, input, candidate, output
        self.W_f = init_weight(concat_size, hidden_size)
        self.b_f = np.ones((1, hidden_size))  # Initialize forget bias to 1
        self.W_i = init_weight(concat_size, hidden_size)
        self.b_i = np.zeros((1, hidden_size))
        self.W_c = init_weight(concat_size, hidden_size)
        self.b_c = np.zeros((1, hidden_size))
        self.W_o = init_weight(concat_size, hidden_size)
        self.b_o = np.zeros((1, hidden_size))

    def forward(self, x, h_prev, c_prev):
        """Forward pass of LSTM cell."""
        concat = np.concatenate([x, h_prev], axis=1)

        f = sigmoid(concat @ self.W_f + self.b_f)
        i = sigmoid(concat @ self.W_i + self.b_i)
        c_tilde = tanh(concat @ self.W_c + self.b_c)
        c = f * c_prev + i * c_tilde
        o = sigmoid(concat @ self.W_o + self.b_o)
        h = o * tanh(c)

        cache = (x, h_prev, c_prev, concat, f, i, c_tilde, c, o, h)
        return h, c, cache

    def backward(self, dh, dc_next, cache):
        """Backward pass of LSTM cell."""
        x, h_prev, c_prev, concat, f, i, c_tilde, c, o, h = cache

        do = dh * tanh(c)
        dc = dh * o * (1 - tanh(c) ** 2) + dc_next

        df = dc * c_prev
        di = dc * c_tilde
        dc_tilde = dc * i
        dc_prev = dc * f

        # Gate gradients
        df_raw = df * f * (1 - f)
        di_raw = di * i * (1 - i)
        dc_tilde_raw = dc_tilde * (1 - c_tilde ** 2)
        do_raw = do * o * (1 - o)

        # Parameter gradients
        dW_f = concat.T @ df_raw
        db_f = df_raw
        dW_i = concat.T @ di_raw
        db_i = di_raw
        dW_c = concat.T @ dc_tilde_raw
        db_c = dc_tilde_raw
        dW_o = concat.T @ do_raw
        db_o = do_raw

        # Input and hidden gradients
        d_concat = (df_raw @ self.W_f.T + di_raw @ self.W_i.T +
                    dc_tilde_raw @ self.W_c.T + do_raw @ self.W_o.T)
        dx = d_concat[:, :self.input_size]
        dh_prev = d_concat[:, self.input_size:]

        grads = {
            'W_f': dW_f, 'b_f': db_f, 'W_i': dW_i, 'b_i': db_i,
            'W_c': dW_c, 'b_c': db_c, 'W_o': dW_o, 'b_o': db_o
        }

        return dx, dh_prev, dc_prev, grads

    def count_parameters(self):
        total = 0
        for attr in ['W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o']:
            total += getattr(self, attr).size
        return total


class BLSTM:
    """
    Bidirectional LSTM for character-level name generation.

    Architecture:
    - Embedding: Learned dense lookup E (vocab_size -> embed_dim).
    - 3 stacked forward LSTM layers (left-to-right)
    - 3 stacked backward LSTM layers (right-to-left)
    - Top-layer hidden states concatenated from both directions
    - Output: Linear layer + softmax
      y_t = softmax(W_out * [h_fwd3_t; h_bwd3_t] + b_out)

    For generation, a 3-layer forward-only LSTM is used at inference time.
    All three stacks (fwd, bwd, gen) share the same embedding matrix E.
    """

    def __init__(self, vocab_size, hidden_size=64, embed_dim=16, learning_rate=0.005):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.lr = learning_rate
        self.num_layers = 3

        # Shared learned character embedding (fwd, bwd, gen all use E[idx])
        self.E = init_weight(vocab_size, embed_dim)

        # 3-layer forward, backward, and generator LSTM stacks
        self.fwd_cells = []
        self.bwd_cells = []
        self.gen_cells = []
        for l in range(self.num_layers):
            in_size = embed_dim if l == 0 else hidden_size
            self.fwd_cells.append(LSTMCell(in_size, hidden_size))
            self.bwd_cells.append(LSTMCell(in_size, hidden_size))
            self.gen_cells.append(LSTMCell(in_size, hidden_size))

        # Output projection (top-layer, 2*hidden_size because bidirectional)
        self.W_out = init_weight(2 * hidden_size, vocab_size)
        self.b_out = np.zeros((1, vocab_size))

        # Generator output projection (top-layer)
        self.W_gen = init_weight(hidden_size, vocab_size)
        self.b_gen = np.zeros((1, vocab_size))

        self._init_adam()

    def _init_adam(self):
        self.m = {}
        self.v = {}
        self.adam_t = 0
        self.param_names = []
        for prefix, cells in [('fwd', self.fwd_cells), ('bwd', self.bwd_cells), ('gen', self.gen_cells)]:
            for l, cell in enumerate(cells):
                for attr in ['W_f', 'b_f', 'W_i', 'b_i', 'W_c', 'b_c', 'W_o', 'b_o']:
                    name = f"{prefix}{l}_{attr}"
                    self.param_names.append(name)
                    self.m[name] = np.zeros_like(getattr(cell, attr))
                    self.v[name] = np.zeros_like(getattr(cell, attr))
        for name in ['W_out', 'b_out', 'W_gen', 'b_gen', 'E']:
            self.param_names.append(name)
            self.m[name] = np.zeros_like(getattr(self, name))
            self.v[name] = np.zeros_like(getattr(self, name))

    def count_parameters(self):
        total = self.E.size
        for cells in [self.fwd_cells, self.bwd_cells, self.gen_cells]:
            for cell in cells:
                total += cell.count_parameters()
        total += self.W_out.size + self.b_out.size
        total += self.W_gen.size + self.b_gen.size
        return total

    def compute_grad_and_loss(self, inputs, targets):
        """Forward + backward. Returns (loss, all_grads dict) without updating weights."""
        T = len(inputs)
        NL = self.num_layers

        # 3-layer Forward LSTM
        fwd_hs = [{} for _ in range(NL)]; fwd_cs = [{} for _ in range(NL)]; fwd_caches = [{} for _ in range(NL)]
        for l in range(NL):
            fwd_hs[l][-1] = np.zeros((1, self.hidden_size)); fwd_cs[l][-1] = np.zeros((1, self.hidden_size))
        for l in range(NL):
            for t in range(T):
                x_in = self.E[inputs[t]].reshape(1, -1) if l == 0 else fwd_hs[l-1][t]
                fwd_hs[l][t], fwd_cs[l][t], fwd_caches[l][t] = self.fwd_cells[l].forward(x_in, fwd_hs[l][t-1], fwd_cs[l][t-1])

        # 3-layer Backward LSTM
        bwd_hs = [{} for _ in range(NL)]; bwd_cs = [{} for _ in range(NL)]; bwd_caches = [{} for _ in range(NL)]
        for l in range(NL):
            bwd_hs[l][T] = np.zeros((1, self.hidden_size)); bwd_cs[l][T] = np.zeros((1, self.hidden_size))
        for l in range(NL):
            for t in reversed(range(T)):
                x_in = self.E[inputs[t]].reshape(1, -1) if l == 0 else bwd_hs[l-1][t]
                bwd_hs[l][t], bwd_cs[l][t], bwd_caches[l][t] = self.bwd_cells[l].forward(x_in, bwd_hs[l][t+1], bwd_cs[l][t+1])

        # Bidirectional output
        probs_bi = []
        for t in range(T):
            combined = np.concatenate([fwd_hs[NL-1][t], bwd_hs[NL-1][t]], axis=1)
            probs_bi.append(softmax(combined @ self.W_out + self.b_out).flatten())
        loss_bi = cross_entropy_loss(probs_bi, targets)

        # 3-layer Generator forward
        gen_hs = [{} for _ in range(NL)]; gen_cs = [{} for _ in range(NL)]; gen_caches = [{} for _ in range(NL)]
        for l in range(NL):
            gen_hs[l][-1] = np.zeros((1, self.hidden_size)); gen_cs[l][-1] = np.zeros((1, self.hidden_size))
        probs_gen = []
        for t in range(T):
            x_in = self.E[inputs[t]].reshape(1, -1)     # embedding lookup
            for l in range(NL):
                gen_hs[l][t], gen_cs[l][t], gen_caches[l][t] = self.gen_cells[l].forward(x_in, gen_hs[l][t-1], gen_cs[l][t-1])
                x_in = gen_hs[l][t]
            probs_gen.append(softmax(gen_hs[NL-1][t] @ self.W_gen + self.b_gen).flatten())
        loss_gen = cross_entropy_loss(probs_gen, targets)

        # Backprop generator
        dE = np.zeros_like(self.E)   # shared embedding gradient from all 3 stacks
        dW_gen = np.zeros_like(self.W_gen); db_gen = np.zeros_like(self.b_gen)
        dh_gen_next = [np.zeros((1, self.hidden_size)) for _ in range(NL)]
        dc_gen_next = [np.zeros((1, self.hidden_size)) for _ in range(NL)]
        gen_grads = [{a: np.zeros_like(getattr(self.gen_cells[l], a)) for a in ['W_f','b_f','W_i','b_i','W_c','b_c','W_o','b_o']} for l in range(NL)]
        for t in reversed(range(T)):
            dy = probs_gen[t].reshape(1, -1).copy(); dy[0, targets[t]] -= 1.0
            dW_gen += gen_hs[NL-1][t].T @ dy; db_gen += dy
            dh_upper = dy @ self.W_gen.T
            for l in reversed(range(NL)):
                dx_l, dh_gen_next[l], dc_gen_next[l], g = self.gen_cells[l].backward(dh_upper + dh_gen_next[l], dc_gen_next[l], gen_caches[l][t])
                for k in gen_grads[l]: gen_grads[l][k] += g[k]
                dh_upper = dx_l
            # dh_upper is now dx from layer 0 → gradient w.r.t. the embedding
            dE[inputs[t]] += dh_upper.flatten()

        # Backprop bidirectional
        dW_out = np.zeros_like(self.W_out); db_out = np.zeros_like(self.b_out)
        dh_fwd_next = [np.zeros((1, self.hidden_size)) for _ in range(NL)]
        dc_fwd_next = [np.zeros((1, self.hidden_size)) for _ in range(NL)]
        dh_bwd_next = [np.zeros((1, self.hidden_size)) for _ in range(NL)]
        dc_bwd_next = [np.zeros((1, self.hidden_size)) for _ in range(NL)]
        fwd_grads = [{a: np.zeros_like(getattr(self.fwd_cells[l], a)) for a in ['W_f','b_f','W_i','b_i','W_c','b_c','W_o','b_o']} for l in range(NL)]
        bwd_grads = [{a: np.zeros_like(getattr(self.bwd_cells[l], a)) for a in ['W_f','b_f','W_i','b_i','W_c','b_c','W_o','b_o']} for l in range(NL)]
        dy_list = []
        for t in range(T):
            dy = probs_bi[t].reshape(1, -1).copy(); dy[0, targets[t]] -= 1.0
            dy_list.append(dy)
            dW_out += np.concatenate([fwd_hs[NL-1][t], bwd_hs[NL-1][t]], axis=1).T @ dy; db_out += dy
        for t in reversed(range(T)):
            dh_upper = (dy_list[t] @ self.W_out.T)[:, :self.hidden_size]
            for l in reversed(range(NL)):
                dx_l, dh_fwd_next[l], dc_fwd_next[l], g = self.fwd_cells[l].backward(dh_upper + dh_fwd_next[l], dc_fwd_next[l], fwd_caches[l][t])
                for k in fwd_grads[l]: fwd_grads[l][k] += g[k]
                dh_upper = dx_l
            dE[inputs[t]] += dh_upper.flatten()   # fwd embedding gradient
        for t in range(T):
            dh_upper = (dy_list[t] @ self.W_out.T)[:, self.hidden_size:]
            for l in reversed(range(NL)):
                dx_l, dh_bwd_next[l], dc_bwd_next[l], g = self.bwd_cells[l].backward(dh_upper + dh_bwd_next[l], dc_bwd_next[l], bwd_caches[l][t])
                for k in bwd_grads[l]: bwd_grads[l][k] += g[k]
                dh_upper = dx_l
            dE[inputs[t]] += dh_upper.flatten()   # bwd embedding gradient

        all_grads = {}
        for l in range(NL):
            for attr in ['W_f','b_f','W_i','b_i','W_c','b_c','W_o','b_o']:
                all_grads[f'fwd{l}_{attr}'] = fwd_grads[l][attr]
                all_grads[f'bwd{l}_{attr}'] = bwd_grads[l][attr]
                all_grads[f'gen{l}_{attr}'] = gen_grads[l][attr]
        all_grads['W_out'] = dW_out; all_grads['b_out'] = db_out
        all_grads['W_gen'] = dW_gen; all_grads['b_gen'] = db_gen
        all_grads['E'] = dE
        return (loss_bi + loss_gen) / 2, all_grads

    def compute_loss(self, inputs, targets):
        """Forward-only pass using the generator LSTM for validation."""
        T = len(inputs)
        NL = self.num_layers
        hs = [{} for _ in range(NL)]; cs = [{} for _ in range(NL)]
        for l in range(NL):
            hs[l][-1] = np.zeros((1, self.hidden_size)); cs[l][-1] = np.zeros((1, self.hidden_size))
        probs = []
        for t in range(T):
            x_in = self.E[inputs[t]].reshape(1, -1)     # embedding lookup
            for l in range(NL):
                hs[l][t], cs[l][t], _ = self.gen_cells[l].forward(x_in, hs[l][t-1], cs[l][t-1])
                x_in = hs[l][t]
            probs.append(softmax(hs[NL-1][t] @ self.W_gen + self.b_gen).flatten())
        return cross_entropy_loss(probs, targets)

    def _adam_step(self, grads):
        """Clip gradients and apply one Adam update step."""
        keys = list(grads.keys())
        clipped = dict(zip(keys, clip_gradients([grads[k] for k in keys])))
        self.adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name in self.param_names:
            grad = clipped[name]
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * grad ** 2
            m_hat = self.m[name] / (1 - beta1 ** self.adam_t)
            v_hat = self.v[name] / (1 - beta2 ** self.adam_t)
            if name in ('W_out', 'b_out', 'W_gen', 'b_gen', 'E'):
                param = getattr(self, name)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                setattr(self, name, param)
            else:
                prefix_end = name.index('_')
                dl = name[:prefix_end]; attr = name[prefix_end+1:]
                direction = dl[:-1]; layer_idx = int(dl[-1])
                cell = self.fwd_cells[layer_idx] if direction == 'fwd' else (self.bwd_cells[layer_idx] if direction == 'bwd' else self.gen_cells[layer_idx])
                param = getattr(cell, attr)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                setattr(cell, attr, param)

    def train_step(self, inputs, targets):
        """Single-sample forward + backward + Adam update."""
        loss, grads = self.compute_grad_and_loss(inputs, targets)
        self._adam_step(grads)
        return loss

    def generate(self, max_len=50, temperature=0.8):
        """Generate using the 3-layer forward-only generator LSTM."""
        NL = self.num_layers
        hs = [np.zeros((1, self.hidden_size)) for _ in range(NL)]
        cs = [np.zeros((1, self.hidden_size)) for _ in range(NL)]
        idx = char_to_idx[SOS]
        name = []

        for _ in range(max_len):
            x_in = self.E[idx].reshape(1, -1)       # embedding lookup
            for l in range(NL):
                hs[l], cs[l], _ = self.gen_cells[l].forward(x_in, hs[l], cs[l])
                x_in = hs[l]
            y = hs[NL-1] @ self.W_gen + self.b_gen
            y = y / temperature
            p = softmax(y).flatten()
            idx = np.random.choice(len(p), p=p)
            ch = idx_to_char[idx]
            if ch == EOS:
                break
            if ch not in (PAD, SOS):
                name.append(ch)

        return ''.join(name)


# MODEL 3: RNN WITH BASIC ATTENTION

class RNNWithAttention:
    """
    RNN with Basic (Additive/Bahdanau) Attention for name generation.

    Architecture:
    - Embedding: Learned dense lookup E (vocab_size -> embed_dim)
      h1_t = tanh(W_xh1 * e_t  + W_hh1 * h1_{t-1} + b_h1)
      h2_t = tanh(W_xh2 * h1_t + W_hh2 * h2_{t-1} + b_h2)
      h3_t = tanh(W_xh3 * h2_t + W_hh3 * h3_{t-1} + b_h3)
    - Attention over top-layer (h3) encoder states
      e_t,i = v^T * tanh(W_a * h3_i + U_a * h3_t)
      alpha_t = softmax(e_t)
      context_t = sum(alpha_t,i * h3_i)
    - Output: y_t = softmax(W_out * [h3_t; context_t] + b_out)
    """

    def __init__(self, vocab_size, hidden_size=128, embed_dim=16, learning_rate=0.005):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.lr = learning_rate

        # Learned character embedding (same approach as VanillaRNN)
        self.E = init_weight(vocab_size, embed_dim)

        # Encoder: 3 stacked RNN layers
        self.W_xh1 = init_weight(embed_dim, hidden_size)
        self.W_hh1 = init_weight(hidden_size, hidden_size)
        self.b_h1  = np.zeros((1, hidden_size))
        self.W_xh2 = init_weight(hidden_size, hidden_size)
        self.W_hh2 = init_weight(hidden_size, hidden_size)
        self.b_h2  = np.zeros((1, hidden_size))
        self.W_xh3 = init_weight(hidden_size, hidden_size)
        self.W_hh3 = init_weight(hidden_size, hidden_size)
        self.b_h3  = np.zeros((1, hidden_size))

        # Attention parameters (operate on top-layer hidden states)
        self.W_a = init_weight(hidden_size, hidden_size)
        self.U_a = init_weight(hidden_size, hidden_size)
        self.v_a = init_weight(hidden_size, 1)

        # Output
        self.W_out = init_weight(hidden_size + hidden_size, vocab_size)
        self.b_out = np.zeros((1, vocab_size))

        self._init_adam()

    def _init_adam(self):
        self.m = {}
        self.v = {}
        self.adam_t = 0
        for name in ['E',
                     'W_xh1', 'W_hh1', 'b_h1',
                     'W_xh2', 'W_hh2', 'b_h2',
                     'W_xh3', 'W_hh3', 'b_h3',
                     'W_a', 'U_a', 'v_a', 'W_out', 'b_out']:
            self.m[name] = np.zeros_like(getattr(self, name))
            self.v[name] = np.zeros_like(getattr(self, name))

    def count_parameters(self):
        total = 0
        for name in ['E',
                     'W_xh1', 'W_hh1', 'b_h1',
                     'W_xh2', 'W_hh2', 'b_h2',
                     'W_xh3', 'W_hh3', 'b_h3',
                     'W_a', 'U_a', 'v_a', 'W_out', 'b_out']:
            total += getattr(self, name).size
        return total

    def _compute_attention(self, encoder_hiddens, current_h):
        """Compute attention weights and context vector over top-layer states."""
        T = len(encoder_hiddens)
        scores = np.zeros(T)

        for i in range(T):
            energy = tanh(encoder_hiddens[i] @ self.W_a + current_h @ self.U_a)
            scores[i] = (energy @ self.v_a).flatten()[0]

        weights = softmax(scores.reshape(1, -1)).flatten()

        context = np.zeros((1, self.hidden_size))
        for i in range(T):
            context += weights[i] * encoder_hiddens[i]

        return context, weights

    def compute_grad_and_loss(self, inputs, targets):
        """Forward + full attention backward. Returns (loss, grads_dict)."""
        T = len(inputs)
        H = self.hidden_size

        # === Forward: 3-layer encoder ===
        xs = {}; hs1, hs2, hs3 = {}, {}, {}
        hs1[-1] = hs2[-1] = hs3[-1] = np.zeros((1, H))
        encoder_hiddens = []
        for t in range(T):
            xs[t]  = self.E[inputs[t]].reshape(1, -1)   # embedding lookup, no one-hot
            hs1[t] = tanh(xs[t]  @ self.W_xh1 + hs1[t-1] @ self.W_hh1 + self.b_h1)
            hs2[t] = tanh(hs1[t] @ self.W_xh2 + hs2[t-1] @ self.W_hh2 + self.b_h2)
            hs3[t] = tanh(hs2[t] @ self.W_xh3 + hs3[t-1] @ self.W_hh3 + self.b_h3)
            encoder_hiddens.append(hs3[t].copy())

        # === Forward: attention + output; cache energies/alphas for full backward ===
        probs_list, contexts = [], []
        all_energies, all_alphas = [], []
        for t in range(T):
            if t == 0:
                context = np.zeros((1, H)); energies_t = []; alpha_t = np.array([])
            else:
                enc = encoder_hiddens[:t]
                energies_t = [tanh(enc[i] @ self.W_a + hs3[t] @ self.U_a) for i in range(t)]
                scores = np.array([(energies_t[i] @ self.v_a).flatten()[0] for i in range(t)])
                alpha_t = softmax(scores.reshape(1, -1)).flatten()
                context = sum(alpha_t[i] * enc[i] for i in range(t))
            contexts.append(context); all_energies.append(energies_t); all_alphas.append(alpha_t)
            probs_list.append(softmax(np.concatenate([hs3[t], context], axis=1) @ self.W_out + self.b_out).flatten())

        loss = cross_entropy_loss(probs_list, targets)

        # === Backward: output layer + full attention gradient ===
        dE     = np.zeros_like(self.E)
        dW_xh1 = np.zeros_like(self.W_xh1); dW_hh1 = np.zeros_like(self.W_hh1); db_h1 = np.zeros_like(self.b_h1)
        dW_xh2 = np.zeros_like(self.W_xh2); dW_hh2 = np.zeros_like(self.W_hh2); db_h2 = np.zeros_like(self.b_h2)
        dW_xh3 = np.zeros_like(self.W_xh3); dW_hh3 = np.zeros_like(self.W_hh3); db_h3 = np.zeros_like(self.b_h3)
        dW_out = np.zeros_like(self.W_out); db_out = np.zeros_like(self.b_out)
        dW_a = np.zeros_like(self.W_a); dU_a = np.zeros_like(self.U_a); dv_a = np.zeros_like(self.v_a)

        # dh3_extra[t] accumulates attention gradients to hs3[t]
        # (from being an encoder state in future steps, or query at current step)
        dh3_extra = [np.zeros((1, H)) for _ in range(T)]

        for t in range(T):
            dy = probs_list[t].reshape(1, -1).copy(); dy[0, targets[t]] -= 1.0
            dW_out += np.concatenate([hs3[t], contexts[t]], axis=1).T @ dy; db_out += dy
            d_combined = dy @ self.W_out.T
            dh3_extra[t] += d_combined[:, :H]   # gradient from output
            d_context = d_combined[:, H:]

            if t > 0:
                enc = encoder_hiddens[:t]; alpha_t = all_alphas[t]; energies_t = all_energies[t]
                # Gradient from context = sum(alpha_i * h_enc_i) to alpha and encoder hiddens
                d_alpha = np.array([np.sum(d_context * enc[i]) for i in range(t)])
                for i in range(t):
                    dh3_extra[i] += alpha_t[i] * d_context          # direct context path
                # Full softmax backward: d_score_i = alpha_i*(d_alpha_i - dot(alpha, d_alpha))
                dot = np.sum(alpha_t * d_alpha)
                d_score = alpha_t * (d_alpha - dot)
                for i in range(t):
                    d_energy_i = d_score[i] * self.v_a.T             # (1, H)
                    dv_a += energies_t[i].T * d_score[i]             # (H, 1)
                    d_energy_raw = d_energy_i * (1 - energies_t[i] ** 2)
                    dW_a += enc[i].T @ d_energy_raw                  # encoder hidden path
                    dU_a += hs3[t].T @ d_energy_raw                  # query path
                    dh3_extra[i] += d_energy_raw @ self.W_a.T        # to encoder hidden
                    dh3_extra[t] += d_energy_raw @ self.U_a.T        # to query

        # === Backward: BPTT through 3-layer RNN using accumulated dh3_extra ===
        dh1_next = np.zeros((1, H)); dh2_next = np.zeros((1, H)); dh3_next = np.zeros((1, H))
        for t in reversed(range(T)):
            dh3     = dh3_extra[t] + dh3_next
            dh3_raw = dh3 * (1 - hs3[t] ** 2)
            dW_xh3 += hs2[t].T @ dh3_raw; dW_hh3 += hs3[t-1].T @ dh3_raw; db_h3 += dh3_raw
            dh3_next = dh3_raw @ self.W_hh3.T
            dh2     = dh3_raw @ self.W_xh3.T + dh2_next
            dh2_raw = dh2 * (1 - hs2[t] ** 2)
            dW_xh2 += hs1[t].T @ dh2_raw; dW_hh2 += hs2[t-1].T @ dh2_raw; db_h2 += dh2_raw
            dh2_next = dh2_raw @ self.W_hh2.T
            dh1     = dh2_raw @ self.W_xh2.T + dh1_next
            dh1_raw = dh1 * (1 - hs1[t] ** 2)
            dW_xh1 += xs[t].T @ dh1_raw; dW_hh1 += hs1[t-1].T @ dh1_raw; db_h1 += dh1_raw
            dh1_next = dh1_raw @ self.W_hh1.T
            # Embedding gradient: sparse row update for this character
            dE[inputs[t]] += (dh1_raw @ self.W_xh1.T).flatten()

        _names = ['E','W_xh1','W_hh1','b_h1','W_xh2','W_hh2','b_h2','W_xh3','W_hh3','b_h3','W_a','U_a','v_a','W_out','b_out']
        _raw   = [dE, dW_xh1, dW_hh1, db_h1, dW_xh2, dW_hh2, db_h2, dW_xh3, dW_hh3, db_h3, dW_a, dU_a, dv_a, dW_out, db_out]
        return loss, dict(zip(_names, _raw))

    def compute_loss(self, inputs, targets):
        """Forward-only pass for validation. Returns loss without gradients."""
        H = self.hidden_size
        h1 = h2 = h3 = np.zeros((1, H))
        encoder_hiddens = []; probs = []
        for t in range(len(inputs)):
            x  = self.E[inputs[t]].reshape(1, -1)       # embedding lookup
            h1 = tanh(x  @ self.W_xh1 + h1 @ self.W_hh1 + self.b_h1)
            h2 = tanh(h1 @ self.W_xh2 + h2 @ self.W_hh2 + self.b_h2)
            h3 = tanh(h2 @ self.W_xh3 + h3 @ self.W_hh3 + self.b_h3)
            encoder_hiddens.append(h3.copy())
            context = np.zeros((1, H)) if t == 0 else self._compute_attention(encoder_hiddens[:t], h3)[0]
            probs.append(softmax(np.concatenate([h3, context], axis=1) @ self.W_out + self.b_out).flatten())
        return cross_entropy_loss(probs, targets)

    def _adam_step(self, grads):
        """Clip gradients and apply one Adam update step."""
        names = ['E','W_xh1','W_hh1','b_h1','W_xh2','W_hh2','b_h2','W_xh3','W_hh3','b_h3','W_a','U_a','v_a','W_out','b_out']
        grad_list = clip_gradients([grads[n] for n in names])
        self.adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, grad in zip(names, grad_list):
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * grad ** 2
            m_hat = self.m[name] / (1 - beta1 ** self.adam_t)
            v_hat = self.v[name] / (1 - beta2 ** self.adam_t)
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, param)

    def train_step(self, inputs, targets):
        """Single-sample forward + backward + Adam update."""
        loss, grads = self.compute_grad_and_loss(inputs, targets)
        self._adam_step(grads)
        return loss

    def generate(self, max_len=50, temperature=0.8):
        """Generate a name with 3-layer encoder and attention over top-layer states."""
        h1 = np.zeros((1, self.hidden_size))
        h2 = np.zeros((1, self.hidden_size))
        h3 = np.zeros((1, self.hidden_size))
        idx = char_to_idx[SOS]
        name = []
        all_hiddens = []

        for _ in range(max_len):
            x  = self.E[idx].reshape(1, -1)             # embedding lookup
            h1 = tanh(x  @ self.W_xh1 + h1 @ self.W_hh1 + self.b_h1)
            h2 = tanh(h1 @ self.W_xh2 + h2 @ self.W_hh2 + self.b_h2)
            h3 = tanh(h2 @ self.W_xh3 + h3 @ self.W_hh3 + self.b_h3)
            all_hiddens.append(h3.copy())

            if len(all_hiddens) > 1:
                context, _ = self._compute_attention(all_hiddens[:-1], h3)
            else:
                context = np.zeros((1, self.hidden_size))

            combined = np.concatenate([h3, context], axis=1)
            y = combined @ self.W_out + self.b_out
            y = y / temperature
            p = softmax(y).flatten()
            idx = np.random.choice(len(p), p=p)
            ch = idx_to_char[idx]
            if ch == EOS:
                break
            if ch not in (PAD, SOS):
                name.append(ch)

        return ''.join(name)


# TRAINING FUNCTION

def train_model(model, model_name, encoded_names, val_encoded_names=None,
                epochs=15, samples_per_epoch=500, batch_size=16,
                lr_min=1e-5, patience=20,
                reduce_lr_patience=4, reduce_lr_factor=0.3):
    """
    Train with:
      - Mini-batch gradient accumulation (batch_size samples averaged before one Adam step)
      - ReduceLROnPlateau: halve LR when val loss stops improving for reduce_lr_patience epochs
      - Early stopping on validation loss (or training loss if no val set)
    """
    print(f"\n{'_' * 70}")
    print(f"Training: {model_name}")
    print(f"{'_' * 70}")
    print(f"Trainable Parameters: {model.count_parameters():,}")

    n = len(encoded_names)
    lr_max = model.lr
    losses, val_losses = [], []
    best_monitor = float('inf')
    epochs_no_improve = 0       # counter for early stopping
    lr_no_improve = 0           # separate counter for LR reduction

    for epoch in range(epochs):
        indices = random.choices(range(n), k=samples_per_epoch)  # with replacement
        random.shuffle(indices)

        epoch_loss = 0.0
        count = 0

        # Mini-batch gradient accumulation
        for b_start in range(0, len(indices), batch_size):
            batch = indices[b_start:b_start + batch_size]
            accum_grads = None
            batch_loss = 0.0

            for idx in batch:
                seq = encoded_names[idx]
                loss, grads = model.compute_grad_and_loss(seq[:-1], seq[1:])
                batch_loss += loss
                if accum_grads is None:
                    accum_grads = {k: g.copy() for k, g in grads.items()}
                else:
                    for k in accum_grads:
                        accum_grads[k] += grads[k]

            # Average gradients over batch, then apply one Adam step
            if accum_grads is None:
                continue
            nb = len(batch)
            for k in accum_grads:
                accum_grads[k] /= nb
            model._adam_step(accum_grads)
            epoch_loss += batch_loss
            count += nb

        avg_loss = epoch_loss / count
        losses.append(avg_loss)

        # Validation loss (forward-only, no gradients)
        if val_encoded_names:
            val_idx = random.sample(range(len(val_encoded_names)),
                                    min(150, len(val_encoded_names)))
            val_loss = sum(model.compute_loss(val_encoded_names[i][:-1],
                                              val_encoded_names[i][1:])
                           for i in val_idx) / len(val_idx)
            val_losses.append(val_loss)
            monitor = val_loss
        else:
            monitor = avg_loss

        # ReduceLROnPlateau: reduce LR when stuck, before giving up entirely
        if monitor < best_monitor - 1e-5:
            best_monitor = monitor
            epochs_no_improve = 0
            lr_no_improve = 0
        else:
            epochs_no_improve += 1
            lr_no_improve += 1
            # Halve LR every reduce_lr_patience epochs of no improvement
            if lr_no_improve >= reduce_lr_patience:
                new_lr = max(model.lr * reduce_lr_factor, lr_min)
                if new_lr < model.lr:
                    model.lr = new_lr
                    print(f"  [ReduceLR] LR → {model.lr:.6f}")
                lr_no_improve = 0   # reset LR counter, keep early-stop counter

        if (epoch + 1) % 3 == 0 or epoch == 0:
            sample = model.generate()
            val_str = f" | Val: {val_losses[-1]:.4f}" if val_losses else ""
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}{val_str}"
                  f" | LR: {model.lr:.6f} | Sample: '{sample}'")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    model.lr = lr_max
    return losses, val_losses


# EVALUATION FUNCTIONS 
def generate_names_from_model(model, n=200, temperature=0.8):
    """Generate n names from a trained model."""
    names = []
    for _ in range(n):
        name = model.generate(temperature=temperature)
        if len(name) > 0:
            names.append(name)
    return names

def compute_novelty(generated, training_set):
    """Compute novelty rate: % of generated names NOT in training set."""
    training_lower = set(n.lower() for n in training_set)
    novel = sum(1 for name in generated if name.lower() not in training_lower)
    return novel / len(generated) * 100 if generated else 0

def compute_diversity(generated):
    """Compute diversity: unique names / total names."""
    if not generated:
        return 0
    return len(set(generated)) / len(generated) * 100


# MAIN EXECUTION

if __name__ == "__main__":

    #  MODEL IMPLEMENTATION

    print(" MODEL IMPLEMENTATION")


    # Train / validation split (80 / 20)
    random.shuffle(encoded_names)
    split = int(0.8 * len(encoded_names))
    train_encoded = encoded_names[:split]
    val_encoded   = encoded_names[split:]
    print(f"Train: {len(train_encoded)} names | Val: {len(val_encoded)} names")

    # Hyperparameters (tuned via TPE hyperparameter search)
    #i used another code to find the best value, but that code is not requires for this assignment, 
    # so i will not include it here. I just want to show the final values that i found for each model.
    HIDDEN_SIZE_RNN = 128
    HIDDEN_SIZE_BLSTM = 64   # Per direction (total 128 combined)
    HIDDEN_SIZE_ATTN = 128
    EMBED_DIM = 32
    LEARNING_RATE = 0.005
    LR_MIN = 1e-5             # ReduceLROnPlateau floor (LR never goes below this)
    EPOCHS = 200
    SAMPLES_PER_EPOCH = 500   # Train samples per epoch
    BATCH_SIZE = 16           # Samples averaged before one Adam step
    PATIENCE = 30             # Early stopping patience
    REDUCE_LR_PATIENCE = 3    # Epochs of no improvement before halving LR
    REDUCE_LR_FACTOR = 0.3    # Multiply LR by this on plateau

    print("\n--- Model 1: Vanilla RNN ---")
    print(f"Architecture: Input({vocab_size}) -> RNN x3(hidden={HIDDEN_SIZE_RNN}, tanh) -> Linear -> Softmax")
    print(f"Hidden Size: {HIDDEN_SIZE_RNN}")
    print(f"Layers: 3")
    print(f"Learning Rate: {LEARNING_RATE}")

    rnn = VanillaRNN(vocab_size, hidden_size=HIDDEN_SIZE_RNN, embed_dim=EMBED_DIM, learning_rate=LEARNING_RATE)
    print(f"Trainable Parameters: {rnn.count_parameters():,}")

    print(f"\n--- Model 2: Bidirectional LSTM ---")
    print(f"Architecture: Input({vocab_size}) -> BiLSTM x3(hidden={HIDDEN_SIZE_BLSTM}x2) -> Linear -> Softmax")
    print(f"  + 3-layer Forward-only LSTM generator for inference")
    print(f"Hidden Size: {HIDDEN_SIZE_BLSTM} per direction ({HIDDEN_SIZE_BLSTM * 2} combined)")
    print(f"Layers: 3 bidirectional + 3 unidirectional generator")
    print(f"Learning Rate: {LEARNING_RATE}")

    blstm = BLSTM(vocab_size, hidden_size=HIDDEN_SIZE_BLSTM, embed_dim=EMBED_DIM, learning_rate=LEARNING_RATE)
    print(f"Trainable Parameters: {blstm.count_parameters():,}")

    print(f"\n--- Model 3: RNN with Attention ---")
    print(f"Architecture: Input({vocab_size}) -> RNN x3(hidden={HIDDEN_SIZE_ATTN}) -> Self-Attention -> Linear -> Softmax")
    print(f"  Attention: Additive (Bahdanau) over top-layer hidden states")
    print(f"Hidden Size: {HIDDEN_SIZE_ATTN}")
    print(f"Layers: 3 + attention layer")
    print(f"Learning Rate: {LEARNING_RATE}")

    attn_rnn = RNNWithAttention(vocab_size, hidden_size=HIDDEN_SIZE_ATTN, embed_dim=EMBED_DIM, learning_rate=LEARNING_RATE)
    print(f"Trainable Parameters: {attn_rnn.count_parameters():,}")

    # TRAINING
  
    print("TRAINING ALL MODELS")
  

    rnn_losses, rnn_val_losses     = train_model(rnn,      "Vanilla RNN",        train_encoded,
                                                 val_encoded_names=val_encoded,
                                                 epochs=EPOCHS, samples_per_epoch=SAMPLES_PER_EPOCH,
                                                 batch_size=BATCH_SIZE, lr_min=LR_MIN, patience=PATIENCE,
                                                 reduce_lr_patience=REDUCE_LR_PATIENCE,
                                                 reduce_lr_factor=REDUCE_LR_FACTOR)
    blstm_losses, blstm_val_losses = train_model(blstm,    "Bidirectional LSTM", train_encoded,
                                                 val_encoded_names=val_encoded,
                                                 epochs=EPOCHS, samples_per_epoch=SAMPLES_PER_EPOCH,
                                                 batch_size=BATCH_SIZE, lr_min=LR_MIN, patience=PATIENCE,
                                                 reduce_lr_patience=REDUCE_LR_PATIENCE,
                                                 reduce_lr_factor=REDUCE_LR_FACTOR)
    attn_losses, attn_val_losses   = train_model(attn_rnn, "RNN with Attention", train_encoded,
                                                 val_encoded_names=val_encoded,
                                                 epochs=EPOCHS, samples_per_epoch=SAMPLES_PER_EPOCH,
                                                 batch_size=BATCH_SIZE, lr_min=LR_MIN, patience=PATIENCE,
                                                 reduce_lr_patience=REDUCE_LR_PATIENCE,
                                                 reduce_lr_factor=REDUCE_LR_FACTOR)

    # QUANTITATIVE EVALUATION

    print("EVALUATION")


    NUM_GENERATE = 200

    print(f"\nGenerating {NUM_GENERATE} names from each model...\n")

    rnn_names = generate_names_from_model(rnn, NUM_GENERATE)
    blstm_names = generate_names_from_model(blstm, NUM_GENERATE)
    attn_names = generate_names_from_model(attn_rnn, NUM_GENERATE)

    results = {}
    for model_name, gen_names in [("Vanilla RNN", rnn_names),
                                   ("BLSTM", blstm_names),
                                   ("RNN+Attention", attn_names)]:
        novelty = compute_novelty(gen_names, training_names)
        diversity = compute_diversity(gen_names)
        results[model_name] = {
            'novelty': novelty,
            'diversity': diversity,
            'names': gen_names,
            'count': len(gen_names)
        }

    print(f"{'Model':<20} {'Generated':<12} {'Novelty %':<12} {'Diversity %':<12}")
    print("-" * 56)
    for model_name, res in results.items():
        print(f"{model_name:<20} {res['count']:<12} {res['novelty']:<12.1f} {res['diversity']:<12.1f}")

    # QUALITATIVE ANALYSIS

    print("QUALITATIVE ANALYSIS")


    for model_name, gen_names in [("Vanilla RNN", rnn_names),
                                   ("BLSTM", blstm_names),
                                   ("RNN+Attention", attn_names)]:
        print(f"\n--- {model_name} ---")
        print(f"Representative samples (first 20):")
        for i, name in enumerate(gen_names[:20]):
            print(f"  {i + 1:2d}. {name}")



    print("FAILURE MODE ANALYSIS")


    for model_name, gen_names in [("Vanilla RNN", rnn_names),
                                   ("BLSTM", blstm_names),
                                   ("RNN+Attention", attn_names)]:
        print(f"\n--- {model_name} ---")

        # Too short names (< 3 chars)
        too_short = [n for n in gen_names if len(n) < 3]
        print(f"  Too short (<3 chars): {len(too_short)} ({len(too_short)/len(gen_names)*100:.1f}%)")
        if too_short[:5]:
            print(f"    Examples: {too_short[:5]}")

        # Too long names (> 25 chars)
        too_long = [n for n in gen_names if len(n) > 25]
        print(f"  Too long (>25 chars): {len(too_long)} ({len(too_long)/len(gen_names)*100:.1f}%)")
        if too_long[:3]:
            print(f"    Examples: {too_long[:3]}")

        # Names with no space (missing last name)
        no_space = [n for n in gen_names if ' ' not in n]
        print(f"  No space (single word): {len(no_space)} ({len(no_space)/len(gen_names)*100:.1f}%)")
        if no_space[:5]:
            print(f"    Examples: {no_space[:5]}")

        # Repeated characters
        repeated = [n for n in gen_names if any(n[i] == n[i+1] == n[i+2] for i in range(len(n)-2) if n[i].isalpha())]
        print(f"  Repeated chars (3+): {len(repeated)} ({len(repeated)/len(gen_names)*100:.1f}%)")
        if repeated[:5]:
            print(f"    Examples: {repeated[:5]}")

        # Average length
        avg_len = np.mean([len(n) for n in gen_names])
        print(f"  Average generated length: {avg_len:.1f} chars")

    # Training name stats for reference
    avg_train_len = np.mean([len(n) for n in training_names])
    print(f"\n  [Reference] Average training name length: {avg_train_len:.1f} chars")


    # LOSS CURVES SUMMARY

    print("TRAINING LOSS SUMMARY")

    print(f"\n{'Epoch':<10} {'Vanilla RNN':<15} {'BLSTM':<15} {'RNN+Attention':<15}")
    print("-" * 55)
    min_len = min(len(rnn_losses), len(blstm_losses), len(attn_losses))
    for i in range(0, min_len, 3):
        print(f"{i + 1:<10} {rnn_losses[i]:<15.4f} {blstm_losses[i]:<15.4f} {attn_losses[i]:<15.4f}")
    print(f"{'Final':<10} {rnn_losses[-1]:<15.4f} {blstm_losses[-1]:<15.4f} {attn_losses[-1]:<15.4f}")

    # ── SAVE PLOTS & RESULTS ────────────────────────────────────────────────

    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Individual train + val loss per model
    for label, tr, va in [
        ("VanillaRNN",  rnn_losses,   rnn_val_losses),
        ("BLSTM",       blstm_losses, blstm_val_losses),
        ("RNNWithAttn", attn_losses,  attn_val_losses),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tr, label="Train loss")
        ax.plot(va, label="Val loss", linestyle="--")
        ax.set_title(f"{label} — Training & Validation Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, f"loss_{label}.png"), dpi=150)
        plt.close(fig)

    # 2. Combined training loss comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rnn_losses,   label="Vanilla RNN")
    ax.plot(blstm_losses, label="BLSTM")
    ax.plot(attn_losses,  label="RNN+Attention")
    ax.set_title("Training Loss — All Models")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "train_loss_comparison.png"), dpi=150)
    plt.close(fig)

    # 3. Combined validation loss comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rnn_val_losses,   label="Vanilla RNN")
    ax.plot(blstm_val_losses, label="BLSTM")
    ax.plot(attn_val_losses,  label="RNN+Attention")
    ax.set_title("Validation Loss — All Models")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "val_loss_comparison.png"), dpi=150)
    plt.close(fig)

    # 4. Novelty & Diversity bar chart
    model_labels   = list(results.keys())
    novelty_vals   = [results[m]['novelty']   for m in model_labels]
    diversity_vals = [results[m]['diversity'] for m in model_labels]
    bar_x = np.arange(len(model_labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bar_x - w/2, novelty_vals,   w, label="Novelty %",   color="steelblue", alpha=0.85)
    ax.bar(bar_x + w/2, diversity_vals, w, label="Diversity %", color="tomato",    alpha=0.85)
    ax.set_xticks(bar_x); ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 115); ax.set_ylabel("Percentage (%)")
    ax.set_title("Novelty & Diversity per Model")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    for i, (nv, dv) in enumerate(zip(novelty_vals, diversity_vals)):
        ax.text(i - w/2, nv + 1, f"{nv:.1f}%", ha='center', fontsize=9)
        ax.text(i + w/2, dv + 1, f"{dv:.1f}%", ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "novelty_diversity.png"), dpi=150)
    plt.close(fig)

    # 5. Name length distribution — training vs generated
    train_lens = [len(n) for n in training_names]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (mname, gnames) in zip(axes, [
        ("Vanilla RNN",   rnn_names),
        ("BLSTM",         blstm_names),
        ("RNN+Attention", attn_names),
    ]):
        gen_lens = [len(n) for n in gnames]
        bins = range(0, max(train_lens + gen_lens) + 3)
        ax.hist(train_lens, bins=bins, alpha=0.5, label="Training",  color="steelblue", density=True)
        ax.hist(gen_lens,   bins=bins, alpha=0.6, label="Generated", color="tomato",    density=True)
        ax.set_title(mname); ax.set_xlabel("Name length")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Density")
    fig.suptitle("Name Length Distribution — Training vs Generated", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "length_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6. Failure mode analysis bar chart
    failure_labels = ["Too short\n(<3)", "Too long\n(>25)", "No space", "Repeated\nchars"]
    bar_x2 = np.arange(len(failure_labels)); width = 0.25
    fm_colors = ["steelblue", "tomato", "seagreen"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (mname, gnames) in enumerate([
        ("Vanilla RNN",   rnn_names),
        ("BLSTM",         blstm_names),
        ("RNN+Attention", attn_names),
    ]):
        n = len(gnames)
        pcts = [
            100 * len([nm for nm in gnames if len(nm) < 3]) / n,
            100 * len([nm for nm in gnames if len(nm) > 25]) / n,
            100 * len([nm for nm in gnames if ' ' not in nm]) / n,
            100 * len([nm for nm in gnames if any(
                nm[j] == nm[j+1] == nm[j+2]
                for j in range(len(nm) - 2) if nm[j].isalpha()
            )]) / n,
        ]
        ax.bar(bar_x2 + i * width, pcts, width, label=mname, color=fm_colors[i], alpha=0.85)
    ax.set_xticks(bar_x2 + width); ax.set_xticklabels(failure_labels)
    ax.set_ylabel("Percentage (%)"); ax.set_title("Failure Mode Analysis")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "failure_modes.png"), dpi=150)
    plt.close(fig)

    # 7. Model parameter count comparison
    param_counts = {
        "Vanilla RNN":   rnn.count_parameters(),
        "BLSTM":         blstm.count_parameters(),
        "RNN+Attention": attn_rnn.count_parameters(),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(param_counts.keys()), list(param_counts.values()),
                  color=["steelblue", "tomato", "seagreen"], alpha=0.85)
    ax.set_ylabel("Parameter count"); ax.set_title("Model Size Comparison")
    mx = max(param_counts.values())
    for bar, v in zip(bars, param_counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + mx * 0.01,
                f"{v:,}", ha='center', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "model_size.png"), dpi=150)
    plt.close(fig)

    # 8. Save all results to JSON
    summary = {
        "hyperparameters": {
            "hidden_size_rnn":    HIDDEN_SIZE_RNN,
            "hidden_size_blstm":  HIDDEN_SIZE_BLSTM,
            "hidden_size_attn":   HIDDEN_SIZE_ATTN,
            "embed_dim":          EMBED_DIM,
            "learning_rate":      LEARNING_RATE,
            "batch_size":         BATCH_SIZE,
            "samples_per_epoch":  SAMPLES_PER_EPOCH,
            "epochs":             EPOCHS,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
            "reduce_lr_factor":   REDUCE_LR_FACTOR,
        },
        "models": {},
    }
    for mname, tr, va, gnames, res_key in [
        ("VanillaRNN",  rnn_losses,   rnn_val_losses,   rnn_names,   "Vanilla RNN"),
        ("BLSTM",       blstm_losses, blstm_val_losses, blstm_names, "BLSTM"),
        ("RNNWithAttn", attn_losses,  attn_val_losses,  attn_names,  "RNN+Attention"),
    ]:
        summary["models"][mname] = {
            "epochs_run":       len(tr),
            "param_count":      param_counts[res_key],
            "final_train_loss": float(tr[-1]),
            "best_train_loss":  float(min(tr)),
            "final_val_loss":   float(va[-1]),
            "best_val_loss":    float(min(va)),
            "novelty_pct":      float(results[res_key]['novelty']),
            "diversity_pct":    float(results[res_key]['diversity']),
            "train_losses":     [float(x) for x in tr],
            "val_losses":       [float(x) for x in va],
            "generated_names":  gnames,
        }
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)



