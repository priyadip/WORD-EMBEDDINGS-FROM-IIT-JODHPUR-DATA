"""
Word2Vec from Scratch (NumPy only)
====================================
Implements CBOW and Skip-gram with Negative Sampling using pure NumPy.
No PyTorch, no Gensim  -  only numpy, random, collections, json.

Architecture:
  W_in  : (vocab_size x embed_dim)  -  input  embeddings
  W_out : (vocab_size x embed_dim)  -  output embeddings
  Loss  : Negative Sampling (binary cross-entropy)
  Update: Stochastic Gradient Descent

Window sweep: trains CBOW and Skip-gram for window sizes [3, 5, 10],
saves results to data/models/scratch_experiment_results.json.
Best model (window=5) saved as scratch_cbow_vectors.npy and scratch_skipgram_vectors.npy.

Output:
  data/models/scratch_cbow_vectors.npy
  data/models/scratch_skipgram_vectors.npy
  data/models/scratch_vocab.json
  data/models/scratch_experiment_results.json
"""

import os
import json
import time
import random
import numpy as np
from collections import Counter

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "corpus.txt")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

EMBED_DIM   = 100        # smaller dim for speed (NumPy CPU training)
WINDOWS     = [3, 5, 10] # window sizes to sweep (same as Gensim script)
DEFAULT_WIN = 5          # best window size - used for final saved model
NEG_SAMPLES = 5
MIN_COUNT   = 5          # minimum word frequency to include in vocabulary
EPOCHS      = 20
LR          = 0.01    # initial learning rate (linearly decayed)
SUBSAMPLE_T = 1e-4
MAX_SENT    = 100_000    # cap sentences for speed; remove for full corpus
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)


# Sigmoid (numerically stable)
def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# 1. Build Vocabulary
def build_vocab(corpus_path):
    print("Building vocabulary ...")
    freq     = Counter()
    all_sents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                freq.update(tokens)
                all_sents.append(tokens)

    # Filter by min_count
    vocab    = {w: c for w, c in freq.items() if c >= MIN_COUNT}
    word2idx = {w: i for i, w in enumerate(sorted(vocab))}
    idx2word = {i: w for w, i in word2idx.items()}
    V        = len(word2idx)

    # Subsampling keep probability: p_keep = min(1, sqrt(t/f) + t/f)
    total     = sum(vocab.values())
    keep_prob = {}
    for w, c in vocab.items():
        f = c / total
        keep_prob[w] = min(1.0, np.sqrt(SUBSAMPLE_T / f) + SUBSAMPLE_T / f)

    # Negative sampling distribution: freq^(3/4), normalised
    freqs    = np.array([vocab[idx2word[i]] for i in range(V)], dtype=np.float64)
    neg_dist = freqs ** 0.75
    neg_dist /= neg_dist.sum()
    neg_table = np.random.choice(V, size=10_000_000, p=neg_dist)  # pre-sample

    print(f"  Vocab size : {V:,}  (from {len(freq):,} unique, min_count={MIN_COUNT})")
    print(f"  Sentences  : {len(all_sents):,}  (using first {min(MAX_SENT, len(all_sents)):,})")
    return word2idx, idx2word, keep_prob, neg_table, all_sents[:MAX_SENT]


# 2. Initialise Embeddings
def init_embeddings(V, D):
    W_in  = (np.random.rand(V, D) - 0.5) / D   # small uniform init
    W_out = np.zeros((V, D), dtype=np.float64)
    return W_in, W_out


# 3. Negative Sampling Loss + Gradients
def ns_update(v_in, v_out_pos, v_out_negs, lr):
    """
    One negative-sampling update step.

    v_in      : (D,)     -  input  embedding of center/context word
    v_out_pos : (D,)     -  output embedding of positive word
    v_out_negs: (K, D)   -  output embeddings of K negative words
    lr        : float

    Returns gradients for in-place update.
    """
    # Positive pair
    score_pos  = sigmoid(v_in @ v_out_pos)           # scalar
    grad_pos   = (1 - score_pos)                     # dL/dscore for positive

    # Negative pairs
    scores_neg = sigmoid(v_in @ v_out_negs.T)        # (K,)
    grads_neg  = -scores_neg                         # dL/dscore for negatives

    # Gradient w.r.t. v_in (sum over pos + all negs)
    grad_v_in  = grad_pos * v_out_pos + (grads_neg[:, None] * v_out_negs).sum(axis=0)

    # Gradient w.r.t. output vectors
    grad_v_out_pos  = grad_pos  * v_in               # (D,)
    grad_v_out_negs = grads_neg[:, None] * v_in      # (K, D)

    return grad_v_in, grad_v_out_pos, grad_v_out_negs


# 4. Skip-gram Training
def train_skipgram(sentences, word2idx, keep_prob, neg_table, window):
    V = len(word2idx)
    W_in, W_out = init_embeddings(V, EMBED_DIM)

    total_pairs = 0
    for sent in sentences:
        l = sum(1 for w in sent if w in word2idx)
        total_pairs += l * 2 * window

    pair_no = 0
    neg_ptr = 0

    print(f"\n  Skip-gram | epochs={EPOCHS} | vocab={V:,} | dim={EMBED_DIM} | window={window}")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_updates  = 0
        random.shuffle(sentences)

        for sent in sentences:
            idxs = [word2idx[w] for w in sent
                    if w in word2idx and random.random() < keep_prob.get(w, 1.0)]
            if len(idxs) < 2:
                continue

            for i, center_idx in enumerate(idxs):
                # Dynamic window: random in [1, window]
                w = random.randint(1, window)
                ctx_range = range(max(0, i - w), min(len(idxs), i + w + 1))

                for j in ctx_range:
                    if j == i:
                        continue
                    pos_idx = idxs[j]

                    # Draw K negative samples (avoiding pos)
                    negs = []
                    while len(negs) < NEG_SAMPLES:
                        neg_idx = neg_table[neg_ptr % len(neg_table)]
                        neg_ptr = (neg_ptr + 1) % len(neg_table)
                        if neg_idx != pos_idx:
                            negs.append(neg_idx)

                    # Linear LR decay
                    lr = max(LR * 0.0001,
                             LR * (1 - pair_no / (total_pairs * EPOCHS)))
                    pair_no += 1

                    v_in       = W_in[center_idx]
                    v_out_pos  = W_out[pos_idx]
                    v_out_negs = W_out[negs]

                    g_in, g_pos, g_negs = ns_update(v_in, v_out_pos, v_out_negs, lr)

                    W_in[center_idx]  += lr * g_in
                    W_out[pos_idx]    += lr * g_pos
                    W_out[negs]       += lr * g_negs

                    # Approximate loss for monitoring
                    epoch_loss += -np.log(sigmoid(v_in @ v_out_pos) + 1e-8)
                    n_updates  += 1

        avg_loss = epoch_loss / max(n_updates, 1)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"updates={n_updates:,}  elapsed={time.time()-t0:.0f}s")

    return W_in, time.time() - t0


# 5. CBOW Training
def train_cbow(sentences, word2idx, keep_prob, neg_table, window):
    V = len(word2idx)
    W_in, W_out = init_embeddings(V, EMBED_DIM)

    total_words = sum(
        sum(1 for w in s if w in word2idx) for s in sentences
    )
    word_no = 0
    neg_ptr = 0

    print(f"\n  CBOW | epochs={EPOCHS} | vocab={V:,} | dim={EMBED_DIM} | window={window}")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_updates  = 0
        random.shuffle(sentences)

        for sent in sentences:
            idxs = [word2idx[w] for w in sent
                    if w in word2idx and random.random() < keep_prob.get(w, 1.0)]
            if len(idxs) < 2:
                continue

            for i, center_idx in enumerate(idxs):
                w = random.randint(1, window)
                ctx_idxs = [idxs[j] for j in range(max(0, i - w),
                                                     min(len(idxs), i + w + 1))
                            if j != i]
                if not ctx_idxs:
                    continue

                # CBOW: mean of context embeddings
                h = W_in[ctx_idxs].mean(axis=0)   # (D,)

                # Negative samples
                negs = []
                while len(negs) < NEG_SAMPLES:
                    neg_idx = neg_table[neg_ptr % len(neg_table)]
                    neg_ptr = (neg_ptr + 1) % len(neg_table)
                    if neg_idx != center_idx:
                        negs.append(neg_idx)

                lr = max(LR * 0.0001,
                         LR * (1 - word_no / (total_words * EPOCHS)))
                word_no += 1

                v_out_pos  = W_out[center_idx]
                v_out_negs = W_out[negs]

                g_h, g_pos, g_negs = ns_update(h, v_out_pos, v_out_negs, lr)

                # Distribute gradient to all context embeddings equally
                W_in[ctx_idxs]    += lr * (g_h / len(ctx_idxs))
                W_out[center_idx] += lr * g_pos
                W_out[negs]       += lr * g_negs

                epoch_loss += -np.log(sigmoid(h @ v_out_pos) + 1e-8)
                n_updates  += 1

        avg_loss = epoch_loss / max(n_updates, 1)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"updates={n_updates:,}  elapsed={time.time()-t0:.0f}s")

    return W_in, time.time() - t0


# 6. Evaluate vectors - avg cosine similarity for known probe pairs
def eval_vectors(vectors, word2idx):
    """Quick intrinsic evaluation: avg cosine similarity for known pairs."""
    probe_pairs = [
        ("research", "study"),
        ("student",  "faculty"),
        ("phd",      "degree"),
        ("exam",     "assessment"),
        ("jodhpur",  "rajasthan"),
    ]
    norms = np.linalg.norm(vectors, axis=1) + 1e-8
    norm_vecs = vectors / norms[:, None]
    scores = []
    for w1, w2 in probe_pairs:
        if w1 in word2idx and w2 in word2idx:
            sim = float(norm_vecs[word2idx[w1]] @ norm_vecs[word2idx[w2]])
            scores.append(sim)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# 7. Nearest Neighbours (cosine similarity)
def nearest_neighbours(word, word2idx, vectors, topn=5):
    if word not in word2idx:
        print(f"  '{word}' not in vocabulary.")
        return []
    idx  = word2idx[word]
    v    = vectors[idx]
    norm = np.linalg.norm(vectors, axis=1) + 1e-8
    sims = (vectors @ v) / (norm * np.linalg.norm(v) + 1e-8)
    sims[idx] = -1   # exclude self
    top_idxs = np.argsort(sims)[::-1][:topn]
    idx2word = {i: w for w, i in word2idx.items()}
    return [(idx2word[i], round(float(sims[i]), 4)) for i in top_idxs]


# 8. Main
def main():
    print("=" * 60)
    print("  Word2Vec from Scratch (NumPy)")
    print("=" * 60)

    word2idx, idx2word, keep_prob, neg_table, sentences = build_vocab(CORPUS_PATH)

    # Save vocabulary
    vocab_path = os.path.join(MODEL_DIR, "scratch_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f)
    print(f"Vocab saved → {vocab_path}\n")

    probe_words = ["research", "student", "phd", "exam", "department"]

    # ── Window size sweep (same as 05_train_word2vec.py) ──────────────────
    print("\n" + "=" * 60)
    print("  [1] Window Size Sweep")
    print("=" * 60)

    sweep_results = []
    for window in WINDOWS:
        for arch, train_fn in [("cbow", train_cbow), ("skipgram", train_skipgram)]:
            print(f"\n{'#'*60}")
            print(f"  {arch.upper()} | window={window}")
            print(f"{'#'*60}")
            vectors, elapsed = train_fn(sentences, word2idx, keep_prob, neg_table, window)
            avg_sim = eval_vectors(vectors, word2idx)
            sweep_results.append({
                "architecture":   arch,
                "sweep":          "window",
                "window":         window,
                "vector_size":    EMBED_DIM,
                "negative":       NEG_SAMPLES,
                "vocab_size":     len(word2idx),
                "train_time_sec": round(elapsed, 2),
                "avg_probe_sim":  avg_sim,
            })
            print(f"\n  window={window} avg_probe_sim={avg_sim:.4f}  time={elapsed:.0f}s")

    # Print sweep summary table
    print("\n" + "=" * 75)
    print(f"{'Arch':10s} {'Window':>8} {'Vocab':>8} {'Time(s)':>8} {'AvgSim':>8}")
    print("-" * 75)
    for r in sweep_results:
        print(f"{r['architecture']:10s} {r['window']:>8d} {r['vocab_size']:>8,} "
              f"{r['train_time_sec']:>8.1f} {r['avg_probe_sim']:>8.4f}")
    print("=" * 75)

    # Save sweep results
    sweep_path = os.path.join(MODEL_DIR, "scratch_experiment_results.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Sweep results saved → {sweep_path}")

    # ── Best models (window=DEFAULT_WIN) ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  [2] Training best models (window={DEFAULT_WIN})")
    print("=" * 60)

    for arch, train_fn in [("skipgram", train_skipgram), ("cbow", train_cbow)]:
        print(f"\n{'#'*60}")
        print(f"  {arch.upper()}")
        print(f"{'#'*60}")

        vectors, _ = train_fn(sentences, word2idx, keep_prob, neg_table, DEFAULT_WIN)

        # Save best vectors
        out_path = os.path.join(MODEL_DIR, f"scratch_{arch}_vectors.npy")
        np.save(out_path, vectors)
        print(f"\n  Vectors saved → {out_path}  shape={vectors.shape}")

        # Quick nearest-neighbour check
        print(f"\n  Nearest neighbours:")
        for word in probe_words:
            nn = nearest_neighbours(word, word2idx, vectors)
            if nn:
                print(f"    '{word}': {[w for w, _ in nn]}")

    print("\nScratch training complete.")


if __name__ == "__main__":
    main()
