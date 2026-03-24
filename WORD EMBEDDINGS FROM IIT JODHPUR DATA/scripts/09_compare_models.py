"""
COMPARISON: Gensim Word2Vec vs From-Scratch Word2Vec (NumPy)
=============================================================
Compares both implementations on:
  1. Nearest neighbours for probe words
  2. Analogy accuracy
  3. Embedding space similarity (CKA score)
  4. Vocabulary coverage

Output:
  data/models/comparison_results.json
  data/figures/comparison_nn.png
  data/figures/comparison_tsne.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "..", "data", "models")
FIG_DIR    = os.path.join(SCRIPT_DIR, "..", "data", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PROBE_WORDS = ["research", "student", "phd", "exam", "department",
               "faculty", "course", "semester", "admission", "jodhpur"]

ANALOGIES = [
    {"label": "UG : btech :: PG : ?",
     "positive": ["pg", "btech"], "negative": ["ug"],
     "expected": ["mtech", "msc", "ma", "postgraduate"],
     "explain": "If UG corresponds to BTech, what does PG correspond to?"},
    {"label": "professor : teaching :: researcher : ?",
     "positive": ["researcher", "teaching"], "negative": ["professor"],
     "expected": ["research", "investigation", "study"],
     "explain": "If a professor's role is teaching, what is a researcher's role?"},
    {"label": "mtech : postgraduate :: btech : ?",
     "positive": ["btech", "postgraduate"], "negative": ["mtech"],
     "expected": ["undergraduate", "ug", "graduate", "bachelor"],
     "explain": "If mtech is postgraduate, what level is btech?"},
    {"label": "mishra : assistant :: das : ?",
     "positive": ["assistant", "das"], "negative": ["mishra"],
     "expected": ["associate"],
     "explain": "If Anand Mishra holds Assistant Professor rank, what rank does Debasis Das hold? (expected: associate)"},
    {"label": "gaurav : harit :: debasis : ?",
     "positive": ["debasis", "harit"], "negative": ["gaurav"],
     "expected": ["das"],
     "explain": "If 'gaurav' pairs with surname 'harit', what surname does 'debasis' pair with? (IIT Jodhpur faculty: Gaurav Harit, Debasis Das)"},
    {"label": "gate : pg :: jee : ?",
     "positive": ["jee", "pg"], "negative": ["gate"],
     "expected": ["ug", "undergraduate"],
     "explain": "If GATE is the entrance for PG programs, what is the entrance for UG programs?"},
    {"label": "jee : btech :: gate : ?",
     "positive": ["gate", "btech"], "negative": ["jee"],
     "expected": ["mtech", "postgraduate"],
     "explain": "If JEE leads to BTech, what degree does GATE lead to?"},
    {"label": "iitm : madras :: iitj : ?",
     "positive": ["iitj", "madras"], "negative": ["iitm"],
     "expected": ["jodhpur"],
     "explain": "If IIT Madras is in Madras (Chennai), IIT Jodhpur is in which city?"},
    {"label": "iitb : bombay :: iitd : ?",
     "positive": ["iitd", "bombay"], "negative": ["iitb"],
     "expected": ["delhi"],
     "explain": "If IIT Bombay is in Bombay (Mumbai), IIT Delhi is in which city?"},
    {"label": "exam : written :: viva : ?",
     "positive": ["viva", "written"], "negative": ["exam"],
     "expected": ["voce", "oral"],
     "explain": "If a regular exam is written, what mode is a viva (viva voce)?"},
    {"label": "conference : paper :: journal : ?",
     "positive": ["journal", "paper"], "negative": ["conference"],
     "expected": ["article", "articles"],
     "explain": "If a conference publishes papers, what does a journal publish?"},
    {"label": "conference : proceedings :: journal : ?",
     "positive": ["journal", "proceedings"], "negative": ["conference"],
     "expected": ["issue", "articles"],
     "explain": "If a conference has proceedings, what does a journal have?"},
    {"label": "international : conference :: national : ?",
     "positive": ["national", "conference"], "negative": ["international"],
     "expected": ["symposium", "conference"],
     "explain": "If a large-scale event is an international conference, what is a smaller national-level event called?"},
    {"label": "anand : mishra :: debasis : ?",
     "positive": ["debasis", "mishra"], "negative": ["anand"],
     "expected": ["das"],
     "explain": "If 'anand' pairs with surname 'mishra', what surname does 'debasis' pair with? (IIT Jodhpur faculty: Anand Mishra, Debasis Das)"},
    {"label": "anand : mishra :: gaurav : ?",
     "positive": ["gaurav", "mishra"], "negative": ["anand"],
     "expected": ["harit"],
     "explain": "If 'anand' pairs with surname 'mishra', what surname does 'gaurav' pair with? (IIT Jodhpur faculty: Anand Mishra, Gaurav Harit)"},
]


# Helper: Cosine similarity wrappers

class GensimWrapper:
    """Thin wrapper so Gensim and Scratch share the same interface."""
    def __init__(self, model_path, label):
        self.label = label
        self.model = Word2Vec.load(model_path)
        self.wv    = self.model.wv

    def neighbours(self, word, topn=5):
        if word not in self.wv:
            return []
        return [(w, round(float(s), 4)) for w, s in self.wv.most_similar(word, topn=topn)]

    def analogy(self, positive, negative, topn=5):
        missing = [w for w in positive + negative if w not in self.wv]
        if missing:
            return None, missing
        res = self.wv.most_similar(positive=positive, negative=negative, topn=topn)
        return [(w, round(float(s), 4)) for w, s in res], []

    def vector(self, word):
        return self.wv[word] if word in self.wv else None

    def vocab(self):
        return set(self.wv.key_to_index.keys())


class ScratchWrapper:
    """Wrapper for NumPy-based scratch vectors."""
    def __init__(self, vectors_path, vocab_path, label):
        self.label   = label
        self.vectors = np.load(vectors_path)
        with open(vocab_path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        # Pre-normalise for fast cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8
        self.norm_vectors = self.vectors / norms

    def _cosine_topn(self, query_vec, topn, exclude_idxs=set()):
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        sims   = self.norm_vectors @ q_norm
        for idx in exclude_idxs:
            sims[idx] = -1
        top_idxs = np.argsort(sims)[::-1][:topn + len(exclude_idxs)]
        results  = [(self.idx2word[i], round(float(sims[i]), 4))
                    for i in top_idxs if i not in exclude_idxs]
        return results[:topn]

    def neighbours(self, word, topn=5):
        if word not in self.word2idx:
            return []
        idx = self.word2idx[word]
        return self._cosine_topn(self.vectors[idx], topn, exclude_idxs={idx})

    def analogy(self, positive, negative, topn=5):
        missing = [w for w in positive + negative if w not in self.word2idx]
        if missing:
            return None, missing
        query = (sum(self.vectors[self.word2idx[w]] for w in positive) -
                 sum(self.vectors[self.word2idx[w]] for w in negative))
        exclude = {self.word2idx[w] for w in positive + negative}
        res = self._cosine_topn(query, topn, exclude_idxs=exclude)
        return res, []

    def vector(self, word):
        return self.vectors[self.word2idx[word]] if word in self.word2idx else None

    def vocab(self):
        return set(self.word2idx.keys())


# CKA (Centred Kernel Alignment)  -  measures embedding space similarity
def cka(X, Y):
    """
    Linear CKA between two embedding matrices X (n×d1) and Y (n×d2).
    Range [0, 1]  -  1 means identical structure, 0 means completely different.
    """
    def centre(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_X = centre(X @ X.T)
    K_Y = centre(Y @ Y.T)
    return np.sum(K_X * K_Y) / (np.linalg.norm(K_X, 'fro') * np.linalg.norm(K_Y, 'fro') + 1e-8)


# Nearest neighbours comparison plot
def plot_nn_comparison(models, probe_words):
    """Heatmap: for each probe word, how many top-5 neighbours are shared?"""
    model_names = [m.label for m in models]
    n_models    = len(models)
    n_words     = len(probe_words)
    overlap     = np.zeros((n_words,), dtype=float)

    nn_data = {}
    for m in models:
        nn_data[m.label] = {}
        for w in probe_words:
            nn_data[m.label][w] = [x for x, _ in m.neighbours(w, topn=5)]

    # Compute pairwise overlap for each word
    for wi, word in enumerate(probe_words):
        sets = [set(nn_data[m.label][word]) for m in models]
        if all(sets):
            common = sets[0].intersection(*sets[1:])
            overlap[wi] = len(common) / 5.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: print nearest neighbours as a table (bar chart)
    ax = axes[0]
    ax.barh(probe_words, overlap, color="#2980b9", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of shared top-5 neighbours (all models)")
    ax.set_title("Neighbour Agreement: Gensim vs Scratch", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(overlap):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

    # Right: show top-3 neighbours per model per word
    ax = axes[1]
    ax.axis("off")
    rows = [["Word"] + [m.label for m in models]]
    for word in probe_words:
        row = [word]
        for m in models:
            nn = nn_data[m.label].get(word, [])
            row.append(", ".join(nn[:3]))
        rows.append(row)

    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.6)
    ax.set_title("Top-3 Neighbours per Model", fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "comparison_nn.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# t-SNE comparison for shared vocabulary
def plot_tsne_comparison(models, words_to_plot):
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 7))
    if len(models) == 1:
        axes = [axes]

    colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#e67e22",
              "#16a085", "#d35400", "#7f8c8d", "#2c3e50", "#f39c12"]

    for ax, model in zip(axes, models):
        vecs  = []
        words = []
        for w in words_to_plot:
            v = model.vector(w)
            if v is not None:
                vecs.append(v)
                words.append(w)

        if len(vecs) < 5:
            ax.set_title(f"{model.label}  -  too few words")
            continue

        X = np.array(vecs)
        perp = min(10, len(X) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=1000, learning_rate="auto", init="pca")
        coords = tsne.fit_transform(X)

        for i, (word, (x, y)) in enumerate(zip(words, coords)):
            ax.scatter(x, y, color=colors[i % len(colors)], s=80, zorder=3)
            ax.annotate(word, (x, y), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")

        ax.set_title(f"t-SNE  -  {model.label}", fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#fafafa")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "comparison_tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# Main comparison
def main():
    print("=" * 65)
    print("  MODEL COMPARISON: Gensim Word2Vec vs From-Scratch (NumPy)")
    print("=" * 65)

    scratch_vocab = os.path.join(MODEL_DIR, "scratch_vocab.json")

    # Load all models
    models = []
    model_configs = [
        ("Gensim-CBOW",     "gensim", os.path.join(MODEL_DIR, "cbow_best.model"),          None),
        ("Gensim-Skipgram", "gensim", os.path.join(MODEL_DIR, "skipgram_best.model"),       None),
        ("Scratch-CBOW",    "scratch", os.path.join(MODEL_DIR, "scratch_cbow_vectors.npy"),     scratch_vocab),
        ("Scratch-Skipgram","scratch", os.path.join(MODEL_DIR, "scratch_skipgram_vectors.npy"), scratch_vocab),
    ]

    for label, kind, path, vocab_path in model_configs:
        if not os.path.exists(path):
            print(f"  [SKIP] {label}  -  file not found: {path}")
            continue
        if kind == "gensim":
            models.append(GensimWrapper(path, label))
        else:
            models.append(ScratchWrapper(path, vocab_path, label))
        print(f"  Loaded: {label}  |  vocab={len(models[-1].vocab()):,}")

    if not models:
        print("No models found. Run 05 and 08 first.")
        return

    results = {}

    print("\n[1] Vocabulary sizes")
    results["vocab_sizes"] = {m.label: len(m.vocab()) for m in models}
    for label, size in results["vocab_sizes"].items():
        print(f"    {label:25s}: {size:,}")

    # Shared vocab across all
    common_vocab = set.intersection(*[m.vocab() for m in models])
    print(f"    Common vocab (all models) : {len(common_vocab):,}")
    results["common_vocab_size"] = len(common_vocab)

    print("\n[2] Nearest Neighbours (top-5)")
    nn_results = {}
    for word in PROBE_WORDS:
        nn_results[word] = {}
        print(f"\n  '{word}':")
        for m in models:
            nn = m.neighbours(word, topn=5)
            nn_results[word][m.label] = nn
            top = [w for w, _ in nn] if nn else ["(not in vocab)"]
            print(f"    {m.label:25s}: {top}")
    results["nearest_neighbours"] = nn_results

    print("\n[3] Analogy Experiments")
    analogy_results = {}
    for analogy in ANALOGIES:
        label   = analogy["label"]
        correct = analogy["expected"]
        analogy_results[label] = {}
        print(f"\n  {label}")
        if analogy.get("explain"):
            print(f"  ({analogy['explain']})")
        for m in models:
            answers, missing = m.analogy(analogy["positive"], analogy["negative"])
            if answers is None:
                print(f"    {m.label:25s}: SKIP (missing: {missing})")
                analogy_results[label][m.label] = {"top": None, "hit": False}
            else:
                top = answers[0][0] if answers else ""
                hit = top in correct
                mark = "HIT" if hit else "MISS"
                print(f"    {m.label:25s}: {[w for w,_ in answers[:3]]}  {mark}")
                analogy_results[label][m.label] = {"top": top, "hit": hit,
                                                    "answers": answers[:3]}
    results["analogies"] = analogy_results

    print("\n[4] CKA Embedding Space Similarity")
    # Use common vocab words (up to 500 for speed)
    sample_words = list(common_vocab)[:500]
    cka_matrix   = {}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            X = np.array([m1.vector(w) for w in sample_words if m1.vector(w) is not None and m2.vector(w) is not None])
            Y = np.array([m2.vector(w) for w in sample_words if m1.vector(w) is not None and m2.vector(w) is not None])
            if len(X) < 10:
                continue
            score = cka(X, Y)
            key   = f"{m1.label} ↔ {m2.label}"
            cka_matrix[key] = round(float(score), 4)
            print(f"    {key:50s}: CKA={score:.4f}")
    results["cka"] = cka_matrix

    print("\n[5] Generating plots …")
    plot_nn_comparison(models, PROBE_WORDS)

    tsne_words = ["research", "student", "phd", "exam", "faculty",
                  "course", "semester", "admission", "laboratory", "thesis",
                  "jodhpur", "rajasthan", "professor", "mtech", "btech"]
    plot_tsne_comparison(models, tsne_words)

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Model':25s} {'Vocab':>8}  {'Analogy Hits':>14}")
    print(f"  {'-'*50}")
    for m in models:
        hits  = sum(1 for a in analogy_results.values()
                    if a.get(m.label, {}).get("hit", False))
        total = len(ANALOGIES)
        print(f"  {m.label:25s} {len(m.vocab()):>8,}  {hits}/{total}")

    print(f"\n  CKA scores (0=different, 1=identical structure):")
    for pair, score in cka_matrix.items():
        print(f"    {pair}: {score:.4f}")

    # ── [6] Window Sweep Experiment Comparison ────────────────────────────
    print("\n" + "=" * 65)
    print("  [6] Window Sweep Comparison: Gensim vs Scratch")
    print("=" * 65)

    gensim_exp_path  = os.path.join(MODEL_DIR, "experiment_results.json")
    scratch_exp_path = os.path.join(MODEL_DIR, "scratch_experiment_results.json")

    sweep_comparison = {}

    gensim_sweep  = []
    scratch_sweep = []

    if os.path.exists(gensim_exp_path):
        with open(gensim_exp_path, "r") as f:
            gensim_sweep = [r for r in json.load(f) if r.get("sweep") == "window"]
    else:
        print("  [SKIP] experiment_results.json not found. Run 05_train_word2vec.py first.")

    if os.path.exists(scratch_exp_path):
        with open(scratch_exp_path, "r") as f:
            scratch_sweep = json.load(f)
    else:
        print("  [SKIP] scratch_experiment_results.json not found. Run 08_word2vec_scratch.py first.")

    if gensim_sweep or scratch_sweep:
        print(f"\n  {'Impl':10s} {'Arch':10s} {'Window':>8} {'AvgSim':>8} {'Time(s)':>8}")
        print("  " + "-" * 50)

        for r in gensim_sweep:
            print(f"  {'Gensim':10s} {r['architecture']:10s} {r['window']:>8d} "
                  f"{r['avg_probe_sim']:>8.4f} {r['train_time_sec']:>8.1f}")

        for r in scratch_sweep:
            print(f"  {'Scratch':10s} {r['architecture']:10s} {r['window']:>8d} "
                  f"{r['avg_probe_sim']:>8.4f} {r['train_time_sec']:>8.1f}")

        # Side-by-side comparison per window per arch
        print(f"\n  AvgSim difference (Gensim - Scratch) per window:")
        print(f"  {'Arch':10s} {'Window':>8} {'Gensim':>8} {'Scratch':>8} {'Diff':>8}")
        print("  " + "-" * 50)
        for arch in ["cbow", "skipgram"]:
            for win in [3, 5, 10]:
                g = next((r["avg_probe_sim"] for r in gensim_sweep
                          if r["architecture"] == arch and r["window"] == win), None)
                s = next((r["avg_probe_sim"] for r in scratch_sweep
                          if r["architecture"] == arch and r["window"] == win), None)
                if g is not None and s is not None:
                    diff = round(g - s, 4)
                    print(f"  {arch:10s} {win:>8d} {g:>8.4f} {s:>8.4f} {diff:>+8.4f}")

        sweep_comparison = {
            "gensim_window_sweep":  gensim_sweep,
            "scratch_window_sweep": scratch_sweep,
        }

    results["window_sweep_comparison"] = sweep_comparison

    # Save results
    out_path = os.path.join(MODEL_DIR, "comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")
    print("\nComparison complete.")


if __name__ == "__main__":
    main()
