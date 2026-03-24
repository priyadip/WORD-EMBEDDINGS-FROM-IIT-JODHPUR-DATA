"""
Word2Vec Model Training
================================
Trains CBOW and Skip-gram models with Negative Sampling using Gensim.
Experiments with:
  (i)   Embedding dimension   : 100, 200, 300
  (ii)  Context window size   : 3, 5, 10
  (iii) Number of negative samples : 5, 10, 15

Best models (dim=300, window=5, neg=10) are saved for Tasks 3 & 4.
All experiment results are saved to data/models/experiment_results.json.
"""

import os
import json
import time
import logging

from gensim.models import Word2Vec

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "corpus.txt")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class CorpusSentences:
    """Iterates corpus.txt yielding token lists  -  can be iterated multiple times."""
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


DIMENSIONS   = [100, 200, 300]
WINDOWS      = [3, 5, 10]
NEG_SAMPLES  = [5, 10, 15]

# Fixed defaults used when sweeping one hyperparameter at a time
DEFAULT_DIM  = 300
DEFAULT_WIN  = 5
DEFAULT_NEG  = 10
EPOCHS       = 10
MIN_COUNT    = 5       # ignore words with total frequency < 5
WORKERS      = 4


def train_model(sg, dim, window, negative, sentences, label):
    """Train one Word2Vec model and return (model, elapsed_seconds)."""
    arch = "skipgram" if sg else "cbow"
    print(f"\n  Training {arch.upper():8s} | dim={dim:3d} win={window:2d} neg={negative:2d}  [{label}]")
    t0 = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        negative=negative,
        sg=sg,            # 0 = CBOW, 1 = Skip-gram
        hs=0,             # use negative sampling (not hierarchical softmax)
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
        seed=42,
    )
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s  |  vocab={len(model.wv):,}")
    return model, elapsed


def eval_model(model):
    """Quick intrinsic evaluation: avg cosine similarity for known pairs."""
    probe_pairs = [
        ("research", "study"),
        ("student", "faculty"),
        ("phd", "degree"),
        ("exam", "assessment"),
        ("jodhpur", "rajasthan"),
    ]
    scores = []
    wv = model.wv
    for w1, w2 in probe_pairs:
        if w1 in wv and w2 in wv:
            scores.append(float(wv.similarity(w1, w2)))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def run_experiments(sentences):
    """
    Sweep one hyperparameter at a time (others fixed at defaults).
    Returns list of result dicts.
    """
    results = []

    configs = []

    # Sweep embedding dimension
    for dim in DIMENSIONS:
        configs.append(dict(dim=dim, window=DEFAULT_WIN, neg=DEFAULT_NEG, sweep="dimension"))

    # Sweep context window
    for win in WINDOWS:
        if win == DEFAULT_WIN:
            continue   # already covered above
        configs.append(dict(dim=DEFAULT_DIM, window=win, neg=DEFAULT_NEG, sweep="window"))

    # Sweep negative samples
    for neg in NEG_SAMPLES:
        if neg == DEFAULT_NEG:
            continue
        configs.append(dict(dim=DEFAULT_DIM, window=DEFAULT_WIN, neg=neg, sweep="negative_samples"))

    for cfg in configs:
        for sg, arch in [(0, "cbow"), (1, "skipgram")]:
            label = f"{arch}_dim{cfg['dim']}_win{cfg['window']}_neg{cfg['neg']}"
            model, elapsed = train_model(
                sg=sg,
                dim=cfg["dim"],
                window=cfg["window"],
                negative=cfg["neg"],
                sentences=sentences,
                label=label,
            )
            avg_sim = eval_model(model)
            results.append({
                "architecture":    arch,
                "sweep":           cfg["sweep"],
                "vector_size":     cfg["dim"],
                "window":          cfg["window"],
                "negative":        cfg["neg"],
                "vocab_size":      len(model.wv),
                "train_time_sec":  round(elapsed, 2),
                "avg_probe_sim":   avg_sim,
                "label":           label,
            })

    return results


def main():
    print("=" * 60)
    print("TASK 2  -  Word2Vec Training")
    print("=" * 60)

    sentences = CorpusSentences(CORPUS_PATH)

    print("\n[1] Running hyperparameter sweep …")
    results = run_experiments(sentences)

    results_path = os.path.join(MODEL_DIR, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Experiment results saved → {results_path}")

    print("\n" + "=" * 75)
    print(f"{'Arch':8s} {'Sweep':16s} {'Dim':>4} {'Win':>4} {'Neg':>4} {'Vocab':>7} {'Time(s)':>8} {'AvgSim':>7}")
    print("-" * 75)
    for r in results:
        print(f"{r['architecture']:8s} {r['sweep']:16s} {r['vector_size']:>4d} "
              f"{r['window']:>4d} {r['negative']:>4d} {r['vocab_size']:>7,} "
              f"{r['train_time_sec']:>8.1f} {r['avg_probe_sim']:>7.4f}")
    print("=" * 75)

    print("\n[2] Training best models (dim=300, win=5, neg=10, epochs=10) …")
    for sg, arch in [(0, "cbow"), (1, "skipgram")]:
        model, _ = train_model(
            sg=sg, dim=300, window=5, negative=10,
            sentences=sentences, label=f"best_{arch}"
        )
        path = os.path.join(MODEL_DIR, f"{arch}_best.model")
        model.save(path)
        print(f"  Saved → {path}")



if __name__ == "__main__":
    main()
