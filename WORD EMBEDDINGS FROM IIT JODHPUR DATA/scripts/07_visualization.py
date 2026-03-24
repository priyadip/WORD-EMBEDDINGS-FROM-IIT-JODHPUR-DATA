"""
TASK 4: Embedding Visualization
=================================
Projects word embeddings into 2D using PCA and t-SNE.
Visualizes semantic clusters for both CBOW and Skip-gram models.

Word groups visualized:
  - Academic levels   : phd, mtech, btech, msc, postgraduate, undergraduate
  - Research terms    : research, publication, journal, conference, thesis, dissertation
  - People roles      : professor, student, faculty, researcher, lecturer, coordinator
  - Infrastructure    : library, hostel, laboratory, campus, canteen, workshop
  - Administrative    : admission, registration, examination, semester, curriculum, syllabus

Output:
  data/figures/pca_cbow.png
  data/figures/pca_skipgram.png
  data/figures/tsne_cbow.png
  data/figures/tsne_skipgram.png
  data/figures/pca_comparison.png      ← CBOW vs Skip-gram side-by-side
  data/figures/tsne_comparison.png
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(SCRIPT_DIR, "..", "data", "models")
FIG_DIR     = os.path.join(SCRIPT_DIR, "..", "data", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

WORD_GROUPS = {
    "Academic Levels":   {
        "words":  ["phd", "mtech", "btech", "msc", "postgraduate", "undergraduate",
                   "doctorate", "diploma", "graduate", "degree"],
        "color":  "#e74c3c",
        "marker": "o",
    },
    "Research":          {
        "words":  ["research", "publication", "journal", "conference", "thesis",
                   "dissertation", "experiment", "analysis", "methodology", "findings"],
        "color":  "#2980b9",
        "marker": "s",
    },
    "People & Roles":    {
        "words":  ["professor", "student", "faculty", "researcher", "lecturer",
                   "coordinator", "supervisor", "scholar", "instructor", "dean"],
        "color":  "#27ae60",
        "marker": "^",
    },
    "Infrastructure":    {
        "words":  ["library", "hostel", "laboratory", "campus", "canteen",
                   "workshop", "auditorium", "classroom", "facility", "building"],
        "color":  "#8e44ad",
        "marker": "D",
    },
    "Administration":    {
        "words":  ["admission", "registration", "examination", "semester",
                   "curriculum", "syllabus", "regulation", "assessment",
                   "attendance", "placement"],
        "color":  "#e67e22",
        "marker": "P",
    },
}


def collect_vectors(wv, groups):
    """
    For each word group, collect words present in vocabulary.
    Returns (words_list, vectors_array, group_labels_list).
    """
    all_words   = []
    all_vectors = []
    all_groups  = []

    for group_name, cfg in groups.items():
        for word in cfg["words"]:
            if word in wv:
                all_words.append(word)
                all_vectors.append(wv[word])
                all_groups.append(group_name)

    return all_words, np.array(all_vectors), all_groups


def plot_embeddings(coords, words, groups, title, ax):
    """Scatter plot of 2D embeddings with group colours and word labels."""
    for group_name, cfg in WORD_GROUPS.items():
        idxs = [i for i, g in enumerate(groups) if g == group_name]
        if not idxs:
            continue
        xs = coords[idxs, 0]
        ys = coords[idxs, 1]
        ax.scatter(xs, ys,
                   c=cfg["color"],
                   marker=cfg["marker"],
                   s=80,
                   label=group_name,
                   alpha=0.85,
                   edgecolors="white",
                   linewidths=0.5)
        for i, idx in enumerate(idxs):
            ax.annotate(
                words[idx],
                (coords[idx, 0], coords[idx, 1]),
                fontsize=7.5,
                alpha=0.9,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Component 1", fontsize=9)
    ax.set_ylabel("Component 2", fontsize=9)
    ax.legend(fontsize=7, loc="best", framealpha=0.7)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#fafafa")


def project_pca(vectors):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(vectors)


def project_tsne(vectors):
    perp = min(30, len(vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    return tsne.fit_transform(vectors)


def visualize_model(model_path, arch_name):
    """Generate PCA and t-SNE plots for one model. Returns (words, groups, pca_coords, tsne_coords)."""
    print(f"\n  Visualizing {arch_name.upper()} …")
    model = Word2Vec.load(model_path)
    wv    = model.wv

    words, vectors, groups = collect_vectors(wv, WORD_GROUPS)
    print(f"    Words found in vocab: {len(words)} / {sum(len(g['words']) for g in WORD_GROUPS.values())}")

    if len(words) < 5:
        print("    [SKIP] Too few words in vocabulary.")
        return None

    pca_coords = project_pca(vectors)
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_embeddings(pca_coords, words, groups, f"PCA  -  {arch_name.upper()} Word Embeddings", ax)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"pca_{arch_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved → {path}")

    tsne_coords = project_tsne(vectors)
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_embeddings(tsne_coords, words, groups, f"t-SNE  -  {arch_name.upper()} Word Embeddings", ax)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"tsne_{arch_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved → {path}")

    return words, groups, pca_coords, tsne_coords


def comparison_plot(results, method):
    """Side-by-side CBOW vs Skip-gram plot for one projection method."""
    archs = [r[0] for r in results]
    if len(archs) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, (arch_name, words, groups, pca_coords, tsne_coords) in zip(axes, results):
        coords = pca_coords if method == "pca" else tsne_coords
        plot_embeddings(coords, words, groups,
                        f"{method.upper()}  -  {arch_name.upper()}", ax)

    fig.suptitle(
        f"{method.upper()}: CBOW vs Skip-gram Embedding Comparison",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"{method}_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot saved → {path}")


def main():
    print("=" * 60)
    print("TASK 4  -  Embedding Visualization")
    print("=" * 60)

    results = []   # (arch_name, words, groups, pca_coords, tsne_coords)

    for arch in ["cbow", "skipgram"]:
        model_path = os.path.join(MODEL_DIR, f"{arch}_best.model")
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {arch}_best.model not found. Run 05_train_word2vec.py first.")
            continue
        out = visualize_model(model_path, arch)
        if out is not None:
            words, groups, pca_coords, tsne_coords = out
            results.append((arch, words, groups, pca_coords, tsne_coords))

    if len(results) == 2:
        comparison_plot(results, "pca")
        comparison_plot(results, "tsne")


    print(f"Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
