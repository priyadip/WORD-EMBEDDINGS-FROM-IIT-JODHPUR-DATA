"""
Dataset Statistics & Word Cloud
=================================================
Reads the preprocessed corpus and metadata to:
  1. Print dataset statistics:
       - Total number of documents
       - Total number of tokens
       - Vocabulary size
       - Average tokens per document
       - Average sentence length
  2. Compute top-50 most frequent words (after removing stopwords)
  3. Generate and save a Word Cloud image

Outputs:
  - Console: statistics table
  - data/output/wordcloud.png : Word Cloud visualisation
  - data/output/top_words.json: top-50 word frequency list
"""

import os
import json
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all systems)
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)


# Paths

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed", "corpus.txt")
META_PATH    = os.path.join(SCRIPT_DIR, "..", "data", "processed", "metadata.json")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "..", "data", "output")
WC_PATH      = os.path.join(OUTPUT_DIR, "wordcloud.png")
TOP_WORDS    = os.path.join(OUTPUT_DIR, "top_words.json")


# English stopwords  -  extended with domain noise
STOP_WORDS = set(stopwords.words("english")) | {
    # common single-letter tokens that may slip through
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # very common web boilerplate words
    "home", "page", "click", "link", "read", "more", "back", "top",
    "next", "prev", "previous", "menu", "search", "contact", "us",
    "site", "map", "copyright", "rights", "reserved", "iitj", "iit",
    "jodhpur",
}


def load_corpus_tokens(corpus_path: str) -> list[str]:
    """
    Read the preprocessed corpus (one sentence per line, space-separated tokens)
    and return a flat list of all tokens.
    """
    tokens = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens.extend(line.strip().split())
    return tokens


def print_statistics(meta: dict, tokens: list[str]):
    """
    Print a formatted summary of dataset statistics to the console.
    """
    n_docs  = meta["total_documents"]
    n_sents = meta["total_sentences"]
    n_toks  = meta["total_tokens"]
    n_vocab = meta["vocabulary_size"]

    avg_tok_per_doc  = n_toks  / n_docs  if n_docs  else 0
    avg_tok_per_sent = n_toks  / n_sents if n_sents else 0

    print("\n" + "=" * 55)
    print("        DATASET STATISTICS ")
    print("=" * 55)
    print(f"  {'Metric':<35} {'Value':>12}")
    print(f"  {'-'*35} {'-'*12}")
    print(f"  {'Total documents':<35} {n_docs:>12,}")
    print(f"  {'Total sentences':<35} {n_sents:>12,}")
    print(f"  {'Total tokens':<35} {n_toks:>12,}")
    print(f"  {'Vocabulary size':<35} {n_vocab:>12,}")
    print(f"  {'Avg tokens / document':<35} {avg_tok_per_doc:>12.1f}")
    print(f"  {'Avg tokens / sentence':<35} {avg_tok_per_sent:>12.1f}")
    print("=" * 55)


def get_top_words(tokens: list[str], n: int = 50) -> list[tuple[str, int]]:
    """
    Count word frequencies, excluding stopwords, and return the top-n pairs.
    """
    filtered = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    freq     = Counter(filtered)
    return freq.most_common(n)


def print_top_words(top_words: list[tuple[str, int]], n: int = 30):
    """Print the top-n words in a readable table."""
    print(f"\n  Top {n} most frequent words (stopwords excluded):")
    print(f"  {'Rank':<6} {'Word':<20} {'Count':>8}")
    print(f"  {'-'*6} {'-'*20} {'-'*8}")
    for rank, (word, count) in enumerate(top_words[:n], start=1):
        print(f"  {rank:<6} {word:<20} {count:>8,}")


def generate_wordcloud(top_words: list[tuple[str, int]], output_path: str):
    """
    Generate a Word Cloud from the frequency dictionary of top words and
    save it as a PNG image.

    The cloud uses a white background and a colour scheme suitable for
    academic reports.
    """
    freq_dict = {word: count for word, count in top_words}

    wc = WordCloud(
        width            = 1200,
        height           = 700,
        background_color = "white",
        colormap         = "viridis",       # professional colour scheme
        max_words        = 200,
        min_font_size    = 10,
        max_font_size    = 120,
        prefer_horizontal= 0.85,
        collocations     = False,           # avoid bigram duplicates
    )
    wc.generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Word Cloud  -  Most Frequent Words in IIT Jodhpur Corpus",
        fontsize=16, fontweight="bold", pad=15
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Word Cloud saved to: {output_path}")


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CORPUS_PATH):
        print(f"[ERROR] Corpus not found at {CORPUS_PATH}.")
        print("Run 03_preprocess.py first.")
        return

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("Loading corpus tokens …")
    tokens = load_corpus_tokens(CORPUS_PATH)

    print_statistics(meta, tokens)

    top_words = get_top_words(tokens, n=200)
    print_top_words(top_words, n=30)

    # Save top-words list as JSON for later reference
    with open(TOP_WORDS, "w", encoding="utf-8") as f:
        json.dump([{"word": w, "count": c} for w, c in top_words], f, indent=2)
    print(f"\n  [OK] Top words saved to: {TOP_WORDS}")

    generate_wordcloud(top_words, WC_PATH)

    print("\nAll outputs written to data/output/")


if __name__ == "__main__":
    main()
