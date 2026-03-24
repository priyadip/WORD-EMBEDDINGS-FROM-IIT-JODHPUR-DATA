"""
Semantic Analysis
==========================
Using the trained CBOW and Skip-gram models:

  1. Top-5 nearest neighbours (cosine similarity) for:
       research, student, phd, exam, department

  2. Analogy experiments (word arithmetic: a - b + c = ?):
       UG : btech :: PG : ?
       mtech : postgraduate :: btech : ?
       gate : pg :: jee : ?
       jee : btech :: gate : ?
       iitm : madras :: iitj : ?
       iitb : bombay :: iitd : ?
       exam : written :: viva : ?
       conference : paper :: journal : ?
       conference : proceedings :: journal : ?
       international : conference :: national : ?
       mishra : assistant :: das : ?
       gaurav : harit :: debasis : ?
       anand : mishra :: debasis : ?
       anand : mishra :: gaurav : ?
       professor : teaching :: researcher : ?

Results are printed to console and saved to data/models/semantic_analysis.json.
"""

import os
import json

from gensim.models import Word2Vec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "..", "data", "models")
OUT_PATH   = os.path.join(MODEL_DIR, "semantic_analysis.json")


PROBE_WORDS = ["research", "student", "phd", "exam", "department"]

# Format: (label, positive=[a,c], negative=[b])
# i.e.  b : a :: c : ?   →  result ≈ a - b + c
ANALOGIES = [
    {
        "label":      "UG : btech :: PG : ?",
        "positive":   ["pg", "btech"],
        "negative":   ["ug"],
        "explain":    "If UG corresponds to BTech, what does PG correspond to?",
    },
    {
        "label":      "professor : teaching :: researcher : ?",
        "positive":   ["researcher", "teaching"],
        "negative":   ["professor"],
        "explain":    "If a professor's role is teaching, what is a researcher's role?",
    },
    {
        "label":      "mtech : postgraduate :: btech : ?",
        "positive":   ["btech", "postgraduate"],
        "negative":   ["mtech"],
        "explain":    "If mtech is postgraduate, what level is btech?",
    },
    {
        "label":      "mishra : assistant :: das : ?",
        "positive":   ["assistant", "das"],
        "negative":   ["mishra"],
        "explain":    "If Anand Mishra holds Assistant Professor rank, what rank does Debasis Das hold? (expected: associate)",
    },
    {
        "label":      "gaurav : harit :: debasis : ?",
        "positive":   ["debasis", "harit"],
        "negative":   ["gaurav"],
        "explain":    "If 'gaurav' pairs with surname 'harit', what surname does 'debasis' pair with? (IIT Jodhpur faculty: Gaurav Harit, Debasis Das)",
    },
    {
        "label":      "gate : pg :: jee : ?",
        "positive":   ["jee", "pg"],
        "negative":   ["gate"],
        "explain":    "If GATE is the entrance for PG programs, what is the entrance for UG programs?",
    },
    {
        "label":      "jee : btech :: gate : ?",
        "positive":   ["gate", "btech"],
        "negative":   ["jee"],
        "explain":    "If JEE leads to BTech, what degree does GATE lead to?",
    },
    {
        "label":      "iitm : madras :: iitj : ?",
        "positive":   ["iitj", "madras"],
        "negative":   ["iitm"],
        "explain":    "If IIT Madras is in Madras (Chennai), IIT Jodhpur is in which city?",
    },
    {
        "label":      "iitb : bombay :: iitd : ?",
        "positive":   ["iitd", "bombay"],
        "negative":   ["iitb"],
        "explain":    "If IIT Bombay is in Bombay (Mumbai), IIT Delhi is in which city?",
    },
    {
        "label":      "exam : written :: viva : ?",
        "positive":   ["viva", "written"],
        "negative":   ["exam"],
        "explain":    "If a regular exam is written, what mode is a viva (viva voce)?",
    },
    {
        "label":      "conference : paper :: journal : ?",
        "positive":   ["journal", "paper"],
        "negative":   ["conference"],
        "explain":    "If a conference publishes papers, what does a journal publish?",
    },
    {
        "label":      "conference : proceedings :: journal : ?",
        "positive":   ["journal", "proceedings"],
        "negative":   ["conference"],
        "explain":    "If a conference has proceedings, what does a journal have?",
    },
    {
        "label":      "international : conference :: national : ?",
        "positive":   ["national", "conference"],
        "negative":   ["international"],
        "explain":    "If a large-scale event is an international conference, what is a smaller national-level event called?",
    },
    {
        "label":      "anand : mishra :: debasis : ?",
        "positive":   ["debasis", "mishra"],
        "negative":   ["anand"],
        "explain":    "If 'anand' pairs with surname 'mishra', what surname does 'debasis' pair with? (IIT Jodhpur faculty: Anand Mishra, Debasis Das)",
    },
    {
        "label":      "anand : mishra :: gaurav : ?",
        "positive":   ["gaurav", "mishra"],
        "negative":   ["anand"],
        "explain":    "If 'anand' pairs with surname 'mishra', what surname does 'gaurav' pair with? (IIT Jodhpur faculty: Anand Mishra, Gaurav Harit)",
    },
]


def nearest_neighbours(wv, word, topn=5):
    """Return top-N most similar words with cosine similarity scores."""
    if word not in wv:
        return None
    return [(w, round(float(s), 4)) for w, s in wv.most_similar(word, topn=topn)]


def solve_analogy(wv, positive, negative, topn=5):
    """Solve word analogy using vector arithmetic. Returns top-N results."""
    # Check all words exist
    missing = [w for w in positive + negative if w not in wv]
    if missing:
        return None, missing
    results = wv.most_similar(positive=positive, negative=negative, topn=topn)
    return [(w, round(float(s), 4)) for w, s in results], []


def analyse_model(model_path, arch_name):
    """Run all semantic analyses on one model. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"  Model: {arch_name.upper()}")
    print(f"{'='*60}")

    model = Word2Vec.load(model_path)
    wv    = model.wv
    print(f"  Vocab size: {len(wv):,}")

    result = {"architecture": arch_name, "vocab_size": len(wv)}

    print(f"\n  [1] Top-5 Nearest Neighbours")
    print(f"  {'-'*55}")
    nn_results = {}
    for word in PROBE_WORDS:
        neighbours = nearest_neighbours(wv, word)
        if neighbours is None:
            print(f"  '{word}': NOT IN VOCABULARY")
            nn_results[word] = None
        else:
            print(f"\n  '{word}':")
            for i, (w, s) in enumerate(neighbours, 1):
                print(f"    {i}. {w:20s}  sim={s:.4f}")
            nn_results[word] = neighbours
    result["nearest_neighbours"] = nn_results

    print(f"\n  [2] Analogy Experiments")
    print(f"  {'-'*55}")
    analogy_results = []
    for analogy in ANALOGIES:
        print(f"\n  {analogy['label']}")
        if analogy.get("explain"):
            print(f"  ({analogy['explain']})")
        answers, missing = solve_analogy(wv, analogy["positive"], analogy["negative"])
        if answers is None:
            print(f"  [SKIP] Missing words: {missing}")
            entry = {**analogy, "answers": None, "missing_words": missing}
        else:
            print(f"  Top answers:")
            for i, (w, s) in enumerate(answers, 1):
                print(f"    {i}. {w:20s}  sim={s:.4f}")
            entry = {**analogy, "answers": answers, "missing_words": []}

        analogy_results.append(entry)
    result["analogies"] = analogy_results

    return result



def main():
    print("=" * 60)
    print("TASK 3  -  Semantic Analysis")
    print("=" * 60)

    all_results = []
    for arch in ["cbow", "skipgram"]:
        model_path = os.path.join(MODEL_DIR, f"{arch}_best.model")
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {arch}_best.model not found. Run 05_train_word2vec.py first.")
            continue
        res = analyse_model(model_path, arch)
        all_results.append(res)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved → {OUT_PATH}")

    if len(all_results) == 2:
        print("\n" + "=" * 60)
        print("  CBOW vs Skip-gram   -   Nearest Neighbours Comparison")
        print("=" * 60)
        cbow_nn = all_results[0]["nearest_neighbours"]
        sg_nn   = all_results[1]["nearest_neighbours"]
        for word in PROBE_WORDS:
            print(f"\n  '{word}'")
            cbow_top = [w for w, _ in (cbow_nn.get(word) or [])]
            sg_top   = [w for w, _ in (sg_nn.get(word) or [])]
            print(f"    CBOW     : {cbow_top}")
            print(f"    Skip-gram: {sg_top}")
            common = set(cbow_top) & set(sg_top)
            print(f"    Common   : {sorted(common)}")

    print("\nTask 3 complete.")


if __name__ == "__main__":
    main()
