"""
Assignment Pipeline Runner
===========================
PRE-CONDITION: Steps 1-3 must already be done (corpus collected & preprocessed).
  - data/processed/corpus.txt must exist before running this file.

This script runs Steps 4 to 9 in order, one by one:

  Step 4: 04_statistics_wordcloud.py  ->  reads corpus.txt, prints stats, saves word cloud
  Step 5: 05_train_word2vec.py        ->  trains Gensim CBOW & Skip-gram, saves best models
  Step 6: 06_semantic_analysis.py     ->  loads best models, runs nearest neighbours + analogies
  Step 7: 07_visualization.py         ->  loads best models, draws PCA + t-SNE plots
  Step 8: 08_word2vec_scratch.py      ->  trains CBOW & Skip-gram from scratch using NumPy only
  Step 9: 09_compare_models.py        ->  compares Gensim vs Scratch models (CKA, neighbours, analogies)

HOW IT WORKS:
  - Each step is run as a separate Python subprocess (like typing `python 04_...py` in terminal).
  - If any step fails, the pipeline stops immediately and prints the error.
  - If a step succeeds, it moves on to the next step automatically.

HOW TO RUN:
    python scripts/run_pipeline.py

OUTPUT LOCATIONS:
    data/output/        ->  word cloud image + top-50 word frequency
    data/models/        ->  trained models + semantic_analysis.json + comparison_results.json
    data/figures/       ->  PCA and t-SNE plots
"""

import sys
import os
import subprocess

# Folder where all the scripts (04 to 09) are located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(name: str, script: str):
    """
    Runs one script as a subprocess.
    - name   : human-readable label printed to console
    - script : filename of the script to run (e.g. '04_statistics_wordcloud.py')
    If the script exits with an error, the whole pipeline stops.
    """
 
    print(f"  RUNNING: {name}")


    # sys.executable = the same Python interpreter currently running this file
    # os.path.join(SCRIPT_DIR, script) = full path to the target script
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, script)],
        check=False   # don't raise exception on failure; we handle it below
    )

    if result.returncode != 0:
        # Non-zero return code means the script crashed or had an error
        print(f"\n[ERROR] Step '{name}' failed (exit code {result.returncode}).")
        print("Fix the error above before continuing.")
        sys.exit(result.returncode)   # stop the whole pipeline

    print(f"\n[OK] Step '{name}' completed successfully.\n")


def main():

    # Steps run in this exact order, one after another
    run_step("Statistics & Word Cloud",       "04_statistics_wordcloud.py")
    run_step("Word2Vec Model Training",       "05_train_word2vec.py")
    run_step("Semantic Analysis",             "06_semantic_analysis.py")
    run_step("PCA & t-SNE Visualization",    "07_visualization.py")
    run_step("Word2Vec from Scratch (NumPy)", "08_word2vec_scratch.py")
    run_step("Compare Gensim vs Scratch",    "09_compare_models.py")

    print("  Outputs:")
    print("    data/output/        -  word cloud + statistics")
    print("    data/models/        -  Word2Vec models + analysis + comparison")
    print("    data/figures/       -  PCA + t-SNE plots")



if __name__ == "__main__":
    main()
