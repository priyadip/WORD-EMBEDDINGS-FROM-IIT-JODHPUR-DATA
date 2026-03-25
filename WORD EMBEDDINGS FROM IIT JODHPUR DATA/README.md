# Word Embeddings from IIT Jodhpur Data

Implementation of Word2Vec (CBOW & Skip-gram) trained on textual data collected from IIT Jodhpur official sources.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

All scripts must be run from the **project root** (`WORD EMBEDDINGS FROM IIT JODHPUR DATA/`).

---

### Step 1  -  Scrape IIT Jodhpur Website

Crawls `iitj.ac.in` and downloads all English web pages and PDFs.

```bash
python scripts/01_scrape_website.py
```

**Output:**
- `data/raw/text/`  -  one `.txt` file per scraped web page
- `data/raw/pdfs/`  -  downloaded PDF files

---

### Step 2  -  Extract Text from PDFs

Extracts English text from the downloaded PDFs using PyMuPDF. Detects and removes non-English (Hindi/Devanagari) content.

```bash
python scripts/02_extract_pdf_text.py
```

**Output:**
- Adds more `.txt` files into `data/raw/text/`

---

### Step 3  -  Preprocess & Build Corpus

Cleans all raw text files and builds the final corpus used for training.

```bash
python scripts/03_preprocess.py
```

**Output:**
- `data/processed/corpus.txt`  -  cleaned corpus (one sentence per line)
- `data/processed/documents/`  -  individual cleaned document files
- `data/processed/metadata.json`  -  dataset statistics

> **Note:** Steps 1, 2, and 3 must be completed before anything else. `corpus.txt` must exist before running any further scripts.

---

## Run All Remaining Steps at Once (Steps 4–9)

Once the corpus is ready, you can run the full pipeline in one command:

```bash
python scripts/run_pipeline.py
```

This automatically runs Steps 4, 5, 6, 7, 8, and 9 in sequence.

---

## Or Run Individual Steps

### Step 4  -  Dataset Statistics & Word Cloud

Computes corpus statistics and generates the word cloud image.

```bash
python scripts/04_statistics_wordcloud.py
```

**Output:**
- `data/output/wordcloud.png`
- `data/output/top_words.json`

---

### Step 5  -  Train Word2Vec Models (Gensim)

Trains CBOW and Skip-gram models with a hyperparameter sweep over embedding dimension, window size, and negative samples.

```bash
python scripts/05_train_word2vec.py
```

**Output:**
- `data/models/cbow_best.model`
- `data/models/skipgram_best.model`
- `data/models/experiment_results.json`

---

### Step 6  -  Semantic Analysis

Computes top-5 nearest neighbours for probe words and runs 15 analogy tests on the best Gensim models.

```bash
python scripts/06_semantic_analysis.py
```

**Output:**
- `data/models/semantic_analysis.json`

---

### Step 7  -  Visualisation (PCA & t-SNE)

Projects word embeddings into 2D using PCA and t-SNE for the Gensim CBOW and Skip-gram models.

```bash
python scripts/07_visualization.py
```

**Output:**
- `data/figures/pca_cbow.png`
- `data/figures/pca_skipgram.png`
- `data/figures/pca_comparison.png`
- `data/figures/tsne_cbow.png`
- `data/figures/tsne_skipgram.png`
- `data/figures/tsne_comparison.png`

---

### Step 8  -  Word2Vec from Scratch (NumPy)

Trains CBOW and Skip-gram models built entirely in NumPy (no Gensim). Sweeps over window sizes 3, 5, 10.

```bash
python scripts/08_word2vec_scratch.py
```

**Output:**
- `data/models/scratch_cbow_vectors.npy`
- `data/models/scratch_skipgram_vectors.npy`
- `data/models/scratch_vocab.json`
- `data/models/scratch_experiment_results.json`

> **Note:** This script is CPU-only and slow (~39 min for CBOW, ~134 min for Skip-gram at window=5).

---

### Step 9  -  Compare Gensim vs Scratch Models

Compares all four models (Gensim CBOW, Gensim Skip-gram, Scratch CBOW, Scratch Skip-gram) using nearest neighbours, analogy accuracy, and CKA similarity.

```bash
python scripts/09_compare_models.py
```

**Output:**
- `data/models/comparison_results.json`
- `data/figures/comparison_nn.png`
- `data/figures/comparison_tsne.png`

---

## Project Structure

```
WORD EMBEDDINGS FROM IIT JODHPUR DATA/
├── scripts/
│   ├── 01_scrape_website.py
│   ├── 02_extract_pdf_text.py
│   ├── 03_preprocess.py
│   ├── 04_statistics_wordcloud.py
│   ├── 05_train_word2vec.py
│   ├── 06_semantic_analysis.py
│   ├── 07_visualization.py
│   ├── 08_word2vec_scratch.py
│   ├── 09_compare_models.py
│   └── run_pipeline.py
├── data/
│   ├── raw/
│   │   ├── text/
│   │   └── pdfs/
│   ├── processed/
│   │   ├── corpus.txt
│   │   ├── metadata.json
│   │   └── documents/
│   ├── models/
│   ├── figures/
│   └── output/
├── report.tex
├── requirements.txt
└── README.md
```
