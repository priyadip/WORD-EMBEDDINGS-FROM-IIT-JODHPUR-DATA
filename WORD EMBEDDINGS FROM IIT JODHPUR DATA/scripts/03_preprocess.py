"""
TASK 1 - Step 3: Corpus Preprocessing
=======================================
Reads all raw text files from data/raw/text/, applies the required
preprocessing pipeline, and writes:

  - data/processed/documents/<doc_id>.txt  : one cleaned file per source
  - data/processed/corpus.txt              : all docs concatenated (one
                                             sentence per line)  -  used by
                                             Word2Vec training in Task 2.

Preprocessing pipeline (as required by the assignment):
  (i)   Removal of boilerplate text and formatting artifacts
  (ii)  Tokenization                     (NLTK word_tokenize)
  (iii) Lowercasing
  (iv)  Removal of excessive punctuation and non-textual content

Additionally:
  - Non-English (non-ASCII / Devanagari) tokens are removed
  - Common English stopwords are NOT removed here  -  Word2Vec benefits from
    context words including stopwords (this step can be done optionally)
  - URLs, email addresses, and HTML entities are stripped
"""

import os
import re
import json
import unicodedata

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Download required NLTK data (silent if already present)
nltk.download("punkt",           quiet=True)
nltk.download("punkt_tab",       quiet=True)
nltk.download("stopwords",       quiet=True)

# Paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(SCRIPT_DIR, "..", "data", "raw",       "text")
DOC_DIR     = os.path.join(SCRIPT_DIR, "..", "data", "processed", "documents")
CORPUS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "corpus.txt")
META_PATH   = os.path.join(SCRIPT_DIR, "..", "data", "processed", "metadata.json")

# Preprocessing helpers

# Regex patterns compiled once for efficiency
RE_URL      = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL    = re.compile(r"\S+@\S+\.\S+")
RE_HTML_ENT = re.compile(r"&[a-z]+;")
# Obfuscated emails: username[at]iitj[dot]ac[dot]in  or  username at iitj dot ac dot in
# Also handles multi-part usernames: pooja dot shivam at iitj dot ac dot in
# Replaces entire obfuscated email with just the first username part
RE_OBF_EMAIL = re.compile(
    r"(\w+(?:\s*(?:\[dot\]|dot)\s*\w+)*)\s*(?:\[at\]|at)\s*\w+\s*(?:\[dot\]|dot)\s*\w+(?:\s*(?:\[dot\]|dot)\s*\w+)*",
    re.I
)          # HTML entities like &amp; &nbsp;
RE_PHONE    = re.compile(r"\b\+?[\d\s\-\(\)]{7,}\b")  # phone numbers
RE_MULTI_WS = re.compile(r"\s+")
# Valid token: starts with a letter, followed by letters/digits, at least 2 chars
# Allows pure words (research) and alphanumeric codes (cs601, me301, phd24)
# Rejects pure numbers (2024), punctuation, and non-ASCII
RE_WORD     = re.compile(r"^[a-z][a-z0-9]{1,}$")

# Boilerplate patterns present on every IIT Jodhpur page (footer/navbar/timestamps)
# Lines matching ANY of these are discarded before sentence tokenisation.
BOILERPLATE_PATTERNS = [
    re.compile(r"copyright.*all rights reserved", re.I),
    re.compile(r"nagaur road.*karwar.*jodhpur", re.I),
    re.compile(r"this portal is owned.*designed.*developed", re.I),
    re.compile(r"for any comments.*enquiries.*feedback.*wim", re.I),
    re.compile(r"last updated\s*:?\s*\d", re.I),
    re.compile(r"important links.*cccd.*iitj.*recruitment", re.I),
    re.compile(r"web information manager", re.I),
    re.compile(r"institute repository", re.I),
    re.compile(r"digital infrastructure.*automation.*iit jodhpur", re.I),
    re.compile(r"^(home|about|contact|login|logout|search|menu|skip to)$", re.I),
]


def is_boilerplate(line: str) -> bool:
    """Return True if the line matches a known boilerplate pattern."""
    s = line.strip()
    return any(p.search(s) for p in BOILERPLATE_PATTERNS)


def remove_provenance_header(text: str) -> str:
    """Strip the SOURCE_URL / SOURCE_PDF header line added by collection scripts."""
    lines = text.split("\n")
    filtered = [l for l in lines if not l.startswith(("SOURCE_URL:", "SOURCE_PDF:"))]
    return "\n".join(filtered)


def remove_formatting_artifacts(text: str) -> str:
    """
    Remove common boilerplate and formatting artifacts:
      - URLs and email addresses
      - HTML/XML tags and entities
      - Phone numbers
      - Sequences of underscores / dashes used as dividers
      - Repeated punctuation (e.g. '......' or '-----')
      - Non-breaking spaces and other special whitespace
    """
    text = RE_URL.sub(" ",      text)    # remove hyperlinks
    # Obfuscated emails: keep only the username as a single joined token
    # e.g. dofa[at]iitj[dot]ac[dot]in       → dofa
    # e.g. pooja dot shivam at iitj dot ac dot in → poojashivam
    def _clean_email(m):
        username = re.sub(r'\s*(?:\[dot\]|dot)\s*', '', m.group(1), flags=re.I)
        return username
    text = RE_OBF_EMAIL.sub(_clean_email, text)
    text = RE_EMAIL.sub(" ",    text)    # remove any remaining real emails
    text = RE_HTML_ENT.sub(" ", text)    # remove HTML entities
    text = RE_PHONE.sub(" ",    text)    # remove phone numbers
    # Remove HTML tags (residual from BeautifulSoup misses)
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove lines that are purely punctuation / symbols (dividers like '----')
    # Also remove known website boilerplate lines (footer, navbar, timestamps)
    lines = text.split("\n")
    lines = [l for l in lines
             if not re.fullmatch(r"[\W_]+", l.strip())
             and not is_boilerplate(l)]
    text = "\n".join(lines)
    # Collapse excessive punctuation: '......' → '.'
    text = re.sub(r"[.]{3,}", ".",  text)
    text = re.sub(r"[-]{3,}", " ",  text)
    text = re.sub(r"[_]{3,}", " ",  text)
    text = re.sub(r"[*]{2,}", " ",  text)
    # Replace non-breaking spaces and other Unicode whitespace with a regular space
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return text


def is_english_sentence(sentence: str) -> bool:
    """
    Returns True if the sentence is English.
    Strips alphanumeric codes (roll numbers, course codes) before checking,
    then always runs langdetect on the remaining alphabetic-only words.
    Falls back to avg-token-length heuristic if langdetect unavailable.
    """
    if not sentence.strip():
        return False

    # Remove alphanumeric codes (roll numbers like b17me038, course codes like cs601)
    # so they don't inflate the avg token length and fool the heuristic
    alpha_only_tokens = [t for t in sentence.split() if t.isalpha()]
    if not alpha_only_tokens:
        return True   # only codes, no alphabetic words  -  keep it

    alpha_text = " ".join(alpha_only_tokens)

    if LANGDETECT_AVAILABLE and len(alpha_text) >= 15:
        try:
            return detect(alpha_text) == "en"
        except LangDetectException:
            pass

    # Fallback heuristic: avg alphabetic token length
    avg_len = sum(len(t) for t in alpha_only_tokens) / len(alpha_only_tokens)
    return avg_len > 3.2


def is_english_token(token: str) -> bool:
    """
    Return True only if the token is a pure lowercase ASCII alphabetic word
    (length ≥ 2). This ensures Devanagari, Chinese, and other non-Latin
    script characters are excluded after lowercasing.
    """
    return bool(RE_WORD.match(token))


def deduplicate_lines(text: str) -> str:
    """Remove duplicate lines within a document (order-preserving)."""
    seen = set()
    unique_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique_lines.append(line)
    return "\n".join(unique_lines)


def preprocess_document(raw_text: str) -> tuple[list[list[str]], str]:
    """
    Full preprocessing pipeline for a single document.

    Steps:
      1. Remove provenance header
      2. Remove formatting artifacts / boilerplate
      3. Deduplicate repeated lines (caused by nested HTML elements)
      4. Sentence tokenization (NLTK sent_tokenize)
      5. Word tokenization     (NLTK word_tokenize)
      6. Lowercasing
      7. Keep only valid English alphabetic tokens (removes punctuation,
         numbers, and non-Latin script tokens)

    Returns:
      sentences : list of token lists  -  one list per sentence
      clean_text: joined clean text (sentences separated by newlines)
    """
    # Step 1: strip provenance header
    text = remove_provenance_header(raw_text)

    # Step 2: remove formatting artifacts
    text = remove_formatting_artifacts(text)

    # Step 3: deduplicate repeated lines
    text = deduplicate_lines(text)

    # Step 4: sentence tokenization  -  preserves sentence boundaries for Word2Vec
    sentences_raw = sent_tokenize(text)

    sentences = []
    seen_sents = set()
    for sent in sentences_raw:
        # Reject non-English sentences (catches garbled Hindi in mixed blocks)
        if not is_english_sentence(sent):
            continue
        # Step 5: word tokenize
        tokens = word_tokenize(sent)
        # Step 6: lowercase
        tokens = [t.lower() for t in tokens]
        # Step 7: keep only pure English alphabetic tokens (≥ 2 chars)
        tokens = [t for t in tokens if is_english_token(t)]
        if len(tokens) >= 3:   # discard trivially short sentences
            key = " ".join(tokens)
            if key not in seen_sents:
                seen_sents.add(key)
                sentences.append(tokens)

    # Build a plain-text representation: one sentence per line, space-separated
    clean_text = "\n".join(" ".join(s) for s in sentences)
    return sentences, clean_text


# Main

def main():
    os.makedirs(DOC_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)

    raw_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".txt")])

    if not raw_files:
        print(f"No raw text files found in {RAW_DIR}.")
        print("Run 01_scrape_website.py and 02_extract_pdf_text.py first.")
        return

    print(f"Preprocessing {len(raw_files)} raw text files …\n")

    all_sentences      = []   # for corpus.txt
    global_seen_sents  = set()  # global cross-document deduplication
    metadata           = []   # per-document stats
    total_tokens       = 0
    vocab              = set()

    for fname in raw_files:
        raw_path = os.path.join(RAW_DIR, fname)
        with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        sentences, clean_text = preprocess_document(raw_text)

        if not sentences:
            print(f"  [SKIP] {fname}  -  no usable text after preprocessing.")
            continue

        # Global cross-document deduplication
        unique_sentences = []
        for sent in sentences:
            key = " ".join(sent)
            if key not in global_seen_sents:
                global_seen_sents.add(key)
                unique_sentences.append(sent)
        sentences = unique_sentences

        if not sentences:
            continue

        # Collect vocabulary and token counts for this doc
        doc_tokens = [tok for sent in sentences for tok in sent]
        doc_vocab  = set(doc_tokens)
        total_tokens += len(doc_tokens)
        vocab.update(doc_vocab)

        # Save individual cleaned document
        clean_text = "\n".join(" ".join(s) for s in sentences)
        out_path = os.path.join(DOC_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        all_sentences.extend(sentences)

        meta_entry = {
            "file":       fname,
            "sentences":  len(sentences),
            "tokens":     len(doc_tokens),
            "vocab_size": len(doc_vocab),
        }
        metadata.append(meta_entry)
        print(f"  [OK] {fname}  |  {len(sentences)} sentences, {len(doc_tokens)} tokens")

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(" ".join(sent) + "\n")

    summary = {
        "total_documents": len(metadata),
        "total_sentences": len(all_sentences),
        "total_tokens":    total_tokens,
        "vocabulary_size": len(vocab),
        "documents":       metadata,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Preprocessing complete.")
    print(f"  Total documents  : {len(metadata)}")
    print(f"  Total sentences  : {len(all_sentences)}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  Vocabulary size  : {len(vocab):,}")
    print(f"  Corpus saved to  : {CORPUS_PATH}")
    print(f"  Metadata saved to: {META_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
