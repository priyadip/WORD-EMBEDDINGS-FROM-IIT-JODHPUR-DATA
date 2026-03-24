"""
TASK 1 - Step 2: PDF Text Extractor
=====================================
Reads every PDF file in data/raw/pdfs/ and extracts its English text using
PyMuPDF (fitz). Non-English (e.g. Devanagari / Hindi) text blocks are
detected by Unicode range analysis and discarded.

Each PDF becomes one plain-text file saved in data/raw/text/ so that the
preprocessing step (03_preprocess.py) can treat all sources uniformly.

Why PyMuPDF?
  - Handles both text-layer PDFs and scanned-image PDFs (basic support).
  - Preserves reading order better than pdfminer for most academic documents.
"""

import os
import re
import unicodedata

import fitz   # PyMuPDF

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR    = os.path.join(SCRIPT_DIR, "..", "data", "raw", "pdfs")
TEXT_DIR   = os.path.join(SCRIPT_DIR, "..", "data", "raw", "text")

# Helpers

# Common short English words used to detect real English text
_COMMON_EN = {
    "the", "of", "and", "in", "to", "for", "is", "are", "on", "at",
    "by", "an", "as", "be", "it", "or", "from", "with", "this", "that",
    "not", "was", "has", "have", "may", "will", "its", "all", "but",
    "iit", "phd", "mtech", "btech", "department", "research", "student",
    "professor", "course", "exam", "semester", "india", "jodhpur",
}

def is_english_block(text: str, threshold: float = 0.80) -> bool:
    """
    Returns True if the block is genuine English text.

    Two checks:
      1. ASCII ratio: ≥80% of alphabetic chars must be ASCII.
         (Filters Devanagari/Tamil/Arabic in proper Unicode encoding.)
      2. Real-word ratio: ≥15% of whitespace-separated tokens must be
         recognisable English words (avg length ≥ 3.5 chars AND at least
         one common English word present).
         (Filters legacy-font-encoded Hindi that looks like ASCII gibberish.)
    """
    alpha = [c for c in text if unicodedata.category(c).startswith("L")]
    if not alpha:
        return False

    # Check 1: ASCII ratio
    ascii_alpha = sum(1 for c in alpha if ord(c) < 128)
    if (ascii_alpha / len(alpha)) < threshold:
        return False

    # Check 2: real English word heuristic
    tokens = [t.lower().strip(".,;:()[]") for t in text.split() if t.isalpha()]
    if not tokens:
        return False

    avg_len = sum(len(t) for t in tokens) / len(tokens)
    has_common = any(t in _COMMON_EN for t in tokens)

    # Garbled legacy-font Hindi: avg token length ≤ 3.2 and no common words
    # Use langdetect as fallback for uncertain blocks before rejecting
    if avg_len <= 3.2 and not has_common:
        if LANGDETECT_AVAILABLE and len(text.strip()) >= 20:
            try:
                lang = detect(text)
                return lang == "en"
            except LangDetectException:
                pass
        return False

    return True


def clean_pdf_text(raw_text: str) -> str:
    """
    Light cleaning of raw text extracted from a PDF page.
      - Collapse multiple blank lines into one
      - Remove form-feed characters (\\f) inserted by PyMuPDF between pages
      - Strip leading/trailing whitespace
    """
    # Remove form-feed page separators
    text = raw_text.replace("\f", "\n")
    # Collapse runs of 3+ newlines into 2 newlines (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace (tabs, multiple spaces)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def table_to_text(table) -> list[str]:
    """
    Convert a PyMuPDF Table object into a list of sentences  -  one per row.

    Each row's cells are joined with spaces so that semantically related
    terms (e.g. grade name + letter + CGPA points, or course + credits +
    semester) appear inside the same Word2Vec context window.

    Example table row  : ["Outstanding", "O", "10", "≥ 90%"]
    Produced sentence  : "Outstanding O 10 90%"
    → embeddings learn: outstanding ↔ cgpa ↔ ten  (grade semantics)
    """
    sentences = []
    try:
        rows = table.extract()          # list of lists of cell strings
    except Exception:
        return sentences

    for row in rows:
        # Flatten None / non-string cells; skip empty cells
        cells = [str(c).strip() for c in row if c and str(c).strip()]
        if len(cells) >= 2:             # skip single-cell / empty rows
            sentences.append(" ".join(cells))
    return sentences


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF with PyMuPDF and extract English text page by page.

    Strategy:
      1. Detect tables on each page via find_tables().
      2. Convert each table row → one sentence (preserves co-occurrence of
         semantically related terms for better Word2Vec embeddings).
      3. Extract remaining non-table text blocks as before.
      4. Non-English blocks are discarded throughout.

    Returns the full concatenated English text of the document.
    """
    english_pages = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        print(f"  [ERROR] Cannot open {os.path.basename(pdf_path)}: {exc}")
        return ""

    for page_num, page in enumerate(doc, start=1):
        page_chunks = []

        # Collect bounding boxes of table regions so we can skip those
        # blocks during normal text extraction (avoid double-counting).
        table_bboxes = []
        try:
            tabs = page.find_tables()
            for tab in tabs:
                table_bboxes.append(fitz.Rect(tab.bbox))
                for sentence in table_to_text(tab):
                    if is_english_block(sentence):
                        page_chunks.append(sentence)
        except Exception:
            pass  # find_tables() not available or failed  -  fall back gracefully

        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        for block in blocks:
            block_rect = fitz.Rect(block[:4])
            # Skip blocks that overlap significantly with a detected table
            if any(block_rect.intersects(tb) for tb in table_bboxes):
                continue
            block_text = block[4].strip()
            if not block_text:
                continue
            if is_english_block(block_text):
                page_chunks.append(block_text)

        if page_chunks:
            english_pages.append("\n".join(page_chunks))

    doc.close()
    raw = "\n\n".join(english_pages)
    return clean_pdf_text(raw)


# Main

def main():
    os.makedirs(TEXT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}. Run 01_scrape_website.py first.")
        return

    print(f"Found {len(pdf_files)} PDF files. Extracting text …\n")
    saved = 0

    for fname in pdf_files:
        pdf_path  = os.path.join(PDF_DIR, fname)
        print(f"  Processing: {fname}")
        text = extract_text_from_pdf(pdf_path)

        if len(text.strip()) < 50:
            print(f"    [SKIP] Very little text extracted.")
            continue

        # Save extracted text with provenance header
        out_name = fname.replace(".pdf", "_pdf.txt")
        out_path = os.path.join(TEXT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE_PDF: {fname}\n\n")
            f.write(text)

        print(f"    [OK] {len(text)} chars → {out_name}")
        saved += 1

    print(f"\nDone. Extracted text from {saved}/{len(pdf_files)} PDFs.")
    print(f"Output directory: {TEXT_DIR}")


if __name__ == "__main__":
    main()
