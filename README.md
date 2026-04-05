# ⚖ Semantic Diff Engine

> Understand **what changed in meaning** between two versions of a legal or policy document — not just what words changed.

Most diff tools show you that a sentence was reworded. This engine tells you *why it matters legally*: whether a clause narrowed its scope, reversed a burden of proof, introduced a new obligation, or added an exception that wasn't there before.

---

## Demo

Upload two versions of a contract or policy document (`.txt` or `.pdf`). The engine segments both into clauses, matches them by semantic similarity, and classifies every meaningful change with a plain-English legal implication.

**Example output:**

```
⚠  [#  NUMERIC CHANGE]         score: 0.9214
   flags  : numeric value changed
   v1     : (a) Payment is due within 30 days of invoice.
   v2     : (a) Payment is due within 45 days of invoice.
   impact : The payment window increased by 15 days, reducing the creditor's
            ability to enforce timely collection.

✚  [+  ADDED OBLIGATION]
   added  : (d) Late payments may incur a penalty of 2%.
   impact : A new financial penalty clause was introduced, directly increasing
            the debtor's liability for delayed payments.

✖  [-  REMOVED OBLIGATION]
   removed: (c) Either party may terminate with 60 days written notice.
   impact : The right to terminate with advance notice was removed, leaving
            parties without a structured exit mechanism.
```

---

## How It Works

The diff is computed in **embedding space**, not character space.

```
Doc v1 ──┐
          ├──▶  Clause Segmenter  ──▶  Embedding Model  ──┐
Doc v2 ──┘     (semantic chunking)    (sentence-transformer) │
                                                              ▼
                                               Nearest-Neighbor Matcher
                                               (cosine similarity matrix)
                                                              │
                                               ┌─────────────┴──────────────┐
                                               ▼                            ▼
                                        Delta Classifier            Added / Removed
                                     (few-shot LLM prompt)        clause detection
                                               │
                                               ▼
                                    Semantic Change Report
                              (label + plain-English implication)
```

**Key design decisions:**

- **Embedding-based matching** — clauses are matched by meaning, not position. A reordered or renumbered clause still finds its correct counterpart.
- **Nearest-neighbor pairing** — builds a full cosine similarity matrix (every v1 clause vs every v2 clause) and uses `argmax` to find the best match per clause. Clauses below the match threshold are flagged as added or removed.
- **Rule-based guards** — numeric value shifts and negation word changes (`not`, `never`, `except`) are caught even when embedding similarity is high, because embeddings are blind to those distinctions.
- **Few-shot classification** — no model fine-tuning. A structured prompt with labelled examples teaches the LLM to output consistent labels and one-sentence implications.
- **Embedding cache** — clause vectors are cached to disk so re-runs on the same document skip re-embedding entirely.

---

## Change Labels

| Label | Meaning |
|---|---|
| `narrowed_scope` | Obligation or right now covers less |
| `expanded_scope` | Obligation or right now covers more |
| `numeric_change` | A number changed (amount, days, percentage) |
| `reversed_burden` | Who bears responsibility flipped |
| `added_obligation` | New duty introduced |
| `removed_obligation` | Duty eliminated |
| `added_carve_out` | New exception or exclusion added |
| `removed_carve_out` | Exception removed — clause tightened |
| `negation_shift` | `shall` → `shall not`, or vice versa |

---

## Project Structure

```
semantic_diff/
│
├── app.py                        # Streamlit web UI
│
├── src/
│   ├── loader.py                 # Reads .txt files from disk
│   ├── pdf_loader.py             # Reads .pdf files using pdfplumber
│   ├── segmenter.py              # Splits documents into clauses (regex-based)
│   ├── embedder.py               # Converts clause text → vectors (sentence-transformer)
│   ├── comparator.py             # Nearest-neighbor matching + cosine similarity
│   ├── classifier.py             # Few-shot LLM classification of change type
│   └── optimizer.py              # Batching, disk cache, deduplication
│
├── sample_docs/
│   ├── contract_v1.txt           # Sample original document
│   └── contract_v2.txt           # Sample amended document
│
├── .env                          # API keys (never commit this)
├── .embedding_cache.json         # Auto-generated embedding cache (gitignored)
├── .gitignore
└── requirements.txt
```

### Module Responsibilities

**`loader.py`** — reads a `.txt` file and returns raw text. One function, one job.

**`pdf_loader.py`** — uses `pdfplumber` to extract text from PDF files page by page. Handles multi-column layouts and most real-world PDFs cleanly.

**`segmenter.py`** — detects clause boundaries using a regex pattern that covers all common legal numbering formats: `1.`, `1.2`, `1.2.3`, `(a)`, `(i)`, `(A)`, `Article 1`, `Section 2.1`, `WHEREAS`, `Schedule A`. Multi-line clauses are accumulated until the next header is detected. Returns structured dicts with `text`, `label`, and nesting `level`.

**`embedder.py`** — loads `all-MiniLM-L6-v2` once at module import time (avoiding the 60–90s reload on every run) and converts a list of clause strings into embedding vectors.

**`comparator.py`** — builds a full `(n × m)` cosine similarity matrix between all v1 and v2 clause embeddings using vectorised numpy. For each v1 clause, finds the best-matching v2 clause. Applies a match threshold (0.50) — unmatched clauses are flagged as removed/added. Applies an unchanged threshold (0.97) and numeric/negation guards — only truly changed pairs are returned.

**`classifier.py`** — sends changed clause pairs to the LLM with a structured few-shot prompt. Handles three prompt types: `changed` (v1 vs v2), `added` (explain what the new clause introduces), `removed` (explain what was lost). Parses the `Label:` / `Implication:` response format and validates the label against the allowed set.

**`optimizer.py`** — three optimisations: (1) disk-based embedding cache keyed by MD5 hash of clause text, (2) batch embedder with configurable batch size and rate-limit pause, (3) exact-text deduplication to skip LLM calls on identical clause pairs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector comparison | `numpy` (cosine similarity, matrix operations) |
| LLM classification | Groq API — `llama3-8b-8192` (free tier) |
| PDF parsing | `pdfplumber` |
| Web UI | `streamlit` |
| Environment config | `python-dotenv` |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/semantic-diff-engine.git
cd semantic-diff-engine
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com). No credit card required.

> **Note:** If you prefer OpenAI instead of Groq, replace the client in `src/classifier.py`:
> ```python
> # swap this:
> from groq import Groq
> client = Groq(api_key=os.getenv("GROQ_API_KEY"))
>
> # for this:
> from openai import OpenAI
> client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
> # and change model to "gpt-4o-mini"
> ```

### 5. Run

**Streamlit web app (recommended):**
```bash
streamlit run app.py
```

**Terminal mode:**
```bash
python src/main.py
```

For terminal mode, set your document paths at the top of `src/main.py`:
```python
PATH_V1 = "sample_docs/contract_v1.txt"   # supports .txt and .pdf
PATH_V2 = "sample_docs/contract_v2.txt"
```

---

## Requirements

```
sentence-transformers
numpy
python-dotenv
groq
pdfplumber
streamlit
```

Python 3.9 or higher recommended.

---

## Supported Document Formats

| Format | Support |
|---|---|
| `.txt` | Full support |
| `.pdf` | Full support via `pdfplumber` |
| `.docx` | Not yet supported (planned) |

### Supported clause numbering patterns

The segmenter detects all of the following:

```
1.          Top-level numbered
1.2         Sub-section
1.2.3       Sub-sub-section
(a) (b)     Lowercase letter
(A) (B)     Uppercase letter
(i) (ii)    Roman numeral
Article 1   Article heading
Section 2   Section heading
Schedule A  Schedule heading
WHEREAS     Recital / preamble keyword
```

Multi-line clauses (where body text wraps onto the next line) are handled correctly — continuation lines are accumulated into the same clause until the next header pattern is detected.

---

## Configuration

Key thresholds in `src/comparator.py`:

```python
MATCH_THRESHOLD     = 0.50   # below this → clauses are unrelated (add/remove)
UNCHANGED_THRESHOLD = 0.97   # above this → clause is unchanged (skip)
```

Adjusting these changes sensitivity:
- **Lower `MATCH_THRESHOLD`** → more aggressive about flagging clauses as removed/added
- **Higher `UNCHANGED_THRESHOLD`** → flags more clauses as changed (more sensitive)
- **Lower `UNCHANGED_THRESHOLD`** → only flags significant semantic shifts

Batch size and cache settings in `src/optimizer.py`:

```python
# In embed_in_batches():
batch_size = 64              # reduce to 32 if hitting rate limits
pause_between_batches = 0.5  # seconds between batches
```

---

## Performance

| Document size | Approx. runtime (after first run) |
|---|---|
| 1–2 pages (10–25 clauses) | 5–10 seconds |
| 5–10 pages (50–100 clauses) | 15–30 seconds |
| 20+ pages (200+ clauses) | 45–90 seconds |

**First run** is slower because `sentence-transformers` downloads the model (~90MB) and the embedding cache is empty. Subsequent runs on the same documents are significantly faster — cached clause embeddings are reused, and identical clauses skip the LLM classifier entirely.

The main runtime bottleneck at scale is the Groq API classifier (one call per changed clause). For documents with many changes, this can be parallelised with `asyncio` — not yet implemented.

---

## Limitations & Known Issues

- **Positional accuracy depends on segmentation quality.** Poorly formatted PDFs (scanned, two-column, or table-heavy) may produce noisy clause splits. Check `pdfplumber`'s output if results look off.
- **Embeddings are weak on numbers and negation.** `"shall be liable"` and `"shall not be liable"` are very close in embedding space. The rule-based guards in `comparator.py` catch most of these, but edge cases exist.
- **LLM classification can be inconsistent** for borderline cases. The few-shot prompt is the main lever for improving this — adding more labelled examples directly improves accuracy without retraining.
- **Clause reordering is handled** — nearest-neighbor matching finds the correct counterpart regardless of position. However, if two clauses have very similar embeddings but different positions, the wrong one may be selected as the best match.

---

## Roadmap

- [ ] `.docx` file support
- [ ] Async LLM calls for faster classification on large documents
- [ ] Confidence scores on LLM classifications
- [ ] Word-level highlight diff within changed clause pairs
- [ ] Export to PDF report
- [ ] Support for custom clause segmentation rules per document type

---

## .gitignore

```gitignore
# Environment
.env

# Embedding cache (auto-regenerated)
.embedding_cache.json

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Streamlit
.streamlit/

# OS
.DS_Store
Thumbs.db
```

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgements

- [`sentence-transformers`](https://www.sbert.net/) — `all-MiniLM-L6-v2` for fast, accurate semantic embeddings
- [`pdfplumber`](https://github.com/jsvine/pdfplumber) — reliable PDF text extraction
- [Groq](https://groq.com/) — fast, free LLM inference for the classifier
- [`streamlit`](https://streamlit.io/) — rapid web UI without frontend overhead
