import hashlib
import json
import os
import time
from embedder import embed_clauses

CACHE_FILE = ".embedding_cache.json"

# ── 1. Disk-based embedding cache ─────────────────────────────────────

def _load_cache() -> dict:
    """Loads the embedding cache from disk. Returns empty dict if none exists."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    """Persists the embedding cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _hash_text(text: str) -> str:
    """Creates a short unique key for a clause string."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def embed_with_cache(clauses: list[dict]) -> list[list[float]]:
    """
    Embeds a list of clause dicts, using a disk cache to skip re-embedding
    clauses that were already processed in a previous run.

    Why this matters: a 100-page contract might have 300 clauses.
    If you tweak one clause and re-run, you'd normally re-embed all 300.
    With caching, only the changed ones get re-embedded.
    """
    cache = _load_cache()
    texts = [c["text"] for c in clauses]

    # Split into cached vs uncached
    uncached_indices = []
    uncached_texts = []
    for i, text in enumerate(texts):
        key = _hash_text(text)
        if key not in cache:
            uncached_indices.append(i)
            uncached_texts.append(text)

    # Embed only the uncached ones (in batches)
    if uncached_texts:
        new_vectors = embed_in_batches(uncached_texts, batch_size=64)
        for idx, text, vec in zip(uncached_indices, uncached_texts, new_vectors):
            cache[_hash_text(text)] = vec

    _save_cache(cache)

    # Reconstruct full embedding list in original order
    return [cache[_hash_text(t)] for t in texts]


# ── 2. Batch embedder for large inputs ────────────────────────────────

def embed_in_batches(
    texts: list[str],
    batch_size: int = 64,
    pause_between_batches: float = 0.5
) -> list[list[float]]:
    """
    Embeds a large list of texts in chunks of batch_size.
    Adds a small pause between batches to avoid rate limit errors.

    batch_size=64 is safe for most free-tier embedding APIs.
    Reduce to 32 if you still hit rate limits.
    """
    all_vectors = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        print(f"  Embedding batch {start // batch_size + 1}"
              f"/{-(-total // batch_size)} ({len(batch)} clauses)...")
        vectors = embed_clauses(batch)
        all_vectors.extend(vectors)

        if start + batch_size < total:
            time.sleep(pause_between_batches)

    return all_vectors


# ── 3. Deduplicate identical clauses before classifying ───────────────

def deduplicate_for_classification(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Separates a list of comparison pairs into:
      - to_classify: pairs where v1 != v2 text (need LLM)
      - skip: pairs where text is identical (no LLM call needed)

    This saves LLM API calls — identical clauses don't need classification.
    On a 200-clause contract, typically 60–70% of clauses are unchanged.
    """
    to_classify = []
    skip = []

    for pair in pairs:
        if pair["v1"].strip().lower() == pair["v2"].strip().lower():
            pair["label"] = "cosmetic_only"
            pair["implication"] = "Clause is identical in both versions."
            pair["changed"] = False
            skip.append(pair)
        else:
            to_classify.append(pair)

    return to_classify, skip