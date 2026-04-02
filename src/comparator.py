import re
import numpy as np

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "except", "unless"}

# Similarity below this = clauses are unrelated, don't pair them
MATCH_THRESHOLD = 0.50

# Similarity above this = clause is unchanged, skip showing it
UNCHANGED_THRESHOLD = 0.97


def _extract_numbers(text: str) -> set[str]:
    """Pulls all numbers from a clause (amounts, days, percentages)."""
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def _has_negation_shift(text_a: str, text_b: str) -> bool:
    """True if one clause has negation words the other doesn't."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    return (words_a & NEGATION_WORDS) != (words_b & NEGATION_WORDS)


def _has_number_shift(text_a: str, text_b: str) -> bool:
    """True if numeric values differ between the two clauses."""
    return _extract_numbers(text_a) != _extract_numbers(text_b)


def _is_section_header(text: str) -> bool:
    """
    Returns True if a clause is just a section header with no body.
    e.g. "Section 1.1 Definitions" or "1. Payment Terms"
    These carry no semantic content — comparing two headers is meaningless.
    """
    stripped = text.strip()
    # Short line (under 60 chars) that matches a heading pattern
    if len(stripped) > 60:
        return False
    header_patterns = [
        r"^(Section|Article|Clause|Schedule)\s+[\d\.]+",
        r"^\d+\.\d*\s+[A-Z][^a-z]{0,40}$",   # "1.1 DEFINITIONS"
        r"^\d+\.\s*[A-Z][^.]{0,40}$",          # "1. Payment Terms"
    ]
    return any(re.match(p, stripped, re.IGNORECASE) for p in header_patterns)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0–1.0."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return float(np.dot(a, b) / (mag_a * mag_b))


def _build_similarity_matrix(
    embeddings_v1: list[list[float]],
    embeddings_v2: list[list[float]]
) -> np.ndarray:
    """
    Builds an (n x m) matrix where entry [i][j] is the cosine similarity
    between clause i from v1 and clause j from v2.
    Vectorised with numpy — fast even for 500x500.
    """
    mat_a = np.array(embeddings_v1)   # shape: (n, dim)
    mat_b = np.array(embeddings_v2)   # shape: (m, dim)

    # Normalise rows
    mat_a = mat_a / (np.linalg.norm(mat_a, axis=1, keepdims=True) + 1e-10)
    mat_b = mat_b / (np.linalg.norm(mat_b, axis=1, keepdims=True) + 1e-10)

    return mat_a @ mat_b.T            # shape: (n, m)


def pair_and_compare(
    clauses_v1: list[dict],
    embeddings_v1: list[list[float]],
    clauses_v2: list[dict],
    embeddings_v2: list[list[float]],
) -> list[dict]:
    """
    Nearest-neighbor clause matching:

    1. Build full similarity matrix (every v1 clause vs every v2 clause).
    2. For each v1 clause, find its best match in v2.
    3. If best match score >= MATCH_THRESHOLD, it's a real pair.
    4. If score >= UNCHANGED_THRESHOLD AND no numeric/negation shift → skip (unchanged).
    5. v2 clauses that were never matched → ADDED.
    6. v1 clauses whose best match was below threshold → REMOVED.
    7. Section headers with no body → always skip.

    Returns only actionable results: changed pairs + added + removed.
    """
    sim_matrix = _build_similarity_matrix(embeddings_v1, embeddings_v2)

    n = len(clauses_v1)
    m = len(clauses_v2)

    results = []
    matched_v2_indices = set()
    clause_counter = 0

    for i in range(n):
        clause_v1 = clauses_v1[i]
        text_v1 = clause_v1["text"]

        # Skip pure section headers — nothing to compare
        if _is_section_header(text_v1):
            continue

        # Find best matching v2 clause
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_j])

        if best_score < MATCH_THRESHOLD:
            # No good match found in v2 → this clause was REMOVED
            clause_counter += 1
            results.append({
                "clause_index":  clause_counter,
                "result_type":   "removed",
                "label":         "removed_obligation",
                "v1":            text_v1,
                "v2":            None,
                "similarity":    round(best_score, 4),
                "changed":       True,
                "reasons":       ["no matching clause found in v2"],
                "implication":   None,   # classifier will fill this
                "level":         clause_v1.get("level", 1),
            })
            continue

        # We have a real match
        matched_v2_indices.add(best_j)
        clause_v2 = clauses_v2[best_j]
        text_v2 = clause_v2["text"]

        # Check hard guards even if score is high
        num_shift  = _has_number_shift(text_v1, text_v2)
        neg_shift  = _has_negation_shift(text_v1, text_v2)
        is_changed = (best_score < UNCHANGED_THRESHOLD) or num_shift or neg_shift

        # Truly unchanged — skip entirely, don't add to results
        if not is_changed:
            continue

        reasons = []
        if best_score < UNCHANGED_THRESHOLD:
            reasons.append("semantic drift")
        if num_shift:
            reasons.append("numeric value changed")
        if neg_shift:
            reasons.append("negation shift")

        clause_counter += 1
        results.append({
            "clause_index":  clause_counter,
            "result_type":   "changed",
            "label":         None,      # filled by classifier
            "v1":            text_v1,
            "v2":            text_v2,
            "similarity":    round(best_score, 4),
            "changed":       True,
            "reasons":       reasons,
            "implication":   None,
            "level":         clause_v1.get("level", 1),
        })

    # v2 clauses never matched by any v1 clause → ADDED
    for j in range(m):
        if j in matched_v2_indices:
            continue
        text_v2 = clauses_v2[j]["text"]
        if _is_section_header(text_v2):
            continue
        clause_counter += 1
        results.append({
            "clause_index":  clause_counter,
            "result_type":   "added",
            "label":         "added_obligation",
            "v1":            None,
            "v2":            text_v2,
            "similarity":    0.0,
            "changed":       True,
            "reasons":       ["new clause with no equivalent in v1"],
            "implication":   None,
            "level":         clauses_v2[j].get("level", 1),
        })

    return results