import re
import numpy as np

# Words that flip meaning — if these differ between clauses, flag it
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "except", "unless"}


def _extract_numbers(text: str) -> set[str]:
    """Pulls all numbers from a clause (dollar amounts, days, percentages, etc.)"""
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def _has_negation_shift(text_a: str, text_b: str) -> bool:
    """
    Returns True if one clause has a negation word the other doesn't.
    e.g. "shall be liable" vs "shall NOT be liable"
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    neg_in_a = words_a & NEGATION_WORDS
    neg_in_b = words_b & NEGATION_WORDS
    return neg_in_a != neg_in_b


def _has_number_shift(text_a: str, text_b: str) -> bool:
    """
    Returns True if the numbers in two clauses differ.
    Catches: $1000 → $1200, 30 days → 14 days, etc.
    """
    return _extract_numbers(text_a) != _extract_numbers(text_b)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def pair_and_compare(
    clauses_v1: list[dict],
    embeddings_v1: list[list[float]],
    clauses_v2: list[dict],
    embeddings_v2: list[list[float]]
) -> list[dict]:
    """
    Pairs clauses by position, computes similarity, and runs the
    number/negation guard on top. Returns enriched result dicts.
    """
    results = []
    num_pairs = min(len(clauses_v1), len(clauses_v2))

    for i in range(num_pairs):
        text_a = clauses_v1[i]["text"]
        text_b = clauses_v2[i]["text"]
        score = cosine_similarity(embeddings_v1[i], embeddings_v2[i])

        number_shift = _has_number_shift(text_a, text_b)
        negation_shift = _has_negation_shift(text_a, text_b)

        # Force-flag as changed if numbers or negation differ, regardless of score
        is_changed = score < 0.97 or number_shift or negation_shift

        # Build a human-readable reason
        reasons = []
        if score < 0.97:
            reasons.append("semantic drift")
        if number_shift:
            reasons.append("numeric value changed")
        if negation_shift:
            reasons.append("negation shift detected")

        results.append({
            "clause_index": i + 1,
            "label_v1": clauses_v1[i]["label"],
            "label_v2": clauses_v2[i]["label"],
            "level": clauses_v1[i]["level"],
            "v1": text_a,
            "v2": text_b,
            "similarity": round(score, 4),
            "changed": is_changed,
            "reasons": reasons if reasons else ["none"]
        })

    return results