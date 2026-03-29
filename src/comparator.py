import numpy as np

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measures how similar two vectors are in terms of direction.
    Returns a float between 0 and 1.
    1.0 = identical meaning, 0.0 = completely unrelated.

    Formula: dot(a, b) / (|a| * |b|)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # avoid division by zero

    return dot_product / (magnitude_a * magnitude_b)


def pair_and_compare(
    clauses_v1: list[str],
    embeddings_v1: list[list[float]],
    clauses_v2: list[str],
    embeddings_v2: list[list[float]]
) -> list[dict]:
    """
    Pairs each clause from v1 with the corresponding clause from v2 (by position).
    Computes cosine similarity for each pair.
    Returns a list of result dicts with the clause texts and their similarity score.
    """
    results = []
    num_pairs = min(len(clauses_v1), len(clauses_v2))  # handle unequal lengths

    for i in range(num_pairs):
        score = cosine_similarity(embeddings_v1[i], embeddings_v2[i])

        results.append({
            "clause_index": i + 1,
            "v1": clauses_v1[i],
            "v2": clauses_v2[i],
            "similarity": round(score, 4),
            "changed": score < 0.97  # threshold: below 0.97 = semantically changed
        })

    return results