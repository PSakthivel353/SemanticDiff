import sys
import os


# Make sure Python can find the src/ modules when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from loader import load_document
from segmenter import segment_into_clauses
from embedder import embed_clauses
from comparator import pair_and_compare


def print_report(results: list[dict]):
    """
    Prints a human-readable diff report to the terminal.
    Shows each clause pair with its similarity score and a CHANGED / UNCHANGED label.
    """
    print("\n" + "=" * 70)
    print("  SEMANTIC DIFF REPORT")
    print("=" * 70)

    for r in results:
        status = "⚠  CHANGED" if r["changed"] else "✓  UNCHANGED"
        score_display = f"{r['similarity']:.4f}"

        print(f"\nClause {r['clause_index']}  |  Similarity: {score_display}  |  {status}")
        print(f"  V1: {r['v1']}")
        print(f"  V2: {r['v2']}")
        print("-" * 70)

    changed_count = sum(1 for r in results if r["changed"])
    print(f"\nSummary: {changed_count} of {len(results)} clauses semantically changed.\n")


def main():
    # Step 1: Load both documents
    print("Loading documents...")
    doc_v1 = load_document("sample_docs/contract_v1.txt")
    doc_v2 = load_document("sample_docs/contract_v2.txt")

    # Step 2: Split into clauses
    print("Segmenting into clauses...")
    clauses_v1 = segment_into_clauses(doc_v1)
    clauses_v2 = segment_into_clauses(doc_v2)

    print(f"  Found {len(clauses_v1)} clauses in v1, {len(clauses_v2)} in v2.")

    # Step 3: Embed all clauses (2 API calls — one per document)
    print("Generating embeddings via OpenAI...")
    embeddings_v1 = embed_clauses(clauses_v1)
    embeddings_v2 = embed_clauses(clauses_v2)

    # Step 4: Compare clause pairs
    print("Computing cosine similarity scores...")
    results = pair_and_compare(clauses_v1, embeddings_v1, clauses_v2, embeddings_v2)

    # Step 5: Print the report
    print_report(results)


if __name__ == "__main__":
    main()