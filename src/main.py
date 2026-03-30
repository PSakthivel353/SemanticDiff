import sys
import os


# Make sure Python can find the src/ modules when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from loader import load_document
from pdf_loader import load_pdf
from segmenter import segment_into_clauses
from embedder import embed_clauses
from comparator import pair_and_compare
from classifier import classify_batch
from optimizer import embed_with_cache, deduplicate_for_classification


def load_file(filepath: str) -> str:
    if filepath.endswith(".pdf"):
        return load_pdf(filepath)
    return load_document(filepath)

def print_report(results: list[dict]):
    print("\n" + "=" * 70)
    print("  SEMANTIC DIFF REPORT")
    print("=" * 70)

    for r in results:
        status = "⚠  CHANGED" if r["changed"] else "✓  UNCHANGED"
        indent = "  " * (r["level"] - 1)  # indent sub-clauses visually

        print(f"\n{indent}Clause {r['label_v1']}  |  Score: {r['similarity']:.4f}  |  {status}")
        if r["changed"]:
            print(f"{indent}  Reason: {', '.join(r['reasons'])}")
        print(f"{indent}  V1: {r['v1']}")
        print(f"{indent}  V2: {r['v2']}")
        print("-" * 70)

    changed = sum(1 for r in results if r["changed"])
    print(f"\nSummary: {changed} of {len(results)} clauses changed.\n")


def main():
    # Step 1: Load both documents
    print("Loading documents...")
    doc_v1 = load_file("sample_docs/contract_v1.pdf")
    doc_v2 = load_file("sample_docs/contract_v2.pdf")

    # Step 2: Split into clauses
    print("Segmenting into clauses...")
    clauses_v1 = segment_into_clauses(doc_v1)
    clauses_v2 = segment_into_clauses(doc_v2)

    print(f"  Found {len(clauses_v1)} clauses in v1, {len(clauses_v2)} in v2.")

    print("Embedding (with cache)...")
    emb_v1 = embed_with_cache(clauses_v1)
    emb_v2 = embed_with_cache(clauses_v2)

    print("Comparing...")
    pairs = pair_and_compare(clauses_v1, emb_v1, clauses_v2, emb_v2)

    print("Classifying changed clauses...")
    to_classify, already_done = deduplicate_for_classification(pairs)
    classified = classify_batch(to_classify)

    # Step 4: Compare clause pairs
    print("Computing cosine similarity scores...")
    results = pair_and_compare(clauses_v1, emb_v1, clauses_v2, emb_v2)

    print("Classifying changed clauses...")
    to_classify, already_done = deduplicate_for_classification(pairs)
    classified = classify_batch(to_classify)

    all_results = classified + already_done
    all_results.sort(key=lambda x: x["clause_index"])

    # Step 5: Print the report
    print_report(all_results)


if __name__ == "__main__":
    main()