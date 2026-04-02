import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from loader import load_document
from pdf_loader import load_pdf
from segmenter import segment_into_clauses
from comparator import pair_and_compare
from classifier import classify_batch
from optimizer import embed_with_cache

LABEL_SYMBOLS = {
    "narrowed_scope":     "◀  NARROWED SCOPE",
    "expanded_scope":     "▶  EXPANDED SCOPE",
    "numeric_change":     "#  NUMERIC CHANGE",
    "reversed_burden":    "⇄  REVERSED BURDEN",
    "added_obligation":   "+  ADDED OBLIGATION",
    "removed_obligation": "-  REMOVED OBLIGATION",
    "added_carve_out":    "+  ADDED CARVE-OUT",
    "removed_carve_out":  "-  REMOVED CARVE-OUT",
    "negation_shift":     "!  NEGATION SHIFT",
    "unknown":            "?  UNKNOWN",
}

TYPE_ICONS = {
    "changed": "⚠",
    "added":   "✚",
    "removed": "✖",
}


def _load_file(filepath: str) -> str:
    if filepath.lower().endswith(".pdf"):
        return load_pdf(filepath)
    return load_document(filepath)


def print_report(results: list[dict]):
    WIDTH = 72
    print("\n" + "═" * WIDTH)
    print("  SEMANTIC DIFF ENGINE — REPORT")
    print("═" * WIDTH)

    if not results:
        print("\n  No meaningful changes detected between the two documents.\n")
        print("═" * WIDTH)
        return

    for r in results:
        rtype  = r.get("result_type", "changed")
        label  = r.get("label", "unknown")
        symbol = LABEL_SYMBOLS.get(label, label)
        icon   = TYPE_ICONS.get(rtype, "⚠")
        indent = "    " * (r.get("level", 1) - 1)

        print(f"\n{indent}{icon}  [{symbol}]  score: {r['similarity']:.4f}")

        if r.get("reasons"):
            print(f"{indent}   flags  : {', '.join(r['reasons'])}")

        if rtype == "removed":
            print(f"{indent}   removed: {r['v1']}")
        elif rtype == "added":
            print(f"{indent}   added  : {r['v2']}")
        else:
            print(f"{indent}   v1     : {r['v1']}")
            print(f"{indent}   v2     : {r['v2']}")

        if r.get("implication"):
            print(f"{indent}   impact : {r['implication']}")

        print(f"{indent}{'─' * (WIDTH - len(indent))}")

    # Summary
    from collections import Counter
    total   = len(results)
    changed = sum(1 for r in results if r["result_type"] == "changed")
    added   = sum(1 for r in results if r["result_type"] == "added")
    removed = sum(1 for r in results if r["result_type"] == "removed")
    labels  = Counter(r.get("label","") for r in results)

    print(f"\n  {'─' * 30}")
    print(f"  Changed clauses  : {changed}")
    print(f"  Added clauses    : {added}")
    print(f"  Removed clauses  : {removed}")
    print(f"  Total flagged    : {total}")
    if labels:
        print(f"\n  By type:")
        for lbl, cnt in labels.most_common():
            print(f"    {LABEL_SYMBOLS.get(lbl, lbl):<28} {cnt}")
    print(f"\n{'═' * WIDTH}\n")


def main():
    t0 = time.time()

    PATH_V1 = "sample_docs/contract_v1.pdf"
    PATH_V2 = "sample_docs/contract_v2.pdf"

    print("\n[1/5] Loading documents...")
    doc_v1 = _load_file(PATH_V1)
    doc_v2 = _load_file(PATH_V2)

    print("[2/5] Segmenting into clauses...")
    clauses_v1 = segment_into_clauses(doc_v1)
    clauses_v2 = segment_into_clauses(doc_v2)
    print(f"      {len(clauses_v1)} clauses in v1 | {len(clauses_v2)} in v2")

    print("[3/5] Embedding (cache-aware)...")
    emb_v1 = embed_with_cache(clauses_v1)
    emb_v2 = embed_with_cache(clauses_v2)

    print("[4/5] Nearest-neighbor matching + cosine comparison...")
    results = pair_and_compare(clauses_v1, emb_v1, clauses_v2, emb_v2)
    print(f"      {len(results)} actionable differences found")

    print("[5/5] Classifying changes with LLM...")
    classify_batch(results)

    print_report(results)
    print(f"  Completed in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()