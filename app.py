import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import json
import tempfile

from loader import load_document
from pdf_loader import load_pdf
from segmenter import segment_into_clauses
from comparator import pair_and_compare
from classifier import classify_batch
from optimizer import embed_with_cache, deduplicate_for_classification

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Diff Engine",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0e0e0f;
    color: #d4d0c8;
}
h1, h2, h3 { font-family: 'Fraunces', serif; font-weight: 600; }

/* Clause card */
.clause-card {
    border: 1px solid #2a2a2e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    background: #141416;
}
.clause-card.changed  { border-left: 3px solid #e07b4f; }
.clause-card.unchanged{ border-left: 3px solid #3a7d5c; }

/* Label badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.badge-narrowed_scope    { background:#3b2a1a; color:#e07b4f; }
.badge-expanded_scope    { background:#1a2e28; color:#4caf84; }
.badge-numeric_change    { background:#2a2a1a; color:#d4b84a; }
.badge-reversed_burden   { background:#2e1a2a; color:#c47fc4; }
.badge-added_obligation  { background:#1a2030; color:#5a9fd4; }
.badge-removed_obligation{ background:#301a1a; color:#d45a5a; }
.badge-added_carve_out   { background:#1a2e28; color:#4caf84; }
.badge-removed_carve_out { background:#2e1a1a; color:#e07b4f; }
.badge-negation_shift    { background:#2e1a1a; color:#e05a5a; }
.badge-cosmetic_only     { background:#1e1e1e; color:#777; }
.badge-unknown           { background:#222; color:#888; }

/* Diff text */
.text-v1 { color: #c47070; font-size: 13px; line-height: 1.7; }
.text-v2 { color: #70b88a; font-size: 13px; line-height: 1.7; }
.implication { color: #9a96c8; font-size: 13px; font-style: italic; margin-top: 8px; }
.score { color: #666; font-size: 11px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────

def load_file(uploaded_file) -> str:
    """Saves the uploaded file to a temp path and reads it."""
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    if suffix == ".pdf":
        return load_pdf(tmp_path)
    return load_document(tmp_path)


@st.cache_data(show_spinner=False)
def run_pipeline(text_v1: str, text_v2: str) -> list[dict]:
    """
    Full pipeline: segment → embed → compare → classify.
    Cached by Streamlit — re-runs only when inputs change.
    This is the main performance optimization for the UI.
    """
    clauses_v1 = segment_into_clauses(text_v1)
    clauses_v2 = segment_into_clauses(text_v2)

    emb_v1 = embed_with_cache(clauses_v1)
    emb_v2 = embed_with_cache(clauses_v2)

    pairs = pair_and_compare(clauses_v1, emb_v1, clauses_v2, emb_v2)

    to_classify, already_done = deduplicate_for_classification(pairs)
    classified = classify_batch(to_classify)

    # Merge back and sort by clause index
    all_results = classified + already_done
    all_results.sort(key=lambda x: x["clause_index"])
    return all_results


def render_clause_card(r: dict):
    """Renders one clause comparison as a styled HTML card."""
    css_class = "changed" if r["changed"] else "unchanged"
    label = r.get("label", "cosmetic_only")
    implication = r.get("implication", "")
    score = r["similarity"]

    card_html = f"""
    <div class="clause-card {css_class}">
        <span class="badge badge-{label}">{label.replace("_", " ")}</span>
        <span class="score" style="margin-left:10px">similarity: {score:.4f}</span>
        <div class="text-v1">▸ V1: {r['v1']}</div>
        <div class="text-v2">▸ V2: {r['v2']}</div>
        {"<div class='implication'>⟶ " + implication + "</div>" if r["changed"] else ""}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────

st.markdown("# ⚖ Semantic Diff Engine")
st.markdown(
    "<p style='color:#666;font-size:13px;margin-top:-12px'>"
    "Upload two versions of a legal or policy document. "
    "The engine finds what the <em>meaning</em> changed — not just the words.</p>",
    unsafe_allow_html=True
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Document V1** — original")
    file_v1 = st.file_uploader("Upload V1", type=["txt", "pdf"], key="v1", label_visibility="collapsed")
with col2:
    st.markdown("**Document V2** — amended")
    file_v2 = st.file_uploader("Upload V2", type=["txt", "pdf"], key="v2", label_visibility="collapsed")

# ── Run ───────────────────────────────────────────────────────────────

if file_v1 and file_v2:
    with st.spinner("Reading documents..."):
        text_v1 = load_file(file_v1)
        text_v2 = load_file(file_v2)

    with st.spinner("Running pipeline — segmenting, embedding, classifying..."):
        results = run_pipeline(text_v1, text_v2)

    # ── Summary bar ───────────────────────────────────────────────────
    total = len(results)
    changed = sum(1 for r in results if r["changed"])
    unchanged = total - changed

    from collections import Counter
    label_counts = Counter(r.get("label","") for r in results if r["changed"])

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total clauses", total)
    m2.metric("Changed", changed, delta=f"{changed/total*100:.0f}%", delta_color="inverse")
    m3.metric("Unchanged", unchanged)

    # Label breakdown
    if label_counts:
        st.markdown("**Change breakdown**")
        cols = st.columns(min(len(label_counts), 4))
        for i, (label, count) in enumerate(label_counts.most_common()):
            cols[i % 4].markdown(
                f"<span class='badge badge-{label}'>{label.replace('_',' ')}</span>"
                f"<span style='color:#666;font-size:12px;margin-left:6px'>{count}</span>",
                unsafe_allow_html=True
            )

    # ── Filter controls ───────────────────────────────────────────────
    st.divider()
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        show_filter = st.selectbox(
            "Show",
            ["All clauses", "Changed only", "Unchanged only"],
            key="filter"
        )
    with filter_col2:
        sort_by = st.selectbox("Sort by", ["Clause order", "Similarity (low→high)"], key="sort")

    # Apply filters
    display = results.copy()
    if show_filter == "Changed only":
        display = [r for r in display if r["changed"]]
    elif show_filter == "Unchanged only":
        display = [r for r in display if not r["changed"]]

    if sort_by == "Similarity (low→high)":
        display.sort(key=lambda x: x["similarity"])

    # ── Render cards ──────────────────────────────────────────────────
    st.markdown(f"<p style='color:#555;font-size:12px'>{len(display)} clauses shown</p>",
                unsafe_allow_html=True)

    for r in display:
        render_clause_card(r)

    # ── Download ──────────────────────────────────────────────────────
    st.divider()
    export = [{
        "clause": r["clause_index"],
        "label": r.get("label",""),
        "similarity": r["similarity"],
        "changed": r["changed"],
        "implication": r.get("implication",""),
        "v1": r["v1"],
        "v2": r["v2"],
    } for r in results]

    st.download_button(
        label="↓ Download report as JSON",
        data=json.dumps(export, indent=2),
        file_name="semantic_diff_report.json",
        mime="application/json",
    )

else:
    st.markdown(
        "<div style='text-align:center;padding:60px 0;color:#333;font-size:14px'>"
        "Upload both documents above to begin analysis.</div>",
        unsafe_allow_html=True
    )