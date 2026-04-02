import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import json
import tempfile
from collections import Counter

from loader import load_document
from pdf_loader import load_pdf
from segmenter import segment_into_clauses
from comparator import pair_and_compare
from classifier import classify_batch
from optimizer import embed_with_cache

st.set_page_config(
    page_title="Semantic Diff Engine",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0e0e0f;
    color: #d4d0c8;
}
h1, h2, h3 { font-family: 'Fraunces', serif; font-weight: 600; }

.clause-card {
    border: 1px solid #2a2a2e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    background: #141416;
}
.clause-card.changed  { border-left: 3px solid #e07b4f; }
.clause-card.added    { border-left: 3px solid #4caf84; }
.clause-card.removed  { border-left: 3px solid #d45a5a; }

.badge { display:inline-block; padding:2px 10px; border-radius:4px;
         font-size:11px; font-weight:500; letter-spacing:0.04em;
         text-transform:uppercase; margin-bottom:8px; }

.badge-narrowed_scope    { background:#3b2a1a; color:#e07b4f; }
.badge-expanded_scope    { background:#1a2e28; color:#4caf84; }
.badge-numeric_change    { background:#2a2a1a; color:#d4b84a; }
.badge-reversed_burden   { background:#2e1a2a; color:#c47fc4; }
.badge-added_obligation  { background:#1a2030; color:#5a9fd4; }
.badge-removed_obligation{ background:#301a1a; color:#d45a5a; }
.badge-added_carve_out   { background:#1a2e28; color:#4caf84; }
.badge-removed_carve_out { background:#2e1a1a; color:#e07b4f; }
.badge-negation_shift    { background:#2e1a1a; color:#e05a5a; }
.badge-unknown           { background:#222;    color:#888; }

.rtype-added   { color:#4caf84; font-size:11px; font-weight:500;
                 text-transform:uppercase; margin-left:8px; }
.rtype-removed { color:#d45a5a; font-size:11px; font-weight:500;
                 text-transform:uppercase; margin-left:8px; }

.text-v1   { color:#c47070; font-size:13px; line-height:1.7; }
.text-v2   { color:#70b88a; font-size:13px; line-height:1.7; }
.text-only { color:#d4d0c8; font-size:13px; line-height:1.7; }
.implication { color:#9a96c8; font-size:13px; font-style:italic; margin-top:8px; }
.score { color:#555; font-size:11px; }
.flags { color:#666; font-size:11px; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────

def load_uploaded(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    return load_pdf(tmp_path) if suffix == ".pdf" else load_document(tmp_path)


def run_pipeline(text_v1: str, text_v2: str) -> list[dict]:
    """Full pipeline. Called only when user clicks Submit."""
    with st.status("Running pipeline...", expanded=True) as status:

        st.write("Segmenting clauses...")
        clauses_v1 = segment_into_clauses(text_v1)
        clauses_v2 = segment_into_clauses(text_v2)
        st.write(f"Found {len(clauses_v1)} clauses in V1, {len(clauses_v2)} in V2.")

        st.write("Generating embeddings...")
        emb_v1 = embed_with_cache(clauses_v1)
        emb_v2 = embed_with_cache(clauses_v2)

        st.write("Matching clauses by semantic similarity...")
        results = pair_and_compare(clauses_v1, emb_v1, clauses_v2, emb_v2)
        st.write(f"{len(results)} meaningful differences found.")

        st.write("Classifying changes with LLM...")
        classify_batch(results)

        status.update(label="Analysis complete.", state="complete")

    return results


def render_card(r: dict):
    """Renders one result as a styled HTML card."""
    rtype = r.get("result_type", "changed")
    label = r.get("label", "unknown")
    score = r["similarity"]
    impl  = r.get("implication", "")
    flags = r.get("reasons", [])

    # Build inner HTML based on result type
    if rtype == "added":
        type_tag   = "<span class='rtype-added'>✚ NEW CLAUSE</span>"
        score_part = ""   # no score for added (no v1 match)
        text_part  = f"<div class='text-only'>▸ {r['v2']}</div>"
    elif rtype == "removed":
        type_tag   = "<span class='rtype-removed'>✖ REMOVED CLAUSE</span>"
        score_part = ""
        text_part  = f"<div class='text-only'>▸ {r['v1']}</div>"
    else:
        type_tag   = ""
        score_part = f"<span class='score'>similarity: {score:.4f}</span>"
        text_part  = (
            f"<div class='text-v1'>▸ V1: {r['v1']}</div>"
            f"<div class='text-v2'>▸ V2: {r['v2']}</div>"
        )

    flag_part = (
        f"<div class='flags'>flags: {', '.join(flags)}</div>"
        if flags and rtype == "changed" else ""
    )

    impl_part = (
        f"<div class='implication'>⟶ {impl}</div>"
        if impl else ""
    )

    html = f"""
    <div class="clause-card {rtype}">
        <span class="badge badge-{label}">{label.replace("_"," ")}</span>
        {type_tag}
        {score_part}
        {flag_part}
        {text_part}
        {impl_part}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────

st.markdown("# ⚖ Semantic Diff Engine")
st.markdown(
    "<p style='color:#555;font-size:13px;margin-top:-12px'>"
    "Upload two versions of a legal or policy document — "
    "only meaningful changes are shown.</p>",
    unsafe_allow_html=True
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Document V1** — original")
    file_v1 = st.file_uploader(
        "Upload V1", type=["txt","pdf"], key="v1", label_visibility="collapsed"
    )
with col2:
    st.markdown("**Document V2** — amended")
    file_v2 = st.file_uploader(
        "Upload V2", type=["txt","pdf"], key="v2", label_visibility="collapsed"
    )

# Submit button — pipeline only runs when clicked
both_uploaded = file_v1 is not None and file_v2 is not None
submit = st.button(
    "⟶  Run Analysis",
    disabled=not both_uploaded,
    type="primary",
    help="Upload both documents first, then click to run."
)

if not both_uploaded:
    st.markdown(
        "<div style='text-align:center;padding:48px 0;color:#333;font-size:13px'>"
        "Upload both documents to enable analysis.</div>",
        unsafe_allow_html=True
    )

elif submit:
    text_v1 = load_uploaded(file_v1)
    text_v2 = load_uploaded(file_v2)
    results = run_pipeline(text_v1, text_v2)
    st.session_state["results"] = results   # persist so filters don't re-run pipeline

# Show results if they exist in session (survives filter widget interactions)
if "results" in st.session_state:
    results = st.session_state["results"]

    if not results:
        st.success("No meaningful differences found between the two documents.")
    else:
        # ── Summary metrics ───────────────────────────────────────────
        st.divider()
        total   = len(results)
        changed = sum(1 for r in results if r["result_type"] == "changed")
        added   = sum(1 for r in results if r["result_type"] == "added")
        removed = sum(1 for r in results if r["result_type"] == "removed")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total flagged", total)
        m2.metric("Changed",  changed)
        m3.metric("Added",    added)
        m4.metric("Removed",  removed)

        # Label breakdown pills
        label_counts = Counter(r.get("label","") for r in results)
        if label_counts:
            st.markdown("**Change types detected**")
            pills_html = " ".join(
                f"<span class='badge badge-{lbl}'>{lbl.replace('_',' ')} ({cnt})</span>"
                for lbl, cnt in label_counts.most_common()
            )
            st.markdown(pills_html, unsafe_allow_html=True)

        # ── Filter controls ───────────────────────────────────────────
        st.divider()
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            show_filter = st.selectbox(
                "Show",
                ["All changes", "Changed only", "Added only", "Removed only"],
            )
        with fc2:
            sort_by = st.selectbox(
                "Sort by",
                ["Document order", "Similarity (low→high)"]
            )

        display = results.copy()
        if show_filter == "Changed only":
            display = [r for r in display if r["result_type"] == "changed"]
        elif show_filter == "Added only":
            display = [r for r in display if r["result_type"] == "added"]
        elif show_filter == "Removed only":
            display = [r for r in display if r["result_type"] == "removed"]

        if sort_by == "Similarity (low→high)":
            display.sort(key=lambda x: x["similarity"])

        st.markdown(
            f"<p style='color:#444;font-size:12px'>{len(display)} results shown</p>",
            unsafe_allow_html=True
        )

        for r in display:
            render_card(r)

        # ── Download ──────────────────────────────────────────────────
        st.divider()
        export = [{
            "type":        r["result_type"],
            "label":       r.get("label",""),
            "similarity":  r["similarity"],
            "v1":          r.get("v1",""),
            "v2":          r.get("v2",""),
            "implication": r.get("implication",""),
        } for r in results]

        st.download_button(
            "↓ Download report (JSON)",
            data=json.dumps(export, indent=2),
            file_name="semantic_diff_report.json",
            mime="application/json",
        )