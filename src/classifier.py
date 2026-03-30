import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Change labels the model can choose from ──────────────────────────
CHANGE_LABELS = [
    "narrowed_scope",        # obligation/right now covers less
    "expanded_scope",        # obligation/right now covers more
    "numeric_change",        # a number changed (amount, days, %)
    "reversed_burden",       # who bears responsibility flipped
    "added_obligation",      # new duty introduced
    "removed_obligation",    # duty eliminated
    "added_carve_out",       # new exception/exclusion added
    "removed_carve_out",     # exception removed (tightened)
    "negation_shift",        # shall → shall not, or vice versa
    "cosmetic_only",         # wording changed, meaning unchanged
]

# ── Few-shot examples baked into the prompt ──────────────────────────
# These teach the model what each label looks like in practice.
# The more examples you add here, the better it gets. No retraining needed.
FEW_SHOT_EXAMPLES = """
Example 1:
V1: The tenant shall pay rent of $1000 per month.
V2: The tenant shall pay rent of $1200 per month.
Label: numeric_change
Implication: The monthly rent obligation increased by $200, directly increasing tenant's financial burden.

Example 2:
V1: The landlord shall provide 30 days written notice before terminating this agreement.
V2: The landlord shall provide 14 days written notice before terminating this agreement.
Label: numeric_change
Implication: Tenant's protection window before termination was cut in half, significantly reducing their ability to find alternative housing.

Example 3:
V1: The tenant is responsible for all utility bills including electricity and water.
V2: The tenant is responsible for electricity bills only. Water is covered by the landlord.
Label: narrowed_scope
Implication: Tenant's financial obligations were narrowed — water costs shifted to the landlord, reducing tenant's recurring expenses.

Example 4:
V1: Pets are not permitted on the premises under any circumstances.
V2: Pets under 10kg are permitted on the premises with prior written approval from the landlord.
Label: added_carve_out
Implication: An absolute prohibition was replaced with a conditional permission, creating a new carve-out for small pets subject to landlord approval.

Example 5:
V1: The contractor shall not be liable for any indirect or consequential damages.
V2: The contractor shall be liable for all indirect and consequential damages.
Label: negation_shift
Implication: A complete liability exclusion was reversed into full liability — a drastic increase in the contractor's legal exposure.

Example 6:
V1: Either party may terminate this agreement with 60 days written notice.
V2: Either party may terminate this agreement with 60 days written notice.
Label: cosmetic_only
Implication: No meaningful change — the clause is identical in obligation and scope.
""".strip()

# ── System prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a legal analyst specializing in contract and policy document review.

Your job: given two versions of a legal clause (V1 = original, V2 = revised), classify the change and explain its legal implication.

You MUST respond in this exact format — nothing else:
Label: <one of the labels below>
Implication: <one sentence, plain English, explaining the legal impact of the change>

Available labels: {", ".join(CHANGE_LABELS)}

Rules:
- If the clause is unchanged or only cosmetically reworded, use cosmetic_only.
- Focus on the legal/practical impact, not just the linguistic difference.
- The Implication sentence must explain WHO is affected and HOW (burden increased/decreased, right expanded/removed, etc.)
- Never add commentary outside the Label/Implication format.

Here are examples of correct classifications:

{FEW_SHOT_EXAMPLES}
"""


def classify_change(text_v1: str, text_v2: str) -> dict:
    """
    Sends a clause pair to the LLM and returns a classification dict:
      - label: one of the CHANGE_LABELS
      - implication: plain-English legal impact sentence
      - raw: the full model response (for debugging)

    Uses Groq's llama3-8b-instruct — free, fast, good enough for this task.
    Swap to claude-3-haiku or gpt-4o-mini if you want higher accuracy.
    """
    user_message = f"V1: {text_v1}\nV2: {text_v2}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,      # low temp = consistent, deterministic output
        max_tokens=120,
    )

    raw = response.choices[0].message.content.strip()
    return _parse_response(raw)


def _parse_response(raw: str) -> dict:
    """
    Parses the model's Label/Implication response into a clean dict.
    Falls back gracefully if the model didn't follow the format exactly.
    """
    label = "unknown"
    implication = raw  # fallback: return raw if parsing fails

    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("label:"):
            label = line.split(":", 1)[1].strip().lower().replace(" ", "_")
        elif line.lower().startswith("implication:"):
            implication = line.split(":", 1)[1].strip()

    # Validate label — if model hallucinated something, default to unknown
    if label not in CHANGE_LABELS:
        label = "unknown"

    return {"label": label, "implication": implication, "raw": raw}


def classify_batch(pairs: list[dict]) -> list[dict]:
    """
    Runs classify_change() on a list of comparator result dicts.
    Only classifies pairs where changed=True — skips unchanged ones.
    Adds 'label' and 'implication' keys to each dict in-place.
    Returns the enriched list.
    """
    for pair in pairs:
        if pair["changed"]:
            result = classify_change(pair["v1"], pair["v2"])
            pair["label"] = result["label"]
            pair["implication"] = result["implication"]
        else:
            pair["label"] = "cosmetic_only"
            pair["implication"] = "No meaningful change detected."
    return pairs