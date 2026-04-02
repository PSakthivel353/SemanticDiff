import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHANGE_LABELS = [
    "narrowed_scope",
    "expanded_scope",
    "numeric_change",
    "reversed_burden",
    "added_obligation",
    "removed_obligation",
    "added_carve_out",
    "removed_carve_out",
    "negation_shift",
]

FEW_SHOT_EXAMPLES = """
Example 1:
V1: The tenant shall pay rent of $1000 per month.
V2: The tenant shall pay rent of $1200 per month.
Label: numeric_change
Implication: Monthly rent increased by $200, directly raising the tenant's financial burden.

Example 2:
V1: The landlord shall provide 30 days written notice before terminating this agreement.
V2: The landlord shall provide 14 days written notice before terminating this agreement.
Label: numeric_change
Implication: Tenant's protection window before termination was cut in half, reducing time to find alternative arrangements.

Example 3:
V1: The tenant is responsible for all utility bills including electricity and water.
V2: The tenant is responsible for electricity bills only. Water is covered by the landlord.
Label: narrowed_scope
Implication: Tenant's financial obligations were narrowed — water costs shifted to the landlord.

Example 4:
V1: Pets are not permitted on the premises under any circumstances.
V2: Pets under 10kg are permitted on the premises with prior written approval.
Label: added_carve_out
Implication: An absolute prohibition was replaced with a conditional permission for small pets.

Example 5:
V1: The contractor shall not be liable for any indirect or consequential damages.
V2: The contractor shall be liable for all indirect and consequential damages.
Label: negation_shift
Implication: A complete liability exclusion was reversed into full liability — drastically increasing the contractor's legal exposure.

Example 6:
V1: (c) "Affiliate" means any subsidiary of a Party.
V2: (c) "Affiliate" means any subsidiary or parent company of a Party.
Label: expanded_scope
Implication: The definition of Affiliate was broadened to include parent companies, expanding the range of entities covered by the agreement.
""".strip()

SYSTEM_PROMPT = f"""You are a legal analyst specializing in contract review.

Given two versions of a legal clause (V1 = original, V2 = revised), classify the change and explain its legal implication.

Respond in EXACTLY this format:
Label: <label>
Implication: <one sentence explaining who is affected and how>

Available labels: {", ".join(CHANGE_LABELS)}

Rules:
- Focus on legal/practical impact, not linguistic difference.
- The implication must state WHO is affected and HOW (burden increased/decreased, right expanded/removed).
- Never add text outside the Label/Implication format.

Examples:
{FEW_SHOT_EXAMPLES}
"""

ADDED_PROMPT = """You are a legal analyst. A new clause was added to a revised document that did not exist in the original.

Clause text: {text}

Respond in EXACTLY this format:
Implication: <one sentence explaining what new obligation, right, or restriction this clause introduces and who it affects>
"""

REMOVED_PROMPT = """You are a legal analyst. A clause from the original document was removed entirely in the revised version.

Clause text: {text}

Respond in EXACTLY this format:
Implication: <one sentence explaining what obligation, right, or protection was lost by removing this clause and who is affected>
"""


def _call_llm(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()


def _parse_changed(raw: str) -> tuple[str, str]:
    """Parses Label + Implication from a changed-clause response."""
    label = "unknown"
    implication = raw
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("label:"):
            candidate = line.split(":", 1)[1].strip().lower().replace(" ", "_")
            if candidate in CHANGE_LABELS:
                label = candidate
        elif line.lower().startswith("implication:"):
            implication = line.split(":", 1)[1].strip()
    return label, implication


def _parse_implication_only(raw: str) -> str:
    """Parses just the Implication line from an added/removed response."""
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("implication:"):
            return line.split(":", 1)[1].strip()
    return raw


def classify_batch(pairs: list[dict]) -> list[dict]:
    """
    Classifies each result dict based on its result_type:
      - 'changed'  → ask LLM to label + explain the change
      - 'added'    → ask LLM to explain what the new clause introduces
      - 'removed'  → ask LLM to explain what was lost

    Mutates each dict in-place, adding 'label' and 'implication'.
    Returns the same list.
    """
    for pair in pairs:
        rtype = pair.get("result_type", "changed")

        try:
            if rtype == "changed":
                raw = _call_llm(
                    SYSTEM_PROMPT,
                    f"V1: {pair['v1']}\nV2: {pair['v2']}"
                )
                label, impl = _parse_changed(raw)
                pair["label"]       = label
                pair["implication"] = impl

            elif rtype == "added":
                raw = _call_llm(
                    "You are a legal analyst.",
                    ADDED_PROMPT.format(text=pair["v2"])
                )
                pair["label"]       = "added_obligation"
                pair["implication"] = _parse_implication_only(raw)

            elif rtype == "removed":
                raw = _call_llm(
                    "You are a legal analyst.",
                    REMOVED_PROMPT.format(text=pair["v1"])
                )
                pair["label"]       = "removed_obligation"
                pair["implication"] = _parse_implication_only(raw)

        except Exception as e:
            pair["label"]       = "unknown"
            pair["implication"] = f"Classification failed: {e}"

    return pairs