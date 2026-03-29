import re

# All the patterns legal documents use for clause numbering
CLAUSE_PATTERNS = [
    r"^\d+\.\d+\.\d+",         # 1.2.3  (sub-sub-section)
    r"^\d+\.\d+",              # 1.2    (sub-section)
    r"^\d+\.",                 # 1.     (top-level numbered)
    r"^\([ivxlcdm]+\)",        # (i), (ii), (iv), (xiv)  — Roman numerals
    r"^\([a-z]\)",             # (a), (b), (c)
    r"^\([A-Z]\)",             # (A), (B), (C)
    r"^[A-Z]{2,}[\s:]",        # WHEREAS, RECITALS, DEFINITIONS, NOW THEREFORE
    r"^Article\s+\d+",         # Article 1, Article 12
    r"^Section\s+\d+",         # Section 1, Section 4.2
    r"^Schedule\s+[A-Z\d]+",   # Schedule A, Schedule 1
    r"^Clause\s+\d+",          # Clause 1
]

# Compile all into one master pattern
MASTER_PATTERN = re.compile(
    "|".join(f"({p})" for p in CLAUSE_PATTERNS),
    re.IGNORECASE
)


def is_clause_start(line: str) -> bool:
    """
    Returns True if a line looks like the beginning of a legal clause.
    Checks against all known numbering patterns.
    """
    return bool(MASTER_PATTERN.match(line.strip()))


def segment_into_clauses(document: str) -> list[dict]:
    """
    Splits a document into clauses. Each clause is returned as a dict with:
      - 'label': the clause number/letter (e.g. "1.", "(a)", "WHEREAS")
      - 'text': the full clause text (may span multiple lines)
      - 'level': nesting depth (1 = top, 2 = sub, 3 = sub-sub)

    Handles multi-line clauses: lines that don't start a new clause
    are appended to the current clause's text.
    """
    lines = document.strip().split("\n")
    clauses = []
    current_clause = None

    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip blank lines

        if is_clause_start(line):
            # Save the previous clause before starting a new one
            if current_clause:
                clauses.append(current_clause)

            label = _extract_label(line)
            current_clause = {
                "label": label,
                "text": line,
                "level": _get_nesting_level(label)
            }
        else:
            # This line is a continuation of the current clause
            if current_clause:
                current_clause["text"] += " " + line

    # Don't forget the last clause
    if current_clause:
        clauses.append(current_clause)

    return clauses


def _extract_label(line: str) -> str:
    """
    Pulls just the clause identifier from the start of a line.
    e.g. "1.2 The tenant shall..." → "1.2"
         "(a) Pets are not..."    → "(a)"
    """
    match = MASTER_PATTERN.match(line.strip())
    if match:
        return match.group(0).strip()
    return ""


def _get_nesting_level(label: str) -> int:
    """
    Determines how deeply nested a clause is based on its label format.
    Level 1: top-level  (1., Article 1, WHEREAS)
    Level 2: sub        (1.1, (a), (i))
    Level 3: sub-sub    (1.1.1)
    """
    if re.match(r"^\d+\.\d+\.\d+", label):
        return 3
    if re.match(r"^\d+\.\d+", label) or re.match(r"^\([a-z]\)|^\([A-Z]\)|^\([ivxlcdm]+\)", label, re.I):
        return 2
    return 1