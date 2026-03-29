import re

def segment_into_clauses(document: str) -> list[str]:
    """
    Splits a document into individual clauses.
    Looks for lines that start with a number and a dot (e.g. "1.", "2.").
    Returns a list of clause strings, stripped of leading/trailing whitespace.
    """
    # Split on lines that begin with a digit followed by a period
    lines = document.strip().split("\n")
    clauses = []

    for line in lines:
        line = line.strip()
        if re.match(r"^\d+\.", line):  # matches "1.", "2.", "10.", etc.
            clauses.append(line)

    return clauses