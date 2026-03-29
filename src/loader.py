def load_document(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()