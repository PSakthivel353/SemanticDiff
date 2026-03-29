import pdfplumber

def load_pdf(filepath: str) -> str:
    """
    Reads a PDF and returns all text as a single string.
    pdfplumber handles multi-column, tables, and most real-world PDFs cleanly.
    """
    full_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)