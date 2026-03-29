from sentence_transformers import SentenceTransformer

# Load the model once
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_clauses(clauses: list[str]) -> list[list[float]]:
    """
    Sends all clauses to a local SentenceTransformer model.
    Returns a list of embedding vectors, one per clause.
    Each vector is 384 dimensions for 'all-MiniLM-L6-v2'.
    """
    # Encode all clauses at once
    embeddings = model.encode(clauses, convert_to_numpy=True)
    
    # Convert to list of lists
    vectors = embeddings.tolist()
    return vectors