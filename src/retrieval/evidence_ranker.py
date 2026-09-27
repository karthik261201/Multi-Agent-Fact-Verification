# find passages most relevant to claim

# Sentence Transformers converts text into embeddings and supports similarity calculations between those embeddings, 
# which is exactly what we need for semantic evidence ranking.

from sentence_transformers import SentenceTransformer


# Load the embedding model. We load it once when this module starts instead
# of loading it every time rank_evidence() runs.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def rank_evidence(claim, passages, top_k=5):
    """
    Rank evidence passages according to how semantically
    relevant they are to the original claim.

    Parameters:
        claim (str): Original claim entered by the user.

        passages (list[dict]): Evidence passages collected from webpages.

        top_k (int): Number of strongest passages to return.

    Returns:
        list[dict]:Top-ranked evidence passages.
    """

    if not passages:
        return []

    # Extract only the text because this is what the
    # Sentence Transformer model needs to encode.
    passage_texts = [ passage["text"] for passage in passages ]

    # Convert the user's claim into an embedding.
    claim_embedding = model.encode([claim])

    # Convert all evidence passages into embeddings.
    passage_embeddings = model.encode(passage_texts)

    # Calculate semantic similarity between the claim and every evidence passage.
    similarity_scores = model.similarity(claim_embedding, passage_embeddings)[0]

    ranked_passages = []

    # Attach the similarity score back to each
    # evidence passage.
    for passage, score in zip(passages, similarity_scores):

        evidence = passage.copy()

        evidence["relevance_score"] = float(score)

        ranked_passages.append(evidence)

    # Highest similarity should appear first.
    ranked_passages.sort(key=lambda item: item["relevance_score"], reverse=True)

    # Return only the strongest evidence.
    return ranked_passages[:top_k]