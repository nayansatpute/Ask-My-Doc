
import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../../vectorstore")
COLLECTION_NAME = "documents"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query: str, n_results: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return [
        {"text": doc, "source": meta["source"], "rank": i}
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]))
    ]