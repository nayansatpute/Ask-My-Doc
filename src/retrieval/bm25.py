
import os

from rank_bm25 import BM25Okapi
import chromadb

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../../vectorstore")
COLLECTION_NAME = "documents"

def get_all_chunks() -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.get()
    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]

def bm25_search(query: str, n_results: int = 5) -> list[dict]:
    chunks = get_all_chunks()
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [{"text": chunks[i]["text"], "source": chunks[i]["source"], "rank": rank}
            for rank, i in enumerate(top_indices)]