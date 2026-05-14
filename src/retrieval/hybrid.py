from semantic import semantic_search
from bm25 import bm25_search

def reciprocal_rank_fusion(semantic_results, bm25_results, k=60) -> list[dict]:
    scores = {}
    all_docs = {}

    for results in [semantic_results, bm25_results]:
        for item in results:
            text = item["text"]
            rank = item["rank"]
            scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)
            all_docs[text] = item["source"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": text, "source": all_docs[text], "score": score} for text, score in ranked]

def hybrid_search(query: str, n_results: int = 5) -> list[dict]:
    semantic = semantic_search(query, n_results=10)
    bm25 = bm25_search(query, n_results=10)
    fused = reciprocal_rank_fusion(semantic, bm25)
    return fused[:n_results]