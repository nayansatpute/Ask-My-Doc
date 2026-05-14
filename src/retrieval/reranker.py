from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, results: list[dict], top_n: int = 3) -> list[dict]:
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)
    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]