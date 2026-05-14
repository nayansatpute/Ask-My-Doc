# 📄 Ask My Doc

A RAG (Retrieval-Augmented Generation) system that answers questions over your documents using hybrid search and LLM generation.

## Architecture
- **Hybrid Retrieval** — BM25 + semantic search fused via Reciprocal Rank Fusion (RRF)
- **Reranking** — Cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- **Generation** — Groq API (Llama 3.1 8B)
- **Vector Store** — ChromaDB
- **Evaluation** — Ragas (Faithfulness: 1.0 | Answer Relevancy: 0.90)
- **UI** — Streamlit

## Quick Start
```bash
git clone https://github.com/nayansatpute/ask-my-doc
cd ask-my-doc
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
echo GROQ_API_KEY=your_key > .env
streamlit run app.py
```

## Evaluation
| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.90 |

## Stack
`Python` `ChromaDB` `sentence-transformers` `rank-bm25` `Groq` `Ragas` `Streamlit`