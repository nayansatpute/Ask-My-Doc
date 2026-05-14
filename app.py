import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/retrieval'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/generation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/ingestion'))

from hybrid import hybrid_search
from reranker import rerank
from llm_client import generate_answer
from document_loader import load_document
from chunker import chunk_text
from vector_store import store_chunks

st.set_page_config(page_title="Ask My Doc", page_icon="📄")
st.title("📄 Ask My Doc")
st.caption("Upload a document and ask questions about it.")

with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or Markdown file", type=["pdf", "md", "txt"])
    if uploaded_file:
        save_path = f"data/raw/{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Ingesting document..."):
            text = load_document(save_path)
            chunks = chunk_text(text)
            store_chunks(chunks, uploaded_file.name)
        st.success(f"✅ Ingested {len(chunks)} chunks!")

st.header("💬 Ask a Question")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching and generating answer..."):
        results = hybrid_search(query, n_results=5)
        reranked = rerank(query, results, top_n=3)
        answer = generate_answer(query, reranked)

    st.markdown("### 🤖 Answer")
    st.write(answer)

    with st.expander("📚 Source Chunks"):
        for i, chunk in enumerate(reranked):
            st.markdown(f"**Chunk {i+1}** — `{chunk['source']}`")
            st.write(chunk["text"])
            st.divider()