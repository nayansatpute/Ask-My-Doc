import chromadb
from sentence_transformers import SentenceTransformer
import os

CHROMA_PATH = "vectorstore"
COLLECTION_NAME = "documents"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)

def store_chunks(chunks: list[str], source_name: str):
    collection = get_collection()
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": source_name} for _ in chunks]
    )
    print(f"✅ Stored {len(chunks)} chunks from '{source_name}'")