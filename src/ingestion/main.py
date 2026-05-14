from document_loader import load_document
from chunker import chunk_text
from vector_store import store_chunks
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def ingest(file_path: str):
    print(f"Loading {file_path}...")
    text = load_document(file_path)
    print(f"Chunking...")
    chunks = chunk_text(text)
    print(f"Got {len(chunks)} chunks. Storing...")
    source_name = os.path.basename(file_path)
    store_chunks(chunks, source_name)

if __name__ == "__main__":
    file_path = sys.argv[1]
    ingest(file_path)