from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

# Test 1: Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("hello world")
print(f"✅ Embeddings working: vector size = {vec.shape}")

# Test 2: ChromaDB
client = chromadb.Client()
col = client.create_collection("test")
print("✅ ChromaDB working")

# Test 3: Groq
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one word"}]
)
print(f"✅ Groq working: {response.choices[0].message.content.strip()}")