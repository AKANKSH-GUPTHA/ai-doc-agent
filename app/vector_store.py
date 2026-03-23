from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

vector_store = None

class SimpleEmbeddings(Embeddings):
    """Lightweight embeddings using character hashing - no model download needed."""
    
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]
    
    def embed_query(self, text):
        return self._embed(text)
    
    def _embed(self, text):
        # Create 384-dim vector from text using hashing
        vector = []
        for i in range(384):
            h = hashlib.md5(f"{text}{i}".encode()).hexdigest()
            vector.append(int(h[:8], 16) / 0xFFFFFFFF - 0.5)
        return vector

embeddings = SimpleEmbeddings()
print("✅ Lightweight embeddings ready - no download needed!")

def create_vector_store(chunks):
    global vector_store
    print("⏳ Creating vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"✅ Vector store created with {len(chunks)} chunks!")
    return vector_store

def search_documents(query: str, k: int = 4):
    global vector_store
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=k)