from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

vector_store = None

# Load embedding model ONCE at startup
print("⏳ Loading embedding model at startup...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embedding model ready!")

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
    results = vector_store.similarity_search(query, k=k)
    return results