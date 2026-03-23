from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Global variable to store our vector database in memory
vector_store = None

def get_embeddings():
    """
    Load the embedding model.
    This converts text into numbers (vectors) so we can search by meaning.
    """
    print("⏳ Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Small, fast, free model
    )
    print("✅ Embedding model loaded!")
    return embeddings

def create_vector_store(chunks):
    """
    Takes text chunks and stores them in FAISS vector database.
    """
    global vector_store

    print("⏳ Creating vector store...")
    embeddings = get_embeddings()

    # Convert chunks to vectors and store in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)

    print(f"✅ Vector store created with {len(chunks)} chunks!")
    return vector_store

def search_documents(query: str, k: int = 4):
    """
    Search for the most relevant chunks for a given query.
    k = number of chunks to return
    """
    global vector_store

    if vector_store is None:
        return []

    results = vector_store.similarity_search(query, k=k)
    return results