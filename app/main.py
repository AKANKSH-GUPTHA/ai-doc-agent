from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.pdf_reader import load_and_chunk_pdf
from app.vector_store import create_vector_store, search_documents
import shutil
import os

app = FastAPI(title="AI Document Intelligence Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

@app.get("/")
def home():
    return {"message": "AI Doc Agent is running!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read and chunk the PDF
    chunks = load_and_chunk_pdf(file_path)

    # Store chunks in vector database
    create_vector_store(chunks)

    return {
        "message": "File uploaded, processed and indexed!",
        "filename": file.filename,
        "total_chunks": len(chunks),
    }

@app.post("/search")
def search(query: str):
    """Search for relevant chunks based on a query"""
    results = search_documents(query, k=4)

    return {
        "query": query,
        "results": [
            {
                "content": r.page_content,
                "page": r.metadata.get("page", 0)
            }
            for r in results
        ]
    }