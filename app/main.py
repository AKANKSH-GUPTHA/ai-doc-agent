from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.pdf_reader import load_and_chunk_pdf
from app.vector_store import create_vector_store, search_documents
from app.llm import answer_question
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
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = load_and_chunk_pdf(file_path)
    create_vector_store(chunks)

    return {
        "message": "File uploaded, processed and indexed!",
        "filename": file.filename,
        "total_chunks": len(chunks),
    }

@app.post("/ask")
def ask_question(question: str):
    """Ask a question about the uploaded document"""

    # Step 1: Find relevant chunks using vector search
    relevant_chunks = search_documents(question, k=4)

    if not relevant_chunks:
        return {"answer": "Please upload a document first!"}

    # Step 2: Send chunks + question to LLM
    answer = answer_question(question, relevant_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources_used": len(relevant_chunks)
    }