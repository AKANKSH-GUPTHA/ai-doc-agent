from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.pdf_reader import load_and_chunk_pdf
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

    return {
        "message": "File uploaded and processed!",
        "filename": file.filename,
        "total_pages": len(set([c.metadata['page'] for c in chunks])),
        "total_chunks": len(chunks),
        "sample_chunk": chunks[0].page_content
    }