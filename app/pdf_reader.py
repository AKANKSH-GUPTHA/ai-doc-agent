from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Loads a PDF and splits it into small chunks.
    Each chunk = ~500 characters with 50 character overlap.
    """

    # Step 1: Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} pages from PDF")

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Split into {len(chunks)} chunks")

    return chunks