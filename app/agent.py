from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from app.vector_store import search_documents
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def search_document(query: str) -> str:
    """Search the uploaded document for specific information."""
    chunks = search_documents(query, k=4)
    if not chunks:
        return "No document uploaded yet."
    return "\n\n".join([c.page_content for c in chunks])

@tool
def summarize_document(topic: str) -> str:
    """Summarize the document or a specific topic from it."""
    chunks = search_documents(topic, k=6)
    if not chunks:
        return "No document uploaded yet."
    return "\n\n".join([c.page_content for c in chunks])

@tool
def extract_key_info(info_type: str) -> str:
    """Extract specific info like skills, experience, education, projects."""
    chunks = search_documents(info_type, k=4)
    if not chunks:
        return "No document uploaded yet."
    return "\n\n".join([c.page_content for c in chunks])

def run_agent(user_query: str) -> str:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    tools = [search_document, summarize_document, extract_key_info]

    # Bind tools to LLM and run directly without AgentExecutor
    llm_with_tools = llm.bind_tools(tools)

    # Build tool map
    tool_map = {t.name: t for t in tools}

    # Decide which tool to use based on query
    if any(word in user_query.lower() for word in ["summarize", "summary", "overview"]):
        result = summarize_document.invoke(user_query)
    elif any(word in user_query.lower() for word in ["skill", "experience", "education", "project"]):
        result = extract_key_info.invoke(user_query)
    else:
        result = search_document.invoke(user_query)

    # Pass result to LLM for final answer
    prompt = f"""Based on this document content:

{result}

Answer this query: {user_query}

Provide a clear, structured answer:"""

    response = llm.invoke(prompt)
    return response.content