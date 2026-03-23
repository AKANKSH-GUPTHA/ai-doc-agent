from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

def answer_question(question: str, context_chunks: list) -> str:
    context = "\n\n".join([chunk.page_content for chunk in context_chunks])

    prompt = ChatPromptTemplate.from_template("""
You are an intelligent document assistant. 
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer clearly and concisely:
""")

    chain = prompt | get_llm()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content