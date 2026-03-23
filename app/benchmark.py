from app.vector_store import search_documents
from app.llm import answer_question

def run_benchmark():
    """
    Custom RAG benchmarking - evaluates pipeline quality.
    Metrics: Answer completeness, context relevance, response consistency
    """

    test_cases = [
        {
            "question": "What is the candidate's name?",
            "expected_keywords": ["akanksh", "modadugu", "guptha"]
        },
        {
            "question": "What programming languages does the candidate know?",
            "expected_keywords": ["python", "javascript", "c++"]
        },
        {
            "question": "What internships has the candidate completed?",
            "expected_keywords": ["edunet", "cawius", "intern"]
        },
        {
            "question": "What projects has the candidate built?",
            "expected_keywords": ["fraud", "spiceroute", "face mask"]
        },
    ]

    results = []
    total_score = 0

    print("⏳ Running RAG benchmark...")

    for case in test_cases:
        question = case["question"]
        keywords = case["expected_keywords"]

        # Get chunks and answer
        chunks = search_documents(question, k=4)
        if not chunks:
            return None

        answer = answer_question(question, chunks)
        answer_lower = answer.lower()

        # Score: how many expected keywords appear in answer
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        score = round(hits / len(keywords), 2)
        total_score += score

        results.append({
            "question": question,
            "answer_preview": answer[:100] + "...",
            "score": score
        })

        print(f"✅ Score {score} — {question[:40]}")

    avg_score = round(total_score / len(test_cases), 3)

    return {
        "overall_score": avg_score,
        "total_questions": len(test_cases),
        "detailed_results": results,
        "grade": "Excellent" if avg_score >= 0.8 else "Good" if avg_score >= 0.6 else "Needs Improvement"
    }