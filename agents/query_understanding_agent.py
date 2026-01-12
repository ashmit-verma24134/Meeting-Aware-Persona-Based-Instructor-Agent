
FOLLOW_UP_KEYWORDS = ["that", "it", "earlier", "previous", "again"]

def is_follow_up(question: str) -> bool:   #Later will introduce langraph in this
 
    if not question:
        return False

    q = question.lower()
    return any(word in q for word in FOLLOW_UP_KEYWORDS)



def understand_query(question: str) -> str:

    if not question or len(question.strip()) < 3:
        return "out_of_scope"

    q = question.lower()

    # Explicit out-of-scope keywords
    out_of_scope_keywords = [
        "weather",
        "capital of",
        "who is",
        "define physics",
        "math formula",
        "faiss",
        "leetcode",
        "codeforces"
    ]

    for kw in out_of_scope_keywords:
        if kw in q:
            return "out_of_scope"

    
    return "meeting_content"

