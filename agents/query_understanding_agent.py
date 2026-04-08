import os
import re
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SUPERVISOR_MODEL = "llama-3.1-8b-instant"

def understand_query(
    question: str,
    recent_history: Optional[List[Dict]] = None,
    user_id: Optional[str] = None,
) -> Dict:

    q = (question or "").strip()

    # Default state for any early exit
    default_response = {
        "ignore": False,
        "standalone_query": q,
        "is_hunting_for_exact_value": False
    }

    if not q:
        return {**default_response, "ignore": True, "standalone_query": ""}

    # 1. Broadly identify if we need LLM resolution 
    REFERENTIAL_PATTERN = re.compile(
        r"\b(this|that|it|those|they|them|he|she|him|her|"
        r"above|previous|earlier|mentioned|same|such)\b",
        re.IGNORECASE,
    )
    
    # Check for "specific value" keywords even in non-referential queries
    VALUE_KEYWORDS = ["id", "key", "secret", "token", "password", "url", "link", "arn"]
    q_lower = q.lower()
    is_hunting_hint = any(word in q_lower for word in VALUE_KEYWORDS)

    # If it's not referential AND we don't need intent classification, exit early
    # BUT since we want AI-driven intent detection, we should usually call the LLM
    # if it looks like a technical query.
    is_referential = bool(REFERENTIAL_PATTERN.search(q))
    
    if not is_referential and not is_hunting_hint:
        return default_response

    if not recent_history and not is_hunting_hint:
        return default_response

    try:
        context = "\n".join(
            f"User: {t.get('question','')}\nAI: {t.get('answer','')}"
            for t in recent_history[-3:]
        ) if recent_history else "No recent history."

        prompt = f"""
        TASK:
        1. Resolve pronouns or vague references using the Chat Context.
        2. Set "is_hunting_for_exact_value" to true ONLY IF the user is asking for a specific technical string (ID, API key, Secret, URL, Token, etc.).
        
        RETURN JSON ONLY:
        {{
            "standalone_query": "resolved question string",
            "is_hunting_for_exact_value": boolean
        }}

        Chat context:
        {context}

        User question:
        {q}
        """.strip()

        response = client.chat.completions.create(
            model=SUPERVISOR_MODEL,
            messages=[
                {"role": "system", "content": "You are a query analyzer. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )

        # Parse the structured response
        analysis = json.loads(response.choices[0].message.content)

        return {
            "ignore": False,
            "standalone_query": analysis.get("standalone_query", q),
            "is_hunting_for_exact_value": bool(analysis.get("is_hunting_for_exact_value", False))
        }

    except Exception as e:
        print(f"understand_query failed: {e}")
        return default_response