import os
import re
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

    if not q:
        return {
            "ignore": True,
            "standalone_query": "",
        }

    REFERENTIAL_PATTERN = re.compile(
        r"\b(this|that|it|those|they|them|he|she|him|her|"
        r"above|previous|earlier|mentioned|same|such)\b",
        re.IGNORECASE,
    )

    is_referential = bool(REFERENTIAL_PATTERN.search(q))

    if not is_referential:
        return {
            "ignore": False,
            "standalone_query": q,
        }

    if not recent_history:
        return {
            "ignore": False,
            "standalone_query": q,
        }

    try:
        context = "\n".join(
            f"User: {t.get('question','')}\nAI: {t.get('answer','')}"
            for t in recent_history[-3:]
        )

        prompt = f"""
You are a reference resolver.

TASK:
- Resolve pronouns or vague references using ONLY the chat context.
- DO NOT reframe the question.
- DO NOT infer intent.
- DO NOT generalize.
- DO NOT add or remove meaning.
- If resolution is unclear, return the original question unchanged.

Chat context:
{context}

User question:
{q}

Resolved standalone question:
""".strip()

        response = client.chat.completions.create(
            model=SUPERVISOR_MODEL,
            messages=[
                {"role": "system", "content": "Resolve references only. Do not rewrite meaning."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=128,
        )

        rewritten = response.choices[0].message.content.strip()

        return {
            "ignore": False,
            "standalone_query": rewritten if rewritten else q,
        }

    except Exception as e:
        print(f" understand_query failed: {e}")


    return {
        "ignore": False,
        "standalone_query": q,
    }
