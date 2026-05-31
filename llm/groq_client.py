"""
llm/groq_client.py - rename this file to groq_client.py

Using Groq with LLaMA 3.1 70B as primary LLM.
Groq is free, extremely fast, and works reliably.

WHY GROQ:
- Free tier with generous limits
- Fastest inference available (good for live demos)
- LLaMA 3.1 70B is strong for entity extraction
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"


def call_llm(prompt: str) -> str:
    """
    Simple wrapper around Groq API call.
    Function named call_llm (not call_groq) so if we
    swap LLM later, rest of code stays the same.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"LLM call failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_response = call_llm("Say hello in one sentence.")
    print("Response:", test_response)
    print("Connection successful.")