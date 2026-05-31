"""
llm/openai_client.py

GPT-4o-mini client setup.
Better structured output than Groq for entity extraction.

WHY GPT-4o-mini:
- More reliable JSON output
- Better at mapping casual language to medical terms
- No daily token limits for our usage level
- Cost: ~$0.002 per conversation (negligible)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"


def call_llm(prompt: str) -> str:
    """
    Wrapper around OpenAI API call.
    Same function name as groq_client.py so
    rest of code works without changes.
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
        print(f"OpenAI API call failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_response = call_llm("Say hello in one sentence.")
    print("Response:", test_response)
    print("Connection successful.")