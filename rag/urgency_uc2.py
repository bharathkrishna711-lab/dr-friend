"""
urgency_uc2.py

Symptom-only urgency assessment for UC2, since no vitals are available in
this pathway (that is UC1's differentiator). Rather than freely generating
advice text, this constrains the LLM to a narrow criteria-matching task:
given the patient's reported symptoms and the retrieved document's own
"When to See a Doctor" / "When It's an Emergency" bullets, output one of
four fixed urgency levels by checking which explicit criteria are met.

WHY THIS IS DIFFERENT FROM FREE-TEXT ADVICE GENERATION:
UC1's urgency engine is fully rule-based and deterministic because it has
numeric vitals to check against fixed thresholds (SpO2 < 93, etc). UC2 has
no vitals, so a numeric rule engine cannot apply directly. Instead of
falling back to purely free-text LLM advice ("if X, see a doctor" prose,
which is what earlier UC2 testing produced), this function constrains the
LLM to classification against explicit, document-sourced criteria only --
closer in spirit to NEWS2's deterministic philosophy, adapted for a
symptom-only context. The LLM is not asked to invent thresholds; it is
asked to match against thresholds already written into the retrieved
WHO/NHS source text.

WHY NOT REUSE core/urgency_engine.py DIRECTLY:
That engine's rules are built around numeric vital thresholds (Rules 1-5)
that UC2 will never receive. Only its symptom-combination logic (Rule 6)
and disease-category floor logic (Rule 7) are conceptually reusable, and
those are reimplemented here in a form that works from retrieved document
criteria rather than a fixed in-code rule table -- since UC2's disease
coverage is meant to grow as more documents are added, unlike UC1's fixed
17-class rule table.
"""

from dotenv import load_dotenv
load_dotenv()

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Fixed urgency levels -- same four levels as UC1, for consistency across
# the whole application, even though the underlying logic differs.
URGENCY_LEVELS = [
    "Self-Care at Home",
    "See a Doctor Soon",
    "See a Doctor Today",
    "Go to Emergency",
]

URGENCY_PROMPT = """You are a strict criteria-matching classifier, not a general medical advisor.

You will be given:
1. A patient's reported symptoms
2. Reference text retrieved from a trusted medical document, which contains
   explicit "When to See a Doctor" and "When It's an Emergency" criteria

Your ONLY task is to check which explicit criteria the patient's symptoms
match, and classify the urgency into exactly one of these four levels:
- "Self-Care at Home" -- no criteria from either section are met
- "See a Doctor Soon" -- mild concern, but no explicit "see a doctor" criteria clearly met yet
- "See a Doctor Today" -- one or more explicit "When to See a Doctor" criteria are met
- "Go to Emergency" -- one or more explicit "When It's an Emergency" criteria are met

RULES:
- Only match against criteria EXPLICITLY present in the reference text below.
- Do not invent new criteria or use outside medical knowledge.
- If any "When It's an Emergency" criterion is met, always classify as
  "Go to Emergency" -- this overrides all other levels.
- Return ONLY a JSON object, no other text, in exactly this format:
{{
    "urgency_level": "<one of the four levels above>",
    "matched_criteria": ["<criterion text that was matched, if any>"],
    "reasoning": "<one sentence explaining the match>"
}}

REFERENCE TEXT (retrieved from trusted medical document):
{context}

PATIENT'S REPORTED SYMPTOMS:
{symptoms}

Your JSON response:"""


def check_uc2_urgency(symptoms: str, retrieved_context: str) -> dict:
    """
    Classifies urgency by matching patient symptoms against explicit
    criteria in the retrieved document context. Returns a structured
    dict, not free text, so the result is safe to display deterministically
    in the UI (e.g. a colored urgency badge, same as UC1's results screen).
    """
    prompt = ChatPromptTemplate.from_template(URGENCY_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    # WHY temperature=0.0 here specifically (lower than UC2's answer
    # generation at 0.2): this is a classification task, not a generation
    # task. We want maximum consistency, not any creative variation.
    parser = StrOutputParser()

    chain = prompt | llm | parser
    raw_response = chain.invoke({
        "context": retrieved_context,
        "symptoms": symptoms,
    })

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        # Fallback if the LLM ever returns malformed JSON -- fail safe
        # toward caution rather than crashing or silently under-triaging.
        return {
            "urgency_level": "See a Doctor Today",
            "matched_criteria": [],
            "reasoning": "Could not parse urgency classification; defaulting to a cautious level.",
        }

    # Validate the returned level is actually one of our four fixed
    # options -- if the LLM drifts from the exact string (same class of
    # issue we saw in UC1's Layer 2), fail safe rather than trust it blindly.
    if result.get("urgency_level") not in URGENCY_LEVELS:
        result["urgency_level"] = "See a Doctor Today"
        result["reasoning"] = result.get("reasoning", "") + " (urgency level normalized to nearest valid option)"

    return result


if __name__ == "__main__":
    # Quick manual test using a real UC2 retrieval result
    from retriever import ask_dr_friend_uc2, load_vector_store

    test_symptoms = "Mild throbbing headache on one side since this morning, a bit sensitive to light, no other symptoms"

    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(test_symptoms)
    context = "\n\n".join(doc.page_content for doc in docs)

    result = check_uc2_urgency(test_symptoms, context)
    print(f"SYMPTOMS: {test_symptoms}\n")
    print(f"URGENCY LEVEL: {result['urgency_level']}")
    print(f"MATCHED CRITERIA: {result['matched_criteria']}")
    print(f"REASONING: {result['reasoning']}")