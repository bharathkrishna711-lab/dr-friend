"""
uc2_conversation.py

Multi-turn, symptoms-only conversation manager for UC2. Gathers as much
relevant detail as possible (symptoms, duration, severity, associated
factors) WITHOUT ever collecting vitals -- that is UC1's differentiator.
UC2 exists specifically for patients without measurement devices.

Once the LLM judges it has enough information, it signals readiness the
same way UC1's conversation.py does, via a [READY_TO_ANALYSE] tag. This
keeps the design pattern consistent across both use cases, and reuses
the same "LLM judges readiness from content, not a fixed turn count"
rationale already justified in UC1.

WHY A SEPARATE FILE FROM UC1's conversation.py:
Per the outline's architecture decision, there is no top-level routing
agent -- the patient explicitly selects UC1/UC2/UC3 at entry. Keeping
UC2's conversation logic fully separate avoids any risk of UC2 changes
affecting UC1's already-tested, working flow.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from retriever import ask_dr_friend_uc2
from urgency_uc2 import check_uc2_urgency
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MAX_TURNS = 7  # Same safety cap as UC1, same rationale: prevent an
                # unresponsive or vague conversation from looping forever.

UC2_CONVERSATION_SYSTEM_PROMPT = """You are Dr. Friend, an AI healthcare guidance assistant for Indian patients
who may not have access to medical devices like thermometers or blood
pressure monitors.

Your goal in this conversation is to gather relevant detail about the
patient's symptoms through natural conversation:
- What symptoms are they experiencing (be specific)
- How long have symptoms been present (onset, duration)
- How severe or how symptoms are progressing (getting better/worse)
- Any relevant associated factors (recent travel, food, exposure to illness)

CRITICAL RULE -- NEVER ASK FOR VITALS:
Do NOT ask for temperature, heart rate, blood pressure, oxygen saturation
(SpO2), or any other measured vital sign. This conversation pathway exists
specifically for patients without access to measurement devices.

Keep your questions natural, empathetic, and concise -- one or two focused
questions per turn, not an overwhelming checklist."""


SUMMARIZER_PROMPT = """Given the following conversation between a patient and Dr. Friend, write
ONE clean, consolidated paragraph describing the patient's symptoms,
duration, severity, and any relevant associated factors mentioned.

Do not include any greetings, questions, or conversational filler --
only the factual symptom description, written as the patient would
describe it to a doctor.

CONVERSATION:
{conversation_text}

Consolidated symptom summary:"""

def run_uc2_pipeline(conversation_history: list) -> dict:
    """
    The full UC2 pipeline, run once check_readiness() confirms enough
    symptom information has been gathered. Mirrors UC1's results
    structure conceptually -- a single results screen showing the
    generated guidance, its sources, and a deterministic urgency level --
    but built on RAG retrieval + symptom-only urgency classification
    instead of UC1's ML model + vitals-based NEWS2 engine.

    Returns a single dict ready to hand to the results screen, so
    app.py doesn't need to know about the individual RAG/urgency
    functions underneath -- same separation of concerns as UC1's
    prediction_agent.py packaging ML + Layer 2 output for app.py.
    """
    # Step 1: consolidate the conversation into one clean symptom summary
    symptom_summary = summarize_symptoms(conversation_history)

    # Step 2: retrieve grounded guidance + real citations
    rag_result = ask_dr_friend_uc2(symptom_summary)

    # Step 3: classify urgency from the same retrieved context, so the
    # urgency decision is grounded in the same source material as the
    # guidance text -- not a separate, potentially inconsistent lookup
    from retriever import load_vector_store
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(symptom_summary)
    retrieved_context = "\n\n".join(doc.page_content for doc in docs)

    urgency_result = check_uc2_urgency(symptom_summary, retrieved_context)

    return {
        "symptom_summary": symptom_summary,
        "guidance": rag_result["answer"],
        "sources": rag_result["sources"],
        "source_excerpts": rag_result["source_excerpts"],
        "chunks_retrieved": rag_result["retrieved_chunks"],
        "urgency_level": urgency_result["urgency_level"],
        "urgency_matched_criteria": urgency_result["matched_criteria"],
        "urgency_reasoning": urgency_result["reasoning"],
    }


def get_uc2_response(conversation_history: list) -> str:
    """
    Generates Dr. Friend's next response in the UC2 symptom-only
    conversation. Same statelessness principle as UC1's
    get_dr_friend_response() -- the full history is resent every call,
    since the LLM has no memory between calls.
    """
    conversation_text = ""
    for message in conversation_history:
        role = "Patient" if message["role"] == "user" else "Dr. Friend"
        conversation_text += f"{role}: {message['content']}\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", UC2_CONVERSATION_SYSTEM_PROMPT),
        ("human", "CONVERSATION SO FAR:\n{conversation}\n\nDr. Friend's next response:"),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    response = chain.invoke({"conversation": conversation_text})
    return response

READINESS_CHECK_PROMPT = """Given the following conversation between a patient and Dr. Friend, determine
if ENOUGH information has been gathered to proceed with analysis.

Enough information means ALL three of these are known:
1. At least 2-3 specific named symptoms
2. Duration/onset (how long symptoms have been present)
3. Severity or progression (getting better/worse/staying the same)

CONVERSATION:
{conversation_text}

Respond with ONLY the word "READY" if all three items above are known,
or ONLY the word "NOT_READY" if any item is still missing. No other text."""


def check_readiness(conversation_history: list) -> bool:
    """
    Separate, focused LLM call whose only job is judging readiness --
    split from get_uc2_response() because asking one call to both
    converse naturally AND reliably append a hidden tag proved
    inconsistent in testing (the same class of instruction-following
    unreliability observed in UC1's Layer 2). A dedicated call with a
    single, narrow yes/no task is more reliable than a compound
    instruction bundled into a conversational response.
    """
    conversation_text = ""
    for message in conversation_history:
        role = "Patient" if message["role"] == "user" else "Dr. Friend"
        conversation_text += f"{role}: {message['content']}\n\n"

    prompt = ChatPromptTemplate.from_template(READINESS_CHECK_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    result = chain.invoke({"conversation_text": conversation_text})
    return "READY" in result.upper() and "NOT_READY" not in result.upper()


def summarize_symptoms(conversation_history: list) -> str:
    """
    Converts the full multi-turn conversation into one clean, consolidated
    symptom description -- used as the query sent to the RAG retriever
    and urgency classifier, instead of the raw multi-turn transcript.

    WHY THIS STEP EXISTS:
    A raw transcript is noisier for embedding-based retrieval than one
    clean sentence -- it includes Dr. Friend's own questions, patient
    filler words, and repeated information. Consolidating first keeps
    the retrieval query focused purely on symptom content, and keeps
    results consistent regardless of how many turns the conversation
    took to gather that information.
    """
    conversation_text = ""
    for message in conversation_history:
        role = "Patient" if message["role"] == "user" else "Dr. Friend"
        conversation_text += f"{role}: {message['content']}\n\n"

    prompt = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    # WHY temperature=0.0: this is a factual consolidation task, not
    # creative generation -- we want the summary to stay strictly
    # faithful to what was actually said, not embellished.
    parser = StrOutputParser()

    chain = prompt | llm | parser
    summary = chain.invoke({"conversation_text": conversation_text})
    return summary


if __name__ == "__main__":
    history = [
        {"role": "assistant", "content": "Hello! I'm Dr. Friend. What symptoms have you been experiencing?"},
        {"role": "user", "content": "I've had a high fever and bad headache behind my eyes for 2 days"},
    ]

    response1 = get_uc2_response(history)
    history.append({"role": "assistant", "content": response1})

    history.append({"role": "user", "content": "Also joint pain and I feel nauseous, no bleeding or anything like that, symptoms seem to be getting worse"})
    response2 = get_uc2_response(history)
    history.append({"role": "assistant", "content": response2})

    if check_readiness(history):
        print("=" * 70)
        print("RUNNING FULL UC2 PIPELINE")
        print("=" * 70)
        result = run_uc2_pipeline(history)

        print(f"\nSYMPTOM SUMMARY:\n{result['symptom_summary']}")
        print(f"\nGUIDANCE:\n{result['guidance']}")
        print(f"\nSOURCES: {result['sources']}")
        print(f"CHUNKS RETRIEVED: {result['chunks_retrieved']}")
        print(f"\nURGENCY LEVEL: {result['urgency_level']}")
        print(f"MATCHED CRITERIA: {result['urgency_matched_criteria']}")
        print(f"URGENCY REASONING: {result['urgency_reasoning']}")