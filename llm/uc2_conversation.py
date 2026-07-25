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

    
    # Step 2: retrieve grounded guidance + real citations. Also pass
    # the raw patient text (not just the LLM-paraphrased summary) so
    # retrieval can catch matches the paraphrase might have missed --
    # see Bug 22 follow-up.
    raw_patient_text = " ".join(
        msg["content"] for msg in conversation_history if msg["role"] == "user"
    )
    rag_result = ask_dr_friend_uc2(symptom_summary, raw_patient_text=raw_patient_text)

    # Step 3: classify urgency from the same retrieved context, so the
    # urgency decision is grounded in the same source material as the
    # guidance text -- not a separate, potentially inconsistent lookup
    from retriever import load_vector_store
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(symptom_summary)
    retrieved_context = "\n\n".join(doc.page_content for doc in docs)

    RED_FLAG_TERMS = [
        "severe", "chest pain", "can't breathe", "cant breathe", "cannot breathe",
        "difficulty breathing", "blood", "bleeding", "fainted", "fainting",
        "unconscious", "confusion", "confused", "sudden", "unbearable",
        "can't move", "cant move", "paralysis", "numbness", "seizure",
        "high fever", "won't stop", "wont stop", "worsening rapidly",
        "getting worse quickly", "passed out", "collapsed",
    ]

    NEGATION_WORDS = ["no", "not", "nothing", "without", "never", "isn't",
                      "isnt", "wasn't", "wasnt", "don't", "dont", "doesn't",
                      "doesnt", "hasn't", "hasnt", "haven't", "havent", "none"]

    def has_unnegated_red_flag(text: str) -> bool:
        """
        Bug 21 fix: plain substring matching on RED_FLAG_TERMS treated
        "nothing severe" the same as "severe pain" -- a patient
        explicitly denying a red-flag symptom got escalated as if they
        had reported it. This checks a small window of words before
        each match for a negation word, and skips the match if found.
        """
        words = text.lower().split()
        text_lower = text.lower()
        for term in RED_FLAG_TERMS:
            if term not in text_lower:
                continue
            first_word = term.split()[0]
            for i, w in enumerate(words):
                if first_word in w:
                    window_start = max(0, i - 4)
                    preceding = words[window_start:i]
                    if not any(neg in preceding for neg in NEGATION_WORDS):
                        return True
        return False

    if rag_result.get("is_fallback_guess", False):
        # SAFETY FLOOR (Bug 13), TIERED (follow-up fix): when the
        # knowledge base had no confident match, urgency is never
        # computed by any LLM on unreliable retrieved context (see Bug
        # 13's original reasoning). But a SINGLE fixed floor
        # ("See a Doctor Today") was found to be poorly calibrated --
        # correct for genuinely concerning fallback cases (e.g. sudden
        # severe pain), but overly alarmist for genuinely mild fallback
        # cases (e.g. a mild, intermittent headache with slight stomach
        # discomfort), where every fallback case landed on the same
        # urgency regardless of how the symptoms actually read.
        #
        # FIX: a deterministic keyword scan (not an LLM judgment -- same
        # rule-based philosophy as the rest of this project) checks the
        # patient's own words for explicit red-flag language. If present,
        # keep the more urgent floor. If absent, use a gentler floor --
        # still recommending a doctor, never "Self-Care at Home", just
        # less alarmist wording for a case that reads as mild.
        has_red_flag = has_unnegated_red_flag(symptom_summary)

        if has_red_flag:
            urgency_result = {
                "urgency_level": "See a Doctor Today",
                "matched_criteria": [],
                "reasoning": (
                    "This symptom pattern did not match our verified knowledge "
                    "base, but includes language suggesting a potentially "
                    "serious presentation, so we are erring on the side of "
                    "caution. Please have this evaluated by a doctor rather "
                    "than relying on self-care."
                ),
            }
        else:
            urgency_result = {
                "urgency_level": "See a Doctor Soon",
                "matched_criteria": [],
                "reasoning": (
                    "This symptom pattern did not match our verified knowledge "
                    "base. It does not appear to include specific red-flag "
                    "language, but since we cannot fully assess it, we still "
                    "recommend a routine doctor visit rather than self-care alone."
                ),
            }
    else:
        urgency_result = check_uc2_urgency(symptom_summary, retrieved_context)

    return {
        "symptom_summary": symptom_summary, 
        "guidance": rag_result["answer"],
        "primary_condition": rag_result["primary_condition"],
        "is_fallback_guess": rag_result.get("is_fallback_guess", False),
        "sources": rag_result["sources"],
        "source_excerpts": rag_result["source_excerpts"],
        "chunks_retrieved": rag_result["retrieved_chunks"],
        "urgency_level": urgency_result["urgency_level"],
        "urgency_matched_criteria": urgency_result["matched_criteria"],
        "urgency_reasoning": urgency_result["reasoning"],
    }


MISSING_INFO_PROMPT = """Given the following conversation between a patient and Dr. Friend,
identify which of these three items are STILL MISSING or unclear:
1. At least 1-3 specific named symptoms (a single clearly named symptom
   is sufficient if that is genuinely all the patient has)
2. Duration/onset (how long symptoms have been present)
3. Severity or progression -- EITHER a severity indication (e.g. "5/10",
   "mild", "quite bad") OR a progression trend (getting better/worse/
   staying the same) is sufficient on its own, not both.

Read through the ENTIRE conversation carefully, including the patient's
answers to Dr. Friend's follow-up questions -- information is often
given in response to a question asked several lines earlier, not just
in the patient's very first message.

First, write ONE short line for each of the three items, stating
specifically what you found for that item in the conversation (or "not
found" if genuinely absent). Then, on a new final line, write ONLY the
missing item(s) by number and name (e.g. "2. Duration/onset"), or, if
all three were found, write exactly: NONE_MISSING

CONVERSATION:
{conversation_text}

Your analysis:"""


def identify_missing_info(conversation_history: list) -> str:
    """
    Separate, narrow LLM call that identifies specifically which
    readiness criteria are still unmet, so get_uc2_response() can
    target its next question directly at the actual gap instead of
    probing generically. Without this, the conversational LLM has no
    visibility into what check_readiness() is actually checking for,
    and can ask many turns of plausible-sounding questions without
    ever converging on the specific missing piece (see Bug 12).
    """
    conversation_text = ""
    for message in conversation_history:
        role = "Patient" if message["role"] == "user" else "Dr. Friend"
        conversation_text += f"{role}: {message['content']}\n\n"

    prompt = ChatPromptTemplate.from_template(MISSING_INFO_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    return chain.invoke({"conversation_text": conversation_text}).strip()


def get_uc2_response(conversation_history: list) -> str:
    """
    Generates Dr. Friend's next response in the UC2 symptom-only
    conversation. Same statelessness principle as UC1's
    get_dr_friend_response() -- the full history is resent every call,
    since the LLM has no memory between calls.

    Now checks which specific readiness criteria are still missing
    (see identify_missing_info / Bug 12) and steers the conversational
    prompt to target that gap directly, once the conversation has moved
    past the first couple of turns -- preventing indefinite generic
    follow-up questions that never converge on what check_readiness()
    actually needs.
    """
    conversation_text = ""
    for message in conversation_history:
        role = "Patient" if message["role"] == "user" else "Dr. Friend"
        conversation_text += f"{role}: {message['content']}\n\n"

    missing_info_note = ""
    if len(conversation_history) >= 3:
        missing = identify_missing_info(conversation_history)
        if missing != "NONE_MISSING":
            missing_info_note = (
                f"\n\nIMPORTANT: Specifically still missing: {missing}. "
                f"Your next question MUST directly ask for this missing "
                f"information rather than asking about a new, unrelated topic."
            )

    prompt = ChatPromptTemplate.from_messages([
        ("system", UC2_CONVERSATION_SYSTEM_PROMPT + missing_info_note),
        ("human", "CONVERSATION SO FAR:\n{conversation}\n\nDr. Friend's next response:"),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    response = chain.invoke({"conversation": conversation_text})

    # Bug 17 fix: the LLM occasionally echoes the "Speaker: message"
    # formatting pattern used internally to build conversation_text
    # (e.g. "Dr. Friend: ..."), likely because that pattern is visible
    # to it in the prompt's conversation history. Strip a leading
    # "Dr. Friend:" prefix if present, since the chat UI already shows
    # who is speaking via the avatar -- a literal prefix in the text
    # itself is redundant and looks like a formatting bug to the user.
    response = response.strip()
    for prefix in ["Dr. Friend:", "Dr Friend:", "Dr. Friend :"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            break

    return response

READINESS_CHECK_PROMPT = """Given the following conversation between a patient and Dr. Friend, determine
if ENOUGH information has been gathered to proceed with analysis.

Enough information means ALL three of these are known:
1. At least 1-3 specific named symptoms (a single clearly named symptom is
   sufficient if that is genuinely all the patient has -- do not require
   multiple symptoms to exist if the patient has only reported one)
2. Duration/onset (how long symptoms have been present)
3. Severity or progression -- EITHER a severity indication (e.g. "5/10",
   "mild", "quite bad") OR a progression trend (getting better/worse/
   staying the same) is sufficient on its own. Do NOT require both a
   severity rating AND an explicit trend statement -- one or the other
   is enough.

Read through the ENTIRE conversation carefully, including the patient's
answers to Dr. Friend's follow-up questions -- information is often
provided in response to a question asked several lines earlier, not
just in the patient's very first message.

First, write ONE short line for each of the three items above, stating
specifically what you found for that item (or "not found" if genuinely
absent from the conversation). Then, on a new final line, write ONLY
the word "READY" if all three items were found, or ONLY the word
"NOT_READY" if any item was genuinely not found.

CONVERSATION:
{conversation_text}

Your analysis:"""


def check_readiness(conversation_history: list) -> bool:
    """
    Separate, focused LLM call whose only job is judging readiness --
    split from get_uc2_response() because asking one call to both
    converse naturally AND reliably append a hidden tag proved
    inconsistent in testing (the same class of instruction-following
    unreliability observed in UC1's Layer 2). A dedicated call with a
    single, narrow yes/no task is more reliable than a compound
    instruction bundled into a conversational response.

    UPDATED (Bug 20 fix): the prompt now asks the LLM to first state
    what it found for each of the three criteria, THEN conclude
    READY/NOT_READY, rather than jumping straight to a bare verdict.
    Testing found the bare-verdict version could fail to recognize
    information that was clearly present in the transcript but
    provided in response to an earlier question (e.g. severity stated
    several lines after the question that asked for it) -- this
    "show your work first" structure gives the LLM room to explicitly
    attribute information across turns before judging, which is a
    more reliable pattern for this kind of multi-part reasoning task.
    Also loosened criterion 1 to accept a single named symptom, since
    the original "2-3 symptoms" wording could never be satisfied by a
    genuinely single-symptom presentation (e.g. an isolated headache
    with no other symptoms), regardless of how many turns were run.
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

    # Only the LAST line should carry the verdict now, since earlier
    # lines contain the per-criterion reasoning (which may legitimately
    # contain the substring "not found" etc. -- checking the whole
    # response could misfire on those lines).
    last_line = result.strip().split("\n")[-1].upper()
    return "READY" in last_line and "NOT_READY" not in last_line


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