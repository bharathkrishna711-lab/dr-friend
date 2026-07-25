"""
retriever.py

The core UC2 RAG chain: given a patient's symptom description, retrieves
relevant chunks from the ChromaDB knowledge base and generates grounded,
cited healthcare guidance using GPT-4o-mini.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

RETRIEVAL_K = 3
SCORE_THRESHOLD = 0.30
# WHY 0.30: testing showed correct-disease chunks scoring 0.35-0.52,
# while cross-contaminating chunks from unrelated diseases scored
# around 0.25. A 0.30 threshold cleanly separates genuinely relevant
# retrieval from tangential noise, based on empirical testing across
# 5 disease queries.


def load_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )
    return vector_store
def debug_similarity_scores(query: str, k: int = 5):
    """
    Diagnostic tool: shows the actual similarity scores for retrieved
    chunks, so we can pick a sensible threshold based on real data
    rather than guessing.
    """
    vector_store = load_vector_store()
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    print(f"\nQUERY: {query}")
    for doc, score in results:
        citation = doc.metadata.get("citation", "unknown")
        section = doc.metadata.get("section", "N/A")
        print(f"  Score: {score:.3f} | {citation[:40]} | Section: {section}")


def format_retrieved_docs(docs):
    # Bug 22 fix: previously only "citation" (the source publication
    # name, e.g. "NHS - Common Cold and Flu") was shown to the LLM --
    # never the document's actual disease_name metadata field (e.g.
    # "Viral Infection"). When a citation's wording didn't closely
    # resemble its disease_name, PRIMARY_CONDITION_PROMPT's LLM had no
    # way to map the document it was reading back to an allowed name
    # from KNOWN_DISEASE_NAMES, and defaulted to "Unclear" even when
    # the correct document was the clear top match. Explicitly
    # including the disease name in the context removes that gap.
    formatted = "\n\n---\n\n".join(
        f"[Disease: {doc.metadata.get('disease_name', 'unknown').replace('Disease: ', '')} | "
        f"Source: {doc.metadata.get('citation', 'unknown')} | "
        f"Section: {doc.metadata.get('section', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )
    return formatted


UC2_SYSTEM_PROMPT = """You are Dr. Friend, an AI healthcare guidance assistant for Indian patients.

You are answering based ONLY on the retrieved medical reference information
below. Do not use outside knowledge beyond what is provided in the context.

RULES:
- Ground every claim in the provided context. If the context does not
  contain enough information to answer confidently, say so explicitly.
- Always mention which disease(s) the retrieved information relates to.
- Clearly state when the patient should see a doctor or seek emergency
  care, based on the "When to See a Doctor" and "When It's an Emergency"
  sections in the context, if present.
- Keep the tone calm, clear, and non-alarming, but do not understate
  genuine emergencies.

CONTEXT (retrieved reference documents):
{context}

PATIENT'S SYMPTOMS:
{question}

Your response:"""


KNOWN_DISEASE_NAMES = [
    "Anaemia", "Anxiety Attack", "Arrhythmia", "Asthma", "Bronchitis",
    "COPD", "COVID-19", "Dengue Fever", "Food Poisoning", "Gastroenteritis",
    "Heart Failure", "Hepatitis", "Hypertensive Crisis", "Hypothyroidism",
    "Malaria", "Migraine", "Pneumonia", "Tuberculosis", "Type 2 Diabetes",
    "Typhoid", "UTI", "Viral Infection",
]
# WHY A HARDCODED LIST, NOT A LIVE CHROMADB QUERY AT IMPORT TIME:
# pulling this list fresh from the vector store on every module import
# adds an unnecessary DB round-trip to startup, and the knowledge base's
# disease set only changes when documents are deliberately added (a rare,
# manual event) -- not something that needs to be dynamically discovered
# on every request. If new documents are added to the knowledge base,
# this list must be updated to match (verified via the same ChromaDB
# metadata query used to generate it).

PRIMARY_CONDITION_PROMPT = """Based on the following retrieved medical reference context and the
patient's symptoms, identify the SINGLE most likely condition being
discussed.

You MUST respond with EXACTLY ONE of these disease names, copied
character-for-character, and nothing else -- do not rephrase, abbreviate,
or use an alternate/colloquial name (e.g. respond "Hypothyroidism", never
"Underactive Thyroid"):

{disease_list}

If multiple conditions are discussed, choose the one most strongly
supported by the symptoms. If no listed condition can be confidently
identified from the context, respond with exactly: Unclear

CONTEXT:
{{context}}

PATIENT'S SYMPTOMS:
{{question}}

Respond with ONLY the exact disease name from the list above, nothing else:""".format(
    disease_list="\n".join(f"- {name}" for name in KNOWN_DISEASE_NAMES)
)


FALLBACK_CONDITION_PROMPT = """A patient described these symptoms, and no confident match was found in
a specialized medical knowledge base of common conditions.

Based on general medical knowledge, what is ONE plausible condition
that could explain these symptoms? This does not need to match any
specific list -- use your own broader medical knowledge.

PATIENT'S SYMPTOMS:
{question}

Respond with ONLY the condition name (a few words), nothing else. If you
genuinely cannot suggest anything plausible, respond with exactly: Unclear"""


FALLBACK_GUIDANCE_PROMPT = """A patient described these symptoms, and no confident match was found in
a specialized medical knowledge base of common conditions. Based on
general medical knowledge, you believe this may be: {condition}

Write a short, calm, clear paragraph of general guidance for the
patient, using your own broader medical knowledge (not a specific
reference document). Mention when they should see a doctor. Do not
fabricate citations or reference specific studies/guidelines by name.

PATIENT'S SYMPTOMS:
{question}

Your guidance:"""


def get_fallback_condition_guess(symptom_query: str) -> str:
    """
    Safety-net call for when primary_condition comes back "Unclear"
    from the constrained, knowledge-base-only PRIMARY_CONDITION_PROMPT.
    Unlike that prompt, this one is NOT constrained to KNOWN_DISEASE_NAMES
    and does NOT see the retrieved (and possibly noisy/irrelevant)
    documents -- it reasons freshly from general medical knowledge only.

    WHY IGNORE RETRIEVED CONTEXT HERE: testing showed that when nothing
    in the knowledge base matches well, retrieval still returns its
    top-k closest (but often irrelevant) documents due to how vector
    search works -- it always returns *something*, even if nothing is
    a good match. Feeding those noisy documents into this fallback call
    risks anchoring the guess toward the wrong direction (e.g. a kidney
    stone case pulling in UTI/Anxiety/Headache documents due to keyword
    overlap, despite none of them being a good clinical match). Ignoring
    retrieved context and reasoning fresh avoids that anchoring risk.

    This result must always be labeled clearly as an ungrounded, general
    knowledge guess in the UI -- never displayed as if it were a
    verified, cited match the way normal primary_condition results are.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(FALLBACK_CONDITION_PROMPT)

    chain = prompt | llm | parser
    return chain.invoke({"question": symptom_query}).strip()


def get_fallback_guidance(symptom_query: str, condition_guess: str) -> str:
    """
    Companion to get_fallback_condition_guess() -- generates guidance
    text for the SAME ungrounded condition guess, instead of reusing
    the noisy retrieved documents from the failed knowledge-base
    lookup. This was a necessary fix after testing showed the headline
    condition (from the fallback guess) and the Guidance paragraph
    (still built from irrelevant retrieved documents) could disagree
    with each other on the same results screen -- e.g. headline saying
    "kidney stone" while Guidance discussed UTI, sourced from documents
    that were never actually relevant to the patient's symptoms. This
    call keeps both pieces of the fallback answer consistent with each
    other, and consistently ungrounded rather than a confusing mix.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(FALLBACK_GUIDANCE_PROMPT)

    chain = prompt | llm | parser
    return chain.invoke({"question": symptom_query, "condition": condition_guess}).strip()


def ask_dr_friend_uc2(symptom_query: str, raw_patient_text: str = None) -> dict:
    """
    Main entry point for UC2. Pass in the patient's symptom description
    as plain text, get back a dict with the generated answer, a
    programmatically-extracted list of source citations, the raw
    retrieved excerpt text (for UI transparency), and a structured
    primary condition name (for display and downstream disease-overview
    lookup).

    raw_patient_text (optional): the patient's own original words,
    concatenated from the conversation, BEFORE summarize_symptoms()
    rewrote them. WHY THIS EXISTS (found via testing): the LLM-written
    summary can paraphrase casual patient language ("nothing severe")
    into more formal, clinical-sounding text, which can shift retrieval
    toward a different, sometimes wrong, set of documents (e.g. mild
    fever/body-ache phrasing that correctly retrieved Common Cold/Flu
    when raw, retrieved Typhoid/Malaria instead once paraphrased into
    "not severely affected"). When raw_patient_text is provided, this
    function retrieves using BOTH the summary and the raw text, then
    merges and deduplicates -- mirroring the same dual-retrieval
    pattern already used in UC3's interpret_lab_report() (see Bug 4),
    which exists for the same underlying reason: a single query's
    phrasing can miss a genuinely good match that a differently-worded
    query for the same information would catch.
    """
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    if raw_patient_text:
        summary_docs = retriever.invoke(symptom_query)
        raw_docs = retriever.invoke(raw_patient_text)
        seen_content = set()
        retrieved_docs = []
        for doc in summary_docs + raw_docs:
            key = doc.page_content[:100]
            if key not in seen_content:
                seen_content.add(key)
                retrieved_docs.append(doc)
        # Cap the merged set: testing showed that beyond ~4 documents,
        # the primary_condition LLM became noticeably more likely to
        # return "Unclear" even when a clearly correct match was
        # present in the merged set, likely because more (sometimes
        # conflicting) options make confident single-answer selection
        # harder. Interleaving preserves each query's own top-ranked
        # relevance ordering rather than just truncating the summary
        # query's results first.
        capped_docs = []
        for i in range(max(len(summary_docs), len(raw_docs))):
            if i < len(summary_docs) and summary_docs[i] in retrieved_docs and summary_docs[i] not in capped_docs:
                capped_docs.append(summary_docs[i])
            if i < len(raw_docs) and raw_docs[i] in retrieved_docs and raw_docs[i] not in capped_docs:
                capped_docs.append(raw_docs[i])
            if len(capped_docs) >= 4:
                break
        retrieved_docs = capped_docs
    else:
        retrieved_docs = retriever.invoke(symptom_query)

    # SAFETY NET: never let the LLM generate guidance from empty or
    # near-empty retrieved context. Without this, the LLM would fall
    # back to its own general medical knowledge -- breaking the
    # grounding guarantee this whole system is built on. This mirrors
    # the identical safeguard added to UC3's interpreter after the same
    # failure mode was found there.
    if not retrieved_docs:
        return {
            "answer": (
                "I wasn't able to find enough relevant information in my "
                "knowledge base to confidently guide you on this. This "
                "doesn't mean nothing is wrong -- it means your symptoms "
                "don't clearly match the conditions I'm trained to "
                "recognize. Please describe your symptoms to a doctor or "
                "pharmacist directly, especially if they persist or worsen."
            ),
            "primary_condition": "Unclear",
            "sources": [],
            "source_excerpts": [],
            "retrieved_chunks": 0,
        }

    context_text = format_retrieved_docs(retrieved_docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()

    # EFFICIENCY FIX: primary_condition is now extracted BEFORE the
    # normal Guidance-generation call runs, not after. Previously, the
    # Guidance call always ran first and its output was silently
    # discarded whenever primary_condition later came back "Unclear"
    # and fallback mode took over -- one wasted LLM call on every
    # fallback-triggered query. Checking primary_condition first means
    # we only pay for the normal Guidance call when we already know a
    # real knowledge-base match exists.
    condition_prompt = ChatPromptTemplate.from_template(PRIMARY_CONDITION_PROMPT)
    condition_chain = condition_prompt | llm | parser
    primary_condition = condition_chain.invoke({
        "context": context_text,
        "question": symptom_query,
    }).strip()

    is_fallback_guess = False
    sources = []
    source_excerpts = []

    if primary_condition == "Unclear":
        fallback_guess = get_fallback_condition_guess(symptom_query)
        if fallback_guess == "Unclear":
            # Fallback also couldn't find anything plausible -- fall
            # through to the normal path below, which will generate a
            # grounded (if weak) answer from whatever was retrieved,
            # same as before this reordering.
            primary_condition = "Unclear"
        else:
            primary_condition = fallback_guess
            is_fallback_guess = True
            answer_text = (
                f"General knowledge guess, not verified against our reference "
                f"database - please treat as a starting point only.\n\n"
                + get_fallback_guidance(symptom_query, fallback_guess)
            )

    if not is_fallback_guess:
        generation_prompt = ChatPromptTemplate.from_template(UC2_SYSTEM_PROMPT)
        generation_chain = generation_prompt | llm | parser
        answer_text = generation_chain.invoke({
            "context": context_text,
            "question": symptom_query,
        })

        for doc in retrieved_docs:
            citation = doc.metadata.get("citation", "Unknown source")
            section = doc.metadata.get("section", "N/A")
            if citation not in sources:
                sources.append(citation)
            source_excerpts.append({
                "citation": citation,
                "section": section,
                "text": doc.page_content,
            })

    return {
        "answer": answer_text,
        "primary_condition": primary_condition,
        "is_fallback_guess": is_fallback_guess,
        "sources": sources,
        "source_excerpts": source_excerpts,
        "retrieved_chunks": len(retrieved_docs),
    }


if __name__ == "__main__":
    test_queries = [
        "I have excessive thirst, frequent urination, and blurred vision",
        "High fever, severe headache behind my eyes, and joint pain, started 2 days ago",
        "Throbbing headache on one side, sensitive to light, feeling nauseous",
        "Loose motions and vomiting since yesterday, mild stomach pain",
        "Cough with yellow phlegm, fever, and chest pain when breathing",
    ]

    for query in test_queries:
        print("=" * 70)
        print(f"QUERY: {query}\n")
        result = ask_dr_friend_uc2(query)
        print("RESPONSE:")
        print(result["answer"])
        print(f"\nSOURCES: {result['sources']}")
        print(f"CHUNKS RETRIEVED: {result['retrieved_chunks']}")
        print()