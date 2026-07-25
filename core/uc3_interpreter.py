"""
uc3_interpreter.py

Stage 4 of UC3: combines flagged abnormal lab values (from
uc3_extractor.py) with patient-reported symptoms into a single RAG
query, retrieves grounded guidance, and produces a structured findings
narrative plus a deterministic urgency level -- reusing the same
rag/retriever.py and rag/urgency_uc2.py infrastructure built for UC2,
per the outline's stated design: RAG and urgency assessment are shared
services across all three use cases, not rebuilt per UC.
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from retriever import load_vector_store, PRIMARY_CONDITION_PROMPT
from urgency_uc2 import check_uc2_urgency

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

RETRIEVAL_K = 3
SCORE_THRESHOLD = 0.30


def build_lab_query(structured_values: list, symptoms: str) -> str:
    """
    Combines flagged abnormal lab values and patient symptoms into one
    query string for retrieval. Only abnormal (High/Low) values are
    included -- Normal values don't need to inform disease retrieval,
    and including them would dilute the query with irrelevant signal.

    WHY THIS FORMAT: writing values as "Test Name: value unit (Flag)"
    keeps the query close to how a doctor would describe abnormal
    findings verbally, which matches the phrasing style of the
    knowledge base documents' own symptom/finding descriptions --
    closer phrasing style improves embedding similarity matching.
    """
    abnormal = [v for v in structured_values if v["computed_flag"] in ("High", "Low")]

    findings_text = "; ".join(
        f"{v['test_name']}: {v['raw_result']} ({v['computed_flag']})"
        for v in abnormal
    )

    query = f"Symptoms: {symptoms}. Abnormal lab findings: {findings_text}."
    return query


INTERPRETATION_PROMPT = """You are Dr. Friend, an AI healthcare guidance assistant for Indian patients.

You are interpreting a patient's lab report results together with their
reported symptoms, using ONLY the retrieved medical reference information
below. Do not use outside knowledge beyond what is provided in the context.

RULES:
- Identify which condition(s) the combination of abnormal lab findings
  and symptoms most likely points to, using ONLY information present in
  the retrieved context below.
- CRITICAL: Do not introduce medical explanations, mechanisms, or
  terminology that are not present in the retrieved context, even if
  you know them to be generally true. For example, do not explain platelet
  physiology, bone marrow function, or electrolyte balance mechanisms
  unless that specific explanation appears in the context text itself.
  If the context does not explain WHY a value matters clinically, simply
  state that it is flagged and note the condition(s) it relates to,
  without adding your own explanation of the underlying mechanism.
- If a retrieved document does not end up being relevant to your final
  interpretation, do not mention it at all -- only discuss documents
  whose content is actually used in your reasoning.
- Explicitly connect specific abnormal values to specific symptoms only
  where the retrieved context directly supports that connection.
- If findings and symptoms point to different possible conditions, say so
  clearly rather than forcing a single diagnosis.
- Keep the tone calm, clear, and non-alarming, but do not understate
  genuinely concerning findings.

CONTEXT (retrieved reference documents):
{context}

PATIENT'S SYMPTOMS:
{symptoms}

ABNORMAL LAB FINDINGS:
{findings}

Your interpretation:"""


def interpret_lab_report(structured_values: list, symptoms: str) -> dict:
    """
    Full Stage 4 pipeline: builds a combined query from abnormal lab
    values + symptoms, retrieves grounded context, generates an
    interpretation connecting values to symptoms, and independently
    classifies urgency from the same retrieved context -- mirroring
    UC2's run_uc2_pipeline() structure for consistency across use cases.
    """
    abnormal = [v for v in structured_values if v["computed_flag"] in ("High", "Low")]
    findings_summary = "; ".join(
        f"{v['test_name']}: {v['raw_result']} ({v['computed_flag']}, normal range {v['raw_range']})"
        for v in abnormal
    )

    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # WHY TWO SEPARATE RETRIEVALS, NOT ONE BLENDED QUERY:
    # A single combined query can let weak/ambiguous symptom text dilute
    # strong lab-value signal (or vice versa) in the embedding space.
    # Retrieving each independently and merging means a distinctive lab
    # pattern (e.g. low platelets + high haematocrit) can surface its
    # matching document even when symptom phrasing alone wouldn't have
    # retrieved it strongly enough to rank in a single blended top-k.
    findings_only_query = f"Abnormal lab findings: {findings_summary}"
    symptoms_only_query = f"Symptoms: {symptoms}"

    findings_docs = retriever.invoke(findings_only_query)
    symptom_docs = retriever.invoke(symptoms_only_query)

    seen_content = set()
    retrieved_docs = []
    for doc in findings_docs + symptom_docs:
        key = doc.page_content[:100]
        if key not in seen_content:
            seen_content.add(key)
            retrieved_docs.append(doc)

    # CRITICAL SAFETY CHECK: never let the LLM generate an interpretation   
    # from empty context. An LLM asked to interpret findings with no
    # retrieved reference material will silently fall back to its own
    # general knowledge instead of refusing -- which breaks the entire
    # grounding guarantee this system is built on. Fail loudly and
    # honestly instead.
    if not retrieved_docs:
        return {
            "abnormal_findings": abnormal,
            "findings_summary": findings_summary,
            "interpretation": (
                "No sufficiently relevant reference material was found in "
                "the knowledge base for this specific combination of findings "
                "and symptoms. Please consult a healthcare professional to "
                "interpret these results directly."
            ),
            "primary_condition": "Unclear",
            "sources": [],
            "source_excerpts": [],
            "chunks_retrieved": 0,
            "urgency_level": "See a Doctor Today",
            "urgency_matched_criteria": [],
            "urgency_reasoning": "Unable to confidently classify urgency without retrieved reference material; defaulting to a cautious recommendation.",
        }

    context_text = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('citation', 'unknown')} | "
        f"Section: {doc.metadata.get('section', 'N/A')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_template(INTERPRETATION_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    interpretation = chain.invoke({
        "context": context_text,
        "symptoms": symptoms,
        "findings": findings_summary,
    })

    # Structured primary_condition field, mirroring UC2's approach in
    # ask_dr_friend_uc2(). Combines symptoms and findings into one
    # "question" input since lab values are often the stronger signal
    # in UC3 (see Bug 4 -- Dengue was correctly identified from lab
    # values alone with no fever mentioned in symptoms).
    condition_prompt = ChatPromptTemplate.from_template(PRIMARY_CONDITION_PROMPT)
    condition_chain = condition_prompt | llm | parser
    primary_condition = condition_chain.invoke({
        "context": context_text,
        "question": f"{symptoms}. Abnormal findings: {findings_summary}",
    }).strip()

    urgency_result = check_uc2_urgency(
        symptoms=f"{symptoms}. Abnormal findings: {findings_summary}",
        retrieved_context=context_text,
    )

    # SAFETY FLOOR (Bug 16): check_uc2_urgency() can only escalate
    # urgency based on criteria it found in the retrieved documents. If
    # the knowledge base has no document covering a particular abnormal
    # finding (e.g. no dedicated kidney/renal disease document to match
    # against elevated creatinine, BUN, low eGFR, or high potassium),
    # those findings are silently never considered at all -- not
    # rejected, just never evaluated. This was found to let a lab
    # report with several genuinely concerning abnormal values (kidney
    # function decline plus dangerous hyperkalemia) reach "Self-Care at
    # Home" simply because none of it matched any retrieved criteria.
    # Deterministic rule, consistent with this project's rule-based-
    # where-possible philosophy: any abnormal (High/Low) lab finding is
    # reason enough to rule out "Self-Care at Home" as an outcome for a
    # UC3 report -- lab abnormalities always warrant at least a routine
    # doctor review, even when no specific matched criterion escalated
    # it further.
    if abnormal and urgency_result["urgency_level"] == "Self-Care at Home":
        urgency_result["urgency_level"] = "See a Doctor Soon"
        urgency_result["reasoning"] = (
            "Abnormal lab findings were detected, but our reference "
            "documents did not contain matching criteria to assess their "
            "urgency further. As a precaution, we recommend a routine "
            "doctor review rather than self-care alone."
        )

    sources = []
    source_excerpts = []
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
        "abnormal_findings": abnormal,
        "findings_summary": findings_summary,
        "interpretation": interpretation,
        "primary_condition": primary_condition,
        "sources": sources,
        "source_excerpts": source_excerpts,
        "chunks_retrieved": len(retrieved_docs),
        "urgency_level": urgency_result["urgency_level"],
        "urgency_matched_criteria": urgency_result["matched_criteria"],
        "urgency_reasoning": urgency_result["reasoning"],
    }


if __name__ == "__main__":
    from uc3_extractor import extract_structured_values
    import os as _os

    apex_report = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "sample_report", "Apex_Path_Labs_Report.pdf")

    structured_values = extract_structured_values(apex_report)
    symptoms = "I've been feeling constantly tired, gaining weight without trying, and I feel cold all the time even when others are comfortable"

    print(f"SYMPTOMS: {symptoms}\n")
    print("ABNORMAL VALUES BEING SENT FOR INTERPRETATION:")
    for v in structured_values:
        if v["computed_flag"] in ("High", "Low"):
            print(f"  - {v['test_name']}: {v['raw_result']} ({v['computed_flag']})")

    print("\n" + "=" * 70)
    result = interpret_lab_report(structured_values, symptoms)

    print("INTERPRETATION:")
    print(result["interpretation"])
    print(f"\nSOURCES: {result['sources']}")
    print(f"CHUNKS RETRIEVED: {result['chunks_retrieved']}")
    print(f"\nURGENCY LEVEL: {result['urgency_level']}")
    print(f"MATCHED CRITERIA: {result['urgency_matched_criteria']}")
    print(f"\nPRIMARY CONDITION: {result['primary_condition']}")