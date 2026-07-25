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
- Match criteria STRICTLY and LITERALLY. If a criterion specifies a
  duration, frequency, or severity qualifier (e.g. "lasting more than
  2 days", "persistent", "severe", "unable to keep fluids down"), do
  NOT count it as matched unless the patient's symptoms explicitly
  include that same qualifier. Simply mentioning a symptom that appears
  in a criterion (e.g. "vomiting") does NOT match a criterion that
  requires a qualified version of it (e.g. "persistent vomiting").
  When in doubt about whether a qualifier is met, do NOT match the
  criterion -- prefer "Self-Care at Home" over an unsupported escalation.
- Return ONLY a JSON object, no other text, in exactly this format:
{{
    "urgency_level": "<one of the four levels above>",
    "matched_criteria": ["<criterion text that was matched, if any>"],
    "patient_evidence": ["<the EXACT phrase from the patient's symptoms that satisfies each matched criterion, quoted verbatim -- if you cannot find an exact phrase that explicitly satisfies a qualifier like 'persistent' or 'severe', do not include that criterion as matched>"],
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
    # Code-level safeguard: reject any matched "persistent"/"severe"/
    # qualifier-based criterion if the patient_evidence doesn't actually
    # contain qualifying language. This does not trust the LLM's own
    # judgment alone -- it verifies the LLM's cited evidence contains
    # real support for persistence/severity claims, since prompt-only
    # instructions were tested and found insufficient to reliably
    # prevent the LLM from treating a bare symptom mention as satisfying
    # a qualified criterion.
    # WHY WE CHECK FOR LAB-VALUE EVIDENCE SEPARATELY (Bug 14):
    # UC3 passes lab findings as evidence (e.g. "C-Reactive Protein
    # (CRP): 9.8 mg/L (High)"), which reads nothing like a symptom
    # sentence but is still strong, objective evidence on its own.
    def is_lab_value_evidence(evidence_text: str) -> bool:
        lab_flag_markers = ["(high)", "(low)", "mg/dl", "mg/l", "uiu/ml",
                             "/cumm", "mil/cumm", "g/dl", "%", "u/l"]
        evidence_lower = evidence_text.lower()
        return any(marker in evidence_lower for marker in lab_flag_markers)

    # BUG 19 FIX -- replaces the old QUALIFIER_WORDS keyword-matching
    # approach entirely. The old approach checked whether the LLM's
    # quoted evidence contained one of a fixed list of "seriousness
    # words" (e.g. "severe", "worsening"). This broke on criteria with
    # OR-conditions -- e.g. "chest pain that is severe, crushing, OR
    # radiating to the arm" -- where evidence genuinely satisfying the
    # criterion via "crushing"/"radiating" got rejected simply because
    # the literal word "severe" wasn't separately present. This
    # incorrectly downgraded a textbook cardiac emergency (crushing
    # chest pain radiating to the arm) all the way to "Self-Care at
    # Home", despite the LLM's original classification being correct.
    #
    # NEW APPROACH: verify that the LLM's quoted patient_evidence is a
    # genuine excerpt of what the patient actually said (a real
    # substring match, allowing for minor whitespace/case differences),
    # rather than checking for specific severity words. This still
    # catches Bug 6's original problem -- an LLM claiming a match with
    # fabricated or unsupported "evidence" -- without rejecting valid
    # matches just because they used different words than our fixed
    # list expected.
    symptoms_lower = symptoms.lower()
    evidence_list = result.get("patient_evidence", [])
    verified_criteria = []
    for i, criterion in enumerate(result.get("matched_criteria", [])):
        evidence_text = evidence_list[i] if i < len(evidence_list) else ""
        evidence_lower = evidence_text.lower().strip()

        if is_lab_value_evidence(evidence_text):
            # Lab findings are objective on their own -- no need to
            # verify against symptom text, which wouldn't contain them.
            verified_criteria.append(criterion)
            continue

        if evidence_lower and evidence_lower in symptoms_lower:
            # The LLM's quoted evidence genuinely appears in what the
            # patient said -- trust the match, whatever words it used.
            verified_criteria.append(criterion)
        # else: silently drop this criterion -- the LLM's quoted
        # evidence doesn't actually appear in the patient's own words,
        # suggesting it may have been fabricated or paraphrased beyond
        # what was actually said.

    if len(verified_criteria) < len(result.get("matched_criteria", [])):
        # Some criteria were rejected -- re-derive urgency level from
        # what's actually left, rather than trusting the LLM's original
        # level which was based on the unverified criteria set.
        result["matched_criteria"] = verified_criteria
        if not verified_criteria:
            result["urgency_level"] = "Self-Care at Home"
            result["reasoning"] = "No criteria could be confirmed against explicit patient-reported evidence; downgraded from initial assessment."

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