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
    formatted = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('citation', 'unknown')} | "
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


def ask_dr_friend_uc2(symptom_query: str) -> dict:
    """
    Main entry point for UC2. Pass in the patient's symptom description
    as plain text, get back a dict with the generated answer AND a
    programmatically-extracted list of source citations.

    WHY WE RETURN SOURCES SEPARATELY, NOT JUST FROM THE LLM'S TEXT:
    Asking the LLM to remember to list its sources in free text is
    unreliable -- it sometimes includes the citation, sometimes doesn't.
    Extracting sources directly from the retriever's actual results
    guarantees the citation is always accurate and always present,
    regardless of what the LLM chooses to write.
    """
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVAL_K, "score_threshold": SCORE_THRESHOLD},
    )

    retrieved_docs = retriever.invoke(symptom_query)
    context_text = format_retrieved_docs(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(UC2_SYSTEM_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()

    generation_chain = prompt | llm | parser
    answer_text = generation_chain.invoke({
        "context": context_text,
        "question": symptom_query,
    })

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
        "answer": answer_text,
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