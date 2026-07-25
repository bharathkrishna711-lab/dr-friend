"""
disease_lookup.py

Deterministic disease overview lookup, shared across UC1/UC2/UC3. Given
a disease name that is ALREADY KNOWN (from UC1's ML prediction, or a
confirmed primary diagnosis in UC2/UC3), retrieves that disease's
Overview section directly from the knowledge base via an exact metadata
match -- NOT similarity search.

WHY METADATA LOOKUP, NOT SIMILARITY SEARCH:
Similarity search (as used in rag/retriever.py) is for the genuinely
ambiguous case: given symptoms alone, which disease does this most
resemble? That is real uncertainty requiring embedding comparison.
Here, the disease is already known with certainty (UC1's classifier
already decided; UC2/UC3 already concluded a primary diagnosis) -- so
retrieving its Overview is a simple, exact lookup by metadata, with zero
risk of retrieving the wrong document. This is more reliable and more
defensible than running another similarity search when the answer is
already known.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

_vector_store = None


def _get_vector_store():
    """Lazily loads and caches the vector store connection."""
    global _vector_store
    if _vector_store is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
        )
    return _vector_store


def get_disease_overview(disease_name: str) -> dict | None:
    """
    Looks up the Overview section for a given disease name via exact
    metadata filtering on ChromaDB's collection -- no embedding
    similarity involved.

    Returns None if no matching document exists in the knowledge base
    (some of UC1's 17 trained diseases may not yet have a corresponding
    knowledge base document) -- callers should handle this by simply
    not showing the panel, never by guessing or falling back to a
    similarity search that could retrieve the wrong disease.
    """
    vector_store = _get_vector_store()

    try:
        results = vector_store.get(
            where={
                "$and": [
                    {"disease_name": {"$eq": f"Disease: {disease_name}"}},
                    {"section": {"$eq": "Overview"}},
                ]
            }
        )
    except Exception:
        return None

    if not results or not results.get("documents"):
        return None

    return {
        "disease_name": disease_name,
        "overview_text": results["documents"][0],
        "citation": results["metadatas"][0].get("citation", "Unknown source"),
    }


if __name__ == "__main__":
    test_diseases = ["Migraine", "Type 2 Diabetes", "Arrhythmia", "Heart Failure", "Some Fake Disease"]
    for disease in test_diseases:
        result = get_disease_overview(disease)
        print(f"\n{'=' * 60}")
        print(f"DISEASE: {disease}")
        if result:
            print(f"CITATION: {result['citation']}")
            print(f"OVERVIEW: {result['overview_text'][:200]}...")
        else:
            print("No overview found in knowledge base.")