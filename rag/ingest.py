"""
ingest.py

Reads all disease documents from rag/knowledge_base/, splits them into
chunks by section, embeds each chunk using OpenAI's embedding model, and
stores them in a persistent ChromaDB vector store.

WHY WE SPLIT BY SECTION, NOT FIXED CHARACTER COUNT:
Our documents use consistent ## headers (Overview, Common Symptoms,
Self-Care Guidance, etc.). Splitting along these headers means each chunk
is a complete, coherent unit of information -- e.g. a query about "when
should I worry" retrieves the whole "When to See a Doctor" section intact,
rather than a random 500-character slice that might cut a sentence in half.

Run this script every time you add or update a document in knowledge_base/.
"""
from dotenv import load_dotenv
load_dotenv()


import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Path setup -- everything relative to this file's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(CURRENT_DIR, "knowledge_base")
CHROMA_DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

# Headers to split on -- matches our document template exactly
HEADERS_TO_SPLIT_ON = [
    ("#", "disease_name"),
    ("##", "section"),
]


def load_documents():
    """
    Loads every .txt file from knowledge_base/ as a raw document.
    Each document keeps its filename as metadata, useful for debugging.
    """
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {KNOWLEDGE_BASE_DIR}")
    return documents



def extract_citation_line(document_text: str) -> str:
    """
    Pulls the '## Source:' line out of a document's raw text.
    """
    match = re.search(r"^## Source:\s*(.+)$", document_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "Unknown source"


def split_documents(documents):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    all_chunks = []

    for doc in documents:
        citation = extract_citation_line(doc.page_content)
        chunks = splitter.split_text(doc.page_content)
        source_file = doc.metadata.get("source", "unknown")
        for chunk in chunks:
            chunk.metadata["source_file"] = os.path.basename(source_file)
            chunk.metadata["citation"] = citation
        all_chunks.extend(chunks)

    print(f"Split into {len(all_chunks)} chunks")
    return all_chunks


def build_vector_store(chunks):
    """
    Embeds every chunk using OpenAI's embedding model and stores the
    result in a persistent ChromaDB collection on disk.

    WHY text-embedding-3-small:
    Same provider as our GPT-4o-mini calls elsewhere in the project --
    no new API key or account needed. Small model is cheap and fast,
    and is more than accurate enough for a knowledge base of this size.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"Vector store built and saved to {CHROMA_DB_DIR}")
    return vector_store


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    build_vector_store(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()