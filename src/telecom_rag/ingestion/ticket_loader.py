"""
Load resolved tickets from ticketsdb and embed them
into the 'tickets' Chroma collection.

Each ticket becomes one Document (no chunking — tickets are concise).
"""

import sqlite3
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from telecom_rag.config import (
    TICKETS_DB_PATH,
    CHROMA_DIR,
    COLLECTION_TICKETS,
    EMBEDDING_MODEL_NAME,
)


def load(db_path: str = TICKETS_DB_PATH) -> list[Document]:
    """Query all resolved tickets and return them as Documents."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tickets WHERE status = 'resolved'"
    ).fetchall()
    conn.close()

    documents: list[Document] = []
    for row in rows:
        text = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "ticket",
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "status": row["status"],
                },
            )
        )
    return documents


def store(documents: list[Document] | None = None) -> int:
    """Embed ticket documents and persist them to Chroma."""
    if documents is None:
        documents = load()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_TICKETS,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore._collection.count()


if __name__ == "__main__":
    docs = load()
    print(f"Loaded {len(docs)} resolved tickets.")
    n = store(docs)
    print(f"Stored {n} vectors in '{COLLECTION_TICKETS}'.")
