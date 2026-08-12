"""
LoadsFAQ entries from a CSV file and embed them into the 'faq' Chroma collection.

"""

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from telecom_rag.config import (
    FAQ_CSV_PATH,
    CHROMA_DIR,
    COLLECTION_FAQ,
    EMBEDDING_MODEL_NAME,
)


def load(csv_path: str = FAQ_CSV_PATH) -> list[Document]:
    """Readsthe FAQ CSV and return a list of Documents."""
    df = pd.read_csv(csv_path)
    documents: list[Document] = []
    for _, row in df.iterrows():
        text = f"Q: {row['question']}\nA: {row['answer']}"
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "faq",
                    "category": row["category"],
                    "faq_id": str(row["id"]),
                },
            )
        )
    return documents


def store(documents: list[Document] | None = None) -> int:
    """Embed documents and persist them to the Chroma FAQ collection.

    Returns the number of vectors stored.
    """
    if documents is None:
        documents = load()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_FAQ,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore._collection.count()


if __name__ == "__main__":
    docs = load()
    print(f"Loaded {len(docs)} FAQ entries.")
    n = store(docs)
    print(f"Stored {n} vectors in '{COLLECTION_FAQ}'.")
