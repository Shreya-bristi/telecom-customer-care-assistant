"""
Load the Telecom Technical Reference Guide (PDF) and embed it
into the 'guides' Chroma collection.

The PDF is split with RecursiveCharacterTextSplitter before embedding.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from telecom_rag.config import (
    GUIDE_PDF_PATH,
    CHROMA_DIR,
    COLLECTION_GUIDES,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load(pdf_path: str = GUIDE_PDF_PATH) -> list[Document]:
    """Parsesthe PDF and split it into overlapping chunks."""
    pages = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = idx

    return chunks


def store(documents: list[Document] | None = None) -> int:
    """Embed chunks and persist them to the Chroma guides collection."""
    if documents is None:
        documents = load()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_GUIDES,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore._collection.count()


if __name__ == "__main__":
    docs = load()
    print(f"Loaded {len(docs)} guide chunks.")
    n = store(docs)
    print(f"Stored {n} vectors in '{COLLECTION_GUIDES}'.")
