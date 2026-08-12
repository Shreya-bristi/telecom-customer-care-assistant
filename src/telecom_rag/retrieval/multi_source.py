"""
Builds a merged retriever that fans out across three Chroma collections
(faq, tickets, guides) and returns a single combined result list.

The retriever is wrapped as a LangChain RunnableLambda so it plugs
directly into an LCEL(LangChain expression language) chain.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from telecom_rag.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    COLLECTION_FAQ,
    COLLECTION_TICKETS,
    COLLECTION_GUIDES,
    K_FAQ,
    K_TICKETS,
    K_GUIDES,
)


def _open_collection(name: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def build_retriever(
    k_faq: int = K_FAQ,
    k_tickets: int = K_TICKETS,
    k_guides: int = K_GUIDES,
) -> RunnableLambda:
    """Return a RunnableLambda that retrieves from all three collections."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    faq_ret = _open_collection(COLLECTION_FAQ, embeddings).as_retriever(
        search_kwargs={"k": k_faq}
    )
    ticket_ret = _open_collection(COLLECTION_TICKETS, embeddings).as_retriever(
        search_kwargs={"k": k_tickets}
    )
    guide_ret = _open_collection(COLLECTION_GUIDES, embeddings).as_retriever(
        search_kwargs={"k": k_guides}
    )

    def _retrieve(query: str) -> list[Document]:
        return faq_ret.invoke(query) + ticket_ret.invoke(query) + guide_ret.invoke(query)

    return RunnableLambda(_retrieve)
