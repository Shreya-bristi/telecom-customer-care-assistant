"""
Assembles the end-to-end RAG chain:

    input_guard → retriever → format → prompt → LLM → output_guard → str

LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true.
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

from telecom_rag.config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_RETRIES
from telecom_rag.chain.prompts import CHAT_PROMPT
from telecom_rag.retrieval.multi_source import build_retriever
from telecom_rag.guardrails.input_check import validate_input
from telecom_rag.guardrails.output_check import validate_output


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """Render retrieved documents into a labelled context string."""
    sections: list[str] = []
    for doc in docs:
        tag = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{tag}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


# ── Chain builder ────────────────────────────────────────────────────────────

def build_chain():
    """Return a fully-wired LCEL chain with guardrails."""
    retriever = build_retriever()

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=LLM_MAX_RETRIES,
    )

    chain = (
        RunnableLambda(validate_input)              # ← input guardrail
        | {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | CHAT_PROMPT
        | llm
        | StrOutputParser()
        | RunnableLambda(validate_output)            # ← output guardrail
    )
    return chain
