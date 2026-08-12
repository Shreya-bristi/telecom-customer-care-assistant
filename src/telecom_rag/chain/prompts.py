"""
Prompt templates for the Telecom RAG chain.

Keeping prompts in a dedicated module makes them easy to version,
A/B test, and reference during LangSmith evaluation.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_TEMPLATE = """\
You are a professional telecom customer-care assistant.
Your job is to help customers resolve technical issues with their mobile service.

RULES:
1. Answer ONLY from the context provided below.
2. If the context is insufficient, say so clearly and suggest the customer
   call 611 or use the MyTelecom app.
3. Be concise and action-oriented — give numbered steps when appropriate.
4. Never fabricate technical details, pricing, or policy information.

Context (retrieved from FAQ, past tickets, and technical guides):
{context}
"""

# ── Assembled chat prompt ────────────────────────────────────────────────────
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]
)
