"""
Streamlit front-end for the Telecom Customer Care RAG Assistant.

Usage:
    streamlit run app.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from telecom_rag.chain.rag_pipeline import build_chain
from telecom_rag.guardrails.input_check import GuardrailError

# ── Config ───────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I set up a personal hotspot?",
]

st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)


@st.cache_resource
def get_chain():
    return build_chain()


# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("RAG-powered · LangChain · LangSmith")
    st.divider()

    st.markdown("**Try a sample question**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []

# ── Main chat ────────────────────────────────────────────────────────────────
st.title("Customer Care Assistant")
st.caption(
    "Ask me anything about your mobile service — "
    "connectivity, billing, SIM, roaming, and more."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Resolve question source
question = st.chat_input("Describe your issue…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            chain = get_chain()
            response = st.write_stream(chain.stream(question))
        except GuardrailError as e:
            response = f"⚠️ {e}"
            st.warning(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
