"""
Centralized config for RAG pipeline
"""

import os
from pathlib import Path

# =================== path ====================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]          
DATA_DIR     = PROJECT_ROOT / "data"
CHROMA_DIR   = str(PROJECT_ROOT / "chroma_store")

FAQ_CSV_PATH     = str(DATA_DIR / "faq.csv")
GUIDE_PDF_PATH   = str(DATA_DIR / "telecom_guide.pdf")
TICKETS_DB_PATH  = str(DATA_DIR / "tickets.db")

#=======================embedding model========================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ========================= ChromaDB collection names==================================
COLLECTION_FAQ     = "faq"
COLLECTION_TICKETS = "tickets"
COLLECTION_GUIDES  = "guides"

# ========================= Chunking settings ===================================
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100

# =============================
K_FAQ     = 3
K_TICKETS = 3
K_GUIDES  = 3

# ===================== LLM settings ========================================
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
LLM_TEMPERATURE = 0
LLM_MAX_RETRIES = 2

# ========================= Langsmith Observability =============================
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "telecom-care-rag")

#============================== Guardrail thresholds ==================================
MAX_INPUT_LENGTH      = 1000      # characters
HALLUCINATION_THRESHOLD = 0.5     # cosine-sim floor for grounding check
