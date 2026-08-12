"""
Ingest all data sources into ChromaDB in a single run.

Usage:
    python -m telecom_rag.ingestion.run_all
"""

import shutil
from telecom_rag.config import CHROMA_DIR
from telecom_rag.ingestion import faq_loader, pdf_loader, ticket_loader


def ingest(clean: bool = False) -> dict[str, int]:
    """Run every loader and return a summary of vectors stored per collection."""
    if clean:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        print("Cleared existing Chroma store.")

    results: dict[str, int] = {}

    print("\n[1/3] Ingesting FAQ entries …")
    faq_docs = faq_loader.load()
    results["faq"] = faq_loader.store(faq_docs)
    print(f"       → {results['faq']} vectors")

    print("[2/3] Ingesting PDF guide …")
    guide_docs = pdf_loader.load()
    results["guides"] = pdf_loader.store(guide_docs)
    print(f"       → {results['guides']} vectors")

    print("[3/3] Ingesting resolved tickets …")
    ticket_docs = ticket_loader.load()
    results["tickets"] = ticket_loader.store(ticket_docs)
    print(f"       → {results['tickets']} vectors")

    print(f"\nIngestion complete. Total: {sum(results.values())} vectors across {len(results)} collections.")
    return results


if __name__ == "__main__":
    ingest(clean=True)
