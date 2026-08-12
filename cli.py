"""
CLI entry point for the Telecom RAG chatbot.

Usage:
    python cli.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
load_dotenv()

from telecom_rag.chain.rag_pipeline import build_chain
from telecom_rag.guardrails.input_check import GuardrailError


def main() -> None:
    print("╔══════════════════════════════════════════╗")
    print("║   Telecom Customer Care Assistant (RAG)  ║")
    print("╚══════════════════════════════════════════╝")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    chain = build_chain()

    while True:
        question = input("Customer: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            print("\nAssistant: ", end="", flush=True)
            for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n")
        except GuardrailError as e:
            print(f"\n⚠️  {e}\n")


if __name__ == "__main__":
    main()
