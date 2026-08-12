"""
Build an evaluation dataset from resolved tickets.

Each ticket becomes a test case:
  - input   : the ticket description (simulates a customer question)
  - expected: the resolution text (the ground-truth answer)

The dataset can be pushed to LangSmith or used locally with
`langsmith.evaluation.evaluate()`.
"""

import sqlite3
from telecom_rag.config import TICKETS_DB_PATH


def build_eval_examples(db_path: str = TICKETS_DB_PATH) -> list[dict]:
    """Return a list of {input, expected_output, metadata} dicts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tickets WHERE status = 'resolved'"
    ).fetchall()
    conn.close()

    examples = []
    for row in rows:
        examples.append(
            {
                "input": row["description"],
                "expected_output": row["resolution"],
                "metadata": {
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "issue_type": row["issue_type"],
                },
            }
        )
    return examples


def push_to_langsmith(dataset_name: str = "telecom-care-eval") -> None:
    """Create (or update) the dataset in LangSmith."""
    from langsmith import Client

    client = Client()
    examples = build_eval_examples()

    # Create dataset if it doesn't exist
    try:
        dataset = client.create_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)

    for ex in examples:
        client.create_example(
            inputs={"question": ex["input"]},
            outputs={"answer": ex["expected_output"]},
            metadata=ex["metadata"],
            dataset_id=dataset.id,
        )
    print(f"Pushed {len(examples)} examples to LangSmith dataset '{dataset_name}'.")


if __name__ == "__main__":
    examples = build_eval_examples()
    print(f"Built {len(examples)} evaluation examples from resolved tickets.\n")
    for ex in examples[:3]:
        print(f"  [{ex['metadata']['ticket_id']}]")
        print(f"  Q: {ex['input'][:80]}…")
        print(f"  A: {ex['expected_output'][:80]}…\n")
