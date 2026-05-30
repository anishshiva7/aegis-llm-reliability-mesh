"""
Test script for Module 1.

Run from the backend/ directory with the venv python:

    ./venv/bin/python -m pytest tests/test_retrieval.py -v

or simply as a script (it falls back to running the checks directly):

    ./venv/bin/python tests/test_retrieval.py

The chunker tests are fast (no model). The end-to-end test loads the embedding
model, so it's slower on the first run while the model downloads/loads.
"""

import sys
from pathlib import Path

# Make `app` importable when run as a plain script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chunker import chunk_text  # noqa: E402


# --------------------------------------------------------------------------- #
# Chunking — fast, deterministic, no model required.
# --------------------------------------------------------------------------- #
def test_chunk_basic_no_overlap():
    text = " ".join(str(i) for i in range(10))  # "0 1 2 ... 9"
    chunks = chunk_text(text, chunk_size=5, chunk_overlap=0)
    assert chunks == ["0 1 2 3 4", "5 6 7 8 9"]


def test_chunk_with_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)
    # step = 5 - 2 = 3 -> windows start at 0, 3, 6, 9
    assert chunks[0] == "0 1 2 3 4"
    assert chunks[1] == "3 4 5 6 7"  # last 2 words of chunk 0 reappear here
    assert chunks[2] == "6 7 8 9"
    assert chunks[-1].endswith("9")  # final word always present


def test_chunk_shorter_than_size():
    chunks = chunk_text("just a few words", chunk_size=100, chunk_overlap=10)
    assert chunks == ["just a few words"]


def test_chunk_empty():
    assert chunk_text("   ", chunk_size=10, chunk_overlap=2) == []


def test_chunk_invalid_overlap():
    try:
        chunk_text("a b c", chunk_size=3, chunk_overlap=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError when overlap >= chunk_size")


# --------------------------------------------------------------------------- #
# End-to-end — ingest then search. Loads the embedding model (slower).
# --------------------------------------------------------------------------- #
def test_end_to_end_search_ranks_relevant_chunk_first():
    from app.services.retrieval import RetrievalEngine

    engine = RetrievalEngine()
    engine.ingest(
        text=(
            "The mitochondria is the powerhouse of the cell. "
            "Python is a popular programming language for data science. "
            "The Eiffel Tower is located in Paris, France."
        ),
        source="facts",
        chunk_size=8,
        chunk_overlap=2,
    )

    results = engine.search("Where is the Eiffel Tower?", top_k=3)
    assert results, "expected at least one result"

    chunk_id, score, record = results[0]
    # The top hit should be the Paris/Eiffel sentence — semantic match.
    assert "Eiffel" in record.text or "Paris" in record.text
    # Cosine similarity is bounded by 1.0; a real match should be clearly > 0.
    assert 0.0 < score <= 1.0001


if __name__ == "__main__":
    # Allow running without pytest installed.
    failures = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
