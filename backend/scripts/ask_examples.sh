#!/usr/bin/env bash
#
# Manual smoke test for the Module 2 /ask endpoint (query router + RAG).
#
# Prereqs — start the server in another terminal:
#   cd backend && bash scripts/run.sh
#
# Then:
#   bash scripts/ask_examples.sh
#
set -euo pipefail

BASE="${AEGIS_BASE_URL:-http://127.0.0.1:8000}"

echo "== seed the index with a document =="
curl -s -X POST "$BASE/ingest" \
  -H "Content-Type: application/json" \
  -d '{
        "text": "The Eiffel Tower is located in Paris, France. It was completed in 1889. Python is a popular programming language for data science.",
        "source": "facts.txt",
        "chunk_size": 10,
        "chunk_overlap": 2
      }'; echo

echo
echo "== RAG_ANSWER: a question answerable from the document =="
curl -s -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is the Eiffel Tower located?", "top_k": 2}'; echo

echo
echo "== DIRECT_ANSWER: a general query unrelated to the document =="
curl -s -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are good strategies for negotiating a salary?"}'; echo

echo
echo "== NEEDS_CLARIFICATION: a vague query =="
curl -s -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "tell me more"}'; echo

echo
echo "== FORCED route: force RAG even on a generic query =="
curl -s -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "hello there", "force_route": "RAG_ANSWER"}'; echo

echo
echo "== trace disabled =="
curl -s -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is the Eiffel Tower located?", "include_trace": false}'; echo
