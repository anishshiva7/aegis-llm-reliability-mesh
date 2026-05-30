#!/usr/bin/env bash
#
# Manual smoke test for the Aegis retrieval API using curl.
#
# Prereqs — in one terminal, start the server:
#   cd backend
#   ./venv/bin/uvicorn app.main:app --reload
#
# Then in another terminal:
#   bash scripts/curl_examples.sh
#
set -euo pipefail

BASE="${AEGIS_BASE_URL:-http://127.0.0.1:8000}"

echo "== health =="
curl -s "$BASE/health"; echo

echo
echo "== ingest raw text =="
curl -s -X POST "$BASE/ingest" \
  -H "Content-Type: application/json" \
  -d '{
        "text": "The mitochondria is the powerhouse of the cell. Python is a popular programming language for data science. The Eiffel Tower is located in Paris, France.",
        "source": "facts.txt",
        "chunk_size": 8,
        "chunk_overlap": 2
      }'; echo

echo
echo "== ingest a file (creates a temp file then uploads it) =="
TMP="$(mktemp -t aegis_doc.XXXXXX)"
printf 'RAG combines retrieval with generation. First retrieve relevant context, then feed it to an LLM.\n' > "$TMP"
curl -s -X POST "$BASE/ingest/file" \
  -F "file=@${TMP};type=text/plain" \
  -F "chunk_size=12" \
  -F "chunk_overlap=3"; echo
rm -f "$TMP"

echo
echo "== search =="
curl -s -X POST "$BASE/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is the Eiffel Tower?", "top_k": 3}'; echo

echo
echo "== stats =="
curl -s "$BASE/stats"; echo
