#!/usr/bin/env bash
#
# Seed the running Aegis backend with the sample documents in ../data so a fresh
# clone can demo grounded retrieval immediately.
#
# Prerequisite: the backend must already be running (see scripts/run.sh).
#
# Usage:
#   bash scripts/seed.sh                       # seed against http://127.0.0.1:8000
#   AEGIS_URL=http://127.0.0.1:8077 bash scripts/seed.sh
#
set -euo pipefail

AEGIS_URL="${AEGIS_URL:-http://127.0.0.1:8000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

echo "Seeding Aegis at $AEGIS_URL with sample documents from $DATA_DIR"

# Fail fast with a friendly message if the server isn't up.
if ! curl -sf "$AEGIS_URL/health" >/dev/null; then
  echo "ERROR: backend not reachable at $AEGIS_URL/health." >&2
  echo "Start it first:  cd backend && bash scripts/run.sh" >&2
  exit 1
fi

for doc in "$DATA_DIR"/*.md; do
  [ -e "$doc" ] || { echo "No .md files found in $DATA_DIR"; exit 1; }
  name="$(basename "$doc")"
  echo "  -> ingesting $name"
  curl -sf -X POST "$AEGIS_URL/ingest/file" -F "file=@$doc" | \
    (command -v jq >/dev/null && jq -c . || cat)
  echo
done

echo
echo "Done. Index stats:"
curl -sf "$AEGIS_URL/stats" | (command -v jq >/dev/null && jq . || cat)

cat <<'EOF'

Try a grounded query now:
  curl -s http://127.0.0.1:8000/ask \
    -H 'Content-Type: application/json' \
    -d '{"query":"What is Acme Cloud'\''s refund policy?"}' | jq '.route, .answer'

Or open the dashboard at http://localhost:3000 and ask:
  "How does Aegis route a query?"          (grounded RAG answer)
  "What is Acme Cloud's refund policy?"    (grounded RAG answer)
  "What is 2 + 2?"                          (direct answer, no retrieval)
EOF
