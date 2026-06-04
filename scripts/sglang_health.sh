#!/bin/bash
# SG-Lang Health Check
# Verifies SG-Lang container is running and responsive before training or evals.
#
# Usage:
#   ./scripts/sglang_health.sh                    # Check default port 1235
#   ./scripts/sglang_health.sh --port 1235        # Check specific port
#   ./scripts/sglang_health.sh --url http://...   # Check custom URL

set -euo pipefail

PORT="${SGLANG_PORT:-1235}"
URL="${SGLANG_URL:-http://localhost:$PORT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      URL="http://localhost:$PORT"
      shift 2
      ;;
    --url)
      URL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Checking SG-Lang health at $URL ..."

for i in 1 2 3; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL/v1/models" --max-time 10 2>/dev/null || echo "000")
  if [ "$HTTP_CODE" = "200" ]; then
    echo "SG-Lang is healthy (HTTP $HTTP_CODE)"
    exit 0
  fi
  echo "  Attempt $i: HTTP $HTTP_CODE"
  sleep 2
done

echo "ERROR: SG-Lang is not reachable at $URL"
echo "Start the SG-Lang container first:"
echo "  docker compose up -d sglang"
exit 1
