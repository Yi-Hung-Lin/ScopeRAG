#!/usr/bin/env bash
# One-command setup: fetch the adapter, build, start vLLM + ScopeRAG API + Open WebUI, and wait until ready.
set -euo pipefail
cd "$(dirname "$0")"
command -v docker >/dev/null || { echo "Docker is required"; exit 1; }
docker info 2>/dev/null | grep -qi nvidia || echo "warning: NVIDIA container runtime not detected (needed for vLLM)"
[ -f .env ] || cp .env.example .env
scripts/fetch_adapter.sh
docker compose --profile ui up -d --build
env_get() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d '"'"'"; }
API_PORT=$(env_get API_PORT); WEBUI_PORT=$(env_get WEBUI_PORT)
echo "waiting for the model to load (first run downloads ~16 GB)..."
for _ in $(seq 720); do curl -sf "http://localhost:${API_PORT:-8021}/health" >/dev/null && break; sleep 5; done
curl -sf "http://localhost:${API_PORT:-8021}/health" >/dev/null || { echo "not ready; see: docker compose logs"; exit 1; }
echo
echo "ScopeRAG is ready."
echo "  Chat UI:     http://localhost:${WEBUI_PORT:-3001}"
echo "  OpenAI API:  http://localhost:${API_PORT:-8021}/v1"
echo "  Quick test:  scripts/try_it.sh"
