#!/bin/bash
# Longspear — Setup and convenience script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ── Functions ─────────────────────────────────────────────

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  setup       Copy .env.example to .env, create data dirs"
    echo "  up          Start all Docker services"
    echo "  down        Stop all Docker services"
    echo "  ingest      Run the full ingest pipeline"
    echo "  ingest-test Run ingest with --test-mode (5 videos per channel)"
    echo "  logs        Follow all container logs"
    echo "  health      Check service health"
    echo "  stats       Show document counts"
    echo "  shell       Open a shell in the app container"
    echo "  test        Run tests inside the app container"
    echo ""
}

cmd_setup() {
    echo "═══ Setting up Longspear ═══"

    if [ ! -f .env ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    else
        echo ".env already exists"
    fi

    mkdir -p data/transcripts/heather_cox_richardson
    mkdir -p data/transcripts/nate_b_jones
    mkdir -p data/processed/heather_cox_richardson
    mkdir -p data/processed/nate_b_jones
    mkdir -p data/vectordb/lancedb

    echo "Data directories created."
    echo ""
    echo "Next: run '$0 up' to start services."
}

cmd_up() {
    echo "═══ Starting Longspear services ═══"
    docker compose up -d
    echo ""
    echo "Waiting for services to be healthy..."
    sleep 5
    docker compose ps
    echo ""
    echo "Note: Ollama will pull models on first run (may take a few minutes)."
    echo "Check: docker compose logs -f ollama-init"
}

cmd_down() {
    docker compose down
}

cmd_ingest() {
    docker compose exec app python -m src.ingest.pipeline "$@"
}

cmd_ingest_test() {
    docker compose exec app python -m src.ingest.pipeline --test-mode
}

cmd_logs() {
    docker compose logs -f "$@"
}

cmd_health() {
    curl -s http://localhost:8000/health | python3 -m json.tool
}

cmd_stats() {
    curl -s http://localhost:8000/stats | python3 -m json.tool
}

cmd_shell() {
    docker compose exec app bash
}

cmd_test() {
    docker compose exec app pytest tests/ -v
}

# ── Main ──────────────────────────────────────────────────

case "${1:-}" in
    setup)       cmd_setup ;;
    up)          cmd_up ;;
    down)        cmd_down ;;
    ingest)      shift; cmd_ingest "$@" ;;
    ingest-test) cmd_ingest_test ;;
    logs)        shift; cmd_logs "$@" ;;
    health)      cmd_health ;;
    stats)       cmd_stats ;;
    shell)       cmd_shell ;;
    test)        cmd_test ;;
    *)           usage ;;
esac
