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
    echo "  monitor     Show transcript counts, DB stats, and Ollama status"
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

    # Verify native Ollama is running
    if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "⚠️  Native Ollama is not running. Start it first:"
        echo "   ollama serve"
        echo "   ollama pull nomic-embed-text"
        echo "   ollama pull mxbai-embed-large"
        exit 1
    fi
    echo "✅ Native Ollama detected (Metal GPU)"

    docker compose up -d
    echo ""
    echo "Waiting for services to be healthy..."
    sleep 5
    docker compose ps
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

cmd_monitor() {
    echo "═══ Longspear Status ═══"
    echo ""

    # Transcript counts
    echo "── Transcripts on disk ──"
    HCR_COUNT=$(ls data/transcripts/heather_cox_richardson/*.vtt 2>/dev/null | wc -l | tr -d ' ')
    NBJ_COUNT=$(ls data/transcripts/nate_b_jones/*.vtt 2>/dev/null | wc -l | tr -d ' ')
    echo "  Heather Cox Richardson: ${HCR_COUNT} VTT files"
    echo "  Nate B Jones:          ${NBJ_COUNT} VTT files"
    echo ""

    # DB counts
    echo "── pgvector (PostgreSQL) ──"
    docker compose exec -T postgres psql -U longspear -d longspear -t -c \
        "SELECT '  nomic:  ' || COUNT(*) || ' docs (persona: ' || COALESCE(STRING_AGG(DISTINCT persona, ', '), 'none') || ')' FROM documents_nomic UNION ALL SELECT '  mxbai:  ' || COUNT(*) || ' docs (persona: ' || COALESCE(STRING_AGG(DISTINCT persona, ', '), 'none') || ')' FROM documents_mxbai;" 2>/dev/null || echo "  (postgres not running)"
    echo ""

    echo "── LanceDB (embedded) ──"
    docker compose exec -T app python -c "
import lancedb
db = lancedb.connect('/app/data/vectordb/lancedb')
for t in db.list_tables():
    tbl = db.open_table(t)
    print(f'  {t}: {tbl.count_rows()} rows')
if not db.list_tables():
    print('  (no tables)')
" 2>/dev/null || echo "  (app not running)"
    echo ""

    # Ollama
    echo "── Ollama (native) ──"
    curl -s http://localhost:11434/api/ps 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    total = 0
    for m in data['models']:
        gb = m['size'] / (1024**3)
        total += gb
        print(f'  {m[\"name\"]:<30} {gb:.1f} GB')
    print(f'  Total: {total:.1f} GB')
except: print('  (ollama not responding)')
" || echo "  (ollama not running)"
    echo ""

    # Disk usage
    echo "── Disk usage ──"
    echo "  Transcripts: $(du -sh data/transcripts/ 2>/dev/null | cut -f1 || echo 'N/A')"
    echo "  LanceDB:     $(du -sh data/vectordb/ 2>/dev/null | cut -f1 || echo 'N/A')"
    echo ""

    # Docker status
    echo "── Docker containers ──"
    docker compose ps 2>/dev/null || echo "  (not running)"
}

cmd_logs() {
    docker compose logs -f "$@"
}

cmd_health() {
    curl -s http://localhost:28000/health | python3 -m json.tool
}

cmd_stats() {
    curl -s http://localhost:28000/stats | python3 -m json.tool
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
    monitor)     cmd_monitor ;;
    logs)        shift; cmd_logs "$@" ;;
    health)      cmd_health ;;
    stats)       cmd_stats ;;
    shell)       cmd_shell ;;
    test)        cmd_test ;;
    *)           usage ;;
esac
