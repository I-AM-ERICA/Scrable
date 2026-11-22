#!/usr/bin/env bash
# SCRABLE (CPU) + backend + frontend on this laptop, all inside WSL, for the customer-reply demo.
# No GPU needed. Everything listens on localhost only. Steps and first-time setup: RUN_LOCAL.md.
#
#   bash scripts/run_local_site.sh        (Ctrl+C stops all three)
set -euo pipefail

SCRABLE_DIR=${SCRABLE_DIR:-$HOME/naver/Scrable}
BACKEND_DIR=${BACKEND_DIR:-$HOME/naver/am-erica-backend}
FRONTEND_DIR=${FRONTEND_DIR:-$HOME/naver/am-erica-frontend}
LOG_DIR=${LOG_DIR:-/tmp/scrable-site}
mkdir -p "$LOG_DIR"

# A one-off internal key shared by SCRABLE and the backend for this run only (never printed).
AI_INTERNAL_KEY=$(openssl rand -hex 16)
export AI_INTERNAL_KEY
export SCRABLE_CPU_BF16=1  # bfloat16 on the CPU: about 3.5 GB of memory, ~2x faster than float32

trap 'echo; echo "Stopping..."; kill 0' INT TERM EXIT

wait_for() {  # wait_for <name> <pid> <log> <command that succeeds when ready>
  local name=$1 pid=$2 log=$3
  shift 3
  until "$@" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name stopped. Last lines of $log:"
      tail -n 20 "$log"
      exit 1
    fi
    sleep 3
  done
}

echo "1/3 SCRABLE: loading the model on the CPU (1-3 minutes; the first run also downloads the embedding model)"
cd "$SCRABLE_DIR"
.venv/bin/python -m scrable.serve --config configs/serve_cpu.json > "$LOG_DIR/scrable.log" 2>&1 &
wait_for SCRABLE $! "$LOG_DIR/scrable.log" grep -q listening "$LOG_DIR/scrable.log"
echo "    ready ($(grep -o 'listening on [^ ]*' "$LOG_DIR/scrable.log"))"

echo "2/3 Backend: starting on localhost:8000 (customer replies go to SCRABLE, other assistants stay mock)"
cd "$BACKEND_DIR"
AI_CUSTOMER_MOCK_ENABLED=false AI_CUSTOMER_HOST=127.0.0.1 AI_CUSTOMER_PORT=50053 \
  AI_CUSTOMER_INTERNAL_KEY="$AI_INTERNAL_KEY" AI_CUSTOMER_DEADLINE=300s \
  ./gradlew bootRun > "$LOG_DIR/backend.log" 2>&1 &
wait_for Backend $! "$LOG_DIR/backend.log" curl -sf localhost:8000/actuator/health
echo "    ready"

echo "3/3 Frontend: starting on localhost:7001"
cd "$FRONTEND_DIR"
# shellcheck disable=SC1091
[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
wait_for Frontend $! "$LOG_DIR/frontend.log" curl -sf localhost:7001
echo "    ready"

echo
echo "Open http://localhost:7001 in the Windows browser and log in with the demo account (RUN_LOCAL.md step 3)."
echo "Logs: $LOG_DIR/{scrable,backend,frontend}.log   Stop: Ctrl+C"
wait
