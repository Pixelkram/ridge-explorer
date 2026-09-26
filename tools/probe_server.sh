#!/usr/bin/env bash
# Start (or restart) the backend for probe testing. Kept in a file so the caller's command line never
# contains the uvicorn pattern (a `pkill -f` from an inline command matched and killed itself).
# Usage: tools/probe_server.sh [n_gpus]   -> logs/probe_test_server.log, logs/server.pid
set -u
cd "$(dirname "$0")/.."
N=${1:-2}
if [ -f logs/server.pid ] && kill -0 "$(cat logs/server.pid)" 2>/dev/null; then
  kill "$(cat logs/server.pid)"; sleep 4; kill -9 "$(cat logs/server.pid)" 2>/dev/null || true
fi
for p in $(pgrep -f "uvicorn backend.main:app"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done; sleep 2
mkdir -p logs
RIDGE_N_GPUS=$N nohup python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 > logs/probe_test_server.log 2>&1 &
echo $! > logs/server.pid
until [ "$(grep -c 'Worker ready' logs/probe_test_server.log 2>/dev/null)" -ge "$N" ]; do
  if grep -q "Traceback" logs/probe_test_server.log 2>/dev/null; then echo "startup error:"; grep -A3 Traceback logs/probe_test_server.log | head; exit 1; fi
  sleep 5
done
echo "server ready ($N workers) pid $(cat logs/server.pid) $(date +%H:%M:%S)"
