#!/usr/bin/env bash
# macOS/Linux/WSL: check prerequisites, install light dependencies, supervise both servers.
set -euo pipefail
SATQUERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SATQUERY_ROOT"
SATQUERY_PYTHON="${SATQUERY_PYTHON:-}"
if [[ -z "$SATQUERY_PYTHON" ]]; then
  for candidate in "$SATQUERY_ROOT/.venv/bin/python" python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info[:2] not in [(3,11),(3,12)])' 2>/dev/null; then
      SATQUERY_PYTHON="$candidate"; break
    fi
  done
fi
[[ -n "$SATQUERY_PYTHON" ]] || { echo 'Install Python 3.11/3.12, or set SATQUERY_PYTHON to its executable.' >&2; exit 1; }
"$SATQUERY_PYTHON" -c 'import sys; assert sys.version_info[:2] in [(3,11),(3,12)], "Python 3.11 or 3.12 is required"; print("Python:",sys.version.split()[0])'
command -v node >/dev/null && command -v npm >/dev/null || { echo 'Install Node.js and npm.' >&2; exit 1; }
node -e 'let [a,b]=process.versions.node.split(".").map(Number);if(!((a===20&&b>=19)||(a===22&&b>=12)||a>22)){console.error("Node 20.19+ or 22.12+ required");process.exit(1)};console.log("Node:",process.versions.node)'
if [[ "${1:-}" == '--check' ]]; then echo 'Prerequisite checks passed.'; exit 0; fi
[[ -f .venv/bin/python ]] || "$SATQUERY_PYTHON" -m venv .venv
SATQUERY_PYTHON="$SATQUERY_ROOT/.venv/bin/python"
"$SATQUERY_PYTHON" -c 'import sys; assert sys.version_info[:2] in [(3,11),(3,12)], "Recreate .venv with Python 3.11/3.12"'
"$SATQUERY_PYTHON" -m pip install -r backend/requirements.txt
if "$SATQUERY_PYTHON" -c 'from backend.app.api.config import Settings; import sys; sys.exit(not Settings().database_url.startswith("postgresql"))'; then
  "$SATQUERY_PYTHON" -m pip install -r backend/requirements-postgres.txt
fi
if [[ ! -d frontend/node_modules ]]; then npm --prefix frontend ci; fi
"$SATQUERY_PYTHON" - <<'PY'
import socket
for port in (8000,5173):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: sock.bind(('127.0.0.1',port))
        except OSError: raise SystemExit(f'Port {port} is occupied; stop that server before launching this script.')
PY
SATQUERY_BACKEND_PID=''; SATQUERY_FRONTEND_PID=''
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$SATQUERY_FRONTEND_PID" ]] || kill "$SATQUERY_FRONTEND_PID" 2>/dev/null || true
  [[ -z "$SATQUERY_BACKEND_PID" ]] || kill "$SATQUERY_BACKEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
"$SATQUERY_PYTHON" -m uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000 &
SATQUERY_BACKEND_PID=$!
"$SATQUERY_PYTHON" - <<'PY'
import time,urllib.request
for _ in range(100):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=1)
        break
    except OSError: time.sleep(.2)
else: raise SystemExit('Backend did not become healthy; inspect its error above.')
PY
# Run Vite directly so the PID being supervised is the actual frontend process.
(cd frontend && exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort) &
SATQUERY_FRONTEND_PID=$!
echo 'Backend: http://127.0.0.1:8000  API docs: http://127.0.0.1:8000/docs'
echo 'Frontend: http://127.0.0.1:5173  Stop both with Ctrl+C.'
while kill -0 "$SATQUERY_BACKEND_PID" 2>/dev/null && kill -0 "$SATQUERY_FRONTEND_PID" 2>/dev/null; do sleep 1; done
echo 'A server exited; stopping the other server.' >&2
exit 1
