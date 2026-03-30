#!/bin/bash
# Taiwan Stock Prediction Web System — Start Script
# Backend  : FastAPI  port 8000
# Frontend : Vite     port 5173
# Tunnel   : ngrok    public URL (optional: pass --tunnel)

ROOT="$(cd "$(dirname "$0")" && pwd)"
NGROK="/mnt/nfs/maokao_2/.local/bin/ngrok"

# ── Load nvm / node ───────────────────────────────────────────
export NVM_DIR="${HOME}/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

# ── Kill any existing processes on these ports ────────────────
kill_port() {
  local pids
  pids=$(lsof -ti:"$1" 2>/dev/null)
  [ -n "$pids" ] && kill $pids 2>/dev/null && sleep 1
}
kill_port 8000
kill_port 5173

echo "=== Taiwan Stock Prediction Web ==="
echo

# ── Install frontend deps if needed ──────────────────────────
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "[frontend] Installing dependencies..."
  cd "$ROOT/frontend" && npm install
fi

# ── Start backend ─────────────────────────────────────────────
echo "[backend]  Starting FastAPI on http://localhost:8000"
cd "$ROOT/backend"
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --loop asyncio \
  >> /tmp/stock_backend.log 2>&1 &
BACKEND_PID=$!

# Wait until backend is ready (max 10s)
for i in $(seq 1 10); do
  sleep 1
  curl -sf http://localhost:8000/api/health > /dev/null 2>&1 && break
  if [ $i -eq 10 ]; then
    echo "[ERROR] Backend failed to start. Check /tmp/stock_backend.log"
    exit 1
  fi
done
echo "  Backend ready (PID $BACKEND_PID)"

# ── Start frontend ────────────────────────────────────────────
echo "[frontend] Starting Vite dev server on http://localhost:5173"
cd "$ROOT/frontend"
npm run dev -- --host 0.0.0.0 >> /tmp/stock_frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait until frontend is ready (max 10s)
for i in $(seq 1 10); do
  sleep 1
  curl -sf http://localhost:5173 > /dev/null 2>&1 && break
  if [ $i -eq 10 ]; then
    echo "[ERROR] Frontend failed to start. Check /tmp/stock_frontend.log"
    kill $BACKEND_PID 2>/dev/null
    exit 1
  fi
done
echo "  Frontend ready (PID $FRONTEND_PID)"

# ── Start ngrok tunnel ────────────────────────────────────────
NGROK_PID=""
if [[ "$*" == *"--tunnel"* ]] || [[ "$1" == "--tunnel" ]]; then
  if [ -x "$NGROK" ]; then
    echo "[ngrok]    Starting tunnel on port 5173..."
    "$NGROK" http 5173 --log=stdout >> /tmp/stock_ngrok.log 2>&1 &
    NGROK_PID=$!
    sleep 4
    # Extract public URL from ngrok API
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
      | python3 -c "import sys,json; t=json.load(sys.stdin).get('tunnels',[]); print(next((x['public_url'] for x in t if 'https' in x.get('public_url','')),''
))" 2>/dev/null)
    if [ -n "$NGROK_URL" ]; then
      echo "  ✓ Public URL: $NGROK_URL"
    else
      echo "  ngrok tunnel started (check http://localhost:4040 for URL)"
    fi
  else
    echo "[ngrok]    Not found at $NGROK — skipping tunnel"
  fi
fi

echo
echo "==================================================="
echo "  Local:  http://localhost:5173"
echo "  API:    http://localhost:8000/api"
[ -n "$NGROK_URL" ] && echo "  Public: $NGROK_URL"
echo
echo "  Logs:   /tmp/stock_backend.log"
echo "          /tmp/stock_frontend.log"
[ -n "$NGROK_PID" ] && echo "          /tmp/stock_ngrok.log"
echo
echo "  Press Ctrl+C to stop all services"
echo "==================================================="

# ── Cleanup on exit ───────────────────────────────────────────
cleanup() {
  echo
  echo "Stopping all services..."
  [ -n "$BACKEND_PID"  ] && kill $BACKEND_PID  2>/dev/null
  [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
  [ -n "$NGROK_PID"    ] && kill $NGROK_PID    2>/dev/null
  kill_port 8000; kill_port 5173
  exit 0
}
trap cleanup INT TERM

# ── Monitor: restart crashed services ─────────────────────────
while true; do
  sleep 5

  # Check backend
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] Backend crashed — restarting..."
    cd "$ROOT/backend"
    python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --loop asyncio \
      >> /tmp/stock_backend.log 2>&1 &
    BACKEND_PID=$!
    sleep 3
    echo "[$(date '+%H:%M:%S')] Backend restarted (PID $BACKEND_PID)"
  fi

  # Check frontend
  if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] Frontend crashed — restarting..."
    cd "$ROOT/frontend"
    npm run dev -- --host 0.0.0.0 >> /tmp/stock_frontend.log 2>&1 &
    FRONTEND_PID=$!
    sleep 3
    echo "[$(date '+%H:%M:%S')] Frontend restarted (PID $FRONTEND_PID)"
  fi
done
