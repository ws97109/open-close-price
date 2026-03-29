#!/bin/bash
# Start the Taiwan Stock Prediction Web System
# Backend: FastAPI on port 8000
# Frontend: Vite dev server on port 5173

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load nvm / node
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

echo "=== Taiwan Stock Prediction Web ==="
echo

# ── Install frontend deps if needed ──────────────────────────
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "[frontend] Installing dependencies..."
  cd "$ROOT/frontend" && npm install
fi

# ── Build frontend (production) OR start dev server ──────────
if [ "$1" = "--prod" ]; then
  echo "[frontend] Building for production..."
  cd "$ROOT/frontend" && npm run build
  echo
  echo "[backend] Starting FastAPI (serves built frontend at http://localhost:8000)"
  cd "$ROOT/backend"
  python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --loop asyncio
else
  # Development mode: run both servers
  echo "[backend]  Starting FastAPI on http://localhost:8000"
  cd "$ROOT/backend"
  python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --loop asyncio --reload &
  BACKEND_PID=$!
  echo "  Backend PID: $BACKEND_PID"
  sleep 2

  echo
  echo "[frontend] Starting Vite dev server on http://localhost:5173"
  cd "$ROOT/frontend"
  npm run dev &
  FRONTEND_PID=$!
  echo "  Frontend PID: $FRONTEND_PID"

  echo
  echo "==================================="
  echo "  Open: http://localhost:5173"
  echo "  API:  http://localhost:8000/api"
  echo "  Ctrl+C to stop both servers"
  echo "==================================="

  trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
  wait
fi
