#!/bin/bash
# ── CoughSense Local Startup Script ──────────────────────────────
set -e

echo ""
echo "  ██████╗ ██████╗ ██╗   ██╗ ██████╗ ██╗  ██╗"
echo " ██╔════╝██╔═══██╗██║   ██║██╔════╝ ██║  ██║"
echo " ██║     ██║   ██║██║   ██║██║  ███╗███████║"
echo " ██║     ██║   ██║██║   ██║██║   ██║██╔══██║"
echo " ╚██████╗╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║"
echo "  ╚═════╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝"
echo "  CoughSense — Respiratory Risk Screening"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "❌  Python 3 not found. Install from https://python.org"
  exit 1
fi

echo "📦  Installing dependencies..."
cd backend
pip install -r requirements.txt -q

echo ""
echo "🧠  Starting backend (model will train on first run ~30s)..."
echo "🌐  Frontend: open frontend/index.html in your browser"
echo "📡  API docs: http://localhost:8000/docs"
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
