#!/bin/sh
set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
APP_MODE="${APP_MODE:-api}"

echo "=========================================================="
echo " Starting TenantEase AI Engine in mode: [$APP_MODE]"
echo " Listening on host: $HOST, port: $PORT"
echo "=========================================================="

if [ "$APP_MODE" = "dashboard" ] || [ "$APP_MODE" = "streamlit" ]; then
    exec streamlit run app.py --server.port "$PORT" --server.address "$HOST" --server.headless true
else
    exec uvicorn api:app --host "$HOST" --port "$PORT"
fi
