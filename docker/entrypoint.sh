#!/bin/sh
# Start the app, training a fallback model first if no artifact exists.
#
# The image no longer trains at build time (issue #28): builds were slow
# and non-reproducible because every image build retrained the model.
# Production should mount an artifact trained in CI; `docker run` with
# nothing mounted still works because we train once at startup.
set -e

MODEL_PATH="${MODEL_PATH:-/app/model/artifacts/model.json}"
export MODEL_PATH

if [ ! -f "$MODEL_PATH" ]; then
    echo "No model artifact at $MODEL_PATH — training a fallback model..."
    python -m model.train
else
    echo "Using existing model artifact at $MODEL_PATH"
fi

exec streamlit run app.py \
    --server.port="${PORT:-7860}" \
    --server.address=0.0.0.0 \
    --server.headless=true
