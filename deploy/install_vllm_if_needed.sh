#!/usr/bin/env bash
set -euo pipefail

if command -v vllm >/dev/null 2>&1; then
  echo "vLLM already installed: $(which vllm)"
else
  echo "Installing vLLM..."
  pip install -U "vllm>=0.6.0"
fi

python - << 'PY'
try:
    import vllm
    print("vLLM import ok")
except Exception as e:
    print("vLLM import failed:", e)
    raise
PY