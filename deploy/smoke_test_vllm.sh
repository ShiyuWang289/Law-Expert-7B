#!/usr/bin/env bash
set -euo pipefail

echo "[1] models endpoint"
curl -s http://127.0.0.1:6006/v1/models | python -m json.tool

echo "[2] chat completion"
curl -s http://127.0.0.1:6006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"law-expert",
    "messages":[
      {"role":"system","content":"你是专业中国法律顾问"},
      {"role":"user","content":"劳动合同到期不续签是否有补偿？"}
    ],
    "temperature":0,
    "max_tokens":128
  }' | python -m json.tool