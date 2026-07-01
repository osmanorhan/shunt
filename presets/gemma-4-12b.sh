#!/usr/bin/env bash
# llama-server preset for Gemma-4-12B-IT (thinking suppressed via grammar)
#
# Sampling: Google/Unsloth defaults
#   temp=1.0  top-p=0.95  top-k=64
# These are baked into shunt's ModelProfile for gemma-4* models.
#
# Gemma-4 notes:
#   - /no_think prefix does NOT work; shunt suppresses thinking via grammar
#     (response_format: json_schema forces JSON from token 1, blocking <think>)
#   - budget_tokens=2048 is sent per-request to cap the reasoning phase
#   - Multi-turn: thinking blocks are excluded from history (only final output kept)
#   - --jinja is still recommended for correctness with future tool-call support
#
# Usage:
#   MODEL=/path/to/gemma-4-12b-it-Q4_K_M.gguf bash presets/gemma-4-12b.sh

set -euo pipefail

PORT="${PORT:-8080}"
CTX="${CTX:-16384}"

# Force the process to only see the first GPU (safest single-card method)
export CUDA_VISIBLE_DEVICES=1

if [ -n "${MODEL:-}" ]; then
    model_args=(--model "$MODEL")
else
    model_args=(-hf "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL")
fi

exec ~/llama.cpp/build/bin/llama-server \
  "${model_args[@]}" \
  --alias "gemma-4-12b" \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --threads -1 \
  --flash-attn on \
  --jinja \
  --host 0.0.0.0 \
  --cont-batching \
  --no-context-shift
