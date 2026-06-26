#!/usr/bin/env bash
# llama-server preset for Qwen3.6-27B (non-thinking / coding mode)
#
# Sampling: Unsloth non-thinking instruct recommendations
#   temp=0.7  top-p=0.8  top-k=20  presence_penalty=1.5
# These are baked into shunt's ModelProfile for qwen3* models.
# The server flags here control context, caching, and speed.
#
# --jinja         required for native tool-call template processing
# -sm layer       split model layers across both GPUs (multi-GPU)
# NOTE: do NOT use --reasoning off here. shunt controls thinking per-request:
#   action selection → enable_thinking:false (2048-token routing call, no reasoning needed)
#   content generation → thinking ON at temp=0.6 (model reasons about code to write)
# NOTE: --spec-type draft-mtp requires MTP layers in the GGUF; Qwen3.6 Q6_K
#       does not include them so MTP is omitted here.
#
# Usage:
#   bash presets/qwen3-6-27b.sh                          # default: HF download Q6_K
#   MODEL=/path/to/model.gguf bash presets/qwen3-6-27b.sh  # local file override

set -euo pipefail

PORT="${PORT:-8080}"
CTX="${CTX:-16384}"

# Use local model file if MODEL is set, otherwise download from HuggingFace
if [ -n "${MODEL:-}" ]; then
    model_args=(--model "$MODEL")
else
    model_args=(-hf "unsloth/Qwen3.6-27B-GGUF:Q6_K")
fi

exec ~/llama.cpp/build/bin/llama-server \
    "${model_args[@]}" \
    --alias "qwen3.6-27b" \
    --port "$PORT" \
    --ctx-size "$CTX" \
    --threads -1 \
    --flash-attn on \
    --jinja \
    -sm layer \
    --min-p 0.00
