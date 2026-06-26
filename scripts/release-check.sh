#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run cargo fmt --all -- --check
run cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
run cargo test --workspace --all-features --locked
run cargo test --workspace --doc --all-features --locked
printf '\n==> cargo build --workspace --release --all-features --locked (warnings denied)\n'
RUSTFLAGS="${RUSTFLAGS:-} -D warnings" \
  cargo build --workspace --release --all-features --locked

if [[ "${SHUNT_RELEASE_LIVE:-0}" == "1" ]]; then
  : "${SHUNT_LLM:?SHUNT_LLM is required when SHUNT_RELEASE_LIVE=1}"
  run cargo test -p shunt-infer --test integration --locked -- --ignored --nocapture
  run cargo test -p shunt-infer --test harness --locked -- run_harness --ignored --nocapture
else
  printf '\nLive model checks skipped. Set SHUNT_RELEASE_LIVE=1 and SHUNT_LLM to enable them.\n'
fi

printf '\nRelease checks passed.\n'
