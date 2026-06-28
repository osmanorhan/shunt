#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

before_fmt_diff="$(git diff --binary -- '*.rs')"
run cargo fmt --all
after_fmt_diff="$(git diff --binary -- '*.rs')"

if [[ "$before_fmt_diff" != "$after_fmt_diff" ]]; then
  printf '\nRustfmt rewrote Rust files. Review and stage the formatting changes, then commit again.\n'
  exit 1
fi

run cargo clippy --workspace --all-targets --locked -- -D warnings
