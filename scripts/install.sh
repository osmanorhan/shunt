#!/usr/bin/env sh
# Install shunt — the AI coding assistant
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/osmanorhan/shunt/main/scripts/install.sh | sh
#
# Environment variables:
#   SHUNT_VERSION      — version tag to install, e.g. v0.3.0 (default: latest)
#   SHUNT_INSTALL_DIR  — directory to install the binary (default: ~/.local/bin)
#   SHUNT_REPO         — GitHub repo slug (default: YOUR_ORG/shunt)
set -eu

SHUNT_REPO="${SHUNT_REPO:-osmanorhan/shunt}"
SHUNT_VERSION="${SHUNT_VERSION:-latest}"
SHUNT_INSTALL_DIR="${SHUNT_INSTALL_DIR:-$HOME/.local/bin}"

# ── Detect OS ────────────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
  Linux)  os_part="linux" ;;
  Darwin) os_part="apple-darwin" ;;
  *)
    echo "error: unsupported operating system: $OS" >&2
    exit 1
    ;;
esac

# ── Detect architecture ───────────────────────────────────────────────────────
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)        arch_part="x86_64" ;;
  arm64|aarch64) arch_part="aarch64" ;;
  *)
    echo "error: unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

# ── Build target triple ───────────────────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
  TARGET="${arch_part}-unknown-${os_part}-musl"
else
  TARGET="${arch_part}-${os_part}"
fi

# ── Resolve version ───────────────────────────────────────────────────────────
if [ "$SHUNT_VERSION" = "latest" ]; then
  printf 'Fetching latest release...\n'
  API_URL="https://api.github.com/repos/${SHUNT_REPO}/releases/latest"
  SHUNT_VERSION="$(curl -fsSL "$API_URL" | grep '"tag_name"' | sed 's/.*"tag_name": *"\(v[^"]*\)".*/\1/')"
  if [ -z "$SHUNT_VERSION" ]; then
    echo "error: could not determine latest version from GitHub API" >&2
    exit 1
  fi
fi

ARCHIVE="shunt-${SHUNT_VERSION}-${TARGET}.tar.gz"
URL="https://github.com/${SHUNT_REPO}/releases/download/${SHUNT_VERSION}/${ARCHIVE}"

# ── Download ──────────────────────────────────────────────────────────────────
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT INT TERM

printf 'Downloading %s...\n' "$ARCHIVE"
if ! curl -fsSL "$URL" -o "$TMPDIR/$ARCHIVE"; then
  echo "error: download failed: $URL" >&2
  exit 1
fi

# ── Verify checksum if sha256sums.txt is available ───────────────────────────
SUMS_URL="https://github.com/${SHUNT_REPO}/releases/download/${SHUNT_VERSION}/sha256sums.txt"
if curl -fsSL "$SUMS_URL" -o "$TMPDIR/sha256sums.txt" 2>/dev/null; then
  cd "$TMPDIR"
  if command -v sha256sum >/dev/null 2>&1; then
    grep "$ARCHIVE" sha256sums.txt | sha256sum --check --status
    printf 'Checksum verified.\n'
  elif command -v shasum >/dev/null 2>&1; then
    grep "$ARCHIVE" sha256sums.txt | shasum -a 256 --check --status
    printf 'Checksum verified.\n'
  fi
  cd - >/dev/null
fi

# ── Install ───────────────────────────────────────────────────────────────────
tar xzf "$TMPDIR/$ARCHIVE" -C "$TMPDIR"

mkdir -p "$SHUNT_INSTALL_DIR"
mv "$TMPDIR/shunt" "$SHUNT_INSTALL_DIR/shunt"
chmod +x "$SHUNT_INSTALL_DIR/shunt"

printf '\nshunt %s installed to %s/shunt\n' "$SHUNT_VERSION" "$SHUNT_INSTALL_DIR"

# ── PATH hint ─────────────────────────────────────────────────────────────────
case ":${PATH}:" in
  *:"${SHUNT_INSTALL_DIR}":*) ;;
  *)
    printf '\nNote: %s is not in your PATH.\n' "$SHUNT_INSTALL_DIR"
    printf 'Add this to your shell profile:\n'
    printf '  export PATH="%s:$PATH"\n' "$SHUNT_INSTALL_DIR"
    ;;
esac
