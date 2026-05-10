#!/usr/bin/env bash
# Konjo Quality Framework — Hook Installer
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
GRN='\033[0;32m'; YEL='\033[0;33m'; RED='\033[0;31m'; RST='\033[0m'; BOLD='\033[1m'
ok()   { echo -e "${GRN}  ✓${RST} $1"; }
warn() { echo -e "${YEL}  ⚠${RST} $1"; }
err()  { echo -e "${RED}  ✗${RST} $1"; }
echo -e "${BOLD}Konjo Quality Framework — Install${RST}"
HOOK_SRC="$REPO_ROOT/.konjo/hooks/pre-commit"
HOOK_DST="$REPO_ROOT/.git/hooks/pre-commit"
[[ ! -f "$HOOK_SRC" ]] && err ".konjo/hooks/pre-commit not found" && exit 1
chmod +x "$HOOK_SRC"
[[ -L "$HOOK_DST" ]] && rm "$HOOK_DST"
ln -sf "../../.konjo/hooks/pre-commit" "$HOOK_DST"
ok "Installed .git/hooks/pre-commit"
check_tool() { command -v "$1" &>/dev/null && ok "$1" || warn "$1 not found — $2"; }
check_tool "python3" "install Python 3.10+"
check_tool "git" "install git"
[[ -f "$REPO_ROOT/Cargo.toml" ]] && check_tool "cargo" "install Rust" || true
[[ -f "$REPO_ROOT/pixi.toml" ]] && check_tool "pixi" "install pixi (for Mojo)" || true
[[ -f "$REPO_ROOT/pyproject.toml" ]] && check_tool "ruff" "pip install ruff" || true
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && ok "ANTHROPIC_API_KEY" || warn "ANTHROPIC_API_KEY not set"
echo "Framework installed. Run: git commit --allow-empty -m 'test: verify konjo hooks'"
