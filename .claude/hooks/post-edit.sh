#!/bin/bash
FILE=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("file_path",""))' 2>/dev/null)
[[ -z "$FILE" ]] && exit 0
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT" || exit 0
case "$FILE" in
  *.rs) echo "→ cargo check"; cargo check --quiet 2>&1 | head -30 ;;
  *.py) command -v ruff &>/dev/null && ruff check "$FILE" --quiet 2>&1 | head -20 ;;
  *.ts|*.tsx) npx tsc --noEmit 2>&1 | head -20 ;;
esac
