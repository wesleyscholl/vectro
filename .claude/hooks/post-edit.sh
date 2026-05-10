#!/bin/bash
# Runs cargo check after any .rs file edit — catches compile errors immediately
FILE=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("file_path",""))' 2>/dev/null)
[[ "$FILE" == *.rs ]] || exit 0
echo "→ cargo check (triggered by edit to $FILE)"
cd /Users/wesleyscholl/lopi && cargo check --quiet 2>&1 | head -30
