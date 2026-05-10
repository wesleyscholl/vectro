---
paths: ["**/api*", "**/server*", "**/webhook*", "**/auth*", "**/routes*", "**/middleware*"]
---
# Security Rules
- Validate all inputs at the API boundary
- Prompt injection: system prompt content must never be controllable by request payload
- Never log raw user input at INFO level — log a hash or truncated prefix
- Rate-limit all API endpoints
- Set per-request timeouts
- Never store API keys in the codebase
