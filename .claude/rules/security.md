---
paths:
  - "**/lopi-ui/**"
  - "**/lopi-webhook/**"
  - "**/lopi-remote/**"
  - "**/api*"
  - "**/server*"
  - "**/webhook*"
  - "**/auth*"
---
# Security Rules

- Validate all inputs at the API boundary: max goal length, max batch size, character set constraints
- Prompt injection is a real attack surface — system prompt content must never be controllable by request payload
- Never log raw user goal content at INFO level or above in production — log a hash or truncated prefix
- Rate-limit all endpoints by default
- Set and enforce per-request timeouts on every agent run
- HMAC-verify all GitHub webhook signatures (HMAC-SHA256 + constant-time comparison — already in v0.3.0, maintain it)
- Telegram bot: validate `chat_id` against config allowlist before executing any command
- Never store API keys or tokens in the codebase — use environment variables
