# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 4.x     | ✅ Yes             |
| < 4.0   | ❌ No              |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### Preferred method — GitHub Security Advisories

Open a private security advisory at:
<https://github.com/wesleyscholl/vectro/security/advisories/new>

### Alternative

Email the maintainer directly.  The address is on the
[GitHub profile](https://github.com/wesleyscholl).

### What to include

- A description of the vulnerability and its potential impact
- Step-by-step instructions to reproduce
- Affected versions
- Any known mitigations or workarounds

---

## Coordinated Disclosure Policy

We follow a **90-day coordinated disclosure** policy:

1. You report the issue privately.
2. We acknowledge receipt **within 72 hours**.
3. We provide a fix or mitigation plan **within 14 days**.
4. After 90 days (or earlier if a fix is shipped), you are free to disclose
   publicly.

We will credit reporters in release notes unless anonymity is requested.

---

## Scope

Vectro is a **pure library** — no network traffic, no server component, no
remote telemetry.

- **In scope:** memory-corruption bugs in quantization kernels, path-traversal
  in file I/O helpers, unsafe deserialization of `.vqz` files.
- **Out of scope:** vulnerabilities in Hugging Face Hub model downloads
  (those are user-initiated and handled by the `huggingface_hub` library).

If you are unsure whether an issue is in scope, err on the side of reporting
privately.
