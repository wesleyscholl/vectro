---
paths: ["**/*.py"]
---
# Python Conventions
- No bare `except:` — catch specific exceptions; log with logging.warning and re-raise
- No mutable default arguments
- `ruff check` and `ruff format` must be clean
- `mypy --strict` must be clean
- No dead code — `vulture` zero tolerance
- `radon cc -n C` zero functions above grade C
- Use `logging` not `print` in production code
