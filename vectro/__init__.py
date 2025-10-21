"""Compatibility shim package so `vectro.src` resolves to the repository `src` package

This file ensures that after an editable install the import `vectro.src.quantizer`
works by placing the existing top-level `src` package into `sys.modules` under
the `vectro.src` name when possible.
"""
from __future__ import annotations
import importlib
import sys

try:
    # try to import the top-level src package (the editable installer maps it)
    _src = importlib.import_module('src')
    # Ensure that `vectro.src` points to the same module object
    sys.modules.setdefault('vectro.src', _src)
except Exception:
    # If the `src` package is not available yet, do nothing — the editable
    # importer will provide it later.
    pass
try:
    _py = importlib.import_module('python')
    sys.modules.setdefault('vectro.python', _py)
except Exception:
    pass

__all__ = ['src', 'python']
