"""Run Mojo test/benchmark scripts and parse their output.

This module provides a simple wrapper that invokes the Mojo runtime (if
available) to run the `src/test.mojo` benchmark script that lives in the repo
and extracts throughput and quality metrics from its stdout. It's intended for
demo/visualization purposes (not for production hot paths).

Behavior:
- If `mojo` is on PATH, `run_mojo_benchmark()` will execute `mojo run src/test.mojo`
  and return a dict with keys: 'throughput' (vec/s float), 'quality' (cosine float),
  'quant_time' (s), 'recon_time' (s). On failure it returns None.
"""
from __future__ import annotations
import subprocess
import shutil
import re
import os
from typing import Optional, Dict


def mojo_binary_path() -> Optional[str]:
    """Return path to mojo binary.

    Checks the following in order:
    1. VECTRO_MOJO_PATH environment variable
    2. ./mojo or ./mojo/bin/mojo inside the repo
    3. mojo on PATH
    """
    env_path = os.environ.get('VECTRO_MOJO_PATH')
    if env_path and os.path.exists(env_path):
        return env_path

    # check common local mojo dir
    local_candidates = [os.path.join(os.getcwd(), 'mojo'), os.path.join(os.getcwd(), 'mojo', 'bin', 'mojo')]
    for p in local_candidates:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    # fallback to PATH
    which = shutil.which('mojo')
    return which


def mojo_available() -> bool:
    """Return True if a mojo binary is locatable."""
    return mojo_binary_path() is not None


def run_mojo_benchmark(timeout: int = 60) -> Optional[Dict[str, float]]:
    """Run the Mojo benchmark script and parse results.

    Returns None on error or when mojo is not available.
    """
    mojo_bin = mojo_binary_path()
    if not mojo_bin:
        return None

    # Run the Mojo test script which prints human-friendly lines we can parse
    cmd = [mojo_bin, "run", "src/test.mojo"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None

    out = proc.stdout + "\n" + proc.stderr

    # Parse throughput lines like: "Quantize throughput: 328000.0 vectors/second"
    m_q = re.search(r"Quantize throughput:\s*([0-9,.]+)\s*vectors/second", out)
    m_r = re.search(r"Reconstruct throughput:\s*([0-9,.]+)\s*vectors/second", out)
    m_cos = re.search(r"Average cosine similarity:\s*([0-9.]+)", out)

    if not m_q and not m_r and not m_cos:
        # Try alternative patterns that might appear in different Mojo outputs
        m_q2 = re.search(r"Quantize throughput:\s*([0-9,.]+)", out)
        m_r2 = re.search(r"Reconstruct throughput:\s*([0-9,.]+)", out)
        m_cos2 = re.search(r"Average cosine similarity:\s*([0-9.]+)", out)
        m_q = m_q or m_q2
        m_r = m_r or m_r2
        m_cos = m_cos or m_cos2

    try:
        quant_vps = float(m_q.group(1).replace(',', '')) if m_q else 0.0
        recon_vps = float(m_r.group(1).replace(',', '')) if m_r else 0.0
        avg_cos = float(m_cos.group(1)) if m_cos else 0.0
    except Exception:
        return None

    # Convert throughput to times per-run assuming the script uses n vectors;
    # We don't have n here, so we'll provide throughput and zeros for times.
    return {
        'throughput': quant_vps,
        'quality': avg_cos,
        'quant_time': 0.0,
        'recon_time': 0.0,
    }
