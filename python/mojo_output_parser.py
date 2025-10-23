import re

"""
Simple parser for Mojo test output produced by `mojo run src/test.mojo`.
It extracts quantize/reconstruct throughput and average cosine similarity.
Expected lines in mojo output (examples):

Quantize throughput: 123456.0 vectors/second
Reconstruct throughput: 654321.0 vectors/second
Average cosine similarity: 0.9998

The parser returns a dict with keys: throughput, quality, quant_time, recon_time, compression_ratio
Where `throughput` is quantize vec/s, `quality` is average cosine, times are None (not provided by mojo), compression_ratio left as None.
"""

QUANT_RE = re.compile(r"Quantize throughput:\s*([0-9.]+)\s*vectors/second")
RECON_RE = re.compile(r"Reconstruct throughput:\s*([0-9.]+)\s*vectors/second")
COS_RE = re.compile(r"Average cosine similarity:\s*([0-9.]+(?:\.[0-9]+)?)")


def parse_mojo_output(path: str) -> dict:
    """Parse mojo output file and return metrics dict."""
    data = {
        'throughput': None,
        'recon_throughput': None,
        'quality': None,
        'quant_time': None,
        'recon_time': None,
        'compression_ratio': None,
    }
    try:
        with open(path, 'r') as f:
            txt = f.read()
    except Exception:
        return data

    m = QUANT_RE.search(txt)
    if m:
        try:
            data['throughput'] = float(m.group(1))
        except Exception:
            pass
    m = RECON_RE.search(txt)
    if m:
        try:
            data['recon_throughput'] = float(m.group(1))
        except Exception:
            pass
    m = COS_RE.search(txt)
    if m:
        try:
            data['quality'] = float(m.group(1))
        except Exception:
            pass

    return data
