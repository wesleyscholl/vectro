"""
Mojo quantizer stubs for Vectro.
This file provides function signatures and lightweight reference implementations
so the Python wrapper can call into Mojo later when available.
"""

# Note: Mojo is under active development; this is a placeholder API.

fn quantize_int8(embeddings: [f32]) -> (q: [i8], scale: f32, dims: i32):
    """Quantize a flat embeddings array to int8 with a global scale.
    Returns tuple (quantized_flat_array, scale, dims_per_vector).
    This stub is intentionally minimal and primarily documents the intended API.
    """
    # Not implemented in Mojo stub; raise to indicate runtime fallback to Python.
    raise NotImplementedError()


fn reconstruct_int8(q: [i8], scale: f32) -> [f32]:
    """Reconstruct float embeddings from int8 and scale."""
    raise NotImplementedError()
