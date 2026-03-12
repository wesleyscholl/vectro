"""
Residual Quantizer (RQ) — Phase 7a Learned Quantization (Mojo implementation).

A Residual Quantizer chains multiple Product Quantizer (PQ) codebooks: each
pass encodes the *residual* (original minus current reconstruction) left by
all previous passes.  The final reconstruction is the sum of all per-pass
decodings.  This mirrors the Python layer (python/rq_api.py) in pure Mojo
with SIMD-accelerated distance computations.

Algorithm
---------
1. Pass 0 — fit PQ codebook on the raw data.          Encode → decode → compute residual.
2. Pass 1 — fit PQ codebook on the residual from pass 0.  Encode → decode → compute residual.
...
N. Pass n_passes-1 — fit PQ codebook on the running residual.

Reconstruction = sum of all per-pass decodings.

Compression ratio
-----------------
    float32 storage: d × 4 bytes per vector
    code storage:    n_passes × n_subspaces bytes per vector  (uint8 per sub-space, K ≤ 256)

Public API
----------
ResidualQuantizer.train(data, n, d)              — fit n_passes codebooks
ResidualQuantizer.encode(data, n, d, out_codes)  — (n, n_passes × n_subspaces) uint8
ResidualQuantizer.decode(codes, n, out_recon)    — (n, d) float32
ResidualQuantizer.mean_cosine(a, b, n, d)        — quality metric
ResidualQuantizer.compression_ratio()            — ratio vs float32
"""

from algorithm import vectorize
from sys.info import simdwidthof
from math import sqrt

alias SIMD_W = simdwidthof[DType.float32]()

# Maximum sub-spaces and centroids supported
alias MAX_SUBSPACES:  Int = 128
alias MAX_CENTROIDS:  Int = 256
alias MAX_RQ_PASSES:  Int = 8


# ─────────────────────────────────────────────────────────────────────────────
# Centroid initialisation (k-means++)
# ─────────────────────────────────────────────────────────────────────────────

fn _l2_sq(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    dim: Int,
) -> Float32:
    """Squared-L2 distance between two vectors of length dim.

    Args:
        a:   Pointer to first vector.
        b:   Pointer to second vector.
        dim: Vector length.
    Returns:
        Squared L2 distance.
    """
    var dist: Float32 = 0.0

    @parameter
    fn _k[w: Int](j: Int):
        var diff = SIMD[DType.float32, w].load(a + j) - SIMD[DType.float32, w].load(b + j)
        dist += (diff * diff).reduce_add()

    vectorize[_k, SIMD_W](dim)
    return dist


fn _nearest_centroid(
    vec:       UnsafePointer[Float32],
    centroids: UnsafePointer[Float32],
    K:         Int,
    sub_dim:   Int,
) -> Int:
    """Return the index of the nearest centroid to vec.

    Args:
        vec:       Query sub-vector pointer (length sub_dim).
        centroids: Centroid table (K × sub_dim), row-major.
        K:         Number of centroids.
        sub_dim:   Sub-space dimensionality.
    Returns:
        Index 0 .. K-1 of the nearest centroid.
    """
    var best_dist: Float32 = 1e38
    var best_k: Int = 0
    for k in range(K):
        var d = _l2_sq(vec, centroids + k * sub_dim, sub_dim)
        if d < best_dist:
            best_dist = d
            best_k    = k
    return best_k


fn _kmeans_init_plusplus(
    data:      UnsafePointer[Float32],
    n:         Int,
    sub_dim:   Int,
    K:         Int,
    centroids: UnsafePointer[Float32],
    seed:      UInt32,
):
    """K-means++ centroid seeding.

    Selects the first centroid uniformly at random, then each subsequent
    centroid with probability proportional to squared distance to the
    nearest already-chosen centroid.

    Args:
        data:      Data matrix (n × sub_dim).
        n:         Number of data points.
        sub_dim:   Dimensionality.
        K:         Number of centroids to initialise.
        centroids: Output buffer (K × sub_dim); written in-place.
        seed:      LCG seed for deterministic behaviour.
    """
    var rng: UInt32 = seed

    # Choose first centroid uniformly
    rng = rng * 1664525 + 1013904223
    var first = Int(rng) % n
    for j in range(sub_dim):
        centroids[j] = data[first * sub_dim + j]

    # Temporary distance buffer
    var dists = UnsafePointer[Float32].alloc(n)
    for i in range(n):
        dists[i] = 1e38

    for c in range(1, K):
        # Update distances to nearest chosen centroid
        var c_ptr = centroids + (c - 1) * sub_dim
        for i in range(n):
            var d = _l2_sq(data + i * sub_dim, c_ptr, sub_dim)
            if d < dists[i]:
                dists[i] = d

        # Weighted random selection
        var total: Float32 = 0.0
        for i in range(n): total += dists[i]

        rng = rng * 1664525 + 1013904223
        var threshold = (Float32(Int(rng & 0xFFFF)) / 65535.0) * total

        var cumsum: Float32 = 0.0
        var chosen: Int = n - 1
        for i in range(n):
            cumsum += dists[i]
            if cumsum >= threshold:
                chosen = i
                break

        var c_out = centroids + c * sub_dim
        for j in range(sub_dim):
            c_out[j] = data[chosen * sub_dim + j]

    dists.free()


fn _kmeans_train(
    data:      UnsafePointer[Float32],
    n:         Int,
    sub_dim:   Int,
    K:         Int,
    centroids: UnsafePointer[Float32],
    max_iter:  Int,
    seed:      UInt32,
):
    """Lloyd's K-means fitting for one sub-space.

    Initialises via k-means++ then runs up to max_iter Lloyd iterations.
    Updates centroids in-place.

    Args:
        data:      Data sub-vectors (n × sub_dim).
        n:         Number of points.
        sub_dim:   Sub-space dimensionality.
        K:         Number of centroids.
        centroids: In/out buffer (K × sub_dim).
        max_iter:  Maximum Lloyd iterations.
        seed:      Seed for k-means++ initialisation.
    """
    _kmeans_init_plusplus(data, n, sub_dim, K, centroids, seed)

    var assign  = UnsafePointer[Int].alloc(n)
    var new_c   = UnsafePointer[Float32].alloc(K * sub_dim)
    var counts  = UnsafePointer[Int].alloc(K)

    for _ in range(max_iter):
        # Assignment
        for i in range(n):
            assign[i] = _nearest_centroid(data + i * sub_dim, centroids, K, sub_dim)

        # Update
        for i in range(K * sub_dim): new_c[i] = 0.0
        for i in range(K): counts[i] = 0

        for i in range(n):
            var k   = assign[i]
            counts[k] += 1
            var c_ptr = new_c + k * sub_dim
            for j in range(sub_dim):
                c_ptr[j] += data[i * sub_dim + j]

        for k in range(K):
            if counts[k] > 0:
                var inv = Float32(1.0) / Float32(counts[k])
                for j in range(sub_dim):
                    centroids[k * sub_dim + j] = new_c[k * sub_dim + j] * inv

    assign.free()
    new_c.free()
    counts.free()


# ─────────────────────────────────────────────────────────────────────────────
# Single-pass PQ codebook
# ─────────────────────────────────────────────────────────────────────────────

struct PQPass:
    """One PQ pass: n_subspaces sub-codebooks, each with K centroids of size sub_dim."""

    var n_subspaces: Int
    var K:           Int
    var sub_dim:     Int
    var d_orig:      Int
    var d_padded:    Int

    # Flat centroid storage: (n_subspaces × K × sub_dim) float32
    var centroids: List[Float32]

    fn __init__(out self, n_subspaces: Int, K: Int, d: Int):
        """Allocate an empty PQ pass for input dimension d.

        Args:
            n_subspaces: Number of sub-spaces M.
            K:           Number of centroids per sub-space (≤ 256).
            d:           Original vector dimensionality.
        """
        self.n_subspaces = n_subspaces
        self.K           = K
        self.d_orig      = d
        # Pad d so it is divisible by n_subspaces
        var pad = (n_subspaces - (d % n_subspaces)) % n_subspaces
        self.d_padded = d + pad
        self.sub_dim  = self.d_padded // n_subspaces
        var total = n_subspaces * K * self.sub_dim
        self.centroids = List[Float32](capacity=total)
        for _ in range(total): self.centroids.append(0.0)

    fn train(mut self, data: UnsafePointer[Float32], n: Int, seed: UInt32, max_iter: Int = 25):
        """Fit one sub-codebook per sub-space.

        Args:
            data:     Input data (n × d_orig), row-major.
            n:        Number of vectors.
            seed:     Base seed; incremented per sub-space.
            max_iter: K-means iterations per sub-space.
        """
        var sub_dim   = self.sub_dim
        var d_padded  = self.d_padded
        var d_orig    = self.d_orig
        var M         = self.n_subspaces

        # Build padded copy if needed
        var padded = UnsafePointer[Float32].alloc(n * d_padded)
        for i in range(n):
            for j in range(d_orig):
                padded[i * d_padded + j] = data[i * d_orig + j]
            for j in range(d_orig, d_padded):
                padded[i * d_padded + j] = 0.0

        # Sub-vector buffer for KMeans
        var sub_buf = UnsafePointer[Float32].alloc(n * sub_dim)

        for m in range(M):
            for i in range(n):
                for j in range(sub_dim):
                    sub_buf[i * sub_dim + j] = padded[i * d_padded + m * sub_dim + j]

            var c_ptr = self.centroids.unsafe_ptr() + m * self.K * sub_dim
            _kmeans_train(sub_buf, n, sub_dim, self.K, c_ptr, max_iter, seed + UInt32(m * 7))

        sub_buf.free()
        padded.free()

    fn encode(self, vec: UnsafePointer[Float32], out_codes: UnsafePointer[UInt8]):
        """Encode one vector into n_subspaces uint8 codes.

        Args:
            vec:       Input float32 vector (d_orig elements).
            out_codes: Output uint8 buffer (n_subspaces elements).
        """
        var sub_dim   = self.sub_dim
        var d_padded  = self.d_padded
        var d_orig    = self.d_orig
        var M         = self.n_subspaces

        var padded = UnsafePointer[Float32].alloc(d_padded)
        for j in range(d_orig):  padded[j] = vec[j]
        for j in range(d_orig, d_padded): padded[j] = 0.0

        for m in range(M):
            var sub    = padded + m * sub_dim
            var c_ptr  = self.centroids.unsafe_ptr() + m * self.K * sub_dim
            out_codes[m] = UInt8(_nearest_centroid(sub, c_ptr, self.K, sub_dim))

        padded.free()

    fn decode(self, codes: UnsafePointer[UInt8], out_vec: UnsafePointer[Float32]):
        """Decode n_subspaces uint8 codes into a float32 vector.

        The output has d_orig valid elements (padding is discarded).

        Args:
            codes:   uint8 buffer (n_subspaces elements).
            out_vec: Output float32 buffer (d_orig elements); must be zeroed.
        """
        var sub_dim   = self.sub_dim
        var d_orig    = self.d_orig
        var M         = self.n_subspaces

        for m in range(M):
            var k     = Int(codes[m])
            var c_ptr = self.centroids.unsafe_ptr() + (m * self.K + k) * sub_dim
            for j in range(sub_dim):
                var global_j = m * sub_dim + j
                if global_j < d_orig:
                    out_vec[global_j] += c_ptr[j]


# ─────────────────────────────────────────────────────────────────────────────
# ResidualQuantizer
# ─────────────────────────────────────────────────────────────────────────────

struct ResidualQuantizer:
    """Multi-pass Residual Quantizer.

    Parameters
    ----------
    n_passes    : number of PQ passes (1–8)
    n_subspaces : M — sub-spaces per PQ pass (divisor of d after padding)
    n_centroids : K — centroids per sub-space (≤ 256)
    seed        : base random seed
    """

    var n_passes:    Int
    var n_subspaces: Int
    var n_centroids: Int
    var seed:        UInt32
    var _d_orig:     Int
    var is_trained:  Bool
    var passes:      List[PQPass]

    fn __init__(
        out self,
        n_passes:    Int = 3,
        n_subspaces: Int = 8,
        n_centroids: Int = 64,
        seed:        UInt32 = 0,
    ):
        """Create an untrained ResidualQuantizer.

        Args:
            n_passes:    Number of PQ passes (stacked residual stages).
            n_subspaces: Sub-spaces per PQ pass.
            n_centroids: Centroids per sub-space (K ≤ 256).
            seed:        Base random seed for reproducibility.
        """
        self.n_passes    = min(n_passes, MAX_RQ_PASSES)
        self.n_subspaces = min(n_subspaces, MAX_SUBSPACES)
        self.n_centroids = min(n_centroids, MAX_CENTROIDS)
        self.seed        = seed
        self._d_orig     = 0
        self.is_trained  = False
        self.passes      = List[PQPass]()

    fn train(mut self, data: UnsafePointer[Float32], n: Int, d: Int):
        """Fit n_passes PQ codebooks on progressively smaller residuals.

        Pass 0: fit on raw data.
        Pass p: fit on (data - sum of previous reconstructions).

        Args:
            data: Row-major (n × d) float32 training buffer.
            n:    Number of training vectors.
            d:    Dimensionality.
        """
        self._d_orig = d
        self.passes  = List[PQPass]()

        var residual = UnsafePointer[Float32].alloc(n * d)
        var recon    = UnsafePointer[Float32].alloc(d)

        # Copy raw data into residual buffer
        for i in range(n * d):
            residual[i] = data[i]

        var codes_buf = UnsafePointer[UInt8].alloc(n * self.n_subspaces)

        for p in range(self.n_passes):
            var pass_ = PQPass(self.n_subspaces, self.n_centroids, d)
            pass_.train(residual, n, self.seed + UInt32(p * 100))

            # Encode all vectors and subtract reconstruction from residual
            for i in range(n):
                var vec_ptr   = residual + i * d
                var code_ptr  = codes_buf + i * self.n_subspaces
                pass_.encode(vec_ptr, code_ptr)

                for j in range(d): recon[j] = 0.0
                pass_.decode(code_ptr, recon)
                for j in range(d): vec_ptr[j] -= recon[j]

            self.passes.append(pass_)

        codes_buf.free()
        recon.free()
        residual.free()
        self.is_trained = True

    fn encode(
        self,
        data:      UnsafePointer[Float32],
        n:         Int,
        out_codes: UnsafePointer[UInt8],
    ):
        """Encode vectors through all passes.

        Output layout: codes for pass 0 || pass 1 || ... || pass n_passes-1,
        each block being n × n_subspaces uint8 values.
        Total output size: n × (n_passes × n_subspaces) bytes.

        Args:
            data:      Input (n × d) float32 buffer.
            n:         Number of vectors.
            out_codes: Output buffer (n × n_passes × n_subspaces bytes).
        """
        var d         = self._d_orig
        var M         = self.n_subspaces
        var residual  = UnsafePointer[Float32].alloc(n * d)
        var recon_row = UnsafePointer[Float32].alloc(d)

        for i in range(n * d): residual[i] = data[i]

        for p in range(self.n_passes):
            var pass_ = self.passes[p]
            for i in range(n):
                var vec_ptr  = residual + i * d
                var code_ptr = out_codes + (p * n + i) * M
                pass_.encode(vec_ptr, code_ptr)

                for j in range(d): recon_row[j] = 0.0
                pass_.decode(code_ptr, recon_row)
                for j in range(d): vec_ptr[j] -= recon_row[j]

        recon_row.free()
        residual.free()

    fn decode(
        self,
        codes:     UnsafePointer[UInt8],
        n:         Int,
        out_recon: UnsafePointer[Float32],
    ):
        """Reconstruct float32 vectors from multi-pass codes.

        Args:
            codes:     uint8 buffer (n × n_passes × n_subspaces), layout from encode().
            n:         Number of vectors.
            out_recon: Output float32 buffer (n × d); must be zeroed.
        """
        var d  = self._d_orig
        var M  = self.n_subspaces
        var row_buf = UnsafePointer[Float32].alloc(d)

        for p in range(self.n_passes):
            var pass_ = self.passes[p]
            for i in range(n):
                var code_ptr = codes + (p * n + i) * M
                var out_row  = out_recon + i * d
                for j in range(d): row_buf[j] = 0.0
                pass_.decode(code_ptr, row_buf)
                for j in range(d): out_row[j] += row_buf[j]

        row_buf.free()

    fn mean_cosine(
        self,
        original:      UnsafePointer[Float32],
        reconstructed: UnsafePointer[Float32],
        n: Int,
    ) -> Float32:
        """Mean per-vector cosine similarity.

        Args:
            original:      Original vectors (n × d).
            reconstructed: Reconstructed vectors (n × d).
            n:             Number of vectors.
        Returns:
            Mean cosine similarity.
        """
        var d     = self._d_orig
        var total: Float32 = 0.0

        for i in range(n):
            var a = original      + i * d
            var b = reconstructed + i * d
            var dot: Float32 = 0.0
            var na:  Float32 = 0.0
            var nb:  Float32 = 0.0

            @parameter
            fn _k[w: Int](j: Int):
                var va = SIMD[DType.float32, w].load(a + j)
                var vb = SIMD[DType.float32, w].load(b + j)
                dot += (va * vb).reduce_add()
                na  += (va * va).reduce_add()
                nb  += (vb * vb).reduce_add()

            vectorize[_k, SIMD_W](d)

            var denom = sqrt(na) * sqrt(nb)
            total += dot / denom if denom > 0.0 else Float32(1.0)

        return total / Float32(n)

    fn compression_ratio(self) -> Float32:
        """Compression ratio vs float32 storage.

        float32: d × 4 bytes/vector
        codes:   n_passes × n_subspaces bytes/vector

        Returns:
            Ratio float32_bytes / code_bytes.
        """
        var code_bytes  = self.n_passes * self.n_subspaces
        var float_bytes = self._d_orig * 4 if self._d_orig > 0 else 1
        return Float32(float_bytes) / Float32(max(code_bytes, 1))

    fn print_info(self):
        """Print a summary of this residual quantizer."""
        print("ResidualQuantizer:"
              + " passes=" + String(self.n_passes)
              + " subspaces=" + String(self.n_subspaces)
              + " centroids=" + String(self.n_centroids)
              + " d=" + String(self._d_orig)
              + " trained=" + String(self.is_trained)
              + " ratio=" + String(self.compression_ratio()) + "x")


fn main():
    """Smoke-test: train a 2-pass RQ on random data and print quality."""
    var n = 128
    var d = 32
    var buf = UnsafePointer[Float32].alloc(n * d)
    var v: UInt32 = 54321
    for i in range(n * d):
        v = v * 1664525 + 1013904223
        buf[i] = (Float32(Int(v & 0xFFFF)) / 32768.0) - 1.0

    var rq = ResidualQuantizer(n_passes=2, n_subspaces=4, n_centroids=32, seed=0)
    rq.train(buf, n, d)
    rq.print_info()

    var M         = rq.n_subspaces
    var n_passes  = rq.n_passes
    var codes_buf = UnsafePointer[UInt8].alloc(n * n_passes * M)
    rq.encode(buf, n, codes_buf)

    var recon = UnsafePointer[Float32].alloc(n * d)
    for i in range(n * d): recon[i] = 0.0
    rq.decode(codes_buf, n, recon)

    var sim = rq.mean_cosine(buf, recon, n)
    print("Mean cosine similarity:", sim)

    buf.free(); codes_buf.free(); recon.free()
