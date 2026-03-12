"""
Autoencoder Codebook — Phase 7b Learned Quantization (Mojo implementation).

A lightweight neural autoencoder that learns a low-dimensional INT8 code
space for embedding vectors.

Architecture
    Encoder:  d  → hidden  →  target_dim   (Linear → ReLU → Linear)
    Decoder:  target_dim → hidden  →  d    (Linear → ReLU → Linear)
    Storage:  encoder output is scaled and rounded to int8.

Training uses mini-batch Adam with cosine loss and L2 regularisation.
Weights are Xavier-initialised.

Public API
----------
Codebook.train(data, n, d, n_epochs, lr, batch_size) — fit autoencoder
Codebook.encode(data, n, d, out_codes)               — float32 → int8
Codebook.decode(codes, n, out_recon)                 — int8 → float32
Codebook.mean_cosine(a, b, n, d) -> Float32          — quality metric
Codebook.compression_ratio()     -> Float32          — storage gain
"""

from algorithm import vectorize
from sys.info import simdwidthof
from math import sqrt

alias SIMD_W = simdwidthof[DType.float32]()

alias INT8_MAX_F: Float32 = 127.0
alias ADAM_BETA1:  Float32 = 0.9
alias ADAM_BETA2:  Float32 = 0.999
alias ADAM_EPS:    Float32 = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Low-level numerics
# ─────────────────────────────────────────────────────────────────────────────

fn _xavier_init(
    out_w: UnsafePointer[Float32],
    fan_in: Int,
    fan_out: Int,
    seed: UInt32,
):
    """Xavier / Glorot uniform initialisation written to out_w.

    Scale  = sqrt(2 / (fan_in + fan_out)).
    Values drawn from a deterministic LCG and scaled to [-scale, +scale].

    Args:
        out_w:   Output weight pointer, must hold fan_in * fan_out floats.
        fan_in:  Input dimension.
        fan_out: Output dimension.
        seed:    LCG seed (mutated per element for variety).
    """
    var scale = sqrt(Float32(2) / Float32(fan_in + fan_out))
    var v: UInt32 = seed
    for i in range(fan_in * fan_out):
        v = v * 1664525 + 1013904223
        # Map LCG output to [-1, 1] then scale
        var u = (Float32(Int(v & 0xFFFF)) / 32768.0) - 1.0
        out_w[i] = u * scale


fn _relu_inplace(buf: UnsafePointer[Float32], n: Int):
    """Apply ReLU in place: buf[i] = max(0, buf[i]).

    Args:
        buf: Pointer to float32 buffer.
        n:   Number of elements.
    """
    @parameter
    fn _k[w: Int](i: Int):
        var v = SIMD[DType.float32, w].load(buf + i)
        SIMD[DType.float32, w].store(buf + i, v.max(SIMD[DType.float32, w](0.0)))

    vectorize[_k, SIMD_W](n)


fn _matmul_add(
    dst: UnsafePointer[Float32],
    src: UnsafePointer[Float32],
    w:   UnsafePointer[Float32],
    n:   Int,
    in_dim: Int,
    out_dim: Int,
):
    """Dense matrix multiply: dst (n × out_dim) = src (n × in_dim) @ w (in_dim × out_dim).

    No bias; result accumulates into dst (caller must zero it beforehand).

    Args:
        dst:     Output buffer (n × out_dim), must be zero-initialised.
        src:     Input buffer (n × in_dim).
        w:       Weight matrix (in_dim × out_dim), row-major.
        n:       Batch / number of rows.
        in_dim:  Input dimensionality.
        out_dim: Output dimensionality.
    """
    for i in range(n):
        for k in range(in_dim):
            var src_val = src[i * in_dim + k]
            var w_row   = w + k * out_dim

            @parameter
            fn _k[ww: Int](j: Int):
                var dv = SIMD[DType.float32, ww].load(dst + i * out_dim + j)
                dv += src_val * SIMD[DType.float32, ww].load(w_row + j)
                SIMD[DType.float32, ww].store(dst + i * out_dim + j, dv)

            vectorize[_k, SIMD_W](out_dim)


fn _matmul_t_add(
    dst:     UnsafePointer[Float32],
    src:     UnsafePointer[Float32],
    w:       UnsafePointer[Float32],
    n:       Int,
    in_dim:  Int,
    out_dim: Int,
):
    """dst (n × in_dim) += src (n × out_dim) @ w^T  (out_dim × in_dim) — weight-transposed GEMM.

    Used in the backward pass where we propagate gradient through W^T.

    Args:
        dst:     Output buffer (n × in_dim), must be zero-initialised.
        src:     Gradient buffer (n × out_dim).
        w:       Weight matrix (in_dim × out_dim), row-major.
        n:       Batch size.
        in_dim:  Original input dimension.
        out_dim: Original output dimension.
    """
    for i in range(n):
        for j in range(out_dim):
            var g = src[i * out_dim + j]
            for k in range(in_dim):
                dst[i * in_dim + k] += g * w[k * out_dim + j]


fn _outer_add(
    dst:     UnsafePointer[Float32],
    a:       UnsafePointer[Float32],
    b:       UnsafePointer[Float32],
    n:       Int,
    rows:    Int,
    cols:    Int,
    l2_reg:  Float32,
    w_ptr:   UnsafePointer[Float32],
):
    """Accumulate outer-product gradient into dst: dst += a^T @ b + l2_reg * w.

    Args:
        dst:    Gradient accumulator (rows × cols), zero before first call.
        a:      Left matrix (n × rows).
        b:      Right matrix (n × cols).
        n:      Batch size.
        rows:   Number of rows in dst.
        cols:   Number of columns in dst.
        l2_reg: L2 regularisation coefficient.
        w_ptr:  Current weight values (for L2 gradient = w itself).
    """
    for k in range(n):
        for r in range(rows):
            var av = a[k * rows + r]
            for c in range(cols):
                dst[r * cols + c] += av * b[k * cols + c]

    # L2 gradient
    var total = rows * cols
    for i in range(total):
        dst[i] += l2_reg * w_ptr[i]


fn _adam_step(
    w:   UnsafePointer[Float32],
    g:   UnsafePointer[Float32],
    m:   UnsafePointer[Float32],
    v:   UnsafePointer[Float32],
    t:   Int,
    lr:  Float32,
    size: Int,
):
    """Apply one Adam update step in-place to weight buffer w.

    Uses beta1=0.9, beta2=0.999, eps=1e-8.

    Args:
        w:    Weight buffer (modified in-place).
        g:    Gradient buffer (same size as w).
        m:    First-moment buffer (same size as w).
        v:    Second-moment buffer (same size as w).
        t:    Current time step (1-indexed).
        lr:   Learning rate.
        size: Number of elements.
    """
    var bc1 = Float32(1.0) - ADAM_BETA1 ** Float32(t)
    var bc2 = Float32(1.0) - ADAM_BETA2 ** Float32(t)

    for i in range(size):
        m[i] = ADAM_BETA1 * m[i] + (Float32(1.0) - ADAM_BETA1) * g[i]
        v[i] = ADAM_BETA2 * v[i] + (Float32(1.0) - ADAM_BETA2) * g[i] * g[i]
        var m_hat = m[i] / bc1
        var v_hat = v[i] / bc2
        w[i] -= lr * m_hat / (sqrt(v_hat) + ADAM_EPS)


fn _cosine_loss_grad(
    recon:  UnsafePointer[Float32],
    target: UnsafePointer[Float32],
    n:      Int,
    d:      Int,
    grad:   UnsafePointer[Float32],
) -> Float32:
    """Cosine loss: L = 1 - mean(cos(recon, target)).  Returns loss, writes grad.

    Gradient of (1 - cos) w.r.t. recon:
        dL/d_recon = -(t_hat - r_hat * cos) / (||recon|| + eps)

    averaged over the batch.

    Args:
        recon:  Reconstructed vectors (n × d).
        target: Target  vectors (n × d).
        n:      Batch size.
        d:      Dimensionality.
        grad:   Output gradient buffer (n × d); overwritten.
    Returns:
        Scalar mean cosine loss over the batch.
    """
    var total_loss: Float32 = 0.0
    var inv_n = Float32(1.0) / Float32(n)

    for i in range(n):
        var r = recon  + i * d
        var t = target + i * d
        var g = grad   + i * d
        var eps: Float32 = 1e-8

        var r_norm: Float32 = 0.0
        var t_norm: Float32 = 0.0
        var dot:    Float32 = 0.0

        @parameter
        fn _k[w: Int](j: Int):
            var rv = SIMD[DType.float32, w].load(r + j)
            var tv = SIMD[DType.float32, w].load(t + j)
            dot    += (rv * tv).reduce_add()
            r_norm += (rv * rv).reduce_add()
            t_norm += (tv * tv).reduce_add()

        vectorize[_k, SIMD_W](d)

        r_norm = sqrt(r_norm) + eps
        t_norm = sqrt(t_norm) + eps
        var cos_val = dot / (r_norm * t_norm)
        total_loss += Float32(1.0) - cos_val

        # Gradient: -(t_hat - r_hat * cos) / r_norm  (averaged over batch)
        var inv_r = Float32(1.0) / r_norm
        var inv_t = Float32(1.0) / t_norm
        for j in range(d):
            var r_hat_j = r[j] * inv_r
            var t_hat_j = t[j] * inv_t
            g[j] = -(t_hat_j - r_hat_j * cos_val) * inv_r * inv_n

    return total_loss * inv_n


# ─────────────────────────────────────────────────────────────────────────────
# Codebook struct
# ─────────────────────────────────────────────────────────────────────────────

struct Codebook:
    """Learned INT8 autoencoder codebook.

    Encoder: d → hidden → target_dim (Linear+ReLU+Linear)
    Decoder: target_dim → hidden → d (Linear+ReLU+Linear)
    INT8 codes: encoder output rounded to int8 with per-calibration scale.
    """

    var target_dim:  Int
    var l2_reg:      Float32
    var seed:        UInt32
    var _d:          Int
    var _hidden:     Int
    var _code_scale: Float32
    var is_trained:  Bool

    # Weight matrices stored as List[Float32] (flat row-major)
    var W1e: List[Float32]   # (d       × hidden)
    var W2e: List[Float32]   # (hidden  × target_dim)
    var W1d: List[Float32]   # (target_dim × hidden)
    var W2d: List[Float32]   # (hidden  × d)

    # Adam moment buffers (same shape as corresponding weights)
    var M1e: List[Float32]
    var V1e: List[Float32]
    var M2e: List[Float32]
    var V2e: List[Float32]
    var M1d: List[Float32]
    var V1d: List[Float32]
    var M2d: List[Float32]
    var V2d: List[Float32]

    fn __init__(
        out self,
        target_dim: Int = 64,
        l2_reg: Float32 = 1e-4,
        seed: UInt32 = 42,
    ):
        """Initialise an untrained Codebook.

        Args:
            target_dim: Compressed code dimensionality.
            l2_reg:     L2 weight regularisation coefficient.
            seed:       Deterministic seed for Xavier initialisation and
                        mini-batch shuffling.
        """
        self.target_dim  = target_dim
        self.l2_reg      = l2_reg
        self.seed        = seed
        self._d          = 0
        self._hidden     = 0
        self._code_scale = 1.0
        self.is_trained  = False

        self.W1e = List[Float32]()
        self.W2e = List[Float32]()
        self.W1d = List[Float32]()
        self.W2d = List[Float32]()
        self.M1e = List[Float32]()
        self.V1e = List[Float32]()
        self.M2e = List[Float32]()
        self.V2e = List[Float32]()
        self.M1d = List[Float32]()
        self.V1d = List[Float32]()
        self.M2d = List[Float32]()
        self.V2d = List[Float32]()

    fn _alloc_weights(mut self, d: Int):
        """Allocate and Xavier-initialise all weight matrices.

        Args:
            d: Input/output dimensionality.
        """
        var hidden = max(self.target_dim * 2, min(d, 128))
        self._d      = d
        self._hidden = hidden

        var sz_w1e = d       * hidden
        var sz_w2e = hidden  * self.target_dim
        var sz_w1d = self.target_dim * hidden
        var sz_w2d = hidden  * d

        # Allocate weight buffers
        self.W1e = List[Float32](capacity=sz_w1e)
        self.W2e = List[Float32](capacity=sz_w2e)
        self.W1d = List[Float32](capacity=sz_w1d)
        self.W2d = List[Float32](capacity=sz_w2d)

        # Temp pointers for xavier init
        var p1e = UnsafePointer[Float32].alloc(sz_w1e)
        var p2e = UnsafePointer[Float32].alloc(sz_w2e)
        var p1d = UnsafePointer[Float32].alloc(sz_w1d)
        var p2d = UnsafePointer[Float32].alloc(sz_w2d)

        _xavier_init(p1e, d,              hidden,           self.seed)
        _xavier_init(p2e, hidden,         self.target_dim,  self.seed + 1)
        _xavier_init(p1d, self.target_dim, hidden,          self.seed + 2)
        _xavier_init(p2d, hidden,          d,               self.seed + 3)

        for i in range(sz_w1e): self.W1e.append(p1e[i])
        for i in range(sz_w2e): self.W2e.append(p2e[i])
        for i in range(sz_w1d): self.W1d.append(p1d[i])
        for i in range(sz_w2d): self.W2d.append(p2d[i])

        p1e.free(); p2e.free(); p1d.free(); p2d.free()

        # Zero-init Adam moments
        self.M1e = List[Float32](capacity=sz_w1e)
        self.V1e = List[Float32](capacity=sz_w1e)
        self.M2e = List[Float32](capacity=sz_w2e)
        self.V2e = List[Float32](capacity=sz_w2e)
        self.M1d = List[Float32](capacity=sz_w1d)
        self.V1d = List[Float32](capacity=sz_w1d)
        self.M2d = List[Float32](capacity=sz_w2d)
        self.V2d = List[Float32](capacity=sz_w2d)

        for _ in range(sz_w1e): self.M1e.append(0.0); self.V1e.append(0.0)
        for _ in range(sz_w2e): self.M2e.append(0.0); self.V2e.append(0.0)
        for _ in range(sz_w1d): self.M1d.append(0.0); self.V1d.append(0.0)
        for _ in range(sz_w2d): self.M2d.append(0.0); self.V2d.append(0.0)

    fn _forward_encode(self, x: UnsafePointer[Float32], bs: Int, z: UnsafePointer[Float32]):
        """Run the encoder: x (bs × d) → z (bs × target_dim).

        Args:
            x:  Input batch (row-major).
            bs: Batch size.
            z:  Output code buffer (bs × target_dim), must be zeroed.
        """
        var h = UnsafePointer[Float32].alloc(bs * self._hidden)
        for i in range(bs * self._hidden): h[i] = 0.0

        _matmul_add(h, x,  self.W1e.unsafe_ptr(), bs, self._d,      self._hidden)
        _relu_inplace(h, bs * self._hidden)
        _matmul_add(z, h,  self.W2e.unsafe_ptr(), bs, self._hidden,  self.target_dim)
        h.free()

    fn _forward_decode(self, z: UnsafePointer[Float32], bs: Int, recon: UnsafePointer[Float32]):
        """Run the decoder: z (bs × target_dim) → recon (bs × d).

        Args:
            z:     Code buffer (row-major).
            bs:    Batch size.
            recon: Output buffer (bs × d), must be zeroed.
        """
        var h = UnsafePointer[Float32].alloc(bs * self._hidden)
        for i in range(bs * self._hidden): h[i] = 0.0

        _matmul_add(h,     z,  self.W1d.unsafe_ptr(), bs, self.target_dim, self._hidden)
        _relu_inplace(h, bs * self._hidden)
        _matmul_add(recon, h,  self.W2d.unsafe_ptr(), bs, self._hidden,    self._d)
        h.free()

    fn train(
        mut self,
        data:       UnsafePointer[Float32],
        n:          Int,
        d:          Int,
        n_epochs:   Int = 100,
        lr:         Float32 = 0.01,
        batch_size: Int = 64,
    ):
        """Fit the autoencoder on the training data.

        Normalises input vectors by L2 norm before training to stabilise
        cosine loss.  After training, calibrates the INT8 code scale using
        up to 1000 training vectors.

        Args:
            data:       Row-major (n × d) float32 training buffer.
            n:          Number of training vectors.
            d:          Dimensionality.
            n_epochs:   Number of epochs.
            lr:         Adam learning rate.
            batch_size: Mini-batch size.
        """
        self._alloc_weights(d)
        var hidden = self._hidden
        var tdim   = self.target_dim

        # L2-normalise a copy of the training data
        var norm_data = UnsafePointer[Float32].alloc(n * d)
        for i in range(n):
            var row  = data + i * d
            var nrow = norm_data + i * d
            var norm: Float32 = 0.0
            for j in range(d): norm += row[j] * row[j]
            norm = sqrt(norm)
            if norm < 1e-8: norm = 1.0
            for j in range(d): nrow[j] = row[j] / norm

        var t: Int = 0   # Adam time step

        for _ in range(n_epochs):
            # Simple sequential batch ordering (no shuffle for determinism)
            var start = 0
            while start < n:
                var end = min(start + batch_size, n)
                var bs  = end - start
                var bx  = norm_data + start * d

                # ── Forward ──────────────────────────────
                var h_enc  = UnsafePointer[Float32].alloc(bs * hidden)
                var z_enc  = UnsafePointer[Float32].alloc(bs * tdim)
                var h_dec  = UnsafePointer[Float32].alloc(bs * hidden)
                var recon  = UnsafePointer[Float32].alloc(bs * d)

                for i in range(bs * hidden): h_enc[i] = 0.0
                for i in range(bs * tdim):   z_enc[i] = 0.0
                for i in range(bs * hidden): h_dec[i] = 0.0
                for i in range(bs * d):      recon[i] = 0.0

                _matmul_add(h_enc, bx,    self.W1e.unsafe_ptr(), bs, d,      hidden)
                _relu_inplace(h_enc, bs * hidden)
                _matmul_add(z_enc, h_enc, self.W2e.unsafe_ptr(), bs, hidden, tdim)
                _matmul_add(h_dec, z_enc, self.W1d.unsafe_ptr(), bs, tdim,   hidden)
                _relu_inplace(h_dec, bs * hidden)
                _matmul_add(recon, h_dec, self.W2d.unsafe_ptr(), bs, hidden, d)

                # ── Loss + output gradient ────────────────
                var dL_drecon = UnsafePointer[Float32].alloc(bs * d)
                _ = _cosine_loss_grad(recon, bx, bs, d, dL_drecon)

                # ── Backward: decoder ─────────────────────
                var dL_dW2d = UnsafePointer[Float32].alloc(hidden * d)
                for i in range(hidden * d): dL_dW2d[i] = 0.0
                _outer_add(dL_dW2d, h_dec, dL_drecon, bs, hidden, d, self.l2_reg, self.W2d.unsafe_ptr())

                var dL_dh_dec = UnsafePointer[Float32].alloc(bs * hidden)
                for i in range(bs * hidden): dL_dh_dec[i] = 0.0
                _matmul_t_add(dL_dh_dec, dL_drecon, self.W2d.unsafe_ptr(), bs, hidden, d)

                # ReLU backward for h_dec
                for i in range(bs * hidden):
                    if h_dec[i] <= 0.0: dL_dh_dec[i] = 0.0

                var dL_dW1d = UnsafePointer[Float32].alloc(tdim * hidden)
                for i in range(tdim * hidden): dL_dW1d[i] = 0.0
                _outer_add(dL_dW1d, z_enc, dL_dh_dec, bs, tdim, hidden, self.l2_reg, self.W1d.unsafe_ptr())

                var dL_dz = UnsafePointer[Float32].alloc(bs * tdim)
                for i in range(bs * tdim): dL_dz[i] = 0.0
                _matmul_t_add(dL_dz, dL_dh_dec, self.W1d.unsafe_ptr(), bs, tdim, hidden)

                # ── Backward: encoder ─────────────────────
                var dL_dW2e = UnsafePointer[Float32].alloc(hidden * tdim)
                for i in range(hidden * tdim): dL_dW2e[i] = 0.0
                _outer_add(dL_dW2e, h_enc, dL_dz, bs, hidden, tdim, self.l2_reg, self.W2e.unsafe_ptr())

                var dL_dh_enc = UnsafePointer[Float32].alloc(bs * hidden)
                for i in range(bs * hidden): dL_dh_enc[i] = 0.0
                _matmul_t_add(dL_dh_enc, dL_dz, self.W2e.unsafe_ptr(), bs, hidden, tdim)

                for i in range(bs * hidden):
                    if h_enc[i] <= 0.0: dL_dh_enc[i] = 0.0

                var dL_dW1e = UnsafePointer[Float32].alloc(d * hidden)
                for i in range(d * hidden): dL_dW1e[i] = 0.0
                _outer_add(dL_dW1e, bx, dL_dh_enc, bs, d, hidden, self.l2_reg, self.W1e.unsafe_ptr())

                # ── Adam updates ──────────────────────────
                t += 1
                _adam_step(self.W1e.unsafe_ptr(), dL_dW1e, self.M1e.unsafe_ptr(), self.V1e.unsafe_ptr(), t, lr, d * hidden)
                _adam_step(self.W2e.unsafe_ptr(), dL_dW2e, self.M2e.unsafe_ptr(), self.V2e.unsafe_ptr(), t, lr, hidden * tdim)
                _adam_step(self.W1d.unsafe_ptr(), dL_dW1d, self.M1d.unsafe_ptr(), self.V1d.unsafe_ptr(), t, lr, tdim * hidden)
                _adam_step(self.W2d.unsafe_ptr(), dL_dW2d, self.M2d.unsafe_ptr(), self.V2d.unsafe_ptr(), t, lr, hidden * d)

                # Free batch buffers
                h_enc.free(); z_enc.free(); h_dec.free(); recon.free()
                dL_drecon.free(); dL_dW2d.free(); dL_dh_dec.free()
                dL_dW1d.free(); dL_dz.free(); dL_dW2e.free()
                dL_dh_enc.free(); dL_dW1e.free()
                start += batch_size

        # Calibrate INT8 scale on up to 1000 normalised training vectors
        var calib_n = min(n, 1000)
        var z_calib = UnsafePointer[Float32].alloc(calib_n * tdim)
        for i in range(calib_n * tdim): z_calib[i] = 0.0
        self._forward_encode(norm_data, calib_n, z_calib)

        var abs_max: Float32 = 0.0
        for i in range(calib_n * tdim):
            var av = z_calib[i] if z_calib[i] >= 0.0 else -z_calib[i]
            if av > abs_max: abs_max = av

        self._code_scale = abs_max / INT8_MAX_F if abs_max > 0.0 else 1.0
        z_calib.free()
        norm_data.free()
        self.is_trained = True

    fn encode(
        self,
        data:      UnsafePointer[Float32],
        n:         Int,
        out_codes: UnsafePointer[Int8],
    ):
        """Encode float32 embeddings to INT8 codes.

        Args:
            data:      Row-major (n × d) float32 input buffer.
            n:         Number of vectors.
            out_codes: Output Int8 buffer (n × target_dim).
        """
        var d   = self._d
        var tdim = self.target_dim

        # L2-normalise
        var norm_data = UnsafePointer[Float32].alloc(n * d)
        for i in range(n):
            var row  = data + i * d
            var nrow = norm_data + i * d
            var norm: Float32 = 0.0
            for j in range(d): norm += row[j] * row[j]
            norm = sqrt(norm)
            if norm < 1e-8: norm = 1.0
            for j in range(d): nrow[j] = row[j] / norm

        var z = UnsafePointer[Float32].alloc(n * tdim)
        for i in range(n * tdim): z[i] = 0.0
        self._forward_encode(norm_data, n, z)

        var inv_scale = Float32(1.0) / self._code_scale
        for i in range(n * tdim):
            var v = z[i] * inv_scale
            v = max(-INT8_MAX_F, min(INT8_MAX_F, v))
            out_codes[i] = Int8(Int(v + 0.5 if v >= 0.0 else v - 0.5))

        z.free()
        norm_data.free()

    fn decode(
        self,
        codes:    UnsafePointer[Int8],
        n:        Int,
        out_recon: UnsafePointer[Float32],
    ):
        """Decode INT8 codes back to float32 embeddings.

        Args:
            codes:     Input Int8 buffer (n × target_dim).
            n:         Number of vectors.
            out_recon: Output float32 buffer (n × d), must be zeroed.
        """
        var tdim = self.target_dim

        var z = UnsafePointer[Float32].alloc(n * tdim)
        for i in range(n * tdim):
            z[i] = Float32(Int(codes[i])) * self._code_scale

        self._forward_decode(z, n, out_recon)
        z.free()

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
            Scalar mean cosine similarity.
        """
        var total: Float32 = 0.0
        var d = self._d

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
        """Storage compression ratio: float32 bytes / int8 code bytes.

        Returns:
            (d × 4) / target_dim — or 0.0 if not yet trained.
        """
        if self._d == 0:
            return 0.0
        return Float32(self._d * 4) / Float32(max(self.target_dim, 1))

    fn print_info(self):
        """Print a summary of this codebook's architecture."""
        print("Codebook: d=" + String(self._d)
              + " hidden=" + String(self._hidden)
              + " target_dim=" + String(self.target_dim)
              + " trained=" + String(self.is_trained)
              + " ratio=" + String(self.compression_ratio()) + "x")


fn main():
    """Smoke-test: train a tiny codebook on random data."""
    var n = 64
    var d = 32
    var buf = UnsafePointer[Float32].alloc(n * d)
    var v: UInt32 = 99991
    for i in range(n * d):
        v = v * 1664525 + 1013904223
        buf[i] = (Float32(Int(v & 0xFFFF)) / 32768.0) - 1.0

    var cb = Codebook(target_dim=16, l2_reg=1e-4, seed=42)
    cb.train(buf, n, d, n_epochs=5, lr=0.01, batch_size=32)
    cb.print_info()

    var codes = UnsafePointer[Int8].alloc(n * 16)
    cb.encode(buf, n, codes)

    var recon = UnsafePointer[Float32].alloc(n * d)
    for i in range(n * d): recon[i] = 0.0
    cb.decode(codes, n, recon)

    var sim = cb.mean_cosine(buf, recon, n)
    print("Mean cosine similarity:", sim)

    buf.free(); codes.free(); recon.free()
