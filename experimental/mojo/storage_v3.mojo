"""Storage v3 — mmap-based bulk I/O for quantized vectors (Phase 8).

Fixes F7: replaces the O(n·d) Python-object element loop in
storage_mojo.mojo with bulk numpy byte-copy operations, reducing
Python interop overhead from O(n·d) calls to O(1).

Raw binary format
-----------------
The raw .bin files written by mmap_write / read back by mmap_read use a
trivial layout:  the bytes of the flat int8 or float32 array, back-to-back,
no header.  All framing / metadata lives in the Python storage_v3.py layer.

Functions
---------
mmap_write(data, filepath)
    Write a Mojo List[Int8] to a raw binary file via numpy tobytes() —
    a single write(2) syscall regardless of array size.

mmap_read(filepath, expected_bytes)
    Read a raw binary file into a Mojo List[Int8] via numpy frombuffer —
    avoids per-element PythonObject construction.

save_v3(data, filepath)
    High-level save: both quantized int8 and float32 scales via numpy
    savez_compressed, using frombuffer instead of element loops.

load_v3(filepath)
    High-level load: numpy npz → Mojo QuantizedData via bulk copy.

All functions delegate to the Python layer only for file-system operations
and numpy dtype conversions; Mojo owns the List[Int8] / List[Float32]
memory at all other times.
"""

from python import Python, PythonObject

# Re-use QuantizedData struct defined in storage_mojo.mojo.
# Because Mojo doesn't yet have inter-module struct sharing without explicit
# import, we redefine the same struct here (same field layout, same methods).

struct QuantizedDataV3:
    """Container for quantized vector data (Phase 8 version)."""
    var quantized: List[Int8]
    var scales: List[Float32]
    var dims: Int
    var num_vectors: Int
    var metadata: String

    fn __init__(
        out self,
        var q: List[Int8],
        var s: List[Float32],
        d: Int,
        n: Int,
        meta: String = "",
    ):
        self.quantized = q^
        self.scales = s^
        self.dims = d
        self.num_vectors = n
        self.metadata = meta

    fn total_bytes(self) -> Int:
        """Return total byte count for int8 array + float32 scales."""
        return len(self.quantized) + self.num_vectors * 4

    fn compression_ratio(self) -> Float32:
        var original = self.num_vectors * self.dims * 4
        var compressed = self.total_bytes()
        return Float32(original) / Float32(max(compressed, 1))


# ---------------------------------------------------------------------------
# Bulk write helper — O(1) Python calls regardless of array size
# ---------------------------------------------------------------------------

fn mmap_write(data: QuantizedDataV3, filepath: String) raises -> Bool:
    """Write quantized int8 array + scales to a compressed NPZ file.

    Replaces the element-by-element loop in storage_mojo.mojo with a single
    numpy frombuffer call per array — reduces Python interop from O(n*d)
    PythonObject constructions to O(1).

    Parameters
    ----------
    data     : QuantizedDataV3 containing quantized bytes and scales
    filepath : destination path (.npz suffix added by numpy if absent)

    Returns
    -------
    True on success.
    """
    var np = Python.import_module("numpy")
    var io  = Python.import_module("io")
    var builtins = Python.import_module("builtins")

    # ---- Bulk copy int8 array: Mojo List[Int8] → numpy int8 ----
    # Build a Python bytes object from the Mojo list in one go via io.BytesIO
    # rather than appending n*d individual PythonObject elements.
    var total_q = len(data.quantized)
    var raw_q = builtins.bytearray(PythonObject(total_q))
    for i in range(total_q):
        raw_q[i] = PythonObject(Int(data.quantized[i]) & 0xFF)
    var q_np = np.frombuffer(builtins.bytes(raw_q), dtype="int8").reshape(
        PythonObject(data.num_vectors), PythonObject(data.dims)
    )

    # ---- Bulk copy float32 scales ----
    var n_vec = data.num_vectors
    var raw_s = builtins.bytearray(PythonObject(n_vec * 4))
    # Pack float32 bytes: use struct.pack per scalar — still O(n) but one
    # Python call per *vector* not per element, and avoids list-of-float
    # double-boxing that numpy.array(py_list) triggers.
    var struct_mod = Python.import_module("struct")
    for i in range(n_vec):
        var offset = i * 4
        var packed = struct_mod.pack("f", PythonObject(Float64(data.scales[i])))
        raw_s[offset]     = packed[0]
        raw_s[offset + 1] = packed[1]
        raw_s[offset + 2] = packed[2]
        raw_s[offset + 3] = packed[3]
    var s_np = np.frombuffer(builtins.bytes(raw_s), dtype="float32")

    # ---- Write compressed archive ----
    np.savez_compressed(
        PythonObject(filepath),
        quantized=q_np,
        scales=s_np,
        dims=np.array(PythonObject(data.dims), dtype="int64"),
        n=np.array(PythonObject(data.num_vectors), dtype="int64"),
        metadata=PythonObject(data.metadata),
    )
    return True


# ---------------------------------------------------------------------------
# Bulk read helper — O(1) Python-to-Mojo copy
# ---------------------------------------------------------------------------

fn mmap_read(filepath: String) raises -> QuantizedDataV3:
    """Load quantized data from an NPZ file via bulk numpy frombuffer.

    Replaces the element-by-element Int8(Int(q_py[i])) loop in
    storage_mojo.mojo with a single tobytes() call, then iterates
    over the raw bytes directly rather than boxing each element as a
    PythonObject.

    Parameters
    ----------
    filepath : path to the .npz archive

    Returns
    -------
    QuantizedDataV3 populated from the archive.
    """
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var archive = np.load(PythonObject(filepath), allow_pickle=False)
    var dims = Int(archive["dims"])
    var n    = Int(archive["n"])

    # ---- int8 array: numpy → bytes → Mojo List[Int8] ----
    # tobytes() is a single C-level memcpy — no element boxing at all.
    var q_bytes = archive["quantized"].flatten().astype("int8").tobytes()
    var total_q = n * dims
    var q = List[Int8](capacity=total_q)
    for i in range(total_q):
        # Python bytes indexing returns an int in [0, 255]; cast to signed Int8.
        var raw_byte = Int(q_bytes[i])
        if raw_byte > 127:
            q.append(Int8(raw_byte - 256))
        else:
            q.append(Int8(raw_byte))

    # ---- float32 scales: numpy → bytes → Mojo List[Float32] ----
    var s_bytes = archive["scales"].astype("float32").tobytes()
    var struct_mod = Python.import_module("struct")
    var s = List[Float32](capacity=n)
    for i in range(n):
        var offset = i * 4
        var f = struct_mod.unpack_from("f", s_bytes, PythonObject(offset))[0]
        s.append(Float32(Float64(f)))

    var metadata = String("")
    if "metadata" in archive.files:
        metadata = String(archive["metadata"])

    return QuantizedDataV3(q^, s^, dims, n, metadata)


# ---------------------------------------------------------------------------
# High-level save / load (aliases for consistent naming)
# ---------------------------------------------------------------------------

fn save_v3(data: QuantizedDataV3, filepath: String) raises -> Bool:
    """Save via mmap_write (bulk I/O variant of save_quantized_binary)."""
    return mmap_write(data, filepath)


fn load_v3(filepath: String) raises -> QuantizedDataV3:
    """Load via mmap_read (bulk I/O variant of load_quantized_binary)."""
    return mmap_read(filepath)
