"""
High-performance binary storage and loading for quantized vectors.
Mojo implementation for efficient file I/O operations.

The on-disk format is an NPZ archive (numpy's compressed array container)
aligned with the Python layer's ``_STORAGE_FORMAT_NAME = "vectro_npz"`` contract.
Required keys:
  quantized  : int8 array of shape (num_vectors, dims)
  scales     : float32 array of shape (num_vectors,)
  dims       : int64 scalar — vector dimensions
  n          : int64 scalar — number of vectors
  metadata   : stored as a plain UTF-8 string in the "metadata" key
"""
from python import Python, PythonObject


struct QuantizedData:
    """Container for quantized vector data."""
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
        meta: String = ""
    ):
        """Initialize quantized data container.
        Args:
            q: Quantized int8 values.
            s: Scale factors for each vector.
            d: Vector dimension.
            n: Number of vectors.
            meta: Optional metadata string.
        """
        self.quantized = q^
        self.scales = s^
        self.dims = d
        self.num_vectors = n
        self.metadata = meta
    
    fn get_vector(self, index: Int) -> Tuple[List[Int8], Float32]:
        """Get a single quantized vector and its scale.
        Args:
            index: Vector index to retrieve.
        Returns:
            Tuple of (quantized vector, scale factor).
        """
        var start_idx = index * self.dims
        var end_idx = start_idx + self.dims
        
        var vec = List[Int8]()
        for i in range(start_idx, end_idx):
            vec.append(self.quantized[i])
        
        return (vec^, self.scales[index])
    
    fn total_size_bytes(self) -> Int:
        """Calculate total memory usage in bytes.
        Returns:
            Total size in bytes.
        """
        var quant_size = len(self.quantized) * 1  # 1 byte per int8
        var scale_size = self.num_vectors * 4      # 4 bytes per float32
        var meta_size = len(self.metadata)
        return quant_size + scale_size + meta_size
    
    fn compression_ratio(self) -> Float32:
        """Calculate compression ratio vs float32 storage.
        Returns:
            Compression ratio (original_size / compressed_size).
        """
        var original_size = self.num_vectors * self.dims * 4  # float32
        var compressed_size = self.total_size_bytes()
        return Float32(original_size) / Float32(compressed_size)


fn save_quantized_binary(data: QuantizedData, filepath: String) raises -> Bool:
    """Save quantized data to an NPZ archive aligned with the vectro_npz format.

    Args:
        data: QuantizedData to save.
        filepath: Destination path.  A ``.npz`` suffix is appended by numpy
            when it is not already present.
    Returns:
        True on success.

    Raises:
        Any exception propagated from numpy on I/O failure.
    """
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    # --- Build Python lists from Mojo containers ---
    var py_q = builtins.list()
    for i in range(len(data.quantized)):
        _ = py_q.append(PythonObject(Int(data.quantized[i])))

    var py_s = builtins.list()
    for i in range(data.num_vectors):
        _ = py_s.append(PythonObject(Float64(data.scales[i])))

    # --- Convert to numpy arrays ---
    var q_np = np.array(py_q, dtype="int8").reshape(
        PythonObject(data.num_vectors), PythonObject(data.dims)
    )
    var s_np = np.array(py_s, dtype="float32")

    # --- Write compressed archive ---
    np.savez_compressed(
        PythonObject(filepath),
        quantized=q_np,
        scales=s_np,
        dims=np.array(PythonObject(data.dims), dtype="int64"),
        n=np.array(PythonObject(data.num_vectors), dtype="int64"),
        metadata=PythonObject(data.metadata),
    )

    print("Saved quantized data to:", filepath)
    print("  Vectors:", data.num_vectors, "  Dims:", data.dims)
    print("  Compression ratio:", data.compression_ratio(), "x")
    return True


fn load_quantized_binary(filepath: String) raises -> QuantizedData:
    """Load quantized data from an NPZ archive written by :func:`save_quantized_binary`.

    Args:
        filepath: Path to the ``.npz`` archive (with or without the suffix).
    Returns:
        Populated :struct:`QuantizedData`.

    Raises:
        Any exception propagated from numpy on I/O failure or format mismatch.
    """
    var np = Python.import_module("numpy")

    # numpy appends .npz automatically; accept either form
    var archive = np.load(PythonObject(filepath), allow_pickle=False)

    var dims = Int(archive["dims"])
    var n = Int(archive["n"])

    # Flatten to 1-D Python lists for iteration
    var q_py = archive["quantized"].flatten().tolist()
    var s_py = archive["scales"].flatten().tolist()

    var q = List[Int8](capacity=n * dims)
    for i in range(n * dims):
        q.append(Int8(Int(q_py[i])))

    var s = List[Float32](capacity=n)
    for i in range(n):
        s.append(Float32(Float64(s_py[i])))

    var metadata = String("")
    if "metadata" in archive.files:
        metadata = String(archive["metadata"])

    print("Loaded quantized data from:", filepath)
    print("  Vectors:", n, "  Dims:", dims)

    return QuantizedData(q^, s^, dims, n, metadata)


struct StorageStats:
    """Statistics about stored quantized data."""
    var original_size_mb: Float32
    var compressed_size_mb: Float32
    var compression_ratio: Float32
    var num_vectors: Int
    var vector_dim: Int
    var avg_scale: Float32
    var min_scale: Float32
    var max_scale: Float32
    
    fn __init__(
        out self,
        orig: Float32,
        comp: Float32,
        ratio: Float32,
        n: Int,
        d: Int,
        avg: Float32,
        min_s: Float32,
        max_s: Float32
    ):
        """Initialize storage statistics.
        Args:
            orig: Original size in MB.
            comp: Compressed size in MB.
            ratio: Compression ratio.
            n: Number of vectors.
            d: Vector dimension.
            avg: Average scale factor.
            min_s: Minimum scale factor.
            max_s: Maximum scale factor.
        """
        self.original_size_mb = orig
        self.compressed_size_mb = comp
        self.compression_ratio = ratio
        self.num_vectors = n
        self.vector_dim = d
        self.avg_scale = avg
        self.min_scale = min_s
        self.max_scale = max_s
    
    fn print_stats(self):
        """Print formatted statistics."""
        print("=" * 70)
        print("Storage Statistics")
        print("=" * 70)
        print("Vectors:", self.num_vectors)
        print("Dimensions:", self.vector_dim)
        print("Original size:", self.original_size_mb, "MB")
        print("Compressed size:", self.compressed_size_mb, "MB")
        print("Compression ratio:", self.compression_ratio, "x")
        print("Space saved:", 
              Int((1.0 - 1.0/self.compression_ratio) * 100), "%")
        print("\nScale Factors:")
        print("  Average:", self.avg_scale)
        print("  Min:", self.min_scale)
        print("  Max:", self.max_scale)
        print("=" * 70)


fn calculate_storage_stats(data: QuantizedData) -> StorageStats:
    """Calculate comprehensive storage statistics.
    Args:
        data: QuantizedData to analyze.
    Returns:
        StorageStats with calculated metrics.
    """
    var original_bytes = data.num_vectors * data.dims * 4
    var compressed_bytes = data.total_size_bytes()
    
    var original_mb = Float32(original_bytes) / (1024.0 * 1024.0)
    var compressed_mb = Float32(compressed_bytes) / (1024.0 * 1024.0)
    var ratio = data.compression_ratio()
    
    # Calculate scale statistics
    var sum_scales: Float32 = 0.0
    var min_scale: Float32 = 1e10
    var max_scale: Float32 = 0.0
    
    for i in range(data.num_vectors):
        var scale = data.scales[i]
        sum_scales += scale
        if scale < min_scale:
            min_scale = scale
        if scale > max_scale:
            max_scale = scale
    
    var avg_scale = sum_scales / Float32(data.num_vectors)
    
    return StorageStats(
        original_mb,
        compressed_mb,
        ratio,
        data.num_vectors,
        data.dims,
        avg_scale,
        min_scale,
        max_scale
    )


fn main() raises:
    """Test storage functionality."""
    print("=" * 70)
    print("Vectro Storage Module (Mojo)")
    print("=" * 70)
    
    # Create sample quantized data
    var sample_q = List[Int8]()
    for i in range(100):
        sample_q.append(Int8(i % 127))
    
    var sample_s = List[Float32]()
    for i in range(10):
        sample_s.append(Float32(i) * 0.01 + 0.05)
    
    var data = QuantizedData(sample_q^, sample_s^, 10, 10, "test_data")
    
    print("\nCreated sample data:")
    print("  Vectors:", data.num_vectors)
    print("  Dimensions:", data.dims)
    print("  Size:", data.total_size_bytes(), "bytes")
    print("  Compression:", data.compression_ratio(), "x")
    
    # Calculate and display statistics
    print("\nCalculating storage statistics...")
    var stats = calculate_storage_stats(data)
    stats.print_stats()
    
    # Test saving
    print("\nTesting save functionality...")
    var success = save_quantized_binary(data, "test_output.vqnt")
    if success:
        print("✓ Save operation successful")
    
    print("\n" + "=" * 70)
    print("Storage module ready for high-performance I/O!")
    print("=" * 70)
