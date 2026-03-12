"""
Artifact migration and inspection tooling (Mojo implementation).

Provides Mojo-native routines for inspecting, validating, and reporting on
Vectro compressed artifacts stored in the NPZ/VQZ formats.  The actual file
I/O for the legacy NPZ format is handled by the Python layer
(python/migration.py); this Mojo module provides the structural logic and
format constants that both layers share.

Breaking changes from v1.x → v2.0
-----------------------------------
* storage_format_version field added (defaults to 1 when absent)
* precision_mode field added (defaults to "int8" for legacy files)
* group_size field added (defaults to 0)
* metadata_json field added (provenance, creation time)

Public API
----------
ArtifactInfo            — struct returned by validate_header()
VqzHeader               — 64-byte VQZ binary header struct
validate_header(bytes)  — parse and validate a raw VQZ header
format_artifact_report(info) -> String   — human-readable report
migration_summary(src_version, dst_version) -> String
"""


# ─────────────────────────────────────────────────────────────────────────────
# Format constants
# ─────────────────────────────────────────────────────────────────────────────

alias VQZ_MAGIC:          UInt32 = 0x56515A33   # "VQZ3" in little-endian
alias VQZ_HEADER_SIZE:    Int    = 64
alias CURRENT_VQZ_VERSION: UInt16 = 3
alias CURRENT_NPZ_VERSION: Int    = 2

# Compression flags (byte 8 of header)
alias COMP_FLAG_ZSTD:  UInt8 = 0x01
alias COMP_FLAG_ZLIB:  UInt8 = 0x02
alias COMP_FLAG_NONE:  UInt8 = 0x00

# Precision / quantization mode codes
alias MODE_INT8:   UInt8 = 0x01
alias MODE_NF4:    UInt8 = 0x02
alias MODE_PQ:     UInt8 = 0x03
alias MODE_BINARY: UInt8 = 0x04
alias MODE_RQ:     UInt8 = 0x05


# ─────────────────────────────────────────────────────────────────────────────
# Header struct
# ─────────────────────────────────────────────────────────────────────────────

struct VqzHeader:
    """64-byte binary header for the VQZ container format.

    Layout (all little-endian)
    --------------------------
    Offset  Size  Field
       0      4   magic          (0x56515A33 == "VQZ3")
       4      2   version        (current = 3)
       6      2   reserved_0     (must be 0)
       8      1   comp_flag      (0=none, 1=zstd, 2=zlib)
       9      1   precision_mode (1=int8, 2=nf4, 3=pq, 4=binary, 5=rq)
      10      2   reserved_1
      12      4   n_vectors      (uint32, number of quantised vectors)
      16      4   dims           (uint32, original embedding dimensionality)
      20      4   n_subspaces    (uint32, PQ sub-spaces; 0 for int8/nf4/binary)
      24      4   n_centroids    (uint32, PQ centroids K; 0 otherwise)
      28      4   body_offset    (uint32, byte offset of body in file)
      32      4   body_bytes     (uint32, compressed body size in bytes)
      36      4   raw_body_bytes (uint32, uncompressed body size)
      40      8   reserved_2     (padding to 48 bytes)
      48     16   reserved_3     (padding to 64 bytes)
    """

    var magic:          UInt32
    var version:        UInt16
    var reserved_0:     UInt16
    var comp_flag:      UInt8
    var precision_mode: UInt8
    var reserved_1:     UInt16
    var n_vectors:      UInt32
    var dims:           UInt32
    var n_subspaces:    UInt32
    var n_centroids:    UInt32
    var body_offset:    UInt32
    var body_bytes:     UInt32
    var raw_body_bytes: UInt32

    fn __init__(out self):
        """Initialise a zeroed VQZ header."""
        self.magic          = 0
        self.version        = 0
        self.reserved_0     = 0
        self.comp_flag      = 0
        self.precision_mode = 0
        self.reserved_1     = 0
        self.n_vectors      = 0
        self.dims           = 0
        self.n_subspaces    = 0
        self.n_centroids    = 0
        self.body_offset    = 0
        self.body_bytes     = 0
        self.raw_body_bytes = 0

    fn is_valid_magic(self) -> Bool:
        """Return True when the magic matches the VQZ3 signature.

        Returns:
            True if magic == VQZ_MAGIC.
        """
        return self.magic == VQZ_MAGIC

    fn is_supported_version(self) -> Bool:
        """Return True when version <= CURRENT_VQZ_VERSION.

        Returns:
            True if version is within the supported range.
        """
        return self.version <= CURRENT_VQZ_VERSION

    fn compression_label(self) -> String:
        """Return a human-readable compression label.

        Returns:
            "zstd", "zlib", or "none".
        """
        if self.comp_flag == COMP_FLAG_ZSTD: return "zstd"
        if self.comp_flag == COMP_FLAG_ZLIB: return "zlib"
        return "none"

    fn precision_label(self) -> String:
        """Return a human-readable precision/mode label.

        Returns:
            "int8", "nf4", "pq", "binary", "rq", or "unknown".
        """
        if self.precision_mode == MODE_INT8:   return "int8"
        if self.precision_mode == MODE_NF4:    return "nf4"
        if self.precision_mode == MODE_PQ:     return "pq"
        if self.precision_mode == MODE_BINARY: return "binary"
        if self.precision_mode == MODE_RQ:     return "rq"
        return "unknown"

    fn expected_raw_body_bytes(self) -> UInt32:
        """Compute expected raw body size based on header fields.

        INT8 / NF4 / Binary: n_vectors × dims × 1 byte quantised data
                             + n_vectors × 4 bytes scales.
        PQ / RQ:             n_vectors × n_subspaces bytes codes.
        Binary:              n_vectors × ceil(dims / 8) bytes.

        Returns:
            Expected uncompressed body size in bytes, or 0 if unknown mode.
        """
        var n = UInt32(self.n_vectors)
        var d = UInt32(self.dims)
        var M = UInt32(self.n_subspaces)
        if self.precision_mode == MODE_INT8:
            return n * d + n * 4
        if self.precision_mode == MODE_NF4:
            return n * ((d + 1) // 2) + n * 4
        if self.precision_mode == MODE_BINARY:
            return n * ((d + 7) // 8)
        if self.precision_mode == MODE_PQ or self.precision_mode == MODE_RQ:
            return n * M
        return 0

    fn print(self):
        """Print a formatted header summary."""
        print("VQZ Header")
        print("  magic          :", String(self.magic))
        print("  version        :", String(Int(self.version)))
        print("  compression    :", self.compression_label())
        print("  precision      :", self.precision_label())
        print("  n_vectors      :", String(Int(self.n_vectors)))
        print("  dims           :", String(Int(self.dims)))
        print("  n_subspaces    :", String(Int(self.n_subspaces)))
        print("  body_offset    :", String(Int(self.body_offset)))
        print("  body_bytes     :", String(Int(self.body_bytes)))
        print("  raw_body_bytes :", String(Int(self.raw_body_bytes)))


# ─────────────────────────────────────────────────────────────────────────────
# Validation result
# ─────────────────────────────────────────────────────────────────────────────

struct ValidationResult:
    """Result of a header or artifact validation check."""

    var valid:       Bool
    var error_count: Int
    var warnings:    Int

    fn __init__(out self, valid: Bool, error_count: Int, warnings: Int):
        """Initialise a ValidationResult.

        Args:
            valid:       True when no hard errors were found.
            error_count: Number of hard errors detected.
            warnings:    Number of non-fatal warnings detected.
        """
        self.valid       = valid
        self.error_count = error_count
        self.warnings    = warnings

    fn print(self):
        """Print a one-line validation summary."""
        if self.valid:
            print("Validation: PASSED  (warnings=" + String(self.warnings) + ")")
        else:
            print("Validation: FAILED  (errors=" + String(self.error_count)
                  + " warnings=" + String(self.warnings) + ")")


# ─────────────────────────────────────────────────────────────────────────────
# Artifact info
# ─────────────────────────────────────────────────────────────────────────────

struct ArtifactInfo:
    """Parsed metadata for a Vectro compressed artifact."""

    var format_version:    Int
    var artifact_type:     String    # "single" or "batch"
    var n_vectors:         Int
    var vector_dim:        Int
    var precision_mode:    String
    var group_size:        Int
    var compression_ratio: Float32
    var needs_upgrade:     Bool
    var file_size_bytes:   Int

    fn __init__(
        out self,
        format_version:    Int,
        artifact_type:     String,
        n_vectors:         Int,
        vector_dim:        Int,
        precision_mode:    String,
        group_size:        Int,
        compression_ratio: Float32,
        needs_upgrade:     Bool,
        file_size_bytes:   Int,
    ):
        """Initialise an ArtifactInfo.

        Args:
            format_version:    Storage format version (1 or 2).
            artifact_type:     "single" for one vector, "batch" for many.
            n_vectors:         Number of compressed vectors.
            vector_dim:        Original embedding dimensionality.
            precision_mode:    Quantization mode name (e.g. "int8").
            group_size:        Quantization group size (0 = per-vector).
            compression_ratio: Achieved compression ratio vs float32.
            needs_upgrade:     True when format_version < CURRENT_NPZ_VERSION.
            file_size_bytes:   File size in bytes.
        """
        self.format_version    = format_version
        self.artifact_type     = artifact_type
        self.n_vectors         = n_vectors
        self.vector_dim        = vector_dim
        self.precision_mode    = precision_mode
        self.group_size        = group_size
        self.compression_ratio = compression_ratio
        self.needs_upgrade     = needs_upgrade
        self.file_size_bytes   = file_size_bytes

    fn print(self):
        """Print a human-readable artifact report."""
        var upg = "[NEEDS UPGRADE]" if self.needs_upgrade else "[current]"
        print("Artifact type  : " + self.artifact_type)
        print("Format version : v" + String(self.format_version) + "  " + upg)
        print("Vectors        : " + String(self.n_vectors)
              + " x " + String(self.vector_dim))
        print("Precision mode : " + self.precision_mode)
        print("Group size     : " + String(self.group_size))
        print("Compression    : " + String(self.compression_ratio) + "x")
        print("File size      : " + String(self.file_size_bytes) + " bytes")


# ─────────────────────────────────────────────────────────────────────────────
# Header validation
# ─────────────────────────────────────────────────────────────────────────────

fn validate_vqz_header(hdr: VqzHeader) -> ValidationResult:
    """Validate a parsed VQZ header for structural correctness.

    Checks:
    1. Magic bytes match VQZ3.
    2. Version is within the supported range.
    3. n_vectors > 0 and dims > 0.
    4. body_offset >= VQZ_HEADER_SIZE.
    5. body_bytes > 0 when comp_flag is set.
    6. raw_body_bytes matches expected size (warning if mismatch).

    Args:
        hdr: A parsed VqzHeader struct.
    Returns:
        ValidationResult with error_count and warnings populated.
    """
    var errors   = 0
    var warnings = 0

    if not hdr.is_valid_magic():
        errors += 1
        print("Error: invalid magic bytes (expected VQZ3)")

    if not hdr.is_supported_version():
        errors += 1
        print("Error: unsupported VQZ version " + String(Int(hdr.version))
              + " (max " + String(Int(CURRENT_VQZ_VERSION)) + ")")

    if hdr.n_vectors == 0:
        errors += 1
        print("Error: n_vectors is 0")

    if hdr.dims == 0:
        errors += 1
        print("Error: dims is 0")

    if Int(hdr.body_offset) < VQZ_HEADER_SIZE:
        errors += 1
        print("Error: body_offset " + String(Int(hdr.body_offset))
              + " is less than header size " + String(VQZ_HEADER_SIZE))

    if hdr.comp_flag != COMP_FLAG_NONE and hdr.body_bytes == 0:
        warnings += 1
        print("Warning: body_bytes is 0 but comp_flag is set")

    var expected = hdr.expected_raw_body_bytes()
    if expected > 0 and hdr.raw_body_bytes != expected:
        warnings += 1
        print("Warning: raw_body_bytes=" + String(Int(hdr.raw_body_bytes))
              + " expected=" + String(Int(expected)))

    return ValidationResult(errors == 0, errors, warnings)


fn make_default_vqz_header(
    n_vectors:      UInt32,
    dims:           UInt32,
    precision_mode: UInt8 = MODE_INT8,
    comp_flag:      UInt8 = COMP_FLAG_ZSTD,
) -> VqzHeader:
    """Construct a valid VQZ header with sensible defaults.

    All reserved fields are zeroed.  body_offset is set to VQZ_HEADER_SIZE.
    raw_body_bytes and body_bytes must be filled in by the caller after
    the body is compressed.

    Args:
        n_vectors:      Number of vectors to be stored.
        dims:           Embedding dimensionality.
        precision_mode: Quantization mode flag (use MODE_* constants).
        comp_flag:      Compression algorithm flag (use COMP_FLAG_* constants).
    Returns:
        A populated VqzHeader struct ready for writing.
    """
    var hdr = VqzHeader()
    hdr.magic          = VQZ_MAGIC
    hdr.version        = CURRENT_VQZ_VERSION
    hdr.comp_flag      = comp_flag
    hdr.precision_mode = precision_mode
    hdr.n_vectors      = n_vectors
    hdr.dims           = dims
    hdr.body_offset    = UInt32(VQZ_HEADER_SIZE)
    return hdr


# ─────────────────────────────────────────────────────────────────────────────
# Migration utilities
# ─────────────────────────────────────────────────────────────────────────────

fn migration_summary(src_version: Int, dst_version: Int) -> String:
    """Build a one-line human-readable migration summary string.

    Args:
        src_version: Source format version.
        dst_version: Destination format version.
    Returns:
        A string like "Migrated v1 → v2  (+precision_mode, +group_size, +metadata_json)".
    """
    if src_version >= dst_version:
        return "No migration needed: already at v" + String(dst_version)

    var changes = ""
    if src_version < 2:
        changes = "+precision_mode, +group_size, +metadata_json"
    return ("Migrated v" + String(src_version)
            + " -> v" + String(dst_version)
            + "  (" + changes + ")")


fn needs_upgrade(format_version: Int) -> Bool:
    """Return True when format_version is below the current standard.

    Args:
        format_version: Parsed storage_format_version from an artifact.
    Returns:
        True if an upgrade is needed.
    """
    return format_version < CURRENT_NPZ_VERSION


fn format_compression_ratio(ratio: Float32) -> String:
    """Return a formatted compression ratio string, e.g. "8.00x".

    Args:
        ratio: Compression ratio value.
    Returns:
        String in the form "<ratio>x".
    """
    return String(ratio) + "x"


fn describe_precision_mode(mode: String) -> String:
    """Return a human-readable description of a precision mode name.

    Args:
        mode: Mode name string (e.g. "int8", "nf4").
    Returns:
        A descriptive string.
    """
    if mode == "int8":   return "INT8 symmetric (4x compression)"
    if mode == "nf4":    return "NF4 normal-float 4-bit (8x compression)"
    if mode == "binary": return "Binary 1-bit sign (32x compression)"
    if mode == "pq":     return "Product Quantization (variable compression)"
    if mode == "rq":     return "Residual Quantization (variable compression)"
    return "Unknown mode: " + mode


fn print_migration_plan(src_version: Int):
    """Print a step-by-step migration plan from src_version to current.

    Args:
        src_version: The artifact's current format version.
    """
    if not needs_upgrade(src_version):
        print("No migration required — artifact is at current version.")
        return

    print("Migration plan: v" + String(src_version)
          + " -> v" + String(CURRENT_NPZ_VERSION))

    if src_version < 2:
        print("  Step 1: Add 'precision_mode' field (default: 'int8')")
        print("  Step 2: Add 'group_size' field (default: 0)")
        print("  Step 3: Add 'storage_format' field (default: 'vectro_npz')")
        print("  Step 4: Bump 'storage_format_version' to 2")
        print("  Step 5: Create 'metadata_json' with migration provenance record")


fn main():
    """Demo: create a default header and validate it."""
    var hdr = make_default_vqz_header(
        n_vectors      = 1000,
        dims           = 768,
        precision_mode = MODE_INT8,
        comp_flag      = COMP_FLAG_ZSTD,
    )
    # Simulate body size for an INT8 artifact
    hdr.raw_body_bytes = hdr.expected_raw_body_bytes()
    hdr.body_bytes     = hdr.raw_body_bytes  # pretend compression ratio = 1.0

    hdr.print()
    print("")

    var result = validate_vqz_header(hdr)
    result.print()
    print("")

    print(migration_summary(1, CURRENT_NPZ_VERSION))
    print_migration_plan(1)
    print("")

    print(describe_precision_mode("nf4"))
    print(format_compression_ratio(Float32(8.0)))
