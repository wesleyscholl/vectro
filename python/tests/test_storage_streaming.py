import numpy as np
from python.storage import VectroStreamingWriter, read_header, read_chunk_blob, peek
import os


def test_streaming_writer_and_peek(tmp_path):
    out = str(tmp_path / 'test_stream.vtrb')
    writer = VectroStreamingWriter(out, codec='zstd:3')
    # create small codebooks and set them
    cb = np.random.standard_normal((2, 8, 4)).astype(np.float32)
    writer.set_codebooks(cb)
    # write two chunks
    q1 = (np.arange(16, dtype=np.int8)).tobytes()
    s1 = (np.arange(4, dtype=np.float32)).tobytes()
    writer.write_chunk(0, 4, q_bytes=q1, scales_bytes=s1, backend='int8')
    q2 = (np.arange(16, 32, dtype=np.int8)).tobytes()
    s2 = (np.arange(4, 8, dtype=np.float32)).tobytes()
    writer.write_chunk(4, 8, q_bytes=q2, scales_bytes=s2, backend='int8')
    writer.finalize()
    # peek should not raise
    peek(out)
    h = read_header(out)
    assert 'chunk_index' in h
    assert h['codebooks']['m'] == 2
    # read first chunk blob
    meta = h['chunk_index'][0]
    blob = read_chunk_blob(out, meta)
    assert len(blob) == meta['length']
