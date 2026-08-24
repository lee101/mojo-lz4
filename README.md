# mojo-lz4

LZ4 block and frame compression implemented in [Mojo](https://www.modular.com/mojo)
and exposed to Python through a small `ctypes` layer. The Python functions keep
the names, arguments, return types, and wire formats of the covered
[`lz4`](https://python-lz4.readthedocs.io/) APIs. Switching the import is enough
for the one-shot APIs:

```python
from mojo_lz4 import block, frame

source = (b"Mojo compresses this data. " * 1000)

raw = block.compress(source)
assert block.decompress(raw) == source

framed = frame.compress(source, content_checksum=True)
assert frame.decompress(framed) == source
```

The output is standard LZ4 data, not a private variant. Tests decode
Mojo-produced blocks and frames with `lz4` 4.4.5, decode upstream-produced data
with Mojo, and cover linked blocks, dictionaries, checksums, and malformed
input.

## Coverage

| upstream area | covered |
| --- | --- |
| `lz4.block` | `compress`, `decompress`, stored sizes, fast acceleration, bytearray results, initial dictionaries |
| `lz4.frame` one-shot | `compress`, `decompress`, `get_frame_info`, all four block sizes, linked-frame decoding, content and block checksums |
| `lz4.frame` incremental compression | `create_compression_context`, `compress_begin`, `compress_chunk`, `compress_flush` |
| file API | `LZ4FrameFile` and `open` in binary and text modes |

There are deliberate limits:

- `mode="high_compression"` and positive frame compression levels use the same
  greedy fast parser as the default mode. They produce correct streams but do
  not provide upstream's HC search or compression ratio.
- Frame compression emits independently compressed blocks even when the frame
  permits linked blocks. The stream is valid, but compression does not gain
  from cross-block history. Frame decompression does support cross-block
  history from upstream encoders.
- Incremental decompression contexts, skippable frames, dictionary-ID frames,
  and the upstream `LZ4FrameCompressor` / `LZ4FrameDecompressor` classes are not
  implemented.
- `LZ4FrameFile` buffers a complete file rather than compressing or
  decompressing it incrementally.

## Install and run

The repository pins its own Mojo nightly and Python environment:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` writes `dist/libmojo-lz4.so`. The activated pixi environment
sets `PYTHONPATH=python`, so the usage example above runs without installing a
wheel.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
x86-64, Python 3.13.14, Mojo `1.1.0.dev2026081105`, and upstream `lz4` 4.4.5.
Each entry is the best of seven timed runs after one warmup. Decode comparisons
use the exact same upstream-produced compressed stream. Relative is upstream
time divided by mojo-lz4 time, so values above 1 mean mojo-lz4 was faster.

| case | mojo-lz4 | upstream lz4 | relative |
| --- | ---: | ---: | ---: |
| block compress, repetitive 8 MiB | 0.43 ms | 0.65 ms | 1.52x |
| block decompress, repetitive 8 MiB | 0.71 ms | 1.24 ms | 1.76x |
| block compress, random 8 MiB | 1.70 ms | 2.57 ms | 1.51x |
| block decompress, random 8 MiB | 0.84 ms | 2.08 ms | 2.47x |
| frame compress, repetitive 8 MiB | 0.47 ms | 0.73 ms | 1.54x |
| frame compress, random 128 MiB | 176.43 ms | 249.35 ms | 1.41x |
| frame decompress, repetitive 8 MiB | 0.79 ms | 7.24 ms | 9.16x |
| frame compress + checksum, repetitive 8 MiB | 3.27 ms | 4.32 ms | 1.32x |
| frame decompress + checksum, repetitive 8 MiB | 3.41 ms | 11.37 ms | 3.33x |

Compression uses an adaptive LZ4 skip search on incompressible input, a 32-bit
candidate check, and SIMD match extension with a scalar remainder. Decode
literal and non-overlapping match copies are SIMD as well. The four independent
xxHash32 accumulators use SIMD with scalar remainder handling. Read-only and
writable contiguous Python buffers cross the FFI by pointer, including NumPy
arrays and block-sized `memoryview` slices. The native kernels write directly
into newly allocated Python `bytes` storage, avoiding a full output copy.

Frames of at least 128 MiB compress independent blocks with up to four Mojo
workers; smaller frames stay serial to avoid launch overhead. Frame assembly
keeps uncompressed source blocks and batch-compression results as zero-copy
views until one final join. There is no GPU path: compression is a
data-dependent hash search, decompression is predominantly copying, and
xxHash32 remains below roughly two arithmetic operations per byte moved, so no
kernel has enough arithmetic intensity to justify transfer and launch costs.

## How it works

`src/lz4.mojo` is one compilation unit containing the LZ4 greedy block encoder,
safe block decoder, and xxHash32 used by frame checksums. The encoder maintains
a 65,536-entry hash table and emits the standard token, literal, 16-bit offset,
and match-length sequence. The decoder validates every input and output bound
before copying, including overlap copies and linked-frame history.

Python owns every allocation. Source and destination buffers cross the C ABI as
integer addresses, are reconstructed in Mojo as
`UnsafePointer[UInt8, AnyOrigin[mut=True]]`, and remain alive for the duration of
one call. Mojo never retains a Python pointer and allocates no heap memory. The
Python frame layer writes and validates the standard frame descriptor, splits
data into blocks, and calls the Mojo kernels for compression, decompression, and
xxHash32.

The frame decoder scans block headers first so it can allocate one contiguous
output buffer. Independent blocks write at the current output address. Linked
blocks use the same base address plus the number of bytes already written,
which exposes up to the preceding 65,535 bytes as legal LZ4 history without
copying a dictionary.

## License

MIT
