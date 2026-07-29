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
x86-64, Python 3.13.14, Mojo `1.0.0b3.dev2026072406`, and upstream `lz4` 4.4.5.
Each entry is the best of seven timed runs after one warmup. Decode comparisons
use the exact same upstream-produced compressed stream. Relative is upstream
time divided by mojo-lz4 time, so values above 1 mean mojo-lz4 was faster.

| case | mojo-lz4 | upstream lz4 | relative |
| --- | ---: | ---: | ---: |
| block compress, repetitive 8 MiB | 0.43 ms | 0.64 ms | 1.50x |
| block decompress, repetitive 8 MiB | 0.72 ms | 1.31 ms | 1.83x |
| block compress, random 8 MiB | 1.94 ms | 2.42 ms | 1.25x |
| block decompress, random 8 MiB | 1.85 ms | 2.44 ms | 1.32x |
| frame compress, repetitive 8 MiB | 0.47 ms | 0.72 ms | 1.52x |
| frame decompress, repetitive 8 MiB | 0.66 ms | 9.64 ms | 14.68x |

Compression uses an adaptive LZ4 skip search on incompressible input, a 32-bit
candidate check, and SIMD match extension with a scalar remainder. Decode
literal and non-overlapping match copies are SIMD as well. Read-only and
writable contiguous Python buffers cross the FFI by pointer, including NumPy
arrays and block-sized `memoryview` slices. The native kernels write directly
into newly allocated Python `bytes` storage, avoiding a full output copy.

No parallel or GPU path is included. Compression is a serial, data-dependent
hash search, and decompression is predominantly copying.

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
