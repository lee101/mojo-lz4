import io
import os
import struct

import lz4.block as upstream_block
import lz4.frame as upstream_frame
import numpy as np
import pytest

from mojo_lz4 import block, frame
from mojo_lz4._lib import buffer_address
from mojo_lz4._lib import decompress_into


def payload(size=300_000):
    seed = bytes((i * 17 + i // 7) & 0xFF for i in range(4096))
    return (b"mojo-lz4 parity data\n" + seed) * (size // (len(seed) + 21) + 1)


@pytest.mark.parametrize("size", [0, 1, 3, 4, 15, 16, 255, 4096, 100_000])
def test_block_roundtrip_sizes(size):
    source = payload(size)[:size]
    assert block.decompress(block.compress(source)) == source


def test_block_published_literal_sequence():
    assert block.decompress(b"\x40Wiki", uncompressed_size=4) == b"Wiki"


@pytest.mark.parametrize("store_size", [False, True])
@pytest.mark.parametrize("source", [b"", b"small input", b"a" * 100_000, payload()])
def test_mojo_blocks_decode_upstream(source, store_size):
    encoded = block.compress(source, store_size=store_size)
    size = -1 if store_size else len(source)
    assert upstream_block.decompress(encoded, uncompressed_size=size) == source


@pytest.mark.parametrize("store_size", [False, True])
def test_upstream_blocks_decode_mojo(store_size):
    source = payload()
    encoded = upstream_block.compress(source, store_size=store_size)
    size = -1 if store_size else len(source)
    assert block.decompress(encoded, uncompressed_size=size) == source


def test_block_fast_and_high_compression_modes_are_compatible():
    source = payload()
    fast = block.compress(source, mode="fast", acceleration=4)
    high = block.compress(source, mode="high_compression", compression=12)
    assert upstream_block.decompress(fast) == source
    assert upstream_block.decompress(high) == source


def test_block_dictionary_cross_compatibility():
    dictionary = payload(80_000)[-65_535:]
    source = dictionary[-6000:] * 3
    mojo_encoded = block.compress(source, store_size=False, dict=dictionary)
    assert (
        upstream_block.decompress(
            mojo_encoded, uncompressed_size=len(source), dict=dictionary
        )
        == source
    )
    upstream_encoded = upstream_block.compress(
        source, store_size=False, dict=dictionary
    )
    assert (
        block.decompress(
            upstream_encoded, uncompressed_size=len(source), dict=dictionary
        )
        == source
    )


def test_block_buffer_inputs_and_bytearray_output():
    source = bytearray(payload(10_000))
    encoded = block.compress(memoryview(source), return_bytearray=True)
    decoded = block.decompress(encoded, return_bytearray=True)
    assert isinstance(encoded, bytearray)
    assert isinstance(decoded, bytearray)
    assert decoded == source


@pytest.mark.parametrize("tail", [0, 1, 31, 32, 33])
def test_simd_match_extension_tail(tail):
    pattern = bytes(range(97))
    source = pattern + pattern + pattern[:tail] + b"final literals"
    encoded = block.compress(source, store_size=False)
    assert upstream_block.decompress(encoded, uncompressed_size=len(source)) == source


@pytest.mark.parametrize("writable", [False, True])
def test_numpy_input_crosses_ffi_without_materializing_bytes(writable):
    source = np.arange(300_000, dtype=np.uint8)
    source.flags.writeable = writable
    expected = source.tobytes()
    address, size, keepalive = buffer_address(source)
    assert address == source.ctypes.data
    assert size == source.nbytes
    encoded = block.compress(source, store_size=False)
    _ = keepalive
    assert block.decompress(encoded, uncompressed_size=source.nbytes) == expected


def test_multibyte_numpy_input_uses_raw_bytes_without_dtype_narrowing():
    source = np.arange(4096, dtype=np.uint32).reshape(64, 64)
    encoded = block.compress(source)
    assert block.decompress(encoded) == source.tobytes()


def test_noncontiguous_buffer_is_rejected_before_ffi():
    source = np.arange(100, dtype=np.uint8)[::2]
    with pytest.raises(BufferError, match="contiguous"):
        block.compress(source)


def test_decompress_into_rejects_out_of_bounds_pointer_arithmetic():
    with pytest.raises(ValueError, match="outside the input"):
        decompress_into(b"abc", 2, 2, bytearray(10), 0, False)
    with pytest.raises(ValueError, match="outside the destination"):
        decompress_into(b"abc", 0, 3, bytearray(2), 3, False)


def test_block_too_small_output_raises():
    encoded = block.compress(payload(), store_size=False)
    with pytest.raises(block.LZ4BlockError):
        block.decompress(encoded, uncompressed_size=10)


def test_block_invalid_input_raises():
    with pytest.raises(block.LZ4BlockError):
        block.decompress(struct.pack("<I", 10) + b"\x00\x00\x00")


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"store_size": False},
        {"content_checksum": True},
        {"block_checksum": True},
        {"block_size": frame.BLOCKSIZE_MAX256KB},
        {
            "block_size": frame.BLOCKSIZE_MAX64KB,
            "block_linked": False,
            "content_checksum": True,
            "block_checksum": True,
        },
    ],
)
def test_mojo_frames_decode_upstream(options):
    source = payload()
    encoded = frame.compress(source, **options)
    assert upstream_frame.decompress(encoded) == source


@pytest.mark.parametrize(
    "block_size",
    [
        frame.BLOCKSIZE_MAX64KB,
        frame.BLOCKSIZE_MAX256KB,
        frame.BLOCKSIZE_MAX1MB,
        frame.BLOCKSIZE_MAX4MB,
    ],
)
def test_frame_compression_supports_every_claimed_block_size(block_size):
    source = payload(5_000_000)[:5_000_000]
    encoded = frame.compress(source, block_size=block_size)
    assert upstream_frame.decompress(encoded) == source


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"store_size": False},
        {"content_checksum": True, "block_checksum": True},
        {"block_size": upstream_frame.BLOCKSIZE_MAX64KB, "block_linked": True},
        {"block_size": upstream_frame.BLOCKSIZE_MAX1MB, "block_linked": False},
    ],
)
def test_upstream_frames_decode_mojo(options):
    source = payload()
    encoded = upstream_frame.compress(source, **options)
    assert frame.decompress(encoded) == source


def test_frame_info_matches_requested_features():
    encoded = frame.compress(
        payload(),
        block_size=frame.BLOCKSIZE_MAX256KB,
        block_linked=False,
        content_checksum=True,
        block_checksum=True,
    )
    info = frame.get_frame_info(encoded)
    assert info == {
        "block_size": 256 * 1024,
        "block_size_id": frame.BLOCKSIZE_MAX256KB,
        "block_linked": False,
        "content_checksum": True,
        "block_checksum": True,
        "skippable": False,
        "content_size": len(payload()),
    }


def test_frame_return_bytearray_and_bytes_read():
    source = payload(10_000)
    encoded = frame.compress(source, return_bytearray=True)
    result, consumed = frame.decompress(
        bytes(encoded) + b"trailing", return_bytearray=True, return_bytes_read=True
    )
    assert isinstance(encoded, bytearray)
    assert isinstance(result, bytearray)
    assert result == source
    assert consumed == len(encoded)


def test_frame_header_corruption_is_detected():
    encoded = bytearray(frame.compress(payload()))
    encoded[6] ^= 1
    with pytest.raises(frame.LZ4FrameError, match="header checksum"):
        frame.decompress(encoded)


def test_frame_block_checksum_corruption_is_detected():
    encoded = bytearray(frame.compress(payload(), block_checksum=True))
    header_size = 15
    block_size = struct.unpack_from("<I", encoded, header_size)[0] & 0x7FFFFFFF
    checksum_position = header_size + 4 + block_size
    encoded[checksum_position] ^= 1
    with pytest.raises(frame.LZ4FrameError, match="block checksum"):
        frame.decompress(encoded)


def test_frame_content_checksum_corruption_is_detected():
    encoded = bytearray(frame.compress(payload(), content_checksum=True))
    encoded[-1] ^= 1
    with pytest.raises(frame.LZ4FrameError, match="content checksum"):
        frame.decompress(encoded)


def test_upstream_linked_blocks_using_history_decode_mojo():
    first = os.urandom(64 * 1024)
    source = first + first + first[:30_000]
    encoded = upstream_frame.compress(
        source,
        block_size=upstream_frame.BLOCKSIZE_MAX64KB,
        block_linked=True,
        store_size=False,
    )
    assert frame.get_frame_info(encoded)["block_linked"]
    assert frame.decompress(encoded) == source


def test_incremental_compression_decodes_upstream():
    source = payload()
    context = frame.create_compression_context()
    parts = [
        frame.compress_begin(
            context,
            source_size=len(source),
            block_size=frame.BLOCKSIZE_MAX64KB,
            block_linked=True,
            content_checksum=True,
            block_checksum=True,
        )
    ]
    parts.extend(
        frame.compress_chunk(context, source[start : start + 33_333])
        for start in range(0, len(source), 33_333)
    )
    parts.append(frame.compress_flush(context))
    assert upstream_frame.decompress(b"".join(parts)) == source


def test_incremental_compression_rejects_declared_size_mismatch():
    context = frame.create_compression_context()
    frame.compress_begin(context, source_size=4)
    frame.compress_chunk(context, b"abc")
    with pytest.raises(frame.LZ4FrameError, match="declared source_size"):
        frame.compress_flush(context)

    context = frame.create_compression_context()
    frame.compress_begin(context, source_size=2)
    with pytest.raises(frame.LZ4FrameError, match="declared source_size"):
        frame.compress_chunk(context, b"abc")


def test_binary_file_roundtrip(tmp_path):
    path = tmp_path / "data.lz4"
    source = payload()
    with frame.open(path, "wb", content_checksum=True) as stream:
        assert stream.write(source) == len(source)
    assert upstream_frame.decompress(path.read_bytes()) == source
    with frame.open(path, "rb") as stream:
        assert stream.read() == source


def test_file_object_roundtrip():
    backing = io.BytesIO()
    with frame.LZ4FrameFile(backing, "w") as stream:
        stream.write(b"abc")
        stream.write(payload(1000))
    backing.seek(0)
    with frame.LZ4FrameFile(backing, "r") as stream:
        assert stream.read() == b"abc" + payload(1000)


def test_text_file_roundtrip(tmp_path):
    path = tmp_path / "text.lz4"
    text = "Mojo and LZ4\n" * 100
    with frame.open(path, "wt", encoding="utf-8") as stream:
        stream.write(text)
    with frame.open(path, "rt", encoding="utf-8") as stream:
        assert stream.read() == text
