"""LZ4 frame format built around the Mojo block codec."""

from __future__ import annotations

import builtins
import ctypes
import io
import struct
from dataclasses import dataclass, field

from ._lib import (
    LibraryError,
    as_buffer,
    as_bytes,
    buffer_address,
    compress_raw,
    decompress_into,
    lib,
    writable_bytes,
    xxh32,
)

MAGIC = b"\x04\x22\x4d\x18"
BLOCKSIZE_DEFAULT = 0
BLOCKSIZE_MAX64KB = 4
BLOCKSIZE_MAX256KB = 5
BLOCKSIZE_MAX1MB = 6
BLOCKSIZE_MAX4MB = 7
COMPRESSIONLEVEL_MIN = 0
COMPRESSIONLEVEL_MINHC = 3
COMPRESSIONLEVEL_MAX = 16

_BLOCK_SIZES = {
    BLOCKSIZE_MAX64KB: 64 * 1024,
    BLOCKSIZE_MAX256KB: 256 * 1024,
    BLOCKSIZE_MAX1MB: 1024 * 1024,
    BLOCKSIZE_MAX4MB: 4 * 1024 * 1024,
}


class LZ4FrameError(RuntimeError):
    pass


def _block_id(block_size: int) -> int:
    value = int(block_size)
    if value == BLOCKSIZE_DEFAULT:
        return BLOCKSIZE_MAX64KB
    if value not in _BLOCK_SIZES:
        raise ValueError("invalid block_size")
    return value


def _header(
    content_size: int | None,
    block_size: int,
    block_linked: bool,
    content_checksum: bool,
    block_checksum: bool,
) -> bytes:
    block_id = _block_id(block_size)
    flg = 0x40
    if not block_linked:
        flg |= 0x20
    if block_checksum:
        flg |= 0x10
    if content_size is not None:
        flg |= 0x08
    if content_checksum:
        flg |= 0x04
    descriptor = bytes((flg, block_id << 4))
    if content_size is not None:
        descriptor += struct.pack("<Q", content_size)
    return MAGIC + descriptor + bytes(((xxh32(descriptor) >> 8) & 0xFF,))


def _encode_blocks(
    data: bytes,
    block_size: int,
    compression_level: int,
    block_checksum: bool,
) -> bytes:
    maximum = _BLOCK_SIZES[_block_id(block_size)]
    acceleration = max(1, -int(compression_level)) if compression_level < 0 else 1
    parts: list[bytes] = []
    for start in range(0, len(data), maximum):
        raw = memoryview(data)[start : start + maximum]
        encoded = compress_raw(raw, acceleration=acceleration)
        if len(encoded) >= len(raw):
            payload = raw.tobytes()
            size_field = len(raw) | 0x80000000
        else:
            payload = encoded
            size_field = len(encoded)
        parts.append(struct.pack("<I", size_field))
        parts.append(payload)
        if block_checksum:
            parts.append(struct.pack("<I", xxh32(payload)))
    return b"".join(parts)


def compress(
    data,
    compression_level=0,
    block_size=BLOCKSIZE_DEFAULT,
    content_checksum=False,
    block_linked=True,
    store_size=True,
    return_bytearray=False,
    block_checksum=False,
    auto_flush=False,
):
    source = as_buffer(data)
    header = _header(
        len(source) if store_size else None,
        block_size,
        bool(block_linked),
        bool(content_checksum),
        bool(block_checksum),
    )
    encoded = header + _encode_blocks(
        source, block_size, int(compression_level), bool(block_checksum)
    )
    encoded += b"\0\0\0\0"
    if content_checksum:
        encoded += struct.pack("<I", xxh32(source))
    _ = auto_flush
    return bytearray(encoded) if return_bytearray else encoded


def _parse_header(data: bytes):
    if len(data) < 7:
        raise LZ4FrameError("frame header is incomplete")
    if data[:4] != MAGIC:
        raise LZ4FrameError("invalid LZ4 frame magic")
    flg, bd = data[4], data[5]
    if (flg >> 6) != 1 or (flg & 0x02):
        raise LZ4FrameError("unsupported LZ4 frame descriptor")
    block_id = (bd >> 4) & 7
    if block_id not in _BLOCK_SIZES or (bd & 0x8F):
        raise LZ4FrameError("invalid LZ4 frame block size")
    position = 6
    content_size = 0
    if flg & 0x08:
        if position + 8 > len(data):
            raise LZ4FrameError("frame header is incomplete")
        content_size = struct.unpack_from("<Q", data, position)[0]
        position += 8
    if flg & 0x01:
        if position + 4 > len(data):
            raise LZ4FrameError("frame header is incomplete")
        raise LZ4FrameError("dictionary-ID frames are not supported")
    if position >= len(data):
        raise LZ4FrameError("frame header is incomplete")
    expected = (xxh32(data[4:position]) >> 8) & 0xFF
    if data[position] != expected:
        raise LZ4FrameError("frame header checksum mismatch")
    position += 1
    info = {
        "block_size": _BLOCK_SIZES[block_id],
        "block_size_id": block_id,
        "block_linked": not bool(flg & 0x20),
        "content_checksum": bool(flg & 0x04),
        "block_checksum": bool(flg & 0x10),
        "skippable": False,
        "content_size": content_size,
    }
    return info, position


def get_frame_info(frame):
    info, _ = _parse_header(as_buffer(frame))
    return info


def _scan_blocks(data: bytes, info: dict, position: int):
    blocks = []
    capacity = 0
    while True:
        if position + 4 > len(data):
            raise LZ4FrameError("frame is incomplete")
        field = struct.unpack_from("<I", data, position)[0]
        position += 4
        if field == 0:
            break
        uncompressed = bool(field & 0x80000000)
        size = field & 0x7FFFFFFF
        if size == 0 or size > info["block_size"]:
            raise LZ4FrameError("invalid LZ4 frame block")
        if position + size > len(data):
            raise LZ4FrameError("frame is incomplete")
        payload_position = position
        position += size
        if info["block_checksum"]:
            if position + 4 > len(data):
                raise LZ4FrameError("frame is incomplete")
            checksum = struct.unpack_from("<I", data, position)[0]
            if xxh32(data[payload_position : payload_position + size]) != checksum:
                raise LZ4FrameError("block checksum mismatch")
            position += 4
        blocks.append((payload_position, size, uncompressed))
        capacity += size if uncompressed else info["block_size"]
    if info["content_checksum"]:
        if position + 4 > len(data):
            raise LZ4FrameError("frame is incomplete")
        content_hash = struct.unpack_from("<I", data, position)[0]
        position += 4
    else:
        content_hash = None
    if info["content_size"]:
        capacity = info["content_size"]
    return blocks, capacity, content_hash, position


def decompress(data, return_bytearray=False, return_bytes_read=False):
    source = as_buffer(data)
    info, position = _parse_header(source)
    blocks, capacity, content_hash, consumed = _scan_blocks(source, info, position)
    destination, destination_address = writable_bytes(max(capacity, 1))
    source_address, _, source_keepalive = buffer_address(source)
    written = 0
    try:
        for payload_position, size, uncompressed in blocks:
            if uncompressed:
                if written + size > capacity:
                    raise LZ4FrameError("frame content exceeds declared size")
                ctypes.memmove(
                    destination_address + written,
                    source_address + payload_position,
                    size,
                )
                written += size
            else:
                produced = decompress_into(
                    source,
                    payload_position,
                    size,
                    destination,
                    written,
                    info["block_linked"],
                )
                written += produced
                if written > capacity:
                    raise LZ4FrameError("frame content exceeds output capacity")
    except LibraryError as exc:
        raise LZ4FrameError(str(exc)) from exc
    if info["content_size"] and written != info["content_size"]:
        raise LZ4FrameError("frame content size mismatch")
    if content_hash is not None:
        address = destination_address
        keepalive = destination
        actual = int(lib().mlz_xxh32(address, written, 0)) & 0xFFFFFFFF
        _ = keepalive
        if actual != content_hash:
            raise LZ4FrameError("content checksum mismatch")
    _ = source_keepalive
    payload = destination if written == capacity and capacity else destination[:written]
    result = bytearray(payload) if return_bytearray else payload
    return (result, consumed) if return_bytes_read else result


@dataclass
class _CompressionContext:
    active: bool = False
    block_size: int = BLOCKSIZE_DEFAULT
    compression_level: int = 0
    block_checksum: bool = False
    content_checksum: bool = False
    declared_size: int | None = None
    content: bytearray = field(default_factory=bytearray)


def create_compression_context():
    return _CompressionContext()


def compress_begin(
    context,
    source_size=0,
    compression_level=0,
    block_size=BLOCKSIZE_DEFAULT,
    content_checksum=0,
    content_size=1,
    block_linked=0,
    frame_type=0,
    auto_flush=1,
    block_checksum=False,
    return_bytearray=False,
):
    if not isinstance(context, _CompressionContext):
        raise TypeError("invalid compression context")
    if frame_type != 0:
        raise ValueError("only standard frames are supported")
    context.active = True
    context.block_size = block_size
    context.compression_level = int(compression_level)
    context.block_checksum = bool(block_checksum)
    context.content_checksum = bool(content_checksum)
    context.declared_size = (
        int(source_size) if content_size and int(source_size) > 0 else None
    )
    context.content.clear()
    encoded = _header(
        context.declared_size,
        block_size,
        bool(block_linked),
        bool(content_checksum),
        bool(block_checksum),
    )
    _ = auto_flush
    return bytearray(encoded) if return_bytearray else encoded


def compress_chunk(context, data, return_bytearray=False):
    if not isinstance(context, _CompressionContext) or not context.active:
        raise RuntimeError("compression context has not been started")
    source = as_buffer(data)
    if (
        context.declared_size is not None
        and len(context.content) + len(source) > context.declared_size
    ):
        raise LZ4FrameError("compressed data exceeds the declared source_size")
    context.content.extend(source)
    encoded = _encode_blocks(
        source,
        context.block_size,
        context.compression_level,
        context.block_checksum,
    )
    return bytearray(encoded) if return_bytearray else encoded


def compress_flush(context, end_frame=True, return_bytearray=False):
    if not isinstance(context, _CompressionContext) or not context.active:
        raise RuntimeError("compression context has not been started")
    encoded = b""
    if end_frame:
        if (
            context.declared_size is not None
            and len(context.content) != context.declared_size
        ):
            raise LZ4FrameError("compressed data does not match the declared source_size")
        encoded = b"\0\0\0\0"
        if context.content_checksum:
            encoded += struct.pack("<I", xxh32(context.content))
        context.active = False
    return bytearray(encoded) if return_bytearray else encoded


class LZ4FrameFile(io.BufferedIOBase):
    """Binary file wrapper; compression is buffered until close."""

    def __init__(
        self,
        filename,
        mode="r",
        *,
        return_bytearray=False,
        source_size=0,
        block_size=BLOCKSIZE_DEFAULT,
        block_linked=True,
        compression_level=0,
        content_checksum=False,
        block_checksum=False,
        auto_flush=False,
    ):
        super().__init__()
        binary_mode = mode.replace("b", "")
        if binary_mode not in {"r", "w", "x", "a"}:
            raise ValueError("mode must be r, w, x, or a")
        self._mode = binary_mode
        self._return_bytearray = bool(return_bytearray)
        self._settings = dict(
            block_size=block_size,
            block_linked=block_linked,
            compression_level=compression_level,
            content_checksum=content_checksum,
            block_checksum=block_checksum,
            store_size=bool(source_size),
            auto_flush=auto_flush,
        )
        self._owns_file = not hasattr(filename, "read") and not hasattr(filename, "write")
        file_mode = binary_mode + "b"
        self._file = builtins.open(filename, file_mode) if self._owns_file else filename
        if binary_mode == "r":
            self._buffer = io.BytesIO(decompress(self._file.read()))
        else:
            self._buffer = io.BytesIO()

    def readable(self):
        return self._mode == "r"

    def writable(self):
        return self._mode in {"w", "x", "a"}

    def seekable(self):
        return True

    def read(self, size=-1):
        if not self.readable():
            raise io.UnsupportedOperation("not readable")
        result = self._buffer.read(size)
        return bytearray(result) if self._return_bytearray else result

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def write(self, data):
        if not self.writable():
            raise io.UnsupportedOperation("not writable")
        return self._buffer.write(as_bytes(data))

    def seek(self, offset, whence=io.SEEK_SET):
        return self._buffer.seek(offset, whence)

    def tell(self):
        return self._buffer.tell()

    def flush(self):
        if self.closed:
            raise ValueError("flush of closed file")

    def close(self):
        if self.closed:
            return
        if self.writable():
            payload = self._buffer.getvalue()
            encoded = compress(payload, **self._settings)
            self._file.write(encoded)
            self._file.flush()
        self._buffer.close()
        if self._owns_file:
            self._file.close()
        super().close()


def open(
    filename,
    mode="rb",
    *,
    encoding=None,
    errors=None,
    newline=None,
    **kwargs,
):
    text = "t" in mode
    raw_mode = mode.replace("t", "").replace("b", "")
    binary = LZ4FrameFile(filename, raw_mode, **kwargs)
    if text:
        return io.TextIOWrapper(
            binary, encoding=encoding, errors=errors, newline=newline
        )
    return binary
