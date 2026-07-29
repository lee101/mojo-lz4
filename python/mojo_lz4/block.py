"""Drop-in one-shot API for :mod:`lz4.block`."""

from __future__ import annotations

import struct

from ._lib import LibraryError, as_buffer, compress_raw, decompress_raw


class LZ4BlockError(RuntimeError):
    pass


def compress(
    source,
    mode="default",
    acceleration=1,
    compression=0,
    return_bytearray=False,
    store_size=True,
    dict=None,
):
    if mode not in {"default", "fast", "high_compression"}:
        raise ValueError("mode must be 'default', 'fast', or 'high_compression'")
    if mode == "fast" and int(acceleration) < 1:
        raise ValueError("acceleration must be at least 1")
    if mode == "high_compression" and compression and not 1 <= int(compression) <= 12:
        raise ValueError("compression must be between 1 and 12")

    data = as_buffer(source)
    step = int(acceleration) if mode == "fast" else 1
    try:
        encoded = compress_raw(data, acceleration=step, dictionary=dict)
    except (LibraryError, OverflowError) as exc:
        raise LZ4BlockError(str(exc)) from exc
    if store_size:
        if len(data) > 0xFFFFFFFF:
            raise OverflowError("LZ4 block size exceeds 32-bit stored-size field")
        encoded = struct.pack("<I", len(data)) + encoded
    return bytearray(encoded) if return_bytearray else encoded


def decompress(source, uncompressed_size=-1, return_bytearray=False, dict=None):
    encoded = as_buffer(source)
    if uncompressed_size is None or int(uncompressed_size) < 0:
        if len(encoded) < 4:
            raise LZ4BlockError("source is too short to contain a stored size")
        capacity = struct.unpack_from("<I", encoded)[0]
        encoded = encoded[4:]
    else:
        capacity = int(uncompressed_size)
    try:
        decoded = decompress_raw(encoded, capacity, dictionary=dict)
    except (LibraryError, OverflowError, ValueError) as exc:
        raise LZ4BlockError(
            f"Decompression failed: corrupt input or insufficient uncompressed_size ({capacity})"
        ) from exc
    return bytearray(decoded) if return_bytearray else decoded
