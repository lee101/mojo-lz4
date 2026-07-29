"""ctypes bridge to the Mojo LZ4 kernels."""

from __future__ import annotations

import ctypes
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-lz4.so")

I = ctypes.c_int64
_I64_MAX = (1 << 63) - 1
_LZ4_MAX_INPUT_SIZE = 0x7E000000
_new_bytes = ctypes.pythonapi.PyBytes_FromStringAndSize
_new_bytes.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_new_bytes.restype = ctypes.py_object
_bytes_data = ctypes.pythonapi.PyBytes_AsString
_bytes_data.argtypes = [ctypes.py_object]
_bytes_data.restype = ctypes.c_void_p


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_get_buffer = ctypes.pythonapi.PyObject_GetBuffer
_get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi.PyBuffer_Release
_release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
_release_buffer.restype = None


class _BufferExport:
    __slots__ = ("buffer", "exported")

    def __init__(self, data):
        self.buffer = _PyBuffer()
        self.exported = False
        _get_buffer(data, ctypes.byref(self.buffer), 0)
        self.exported = True

    def __del__(self):
        if self.exported:
            _release_buffer(ctypes.byref(self.buffer))


_SIGNATURES = {
    "mlz_compress": ([I, I, I, I, I, I, I], I),
    "mlz_decompress": ([I, I, I, I, I], I),
    "mlz_xxh32": ([I, I, I], I),
}


class LibraryError(RuntimeError):
    pass


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB_PATH):
            raise LibraryError("shared library is missing; run `pixi run build`")
        _library = ctypes.CDLL(LIB_PATH)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def as_buffer(data):
    if isinstance(data, bytes):
        return data
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise TypeError("a bytes-like object is required") from exc
    if not view.c_contiguous:
        raise BufferError("source buffer is not C-contiguous")
    return view.cast("B")


def as_bytes(data) -> bytes:
    source = as_buffer(data)
    return source if isinstance(source, bytes) else source.tobytes()


def bytes_address(data: bytes) -> tuple[int, ctypes.c_char_p]:
    keepalive = ctypes.c_char_p(data)
    return ctypes.cast(keepalive, ctypes.c_void_p).value or 0, keepalive


def bytearray_address(data: bytearray) -> tuple[int, object]:
    keepalive = (ctypes.c_ubyte * len(data)).from_buffer(data)
    return ctypes.addressof(keepalive), keepalive


def writable_bytes(size: int) -> tuple[bytes, int]:
    if not 0 <= size <= _I64_MAX:
        raise OverflowError("buffer size is outside the native ABI range")
    data = _new_bytes(None, size)
    address = int(_bytes_data(data))
    if not address:
        raise MemoryError("Python returned a null bytes buffer")
    return data, address


def buffer_address(data) -> tuple[int, int, object]:
    if isinstance(data, bytes):
        storage = data or b"\0"
        address, keepalive = bytes_address(storage)
        return address, len(data), keepalive
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise TypeError("a bytes-like object is required") from exc
    if not view.c_contiguous:
        raise BufferError("source buffer is not C-contiguous")
    view = view.cast("B")
    if not view:
        storage = b"\0"
        address, keepalive = bytes_address(storage)
        return address, 0, (view, keepalive)
    keepalive = _BufferExport(view)
    address = int(keepalive.buffer.buf)
    size = int(keepalive.buffer.len)
    if not address:
        raise BufferError("buffer exporter returned a null pointer")
    if not 0 <= size <= _I64_MAX:
        raise OverflowError("buffer size is outside the native ABI range")
    return address, size, (view, keepalive)


_thread_state = threading.local()


def _hash_table():
    table = getattr(_thread_state, "hash_table", None)
    if table is None:
        table = (ctypes.c_int32 * 65536)()
        _thread_state.hash_table = table
    return table


def xxh32(data, seed: int = 0) -> int:
    seed = int(seed)
    if not 0 <= seed <= 0xFFFFFFFF:
        raise OverflowError("seed must fit in an unsigned 32-bit integer")
    address, size, keepalive = buffer_address(data)
    value = int(lib().mlz_xxh32(address, size, seed))
    _ = keepalive
    return value & 0xFFFFFFFF


def compress_raw(data, acceleration: int = 1, dictionary=None) -> bytes:
    prefix = b"" if dictionary is None else as_bytes(dictionary)[-65535:]
    if prefix:
        source = as_bytes(data)
        combined = prefix + source
        source_address, keepalive = bytes_address(combined)
        source_size = len(source)
        total_size = len(combined)
        initial_size = len(prefix)
    else:
        source_address, source_size, keepalive = buffer_address(data)
        total_size = source_size
        initial_size = 0
    if total_size > _LZ4_MAX_INPUT_SIZE:
        raise OverflowError("LZ4 block input exceeds the format limit")
    acceleration = int(acceleration)
    if not 1 <= acceleration <= _I64_MAX:
        raise OverflowError("acceleration is outside the native ABI range")
    capacity = source_size + source_size // 255 + 16
    destination, destination_address = writable_bytes(max(capacity, 1))
    table = _hash_table()
    result = lib().mlz_compress(
        source_address,
        total_size,
        initial_size,
        destination_address,
        capacity,
        ctypes.addressof(table),
        acceleration,
    )
    _ = keepalive
    if result < 0:
        raise LibraryError(f"compression failed with error {result}")
    return destination[:result]


def decompress_raw(data, capacity: int, dictionary=None) -> bytes:
    prefix = b"" if dictionary is None else as_bytes(dictionary)[-65535:]
    if capacity < 0:
        raise ValueError("uncompressed_size must be non-negative")
    if capacity > _LZ4_MAX_INPUT_SIZE:
        raise OverflowError("LZ4 block output exceeds the format limit")
    destination, destination_address = writable_bytes(max(len(prefix) + capacity, 1))
    if prefix:
        ctypes.memmove(destination_address, prefix, len(prefix))
    source_address, source_size, source_keepalive = buffer_address(data)
    result = lib().mlz_decompress(
        source_address,
        source_size,
        destination_address,
        len(prefix) + capacity,
        len(prefix),
    )
    _ = source_keepalive
    if result < 0:
        raise LibraryError(f"decompression failed with error {result}")
    if not prefix and result == capacity:
        return b"" if capacity == 0 else destination
    return destination[len(prefix) : len(prefix) + result]


def decompress_into(
    source,
    source_offset: int,
    source_size: int,
    destination: bytes | bytearray,
    written: int,
    linked: bool,
) -> int:
    source_address, actual_source_size, source_keepalive = buffer_address(source)
    if source_offset < 0 or source_size < 0:
        raise ValueError("source offset and size must be non-negative")
    if source_offset > actual_source_size or source_size > actual_source_size - source_offset:
        raise ValueError("source slice is outside the input buffer")
    if written < 0 or written > len(destination):
        raise ValueError("written is outside the destination buffer")
    if isinstance(destination, bytes):
        destination_address = int(_bytes_data(destination))
        destination_keepalive = destination
    else:
        destination_address, destination_keepalive = bytearray_address(destination)
    if linked:
        dst_address = destination_address
        dst_capacity = len(destination)
        initial_size = written
    else:
        dst_address = destination_address + written
        dst_capacity = len(destination) - written
        initial_size = 0
    result = lib().mlz_decompress(
        source_address + source_offset,
        source_size,
        dst_address,
        dst_capacity,
        initial_size,
    )
    _ = source_keepalive, destination_keepalive
    if result < 0:
        raise LibraryError(f"decompression failed with error {result}")
    return result
