"""LZ4 block codec and xxHash32 kernels exposed through a small C ABI."""

from max.algorithm import parallelize
from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


@always_inline
def copy_bytes(
    dst: BPtr,
    dst_pos: Int,
    src: BPtr,
    src_pos: Int,
    size: Int,
):
    comptime BYTE_W = simdwidthof[DType.float64]() * 8
    var i = 0
    while i + BYTE_W <= size:
        var values = src.load[width=BYTE_W, alignment=1](src_pos + i)
        dst.store[alignment=1](dst_pos + i, values)
        i += BYTE_W
    while i < size:
        dst[dst_pos + i] = src[src_pos + i]
        i += 1


@always_inline
def read32(src: BPtr, i: Int) -> UInt32:
    return (
        UInt32(src[i])
        | (UInt32(src[i + 1]) << 8)
        | (UInt32(src[i + 2]) << 16)
        | (UInt32(src[i + 3]) << 24)
    )


@always_inline
def rotate_left(v: UInt32, amount: Int) -> UInt32:
    return (v << UInt32(amount)) | (v >> UInt32(32 - amount))


@always_inline
def hash_sequence(src: BPtr, i: Int) -> Int:
    return Int((read32(src, i) * UInt32(2654435761)) >> 16)


def emit_length(dst: BPtr, pos: Int, value: Int) -> Int:
    var op = pos
    var remaining = value
    while remaining >= 255:
        dst[op] = UInt8(255)
        op += 1
        remaining -= 255
    dst[op] = UInt8(remaining)
    return op + 1


def compress_block(
    src: BPtr,
    src_size: Int,
    initial_size: Int,
    dst: BPtr,
    dst_capacity: Int,
    table: I32Ptr,
    acceleration: Int,
) -> Int:
    for j in range(65536):
        table[j] = Int32(-1)

    var dict_pos = 0
    while dict_pos + 4 <= initial_size:
        table[hash_sequence(src, dict_pos)] = Int32(dict_pos)
        dict_pos += 1

    var anchor = initial_size
    var ip = initial_size
    var op = 0
    var step = acceleration
    if step < 1:
        step = 1
    var search_attempts = step << 6

    # Leaving five final literals satisfies the canonical LZ4 block constraints.
    while ip + 12 <= src_size:
        var h = hash_sequence(src, ip)
        var match_pos = Int(table[h])
        table[h] = Int32(ip)

        var matched = False
        if match_pos >= 0 and ip - match_pos <= 65535:
            matched = read32(src, match_pos) == read32(src, ip)

        if not matched:
            var skip = search_attempts >> 6
            search_attempts += 1
            ip += skip
            continue

        var literal_size = ip - anchor
        var match_size = 4
        comptime BYTE_W = simdwidthof[DType.float64]() * 8
        while ip + match_size + BYTE_W <= src_size - 5:
            var match_values = src.load[width=BYTE_W, alignment=1](
                match_pos + match_size
            )
            var input_values = src.load[width=BYTE_W, alignment=1](
                ip + match_size
            )
            if match_values != input_values:
                break
            match_size += BYTE_W
        while (
            ip + match_size < src_size - 5
            and src[match_pos + match_size] == src[ip + match_size]
        ):
            match_size += 1

        # Worst-case space for token, extensions, literals, offset, and match length.
        if op + literal_size + literal_size // 255 + match_size // 255 + 8 > dst_capacity:
            return -2

        var token_pos = op
        op += 1
        var literal_token = literal_size
        if literal_token > 15:
            literal_token = 15
        var match_token = match_size - 4
        if match_token > 15:
            match_token = 15
        dst[token_pos] = UInt8((literal_token << 4) | match_token)

        if literal_size >= 15:
            op = emit_length(dst, op, literal_size - 15)
        copy_bytes(dst, op, src, anchor, literal_size)
        op += literal_size

        var offset = ip - match_pos
        dst[op] = UInt8(offset & 255)
        dst[op + 1] = UInt8((offset >> 8) & 255)
        op += 2
        if match_size - 4 >= 15:
            op = emit_length(dst, op, match_size - 19)

        ip += match_size
        anchor = ip
        search_attempts = step << 6

        # Seed a position near the end of the match without walking every byte.
        if ip >= 2 and ip + 2 < src_size:
            table[hash_sequence(src, ip - 2)] = Int32(ip - 2)

    var literal_size = src_size - anchor
    if op + literal_size + literal_size // 255 + 2 > dst_capacity:
        return -2
    var token_size = literal_size
    if token_size > 15:
        token_size = 15
    dst[op] = UInt8(token_size << 4)
    op += 1
    if literal_size >= 15:
        op = emit_length(dst, op, literal_size - 15)
    copy_bytes(dst, op, src, anchor, literal_size)
    return op + literal_size


def compress_blocks(
    src: BPtr,
    src_size: Int,
    dst: BPtr,
    dst_stride: Int,
    tables: I32Ptr,
    results: I64Ptr,
    block_size: Int,
    acceleration: Int,
) -> Int:
    var block_count = (src_size + block_size - 1) // block_size

    @parameter
    def compress_one(block: Int):
        var offset = block * block_size
        var size = min(block_size, src_size - offset)
        results[block] = Int64(
            compress_block(
                src + offset,
                size,
                0,
                dst + block * dst_stride,
                dst_stride,
                tables + block * 65536,
                acceleration,
            )
        )

    parallelize[compress_one](block_count, min(block_count, 4))
    return block_count


def decompress_block(
    src: BPtr,
    src_size: Int,
    dst: BPtr,
    dst_capacity: Int,
    initial_size: Int,
) -> Int:
    var ip = 0
    var op = initial_size

    while ip < src_size:
        var token = Int(src[ip])
        ip += 1

        var literal_size = token >> 4
        if literal_size == 15:
            while True:
                if ip >= src_size:
                    return -1
                var extension = Int(src[ip])
                ip += 1
                literal_size += extension
                if extension != 255:
                    break

        if ip + literal_size > src_size:
            return -1
        if op + literal_size > dst_capacity:
            return -2
        copy_bytes(dst, op, src, ip, literal_size)
        ip += literal_size
        op += literal_size

        if ip == src_size:
            return op - initial_size
        if ip + 2 > src_size:
            return -1

        var offset = Int(src[ip]) | (Int(src[ip + 1]) << 8)
        ip += 2
        if offset == 0 or offset > op:
            return -3

        var match_size = (token & 15) + 4
        if (token & 15) == 15:
            while True:
                if ip >= src_size:
                    return -1
                var extension = Int(src[ip])
                ip += 1
                match_size += extension
                if extension != 255:
                    break
        if op + match_size > dst_capacity:
            return -2

        var match_pos = op - offset
        comptime BYTE_W = simdwidthof[DType.float64]() * 8
        if offset >= BYTE_W:
            copy_bytes(dst, op, dst, match_pos, match_size)
        else:
            for j in range(match_size):
                dst[op + j] = dst[match_pos + j]
        op += match_size

    return op - initial_size


def xxh32(src: BPtr, size: Int, seed: UInt32) -> UInt32:
    var ip = 0
    var h: UInt32
    if size >= 16:
        comptime W = simdwidthof[DType.float64]()
        var accumulators = SIMD[DType.uint32, W](
            seed + UInt32(2654435761) + UInt32(2246822519),
            seed + UInt32(2246822519),
            seed,
            seed - UInt32(2654435761),
        )
        var words = src.bitcast[UInt32]()
        while ip + 16 <= size:
            accumulators += (
                words.load[width=W, alignment=1](ip // 4)
                * SIMD[DType.uint32, W](2246822519)
            )
            accumulators = (
                (accumulators << SIMD[DType.uint32, W](13))
                | (accumulators >> SIMD[DType.uint32, W](19))
            ) * SIMD[DType.uint32, W](2654435761)
            ip += 16
        h = SIMD[DType.uint32, W](
            rotate_left(accumulators[0], 1),
            rotate_left(accumulators[1], 7),
            rotate_left(accumulators[2], 12),
            rotate_left(accumulators[3], 18),
        ).reduce_add()
    else:
        h = seed + UInt32(374761393)

    h += UInt32(size)
    while ip + 4 <= size:
        h += read32(src, ip) * UInt32(3266489917)
        h = rotate_left(h, 17) * UInt32(668265263)
        ip += 4
    while ip < size:
        h += UInt32(src[ip]) * UInt32(374761393)
        h = rotate_left(h, 11) * UInt32(2654435761)
        ip += 1
    h ^= h >> 15
    h *= UInt32(2246822519)
    h ^= h >> 13
    h *= UInt32(3266489917)
    h ^= h >> 16
    return h


@export("mlz_compress")
def mlz_compress(
    src_addr: Int,
    src_size: Int,
    initial_size: Int,
    dst_addr: Int,
    dst_capacity: Int,
    table_addr: Int,
    acceleration: Int,
) abi("C") -> Int:
    return compress_block(
        BPtr(unsafe_from_address=src_addr),
        src_size,
        initial_size,
        BPtr(unsafe_from_address=dst_addr),
        dst_capacity,
        I32Ptr(unsafe_from_address=table_addr),
        acceleration,
    )


@export("mlz_decompress")
def mlz_decompress(
    src_addr: Int,
    src_size: Int,
    dst_addr: Int,
    dst_capacity: Int,
    initial_size: Int,
) abi("C") -> Int:
    return decompress_block(
        BPtr(unsafe_from_address=src_addr),
        src_size,
        BPtr(unsafe_from_address=dst_addr),
        dst_capacity,
        initial_size,
    )


@export("mlz_compress_blocks")
def mlz_compress_blocks(
    src_addr: Int,
    src_size: Int,
    dst_addr: Int,
    dst_stride: Int,
    table_addr: Int,
    result_addr: Int,
    block_size: Int,
    acceleration: Int,
) abi("C") -> Int:
    return compress_blocks(
        BPtr(unsafe_from_address=src_addr),
        src_size,
        BPtr(unsafe_from_address=dst_addr),
        dst_stride,
        I32Ptr(unsafe_from_address=table_addr),
        I64Ptr(unsafe_from_address=result_addr),
        block_size,
        acceleration,
    )


@export("mlz_xxh32")
def mlz_xxh32(src_addr: Int, size: Int, seed: Int) abi("C") -> Int:
    return Int(xxh32(BPtr(unsafe_from_address=src_addr), size, UInt32(seed)))
