"""Benchmark mojo-lz4 against the upstream Python lz4 bindings."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import lz4.block as upstream_block
import lz4.frame as upstream_frame
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from mojo_lz4 import block, frame  # noqa: E402


def best_time(function, repetitions=7):
    function()
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def machine():
    model = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model}; {platform.system()} {platform.machine()}; Python {platform.python_version()}"


def main():
    size = 8 * 1024 * 1024
    record = (
        b'{"time":"2026-07-29T12:00:00Z","level":"info",'
        b'"service":"compressor","message":"request completed","status":200}\n'
    )
    text = (record * (size // len(record) + 1))[:size]
    random_data = np.random.default_rng(7).integers(
        0, 256, size=size, dtype=np.uint8
    ).tobytes()
    large_random_data = np.random.default_rng(11).integers(
        0, 256, size=128 * 1024 * 1024, dtype=np.uint8
    ).tobytes()

    upstream_text_block = upstream_block.compress(text, store_size=False)
    upstream_random_block = upstream_block.compress(random_data, store_size=False)
    upstream_text_frame = upstream_frame.compress(
        text,
        block_size=upstream_frame.BLOCKSIZE_MAX4MB,
        block_linked=False,
        store_size=False,
    )
    upstream_text_checksum_frame = upstream_frame.compress(
        text,
        block_size=upstream_frame.BLOCKSIZE_MAX4MB,
        block_linked=False,
        store_size=False,
        content_checksum=True,
    )

    cases = [
        (
            "block compress, repetitive 8 MiB",
            lambda: block.compress(text, store_size=False),
            lambda: upstream_block.compress(text, store_size=False),
        ),
        (
            "block decompress, repetitive 8 MiB",
            lambda: block.decompress(
                upstream_text_block, uncompressed_size=len(text)
            ),
            lambda: upstream_block.decompress(
                upstream_text_block, uncompressed_size=len(text)
            ),
        ),
        (
            "block compress, random 8 MiB",
            lambda: block.compress(random_data, store_size=False),
            lambda: upstream_block.compress(random_data, store_size=False),
        ),
        (
            "block decompress, random 8 MiB",
            lambda: block.decompress(
                upstream_random_block, uncompressed_size=len(random_data)
            ),
            lambda: upstream_block.decompress(
                upstream_random_block, uncompressed_size=len(random_data)
            ),
        ),
        (
            "frame compress, repetitive 8 MiB",
            lambda: frame.compress(
                text,
                block_size=frame.BLOCKSIZE_MAX4MB,
                block_linked=False,
                store_size=False,
            ),
            lambda: upstream_frame.compress(
                text,
                block_size=upstream_frame.BLOCKSIZE_MAX4MB,
                block_linked=False,
                store_size=False,
            ),
        ),
        (
            "frame compress, random 128 MiB",
            lambda: frame.compress(
                large_random_data,
                block_size=frame.BLOCKSIZE_MAX4MB,
                block_linked=False,
                store_size=False,
            ),
            lambda: upstream_frame.compress(
                large_random_data,
                block_size=upstream_frame.BLOCKSIZE_MAX4MB,
                block_linked=False,
                store_size=False,
            ),
        ),
        (
            "frame decompress, repetitive 8 MiB",
            lambda: frame.decompress(upstream_text_frame),
            lambda: upstream_frame.decompress(upstream_text_frame),
        ),
        (
            "frame compress + checksum, repetitive 8 MiB",
            lambda: frame.compress(
                text,
                block_size=frame.BLOCKSIZE_MAX4MB,
                block_linked=False,
                store_size=False,
                content_checksum=True,
            ),
            lambda: upstream_frame.compress(
                text,
                block_size=upstream_frame.BLOCKSIZE_MAX4MB,
                block_linked=False,
                store_size=False,
                content_checksum=True,
            ),
        ),
        (
            "frame decompress + checksum, repetitive 8 MiB",
            lambda: frame.decompress(upstream_text_checksum_frame),
            lambda: upstream_frame.decompress(upstream_text_checksum_frame),
        ),
    ]

    print(f"Machine: {machine()}")
    print(
        "Mojo:",
        subprocess.run(
            ["mojo", "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    print(f"Upstream: lz4 {__import__('lz4').__version__}")
    print()
    print("| case | mojo-lz4 | upstream lz4 | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, upstream_function in cases:
        mojo_seconds, mojo_result = best_time(mojo_function)
        upstream_seconds, upstream_result = best_time(upstream_function)
        if bytes(mojo_result) != bytes(upstream_result):
            # Compressed byte streams need only decode to the same source.
            if "compress" not in name or "decompress" in name:
                raise AssertionError(f"benchmark outputs differ for {name}")
        relative = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {relative:.2f}x |"
        )


if __name__ == "__main__":
    main()
