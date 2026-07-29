#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${repo_root}/dist"
mojo build --emit shared-lib "${repo_root}/src/lz4.mojo" -o "${repo_root}/dist/libmojo-lz4.so"
