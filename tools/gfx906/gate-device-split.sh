#!/usr/bin/env bash
# gate-device-split.sh - P5.2 gate: the opt-in --gpus/--layer-split stage math of the PORTED engines
# (tools/gfx906/device_split.hpp), compiled and run with plain g++ so it needs no CUDA, no HIP, no GPU, no model and
# no RAM headroom.  It makes no runtime claim: it checks parsing and arithmetic only.
#
# The full HIP build + ctest gate that this change still needs (it is NOT run here - the build takes ~8 min at high
# memory and this box may have its RAM committed elsewhere) is printed at the end.
#
# Usage: tools/gfx906/gate-device-split.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=${TMPDIR:-/tmp}/test_device_split.$$

echo "=== [1/2] compile the standalone gate (plain g++, C++17, warnings are errors) ==="
g++ -std=c++17 -O0 -Wall -Wextra -Werror -I tools/gfx906 -o "$OUT" tools/gfx906/test_device_split.cpp || {
    echo "GATE FAIL: compile"; exit 1; }

echo "=== [2/2] run it ==="
"$OUT"; rc=$?
rm -f "$OUT"
[ $rc -eq 0 ] || { echo "GATE FAIL: test_device_split (rc=$rc)"; exit 1; }

cat <<'TEXT'

--- Syntax-only check of the two engine files this change touches (no link, no GPU; needs ROCm + the ggml checkout):
    WT=$(pwd)
    hipcc --offload-arch=gfx906 -fsyntax-only -std=c++20 -DDS41_HAVE_DSPARK -DGGML_USE_CPU -DGGML_USE_CUDA \
        -DSTRATA_HIP_GFX906=1 -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_Q6K_EXPERTS=1 -D__HIP_PLATFORM_AMD__=1 \
        -I$WT/include -I$WT/tools/v41/engine -I$WT/tools/gfx906 -I<ggml>/ggml/src/../include \
        -isystem $WT/include/strata/platform/hip_compat -isystem /opt/rocm/include \
        tools/v41/engine/ds41_generate.cpp
    (same, without -DDS41_HAVE_DSPARK and with -I$WT/tools/glm -I$WT/tools/ds4 for tools/glm/glm_generate.cpp)

--- Full HIP build gate + ctest vs the baseline fail-set (MEMORY PRECONDITION: this box has no swap; keep at least
    2 GiB MemAvailable - the -j4 here is what tools/gfx906/gate-route-census.sh watches for that.  ~8 minutes):
    cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
        -DSTRATA_GGML_CUDA=ON -DSTRATA_GGML_DIR=/ai/engines/gfx906/llama.cpp-xolotl \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
    cmake --build build-906 -j4
    cd build-906 && ctest -j4
    # the baseline fail-set on this fork (docs/gfx906/START-HERE.md, gate-route-census.sh):
    #   mmvq_il_parity  ple_parity  split_lend_complement_test  verify_parity
    # anything else failing is this change.  No GPU process, no engine run and no benchmark is part of P5.2:
    # runtime validation belongs to the P5.3 split-smoke task, which is the one with hardware access.
TEXT
echo "GATE DONE (standalone part)"
