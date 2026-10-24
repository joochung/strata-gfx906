#!/usr/bin/env bash
# gate-device-split.sh - P5.2/P5.3 gate: the opt-in --gpus/--layer-split stage math (tools/gfx906/device_split.hpp)
# and the stage-boundary relay's staging arithmetic (tools/gfx906/relay_staging.hpp) of the PORTED engines, compiled
# and run with plain g++ so it needs no CUDA, no HIP, no GPU, no model and no RAM headroom.  It makes no runtime
# claim: it checks parsing and arithmetic only.
#
# The full HIP build + ctest gate that this change still needs (it is NOT run here - the build takes ~8 min at high
# memory and this box may have its RAM committed elsewhere) is printed at the end.
#
# Usage: tools/gfx906/gate-device-split.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=${TMPDIR:-/tmp}/test_device_split.$$

echo "=== [1/3] compile the standalone gate (plain g++, C++17, warnings are errors) ==="
g++ -std=c++17 -O0 -Wall -Wextra -Werror -I tools/gfx906 -o "$OUT" tools/gfx906/test_device_split.cpp || {
    echo "GATE FAIL: compile"; exit 1; }

echo "=== [2/3] run it ==="
"$OUT"; rc=$?
rm -f "$OUT"
[ $rc -eq 0 ] || { echo "GATE FAIL: test_device_split (rc=$rc)"; exit 1; }

# ---- [3] the gate's copied geometry vs the ENGINE'S OWN loader, when both are on this box
# test_split_geometry.cpp works from a HAND-COPY of the checkpoint's source lists, because the standalone gate must not
# need ggml, a model or a GPU.  That copy can drift from the file - and every boundary rule in this file is derived from
# those two lists, so a drift is not a typo, it is a wrong split rule.  When the HIP build has been made and the GGUF is
# present, cross-check the copy against ds41_read_geometry(), the same reader the runtime uses (host-only; it validates
# the invariants the engine relies on).  Skipped, not failed, when either is missing: the gate still runs on a box with
# no build and no model.
GGUF=${GGUF:-/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/V41-GSQ.gguf}
GEOM_BIN=build-906/test_ds41_geometry
if [ -x "$GEOM_BIN" ] && [ -r "$GGUF" ]; then
    echo "=== [3/3] the copied geometry against the engine's own reader ($GGUF) ==="
    OUT3=${TMPDIR:-/tmp}/geom_oracle.$$
    if ! "$GEOM_BIN" "$GGUF" > "$OUT3" 2>&1; then
        echo "GATE FAIL: $GEOM_BIN did not accept the checkpoint (see $OUT3)"; exit 1
    fi
    # cand L20 2048x8  and  the per-layer source map ( "39:20/36" = layer 39 reads layer 20's KV, 36's index keys)
    if ! grep -q "cand L20 2048x8" "$OUT3"; then
        echo "GATE FAIL: the checkpoint's candidate layer / block count is not L20 2048x8 as test_split_geometry assumes"
        sed -n 's/^.*\(cand L[0-9]* [0-9]*x[0-9]*\).*/  oracle says: \1/p' "$OUT3" | head -1
        rm -f "$OUT3"; exit 1
    fi
    # the last layer's sources, which set the boundary floor: kv 20, index 36 -> floor 37
    last=$(grep -o "39:[0-9]*/[0-9]*" "$OUT3" | head -1)
    if [ "$last" != "39:20/36" ]; then
        echo "GATE FAIL: layer 39's sources are ${last:-not found}, expected 39:20/36 - the boundary floor (last shared"
        echo "           cache layer + 1) moved, so split_min_boundary, the gate and the witness's SPLIT must move with it"
        rm -f "$OUT3"; exit 1
    fi
    echo "ok: candidate layer and the source map match the gate's copy (floor 37, witness --layer-split 37)"
    rm -f "$OUT3"
else
    echo "=== [3/3] skipped: the geometry oracle needs $GEOM_BIN (make the HIP build) and $GGUF ==="
fi

cat <<'TEXT'

--- Syntax-only check of the engine files (no link, no GPU; needs ROCm + the ggml checkout).  The flags below are
    the real ones, harvested from a configured gfx906 build tree's build.ninja (tools/gfx906/syntax-check.sh runs
    them and prints its exit code for each file):
    WT=$(pwd)
    /opt/rocm-7.2.0/lib/llvm/bin/clang++ --offload-arch=gfx906 -fsyntax-only -x hip -std=c++20 \
        -DGGML_USE_CPU -DGGML_USE_CUDA -DSTRATA_HIP_GFX906=1 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERSION='"gate"' \
        -D__HIP_PLATFORM_AMD__=1 -DUSE_PROF_API=1 -DSTRATA_NATIVE_EXPERTS=1 \
        -I/opt/rocm-7.2.0/include -I$WT/include -I$WT/tools/glm -I$WT/tools/ds4 -I$WT/tools/gfx906 \
        -I/ai/engines/gfx906/llama.cpp-xolotl/ggml/src \
        -I/ai/engines/gfx906/llama.cpp-xolotl/ggml/src/../include \
        -isystem $WT/include/strata/platform/hip_compat \
        tools/glm/glm_dense.cpp tools/glm/glm_generate.cpp

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
