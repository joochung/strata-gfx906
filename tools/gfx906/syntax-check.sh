#!/usr/bin/env bash
# syntax-check.sh - P5.3: compile the touched engine files with the REAL gfx906 build flags, syntax only.
#
# The flags are harvested from a configured gfx906 build tree's build.ninja (the CXX_COMPILER rule for the
# glm_dense / glm_generate objects in /ai/engines/gfx906/worktrees/xolotl-merge/build-906/build.ninja: DEFINES,
# FLAGS and INCLUDES verbatim), with -fsyntax-only added and no -o: no object file, no link, no GPU process.
# This is the cheapest gate that catches a type error in the engine's new code; the full build + ctest is
# tools/gfx906/build-906-gate.sh and belongs to the parent (this box may have its RAM committed elsewhere).
#
# Usage: tools/gfx906/syntax-check.sh [file ...]     (default: the GLM engine's two files)
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
GGML=${GGML:-/ai/engines/gfx906/llama.cpp-xolotl}
CXX=${CXX:-/opt/rocm-7.2.0/lib/llvm/bin/clang++}
[ -x "$CXX" ] || { echo "syntax-check: no compiler at $CXX (set CXX=)"; exit 1; }
[ -d "$GGML/ggml/include" ] || { echo "syntax-check: no ggml at $GGML (set GGML=)"; exit 1; }

FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
    FILES=(tools/glm/glm_dense.cpp tools/glm/glm_generate.cpp)
fi

# -x hip + --offload-arch=gfx906: the device-side compile the HIP target really does (the objects are built by the
# HIP toolchain, so the engine's headers are parsed for the gfx906 profile too).
rc=0
for f in "${FILES[@]}"; do
    echo "=== syntax-check $f ==="
    "$CXX" --offload-arch=gfx906 -fsyntax-only -x hip -std=c++20 \
        -DGGML_USE_CPU -DGGML_USE_CUDA -DSTRATA_HIP_GFX906=1 -DSTRATA_Q6K_EXPERTS=1 \
        -DSTRATA_NATIVE_EXPERTS=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -DSTRATA_VERSION='"syntax-check"' \
        -I/opt/rocm-7.2.0/include \
        -I"$WT/include" -I"$WT/tools/glm" -I"$WT/tools/ds4" -I"$WT/tools/gfx906" \
        -I"$GGML/ggml/src" -I"$GGML/ggml/src/../include" \
        -isystem "$WT/include/strata/platform/hip_compat" \
        "$f"
    one=$?
    echo "--- $f: exit $one"
    [ $one -eq 0 ] || rc=$one
done
echo "syntax-check: $(( ${#FILES[@]} )) file(s), $( [ $rc -eq 0 ] && echo 'all exit 0' || echo "LAST FAILURE $rc" )"
exit $rc
