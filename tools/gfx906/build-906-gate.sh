#!/usr/bin/env bash
# build-906-gate.sh - configure + build + ctest the gfx906 HIP target behind a memory watchdog.
#
# Why the watchdog: this box has no swap and the HIP translation units are large. A build that outruns
# the RAM left standing does not fail cleanly - the kernel picks a victim, and on a box with ROCm
# resident that can be the engine. The watchdog watches MemAvailable and kills the build (never
# anything else) before that happens, then says so.
#
# Nothing here touches the GPU: no engine is run, no model is loaded, no benchmark runs. The ctest
# battery is host-side; the fork's known baseline failures are named at the end.
#
# Usage: tools/gfx906/build-906-gate.sh [jobs]      (default jobs = 4)
set -uo pipefail
JOBS="${1:-4}"
FLOOR_GIB=2                      # kill the build below this MemAvailable
WT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$WT/build-906"
LOG="$WT/logs/build-906-gate-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$WT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "== build-906-gate: worktree $WT, jobs $JOBS, floor ${FLOOR_GIB} GiB, log $LOG"

avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
echo "MemAvailable at start: $(avail_gib) GiB"
if [ "$(avail_gib)" -lt "$FLOOR_GIB" ]; then
  echo "REFUSING: MemAvailable is below the ${FLOOR_GIB} GiB floor before the build even starts."
  echo "Free some RAM (the engine holding it is usually the answer) and re-run."
  exit 3
fi

# Watchdog: kill only this script's build processes, by this script's own process group.
(
  while :; do
    sleep 5
    if [ "$(avail_gib)" -lt "$FLOOR_GIB" ]; then
      echo "WATCHDOG: MemAvailable $(avail_gib) GiB < ${FLOOR_GIB} GiB - killing the build (pid $$)"
      pkill -TERM -P "$$" -f 'ninja|cc1plus|clang' 2>/dev/null
      kill -TERM "$$" 2>/dev/null
      exit 0
    fi
  done
) &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT

cmake -S "$WT" -B "$BUILD" -G Ninja \
  -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_CUDA=ON -DSTRATA_GGML_DIR=/ai/engines/gfx906/llama.cpp-xolotl \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  || { echo "CONFIGURE FAILED"; exit 1; }

cmake --build "$BUILD" -j "$JOBS" || { echo "BUILD FAILED"; exit 1; }
echo "BUILD OK"

cd "$BUILD"
ctest -j "$JOBS" --output-on-failure | tail -40
echo
echo "Expected failures on this fork, all pre-existing on main (see docs/gfx906/START-HERE.md):"
echo "  mmvq_il_parity  ple_parity  split_lend_complement_test  verify_parity"
echo "Anything else in that list is the change under test, not the baseline."
