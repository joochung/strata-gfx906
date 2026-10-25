#!/usr/bin/env bash
# gate-relay-ring-cover.sh - P5.3b host gate: what a staged run's window ring actually owes its queries, and what it does NOT.
#
# Compiled and run with plain g++: no CUDA, no HIP, no GPU, no model, no RAM headroom.  It replays the engine's own ring
# arithmetic - the per-pass write-back and the per-query read (build_attn through i_wslot / i_widx), and the window the
# position tables ask for (upload_pos_tables) - for the two tilings a staged run could use over the same geometry:
#
#   RELEASE   each piece fills the column map's room for its window, exactly as the release tiles a staged prompt.
#   CARRY     the ring carry this branch first tried: the piece that opens a window re-runs the row below as its row 0.
#
# WHAT IT PROVES (and it is the opposite of what the carry was for):
#   * A contiguous prefill piece reads the ring ONLY for positions below its own first row, and a piece's window rows above
#     that come from the piece above's write-back - which the release's tiling leaves complete.  The first decode's window
#     is COMPLETE under the release's tiling (SWA-1 of SWA-1 rows) and STALE under the carry.
#   * The carry RUNS rows twice, which double-advances the recurrent state - strictly worse than the hole it was meant to close.
#
# So the ring is NOT the staged token-2 defect, and the tiling is not where the fix belongs.  This gate is kept as the
# red-control for that conclusion: if a future change reintroduces a carry, this gate says so with numbers.
#
# Usage: tools/gfx906/gate-relay-ring-cover.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=${TMPDIR:-/tmp}/test_relay_ring_cover.$$

echo "=== [1/2] compile the standalone gate (plain g++, C++17, warnings are errors) ==="
g++ -std=c++17 -O0 -Wall -Wextra -Werror -o "$OUT" tools/gfx906/test_relay_ring_cover.cpp || {
    echo "GATE FAIL: compile"; exit 1; }

echo "=== [2/2] run it ==="
"$OUT"; rc=$?
rm -f "$OUT"
[ $rc -eq 0 ] || { echo "GATE FAIL: test_relay_ring_cover (rc=$rc)"; exit 1; }
echo "gate-relay-ring-cover: OK"
