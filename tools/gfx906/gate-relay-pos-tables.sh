#!/usr/bin/env bash
# gate-relay-pos-tables.sh - P5.3b host gate: the staged run's FIRST DECODE read set must equal the unsplit run's.
#
# Compiled and run with plain g++: no CUDA, no HIP, no GPU, no model, no RAM headroom.  It checks the arithmetic of the
# relay seam (tools/v41/engine/ds41_dense.cpp: Impl::upload_pos_tables and Ds41Dense::relay_write_in), NOT runtime output.
#
# The gate carries BOTH models of the seam in one test binary: the PRE-FIX model (a relayed stage's position tables -
# i_pos / i_widx / i_wmask / i_wslot - left stale or zeroed) and the POST-FIX model (relay_write_in refreshes them).  It
# proves the gate sees the defect on the pre-fix tree (every case diverges) and closes it on the post-fix tree (every
# case matches).  This is the fail-before / pass-after evidence for the fix that makes the witness runnable; the gate
# itself never runs the real engine, so it passes on this tree and would have failed on the pre-fix tree's model too.
#
# Usage: tools/gfx906/gate-relay-pos-tables.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=${TMPDIR:-/tmp}/test_relay_pos_tables.$$

echo "=== [1/2] compile the standalone gate (plain g++, C++17, warnings are errors) ==="
g++ -std=c++17 -O0 -Wall -Wextra -Werror -o "$OUT" tools/gfx906/test_relay_pos_tables.cpp || {
    echo "GATE FAIL: compile"; exit 1; }

echo "=== [2/2] run it (gated on the REAL ds41_dense.cpp for the post-fix verdict) ==="
"$OUT" --real tools/v41/engine/ds41_dense.cpp; rc=$?
rm -f "$OUT"
[ $rc -eq 0 ] || { echo "GATE FAIL: test_relay_pos_tables (rc=$rc)"; exit 1; }
echo "gate-relay-pos-tables: OK"
