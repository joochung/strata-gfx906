#!/usr/bin/env bash
# gate-relay-slot-collision.sh - P5.3b host gate: the ring write-back table (i_wslot / i_sslot) is ONE table per instance,
# sized by the LAST pass that ran - and a staged run does NOT corrupt it.
#
# Compiled and run with plain g++: no CUDA, no HIP, no GPU, no model, no RAM headroom.
#
# This gate was written to prove a SECOND root cause for the staged run's diverging first decode token: that a pass whose
# write-back addresses entries the previous (narrower) pass never published writes rows into OTHER positions' ring rows.  It
# does not prove that.  Replayed over every geometry below the ring's width, and over the witness geometry (the 14-token
# prompt at --relay-chunk 64, where the first decode token diverged), a staged run's write-backs stay inside entries its own
# earlier passes wrote, an unsplit control's do too, and the two shapes leave the ring IDENTICAL, slot for slot.  The theory
# is refuted the same way the ring-reach theory was (90230297, reverted by eb1fed5e), and the engine change built on it was
# reverted with this gate turned into a red control.
#
# WHAT IT KEEPS RED, so the conclusion cannot rot into an assumption:
#   * a write-back that addresses an entry no pass in the run published (an unpublished i_wslot entry is real damage in a
#     real run: it lands on a stale or zeroed allocation, in another position's ring row);
#   * a staged ring that differs from the unsplit ring at ANY slot - the engine's reads are by position (i_widx: slot
#     q % RAW), so an identical ring is exactly the condition under which both shapes' queries read the same rows;
#   * a fix rule that COSTS work where it is not owed (the re-feed must be zero rows for a pass no wider than what the run
#     already published, and must never exceed the ring's width);
#   * ENGINE DRIFT: the gate greps the real ds41_dense.cpp for the arithmetic it models (the ring's width, the
#     position-indexed read, i_wslot's scope / allocation / publish rule, the graph's set_rows write-back through a view
#     whose extent does not clip the index, the NEG_INF mask, and relay_write_in's table refresh).  If a line moves, the
#     gate names the assumption that went stale instead of quietly proving a fiction.
#
# Usage: tools/gfx906/gate-relay-slot-collision.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=${TMPDIR:-/tmp}/test_relay_slot_collision.$$

echo "=== [1/2] compile the standalone gate (plain g++, C++17, warnings are errors) ==="
g++ -std=c++17 -O0 -Wall -Wextra -Werror -o "$OUT" tools/gfx906/test_relay_slot_collision.cpp || {
    echo "GATE FAIL: compile"; exit 1; }

echo "=== [2/2] run it (gated on the REAL ds41_dense.cpp for the arithmetic it models) ==="
"$OUT" --real tools/v41/engine/ds41_dense.cpp; rc=$?
rm -f "$OUT"
[ $rc -eq 0 ] || { echo "GATE FAIL: test_relay_slot_collision (rc=$rc)"; exit 1; }
echo "gate-relay-slot-collision: OK"
