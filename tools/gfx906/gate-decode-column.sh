#!/usr/bin/env bash
# gate-decode-column.sh - P5.3b host gate: does the FIRST DECODE pass land on a different STATE COLUMN in a staged run
# than in an unsplit one, at the geometry witness #7 ran (prompt 14 tokens, relay unit 64)?
#
# Plain g++, no GPU, no model, seconds.  Prints both rules' column for every reachable position:
#   UNSPLIT  Ds41Dense::Impl::begin_impl leaves im.col0 = 0 for every pass ("Unsplit, this is pos0 at column 0").
#   STAGED   begin_impl sets im.col0 = stage_col0(pos0, n) = (pos0 / S) % (NC / S) * S - the columns CYCLE by window.
#
# PASSES when the two are equal for every position the witness's run can reach and only part company beyond it, which is
# what rules the decode column out as the token-2 seam and bounds any future claim about it.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=${TMPDIR:-/tmp}/test_decode_column.$$
g++ -std=c++17 -O0 -Wall -Wextra -Werror -o "$OUT" tools/gfx906/test_decode_column.cpp || { echo "GATE FAIL: compile"; exit 1; }
"$OUT"; rc=$?
rm -f "$OUT"
[ $rc -eq 0 ] || { echo "GATE FAIL: test_decode_column (rc=$rc)"; exit 1; }
echo "gate-decode-column: OK"
