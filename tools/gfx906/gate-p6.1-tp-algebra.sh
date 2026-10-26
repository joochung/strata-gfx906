#!/usr/bin/env bash
# gate-p6.1-tp-algebra.sh - host-only gate for the P6.1 sharding-algebra model (task #36).
# Builds tools/gfx906/p6_tp_algebra.cpp with plain g++ (no ggml, no GPU, no cmake) and
# asserts: the clean plan PASSES, and EACH red control turns it RED on its own. A red that
# does not turn the gate red is a gate failure: it means the invariant does not actually
# watch that defect class (the lesson of gates that encode the same misreading as the code).
set -u
cd "$(dirname "$0")/../.." || exit 1
BIN=build-906/p6_tp_algebra
mkdir -p build-906
g++ -O2 -std=c++17 -o "$BIN" tools/gfx906/p6_tp_algebra.cpp || { echo "GATE FAIL: compile"; exit 1; }

fails=0

out=$("$BIN"); rc=$?
echo "$out" | tail -3
if [ $rc -ne 0 ] || ! echo "$out" | grep -q "P6.1 GATE: PASS"; then
    echo "GATE FAIL: clean run did not pass (rc=$rc)"; fails=$((fails+1));
fi
# the error table must land (it is the host-number provenance for later calibration)
for needle in "attn_output    f32" "attn_output    f16" "attn_output    q8" \
              "shared_ffn     f32" "shared_ffn     f16" "shared_ffn     q8"; do
    echo "$out" | grep -q "$needle" || { echo "GATE FAIL: error table missing row: $needle"; fails=$((fails+1)); }
done
# refusal of unplaceable widths must be visible
echo "$out" | grep -q "usable TP widths" || { echo "GATE FAIL: width refusal note missing"; fails=$((fails+1)); }

# each red control ALONE must turn the gate red, with its invariant named
declare -A NEEDLE=(
    [flip-qb-direction]="partials cannot line up"
    [straddle-ogroup]="shard OVERLAP"
    [drop-replicate]="not the full span"
    [dtype-rounding]="truncation"
)
for r in flip-qb-direction straddle-ogroup drop-replicate dtype-rounding; do
    out=$("$BIN" --red="$r"); rc=$?
    if [ $rc -eq 0 ]; then
        echo "GATE FAIL: red control $r left the gate GREEN (invariant is blind to it)"
        fails=$((fails+1)); continue
    fi
    if ! echo "$out" | grep -qF -- "${NEEDLE[$r]}"; then
        echo "GATE FAIL: red control $r exited $rc but did not name its invariant (${NEEDLE[$r]})"
        fails=$((fails+1)); continue
    fi
    echo "red control $r: RED as required ($(echo "$out" | grep -c FAIL) failures named)"
done

if [ $fails -eq 0 ]; then
    echo "P6.1 GATE SCRIPT: PASS (clean green + 4 independent reds + 6 error-table rows)"
    exit 0
fi
echo "P6.1 GATE SCRIPT: FAIL ($fails problems)"
exit 1
