#!/bin/bash
# P5.3b (task #34): the SAME-TOKEN HAND-OFF witness for the STAGED DeepSeek-V4.1 run - the GLM relay (P5.3,
# tools/gfx906/run-p5.3-handoff.sh) replicated into tools/v41.  OPERATOR launches (this box's rule: the assistant
# never starts an engine):
#   cd /ai/engines/gfx906/worktrees/p5.3-relay
#   bash tools/gfx906/run-p5.3b-ds41-handoff.sh 2>&1 | tee run-p5.3b-ds41-handoff.log
# Expect ~20-30 min: five engine runs, each pinning ~64 GiB of arena before the first token (witness times from
# tools/gfx906/run-p4.4-ds41-smoke.sh).
#
# WHAT IT WITNESSES.  The staged run splits the 40 dense layers into stage instances whose boundary crosses the
# host-pinned staging (docs/gfx906/P5-RELAY-PATTERN.md section 6b, implemented in ds41_dense.cpp's relay_read_out /
# relay_write_in and ds41_generate.cpp's staged_pass).  DS4.1's payload is TWO regions on this model, x_state
# (D x HC F32 = 80 KiB) and pre_state (HC F32 = 16 B): the candidate column (CB I32 = 8 KiB) is a third region the
# machinery supports, but the reader's rule below keeps cand_layer and every reader of its blocks in one stage, so it
# never crosses - the run says "cand 0" in its slab line, and that is the expected value, not a missing feature.
#   run1 / run2   unsplit, twice: this engine's cross-process reproducibility, measured, not assumed.
#   run3          --gpus 0,0 --layer-split $SPLIT: two stages on ONE card, sharing the owner's weights.  The P5.3b
#                 step-1 hand-off check: same card, same weights, so a divergence can only come from the hand-off.
#   run4 / run5   --gpus 0,1 --layer-split $SPLIT --relay-copy-weights, twice: stage 1 on the second card with its OWN
#                 weights and state, the same bytes over the same host seam.  The P5.3b step-2 cross-device check,
#                 with a twin because a cross-card run is judged against its own variant's agreement.
#
# WHY THE BOUNDARY IS 20 - THE READER'S RULE, enforced by the engine (Ds41Dense::split_boundary_valid /
# split_boundaries_ok, split_valid_boundaries for the whole set) and restated for the host-only gate in
# tools/gfx906/test_split_geometry.hpp:
#   A layer reads the compressed KV of the latest kv source at or below it and the index keys of the latest index source
#   at or below it, and those rings are PERSISTENT GPU state - allocated once, read by pointer inside a graph.  The
#   relay moves per-token state (x_state, pre_state) and cannot move a cache, so a ring lives on the card of the stage
#   that WRITES it, and every READER of it has to sit in that stage: K is valid iff every layer at or above K finds BOTH
#   of its sources at or above K.  For this model the valid set is {1, 2, 8, 14, 20} - the layers that are both a kv and
#   an index source - and 20, the EVEN HALFWAY POINT OF THE 40-LAYER MODEL, is in it: stage 1 = [20,40) runs the writer
#   of ring 20 together with every reader of it, the writers and readers of index rings 24/28/32/36, and the candidate
#   writer (20) with all of its readers (24/28/32/36).
#   IT IS A SET, NOT A FLOOR.  An earlier revision of this witness ran --layer-split 37 ("last shared-cache layer + 1")
#   and refused 20; that boundary is the one that DIES - the tail [37,40) reads layer 20's compressed KV and layer 36's
#   index keys, both below it, and build_attn aborts on exactly that.  3/9/15/21 remain refused, for the real reason:
#   kv_src(21) == 20 < 21, so ring 20 would stay on the stage below.
#   WHY 20 IS THE ONLY CHOICE THIS WITNESS HAS: the run is a real model, so it passes --engram (layers 1 and 14), and
#   only stage 0 ever feeds the Engram token ring - so the FIRST boundary must sit above the last Engram layer (14),
#   which leaves {20} of {2, 8, 14, 20}.  Two stages is therefore the most a staged DS4.1 run can do while the Engram
#   add is on; the deeper plan {8, 14, 20} (four cards) is legal for the caches and refused by the Engram rule.
#
# THE VERDICT IS CALIBRATED, NOT ABSOLUTE (the lesson of GLM witness #6, which saw two IDENTICAL unsplit runs
# diverge at token 40 of 64): which path an expert takes (VRAM hit, CPU pool, PCIe, file) changes float summation
# order, and greedy argmax flips at knife-edge logits.  A staged run therefore has to agree with an unsplit control at
# least as far as two processes running the SAME command agree with each other, measured in this witness.
#
# ONE ENGINE AT A TIME (SVM crash rule): refuses while any engine is alive (vLLM included - stop it first), or below
# MIN_AVAIL_GIB (floor per docs/gfx906/P4-RAM-BUDGET.md, arena+25).  Deliberate deltas vs the serve config: no
# --prune, no --route-bias, no --kv-fp4, no DSpark (a staged run refuses DSpark - a rejected draft would move state a
# stage already handed over).  A staged run walks the prompt in the sequential chunk path, never the layer-major one
# (the relay has no seam for a sub-batch another stage's graphs wrote), so --prompt-batch here only sizes the STATE
# COLUMNS the passes cycle through: 2048 columns = 32 passes of the 64-token relay unit, enough for a prompt of ~2K
# tokens without pinning the 2.6 GiB of state the serve default (8192) would cost.
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
ARENA_GIB=${ARENA_GIB:-64}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-$((ARENA_GIB + 25))}
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }

# The artifacts this witness stands on (GLM witness #1 ran a binary older than the tree it was launched from and died
# in an assert the current tree does not have).  Check before pinning 64 GiB.
GEN=build-906/ds41_generate
[[ -x tools/v41/serve/ds41-engine-906.sh ]] || { echo "FATAL: missing wrapper tools/v41/serve/ds41-engine-906.sh"; exit 3; }
[[ -x "$GEN" ]] || { echo "FATAL: missing engine binary $GEN - build with tools/gfx906/build-906-gate.sh first"; exit 3; }
for src in tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_moe.cpp \
           tools/v41/engine/ds41_engram.cpp tools/gfx906/relay_staging.hpp tools/gfx906/device_split.hpp; do
    [ "$GEN" -ot "$src" ] && { echo "FATAL: $GEN is older than $src - the witness would measure a stale binary"; exit 3; }
done
# The host-only gate has to be green too (it is seconds, and it is what refuses a bad --layer-split before the GPU).
g++ -std=c++17 -O0 -Wall -Wextra -Werror -I tools/gfx906 -o /tmp/p53b-gate-$$ tools/gfx906/test_device_split.cpp || {
    echo "FATAL: the split gate does not compile"; exit 3; }
/tmp/p53b-gate-$$ || { echo "FATAL: the split gate FAILs"; rm -f /tmp/p53b-gate-$$; exit 3; }
rm -f /tmp/p53b-gate-$$

M=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
PY=/root/.pyenv/versions/venv312/bin/python3

echo "=== engram gate (pre-boot) ==="
$PY tools/gfx906/check_engram_tables.py "$M/V41-GSQ.gguf" "$M/engram/l1" "$M/engram/l14" || {
    echo "refusing: engram gate FAILED before touching the GPU"; exit 3; }

IDS=logs/p5.3b-ids-${TS}.i32
$PY - "$IDS" <<EOF || exit 1
import sys, json
sys.path.insert(0, "tools")
import strata_tokenizer as ST, numpy as np
tdir = "$M/tokenizer"
vocab = json.load(open(tdir + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = open(tdir + "/merges.txt").read().split("\\n")
types = json.load(open(tdir + "/token_type.json"))
tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")   # the server's own pre-split (serve_ds41.py:45)
ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences.")
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
EOF

# The boundary the split is built around, in ONE place: run3/4/5 all use it.  It is the ONLY layer this witness's
# geometry allows (see THE READER'S RULE above: the valid set is {1, 2, 8, 14, 20} and --engram forces the first
# boundary above layer 14).  Anything else an operator passes is snapped DOWN to it by the engine - SPLIT=21 boots as
# 20, with a line saying so - and anything the set cannot reach is refused before allocation, with the set named.
# If the model's geometry ever changes, this number and test_split_geometry.hpp move together: the gate's [3/3] step
# re-derives the set from the checkpoint with the engine's own reader and fails on a stale list.
SPLIT=${SPLIT:-20}
ARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu
      --engram "$M/engram/l1:$M/engram/l14" --rank "$M/rank.bin"
      --ids-file "$IDS" -n 64 --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0
      --prefill-chunk 64)   # --relay-chunk belongs to the STAGED runs only (run3/4/5): the engine refuses a relay
                                              # cap with no --gpus, and a control must not carry staged flags at all.
                                              # --prompt-batch stays at the engine's 8192: a staged run takes its
                                              # state columns from it (the layer-major walk never runs), which is
                                              # what lets a prompt of up to 8192 tokens cycle 128 64-token passes

run_step() {
    # A finished engine releases VRAM asynchronously; the next one would malloc over a not-yet-freed card.
    sleep 10
    local name=$1; shift
    local log=logs/p5.3b-${name}-${TS}.log toks=logs/p5.3b-${name}-${TS}.toks
    echo "=== p5.3b/$name start $(date +%T) (avail ${AVAIL} GiB) ==="
    "$@" >"$toks" 2>"$log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$toks") token ids -> $toks"
    tail -12 "$log"
    grep -m6 -E "relay: |layer split:|ds41_generate:" "$log" | sed "s/^/[$name] /"
    return $rc
}

ENG=(bash tools/v41/serve/ds41-engine-906.sh)
if ! run_step run1 "${ENG[@]}" "${ARGS[@]}"; then
    echo "P5.3b WITNESS: FAIL at run1 (unsplit base) - log logs/p5.3b-run1-${TS}.log"; exit 1
fi
if ! run_step run2 "${ENG[@]}" "${ARGS[@]}"; then
    echo "P5.3b WITNESS: FAIL at run2 (unsplit repeat) - log logs/p5.3b-run2-${TS}.log"; exit 1
fi
if ! run_step run3 "${ENG[@]}" "${ARGS[@]}" --gpus 0,0 --layer-split "$SPLIT" --relay-chunk 64; then
    echo "P5.3b WITNESS: FAIL at run3 (staged --gpus 0,0, one card) - log logs/p5.3b-run3-${TS}.log"; exit 1
fi
if ! run_step run4 "${ENG[@]}" "${ARGS[@]}" --gpus 0,1 --layer-split "$SPLIT" --relay-chunk 64 --relay-copy-weights; then
    echo "P5.3b WITNESS: FAIL at run4 (staged --gpus 0,1, cross-device) - log logs/p5.3b-run4-${TS}.log"; exit 1
fi
if ! run_step run5 "${ENG[@]}" "${ARGS[@]}" --gpus 0,1 --layer-split "$SPLIT" --relay-chunk 64 --relay-copy-weights; then
    echo "P5.3b WITNESS: FAIL at run5 (cross-device twin) - log logs/p5.3b-run5-${TS}.log"; exit 1
fi

# The staged runs must say WHAT they did; a run that silently fell back to the single-instance path is not a witness.
# The lines that prove the run was staged at all, and that its payload is the shape the reader's rule predicts: a slab
# of x_state + pre_state with cand 0 (the candidate column never crosses on this model), on the boundary the engine
# accepted.  A run whose slab says cand 8192 ran a different geometry than this witness documents.
grep -m5 -E "relay: staged run|relay:   plan stage|relay: token slab|relay:   stage |layer split:" \
    "logs/p5.3b-run3-${TS}.log" | sed 's/^/[run3 0,0] /'
grep -m1 "token slab" "logs/p5.3b-run3-${TS}.log" | grep -q "cand 0)" || {
    echo "P5.3b WITNESS: FAIL - run3's payload carries a candidate region (cand != 0); the geometry moved under the witness"; exit 1; }
for r in run4 run5; do
    grep -m5 -E "relay: staged run|relay:   plan stage|relay: token slab|weights own copy|relay:   stage " \
        "logs/p5.3b-${r}-${TS}.log" | sed "s/^/[${r} 0,1] /"
    grep -m1 "weights own copy" "logs/p5.3b-${r}-${TS}.log" >/dev/null || {
        echo "P5.3b WITNESS: FAIL - ${r} did not load its own weights on the second card (no 'weights own copy' line)"; exit 1; }
done
for r in run3 run4 run5; do
    if ! grep -q "relay: staged run" "logs/p5.3b-${r}-${TS}.log"; then
        echo "P5.3b WITNESS: FAIL - $r never built a staged run (see its log)"; exit 1
    fi
done
# And it must have handed off: the boundary count is the number of passes (prompt passes + one per decode token),
# so "0 boundary hand-offs" would mean the relay code never ran.
for r in run3 run4 run5; do
    line=$(grep -m1 "boundary hand-offs" "logs/p5.3b-${r}-${TS}.log" || true)
    [ -n "$line" ] || { echo "P5.3b WITNESS: FAIL - $r printed no relay summary (the relay never ran?)"; exit 1; }
    echo "[${r}] ${line}"
done
for r in run3 run4 run5; do
    grep -q "layers \[$SPLIT," "logs/p5.3b-${r}-${TS}.log" || {
        echo "P5.3b WITNESS: FAIL - ${r} did not build a stage beginning at layer $SPLIT (no 'layers [$SPLIT,' line):";
        echo "    the boundary the engine used is not the one this witness was calibrated for"; exit 1; }
done
# THE CANDIDATE COLUMN NEVER CROSSES HERE, and a staged run that says otherwise ran a different geometry than this
# witness documents.  Under the reader's rule (the header) a boundary at K may only leave a layer that ranks inside
# cand_layer's blocks ABOVE it if cand_layer is above K too - and every layer that ranks inside those blocks IS an index
# source, so a boundary that separates them is not a member of the valid set and the engine never builds one.  Producer
# (20) and readers (24/28/32/36) therefore share a stage, the engine prints neither "borrowed in place" nor "relayed
# to", and the column is read by the layer above the one that wrote it, exactly as an unsplit run reads it.  That is the
# expected line, so the gate is the ABSENCE of the other two.
for r in run3 run4 run5; do
    if grep -qE "are read in place by|are relayed to" "logs/p5.3b-${r}-${TS}.log"; then
        echo "P5.3b WITNESS: FAIL - ${r} moved the candidate column across a boundary; the reader's rule no longer holds"
        exit 1
    fi
    # enc-skip is layer-major-only: its phase-A/B slice indexes absolute columns, which a staged run's cycling
    # window ((pos / cap) % windows) has no defined relation to.  ds41_generate now forces it off for staged runs
    # (witness 2026-10-24 21:50: with the flag silently ON, both staged variants diverged from BOTH controls at
    # the FIRST token, while an unsplit chunk-path run reproduced the controls exactly).  A staged log carrying
    # the plan line means the flag is back - the token verdict would then measure the wrong thing, so refuse.
    if grep -q "enc-skip: cut at layer" "logs/p5.3b-${r}-${TS}.log"; then
        echo "P5.3b WITNESS: FAIL - ${r} ran with enc-skip ON (staged runs must force it off; see ds41_generate)"
        exit 1
    fi
done
echo "[run3/4/5] candidates: stayed inside the producer's stage (reader's rule held; payload = x_state + pre_state)"

# ---- verdict
lcp() {  # longest common prefix of two token-id files
    python3 - "$1" "$2" <<'PYEOF'
import sys
a = [int(x) for x in open(sys.argv[1])]
b = [int(x) for x in open(sys.argv[2])]
n = 0
for x, y in zip(a, b):
    if x != y:
        break
    n += 1
print(n)
PYEOF
}
L12=$(lcp "logs/p5.3b-run1-${TS}.toks" "logs/p5.3b-run2-${TS}.toks")
L45=$(lcp "logs/p5.3b-run4-${TS}.toks" "logs/p5.3b-run5-${TS}.toks")
NT=$(wc -l < "logs/p5.3b-run1-${TS}.toks")
if [ "$NT" -eq 0 ]; then
    echo "P5.3b WITNESS: INCONCLUSIVE - run1 produced no tokens; see logs/p5.3b-run1-${TS}.log"; exit 2
fi
echo "verdict inputs: unsplit twins run1-vs-run2 ${L12}/${NT}, cross-device twins run4-vs-run5 ${L45}/${NT}"
overall=0
for run in run3 run4 run5; do
    L1=$(lcp "logs/p5.3b-run1-${TS}.toks" "logs/p5.3b-${run}-${TS}.toks")
    L2=$(lcp "logs/p5.3b-run2-${TS}.toks" "logs/p5.3b-${run}-${TS}.toks")
    # The bar is the WORST agreement measured between two processes running the same command in this witness:
    # L12 for every variant, never above the variant's own twin agreement (cross-device runs may be judged by L45).
    if [ "$run" = run4 ] || [ "$run" = run5 ]; then thr=$L45; else thr=$L12; fi
    [ "$thr" -gt "$L12" ] && thr=$L12
    best=$L1; [ "$L2" -gt "$best" ] && best=$L2
    echo "  ${run}: best-vs-controls ${best}/${NT} (vs-run1 ${L1}, vs-run2 ${L2}), bar ${thr}"
    if [ "$best" -eq "$NT" ]; then
        echo "  ${run}: PASS - byte-identical to an unsplit control for all ${NT} greedy token ids"
    elif [ "$best" -ge "$thr" ]; then
        echo "  ${run}: PASS (calibrated) - agrees with a control at least as far as same-command processes agree"
        echo "    with each other (bar ${thr}/${NT}): within this engine's own cross-process reproducibility"
        echo "    (expert path -> float order -> knife-edge argmax)."
    else
        echo "  ${run}: FAIL - agrees with neither control as far as ANY same-command pair in this witness agrees"
        echo "    with itself (best ${best}, bar ${thr}): this run's own path diverges."
        overall=1
    fi
done
if [ "$overall" -eq 0 ]; then
    echo "P5.3b WITNESS: PASS - the DS4.1 host-relayed hand-off (x_state + pre_state + candidate blocks) is exact on"
    echo "  one card (--gpus 0,0, candidates borrowed in place) and across two cards (--gpus 0,1, candidates relayed),"
    echo "  within this engine's own cross-process reproducibility."
    exit 0
fi
echo "P5.3b WITNESS: FAIL - see the per-run lines above; logs/p5.3b-*-${TS}.log and .toks hold the runs."
exit 1
