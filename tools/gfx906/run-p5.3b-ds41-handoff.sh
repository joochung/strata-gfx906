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
# host-pinned staging (docs/gfx906/P5-RELAY-PATTERN.md, implemented in ds41_dense.cpp's relay_read_out /
# relay_write_in and ds41_generate.cpp's staged_pass).  DS4.1 relays THREE regions per token, not one: x_state
# (D x HC F32 = 80 KiB), pre_state (HC F32 = 16 B) and - when the boundary separates layer 20 from the indexers that
# rank inside its blocks - the candidate blocks (CB I32 = 8 KiB).
#   run1 / run2   unsplit, twice: this engine's cross-process reproducibility, measured, not assumed.
#   run3          --gpus 0,0 --layer-split 21: two stages on ONE card, sharing the owner's weights, with the
#                 candidate column borrowed in place (no copy).  The P5.3b step-1 hand-off check.
#   run4 / run5   --gpus 0,1 --layer-split 21 --relay-copy-weights, twice: stage 1 on the second card with its OWN
#                 weights and state, the same bytes over the same host seam.  The P5.3b step-2 cross-device check,
#                 with a twin because a cross-card run is judged against its own variant's agreement.
#
# WHY THE BOUNDARY IS 21 AND NOT, SAY, 15.  Two geometry rules, both enforced by the engine (ds41_dense.cpp) and
# restated for a host-only gate in tools/gfx906/test_split_geometry.hpp:
#   THE KV RULE    - the layer BEFORE a boundary must produce both the compressed KV and the index keys its
#                    successor reads.  For this model that allows a boundary after layers 2, 8, 14, 20 (and after 20
#                    the successor's sources are its own), so --layer-split 21 is legal and 16/19/25 are refused.
#   THE CANDIDATE  - layer 20 (cand_layer) picks the 2048 blocks the indexers of 24/28/32/36 rank inside.  A boundary
#                    at 21 puts the producer alone in stage 0, so exactly one boundary crosses the column: relayable
#                    across cards (run4/run5), borrowable in place on one card (run3).  Anything farther strands the
#                    producer and the engine refuses the plan.
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

# The boundary the split is built around, in ONE place: run3/4/5 all use it, and the engine refuses anything else
# (see the two rules above).  If the model's geometry ever changes, this number and test_split_geometry.hpp move.
# = cand_layer + 1: the boundary BOTH rules allow AND that separates layer 20's candidate blocks from the indexers
# above them, so the run exercises the DS4.1-specific third region of the payload.
SPLIT=21
ARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu
      --engram "$M/engram/l1:$M/engram/l14" --rank "$M/rank.bin"
      --ids-file "$IDS" -n 64 --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0
      --prefill-chunk 64 --relay-chunk 64)   # --prompt-batch stays at the engine's 8192: a staged run takes its
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
if ! run_step run3 "${ENG[@]}" "${ARGS[@]}" --gpus 0,0 --layer-split "$SPLIT"; then
    echo "P5.3b WITNESS: FAIL at run3 (staged --gpus 0,0, one card) - log logs/p5.3b-run3-${TS}.log"; exit 1
fi
if ! run_step run4 "${ENG[@]}" "${ARGS[@]}" --gpus 0,1 --layer-split "$SPLIT" --relay-copy-weights; then
    echo "P5.3b WITNESS: FAIL at run4 (staged --gpus 0,1, cross-device) - log logs/p5.3b-run4-${TS}.log"; exit 1
fi
if ! run_step run5 "${ENG[@]}" "${ARGS[@]}" --gpus 0,1 --layer-split "$SPLIT" --relay-copy-weights; then
    echo "P5.3b WITNESS: FAIL at run5 (cross-device twin) - log logs/p5.3b-run5-${TS}.log"; exit 1
fi

# The staged runs must say WHAT they did; a run that silently fell back to the single-instance path is not a witness.
grep -m4 -E "relay: staged run|relay: layer 20|relay: [0-9]+ state columns|relay: [0-9]+ passes" \
    "logs/p5.3b-run3-${TS}.log" | sed 's/^/[run3 0,0] /'
for r in run4 run5; do
    grep -m4 -E "relay: staged run|relay: layer 20|weights own copy|relay: [0-9]+ passes" \
        "logs/p5.3b-${r}-${TS}.log" | sed "s/^/[${r} 0,1] /"
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
# A one-card split must BORROW layer 20's candidate column (no copy); a two-card split must RELAY it.  If either
# prints the other's line, the slab the staging was sized from is the wrong one.
if grep -q "are read in place by" "logs/p5.3b-run3-${TS}.log"; then
    echo "[run3] candidates: borrowed in place (as a same-card split must)"
else
    echo "P5.3b WITNESS: FAIL - run3 did not report borrowing layer 20's candidates"; exit 1
fi
for r in run4 run5; do
    if grep -q "are relayed to" "logs/p5.3b-${r}-${TS}.log"; then
        echo "[${r}] candidates: relayed across the boundary (as a cross-card split must)"
    else
        echo "P5.3b WITNESS: FAIL - ${r} did not report relaying layer 20's candidates"; exit 1
    fi
done

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
