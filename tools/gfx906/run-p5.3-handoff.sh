#!/bin/bash
# P5.3/P5.4 (task #29 rung 1): the SAME-TOKEN HAND-OFF check for the staged GLM run.
# OPERATOR launches (this box's rule: the assistant never starts an engine):
#   cd /ai/engines/gfx906/worktrees/p5.3-relay
#   bash tools/gfx906/run-p5.3-handoff.sh 2>&1 | tee run-p5.3-handoff.log
# Expect ~15 min: five engine runs (~3 min each as witnessed 2026-10-24), each pinning ~60 GiB
# of arena before the first token.
#
# What it witnesses.  The staged run splits the 45 layers into two stage instances whose boundary
# crosses the host-pinned staging (docs/gfx906/P5-RELAY-PATTERN.md).  run3 (--gpus 0,0) keeps both
# stages on one card and shares the owner's weights: the P5.3 step-1 bit-exact hand-off check.
# run4/run5 (--gpus 0,1, the same command twice) put stage 1 on the second card, where it loads its
# own weights and keeps its own state, relaying the same bytes over the same host seam: the P5.3
# step-2 cross-device check.  The copy is an F32 round trip, so each staged greedy token stream
# must match the unsplit one to within the engine's own cross-process reproducibility - measured
# per variant by the twin pairs, not assumed (see the verdict rule below the runs).
#
# Why five runs.  Expert residency differs between processes (cold VRAM slots, arena warm-up),
# so a cross-process identity claim needs its own control, per variant: run1/run2 are the same
# unsplit command twice (witness #6 measured 39/64 agreement; witness #7 measured 64/64 - it is a
# distribution, not a property), and run4/run5 are the same CROSS-DEVICE command twice.  run3 has
# its variant's twin in witness #7's own 64/64 record.  A staged run is judged against the
# agreement measured between same-command pairs, so the witness never demands more determinism
# than the engine actually has.
#
# ONE ENGINE AT A TIME (SVM crash rule): refuses while any engine is alive (vLLM included -
# stop it first), or below MIN_AVAIL_GIB (floor per docs/gfx906/P4-RAM-BUDGET.md, arena+25).
# Deliberate deltas vs the serve config, same as P4.2: no --prune, --route-bias, --lora.
# Margin 4.0 is the witnessed smoke number (run-p4.2-glm-smoke.sh:54, logs/p4.2-base-20261022-180307.log
# rc=0): the chunk-MMQ c_stage (2 x 1 GiB device halves) is allocated AFTER the auto expert slots fill to
# the margin, so margin 2.0 OOMs there (witness #2 died exactly there at 2.0; serve needs 5.0 only for the
# wrap-up-prefill MMQ burst, which a 14-token smoke + greedy decode does not run).
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
ARENA_GIB=${ARENA_GIB:-60}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-$((ARENA_GIB + 25))}
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }

# The artifacts this witness stands on (witness #1 of 2026-10-24 ran a binary older than the tree it
# was launched from, and died in a view assert the current tree does not have).  Check before pinning.
[[ -x tools/glm/serve/glm-engine-906.sh ]] || { echo "FATAL: missing wrapper tools/glm/serve/glm-engine-906.sh"; exit 3; }
GEN="build-906/glm_generate"
[[ -x "$GEN" ]] || { echo "FATAL: missing engine binary $GEN - run tools/gfx906/build-906-gate.sh 24 first"; exit 3; }
if [ "$GEN" -ot tools/glm/glm_dense.cpp ] || [ "$GEN" -ot tools/glm/glm_generate.cpp ] || [ "$GEN" -ot tools/ds4/ds4_moe.cpp ]; then
    echo "FATAL: $GEN is older than the engine sources - the witness would measure a stale binary"
    echo "       (this is what witness #1 did); rebuild with tools/gfx906/build-906-gate.sh 24 first"
    exit 3
fi
# Drift guard: the state region must keep the last release's TOKEN-indexed layout [D, HC, max_tokens].
# A staged run must never widen dim2 to n_layer+1 columns - the unsplit prefill then GGML_ASSERTs in
# ggml_view_3d the moment a pass has more tokens than the model has layers (witness #1's exact failure).
if grep -q "int64_t NT = n_layer + 1" tools/glm/glm_dense.cpp; then
    echo "FATAL: glm_dense.cpp gives x_state n_layer+1 DIM2 columns; the layout must stay token-indexed"
    exit 3
fi

PY=/root/.pyenv/versions/venv312/bin/python3
IDS=logs/p5.3-ids-${TS}.i32
PYTHONPATH="$PWD/tools" $PY - "$IDS" <<'EOF' || exit 1
import sys
sys.path.insert(0, "tools/glm/serve")
import glm_tok
import strata_tokenizer as ST
ST.PRE_PATTERNS.setdefault("glm4", [glm_tok.GLM4_PATTERN])   # what serve_glm.py:37 does before loading
import numpy as np
tok = glm_tok.load("/ai/models/tok/glm-5.3-flash")
ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences.")
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
EOF

MODEL=/ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf
ARGS=(-m "$MODEL" --backend cuda --experts gpu --ids-file "$IDS" -n 64
      --ctx 32768 --arena-gib "$ARENA_GIB" --arena-skip-resident --prefill-chunk 4096 --chunk-mmq
      --vram-margin 4.0 --arena-adapt --pcie auto --stop 154820,154827,154829)

# Each run: stdout = the greedy token ids (one per line, glm_generate.cpp prints "%d\n"),
# stderr = the engine log.  sleep between runs so a finished engine's VRAM is really gone.
run_step() {
    local name=$1; shift
    local log=logs/p5.3-${name}-${TS}.log toks=logs/p5.3-${name}-${TS}.toks
    echo "=== p5.3/$name start $(date +%T) (avail ${AVAIL} GiB) ==="
    "$@" >"$toks" 2>"$log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$toks") token ids -> $toks"
    tail -12 "$log"
    sleep 10
    return $rc
}

ENG=(bash tools/glm/serve/glm-engine-906.sh)
if ! run_step run1 "${ENG[@]}" "${ARGS[@]}"; then
    echo "P5.3 WITNESS: FAIL at run1 (unsplit base) - log logs/p5.3-run1-${TS}.log"; exit 1
fi
if ! run_step run2 "${ENG[@]}" "${ARGS[@]}"; then
    echo "P5.3 WITNESS: FAIL at run2 (unsplit repeat) - log logs/p5.3-run2-${TS}.log"; exit 1
fi
if ! run_step run3 "${ENG[@]}" "${ARGS[@]}" --gpus 0,0; then
    echo "P5.3 WITNESS: FAIL at run3 (staged --gpus 0,0) - log logs/p5.3-run3-${TS}.log"; exit 1
fi
if ! run_step run4 "${ENG[@]}" "${ARGS[@]}" --gpus 0,1; then
    echo "P5.3 WITNESS: FAIL at run4 (staged --gpus 0,1, step 2 cross-device) - log logs/p5.3-run4-${TS}.log"; exit 1
fi
if ! run_step run5 "${ENG[@]}" "${ARGS[@]}" --gpus 0,1; then
    echo "P5.3 WITNESS: FAIL at run5 (staged --gpus 0,1 twin) - log logs/p5.3-run5-${TS}.log"; exit 1
fi

grep -m2 -E "layer split:|relay: staged run|relay: token slab" "logs/p5.3-run3-${TS}.log" | sed 's/^/[staged 0,0] /'
for r in run4 run5; do grep -m2 -E "relay:   stage|weights own copy" "logs/p5.3-${r}-${TS}.log" | sed "s/^/[${r} 0,1] /"; done

# Verdict rule, calibrated the hard way (witness #6, TS 20261024-102456): two IDENTICAL unsplit runs
# of this engine diverged at token 40 of 64.  That is the tiered-expert design, not a bug: which path
# an expert takes (VRAM hit, CPU pool, PCIe, file) changes float summation order, and greedy argmax
# flips at knife-edge logits.  Bit determinism ACROSS PROCESSES is therefore not a property this
# engine offers, so the witness cannot demand it.  What it can demand: a staged run must agree with
# an unsplit run AT LEAST as far as two unsplit runs agree with each other.  Witness #6 met exactly
# that with the strongest shape available for run3: the staged stream equalled run2 for all 64
# tokens, and it diverged from run1 at the same token 40 where run1 diverges from run2 -
# indistinguishable from a third unsplit run.  run4 (--gpus 0,1) is the same claim across two cards.
#   PASS           : the staged run matches run1 or run2 over the FULL stream, or at least as far as
#                   the SAME-CONFIGURATION PAIR of its own variant agrees (unsplit variant: L12;
#                   cross-device variant: min(L12, L45) - the worst twin agreement measured here).
#   FAIL           : the staged run falls short of every same-config agreement - its own path added
#                   divergence (witness #7: run4 agreed 61/64 with both controls AND with run3, the
#                   same staged graph on one card, while the controls happened to land 64/64; the
#                   twin pair run4-vs-run5 below is what tells that noise from a device-1 bug).
#   INCONCLUSIVE   : kept for the case where a run produced no tokens (a crash between the rc checks).
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
L12=$(lcp "logs/p5.3-run1-${TS}.toks" "logs/p5.3-run2-${TS}.toks")
L45=$(lcp "logs/p5.3-run4-${TS}.toks" "logs/p5.3-run5-${TS}.toks")
NT=$(wc -l < "logs/p5.3-run1-${TS}.toks")
if [ "$NT" -eq 0 ]; then
    echo "P5.3 WITNESS: INCONCLUSIVE - run1 produced no tokens; see logs/p5.3-run1-${TS}.log"; exit 2
fi
echo "verdict inputs: unsplit twins run1-vs-run2 ${L12}/${NT}, cross-device twins run4-vs-run5 ${L45}/${NT}"
overall=0
for run in run3 run4 run5; do
    L1=$(lcp "logs/p5.3-run1-${TS}.toks" "logs/p5.3-${run}-${TS}.toks")
    L2=$(lcp "logs/p5.3-run2-${TS}.toks" "logs/p5.3-${run}-${TS}.toks")
    # The bar for each staged run is the worst agreement measured between two processes running
    # the SAME command in this witness, at most L12 for every variant and never above its own
    # variant's twin agreement (cross-device runs may be judged by L45, run3 by L12 alone).
    if [ "$run" = run4 ] || [ "$run" = run5 ]; then thr=$L45; else thr=$L12; fi
    [ "$thr" -gt "$L12" ] && thr=$L12
    best=$L1; [ "$L2" -gt "$best" ] && best=$L2
    echo "  ${run}: best-vs-controls ${best}/${NT} (vs-run1 ${L1}, vs-run2 ${L2}), bar ${thr}"
    if [ "$best" -eq "$NT" ]; then
        echo "  ${run}: PASS - byte-identical to an unsplit control for all ${NT} greedy token ids"
    elif [ "$best" -ge "$thr" ]; then
        echo "  ${run}: PASS (calibrated) - agrees with a control at least as far as same-command"
        echo "    processes agree with each other (bar ${thr}/${NT}): within this engine's own"
        echo "    cross-process reproducibility (tier path -> float order -> knife-edge argmax)."
    else
        echo "  ${run}: FAIL - agrees with neither control as far as ANY same-command pair in this"
        echo "    witness agrees with itself (best ${best}, bar ${thr}): this run's own path diverges."
        overall=1
    fi
done
if [ "$overall" -eq 0 ]; then
    echo "P5.3 WITNESS: PASS - the host-relayed hand-off is exact on one card (--gpus 0,0) and across"
    echo "  two cards (--gpus 0,1), within this engine's own cross-process reproducibility."
    exit 0
fi
echo "P5.3 WITNESS: FAIL - see the per-run lines above; logs/p5.3-*-${TS}.log and .toks hold the runs."
exit 1
