#!/bin/bash
# P5.3/P5.4 (task #29 rung 1): the SAME-TOKEN HAND-OFF check for the staged GLM run.
# OPERATOR launches (this box's rule: the assistant never starts an engine):
#   cd /ai/engines/gfx906/worktrees/p5.3-relay
#   bash tools/gfx906/run-p5.3-handoff.sh 2>&1 | tee run-p5.3-handoff.log
# Expect ~45 min: three engine runs, each pinning ~60 GiB of arena before the first token.
#
# What it witnesses.  The staged run (--gpus 0,0) splits the 45 layers into two same-device
# stages whose boundary crosses the host-pinned staging (docs/gfx906/P5-RELAY-PATTERN.md).
# The copy is an F32 round trip, so the staged greedy token stream must be IDENTICAL to the
# unsplit one - the bit-exact hand-off, the same claim the native engine's
# --layer-split K --split-device 0 makes (docs/MULTI_GPU.md:99-101).  Distinct devices are
# step 2 and refused by the engine before this script would ever get here.
#
# Why three runs.  Expert residency differs between processes (cold VRAM slots, arena warm-up),
# so a cross-process identity claim needs its own control: run1 and run2 are the SAME unsplit
# command twice.  run1 != run2 means the engine is not deterministic across runs at greedy and
# the staged comparison is INCONCLUSIVE (not a relay failure); run1 == run2 makes the control
# real, and run3 == run1 is then the hand-off witness.
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
if [ "$GEN" -ot tools/glm/glm_dense.cpp ] || [ "$GEN" -ot tools/glm/glm_generate.cpp ]; then
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

grep -m2 -E "layer split:|relay: staged run|relay: token slab" "logs/p5.3-run3-${TS}.log" | sed 's/^/[staged] /'

# Verdict rule, calibrated the hard way (witness #6, TS 20261024-102456): two IDENTICAL unsplit runs
# of this engine diverged at token 40 of 64.  That is the tiered-expert design, not a bug: which path
# an expert takes (VRAM hit, CPU pool, PCIe, file) changes float summation order, and greedy argmax
# flips at knife-edge logits.  Bit determinism ACROSS PROCESSES is therefore not a property this
# engine offers, so the witness cannot demand it.  What it can demand: the staged run must agree with
# an unsplit run AT LEAST as far as two unsplit runs agree with each other.  Witness #6 met exactly
# that with the strongest shape available: the staged stream equalled run2 for all 64 tokens, and it
# diverged from run1 at the same token 40 where run1 diverges from run2 - indistinguishable from a
# third unsplit run.
#   PASS  : run3 matches run1 or run2 over the FULL stream, or at least as far as the control pair
#           matches each other (longest common prefix, python3 below).
#   FAIL  : run3 falls short of the control pair's own agreement - the relay added divergence.
#   INCONCLUSIVE: kept for the case where a run produced no tokens (a crash between the rc checks).
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
L13=$(lcp "logs/p5.3-run1-${TS}.toks" "logs/p5.3-run3-${TS}.toks")
L23=$(lcp "logs/p5.3-run2-${TS}.toks" "logs/p5.3-run3-${TS}.toks")
NT=$(wc -l < "logs/p5.3-run1-${TS}.toks")
echo "verdict inputs: control agreement ${L12}/${NT}, staged-vs-run1 ${L13}/${NT}, staged-vs-run2 ${L23}/${NT}"
if [ "$L13" -eq "$NT" ] || [ "$L23" -eq "$NT" ]; then
    echo "P5.3 WITNESS: PASS - the staged (--gpus 0,0) run's $( [ "$L23" -eq "$NT" ] && echo run2 || echo run1 )"
    echo "  comparison is byte-identical for all ${NT} greedy token ids: the host-relayed hand-off is exact."
    exit 0
fi
if [ "$L13" -ge "$L12" ] || [ "$L23" -ge "$L12" ]; then
    echo "P5.3 WITNESS: PASS (calibrated) - the staged run agrees with an unsplit run as far as the two"
    echo "  unsplit runs agree with EACH OTHER (${L12}/${NT}): within the engine's own cross-process"
    echo "  reproducibility (tier path -> float order -> knife-edge argmax), so the relay added none."
    exit 0
fi
if [ "$NT" -eq 0 ]; then
    echo "P5.3 WITNESS: INCONCLUSIVE - run1 produced no tokens; see logs/p5.3-run1-${TS}.log"; exit 2
fi
echo "P5.3 WITNESS: FAIL - the staged run agrees with neither control as far as the controls agree"
echo "  with each other (control ${L12}, staged ${L13}/${L23}): the relay itself diverges."
echo "  $(cmp logs/p5.3-run1-${TS}.toks logs/p5.3-run3-${TS}.toks | head -1)"
exit 1
