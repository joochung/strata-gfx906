#!/bin/bash
# P5.4 (task #29): the 2-then-4-card SPLIT SMOKE for the ported engines.  Launch through the GPU wrapper
# (it stops the stack, runs, and restarts vLLM on exit):
#   /ai/engines/gfx906/unverbraucht/run-with-free-gpus.sh bash -c \
#     "cd /ai/engines/gfx906/worktrees/p5.3-relay && bash tools/gfx906/run-p5.4-split-smoke.sh"
# Expect ~30 min: seven engine runs (four GLM ~3 min, three DS4.1 ~3.5 min), plus 10 s driver settles.
#
# WHAT IS NEW HERE (the 2-card rungs were already witnessed and are NOT repeated):
#   * FOUR stages (--gpus 0,1,2,3, three boundaries, three host-synchronized hand-offs per pass) have never run.
#     P5.3 witnessed one boundary on GLM, P5.3b one boundary on DS4.1 (witness #13, 64/64 with slots pinned).
#   * Per-card VRAM during a staged run, sampled at 5 s intervals while every boot lives (claim 1: compare the
#     measured per-card peaks with the P5.0 design table, docs/gfx906/P5-MULTI-GPU-DESIGN.md).
#   * Claim 3 (per-stage caches holding their OWN layers' experts, the #25 header arithmetic ~65-75% hot) is
#     ANSWERED STRUCTURALLY by the current code BEFORE this witness: both engines pin their expert tier to ONE
#     card - the owner's, --gpus[0] - by design (ds41_generate.cpp: "The tier stays on ONE card... it does not
#     care which cards the layers ran on").  So the honest prediction is REFUTED for the current build: staged
#     runs today split DENSE layers and share ONE expert cache; per-stage expert caches ARE task #39 (P6.4).
#     The witness confirms or refutes with the engine's own tier lines and the sampler, and reports which.
#
# GEOMETRY, one paragraph each:
#   GLM (45 layers) is a plain per-layer-KV model: no persistent ring crosses a boundary, so ANY boundary is
#   legal and --layer-split is left at auto (equal chunks: ~11/11/11/12 at 4 stages).
#   DS4.1 enforces THE READER'S RULE (valid boundary set {1,2,8,14,20}, Ds41Dense::split_boundaries_ok) AND the
#   Engram rule (only stage 0 feeds the Engram token ring, so the first boundary must sit above the last Engram
#   layer, 14) - which with --engram leaves {20} and therefore TWO stages as the maximum.  The 4-stage plan
#   {8,14,20} is legal for the caches and refused with Engram ON; the engine's own refusal names the fix ("Drop
#   --engram to split...").  So BOTH DS4.1 arms here run WITHOUT --engram, control included: the comparison is
#   split-vs-unsplit of the same (reduced) model, never staged-engram-vs-unsplit-engram.  The engine prints its
#   "no --engram dirs: the Engram add is skipped (not the real model)" warning on every DS4.1 arm; that is this
#   witness's deliberate shape, not an accident.  (Staged-with-engram 2-card is already witnessed in P5.3b #13.)
#
# THE VERDICT IS CALIBRATED, not absolute (the engine is not bit-deterministic across processes: expert tier
# path -> float summation order -> knife-edge argmax).  Every arm pins --slots 1500 (the residency rule: same
# cap + same rank.bin seed = same resident expert head in every arm) and the staged arms are judged against the
# agreement measured between TWO processes running the SAME command (g_ctl-vs-g_ctlb for GLM, d_4-vs-d_4b for
# DS4.1), exactly as P5.3b did.  A staged arm matching its control at least that far PASSES; full identity is
# reported when it happens (it did in P5.3b #13) but is not the bar.
#
# ONE ENGINE AT A TIME (SVM crash rule): refuses while any engine is alive; the GPU wrapper guarantees that.
# MemAvailable floor arena+25 as in P4.1.  No throughput claims here (that is #30); the wall window is noted.
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
ARENA_GIB=${ARENA_GIB:-60}
MIN_AVAIL_GIB=$(( ARENA_GIB + 25 ))
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }
SLOTS=${SLOTS:-1500}

# Binary freshness (witness #1 of 2026-10-24 measured a stale binary): both engines must be newer than every
# engine source they contain.
GEN_G=build-906/glm_generate; GEN_D=build-906/ds41_generate
for g in "$GEN_G" "$GEN_D"; do
    [[ -x "$g" ]] || { echo "FATAL: missing $g - build with tools/gfx906/build-906-gate.sh first"; exit 3; }
done
for src in tools/glm/glm_dense.cpp tools/glm/glm_generate.cpp tools/ds4/ds4_moe.cpp; do
    [ "$GEN_G" -ot "$src" ] && { echo "FATAL: $GEN_G older than $src"; exit 3; }
done
for src in tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_moe.cpp \
           tools/v41/engine/ds41_engram.cpp tools/gfx906/relay_staging.hpp tools/gfx906/device_split.hpp; do
    [ "$GEN_D" -ot "$src" ] && { echo "FATAL: $GEN_D older than $src"; exit 3; }
done
g++ -std=c++17 -O0 -Wall -Wextra -Werror -I tools/gfx906 -o /tmp/p54-gate-$$ tools/gfx906/test_device_split.cpp || {
    echo "FATAL: the split gate does not compile"; exit 3; }
/tmp/p54-gate-$$ || { echo "FATAL: the split gate FAILs"; rm -f /tmp/p54-gate-$$; exit 3; }
rm -f /tmp/p54-gate-$$

M=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
GLM_MODEL=/ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf
PY=/root/.pyenv/versions/venv312/bin/python3

# ---- prompts (the P4.2 and P4.4 smoke prompts, same tokenizer wiring as those witnesses).
# PROMPT_REPEAT multiplies the prompt (40x is ~560 tokens: the staged prefill then MOVES STATE COLUMNS MID-PREFILL,
# the geometry the P5.3b gates model but no hardware run had witnessed until this knob was added).
export PROMPT_REPEAT="${PROMPT_REPEAT:-1}"
DECODE_N="${DECODE_N:-128}"
IDS_G=logs/p5.4-glm-ids-${TS}.i32
PYTHONPATH="$PWD/tools" $PY - "$IDS_G" <<'EOF' || exit 1
import sys
sys.path.insert(0, "tools/glm/serve")
import glm_tok, os
import strata_tokenizer as ST
ST.PRE_PATTERNS.setdefault("glm4", [glm_tok.GLM4_PATTERN])   # what serve_glm.py:37 does before loading
import numpy as np
tok = glm_tok.load("/ai/models/tok/glm-5.3-flash")
ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences." * int(os.environ.get("PROMPT_REPEAT", "1")))
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"glm ids: {len(ids)} -> {sys.argv[1]}")
EOF
IDS_D=logs/p5.4-ds41-ids-${TS}.i32
$PY - "$IDS_D" <<EOF || exit 1
import sys, json
sys.path.insert(0, "tools")
import strata_tokenizer as ST, numpy as np, os
tdir = "$M/tokenizer"
vocab = json.load(open(tdir + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = open(tdir + "/merges.txt").read().split("\\n")
types = json.load(open(tdir + "/token_type.json"))
tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")
import re
base = "What is a mixture-of-experts model? Answer in two sentences."
reps = int(os.environ.get("PROMPT_REPEAT", "1"))
# Numbered repetitions: 521 identical sentences ended in EOS on the FIRST sampled token (both arms, 2026-10-26
# session 3), which tests nothing. The numbers keep the length and the repetition rhythm but make the context
# non-degenerate, and the tail asks for a long answer so decode has depth to compare.
text = "".join(f"{i}. {base}" for i in range(reps)) + (" Now write a long detailed explanation." if reps > 1 else "")
ids = tok.encode(text)
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"ds41 ids: {len(ids)} -> {sys.argv[1]}")
EOF

# ---- per-card VRAM sampler: one CSV per arm, 5 s cadence, killed when the run exits
start_sampler() {
    local outf=$1
    ( while :; do
        rocm-smi --showmeminfo vram 2>/dev/null | awk -v ts="$(date +%s)" '
            /^GPU\[/{card=$1} /Total Used Memory/{gsub(/[^0-9]/,"",$NF); print ts, card, $NF}' >> "$outf"
        sleep 5
      done ) &
    SAMPLER_PID=$!
}
stop_sampler() { [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=; }

run_one() {   # run_one <name> <engine> [extra engine args...]
    sleep 10
    local name=$1 eng=$2; shift 2
    local log=logs/p5.4-${name}-${TS}.log toks=logs/p5.4-${name}-${TS}.toks vram=logs/p5.4-${name}-${TS}.vram
    echo "=== p5.4/$name start $(date +%T) ==="
    start_sampler "$vram"
    bash "$eng" "$@" >"$toks" 2>"$log"
    local rc=$?
    stop_sampler
    echo "--- $name rc=$rc, $(wc -l < "$toks") token ids -> $log"
    grep -m1 -E "^prompt: |^decode: " "$log" | sed "s/^/[$name] /"
    [ $rc -eq 0 ] || { echo "P5.4: $name FAILED (rc=$rc)"; return 1; }
}

# GLM args: the witnessed P4.2/P5.3 shape (margin 4.0) + the slots pin.
GARGS=(-m "$GLM_MODEL" --backend cuda --experts gpu --ids-file "$IDS_G" -n 64
       --ctx 32768 --arena-gib "$ARENA_GIB" --arena-skip-resident --prefill-chunk 4096 --chunk-mmq
       --vram-margin 4.0 --arena-adapt --pcie auto --stop 154820,154827,154829 --slots "$SLOTS")
# DS4.1 args: the P4.4 shape WITHOUT --engram (see the header: 4 stages and Engram cannot coexist),
# rank.bin kept (it ranks the arena seeding, independent of Engram), the slots pin, and the staged flags
# only on the staged arm.
DARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu --rank "$M/rank.bin"
       --ids-file "$IDS_D" -n "$DECODE_N" --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0 --slots "$SLOTS"
       --prefill-chunk 64)

# ARMS: comma list to re-run a subset after a partial witness (default: everything).
ARMS="${ARMS:-g_ctl,g_2,g_4,g_ctlb,d_ctl,d_4,d_4b}"
has() { [[ ",$ARMS," == *",$1,"* ]]; }

# ---- GLM ladder: control, 2 cards, 4 cards, control twin
has g_ctl  && { run_one g_ctl tools/glm/serve/glm-engine-906.sh "${GARGS[@]}" || exit 1; }
# g_ctlc: the admission-EQUALIZED control (--pf-b 0). The staged decode loop issues NO predicted prefetch (20.6%%
# vs 12.4%% hit on 2026-10-26), and WHICH experts compute on GPU vs CPU changes the float summation order, so an
# un-equalized control is not a fair parity partner for a staged run (the DS4.1 witness's "1e-1 residual from p15"
# was this same confound). If g_ctlc matches the staged arms where g_ctl does not, the relay algebra is sound.
has g_ctlc && { run_one g_ctlc tools/glm/serve/glm-engine-906.sh "${GARGS[@]}" --pf-b 0 || exit 1; }
has g_2    && { run_one g_2   tools/glm/serve/glm-engine-906.sh "${GARGS[@]}" --gpus 0,1 || exit 1; }
has g_4    && { run_one g_4   tools/glm/serve/glm-engine-906.sh "${GARGS[@]}" --gpus 0,1,2,3 || exit 1; }
has g_ctlb && { run_one g_ctlb tools/glm/serve/glm-engine-906.sh "${GARGS[@]}" || exit 1; }

# ---- DS4.1 ladder: control and the 4-stage arm (both WITHOUT Engram), plus the staged twin for the bar
has d_ctl && { run_one d_ctl tools/v41/serve/ds41-engine-906.sh "${DARGS[@]}" || exit 1; }
has d_4   && { run_one d_4   tools/v41/serve/ds41-engine-906.sh "${DARGS[@]}" \
            --gpus 0,1,2,3 --layer-split 8,14,20 --relay-chunk 64 --relay-copy-weights || exit 1; }
has d_4b  && { run_one d_4b  tools/v41/serve/ds41-engine-906.sh "${DARGS[@]}" \
            --gpus 0,1,2,3 --layer-split 8,14,20 --relay-chunk 64 --relay-copy-weights || exit 1; }

# ---- structural guards: the staged runs must SAY they staged, with the geometry this witness claims.
echo
echo "== structural guards =="
fail=0
for arm in g_2 g_4; do
    log=logs/p5.4-${arm}-${TS}.log
    [ -f "$log" ] || continue   # ARMS subset re-run: only judge what ran in this session
    grep -q "relay: staged run" "$log" || { echo "FAIL $arm: never built a staged run"; fail=1; }
    want=$([ "$arm" = g_2 ] && echo 2 || echo 4)
    grep -q "relay: staged run, $want stage(s)" "$log" || {
        echo "FAIL $arm: the staged-run line does not name $want stages"; fail=1; }
    nst=$(grep -c "relay:   stage " "$log" || true)
    [ "$nst" -eq "$want" ] || { echo "FAIL $arm: $nst stage lines, expected $want"; fail=1; }
    # GLM prints no hand-off SUMMARY line, so that a boundary moved real state is proven FUNCTIONALLY by the token
    # verdict below: if a hand-off never ran, the later stages compute on stale columns and tokens diverge at once.
    grep "relay:   stage " "$log" | sed "s/^/[$arm] /"
done
log=logs/p5.4-d_4-${TS}.log
if [ -f "$log" ]; then
 grep -q "relay: staged run" "$log" || { echo "FAIL d_4: never built a staged run"; fail=1; }
for pat in "layers [0,8)" "layers [8,14)" "layers [14,20)" "layers [20,40)"; do
    grep -qF "$pat" "$log" || { echo "FAIL d_4: no stage matching '$pat'"; fail=1; }
done
grep -q "weights own copy" "$log" || { echo "FAIL d_4: stage 1..3 never loaded own weights"; fail=1; }
line=$(grep -m1 "boundary hand-offs" "$log" || true)
echo "d_4: ${line:-FAIL no relay summary}"
[ -n "$line" ] || fail=1
fi
[ "$fail" -eq 0 ] && echo "structural guards: PASS (staged runs built, at the boundaries claimed)"

# ---- claim 1: per-card VRAM peaks, from the sampler (compare with the P5.0 design table by hand in the receipt)
echo
echo "== per-card VRAM peaks (GiB) during each boot (5 s sampler; peak Total Used Memory) =="
python3 - "$TS" <<'EOF'
import glob, sys, collections
for f in sorted(glob.glob(f"logs/p5.4-*-{sys.argv[1]}.vram")):
    arm = f.split("-")[1]
    peak = collections.defaultdict(int)
    for line in open(f):
        _, card, used = line.split()
        n = int(card.strip("GPU[]:"))
        peak[n] = max(peak[n], int(used))
    row = "  ".join(f"card{c}={v/2**30:.2f}" for c, v in sorted(peak.items()))
    print(f"  {arm:6s} {row}")
EOF

# ---- claim 3: the tier lines, verbatim - ONE tier on the owner's card, or per-stage caches?
echo
echo "== expert tier lines (claim 3: confirm or refute per-stage caches with the engine's own numbers) =="
for arm in g_ctl g_4 d_ctl d_4; do
    [ -f "logs/p5.4-${arm}-${TS}.log" ] || continue
    grep -m2 -E "tier: [0-9]+ experts in VRAM|expert slots" "logs/p5.4-${arm}-${TS}.log" | sed "s/^/[$arm] /"
done

# ---- token verdict, calibrated per engine family
echo
echo "== token verdict =="
lcp() {
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
overall=$fail   # a structural guard failure cannot be washed out by a token verdict
[ -f logs/p5.4-g_ctl-${TS}.toks ] && [ -f logs/p5.4-g_ctlb-${TS}.toks ] && {
GLCP=$(lcp logs/p5.4-g_ctl-${TS}.toks logs/p5.4-g_ctlb-${TS}.toks)
NTG=$(wc -l < logs/p5.4-g_ctl-${TS}.toks)
echo "GLM bar (unsplit twins): ${GLCP}/${NTG}"
for arm in g_2 g_4; do
    best=0
    for ctl in g_ctl g_ctlb; do
        l=$(lcp logs/p5.4-${ctl}-${TS}.toks logs/p5.4-${arm}-${TS}.toks)
        [ "$l" -gt "$best" ] && best=$l
    done
    if [ "$best" -ge "$GLCP" ]; then echo "  $arm: ${best}/${NTG} >= bar ${GLCP} - PASS (calibrated)"; else
        echo "  $arm: ${best}/${NTG} < bar ${GLCP} - FAIL (diverges worse than the engine diverges from itself)"; overall=1; fi
done
}
[ -f logs/p5.4-d_ctl-${TS}.toks ] && [ -f logs/p5.4-d_4-${TS}.toks ] && {
DLCP=$(lcp logs/p5.4-d_4-${TS}.toks logs/p5.4-d_4b-${TS}.toks)
NTD=$(wc -l < logs/p5.4-d_ctl-${TS}.toks)
echo "DS4.1 bar (staged twins): ${DLCP}/${NTD}"
best=0
l=$(lcp logs/p5.4-d_ctl-${TS}.toks logs/p5.4-d_4-${TS}.toks); [ "$l" -gt "$best" ] && best=$l
l=$(lcp logs/p5.4-d_ctl-${TS}.toks logs/p5.4-d_4b-${TS}.toks); [ "$l" -gt "$best" ] && best=$l
if [ "$best" -ge "$DLCP" ]; then echo "  d_4: ${best}/${NTD} >= bar ${DLCP} - PASS (calibrated)"; else
    echo "  d_4: ${best}/${NTD} < bar ${DLCP} - FAIL"; overall=1; fi
}
echo "artifacts: logs/p5.4-*-${TS}.{log,toks,vram}"
[ "$overall" -eq 0 ] && { echo "P5.4 WITNESS: PASS"; exit 0; }
echo "P5.4 WITNESS: FAIL"; exit 1
