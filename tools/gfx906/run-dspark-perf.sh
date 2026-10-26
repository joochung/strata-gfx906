#!/usr/bin/env bash
# run-dspark-perf.sh - DS4.1 prefill + decode performance WITH AND WITHOUT MTP (DSpark), one card, unsplit.
#
# WHY.  DSpark is this checkpoint's MTP: the mtp.0-2 draft stack (V41-MTP-Vision.Q2_K.gguf sidecar) drafts
# STRATA_DSPARK_DEPTH tokens and ONE dense pass verifies them (Leviathan-style, confidence-gated at
# STRATA_DSPARK_CONF).  It is opt-in, and a STAGED (--layer-split) run REFUSES it (ds41_generate.cpp: a draft's k
# positions cannot span a staged column boundary).  So the honest measurement of "MTP on this engine" is the
# single-card unsplit shape: control vs DSpark, everything else identical, default VRAM budgets (this is a
# performance test - the --slots pinning the correctness witness used would hide the VRAM cost of the drafter,
# which is part of what we measure: the sidecar loads BEFORE the expert cache measures free VRAM, so the cache
# sizes around it).
#
# WHAT.  Same tokenized prompt both runs (~PROMPT_TARGET ids, real document text, not a repeated toy line -
# repetition would flatter the expert cache).  Per run the engine prints "prompt: N tokens in X s = Y tok/s",
# "decode: N tokens in X s = Y tok/s; experts ...", and (DSpark only) "DSpark: V verify passes, a accepted
# drafts/pass = t tokens/pass, f% full, ...".  This script parses those three lines into a side-by-side summary.
#
# SAFETY.  Refuses if any engine is alive (one-HIP-engine SVM rule, vLLM included); refuses below (arena+25) GiB
# RAM; sleeps between boots (a finished engine frees VRAM asynchronously).  Orchestrator launches it through the
# GPU wrapper, which stops and restarts the stack:
#   /ai/engines/gfx906/unverbraucht/run-with-free-gpus.sh \
#       "cd /ai/engines/gfx906/worktrees/p5.3-relay && bash tools/gfx906/run-dspark-perf.sh"
set -u
cd "$(dirname "$0")/../.." || exit 1

TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
OUT=logs/dspark-perf-${TS}
PY=/root/.pyenv/versions/venv312/bin/python3
M=${M:-/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored}
SIDECAR=${SIDECAR:-$M/V41-MTP-Vision.Q2_K.gguf}
PROMPT_TARGET=${PROMPT_TARGET:-1200}
N_TOK=${N_TOK:-256}
ARENA_GIB=${ARENA_GIB:-64}

if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - stop it first (one HIP engine at a time)"; exit 3
fi
MIN_AVAIL_GIB=$(( ARENA_GIB + 25 ))
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }
[ -f "$SIDECAR" ] || { echo "refusing: sidecar $SIDECAR missing"; exit 3; }

GEN=build-906/ds41_generate
ENG=(bash tools/v41/serve/ds41-engine-906.sh)
[[ -x "$GEN" ]] || { echo "FATAL: missing $GEN - build with tools/gfx906/build-906-gate.sh first"; exit 3; }

# ---- the prompt: real document text, truncated to PROMPT_TARGET ids (both runs get the SAME file)
IDS=$OUT.i32
$PY - "$IDS" "$PROMPT_TARGET" <<'EOF' || exit 1
import sys, json
sys.path.insert(0, "tools")
import strata_tokenizer as ST, numpy as np
tdir = "/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/tokenizer"
vocab = json.load(open(tdir + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = open(tdir + "/merges.txt").read().split("\n")
types = json.load(open(tdir + "/token_type.json"))
tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")
text = ""
for f in ("docs/DEEPSEEK_V41.md", "docs/gfx906/P5-MULTI-GPU-DESIGN.md", "docs/gfx906/P5-RELAY-PATTERN.md"):
    text += open(f).read() + "\n\n"
ids = tok.encode(text)[: int(sys.argv[2])]
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
EOF

ARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu
      --engram "$M/engram/l1:$M/engram/l14" --rank "$M/rank.bin"
      --ids-file "$IDS" -n "$N_TOK" --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0
      --prefill-chunk 64)

run_one() {   # run_one <name> <env assignments...>
    sleep 10
    local name=$1; shift
    echo "=== $name start $(date +%T) ==="
    env "$@" "${ENG[@]}" "${ARGS[@]}" >"$OUT.$name.toks" 2>"$OUT.$name.log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$OUT.$name.toks") token ids"
    grep -m1 -E "^prompt: " "$OUT.$name.log" | sed "s/^/[$name] /"
    grep -m1 -E "^decode: " "$OUT.$name.log" | sed "s/^/[$name] /"
    grep -m1 -E "^DSpark: " "$OUT.$name.log" | sed "s/^/[$name] /"
    grep -m1 -E "experts in VRAM" "$OUT.$name.log" | sed "s/^/[$name] /"
    [ $rc -eq 0 ] || { echo "DSPARK-PERF: $name FAILED (rc=$rc) - see $OUT.$name.log"; return 1; }
}

run_one control || exit 1
run_one dspark \
    STRATA_DSPARK="$SIDECAR" \
    STRATA_DSPARK_KEEP="$PWD/tools/v41/serve/dspark_expert_counts.txt:64" \
    STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}" \
    STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}" || exit 1

echo
echo "== SUMMARY ($PROMPT_TARGET-token prompt, $N_TOK decode tokens, one card, unsplit, default budgets) =="
for name in control dspark; do
    p=$(grep -m1 -oE "prompt: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s" "$OUT.$name.log" | grep -oE "[0-9.]+ tok/s")
    d=$(grep -m1 -oE "decode: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s" "$OUT.$name.log" | grep -oE "[0-9.]+ tok/s")
    s=$(grep -m1 -oE "DSpark: .*" "$OUT.$name.log" || true)
    echo "  $name: prompt $p | decode $d ${s:+| $s}"
done
echo "artifacts: $OUT.* (logs, tokens)"
