#!/usr/bin/env bash
# run-p6.2-tp-shard.sh - the P6.2 same-device shard spike: measure the sharded attention seam's OVERHEAD and
# NUMERICS against the unsliced path of the SAME binary, on ONE card.
#
# WHY.  P6 (docs/gfx906/P6-TENSOR-SPLIT.md) splits a LAYER across cards; the decode verdict hinges on the per-pass
# seam cost, and the plan's kill gate is written on the SEAM PLUMBING ALONE: "if the same-device two-shard run's
# overhead exceeds 10% at n=64, cross-card prefill TP cannot reach its ~1.8x GEMM halving and is NO-GO" - because
# on one device nothing is halved, so whatever the sharded graph costs over the plain one is pure seam plumbing
# (extra FA launches, two sliced mul_mats, the f32 reduce, the strided q slice).  The numerics half is the same
# run: a different f32 summation ORDER (two partials summed instead of one reduction), so tokens may knife-edge
# flip - the P6.1 host gate (p6_tp_algebra.cpp) bounded the induced error at f32 ~1e-4; this run witnesses what
# that error does to a real greedy decode.
#
# WHAT.  Two runs, same binary, same prompt, same --slots pin (the residency rule: an equal expert cap makes the
# tier path identical, so any rate or token difference is the seam and not the cache):
#   control : default path (tp_shards=1, byte-identical to the last release)
#   tp      : --tp-shards 2 --tp-layers 0..9 (the spike's default TPL): 10 layers sharded, which bounds the
#             sliced-weight VRAM (~100 MB/layer) inside one card.  Per-pass overhead is reported for the WHOLE
#             run (40 layers, 10 sharded) AND extrapolated per-sharded-layer, since a real TP splits EVERY layer.
# Parses the engine's own "prompt:" / "decode:" rate lines and the "experts in VRAM" line; diffs the token files.
#
# SAFETY.  Refuses if any engine is alive (one-HIP-engine SVM rule, vLLM included); refuses below (arena+25) GiB
# RAM; sleeps between boots (a finished engine frees VRAM asynchronously).  Orchestrator launches it through the
# GPU wrapper, which stops and restarts the stack:
#   /ai/engines/gfx906/unverbraucht/run-with-free-gpus.sh \
#       "cd /ai/engines/gfx906/worktrees/p5.3-relay && bash tools/gfx906/run-p6.2-tp-shard.sh"
set -u
cd "$(dirname "$0")/../.." || exit 1

TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
OUT=logs/p6.2-tp-${TS}
PY=/root/.pyenv/versions/venv312/bin/python3
M=${M:-/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored}
PROMPT_TARGET=${PROMPT_TARGET:-512}
N_TOK=${N_TOK:-128}
ARENA_GIB=${ARENA_GIB:-64}
SLOTS=${SLOTS:-1500}          # the residency pin, BOTH runs (the residency rule; see header)
TPL=${TPL:-0,1,2,3,4,5,6,7,8,9}

if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - stop it first (one HIP engine at a time)"; exit 3
fi
MIN_AVAIL_GIB=$(( ARENA_GIB + 25 ))
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }

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
for f in ("docs/DEEPSEEK_V41.md", "docs/gfx906/P6-TENSOR-SPLIT.md"):
    text += open(f).read() + "\n\n"
ids = tok.encode(text)[: int(sys.argv[2])]
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
EOF

ARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu
      --engram "$M/engram/l1:$M/engram/l14" --rank "$M/rank.bin"
      --ids-file "$IDS" -n "$N_TOK" --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0
      --prefill-chunk 64 --slots "$SLOTS")

run_one() {   # run_one <name> [ENV=k=v ...] [extra engine args...]   (ENV= must come first, if any)
    sleep 10
    local name=$1; shift
    local envkv=()
    if [ "${1#ENV=}" != "${1:-}" ]; then envkv=("${1#ENV=}"); shift; fi
    echo "=== $name start $(date +%T) ==="
    env "${envkv[@]}" "${ENG[@]}" "${ARGS[@]}" "$@" >"$OUT.$name.toks" 2>"$OUT.$name.log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$OUT.$name.toks") token ids"
    grep -m1 -E "^prompt: " "$OUT.$name.log" | sed "s/^/[$name] /"
    grep -m1 -E "^decode: " "$OUT.$name.log" | sed "s/^/[$name] /"
    grep -m1 -E "experts in VRAM" "$OUT.$name.log" | sed "s/^/[$name] /"
    [ $rc -eq 0 ] || { echo "P6.2: $name FAILED (rc=$rc) - see $OUT.$name.log"; return 1; }
}

run_one control || exit 1
# the sharded run: its loader MUST report 6 sliced copies per sharded layer (3 tensors x 2 shards), else the flag
# silently did not reach the graph and the comparison would be a no-op measured twice.  STRATA_TP_NO_FA=1 keeps the
# same sliced weights and GEMM shape but runs ONE full-head flash-attn - the difference between the two tp arms
# prices the per-shard FA launches alone (see ds41_dense.cpp's attribution comment).
N_SLICES=$(( $(echo "$TPL" | awk -F, '{print NF}') * 6 ))
run_one tp --tp-shards 2 --tp-layers "$TPL" || exit 1
if ! grep -q "p6.2 tp seam: $N_SLICES sliced weight copies" "$OUT.tp.log"; then
    echo "P6.2: the tp run did not build the expected $N_SLICES sliced weight copies - refusing to compare"; exit 1
fi
if [ "${ARM_NO_FA:-1}" = 1 ]; then
    run_one tp_nofa ENV=STRATA_TP_NO_FA=1 --tp-shards 2 --tp-layers "$TPL" || exit 1
    if ! grep -q "p6.2 tp seam: $N_SLICES sliced weight copies" "$OUT.tp_nofa.log"; then
        echo "P6.2: the tp_nofa run did not build the expected $N_SLICES sliced weight copies"; exit 1
    fi
fi

echo
echo "== SUMMARY ($PROMPT_TARGET-token prompt, $N_TOK decode tokens, chunk 64, one card, slots $SLOTS both, ${TPL} sharded) =="
rate() { grep -m1 -oE "$1: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s" "$2" | grep -oE "[0-9.]+ tok/s"; }
for name in control tp tp_nofa; do
    [ -f "$OUT.$name.log" ] || continue
    p=$(rate prompt "$OUT.$name.log"); d=$(rate decode "$OUT.$name.log")
    v=$(grep -m1 -oE "[0-9]+ experts in VRAM" "$OUT.$name.log" || echo "?")
    echo "  $name: prompt $p | decode $d | $v"
done
python3 - "$OUT" <<'EOF'
import sys
out = sys.argv[1]
def rate(name, kind):
    import re, os
    if not os.path.exists(f"{out}.{name}.log"):
        return None
    s = open(f"{out}.{name}.log", errors="ignore").read()
    m = re.search(rf"{kind}: \d+ tokens in [\d.]+ s = ([\d.]+) tok/s", s)
    return float(m.group(1)) if m else None
for kind in ("prompt", "decode"):
    c = rate("control", kind)
    for arm in ("tp", "tp_nofa"):
        t = rate(arm, kind)
        if c and t:
            print(f"  {arm} {kind} overhead vs control: {100*(c-t)/c:+.2f}% (slower is positive)")
a = open(f"{out}.control.toks").read().split()
for arm in ("tp", "tp_nofa"):
    b = open(f"{out}.{arm}.toks").read().split()
    n = min(len(a), len(b))
    div = next((i for i in range(n) if a[i] != b[i]), n)
    print(f"  {arm} tokens: {len(b)} vs control {len(a)}; identical through index {div} "
          f"({'ALL' if div == n else 'first difference at ' + str(div)})")
EOF
echo "artifacts: $OUT.* (logs, tokens)"
