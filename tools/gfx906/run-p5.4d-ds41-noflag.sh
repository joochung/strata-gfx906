#!/usr/bin/env bash
# P5.4d (#39 precondition): the cross-device slot-budget fix, witnessed.
#
# THE DEFECT (P5.3b receipt "residency_equalization"; P5.5 receipt "preconditions"): a staged DS4.1 run
# refused to build across cards unless --relay-copy-weights was given, and that flag gives EVERY stage its
# own weight load - including STAGE 0, whose card already holds the parent's whole dense half. Witnessed
# in session 20261026-235443: the tier sized itself from the owner card's free VRAM at decode time and saw
#   control  "VRAM after the prompt: 23.43 GiB free -> 22.43 GiB of expert slots"
#   staged   "VRAM after the prompt: 17.38 GiB free -> 16.38 GiB of expert slots"   (~6 GiB gone)
# - a stage-0 duplicate of ~6 GiB sat on the tier's card. Every default-budget staged number inherited it.
#
# THE FIX (this commit): the generate-level cross-card refusal is removed. Ds41Dense::build_stages already
# applies the per-card rule itself - a stage on the owner's backend ADOPTS the owner's tensors, a stage on a
# foreign backend loads its OWN layer range there - and the real cross-card constraint (the ring rule: a
# boundary may cross a card only at the successor's own kv AND index source layer) is enforced there. The
# same shape was already witnessed on GLM (P5.4c g_2p: stage 0 shared + stage 1 own copy, 64/64 EXACT).
#
# ARMS (one session; --slots pinned per the residency rule; DS4.1's --pcie default 0.75 is deterministic):
#   d_ctl   unsplit control
#   d_2n    --gpus 0,1 --layer-split 20, NO --relay-copy-weights   (the fixed path)
#   d_2f    --gpus 0,1 --layer-split 20 WITH --relay-copy-weights  (regression guard for the copy path)
#
# PASS BAR:
#   1. d_2n BUILDS (the refusal is gone) and reports stage 0 weights "shared", stage 1 "own copy";
#   2. d_2n's decode-time budget line reads >= 22.5 GiB free (control-class), NOT ~17.4 (the deficit);
#   3. d_2n's and d_2f's token streams are EXACT vs d_ctl (calibrated parity bar, --slots pinned);
#   4. d_2f still shows the ~17 GiB budget line and stays bit-equal (the flag path unchanged).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
TS=$(date +%Y%m%d-%H%M%S)
if pgrep -f 'glm_generate|ds41_generate|strata ' >/dev/null; then
    echo "P5.4d: refusing - an engine is already running (one HIP engine at a time)"; exit 1
fi
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge 60 ] || { echo "P5.4d: MemAvailable ${AVAIL} GiB < 60 - refusing to load"; exit 1; }

M=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
PY=/root/.pyenv/versions/venv312/bin/python3
SLOTS="${SLOTS:-1500}"
IDS_D=logs/p5.4d-ds41-ids-${TS}.i32
$PY - "$IDS_D" <<EOF || exit 1
import sys, json
sys.path.insert(0, "tools")
import strata_tokenizer as ST, numpy as np
tdir = "$M/tokenizer"
vocab = json.load(open(tdir + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
merges = open(tdir + "/merges.txt").read().split("\n")
types = json.load(open(tdir + "/token_type.json"))
tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")
ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences.")
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"ds41 ids: {len(ids)} -> {sys.argv[1]}")
EOF

run_one() {   # run_one <name> [extra args...]
    sleep 10
    local name=$1; shift
    local log=logs/p5.4d-${name}-${TS}.log toks=logs/p5.4d-${name}-${TS}.toks
    echo "=== p5.4d/$name start $(date +%T) ==="
    bash tools/v41/serve/ds41-engine-906.sh "$@" >"$toks" 2>"$log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$toks") token ids -> $log"
    grep -m1 "VRAM after the prompt" "$log" | sed "s/^/[$name] /"
    grep -E "relay:   stage [0-9]+:.*(shared|own copy)" "$log" | sed "s/^/[$name] /"
    [ $rc -eq 0 ] || { echo "P5.4d: $name FAILED (rc=$rc)"; return 1; }
}

DARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu --rank "$M/rank.bin"
       --ids-file "$IDS_D" -n 64 --arena-gib 60 --arena-adapt --vram-margin 1.0 --slots "$SLOTS"
       --prefill-chunk 64)

run_one d_ctl "${DARGS[@]}" || exit 1
run_one d_2n "${DARGS[@]}" --gpus 0,1 --layer-split 20 --relay-chunk 64 || exit 1
run_one d_2f "${DARGS[@]}" --gpus 0,1 --layer-split 20 --relay-chunk 64 --relay-copy-weights || exit 1

lcp() { local a=$1 b=$2 i=0; while IFS= read -r x <&3 && IFS= read -r y <&4; do
            [ "$x" = "$y" ] || break; i=$((i+1)); done 3<"$a" 4<"$b"; echo $i; }
A=logs/p5.4d-d_ctl-${TS}.toks
echo
echo "VERDICT:"
for arm in d_2n d_2f; do
    B=logs/p5.4d-${arm}-${TS}.toks
    NA=$(wc -l < "$A")
    if cmp -s "$A" "$B"; then echo "  $arm vs control: EXACT $NA/$NA"
    else echo "  $arm vs control: diverges - common prefix $(lcp "$A" "$B")/$NA"; fi
    line=$(grep -m1 "VRAM after the prompt" "logs/p5.4d-${arm}-${TS}.log" || true)
    echo "  $arm decode-time budget: ${line:-MISSING}"
done
echo "  control decode-time budget: $(grep -m1 'VRAM after the prompt' logs/p5.4d-d_ctl-${TS}.log || echo MISSING)"
echo "PASS = d_2n built with stage 0 shared / stage 1 own copy, its budget line is control-class, both staged"
echo "arms EXACT. logs/p5.4d-*-${TS}.*"
