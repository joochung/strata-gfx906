#!/usr/bin/env bash
# P5.4c (#40): the GLM staged-divergence DISCRIMINATOR.  One question: with admission made
# deterministic, does a staged GLM run still diverge from its control?
#
# Why this probe exists.  P5.4 (receipt 2026-10-26-p5.4-split-smoke.json) saw GLM's staged arms diverge
# from control at token 16 while the unsplit twins matched 64/64, and --pf-b 0 moved the divergence to 26
# without removing it.  Every arm there ran --pcie auto - the FreeToken-style ADMISSION whose PCIe share
# is an EMA of MEASURED per-path cost (tools/ds4/ds4_moe.cpp:337-339).  The staged shape runs ~2.7 ms/pass
# slower (the relay), so auto converges to a DIFFERENT pcie/cpu mix by design - 28.5% vs 23.2% observed.
# A CPU-computed expert is not bit-identical to a GPU-computed one (different summation order), so timing
# -> admission -> float order -> knife-edge argmax.  The DS4.1 arms, which went bitwise 128/128, left
# --pcie at the deterministic default 0.25 (error-diffusion carry advances per lookup, so the SAME call
# sequence assigns the SAME paths in both arms).  This probe takes the confound away on both sides and
# re-runs the three shapes:
#
#   g_ctlp  control, --pcie 0.25 --pf-b 0                     (deterministic admission, no prefetch)
#   g_00p   --gpus 0,0 same-device relay (relay algebra with IDENTICAL residency environment)
#   g_2p    --gpus 0,1 cross-card, same deterministic args
#
# Verdict logic (the script prints it):
#   all three 64/64 EXACT   -> the staged relay algebra is SOUND; P5.4's GLM divergence was the
#                              timing-adaptive admission, and the residency rule extends: a staged-vs-
#                              control parity witness must pin --pcie (auto is a function of wall time).
#   g_00p EXACT, g_2p not   -> residency/transport artifact remains even deterministic -> name it.
#   g_00p diverges too      -> real staged-dense defect; next step is the STRATA_DUMP diff campaign.
#
# Static finding shipped alongside (NOT what the witnessed arms exercised, but a real defect): with
# --route-bias/--route-bias-file set, a staged GLM run never updates the bias for decode - set_bias()
# lives only on the unsplit step() path (glm_generate.cpp:902) and a staged step() returns at :899 - so
# staged decodes route with the build-time prune-only vector.  That is DS4.1 root cause #2
# (a6bfab87) in GLM's copy.  It is INERT with the bias off, which is every arm here.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
TS=$(date +%Y%m%d-%H%M%S)

# guards: one engine at a time (crash rule), and the RAM the load needs
if pgrep -f 'glm_generate|ds41_generate|strata ' >/dev/null; then
    echo "P5.4c: refusing - an engine is already running (one HIP engine at a time)"; exit 1
fi
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge 60 ] || { echo "P5.4c: MemAvailable ${AVAIL} GiB < 60 - refusing to load"; exit 1; }

M=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
GLM_MODEL=/ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf
PY=/root/.pyenv/versions/venv312/bin/python3
SLOTS="${SLOTS:-1500}"

# the exact P5.4 prompt (14 tokens at PROMPT_REPEAT=1) - the stream the token-16 divergence was measured on
IDS_G=logs/p5.4c-glm-ids-${TS}.i32
PYTHONPATH="$PWD/tools" $PY - "$IDS_G" <<'EOF' || exit 1
import sys
sys.path.insert(0, "tools/glm/serve")
import glm_tok
import strata_tokenizer as ST
ST.PRE_PATTERNS.setdefault("glm4", [glm_tok.GLM4_PATTERN])
import numpy as np
tok = glm_tok.load("/ai/models/tok/glm-5.3-flash")
ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences.")
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"glm ids: {len(ids)} -> {sys.argv[1]}")
EOF

run_one() {   # run_one <name> [extra engine args...]
    sleep 10
    local name=$1; shift
    local log=logs/p5.4c-${name}-${TS}.log toks=logs/p5.4c-${name}-${TS}.toks
    echo "=== p5.4c/$name start $(date +%T) ==="
    bash tools/glm/serve/glm-engine-906.sh "$@" >"$toks" 2>"$log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$toks") token ids -> $log"
    grep -m1 -E "^prompt: |^decode: " "$log" | sed "s/^/[$name] /"
    grep -m1 "experts/token" "$log" | sed "s/^/[$name] /"
    return $rc
}

# the deterministic set: --pcie 0.25 (NOT auto - the probe's whole point), --pf-b 0 both sides, slots pinned
GARGS=(-m "$GLM_MODEL" --backend cuda --experts gpu --ids-file "$IDS_G" -n 64
       --ctx 32768 --arena-gib 60 --arena-skip-resident --prefill-chunk 4096 --chunk-mmq
       --vram-margin 4.0 --arena-adapt --pcie 0.25 --pf-b 0 --stop 154820,154827,154829 --slots "$SLOTS")

run_one g_ctlp "${GARGS[@]}" || exit 1
run_one g_00p "${GARGS[@]}" --gpus 0,0 || exit 1
run_one g_2p  "${GARGS[@]}" --gpus 0,1 || exit 1

lcp() { local a=$1 b=$2 i=0; while IFS= read -r x <&3 && IFS= read -r y <&4; do
            [ "$x" = "$y" ] || break; i=$((i+1)); done 3<"$a" 4<"$b"; echo $i; }
A=logs/p5.4c-g_ctlp-${TS}.toks; B=logs/p5.4c-g_00p-${TS}.toks; C=logs/p5.4c-g_2p-${TS}.toks
NA=$(wc -l < "$A")
echo
echo "VERDICT (control has $NA tokens; bar = EXACT stream equality, the P5.4 calibrated bar):"
for pair in "00p:$A:$B" "2p:$A:$C"; do
    n=${pair%%:*}; rest=${pair#*:}; f1=${rest%%:*}; f2=${rest#*:}
    if cmp -s "$f1" "$f2"; then echo "  staged $n vs control: EXACT $NA/$NA";
    else echo "  staged $n vs control: diverges - common prefix $(lcp "$f1" "$f2")/$NA"; fi
done
echo "and the control-vs-control reference is the P5.4 twins (64/64 EXACT, same prompt, same bar)."
echo "P5.4c done $(date +%T). logs/p5.4c-*-${TS}.*"
