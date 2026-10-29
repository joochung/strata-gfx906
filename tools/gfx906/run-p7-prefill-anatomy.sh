#!/usr/bin/env bash
# run-p7-prefill-anatomy.sh - P7 EXP1: decompose DS4.1 prefill (the ~40 tok/s chunked path and the batch
# path) on ONE machine, ONE prompt, with the instruments this box has never run: MIMO_CHUNK_PROF=1
# (run_chunk's wall split: file waits / copy waits / sync / GPU grouped kernels / RESIDENT kernels /
# PTQ1->Q2_0 transcode) plus the always-on `prompt split: dense/experts/file` line.
#
# Why: the receipts say prefill is 75-82% expert-tier time (logs/p6.2-*.control.log: 512 tok 27.2 tok/s
# = dense 3.2 s + experts 15.5 s, file wait 6.2 s; logs/dspark-perf-20261025: 1200 tok 39.2 tok/s =
# dense 7.5 + experts 23.0, file 7.9). Whether a 4-card shape can help depends entirely on WHICH expert
# cost dominates: resident grouped kernels (splits across cards), file streaming (NVMe is shared - no
# card count helps the disk), PCIe upload (4 links DO help), or the transcode (4x cards 4x the transcode
# - a reason to pre-transcode instead). No run on this box has ever printed the split. This runner makes it.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
TS=$(date +%Y%m%d-%H%M%S)
OUT=logs/p7-anatomy-$TS
M=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
GEN=build-906/ds41_generate
ENG=(bash tools/v41/serve/ds41-engine-906.sh)   # THIS box's wrapper (build dir + DSpark-off), as in p6.2
PY=/root/.pyenv/versions/venv312/bin/python3
N_TOK=8                     # prefill is the subject; decode is measured elsewhere (P6.4/#46: -25% with shards)
SLOTS=1500
ARENA_GIB=64
PROMPT_N=1200               # the 39.2 tok/s reference shape (dspark-perf-20261025 control)
free_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
[ "$(free_gib)" -ge 65 ] || { echo "refusing: MemAvailable $(free_gib) GiB < 65"; exit 3; }
for pat in ds41_generate glm_generate; do pgrep -f "$pat --serve" >/dev/null && { echo "refusing: $pat alive"; exit 3; }; done
pgrep -f "serve_ds41.py|serve_glm.py" >/dev/null && { echo "refusing: a serve wrapper is alive"; exit 3; }
for src in tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_moe.cpp src/kernels/cpu/pool.cpp; do
    [ "$GEN" -nt "$src" ] || { echo "refusing: $GEN older than $src"; exit 3; }
done
mkdir -p logs

IDS=$OUT.i32
$PY - "$IDS" "$PROMPT_N" <<'EOF' || exit 1
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
[ -s "$IDS" ] || { echo "FAIL: prompt builder produced no ids"; exit 1; }

ARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu
      --engram "$M/engram/l1:$M/engram/l14" --rank "$M/rank.bin"
      --ids-file "$IDS" -n "$N_TOK" --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0
      --slots "$SLOTS")

run_one() {   # run_one <name> [ENV=k=v] [extra engine args...]
    sleep 8
    local name=$1; shift
    local envkv=()
    if [ $# -gt 0 ] && [ "${1#*=}" != "$1" ] && [[ "$1" == *=* ]]; then envkv=("$1"); shift; fi
    echo "=== $name start $(date +%T) ==="
    env "${envkv[@]}" "${ENG[@]}" "${ARGS[@]}" "$@" >"$OUT.$name.toks" 2>"$OUT.$name.log"
    local rc=$?
    echo "--- $name rc=$rc"
    grep -m1 -E "^prompt: " "$OUT.$name.log" | sed "s/^/[$name] /"
    grep -m1 -E "^prompt split: " "$OUT.$name.log" | sed "s/^/[$name] /"
    grep -m1 -E "^chunk prof: " "$OUT.$name.log" | sed "s/^/[$name] /"
    grep -m1 -E "^chunk prof MMQ: " "$OUT.$name.log" | sed "s/^/[$name] /"
    [ $rc -eq 0 ] || { echo "P7: $name FAILED (rc=$rc) - see $OUT.$name.log"; return 1; }
}

# warm: throwaway boot so every measured arm sees the same page cache (the w3 lesson: a cold first boot
# moves the control 2.4x by itself).
run_one warm MIMO_CHUNK_PROF=0 -n 1 --prefill-chunk 64 || { echo "P7: warm boot failed"; exit 1; }

for r in 1 2; do
    run_one "chunk-r$r" MIMO_CHUNK_PROF=1 --prefill-chunk 64
    run_one "batch-r$r" MIMO_CHUNK_PROF=1 --prefill-chunk 2048 --prompt-batch 2048
done

echo "== anatomy summary =="
$PY - "$OUT" "$OUT.summary.jsonl" <<'EOF'
import re, sys, json
out, js = sys.argv[1], sys.argv[2]
def grab(name, path, rx):
    m = re.search(rx, open(path, errors="replace").read())
    return m.groupdict() if m else {}
rows = []
for name in ("chunk-r1", "batch-r1", "chunk-r2", "batch-r2"):
    log = f"{out}.{name}.log"
    row = {"arm": name}
    row.update(grab(name, log, r"^prompt: (?P<toks>\d+) tokens in (?P<s>[\d.]+) s = (?P<tok_s>[\d.]+) tok/s \(experts (?P<eh>\d+) hit / (?P<ec>\d+) cpu / (?P<ep>\d+) pcie / (?P<ef>\d+) file", ))
    row.update(grab(name, log, r"^prompt split: dense (?P<dense_s>[\d.]+) s, experts (?P<experts_s>[\d.]+) s \(file wait (?P<file_s>[\d.]+) s\)"))
    row.update(grab(name, log, r"^chunk prof: (?P<calls>\d+) calls, (?P<launches>\d+) launches: wall (?P<wall>\d+) ms = waits on reads (?P<reads>\d+) \+ wave/copy waits (?P<copies>\d+) \+ final sync (?P<sync>\d+) \+ other host (?P<other>\d+); reader wall (?P<reader>\d+) ms; GPU grouped kernels (?P<kern>\d+) ms \(resident (?P<resident>\d+)\); file tier (?P<file_gb>[\d.]+) GB read in (?P<rd_busy>\d+) ms of reader time = (?P<file_gbs>[\d.]+) GB/s"))
    row.update(grab(name, log, r"^chunk prof MMQ: (?P<mmq_sub>\d+) sub-runs; PTQ1_0 -> Q2_0 transcode (?P<tc>\d+) ms GPU \(inside the grouped kernels\) over (?P<tc_launches>\d+) sub-runs, (?P<tc_gb>[\d.]+) GB of ternary read = (?P<tc_gbs>\d+) GB/s"))
    rows.append(row)
    print(name, json.dumps(row))
with open(js, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
EOF
echo "logs: $OUT.*.log  summary: $OUT.summary.jsonl"
