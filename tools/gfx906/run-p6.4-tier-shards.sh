#!/bin/bash
# P6.4 (task #39): the EXPERT-PARALLEL TIER scaling witness.  --tier-cards N stripes the routed-expert cache
# over N cards (expert e on shard e%N; each shard its own VRAM cache + arena slice + CPU pool; no GEMM split,
# no allreduce - the P6.2 tensor-TP verdict stands).  Launch through the GPU wrapper:
#   /ai/engines/gfx906/unverbraucht/run-with-free-gpus.sh bash -c \
#     "cd /ai/engines/gfx906/worktrees/p5.3-relay && bash tools/gfx906/run-p6.4-tier-shards.sh"
#
# WHAT THIS MEASURES (the task exit): decode hit rate and tok/s at 1 / 2 / 4 cards against the single-card
# control, plus an EQUAL-CAPACITY 4-shard arm (--slots 375 = 4x375 = 1500 = the control's 1500: it separates
# the CAPACITY gain (more experts cached) from the PLACEMENT gain (same cache, four byte-supply pipes)).
# Arms INTERLEAVE (ctl,t2,t4,t4e per round, ROUNDS>=2) and every arm gets its WITHIN-ARM SPREAD - the standing
# rule from the DSpark swap-order repeat: no rate claim without same-session repeated boots.
#
# THE PARITY BAR IS NOT bit-parity vs control.  The fan-in sums shard partials in shard order, a different
# float grouping than the single-card sum (host gate tools/v41/engine/test_ds41_moe.cpp --shards: 5e-8, red
# coverage controls 0.55+).  Knife-edge token flips are EXPECTED; the common token prefix vs the control of
# the same round is reported, never gated.  --slots pinned everywhere (residency rule).
#
# Host gates that run BEFORE any GPU boot: the fan-out gate above (PASS line required), and the binary
# freshness guard.  MemAvailable floor = arena + 25 (the engine slices --arena-gib by tcards internally; the
# gate keeps the total, so the floor is the same as P5.4).  ONE ENGINE AT A TIME (SVM crash rule).
#
# Cost: each boot ~3.5 min.  Default: 1 throwaway WARM boot (the first boot after vLLM evicts the GGUF page
# cache measures 2.4x slow - P5.5 lesson) + 2 rounds x 4 arms = 9 boots ~= 40 min.
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

GEN_D=build-906/ds41_generate
[[ -x "$GEN_D" ]] || { echo "FATAL: missing $GEN_D - build with tools/gfx906/build-906-gate.sh first"; exit 3; }
for src in tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_moe.cpp \
           tools/v41/engine/ds41_moe.hpp tools/v41/engine/ds41_engram.cpp; do
    [ "$GEN_D" -ot "$src" ] && { echo "FATAL: $GEN_D older than $src"; exit 3; }
done

# the host fan-out gate (tools/v41/engine/test_ds41_moe.cpp --shards): subset sums to the full tier within
# float association, deterministically, and BOTH coverage red controls FAIL loud.  Required PASS before GPU.
echo "=== host fan-out gate (test_ds41_moe --shards 4) ==="
M=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
GATE_LOG=logs/p6.4-hostgate-${TS}.log
./build-906/test_ds41_moe "$M/V41-GSQ.gguf" --layers 0 --tokens 4 --shards 4 >"$GATE_LOG" 2>&1 \
    || { echo "FATAL: host fan-out gate exit nonzero - see $GATE_LOG"; exit 3; }
grep -q -- "-> PASS" "$GATE_LOG" || { echo "FATAL: host fan-out gate not PASS - see $GATE_LOG"; exit 3; }
GATE_N=$(grep -c "shards 4: rel" "$GATE_LOG")
[ "$GATE_N" -gt 0 ] || { echo "FATAL: gate ran no shard checks"; exit 3; }
echo "host gate PASS ($GATE_N shard checks, red controls in $GATE_LOG)"

PY=/root/.pyenv/versions/venv312/bin/python3
DECODE_N="${DECODE_N:-128}"
ROUNDS="${ROUNDS:-2}"
export PROMPT_REPEAT="${PROMPT_REPEAT:-1}"
IDS_D=logs/p6.4-ds41-ids-${TS}.i32
$PY - "$IDS_D" <<EOF || exit 1
import sys, json
sys.path.insert(0, "tools")
import strata_tokenizer as ST, numpy as np, os
tdir = "$M/tokenizer"
vocab = json.load(open(tdir + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = open(tdir + "/merges.txt").read().split("\n")
types = json.load(open(tdir + "/token_type.json"))
tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")
base = "What is a mixture-of-experts model? Answer in two sentences."
reps = int(os.environ.get("PROMPT_REPEAT", "1"))
text = "".join(f"{i}. {base}" for i in range(reps)) + (" Now write a long detailed explanation." if reps > 1 else "")
ids = tok.encode(text)
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"ds41 ids: {len(ids)} -> {sys.argv[1]}")
EOF

# engine shape: the P5.4/P5.5 DS4.1 arms WITHOUT --engram, rank.bin kept, --pcie pinned 0.75 (the deterministic
# default - pinning it spells out that no timing-adaptive admission is in any arm), --slots pinned.
SLOTS=${SLOTS:-1500}
DARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu --rank "$M/rank.bin"
       --ids-file "$IDS_D" -n "$DECODE_N" --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0
       --slots "$SLOTS" --pcie 0.75)

J=logs/p6.4-tier-shards-${TS}.jsonl
run_one() {   # run_one <name> <tier-cards> [extra engine args...]
    sleep 10
    local name=$1 cards=$2; shift 2
    local log=logs/p6.4-${name}-${TS}.log toks=logs/p6.4-${name}-${TS}.toks
    echo "=== p6.4/$name (tier-cards $cards) start $(date +%T) ==="
    bash tools/v41/serve/ds41-engine-906.sh "${DARGS[@]}" --tier-cards "$cards" "$@" >"$toks" 2>"$log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$toks") token ids -> $log"
    [ $rc -eq 0 ] || { echo "P6.4: $name FAILED (rc=$rc)"; return 1; }
    grep -m1 -E "^decode: " "$log" | sed "s/^/[$name] /"
    # guards: the shard line the engine prints, and nothing else the engine refused to say
    local shardline; shardline=$(grep -m1 -oE "tier: [0-9]+ shards" "$log")
    if [ "$cards" -gt 1 ]; then
        [ -n "$shardline" ] || { echo "P6.4 GUARD FAIL: $name (--tier-cards $cards) printed no shard line"; return 1; }
    else
        [ -z "$shardline" ] || { echo "P6.4 GUARD FAIL: control $name printed a shard line"; return 1; }
    fi
    grep -iE "error|refus" "$log" | grep -v "no --engram" | sed "s/^/[$name ERR] /"
    # parse the decode line: tokens, tok/s, expert path counters; and the token stream's common prefix vs round 1 ctl
    "$PY" - "$name" "$cards" "$TS" "$J" "$toks" <<'EOF'
import re, sys, json, os
name, cards, ts, jf, tokf = sys.argv[1:6]
log = f"logs/p6.4-{name}-{ts}.log"
s = open(log, errors="replace").read()
m = re.search(r"decode: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s; experts (\d+) hit / (\d+) cpu / (\d+) pcie / (\d+) file", s)
if not m:
    print(f"[{name}] PARSE FAIL: no decode: line"); sys.exit(1)
nt, sec, tps, hit, cpu, pcie, f = m.groups()
hit, cpu, pcie, f = map(int, (hit, cpu, pcie, f))
lk = hit + cpu + pcie + f
pm = re.search(r"prompt: \d+ tokens in [\d.]+ s = ([\d.]+) tok/s \(experts (\d+) hit / (\d+) cpu / (\d+) pcie / (\d+) file\)", s)
seedm = re.search(r"tier: (\d+) shards \(expert e on card e%\d+\): (\d+) experts in VRAM, (\d+) in the RAM arenas \(([\d.]+) GiB\)", s)
single_seed = re.search(r"tier: (\d+) experts in VRAM, (\d+) in the RAM arena \(([\d.]+) GiB\)", s)
tokens = open(tokf).read().split()
rec = {"arm": name, "tier_cards": int(cards), "decode_tokens": int(nt), "decode_s": float(sec),
       "decode_tps": float(tps), "lookups": lk, "hit": hit, "cpu": cpu, "pcie": pcie, "file": f,
       "hit_frac": round(hit / lk, 4) if lk else None,
       "prompt_tps": float(pm.group(1)) if pm else None,
       "vram_experts": int(seedm.group(2)) if seedm else (int(single_seed.group(1)) if single_seed else None),
       "arena_gib": float(seedm.group(4)) if seedm else (float(single_seed.group(3)) if single_seed else None),
       "ntok_out": len(tokens), "first_tokens": tokens[:12]}
with open(jf, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(f"[{name}] decode {tps} tok/s hit {rec['hit_frac']} cpu {cpu} pcie {pcie} file {f} "
      f"vram_experts {rec['vram_experts']} arena {rec['arena_gib']} GiB")
EOF
    [ $? -eq 0 ] || return 1
}

# throwaway: the first boot after vLLM evicts the GGUF page cache is a 2.4x-slow artifact (P5.5 lesson)
if [ "${WARM:-1}" = 1 ]; then
    echo "=== p6.4 warm (throwaway) ==="; run_one warm 1 || exit 1
fi

# interleaved arms, ROUNDS rounds; ARMS=subset knob for partial re-runs (witness rule: same-session pairs)
ARMS="${ARMS:-ctl,t2,t4,t4e}"
has() { [[ ",$ARMS," == *",$1,"* ]]; }
for r in $(seq 1 "$ROUNDS"); do
    has ctl && { run_one "ctl-r$r" 1 || exit 1; }
    has t2  && { run_one "t2-r$r"  2 || exit 1; }
    has t4  && { run_one "t4-r$r"  4 || exit 1; }
    # equal-capacity 4-shard: SLOTS/4 per shard so 4 x SLOTS/4 = the control's cache, four pipes, same capacity
    has t4e && { run_one "t4e-r$r" 4 --slots $(( SLOTS / 4 )) || exit 1; }
done

# ---- summary: per-arm mean +/- spread over rounds, paired against the SAME-round control
echo "=== P6.4 SUMMARY (session $TS) ==="
"$PY" - "$J" <<'EOF'
import json, sys, statistics as st
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = {}
for r in recs:
    if r["arm"].startswith("warm"): continue
    by.setdefault(r["arm"].rsplit("-r", 1)[0], []).append(r)
def col(vals):
    if not vals: return "-"
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f}+-{sd:.2f} (n={len(vals)})"
for arm in ("ctl", "t2", "t4", "t4e"):
    rs = by.get(arm, [])
    if not rs: continue
    print(f"{arm:4s} tier_cards={rs[0]['tier_cards']} vram_experts={col([r['vram_experts'] for r in rs if r['vram_experts']])} "
          f"arena_GiB={col([r['arena_gib'] for r in rs if r['arena_gib']])} "
          f"hit={col([100*r['hit_frac'] for r in rs if r['hit_frac'] is not None])}% "
          f"decode_tok/s={col([r['decode_tps'] for r in rs])} "
          f"file_reads={col([r['file'] for r in rs])}")
rounds = {}
for r in recs:
    if r["arm"].startswith("warm"): continue
    rounds.setdefault(r["arm"].split("-r")[1] if "-r" in r["arm"] else "?", {})[
        r["arm"].rsplit("-r", 1)[0]] = r
for rnd, arms in sorted(rounds.items()):
    if "ctl" not in arms: continue
    line = f"round {rnd}: decode paired ratios vs ctl ({arms['ctl']['decode_tps']:.2f} tok/s):"
    for a, r in sorted(arms.items()):
        if a != "ctl":
            line += f" {a} {100*(r['decode_tps']/arms['ctl']['decode_tps']-1):+.1f}%"
    print(line)
EOF
echo "session $TS done; per-boot records in $J"
