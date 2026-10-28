#!/bin/bash
# P6.4b (task #42): the EXPERT-PARALLEL TIER witness for GLM Maya-L - the CACHE-STARVED engine the shard
# design was built for.  DS4.1's P6.4 verdict (docs/gfx906/P6-TENSOR-SPLIT.md section 10) was NO-GO because
# its decode hit 99.23% at slots 1500: nothing left to gain, only the N serialized tier submissions to pay.
# GLM Maya-L is the opposite operating point: hit 12.4%, and the CPU pool is 75.9% of the decode wall
# (275-285 of 306-337 ms/token).  Here the shards divide the three things DS4.1 could not: pool CORES
# (disjoint physical worker slices), RAM arenas, and byte supply (N file readers + N PCIe lanes).
#   SUCCESS GATE: GLM decode wall must DROP (pool work divides), not just hit rise.
# Launch through the GPU wrapper (it stops the stack, waits for idle VRAM, restarts on exit):
#   /ai/engines/gfx906/unverbraucht/run-with-free-gpus.sh bash -c \
#     "cd /ai/engines/gfx906/worktrees/p5.3-relay && bash tools/gfx906/run-p6.4b-glm-tier-shards.sh"
#
# Arms INTERLEAVE (ctl,t2,t4 per round, ROUNDS>=2) with the WITHIN-ARM SPREAD reported - the standing rule
# (DSpark swap-order, P5.5, P6.4): a same-session pair without repetition supports no rate claim.
# THE PARITY BAR IS NOT bit-parity vs control: the fan-in sums shard partials in shard order, a different
# float grouping than the single-card sum (host gate test_ds4_moe --shards: 3.8e-08, red controls 0.44+).
# Knife-edge token flips are EXPECTED; the common token prefix vs the same round's ctl is reported, gated
# only as an informational prefix length.
#
# Host gates BEFORE any GPU boot: test_ds4_moe --shards 4 (real Maya-L experts, must show rel < 1e-5 lines),
# binary freshness.  MemAvailable floor = arena + 25.  ONE ENGINE AT A TIME (SVM crash rule).
#
# Cost per boot ~4-6 min (156 GB GGUF).  Default: throwaway WARM boot + 2 rounds x 3 arms = 7 boots ~= 45 min.
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

MODEL=/ai/models/hf.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-L/GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf
GEN_G=build-906/glm_generate
[[ -x "$GEN_G" ]] || { echo "FATAL: missing $GEN_G - build with tools/gfx906/build-906-gate.sh first"; exit 3; }
for src in tools/ds4/ds4_moe.cpp tools/ds4/ds4_moe.hpp tools/glm/glm_generate.cpp src/kernels/cpu/pool.cpp; do
    [ "$GEN_G" -ot "$src" ] && { echo "FATAL: $GEN_G older than $src"; exit 3; }
done

# the host fan-out gate on REAL Maya-L experts (arm S of tools/ds4/test_ds4_moe.cpp).  NOTE the pre-existing
# arm-2 L20 "rel nan" failures on this model (witnessed identical at HEAD, NOT from --tier-cards): the gate
# here is arm S only (--no-synthetic --no-real), whose own check count must be > 0 with 0 failures.
echo "=== host fan-out gate (test_ds4_moe --shards 4, arm S only) ==="
GATE_LOG=logs/p6.4b-hostgate-${TS}.log
./build-906/test_ds4_moe --no-synthetic --no-real --shards 4 --gguf "$MODEL" >"$GATE_LOG" 2>&1 \
    || { echo "FATAL: host fan-out gate exit nonzero - see $GATE_LOG"; exit 3; }
GATE_N=$(grep -c "shards 4: rel" "$GATE_LOG")
GATE_F=$(grep -m1 -oE "[0-9]+ failures" "$GATE_LOG" | awk '{print $1}')
[ "$GATE_N" -gt 0 ] && [ "${GATE_F:-1}" = 0 ] || { echo "FATAL: gate checks $GATE_N failures ${GATE_F:-?} - see $GATE_LOG"; exit 3; }
grep "shards 4: rel" "$GATE_LOG" | sed 's/^/[hostgate] /'

PY=/root/.pyenv/versions/venv312/bin/python3
DECODE_N="${DECODE_N:-128}"
ROUNDS="${ROUNDS:-2}"
IDS_G=logs/p6.4b-glm-ids-${TS}.i32
# the P4.2 prompt recipe (tokenizer build exactly as serve_glm.py does); DECODE_N tokens after it
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
print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
EOF

# engine shape: the witnessed P4.2/serve recipe (margin 4.0 per the c_stage arithmetic in run-p4.2; --vram-lru
# and --arena-adapt as production runs them; --pcie pinned so no timing-adaptive admission is in any arm -
# the task #40 confound spelled out of the protocol).  --slots AUTO here: the capacity gain IS the point on
# this engine (each shard sizes to its own card); vram_experts is reported per arm so the reader can divide.
GARGS=(-m "$MODEL" --backend cuda --experts gpu --ids-file "$IDS_G" -n "$DECODE_N"
       --arena-gib "$ARENA_GIB" --arena-adapt --vram-lru --vram-margin 4.0 --pcie 0.25 --lora
       /ai/models/hf.co/GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf)

J=logs/p6.4b-glm-tier-shards-${TS}.jsonl
run_one() {   # run_one <name> <tier-cards> [extra engine args...]
    sleep 10
    local name=$1 cards=$2; shift 2
    local log=logs/p6.4b-${name}-${TS}.log toks=logs/p6.4b-${name}-${TS}.toks
    echo "=== p6.4b/$name (tier-cards $cards) start $(date +%T) ==="
    bash tools/glm/serve/glm-engine-906.sh "${GARGS[@]}" --tier-cards "$cards" "$@" >"$toks" 2>"$log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$toks") token ids -> $log"
    [ $rc -eq 0 ] || { echo "P6.4B: $name FAILED (rc=$rc)"; return 1; }
    grep -m1 -E "^decode : " "$log" | sed "s/^/[$name] /"
    # guard: the shard line must appear exactly when cards > 1
    local shardline; shardline=$(grep -m1 -oE "tier: [a-z ]+, [0-9]+ shards" "$log")
    if [ "$cards" -gt 1 ]; then
        [ -n "$shardline" ] || { echo "P6.4B GUARD FAIL: $name (--tier-cards $cards) printed no shard line"; return 1; }
    else
        [ -z "$shardline" ] || { echo "P6.4B GUARD FAIL: control $name printed a shard line"; return 1; }
    fi
    grep -iE "error|refus|traceback" "$log" | sed "s/^/[$name ERR] /"
    "$PY" - "$name" "$cards" "$TS" "$J" "$toks" <<'EOF'
import re, sys, json
name, cards, ts, jf, tokf = sys.argv[1:6]
log = f"logs/p6.4b-{name}-{ts}.log"
s = open(log, errors="replace").read()
dm = re.search(r"decode : (\d+) tokens, (\d+) forward passes in ([\d.]+) s = ([\d.]+) tok/s", s)
pm = re.search(r"prefill: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s", s)
em = re.search(r"experts/token: hit ([\d.]+)% .*?cpu ([\d.]+)%, pcie ([\d.]+)%, file tier (\d+)", s)
tm = re.search(r"tier ms/token: wall ([\d.]+) \(gpu hits ([\d.]+), pcie ([\d.]+), cpu pool ([\d.]+)\)", s)
seed = re.search(r"tier: \w+, (\d+) shards \(expert e on card e%\d+\): (\d+) resident slots, arenas ([\d.]+) GiB \((\d+) experts\)", s)
single = re.search(r"tier: \w+, (\d+) resident slots, arena ([\d.]+) GiB \((\d+) experts\)", s)
if not (dm and em and tm):
    print(f"[{name}] PARSE FAIL: decode/experts/tier line missing"); sys.exit(1)
nt, steps, sec, tps = float(dm.group(1)), int(dm.group(2)), float(dm.group(3)), float(dm.group(4))
hit, cpu, pcie, file_t = float(em.group(1)), float(em.group(2)), float(em.group(3)), int(em.group(4))
wall, hit_ms, pcie_ms, cpu_ms = map(float, tm.groups())
tokens = open(tokf).read().split()
rec = {"arm": name, "tier_cards": int(cards), "decode_tokens": int(nt), "decode_s": sec, "decode_tps": tps,
       "hit_pct": hit, "cpu_pct": cpu, "pcie_pct": pcie, "file_tier": file_t,
       "tier_wall_ms_tok": wall, "tier_hit_ms_tok": hit_ms, "tier_pcie_ms_tok": pcie_ms, "tier_cpu_ms_tok": cpu_ms,
       "prefill_tps": float(pm.group(3)) if pm else None,
       "vram_slots": int(seed.group(2)) if seed else (int(single.group(1)) if single else None),
       "arena_gib": float(seed.group(3)) if seed else (float(single.group(2)) if single else None),
       "ntok_out": len(tokens), "first_tokens": tokens[:12]}
with open(jf, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(f"[{name}] decode {tps} tok/s  hit {hit}% cpu {cpu}%  wall {wall} ms/tok (cpu pool {cpu_ms}) "
      f"slots {rec['vram_slots']} arena {rec['arena_gib']} GiB")
EOF
    [ $? -eq 0 ] || return 1
}

# throwaway: the first boot after vLLM evicts the GGUF page cache is a 2.4x-slow artifact (P5.5 lesson).
# NEVER WARM=0 after a vLLM restart (P6.4 lesson).
if [ "${WARM:-1}" = 1 ]; then
    echo "=== p6.4b warm (throwaway) ==="; run_one warm 1 || exit 1
fi
# t2 smoke boot: multi-shard init (tier_x on device 1, divided arenas, disjoint pool slices, seed) proven
# with -n 2 before the ladder spends a boot on it.  Its record lands in the jsonl like any other boot.
if [ "${SMOKE:-1}" = 1 ]; then
    echo "=== p6.4b t2-smoke ==="; run_one t2-smoke 2 -n 2 || exit 1
fi

# interleaved arms, ROUNDS rounds (perf witness rule); ARMS=subset knob for partial re-runs
ARMS="${ARMS:-ctl,t2,t4}"
has() { [[ ",$ARMS," == *",$1,"* ]]; }
for r in $(seq 1 "$ROUNDS"); do
    has ctl && { run_one "ctl-r$r" 1 || exit 1; }
    has t2  && { run_one "t2-r$r"  2 || exit 1; }
    has t4  && { run_one "t4-r$r"  4 || exit 1; }
    # EQUAL-CAPACITY 4-shard (the t4e attribution of P6.4, mirrored): --slots 484 x 4 = the control's auto 1934.
    # Separates the CAPACITY gain (4x slots -> hit 21.8% -> 78.7%) from the FAN-OUT gain (same cache, four
    # pools / arenas / byte pipes).  SLOTS_PER_SHARD overrides 484 if the control's auto sizing moves.
    has t4e && { run_one "t4e-r$r" 4 --slots "${SLOTS_PER_SHARD:-484}" || exit 1; }
done

echo "=== P6.4b SUMMARY (session $TS) ==="
"$PY" - "$J" <<'EOF'
import json, sys, statistics as st
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = {}
for r in recs:
    if r["arm"].startswith("warm"): continue
    by.setdefault(r["arm"].rsplit("-r", 1)[0], []).append(r)
def col(vals, fmt="{:.2f}"):
    vals = [v for v in vals if v is not None]
    if not vals: return "-"
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return (fmt + "+-" + fmt + " (n={})").format(m, sd, len(vals))
for arm in ("ctl", "t2", "t4", "t4e"):
    rs = by.get(arm, [])
    if not rs: continue
    print(f"{arm:4s} tier_cards={rs[0]['tier_cards']} slots={col([r['vram_slots'] for r in rs], '{:.0f}')} "
          f"arena_GiB={col([r['arena_gib'] for r in rs])} hit%={col([r['hit_pct'] for r in rs], '{:.1f}')} "
          f"cpu%={col([r['cpu_pct'] for r in rs], '{:.1f}')}")
    print(f"     decode_tok/s={col([r['decode_tps'] for r in rs])} "
          f"tier_wall_ms/tok={col([r['tier_wall_ms_tok'] for r in rs])} "
          f"cpu_pool_ms/tok={col([r['tier_cpu_ms_tok'] for r in rs])} "
          f"prefill_tok/s={col([r['prefill_tps'] for r in rs])}")
# paired ratios vs the SAME-round control (the only comparison the interleaving buys)
rounds = {}
for r in recs:
    if r["arm"].startswith("warm"): continue
    rnd = r["arm"].split("-r")[1] if "-r" in r["arm"] else "?"
    rounds.setdefault(rnd, {})[r["arm"].rsplit("-r", 1)[0]] = r
for rnd, arms in sorted(rounds.items()):
    if "ctl" not in arms: continue
    for arm in ("t2", "t4", "t4e"):
        if arm in arms:
            a, c = arms[arm], arms["ctl"]
            print(f"round {rnd}: {arm}/ctl decode {a['decode_tps']/c['decode_tps']:.3f} "
                  f"tier_wall {a['tier_wall_ms_tok']/c['tier_wall_ms_tok']:.3f} "
                  f"cpu_pool {a['tier_cpu_ms_tok']/c['tier_cpu_ms_tok']:.3f} "
                  f"prefill {(a['prefill_tps']/c['prefill_tps']):.3f}" if a['prefill_tps'] and c['prefill_tps'] else "")
EOF
echo "jsonl: logs/p6.4b-glm-tier-shards-${TS}.jsonl"
