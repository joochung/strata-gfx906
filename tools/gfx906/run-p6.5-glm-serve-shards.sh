#!/bin/bash
# P6.5 (task #44): SERVE-path witness for GLM --tier-cards.  P6.4b (task #42) proved the 4-card expert-parallel
# tier on single-shot boots (+66% decode, hit 21.8 -> 78.7%); the serve path refused shards until #44 removed
# the refusal (the audit: serve's decode already goes through the shard seam, elastic --vram-grow is shard 0's by
# construction, INFO/DONE/saliency counts aggregate).  This witness runs the REAL stack (serve_glm.py over
# serve/server.py over glm_generate --serve) with a fixed six-request chat set and asks the same question as
# P6.4b: does the AGGREGATED VRAM EXPERT CACHE carry into serving?
#
# Protocol (the standing perf rule, DSpark/P5.5/P6.4): ONE throwaway WARM boot (page cache - never skip after a
# vLLM (re)start, the 2.4x first-boot artifact), then ROUNDS x (ctl, s4) INTERLEAVED, within-arm spread reported.
# Arms differ by exactly one flag (--tier-cards 4); the configs say so.  Rate claims come from the PAIRED ratios
# per round, not from single boots.  Token streams are NOT compared bit-for-bit: shard residency differs by
# design (P6.4b rule) and the production admission is --pcie auto (timing-adaptive - the #40 confound is fine
# for a RATE witness, fatal for a PARITY witness, so none is claimed here).
#
# Launch through the GPU wrapper (it stops the stack, waits for idle VRAM, restarts on exit):
#   /ai/engines/gfx906/unverbraucht/run-with-free-gpus.sh bash -c \
#     "cd /ai/engines/gfx906/worktrees/p5.3-relay && bash tools/gfx906/run-p6.5-glm-serve-shards.sh"
#
# Per boot: engine load ~3 min + six 128-token greedy requests ~2 min.  Default warm + 2 rounds x 2 arms = 5
# boots ~= 35-40 min.  ONE ENGINE AT A TIME (SVM crash rule); MemAvailable >= 85 GiB (the serve scope caps at
# 84G, and vLLM must be down for the wrapper to run this anyway).
set -u
cd "$(dirname "$0")/../.."
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate|serve_glm" >/dev/null 2>&1; then
    echo "refusing: another engine/server is alive - one HIP engine at a time (SVM crash rule)"; exit 3
fi
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge 85 ] || { echo "refusing: ${AVAIL} GiB available < 85 (serve scope MemoryMax=84G)"; exit 3; }

GEN_G=build-906/glm_generate
[[ -x "$GEN_G" ]] || { echo "FATAL: missing $GEN_G - build first"; exit 3; }
for src in tools/glm/glm_generate.cpp tools/ds4/ds4_moe.cpp; do
    [ "$GEN_G" -ot "$src" ] && { echo "FATAL: $GEN_G older than $src"; exit 3; }
done

PORT=${PORT:-8147}
ROUNDS=${ROUNDS:-2}
MAXTOK=${MAXTOK:-128}
JSONL=logs/p6.5-serve-$TS.jsonl
PY=/root/.pyenv/versions/venv312/bin/python3
CFG_CTL=tools/glm/serve/strata-glm-mayal-906-ctl.json
CFG_T4=tools/glm/serve/strata-glm-mayal-906-t4.json

# per-boot engine stderr (config "log" rewritten per boot so boots never share a file)
mk_cfg(){ # arm round -> path
    local arm=$1 r=$2 base logn out
    [ "$arm" = s4 ] && base=$CFG_T4 || base=$CFG_CTL
    logn="$PWD/logs/p6.5-engine-$arm-r$r-$TS.log"
    out="logs/p6.5-cfg-$arm-r$r-$TS.json"
    sed "s#\"log\": \"[^\"]*\"#\"log\": \"$logn\"#" "$base" > "$out"
    echo "$out $logn"
}

boot(){ # arm round
    local arm=$1 r=$2 cfg elog pid waited=0
    read -r cfg elog <<<"$(mk_cfg "$arm" "$r")" || return 1
    echo "=== p6.5/$arm-r$r server start $(date +%H:%M:%S) (engine log: ${cfg%.json}.log or config) ==="
    $PY tools/glm/serve/serve_glm.py --engine strata --config "$cfg" --port "$PORT" \
        >"logs/p6.5-$arm-r$r-$TS.server.log" 2>&1 & pid=$!
    # wait for the engine to report READY/loaded through /health (load is ~3 min for 156 GB)
    while :; do
        if ! kill -0 $pid 2>/dev/null; then echo "FAIL $arm-r$r: server died - see logs/p6.5-$arm-r$r-$TS.server.log"; return 1; fi
        st=$($PY - "$PORT" <<'EOF' 2>/dev/null || true
import json, sys, urllib.request
try:
    d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=3))
    print("loaded" if d.get("loaded") else "loading")
except Exception:
    print("wait")
EOF
)
        [ "$st" = loaded ] && break
        waited=$((waited + 5)); [ $waited -ge 480 ] && { echo "FAIL $arm-r$r: not loaded after 480 s"; kill $pid 2>/dev/null; return 1; }
        sleep 5
    done
    echo "[$arm-r$r] loaded after ${waited} s; INFO: $(grep -m1 'INFO ctx=' "$elog" 2>/dev/null || echo none yet)"
    # the fixed request set (greedy, MAXTOK each): identical for every boot
    $PY - "$PORT" "$arm" "$r" "$TS" "$JSONL" "$MAXTOK" <<'EOF' || echo "WARN $arm-r$r: client exit nonzero"
import json, sys, time, urllib.request
port, arm, rnd, ts, jsonl, maxtok = sys.argv[1:7]
PROMPTS = [
    "Explain, in order, what happens to one token inside a mixture-of-experts transformer layer, from embedding to output.",
    "Write a short technical note comparing PCIe 4.0 x16 bandwidth with HBM2 bandwidth. Enumerate three consequences for MoE inference.",
    "List and number the steps needed to cache 15,000 expert tensors across four GPU cards. Then state the failure modes.",
    "A model server answers 1.9 tokens per second with one card and 3.2 with four. Enumerate the three most likely mechanisms.",
    "Describe why dividing CPU worker cores across cards does NOT speed up a memory-bandwidth-bound expert pool.",
    "Summarize in ten numbered points how a hierarchical SSD -> RAM -> VRAM expert cache decides where an expert lives.",
]
for i, p in enumerate(PROMPTS):
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": p}],
                       "max_tokens": int(maxtok), "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        el = time.time() - t0
        u = d.get("usage", {})
        ct = d.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        row = {"ts": ts, "arm": arm, "round": int(rnd), "req": i, "s": round(el, 2),
               "completion_tokens": u.get("completion_tokens", 0), "prompt_tokens": u.get("prompt_tokens", 0),
               "chars": len(ct)}
    except Exception as e:
        row = {"ts": ts, "arm": arm, "round": int(rnd), "req": i, "s": round(time.time() - t0, 2),
               "completion_tokens": 0, "prompt_tokens": 0, "chars": 0, "err": str(e)[:140]}
    with open(jsonl, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{arm} r{rnd} q{i}] {row['completion_tokens']} tok in {row['s']}s" + (" ERR " + row.get("err", "") if row.get("err") else ""))
EOF
    kill $pid 2>/dev/null   # SIGTERM -> server.py's Ctrl+C path: QUIT to the engine, then terminate()
    for _ in $(seq 30); do pgrep -f glm_generate >/dev/null 2>&1 || break; sleep 2; done
    pkill -f "glm_generate --serve" 2>/dev/null   # belt and braces: a held GPU wedges the next boot
    wait $pid 2>/dev/null
    sleep 15   # SIGABRT'd/KILLed HIP procs free VRAM asynchronously (crash-rule lesson)
}

WARM=${WARM:-1}
[ "$WARM" = 1 ] && { echo "=== throwaway WARM boot (page cache; the 2.4x first-boot artifact) ==="; boot s4 0 || exit 1; }
for r in $(seq 1 "$ROUNDS"); do
    boot ctl "$r" || exit 1
    boot s4  "$r" || exit 1
done

echo
echo "== INFO lines (engine logs): experts_vram must be ~4x on s4 (the #44 INFO aggregation check) =="
for f in logs/p6.5-engine-*-$TS.log; do
    [ -f "$f" ] || continue
    echo "  $(basename "$f"): $(grep -m1 -o 'INFO ctx=[0-9]* experts_vram=[0-9]* arena_experts=[0-9]*' "$f" | tr '\n' ' ')"
done

echo
echo "== paired summary (per round: sum over the six requests; s4/ctl ratio) =="
$PY - "$JSONL" <<'EOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
agg = collections.defaultdict(lambda: {"s": 0.0, "tok": 0, "err": 0})
for r in rows:
    k = (r["round"], r["arm"])
    agg[k]["s"] += r["s"]; agg[k]["tok"] += r["completion_tokens"]; agg[k]["err"] += 1 if r.get("err") else 0
rounds = sorted({k[0] for k in agg if k[1] == "ctl" and k[0] > 0})
for rd in rounds:
    c, s = agg.get((rd, "ctl")), agg.get((rd, "s4"))
    if not c or not s: continue
    rc, rs = c["tok"] / max(c["s"], 1e-9), s["tok"] / max(s["s"], 1e-9)
    print(f"  round {rd}: ctl {rc:5.2f} tok/s ({c['tok']} tok, err {c['err']})   s4 {rs:5.2f} tok/s"
          f" ({s['tok']} tok, err {s['err']})   ratio {rs / max(rc, 1e-9):.3f}")
per = collections.defaultdict(list)
for rd in rounds:
    c, s = agg.get((rd, "ctl")), agg.get((rd, "s4"))
    if c and s and c["tok"] and s["tok"]:
        per["ctl"].append(c["tok"] / c["s"]); per["s4"].append(s["tok"] / s["s"])
for a, v in per.items():
    if len(v) >= 2:
        m = sum(v) / len(v)
        print(f"  {a}: mean {m:.2f} tok/s over {len(v)} rounds, within-arm spread {min(v):.2f}-{max(v):.2f}")
EOF

echo
echo "artifacts: logs/p6.5-serve-$TS.jsonl, logs/p6.5-*-$TS.{log,server.log}, logs/p6.5-cfg-*-$TS.json"
echo "P6.5 done $(date +%H:%M:%S)"
