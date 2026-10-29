#!/usr/bin/env bash
# run-p46-ds41-serve-tiers.sh - task #46: DS4.1 --serve x --tier-cards, ctl vs t4 UNDER P4b concurrency
# (docs/gfx906/P6-TENSOR-SPLIT.md section 10-12, docs/gfx906/P4B-SESSIONS.md).
#
# P6.4 measured DS4.1 shards on ONE-SHOT workloads: NO-GO (the ctl hit was 99.23%, the capacity prize
# +0.2 pt, the paired decode ratio 0.72-0.91). P4b changed the load shape the server serves: concurrent
# sessions now slice turns and park per conversation, which presses the arena in a way one-shot never
# did. This runner re-measures THAT: identical concurrent-replay load on two arms that differ by exactly
# one flag (--tier-cards 4), warm + two interleaved rounds, within-arm spread reported, paired ratios per
# round (the standing perf rule). If concurrency starves the ctl cache the P6.4 verdict may flip; if not,
# the single-workload NO-GO stands and this records why under the new load.
#
# Refusals: MemAvailable < 65 GiB; a live engine/serve wrapper; a binary older than its sources.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
PY=/root/.pyenv/versions/venv312/bin/python3
CFG0=tools/v41/serve/strata-ds41-906.json
PORT=${PORT:-8152}
JSONL=logs/p46-ds41-tiers-$TS.jsonl
R=benchmarks/gfx906/receipts/$(date +%Y-%m-%d)-p46-ds41-serve-tiers.json
ROUNDS=${ROUNDS:-2}
T1MAX=${T1MAX:-64}
T2MAX=${T2MAX:-32}
SLOTS=${SLOTS:-4}

free_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
[ "$(free_gib)" -ge 65 ] || { echo "refusing: MemAvailable $(free_gib) GiB < 65 (the batch-script rule)"; exit 3; }
for pat in ds41_generate glm_generate; do pgrep -f "$pat --serve" >/dev/null && { echo "refusing: $pat alive"; exit 3; }; done
pgrep -f "serve_ds41.py|serve_glm.py" >/dev/null && { echo "refusing: a serve wrapper is alive"; exit 3; }
for src in tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_moe.cpp; do
    [ build-906/ds41_generate -nt "$src" ] || { echo "refusing: ds41_generate older than $src"; exit 3; }
done

mk_cfg() { # $1 = arm (ctl|t4) -> "cfg elog"
    local cfg="logs/p46-$TS.$1.json" elog="logs/p46-$TS.$1.engine.log"
    $PY - "$CFG0" "$cfg" "$ROOT" "$elog" "$1" "$SLOTS" <<'EOF'
import json, sys
src, dst, root, elog, arm, slots = sys.argv[1:7]
cfg = json.load(open(src))
cfg["exe"] = root + "/tools/v41/serve/ds41-engine-906.sh"
cfg["cwd"] = root
cfg["log"] = elog                                        # replace the WHOLE value (the doubled-path trap)
cfg["args"] = list(cfg["args"]) + ["--park-slots", slots]
if arm == "t4":
    cfg["args"] += ["--tier-cards", "4"]
    cfg.pop("gpu", None)   # the "gpu" key pins HIP_VISIBLE_DEVICES (server.py child_env); shards need all cards
json.dump(cfg, open(dst, "w"), indent=1)
EOF
    echo "$cfg $elog"
}

boot() { # $1 = arm -> sets SPID ELOG; waits for /health loaded
    local cfg
    read -r cfg ELOG <<<"$(mk_cfg "$1")" || return 1
    echo "=== p46/$1 server start $(date +%H:%M:%S) (engine log: $ELOG) ==="
    STRATA_SERVE_STEP_MAX=96 STRATA_NO_LARGEPAGES=1 STRATA_SERVE_PARK_MIN=${PARK_MIN:-64} \
        $PY tools/v41/serve/serve_ds41.py \
        --engine strata --config "$cfg" --port "$PORT" >"logs/p46-$TS.$1.server.log" 2>&1 & SPID=$!
    local waited=0 st
    while :; do
        kill -0 $SPID 2>/dev/null || { echo "FAIL $1: server died - see logs/p46-$TS.$1.server.log"; return 1; }
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
        waited=$((waited + 5)); [ $waited -ge 600 ] && { echo "FAIL $1: not loaded after 600 s"; kill $SPID 2>/dev/null; return 1; }
        sleep 5
    done
    echo "[$1] loaded after ${waited} s"
}

kill_server() {
    [ -n "${SPID:-}" ] && kill $SPID 2>/dev/null
    for _ in $(seq 30); do pgrep -f "ds41_generate --serve" >/dev/null || break; sleep 2; done
    pkill -f "ds41_generate --serve" 2>/dev/null
    unset SPID; sleep 15                                   # the async VRAM-free rule after a killed engine
}

client() { # $1=phase (warm|measure) $2=arm $3=round
    $PY - "$PORT" "$1" "$2" "$3" "$TS" "$JSONL" "$T1MAX" "$T2MAX" <<'EOF'
import json, sys, threading, time, urllib.request
port, phase, arm, rnd, ts, jsonl, t1max, t2max = sys.argv[1:9]
CONVS = {
    "A": ("Explain, in numbered order, what happens to one token inside a mixture-of-experts transformer "
          "layer from embedding to output logits.",
          "Now rewrite step 3 assuming two experts are pruned at runtime. Keep it under 80 words."),
    "B": ("A machine has four GPUs on PCIe 4.0 x16 and one NVMe holding model experts. Enumerate the three "
          "memory tiers and give a bandwidth number for each.",
          "Which tier does a hot expert land in first when the VRAM cache is half full? Answer in two sentences."),
    "C": ("Write a short technical note comparing HBM2 and GDDR6 for inference. Enumerate exactly three "
          "consequences, no more.",
          "Add one sentence about power draw to your note and number it four."),
}
def ask(msgs, maxtok):
    body = {"messages": msgs, "max_tokens": maxtok, "temperature": 0, "reasoning_effort": "none", "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = []
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except Exception:
                    continue
                ch = (j.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.time() - t0
                text.append(piece or "")
        return {"text": "".join(text), "ttft": first, "wall": time.time() - t0, "error": None}
    except Exception as e:
        return {"text": "", "ttft": None, "wall": time.time() - t0, "error": str(e)}
out = []
lock = threading.Lock()
def turn(conv, deep):
    r = ask([{"role": "user", "content": CONVS[conv][0]}], int(t1max))
    r.update(conv=conv, turn=1)
    with lock:
        out.append(r)
    if not deep or r["error"]:
        return
    r2 = ask([{"role": "user", "content": CONVS[conv][0]}, {"role": "assistant", "content": r["text"]},
              {"role": "user", "content": CONVS[conv][1]}], int(t2max))
    r2.update(conv=conv, turn=2)
    with lock:
        out.append(r2)
ths = [threading.Thread(target=turn, args=(c, phase == "measure")) for c in sorted(CONVS)]
for t in ths:
    t.start()
for t in ths:
    t.join(900)
with open(jsonl, "a") as fh:
    for r in out:
        r.update(phase=phase, arm=arm, round=rnd, ts=ts)
        fh.write(json.dumps(r) + "\n")
errs = [r for r in out if r["error"]]
print("%s/%s r%s: %d requests, %d errors" % (arm, phase, rnd, len(out), len(errs)))
EOF
}

logstats() { # $1=engine log $2=start byte $3=arm $4=round -> appends one JSONL line with tier stats
    $PY - "$ELOG" "$2" "$3" "$4" "$JSONL" <<'EOF'
import json, re, sys
elog, start, arm, rnd, jsonl = sys.argv[1:6]
data = open(elog, "r", errors="replace").read()[int(start):]
dec = re.findall(r"serve: (\d+) tokens in ([\d.]+) s = [\d.]+ tok/s \([^)]*\); experts: (\d+) hit, (\d+) cpu, (\d+) pcie, (\d+) file", data)
prompt = re.findall(r"serve: prompt \d+ tokens \(\d+ [^)]*\) in [\d.]+ s = [\d.]+ tok/s \(experts (\d+) hit / (\d+) cpu / (\d+) pcie / (\d+) file", data)
toks = sum(int(d[0]) for d in dec); secs = sum(float(d[1]) for d in dec)
h = sum(int(d[2]) for d in dec); c = sum(int(d[3]) for d in dec); p = sum(int(d[4]) for d in dec); f = sum(int(d[5]) for d in dec)
ph = sum(int(d[0]) for d in prompt); pc = sum(int(d[1]) for d in prompt)
pp = sum(int(d[2]) for d in prompt); pf = sum(int(d[3]) for d in prompt)
line = {"arm": arm, "round": rnd, "decode_tokens": toks, "decode_s": round(secs, 2),
        "decode_toks_s": round(toks / secs, 3) if secs else None,
        "decode_hit_pct": round(100.0 * h / max(1, h + c + p + f), 2),
        "decode_lookups": h + c + p + f, "decode_cpu": c, "decode_pcie": p, "decode_file": f,
        "prompt_hit_pct": round(100.0 * ph / max(1, ph + pc + pp + pf), 2) if (ph + pc + pp + pf) else None}
with open(jsonl, "a") as fh:
    fh.write(json.dumps(line) + "\n")
print("stats[%s r%s]: decode %s tok/s, hit %s%% (lookups %d), prompt hit %s%%"
      % (arm, rnd, line["decode_toks_s"], line["decode_hit_pct"], line["decode_lookups"], line["prompt_hit_pct"]))
EOF
}

for r in $(seq 1 "$ROUNDS"); do
    for arm in ctl t4; do
        boot "$arm" || exit 1
        START=$(wc -c <"$ELOG")
        client warm "$arm" "$r"
        sleep 5
        START=$(wc -c <"$ELOG")                # warm's traffic is NOT in the measured window
        client measure "$arm" "$r"
        logstats "$ELOG" "$START" "$arm" "$r"
        kill_server
    done
done

echo "== verdict =="
$PY - "$JSONL" "$R" "$TS" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
stats = [r for r in rows if "decode_toks_s" in r]
reqs = [r for r in rows if "phase" in r]
errs = [r for r in reqs if r.get("error")]
meas = [r for r in reqs if r.get("phase") == "measure"]
by = {}
for s in stats:
    by[(s["arm"], s["round"])] = s
rounds = sorted({k[1] for k in by})
ratios = {}
for rd in rounds:
    a, b = by.get(("ctl", rd)), by.get(("t4", rd))
    if a and b and a["decode_toks_s"] and b["decode_toks_s"]:
        ratios[rd] = {"decode_t4/ctl": round(b["decode_toks_s"] / a["decode_toks_s"], 3),
                      "hit_ctl": a["decode_hit_pct"], "hit_t4": b["decode_hit_pct"]}
go = bool(ratios) and all(v["decode_t4/ctl"] > 1.0 for v in ratios.values()) and not errs
json.dump({"gate": "#46 DS4.1 --serve x --tier-cards under P4b concurrent sessions", "ts": sys.argv[3],
           "commit": __import__("subprocess").run(["git", "rev-parse", "--short", "HEAD"],
                                                  capture_output=True, text=True).stdout.strip(),
           "arms_differ_by": "--tier-cards 4 (t4 also omits the config gpu key: shards need all cards)",
           "load": "concurrent 3-session x 2-turn replay, warm burst before each measured window",
           "per_arm": stats, "paired": ratios, "requests": len(reqs), "errors": len(errs),
           "verdict": ("GO: t4 decode beats ctl under concurrent load in every paired round" if go else
                       "NO-GO: the P6.4 single-workload verdict stands under the concurrent load too "
                       "(paired decode ratio at or below 1.0; see per-round numbers)")},
          open(sys.argv[2], "w"), indent=1)
print(json.dumps({"paired": ratios, "errors": len(errs), "requests": len(reqs),
                  "verdict_pass": go}, indent=1))
EOF
echo "jsonl: $JSONL ; receipt: $R"
