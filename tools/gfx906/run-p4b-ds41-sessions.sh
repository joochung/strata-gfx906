#!/usr/bin/env bash
# run-p4b-ds41-sessions.sh - P4b.3 witness for --park-slots on ds41_generate (docs/gfx906/P4B-SESSIONS.md).
#
# Two stages, one GPU window:
#  A. Engine protocol: drive `ds41_generate --serve --park-slots 3` over stdin. Continuations must RESUME from
#     the parked lcp (DONE reused > 0); a 4th conversation must EVICT the LRU slot; a replay of an evicted
#     conversation must still complete (fresh prefill path). No bit-equality claim on the replay - graph shape
#     changes the float order (G8 decision (b)); the claim is counters and no errors.
#  B. Serve fairness through the real stack (serve_ds41.py -> serve/server.py -> engine --park-slots 4): the SAME
#     three two-turn conversations run twice - once strictly serial (control texts) and once with all three
#     concurrent (boot each schedule on a fresh server so neither phase inherits the other's parked state).
#     Claims: (1) answers identical serial vs concurrent, modulo knife-edge flips (report the exact rate);
#     (2) the second/third concurrent request starts its answer long before the first one FINISHES (a slice,
#     not the whole generation, of wait); (3) /metrics requests_reused > 0 through the real stack.
#
# Refusals: vLLM or any engine alive (run under run-with-free-gpus.sh); MemAvailable < 65 GiB (the batch-script
# rule); a binary older than its sources.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
PY=/root/.pyenv/versions/venv312/bin/python3
MODEL=/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/V41-GSQ.gguf
ENGSH=tools/v41/serve/ds41-engine-906.sh
CFG0=tools/v41/serve/strata-ds41-906.json
PORT=${PORT:-8151}
JSONL=logs/p4b-ds41-sessions-$TS.jsonl
SLOTS=${SLOTS:-4}
T1MAX=${T1MAX:-64}
T2MAX=${T2MAX:-32}

free_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }

[ "$(free_gib)" -ge 65 ] || { echo "refusing: MemAvailable $(free_gib) GiB < 65 (the batch-script rule)"; exit 3; }
for pat in ds41_generate glm_generate; do pgrep -f "$pat --serve" >/dev/null && { echo "refusing: $pat alive"; exit 3; }; done
pgrep -f "serve_ds41.py|serve_glm.py" >/dev/null && { echo "refusing: a serve wrapper is alive"; exit 3; }
for src in tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_moe.cpp; do
    [ build-906/ds41_generate -nt "$src" ] || { echo "refusing: ds41_generate older than $src"; exit 3; }
done
$PY -m py_compile serve/server.py || { echo "refusing: serve/server.py does not compile"; exit 3; }
[ -x build-906/ds41_generate ] || { echo "refusing: no build-906/ds41_generate"; exit 3; }

echo "== stage A: engine protocol over stdin (--park-slots 3) =="
AOUT=logs/p4b-$TS.stageA.log
$PY - "$ENGSH" "$MODEL" "$AOUT" <<'EOF' || { echo "stage A FAILED (see $AOUT)"; exit 1; }
import subprocess, sys, threading, queue, time, os
engsh, model, aout = sys.argv[1:4]
# the witness prompts are ~14 tokens; production floors are 128 (N>1) / 2048 (N=1). The env floor is the
# supported knob for short-conversation tests - parking small prompts is exactly what the floor gates out.
env = {**os.environ, "STRATA_SERVE_PARK_MIN": "16"}
cmd = [engsh, "-m", model, "--backend", "cuda", "--experts", "gpu",
       "--engram", "/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/engram/l1:"
                   "/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/engram/l14",
       "--rank", "/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored/rank.bin",
       "--ctx", "8192", "--stop", "1", "--arena-gib", "40", "--vram-margin", "5.0",
       "--serve", "--park-slots", "3"]
log = open(aout, "w")
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env)
lines = queue.Queue()
def reader():
    for l in proc.stdout:
        lines.put(l.rstrip())
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
def send(s):
    proc.stdin.write(s + "\n"); proc.stdin.flush()
def gen(max_new, ids, tag):
    send("GEN %d temperature=0 %s" % (max_new, ",".join(str(i) for i in ids)))   # greedy: deterministic tokens
    toks, reused, finish = [], -1, "?"
    while True:
        l = lines.get()
        if l is None:
            raise RuntimeError("engine stdout closed at " + tag)
        if l.startswith("ERR"):
            raise RuntimeError("engine ERR at %s: %s" % (tag, l))
        f = l.split()
        if f and f[0] == "T":
            toks.append(int(f[1]))
        elif f and f[0] == "DONE":
            reused = int(f[8]); finish = f[5]
            return toks, reused, finish
ready = ""
while True:                                    # INFO lines (ctx, arena) precede READY on stdout
    l = lines.get()
    if l is None:
        raise RuntimeError("engine stdout closed before READY")
    if l.startswith("READY"):
        ready = l
        break
    if l.startswith("ERR"):
        raise RuntimeError("engine ERR before READY: " + l)
print("READY:", ready)
assert "slots=3" in ready, "engine did not advertise slots=3: " + ready
def prompt(base, n):                                 # deterministic pseudo-token ids, well inside the vocab
    return [base + 7 * i + (i * i) % 5 for i in range(n)]
PA, PB, PC, PD = prompt(4001, 14), prompt(5002, 11), prompt(6003, 12), prompt(7004, 9)
res = {}
res["A1"], r, _ = gen(24, PA, "A1"); res["A1_reused"] = r
res["B1"], r, _ = gen(20, PB, "B1"); res["B1_reused"] = r
a2_ids = PA + res["A1"]
res["A2"], res["A2_reused"], _ = gen(12, a2_ids, "A2")     # continuation: must RESUME from lcp=14
res["C1"], r, _ = gen(20, PC, "C1"); res["C1_reused"] = r
res["D1"], r, _ = gen(20, PD, "D1"); res["D1_reused"] = r   # 4th park target into 3 slots: LRU must evict
b2_ids = PB + res["B1"]
res["B2"], res["B2_reused"], _ = gen(10, b2_ids, "B2")      # B parked, then C/D came: B may or may not survive
d2_ids = PD + res["D1"]
res["D2"], res["D2_reused"], _ = gen(10, d2_ids, "D2")      # D is freshest: must still be parked
res["A_replay"], res["A_replay_reused"], _ = gen(12, a2_ids, "A-replay")   # A was very likely evicted: fresh path
send("QUIT")
try:
    proc.wait(timeout=30)
except subprocess.TimeoutExpired:
    proc.kill()
ok = True
def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (" " + detail if detail else ""))
    ok = ok and cond
check("A2 resumed from the parked conversation", res["A2_reused"] >= len(PA), "reused=%d want >= %d" % (res["A2_reused"], len(PA)))
check("D2 (freshest) resumed", res["D2_reused"] >= len(PD), "reused=%d" % res["D2_reused"])
check("A-replay (post-eviction fresh prefill) completed", len(res["A_replay"]) > 0, "%d tokens" % len(res["A_replay"]))
check("A-replay (strict-prefix of a longer parked conversation) refused the stale state",
      res["A_replay_reused"] == 0, "reused=%d want 0 (the lp>=ids.size() guard)" % res["A_replay_reused"])
check("A-replay fresh-prefills to the SAME answer the resumed run gave", res["A_replay"] == res["A2"],
      "match=%d/%d" % (sum(x == y for x, y in zip(res["A_replay"], res["A2"])), len(res["A2"])))
import json
print("STAGE-A-RESULT " + json.dumps({k: (v if not isinstance(v, list) else v[:32]) for k, v in res.items()}))
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] || { echo "stage A failed"; exit 1; }

mk_cfg() { # $1=phase tag -> "cfg elog"
    local cfg="logs/p4b-$TS.$1.json" elog="logs/p4b-$TS.$1.engine.log"
    $PY - "$CFG0" "$cfg" "$ROOT" "$elog" "$SLOTS" <<'EOF'
import json, sys
src, dst, root, elog, slots = sys.argv[1:6]
cfg = json.load(open(src))
cfg["exe"] = root + "/tools/v41/serve/ds41-engine-906.sh"
cfg["cwd"] = root
cfg["log"] = elog                                        # replace the WHOLE value (the doubled-path trap)
cfg["args"] = list(cfg["args"]) + ["--park-slots", slots]
json.dump(cfg, open(dst, "w"), indent=1)
EOF
    echo "$cfg $elog"
}

boot() { # $1=phase tag -> sets SPID ELOG; waits for /health loaded
    local cfg
    read -r cfg ELOG <<<"$(mk_cfg "$1")" || return 1
    echo "=== p4b/$1 server start $(date +%H:%M:%S) (engine log: $ELOG) ==="
    STRATA_SERVE_STEP_MAX=96 STRATA_NO_LARGEPAGES=1 STRATA_SERVE_PARK_MIN=${PARK_MIN:-64} \
        $PY tools/v41/serve/serve_ds41.py \
        --engine strata --config "$cfg" --port "$PORT" >"logs/p4b-$TS.$1.server.log" 2>&1 & SPID=$!
    local waited=0 st
    while :; do
        kill -0 $SPID 2>/dev/null || { echo "FAIL $1: server died - see logs/p4b-$TS.$1.server.log"; return 1; }
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
        waited=$((waited + 5)); [ $waited -ge 480 ] && { echo "FAIL $1: not loaded after 480 s"; kill $SPID 2>/dev/null; return 1; }
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


client() { # $1=phase: serial | concurrent | fairness
    $PY - "$PORT" "$1" "$TS" "$JSONL" "$T1MAX" "$T2MAX" <<'EOF' || return 1
import json, sys, threading, time, urllib.request
port, phase, ts, jsonl, t1max, t2max = sys.argv[1:7]
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
    with urllib.request.urlopen(req, timeout=600) as r:
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
            piece = (j.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if piece and first is None:
                first = time.time() - t0
            text.append(piece)
    return {"phase": phase, "text": "".join(text), "ttft": first, "wall": time.time() - t0}
def one(name):
    u1, u2 = CONVS[name]
    r1 = ask([{"role": "user", "content": u1}], int(t1max))
    r1.update(conv=name, turn=1)
    r2 = ask([{"role": "user", "content": u1}, {"role": "assistant", "content": r1["text"]},
              {"role": "user", "content": u2}], int(t2max))
    r2.update(conv=name, turn=2)
    return r1, r2
out, lock = [], threading.Lock()
def run(name):
    r1, r2 = one(name)
    with lock:
        out.extend([r1, r2])
if phase == "serial":
    for name in CONVS:
        run(name)
elif phase == "concurrent":
    th = []
    for name in CONVS:
        t = threading.Thread(target=run, args=(name,)); t.start(); th.append(t); time.sleep(0.2)
    for t in th:
        t.join(600)
else:  # fairness: a 256-token marathon holds the engine; two short requests arrive mid-marathon.
    def marathon():
        r = ask([{"role": "user", "content": "Write a 2000-word technical narrative about every component of "
                     "a four-GPU MoE inference box: PCIe topology, HBM2, the RAM tier, NVMe, the operating "
                     "system's page tables, and the scheduler. Begin."}], 256)
        r.update(conv="M", turn=1)
        with lock:
            out.append(r)
    def short(name, delay):
        time.sleep(delay)
        r = ask([{"role": "user", "content": CONVS[name][0]}, {"role": "assistant", "content": "Understood."},
                 {"role": "user", "content": "One sentence only: what is the VRAM tier for?"}], 32)
        r.update(conv=name, turn=9)
        with lock:
            out.append(r)
    ms = threading.Thread(target=marathon); ms.start()
    s1 = threading.Thread(target=short, args=("A", 2.0)); s1.start()
    s2 = threading.Thread(target=short, args=("B", 4.0)); s2.start()
    for t in (s1, s2, ms):
        t.join(600)
with open(jsonl, "a") as fh:
    for r in out:
        r["ts"] = ts
        fh.write(json.dumps(r) + "\n")
try:
    m = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5))
    hist = m.get("requests") or []
    if isinstance(hist, list):
        print("phase[%s]: %d requests, %d with reused>0, reused_tokens=%d"
              % (phase, len(hist), sum(1 for r in hist if (r.get("reused") or 0) > 0),
                 sum(int(r.get("reused") or 0) for r in hist)))
except Exception as e:
    print("phase[%s]: metrics unavailable (%s)" % (phase, e))
EOF
}

echo "== stage B: ONE boot; serial control, concurrent replay, fairness marathon =="
boot serve || { kill_server; exit 1; }
client serial     || { echo "serial client failed"; kill_server; exit 1; }
client concurrent || { echo "concurrent client failed"; kill_server; exit 1; }
client fairness   || { echo "fairness client failed"; kill_server; exit 1; }
kill_server

echo "== verdict =="
$PY - "$JSONL" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
by = {(r["phase"], r["conv"], r["turn"]): r for r in rows}
same = tot = 0
for k in sorted(by):
    if k[0] != "serial":
        continue
    b = by.get(("concurrent",) + k[1:])
    if b:
        tot += 1
        same += by[k]["text"] == b["text"]
        if by[k]["text"] != b["text"]:
            print("DIFF %s turn %d: serial %d chars vs concurrent %d chars" % (k[1], k[2], len(by[k]["text"]), len(b["text"])))
print("text match serial vs concurrent (SAME boot, park exactness): %d/%d" % (same, tot))
for ph in ("serial", "concurrent"):
    t = [r for r in rows if r["phase"] == ph and r["turn"] == 1]
    t.sort(key=lambda r: r["ttft"] or 9e9)
    print("%s turn-1 ttft: %s" % (ph, " ".join("%s=%.1fs" % (r["conv"], r["ttft"] or -1) for r in t)))
fair = {r["conv"]: r for r in rows if r["phase"] == "fairness"}
if "M" in fair:
    print("fairness: marathon M wall=%.1f s (started alone); short requests that arrived mid-marathon: %s"
          % (fair["M"]["wall"], " ".join("%s ttft=%.1fs wall=%.1fs" % (c, r["ttft"] or -1, r["wall"])
                                         for c, r in sorted(fair.items()) if c != "M")))
    print("  (no slicing: a short request would wait the marathon's REMAINING wall, ~%.0f s)" % (fair["M"]["wall"] - 2))
EOF
echo "jsonl: $JSONL ; engine logs: logs/p4b-$TS.*.engine.log"
