#!/usr/bin/env bash
# run-p4b-glm-sessions.sh - the P4b.4 witness for --park-slots on glm_generate (the GLM mirror of
# run-p4b-ds41-sessions.sh; docs/gfx906/P4B-SESSIONS.md sections 5 + 6).
#
# Stage A (engine protocol, deterministic): glm_generate --serve --park-slots 3 over stdin. A resumed
# continuation must reuse the parked prefix (reused > 0); a resubmitted STRICT PREFIX of a longer parked
# conversation must be REFUSED the stale state (the 5b4643df guard, ported verbatim in ec174b17) and
# fresh-prefill to the SAME answer the resumed run gave; slot pressure (4 conversations, 3 slots) must
# answer every turn through the LRU eviction.
#
# Stage B (ONE boot of the real stack, serve_glm.py --park-slots 4): six conversations x two turns run
# serially, then concurrently, then a fairness phase (one long marathon + two short arrivals). Gates:
# stage-A exactness + fairness (short TTFT < 60% of the marathon's wall) + zero request errors. Text
# equality serial-vs-concurrent is INFORMATIONAL only - the standing residency knife-edge rule (section 5).
#
# Refusals: MemAvailable < 65 GiB; a live GLM engine/serve wrapper; a binary older than its sources.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
PY=/root/.pyenv/versions/venv312/bin/python3
CFG0=tools/glm/serve/strata-glm-mayal-906-ctl.json
PORT=${PORT:-8141}
JSONL=logs/p4b-glm-sessions-$TS.jsonl
R=benchmarks/gfx906/receipts/$(date +%Y-%m-%d)-p4b4-glm-sessions.json
SLOTS=${SLOTS:-4}
T1MAX=${T1MAX:-48}          # GLM decode is ~2-3x slower than DS4.1: shorter turns, same shapes
T2MAX=${T2MAX:-24}
MARATHON=${MARATHON:-128}
PARK_MIN=${PARK_MIN:-64}

free_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
[ "$(free_gib)" -ge 65 ] || { echo "refusing: MemAvailable $(free_gib) GiB < 65 (the batch-script rule)"; exit 3; }
pgrep -f "glm_generate --serve" >/dev/null && { echo "refusing: glm_generate --serve alive"; exit 3; }
pgrep -f "serve_glm.py" >/dev/null && { echo "refusing: a serve_glm.py is alive"; exit 3; }
for src in tools/glm/glm_generate.cpp tools/glm/glm_dense.cpp; do
    [ build-906/glm_generate -nt "$src" ] || { echo "refusing: glm_generate older than $src"; exit 3; }
done
$PY -m py_compile serve/server.py || { echo "refusing: serve/server.py does not compile"; exit 3; }
[ -x build-906/glm_generate ] || { echo "refusing: no build-906/glm_generate"; exit 3; }

ENGSH=$($PY -c "import json;print(json.load(open('$CFG0'))['exe'])")
ARGS=$($PY -c "import json;print(json.dumps(json.load(open('$CFG0'))['args']))")

echo "== stage A: engine protocol over stdin (--park-slots 3) =="
AOUT=logs/p4b-glm-$TS.stageA.log
$PY - "$ENGSH" "$AOUT" "$ARGS" <<'EOF' || { echo "stage A FAILED (see $AOUT)"; exit 1; }
import json, os, subprocess, sys, threading, queue
engsh, aout, args_json = sys.argv[1:4]
engine_args = json.loads(args_json)
env = {**os.environ, "STRATA_SERVE_PARK_MIN": "16"}        # the supported short-conversation knob
cmd = [engsh, *engine_args, "--serve", "--park-slots", "3"]
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
    send("GEN %d temperature=0 %s" % (max_new, ",".join(str(i) for i in ids)))    # greedy: deterministic
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
            return toks, int(f[8]), f[5]
while True:                                    # INFO lines may precede READY on stdout
    l = lines.get()
    if l is None:
        raise RuntimeError("engine stdout closed before READY")
    if l.startswith("READY"):
        print("READY:", l)
        assert "slots=3" in l, "engine did not advertise slots=3: " + l
        break
    if l.startswith("ERR"):
        raise RuntimeError("engine ERR before READY: " + l)
def prompt(base, n):                                # deterministic pseudo-token ids, well inside GLM's vocab
    return [base + 7 * i + (i * i) % 5 for i in range(n)]
PA, PB, PC, PD = prompt(4001, 14), prompt(5002, 11), prompt(6003, 12), prompt(7004, 9)
res = {}
res["A1"], r, _ = gen(24, PA, "A1"); res["A1_reused"] = r
res["B1"], r, _ = gen(20, PB, "B1"); res["B1_reused"] = r
a2_ids = PA + res["A1"]
res["A2"], res["A2_reused"], _ = gen(12, a2_ids, "A2")     # continuation: must resume from the parked state
res["C1"], r, _ = gen(20, PC, "C1"); res["C1_reused"] = r
res["D1"], r, _ = gen(20, PD, "D1"); res["D1_reused"] = r   # 4th park target into 3 slots: LRU must evict
d2_ids = PD + res["D1"]
res["D2"], res["D2_reused"], _ = gen(10, d2_ids, "D2")      # D is freshest: must still be parked
res["A_replay"], res["A_replay_reused"], _ = gen(12, a2_ids, "A-replay")   # a strict prefix of the longer parked A
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
check("A-replay (strict-prefix of a longer parked conversation) completed", len(res["A_replay"]) > 0, "%d tokens" % len(res["A_replay"]))
check("A-replay refused the stale state", res["A_replay_reused"] == 0,
      "reused=%d want 0 (the lp>=ids.size() guard)" % res["A_replay_reused"])
check("A-replay fresh-prefills to the SAME answer the resumed run gave", res["A_replay"] == res["A2"],
      "match=%d/%d" % (sum(x == y for x, y in zip(res["A_replay"], res["A2"])), len(res["A2"])))
print("STAGE-A-RESULT " + json.dumps({k: (v if not isinstance(v, list) else v[:32]) for k, v in res.items()}))
with open(aout, "a") as fh:
    fh.write("STAGE-A-RESULT " + json.dumps({k: (v if not isinstance(v, list) else v[:32]) for k, v in res.items()}) + "\n")
if not ok:
    sys.exit(1)
EOF

echo "== stage B: ONE boot; serial control, concurrent replay, fairness marathon =="
read -r CFG ELOG <<<"$($PY - "$CFG0" "$ROOT" "$TS" "$SLOTS" <<'EOF'
import json, sys
src, root, ts, slots = sys.argv[1:5]
cfg = json.load(open(src))
cfg["exe"] = root + "/tools/glm/serve/glm-engine-906.sh"
cfg["cwd"] = root
cfg["log"] = "logs/p4b-glm-%s.serve.engine.log" % ts        # replace the WHOLE value (doubled-path trap)
cfg["args"] = list(cfg["args"]) + ["--park-slots", slots]
dst = "logs/p4b-glm-%s.serve.json" % ts
json.dump(cfg, open(dst, "w"), indent=1)
print(dst, cfg["log"])
EOF
)"
echo "=== p4b-glm/serve server start $(date +%H:%M:%S) (engine log: $ELOG) ==="
STRATA_SERVE_PARK_MIN=$PARK_MIN STRATA_NO_LARGEPAGES=1 \
    $PY tools/glm/serve/serve_glm.py --engine strata --config "$CFG" --port "$PORT" >"logs/p4b-glm-$TS.serve.server.log" 2>&1 & SPID=$!
waited=0
while :; do
    kill -0 $SPID 2>/dev/null || { echo "FAIL: server died - see logs/p4b-glm-$TS.serve.server.log"; exit 1; }
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
    waited=$((waited + 5)); [ $waited -ge 600 ] && { echo "FAIL: not loaded after 600 s"; kill $SPID 2>/dev/null; exit 1; }
    sleep 5
done
echo "[serve] loaded after ${waited} s"

phase() { # $1 = serial | concurrent | fairness
    $PY - "$PORT" "$1" "$TS" "$JSONL" "$T1MAX" "$T2MAX" "$MARATHON" <<'EOF'
import json, sys, threading, time, urllib.request
port, phase, ts, jsonl, t1max, t2max, marathon = sys.argv[1:8]
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
    body = {"messages": msgs, "max_tokens": maxtok, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = []; n_content = 0
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
                if j.get("error"):                       # a server error JSON: record it, never count it as a stream
                    return {"phase": phase, "text": "", "ttft": first, "wall": time.time() - t0,
                            "error": str(j["error"])[:200]}
                ch = (j.get("choices") or [{}])[0]
                dl = ch.get("delta") or {}
                # TTFT counts the FIRST STREAMED TOKEN, reasoning or content: GLM always thinks first (its config's
                # reasoning_budget_tokens, and reasoning_effort=none does not switch the template off the way
                # DS4.1's does). What the slice seam buys is the engine - the first token either channel proves it.
                piece = dl.get("content") or dl.get("reasoning_content")
                if dl.get("content"):
                    n_content += 1
                if piece:
                    if first is None:
                        first = time.time() - t0
                    text.append(piece)
        return {"phase": phase, "text": "".join(text), "ttft": first, "content_toks": n_content,
                "wall": time.time() - t0, "error": None}
    except Exception as e:
        return {"phase": phase, "text": "", "ttft": None, "wall": time.time() - t0, "error": str(e)}
out = []
lock = threading.Lock()
def turn(conv, user, maxtok):
    r = ask([{"role": "user", "content": user}], int(maxtok))
    r.update(conv=conv, turn=1)
    with lock:
        out.append(r)
    r2 = ask([{"role": "user", "content": user}, {"role": "assistant", "content": r["text"]},
              {"role": "user", "content": CONVS[conv][1]}], int(t2max))
    r2.update(conv=conv, turn=2)
    with lock:
        out.append(r2)
if phase == "fairness":
    def marathon_run():
        r = ask([{"role": "user", "content":
                  "Write a very detailed, numbered technical manual (leave nothing out) about serving a "
                  "mixture-of-experts model on four GPUs with a RAM and NVMe expert cache: boot order, the "
                  "arena, admission, rebalancing, decode, failure handling, and a worked example. Continue "
                  "with as many numbered sections as you can."}], int(marathon))
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
    ms = threading.Thread(target=marathon_run); ms.start()
    s1 = threading.Thread(target=short, args=("A", 2.0)); s1.start()
    s2 = threading.Thread(target=short, args=("B", 4.0)); s2.start()
    for t in (s1, s2, ms):
        t.join(900)
else:
    ths = {
        "serial": [threading.Thread(target=turn, args=(c, CONVS[c][0], t1max)) for c in sorted(CONVS)],
        "concurrent": [threading.Thread(target=turn, args=(c, CONVS[c][0], t1max)) for c in sorted(CONVS)],
    }[phase]
    for t in ths:
        t.start()
        if phase == "serial":
            t.join(900)
    if phase == "concurrent":
        for t in ths:
            t.join(900)
# usage counters: /metrics keeps per-request reused tokens when the server supports it
try:
    m = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5))
    hist = m.get("requests") or []
    if isinstance(hist, list):
        print("phase[%s]: %d requests, %d with reused>0, reused_tokens=%d"
              % (phase, len(hist), sum(1 for r in hist if (r.get("reused") or 0) > 0),
                 sum(int(r.get("reused") or 0) for r in hist)))
except Exception as e:
    print("phase[%s]: metrics unavailable (%s)" % (phase, e))
with open(jsonl, "a") as fh:
    for r in out:
        r["ts"] = ts
        fh.write(json.dumps(r) + "\n")
EOF
}
phase serial
phase concurrent
phase fairness
kill $SPID 2>/dev/null
for _ in $(seq 30); do pgrep -f "glm_generate --serve" >/dev/null || break; sleep 2; done
pkill -f "glm_generate --serve" 2>/dev/null
sleep 10                                        # the async VRAM-free rule after a killed engine

echo "== verdict =="
$PY - "$JSONL" "$AOUT" "$R" "$TS" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
stageA = {}
for line in open(sys.argv[2]):
    if line.startswith("STAGE-A-RESULT"):
        stageA = json.loads(line.split("STAGE-A-RESULT ", 1)[1])
by = {(r["phase"], r["conv"], r["turn"]): r for r in rows}
same = tot = 0
for k in sorted(by):
    if k[0] != "serial":
        continue
    b = by.get(("concurrent",) + k[1:])
    if b:
        tot += 1
        same += by[k]["text"] == b["text"]
fair = {r["conv"]: r for r in rows if r["phase"] == "fairness"}
M = fair.get("M", {})
fairness = {"short_ttft_s": {c: r.get("ttft") for c, r in sorted(fair.items()) if c != "M"},
            "marathon_wall_s": M.get("wall"),
            "short_beat_marathons_wall": bool(M) and all(
                (fair[c].get("ttft") or 9e9) < 0.6 * M["wall"] for c in fair if c != "M")}
errors = [r for r in rows if r.get("error")]
stage_a_pass = (bool(stageA.get("A2_reused")) and bool(stageA.get("D2_reused"))
                and "A_replay" in stageA and stageA.get("A_replay_reused") == 0
                and stageA.get("A_replay") == stageA.get("A2"))
json.dump({"gate": "P4b.4 witness: GLM serve x park slots x sliced turns", "ts": sys.argv[4],
           "commit": __import__("subprocess").run(["git", "rev-parse", "--short", "HEAD"],
                                                  capture_output=True, text=True).stdout.strip(),
           "stageA": {k: v for k, v in stageA.items() if "reused" in k or k.endswith("replay")},
           "fairness": fairness,
           "text_match_serial_vs_concurrent": "%d/%d (informational - residency knife-edge)" % (same, tot),
           "requests": len(rows), "errors": len(errors),
           "verdict": "GO: park/unpark exact (stage A), slicing fair, all requests completed"
           if stage_a_pass and fairness["short_beat_marathons_wall"] and not errors else "NO-GO or INCOMPLETE"},
          open(sys.argv[3], "w"), indent=1)
print(json.dumps({"stage_a_pass": stage_a_pass, "fairness": fairness, "requests": len(rows),
                  "errors": len(errors)}, indent=1))
EOF
echo "jsonl: $JSONL ; receipt: $R ; engine logs: logs/p4b-glm-$TS.*.engine.log"
