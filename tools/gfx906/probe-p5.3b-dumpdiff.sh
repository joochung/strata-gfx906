#!/usr/bin/env bash
# probe-p5.3b-dumpdiff.sh - P5.3b DISCRIMINATING PROBE: at the witness's exact geometry (14-token prompt, --relay-chunk
# 64, split 20), HOW FAR APART are a staged run's and an unsplit control's logits, row by row through the decode chain?
#
# WHY.  Witness #7 (TS 20261025-001554, tree bb4c6871): token 1 EQUALS both controls (the prefill + relay row is right),
# token 2 is wrong, and all three staged variants are byte-identical to EACH OTHER - a systematic difference, not noise.
# A tokens file cannot separate the three remaining hypotheses; a LOGIT MAGNITUDE can:
#   (A) DECODE-PASS SEAM - the prefill row agrees and the first DECODE row (position 14) differs by something big: the
#       one-row staged pass reads state the unsplit pass does not.  Fix belongs in that pass.
#   (B) KNIFE EDGE - every row agrees to ~1e-6 and only greedy argmax flips.  That is the witness's own calibrated bar
#       (GLM twins flip the same way), not a state bug; P5.3b then CLOSES by the same rule the GLM witness used.
#   (C) PREFILL/RELAY SEAM - the prefill row (which produced the CORRECT token 1!) differs materially: token 1 agreed by
#       luck (margin), and the fix belongs upstream.  Unlikely given 64/64 staged-variant self-agreement, but cheap to kill.
#
# HOW.  Two engine boots, each dumping EVERY row:
#   ctrl    the witness's unsplit control flags (run1) + STRATA_DUMP_STEPS=<dir> + --dump-last (the prefill row, P)
#   staged  the witness's run3 flags exactly (--gpus 0,0 --layer-split 20 --relay-chunk 64) + the same dumps
# The engine writes <dir>.step%03d.bin per decode step (float32 x n_vocab; step001 = the row that emits token 2, the
# witness's failing row) and --dump-last holds the prefill's last row (the row that emits token 1).  N_TOK defaults to 8:
# enough rows to watch the difference grow, ~3 min per boot.  Refuses if any engine is alive (one-HIP-engine rule).
#
# Orchestrator launches it via the GPU wrapper (stack stopped by it):
#   /ai/engines/gfx906/unverbraucht/run-with-free-gpus.sh \
#       "cd /ai/engines/gfx906/worktrees/p5.3-relay && bash tools/gfx906/probe-p5.3b-dumpdiff.sh"
set -u
cd "$(dirname "$0")/../.." || exit 1

TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
OUT=logs/p5.3b-dumpdiff-${TS}
PY=/root/.pyenv/versions/venv312/bin/python3
M=${M:-/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored}
SPLIT=${SPLIT:-20}
ARENA_GIB=${ARENA_GIB:-64}
N_TOK=${N_TOK:-8}

# ---- refuse if an engine is alive (SVM crash rule: one HIP engine at a time, vLLM included)
if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
    echo "refusing: another engine is alive - stop it first (one HIP engine at a time)"; exit 3
fi
MIN_AVAIL_GIB=$(( ARENA_GIB + 25 ))
AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }

# ---- the binary must be NEWER than every source it measures (witness #1's lesson)
GEN=build-906/ds41_generate
ENG=(bash tools/v41/serve/ds41-engine-906.sh)
[[ -x "$GEN" ]] || { echo "FATAL: missing $GEN - build with tools/gfx906/build-906-gate.sh first"; exit 3; }
[[ -x tools/v41/serve/ds41-engine-906.sh ]] || { echo "FATAL: missing tools/v41/serve/ds41-engine-906.sh"; exit 3; }
for src in tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_moe.cpp \
           tools/v41/engine/ds41_engram.cpp tools/gfx906/relay_staging.hpp tools/gfx906/device_split.hpp; do
    [ "$src" -nt "$GEN" ] && { echo "FATAL: $GEN is older than $src - the probe would measure a stale binary"; exit 3; }
done
# the step-dump instrument must be IN the binary (this probe measures nothing without it)
strings "$GEN" | grep -q "STRATA_DUMP_STEPS" || { echo "FATAL: $GEN predates the STRATA_DUMP_STEPS instrument"; exit 3; }
strings "$GEN" | grep -q "STRATA_DUMP_STATE" || { echo "FATAL: $GEN predates the STRATA_DUMP_STATE instrument"; exit 3; }
# host gates green before the GPU is touched (seconds)
bash tools/gfx906/gate-relay-pos-tables.sh >/dev/null 2>&1 || { echo "FATAL: gate-relay-pos-tables FAILs"; exit 3; }
bash tools/gfx906/gate-device-split.sh    >/dev/null 2>&1 || { echo "FATAL: gate-device-split FAILs"; exit 3; }

# ---- the SAME prompt the witness tokenizes (run-p5.3b-ds41-handoff.sh's own path, verbatim)
IDS=$OUT.i32
$PY - "$IDS" <<EOF || exit 1
import sys, json
sys.path.insert(0, "tools")
import strata_tokenizer as ST, numpy as np
tdir = "$M/tokenizer"
vocab = json.load(open(tdir + "/vocab.json"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = open(tdir + "/merges.txt").read().split("\\n")
types = json.load(open(tdir + "/token_type.json"))
tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")   # the server's own pre-split (serve_ds41.py:45)
ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences.")
np.array(ids, dtype=np.int32).tofile(sys.argv[1])
print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
EOF
NPRO=$($PY -c "import numpy as np,sys; print(len(np.fromfile(sys.argv[1],dtype=np.int32)))" "$IDS")
echo "prompt: $NPRO tokens; first decode pass runs at position $NPRO (its logits emit TOKEN 2, the witness's failing row)"

# ---- the witness's engine args verbatim (run-p5.3b-ds41-handoff.sh's ARGS array)
ARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu
      --engram "$M/engram/l1:$M/engram/l14" --rank "$M/rank.bin"
      --ids-file "$IDS" -n "$N_TOK" --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0
      --prefill-chunk 64)
STAGED=(--gpus 0,0 --layer-split "$SPLIT" --relay-chunk 64)

run_one() {   # run_one <name> <extra args...>
    sleep 10                                # a finished engine frees VRAM asynchronously
    local name=$1; shift
    echo "=== $name start $(date +%T) ==="
    mkdir -p "$OUT.$name"
    STRATA_DUMP_STEPS="$OUT.$name/step" STRATA_DUMP_STATE="$OUT.$name/state" STRATA_DUMP_RELAY="$OUT.$name/relay" \
        "${ENG[@]}" "${ARGS[@]}" --dump-last "$OUT.$name.P.bin" "$@" \
        >"$OUT.$name.toks" 2>"$OUT.$name.log"
    local rc=$?
    echo "--- $name rc=$rc, $(wc -l < "$OUT.$name.toks") token ids, $(ls "$OUT.$name"/step.step*.bin 2>/dev/null | wc -l) step dumps"
    grep -m4 -E "relay: staged run|layer split:|boundary hand-offs|prompt:" "$OUT.$name.log" | sed "s/^/[$name] /"
    [ $rc -eq 0 ] || { echo "PROBE: $name FAILED (rc=$rc) - see $OUT.$name.log"; return 1; }
}

run_one ctrl || exit 1
run_one staged "${STAGED[@]}" || exit 1

# the staged run must actually have staged, with the flags that matter
grep -q "relay: staged run" "$OUT.staged.log" || { echo "PROBE: staged run never built a staged run"; exit 1; }
grep -q "enc-skip: cut at layer" "$OUT.staged.log" && { echo "PROBE: staged ran with enc-skip ON"; exit 1; }

# ---- the row-by-row diff
$PY - "$OUT" "$NPRO" <<'EOF' || exit 1
import sys, numpy as np, glob, os
out, npro = sys.argv[1], int(sys.argv[2])

tk_c = np.array([int(x) for x in open(out + ".ctrl.toks")], dtype=np.int64)
tk_s = np.array([int(x) for x in open(out + ".staged.toks")], dtype=np.int64)
n = min(len(tk_c), len(tk_s)); lcp = 0
while lcp < n and tk_c[lcp] == tk_s[lcp]:
    lcp += 1
print("== TOKENS ==")
print(f"  ctrl   : {tk_c[:12].tolist()}")
print(f"  staged : {tk_s[:12].tolist()}")
print(f"  longest common prefix {lcp}/{n} (the witness's shape is lcp 1)")

def rows(side, tag):
    # run_one passes STRATA_DUMP_STEPS="$OUT.$name/step", so the engine writes "$OUT.$name/step.stepNNN.bin" INSIDE
    # the per-side directory; --dump-last is a plain path given to us on the CLI.  (First probe run counted "0 step
    # dumps" because the count globbed "$OUT.$name.step*.bin" - a sibling of the directory, not inside it.)
    p = f"{out}.{side}/{tag}" if tag != "P.bin" else f"{out}.{side}.P.bin"
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None

def cmp(tag, c, s, note=""):
    print(f"\n== {tag} ==")
    if c is None or s is None:
        print("  MISSING dump on one side - inconclusive for this row.")
        return None
    nv = min(len(c), len(s))
    d = np.abs(c[:nv].astype(np.float64) - s[:nv].astype(np.float64))
    ac, asx = int(np.argmax(c[:nv])), int(np.argmax(s[:nv]))
    marg = np.sort(c[:nv])[::-1]
    print(f"  max abs diff {d.max():.6e}   mean {d.mean():.6e}")
    for thr in (1e-6, 1e-4, 1e-2, 1e-1):
        print(f"    entries > {thr:.0e} : {int((d > thr).sum())} / {nv}")
    print(f"  argmax {ac} vs {asx}: {'SAME' if ac == asx else 'DIFFER'}   (control top-2 margin {marg[0]-marg[1]:.4f})")
    if note:
        print(f"  {note}")
    return d.max(), (ac == asx), (marg[0] - marg[1])

P = cmp("P row: prefill last row (position %d) - emits TOKEN 1" % (npro - 1),
        rows("ctrl", "P.bin"), rows("staged", "P.bin"),
        "token 1 AGREED in the witness; P big anyway means it agreed by margin luck (hypothesis C)")
stepres = []
for idx in range(1, 6):
    f = ".step%03d.bin" % idx
    c, s = rows("ctrl", f), rows("staged", f)
    if c is None and s is None:
        break
    which = " - THE WITNESS'S FAILING ROW (emits token 2)" if idx == 1 else ""
    stepres.append((idx, cmp(f"decode step {idx}: row that emits token {idx+1}{which}", c, s)))

print("\n== VERDICT ==")
if P is None or not stepres or stepres[0][1] is None:
    print("  inconclusive: missing dumps; check the .log files and ls " + out + ".*")
else:
    pmax, psame, pmarg = P
    d1, dsame, dmarg = stepres[0][1]
    grow = [r[1][0] for r in stepres if r[1]]
    print(f"  P {pmax:.2e} (argmax {'same' if psame else 'DIFFER'}, top-2 margin {pmarg:.4f}); "
          f"step1 {d1:.2e} (argmax {'same' if dsame else 'DIFFER'}, margin {dmarg:.4f}); growth per row: "
          + " -> ".join(f"{g:.1e}" for g in grow))
    if pmax > 1e-3 and not psame:
        print("  (C) PREFILL/RELAY seam - the P row itself differs materially.")
    elif d1 > 1e-2 and not dsame:
        print("  (A) DECODE-PASS SEAM - the first one-row staged pass reads state the unsplit pass does not.")
        print("      Next: dump that pass's relay payload on both sides and diff it.")
    elif d1 < 1e-4 and dsame and lcp >= 2:
        print("  (B') the row that failed in the witness AGREES here and the tokens now agree - geometry drifted;")
        print("       rerun with N_TOK=64 to reproduce the witness row-for-row before concluding anything.")
    elif d1 < 1e-4 and not dsame:
        print("  (B) KNIFE EDGE - step 1 agrees to float-order noise and only argmax flips (control margin above).")
        print("      The witness's calibrated bar covers this: it is the same flip its own twins showed.")
    else:
        print("  MIXED - read the per-threshold counts above; a whole-row shift is the payload, a few entries are float order.")
EOF
rc=$?

# ---- the state-column comparison (STRATA_DUMP_STATE) ------------------------------
# The relay carries EXACTLY x_state + pre_state (+ cand when the geometry crosses it), and the P row proved the
# PREFILL output bitwise right.  So the decisive comparison is the state AFTER the passes: the unsplit instance's
# columns and the staged consumer's (s1) columns describe the same quantity (all 40 layers' output for those
# positions); if they are equal at "pre" but the step001 row is wrong, the bug is in what the decode pass EXECUTES
# given identical state, not in what it was handed.  s0 is the producer's mid-depth state - no unsplit reference
# exists for it (unsplit overwrites layers 0-19's output within the same pass), so it is reported for scale and for
# NaN/zero triage of stage 0's decode-pass output.  Payload files (STRATA_DUMP_RELAY) are the same bytes s0 shows
# after a pass, at the boundary itself.
$PY - "$OUT" <<'EOF'
import os, sys, glob
import numpy as np
out = sys.argv[1]
XS, PS = 81920, 16   # D*HC*4 and HC*4 bytes per token column (V41-GSQ: 5120 * 4 * 4)
def stfile(side, tag, who):
    p = f"{out}.{side}/state.st.{tag}.{who}.bin"
    return np.fromfile(p, dtype=np.uint8) if os.path.exists(p) else None
def report(tag, a, b, la, lb):
    if a is None or b is None:
        print(f"  {tag}: missing dump ({la if a is None else ''}{lb if b is None else ''})")
        return None
    n = min(len(a), len(b))
    neq = int((a[:n] != b[:n]).sum())
    af, bf = a[:n].view(np.float32), b[:n].view(np.float32)
    m = np.abs(af.astype(np.float64) - bf.astype(np.float64))
    first = int(np.argmax(a[:n] != b[:n])) if neq else -1
    col = first // XS if first >= 0 else -1
    print(f"  {tag} vs {la}: {n} B, {neq} differing bytes ({100.0*neq/n:.3f}%), max float diff {m.max():.3e}"
          + (f", first at byte {first} = column {col}" if first >= 0 else ""))
    return neq
print("== STATE COLUMNS (x_state then pre_state, columns 0..ncols-1 of the last pass's window) ==")
# The layer-19 boundary comparison: the unsplit control's mid-pass state after its first 20 layers (a depth it
# never otherwise exposes) vs stage 0's output at the same depth of the same DECODE pass.  EQUAL -> stage 0 is
# innocent and the consumer's execution of layers 20-39 is what differs; DIFFER -> stage 0's decode pass is.
import os as _os
for p in sorted(_os.listdir(f"{out}.ctrl")) if _os.path.isdir(f"{out}.ctrl") else []:
    if not p.startswith("state.st.midL19.p"):
        continue
    tagw = p[len("state.st.midL19."):][:4]   # e.g. "p14."
    a = np.fromfile(f"{out}.ctrl/" + p, dtype=np.uint8)
    q = f"{out}.staged/state.st.midL19.{tagw[:-1]}.s0.bin"
    if not _os.path.exists(q):
        continue
    b = np.fromfile(q, dtype=np.uint8)
    n = min(len(a), len(b))
    ncols = n // 81936
    ax, bx = a[:ncols * 81920].view(np.float32), b[:ncols * 81920].view(np.float32)
    d = np.abs(ax.astype(np.float64) - bx.astype(np.float64))
    print(f"  midL19 {tagw[:-1]}: ctrl-after-layer-19 vs stage0-output: {int((d > 0).sum())} floats differ, "
          f"max {d.max():.3e}  -> " + ("STAGE 0 INNOCENT (bitwise)" if (d == 0).all() else "STAGE 0 DIFFERS"))
# the post-relay dumps first: the consumer's columns BETWEEN relay_write_in and its own pass.  If these carry the
# payload's real pre-mix floats, the relay lands and the consumer's EXECUTION is what leaves zeros; if these are
# already zero, relay_write_in's pre_state copy is the broken edge.
for p in sorted(glob.glob(f"{out}.staged/state.st.bnd*.postrelay.bin")):
    a = np.fromfile(p, dtype=np.float32)
    ncols = len(a) // 81936
    pre = a[ncols * 81920 // 4:]
    print(f"  {os.path.basename(p)}: {ncols} cols, pre_state nonzero {int((pre != 0).sum())}/{len(pre)}, "
          f"col0 {pre[:4].tolist()}")
for tag in ("pre", "step001"):
    full = stfile("ctrl", tag, "full")
    s1 = stfile("staged", tag, "s1")
    s0 = stfile("staged", tag, "s0")
    r1 = report(tag, full, s1, "ctrl-full", "staged-s1")
    report(tag, full, s0, "ctrl-full", "staged-s0 (mid-depth, scale only)")
    if r1 is not None and tag == "pre":
        print(f"  -> prefill state columns {'IDENTICAL' if r1 == 0 else 'DIVERGE'}: "
              + ("the relayed set is right; the decode pass itself is what differs" if r1 == 0
                 else "the consumer's state after PREFILL already differs from the control's"))
print("== RELAY PAYLOADS (staged; boundary = stage 0's decode-pass output) ==")
for p in sorted(glob.glob(f"{out}.staged/relay.bnd*.bin")):
    a = np.fromfile(p, dtype=np.float32)
    xs_n = min(len(a), XS // 4 * (1 if ".n1." in p else 14))
    x = a[: len(a) - (len(a) % 1)]
    print(f"  {os.path.basename(p)}: {len(a)*4} B, NaN {int(np.isnan(x).sum())}, "
          f"zeros {int((x == 0).sum())}/{len(x)}, absmax {np.abs(x[np.isfinite(x)]).max() if np.isfinite(x).any() else float('nan'):.3f}")
EOF
rc2=$?
[ $rc2 -eq 0 ] || rc=$rc2
echo
echo "probe-p5.3b-dumpdiff: artifacts in $OUT.* (logs, token ids, row dumps)"
exit $rc
