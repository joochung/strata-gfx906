#!/usr/bin/env bash
# probe-p5.3b-dumpdiff.sh - P5.3b DISCRIMINATING PROBE: where do a staged run and an unsplit control stop agreeing,
# INSIDE the network, at the geometry witness #7 failed on (prompt 14 tokens, first decode at position 14)?
#
# WHY THIS PROBE AND NOT ANOTHER THEORY.  Witness #7 (logs ts 20261025-001554) has token 1 correct and token 2 wrong, with
# every staged variant byte-identical to every other and the pos-tables fix IN the tree.  That rules out anything
# run-to-run and anything variant-specific, and leaves exactly one open question: at the first decode pass, is the staged
# run's divergence
#   (A) AT THE HEAD - the logits at position 14 differ from the control's, and the argmax flips because some layer's output
#       already differs, or
#   (B) UPSTREAM OF THE HEAD AND BOUNDED - the logits agree to ~1e-7 and only the greedy argmax flips on a knife edge
#       (float-order noise, the calibrated bar of the witness itself), or
#   (C) IN THE PREFILL, invisibly - the state arriving at position 14 already differs (the x_state / pre_state payload or
#       the ring rows it reads), in which case the logits differ and the fix belongs in the seam that produced them.
# A token comparison cannot separate these: (A) and (C) both look like "token 2 wrong".  A LOGIT DIFFERENCE does, because its
# MAGNITUDE says how far upstream the disagreement started (1e-6 = summation order; 1e-1 = a different state) and its
# SHAPE says whether one row or every row moved.
#
# WHAT IT RUNS (two runs, nothing else):
#   ctrl      the witness's unsplit control (run1/run2 flags exactly, --prompt-batch at the engine default)
#   staged    the witness's run3 flags exactly: --gpus 0,0 --layer-split $SPLIT --relay-chunk 64
# both with --dump-last TWICE IN THE SAME PROCESS, which is the point of the probe: ds41_generate writes that file when the
# PREFILL finishes (ds41_generate.cpp:2575) and AGAIN after the decode loop (it re-opens the file, truncating it), so each
# run yields two dumps from one engine boot -
#     P = the prompt's last row  (position n_prompt - 1 = 13)   -> the PREFILL + relay verdict
#     D = the LAST generated row (position n_prompt + n_tok - 1) -> the DECODE-chain verdict
# and the two runs' P files and D files are compared separately by the numpy step.  If P agrees and D diverges, the prefill
# and the relay are innocent and the seam is the staged one-row decode pass; if P already diverges, the head is only the
# messenger and the fix belongs upstream of it.
#
# OPERATOR launches (this box's rule: the assistant never starts an engine run).  Refuses if any engine is alive.
#   cd /ai/engines/gfx906/worktrees/p5.3-relay
#   bash tools/gfx906/probe-p5.3b-dumpdiff.sh 2>&1 | tee probe-p5.3b-dumpdiff.log
# set -u
# cd "$(dirname "$0")/../.." || exit 1

# TS=$(date +%Y%m%d-%H%M%S)
# mkdir -p logs
# OUT=logs/p5.3b-dumpdiff-${TS}
# PY=/root/.pyenv/versions/venv312/bin/python3
# M=${M:-/ai/models/hf.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored}
# SPLIT=${SPLIT:-20}
# ARENA_GIB=${ARENA_GIB:-64}
# N_TOK=${N_TOK:-8}                     # enough decodes to see token 2 (the witness's failure) and its neighbours

# ---- refuse if an engine is alive (SVM crash rule: one HIP engine at a time, vLLM included)
# if pgrep -f "engine/strata|vllm|glm_generate|ds41_generate" >/dev/null 2>&1; then
#     echo "refusing: another engine is alive - stop it first (one HIP engine at a time)"; exit 3
# fi
# MIN_AVAIL_GIB=$(( ARENA_GIB + 25 ))
# AVAIL=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
# [ "$AVAIL" -ge "$MIN_AVAIL_GIB" ] || { echo "refusing: ${AVAIL} GiB available < ${MIN_AVAIL_GIB}"; exit 3; }

# ---- the artifacts the probe stands on: a binary NEWER than every source it measures (witness #1's lesson)
# GEN=build-906/ds41_generate
# ENG=(bash tools/v41/serve/ds41-engine-906.sh)
# [[ -x "$GEN" ]] || { echo "FATAL: missing $GEN - build with tools/gfx906/build-906-gate.sh first"; exit 3; }
# [[ -x tools/v41/serve/ds41-engine-906.sh ]] || { echo "FATAL: missing tools/v41/serve/ds41-engine-906.sh"; exit 3; }
# for src in tools/v41/engine/ds41_dense.cpp tools/v41/engine/ds41_generate.cpp tools/v41/engine/ds41_moe.cpp \
#            tools/v41/engine/ds41_engram.cpp tools/gfx906/relay_staging.hpp tools/gfx906/device_split.hpp; do
#     [ "$src" -nt "$GEN" ] && { echo "FATAL: $GEN is older than $src - the probe would measure a stale binary"; exit 3; }
# done
# the host gates must be green before the GPU is touched (they are seconds)
# bash tools/gfx906/gate-relay-pos-tables.sh >/dev/null 2>&1 || { echo "FATAL: gate-relay-pos-tables FAILs"; exit 3; }
# bash tools/gfx906/gate-device-split.sh    >/dev/null 2>&1 || { echo "FATAL: gate-device-split FAILs"; exit 3; }

# ---- the SAME prompt the witness uses (run-p5.3b-ds41-handoff.sh's own tokenizer path, verbatim)
# IDS=$OUT.i32
# $PY - "$IDS" "$M" <<'EOF' || exit 1
# import sys, json
# sys.path.insert(0, "tools")
# import strata_tokenizer as ST, numpy as np
# tdir = sys.argv[2] + "/tokenizer"
# vocab = json.load(open(tdir + "/vocab.json"))
# tokens = [None] * len(vocab)
# for t, i in vocab.items():
#     tokens[i] = t
# merges = open(tdir + "/merges.txt").read().split("\n")
# types = json.load(open(tdir + "/token_type.json"))
# tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")   # the server's own pre-split (serve_ds41.py:45)
# ids = tok.encode("What is a mixture-of-experts model? Answer in two sentences.")
# np.array(ids, dtype=np.int32).tofile(sys.argv[1])
# print(f"prompt ids: {len(ids)} tokens -> {sys.argv[1]}")
# EOF
# NPRO=$($PY -c "import numpy as np,sys; print(len(np.fromfile(sys.argv[1],dtype=np.int32)))" "$IDS")
# echo "prompt length: $NPRO tokens; the FIRST DECODE is a 1-row pass at position $NPRO"

# ---- the witness's engine args, built the same way (run-p5.3b-ds41-handoff.sh's ARGS array)
# ARGS=(-m "$M/V41-GSQ.gguf" --backend cuda --experts gpu
#       --engram "$M/engram/l1:$M/engram/l14" --rank "$M/rank.bin"
#       --ids-file "$IDS" -n "$N_TOK" --arena-gib "$ARENA_GIB" --arena-adapt --vram-margin 1.0
#       --prefill-chunk 64)
# STAGED=(--gpus 0,0 --layer-split "$SPLIT" --relay-chunk 64)

# run_one() {   # run_one <name> <extra args...>
#     sleep 10                                # a finished engine frees VRAM asynchronously
#     local name=$1; shift
#     echo "=== $name start $(date +%T) ==="
#     # --dump-last is written when the PREFILL finishes and AGAIN after the decode loop (ds41_generate re-opens the file,
#     # truncating).  Watch it and snapshot the FIRST write, so one boot yields BOTH the prompt's last row (P) and the last
#     # generated row (D).  A dump of one vocab's floats appears once per write, so the file GROWS then SHRINKS: snapshot the
#     # first stable size, then take the final file as D.
#     local watch=$OUT.$name.last.bin
#     ( while :; do
#         if [ -f "$watch" ]; then
#             sz1=$(stat -c %s "$watch" 2>/dev/null || echo 0); sleep 1; sz2=$(stat -c %s "$watch" 2>/dev/null || echo 0)
#             if [ "$sz1" = "$sz2" ] && [ "$sz1" -gt 0 ] && [ ! -f "$OUT.$name.P.bin" ]; then
#                 cp "$watch" "$OUT.$name.P.bin" 2>/dev/null || true; fi
#         fi
#         [ -f "$watch" ] || { sleep 2; continue; }
#         if ! pgrep -f "ds41_generate" >/dev/null 2>&1; then break; fi
#       done ) &
#     local watcher=$!
#     "${ENG[@]}" "${ARGS[@]}" "$@" --dump-last "$watch" >"$OUT.$name.toks" 2>"$OUT.$name.log"
#     local rc=$?
#     wait "$watcher" 2>/dev/null || true
#     cp "$watch" "$OUT.$name.D.bin" 2>/dev/null || true
#     echo "--- $name rc=$rc, $(wc -l < "$OUT.$name.toks") token ids"
#     tail -6 "$OUT.$name.log" | sed "s/^/[$name] /"
#     grep -m4 -E "relay: staged run|relay: token slab|layer split:|boundary hand-offs" "$OUT.$name.log" | sed "s/^/[$name] /"
#     [ $rc -eq 0 ] || { echo "PROBE: $name FAILED (rc=$rc) - see $OUT.$name.log"; return 1; }
# }

# run_one ctrl || exit 1
# run_one staged "${STAGED[@]}" || exit 1

# the staged run must actually have staged (a run that silently fell back measures nothing)
# grep -q "relay: staged run" "$OUT.staged.log" || { echo "PROBE: staged run never built a staged run"; exit 1; }
# grep -q "layers \[$SPLIT," "$OUT.staged.log" || { echo "PROBE: staged run did not build a stage at layer $SPLIT"; exit 1; }
# if grep -q "enc-skip: cut at layer" "$OUT.staged.log"; then echo "PROBE: staged ran with enc-skip ON (must be forced off)"; exit 1; fi

# ---- the diff
# $PY - "$OUT" "$NPRO" <<'EOF' || exit 1
# import sys, numpy as np, os
# out, npro = sys.argv[1], int(sys.argv[2])

# tk_c = np.array([int(x) for x in open(out + ".ctrl.toks")], dtype=np.int64)
# tk_s = np.array([int(x) for x in open(out + ".staged.toks")], dtype=np.int64)
# print("== TOKENS ==")
# n = min(len(tk_c), len(tk_s))
# lcp = 0
# for i in range(n):
#     if tk_c[i] != tk_s[i]:
#         break
#     lcp += 1
# print(f"  ctrl   : {tk_c[:12].tolist()}{' ...' if len(tk_c) > 12 else ''}")
# print(f"  staged : {tk_s[:12].tolist()}{' ...' if len(tk_s) > 12 else ''}")
# print(f"  longest common prefix {lcp}/{n}; first difference at generated-token index {lcp} (0-based),")
# print(f"  i.e. the decode at position {npro + lcp - 1}.  The witness's shape is lcp = 1 (token 2 wrong).")

# def cmp(tag, pc, ps, pos_label):
#     print(f"\n== {tag}  ({pos_label}) ==")
#     if pc is None or ps is None:
#         print("  MISSING dump - the watcher did not catch this write; see the .log files.")
#         return None
#     nv = min(len(pc), len(ps))
#     d = np.abs(pc[:nv].astype(np.float64) - ps[:nv].astype(np.float64))
#     ac, asx = int(np.argmax(pc[:nv])), int(np.argmax(ps[:nv]))
#     print(f"  dump shape: ctrl {len(pc)} floats, staged {len(ps)} floats (equal: {len(pc) == len(ps)})")
#     print(f"  max abs logit diff  : {d.max():.6e}")
#     print(f"  mean abs logit diff : {d.mean():.6e}")
#     for thr in (1e-6, 1e-4, 1e-2, 1e-1):
#         print(f"  entries differing > {thr:.0e} : {int((d > thr).sum())} / {nv}")
#     print(f"  argmax: ctrl {ac}, staged {asx}  {'SAME' if ac == asx else 'DIFFER'}")
#     top = np.argsort(-d)[:5]
#     print("  the 5 entries that moved most (vocab id, ctrl, staged, |diff|):")
#     for i in top:
#         print(f"    {int(i):7d}  {pc[i]: .5f}  {ps[i]: .5f}  {d[i]:.6e}")
#     marg = np.sort(pc[:nv])[::-1]
#     if len(marg) > 1:
#         print(f"  control's top-2 logit margin: {marg[0] - marg[1]:.6f}  (the knife-edge scale this model runs on)")
#     return d.max(), (ac == asx)

# def rows(p):
#     return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None

# P = cmp("PREFILL'S LAST ROW: position %d, the row that produces TOKEN 1" % (npro - 1),
#         rows(out + ".ctrl.P.bin"), rows(out + ".staged.P.bin"), "the prefill + relay verdict")
# D = cmp("LAST GENERATED ROW: position %d, the end of the decode chain" % (npro + len(tk_c) - 1),
#         rows(out + ".ctrl.D.bin"), rows(out + ".staged.D.bin"), "the decode-chain verdict")

# print("\n== WHAT THIS DISCRIMINATES (the three hypotheses a token file cannot separate) ==")
# if P is None or D is None:
#     print("  inconclusive: one of the two dumps is missing.")
# else:
#     pmax, psame = P
#     dmax, dsame = D
#     if pmax < 1e-5 and psame and dmax > 1e-3:
#         print(f"  (A) THE DECODE PASS IS THE SEAM: the prefill's last row agrees to {pmax:.1e} and the argmax matches,")
#         print(f"      yet the decode chain ends {dmax:.1e} apart.  The staged ONE-ROW PASS at position {npro} is where the")
#         print("      state first moves differently: dump the boundary x_state / pre_state of THAT pass, not the prefill's.")
#     elif pmax < 1e-5 and psame and dmax < 1e-4 and not dsame:
#         print("  (B) KNIFE EDGE: every logit agrees to float-order noise and only the greedy argmax flipped.  That is the")
#         print("      witness's own calibrated bar (two processes, same command, same flip), not a state bug.")
#     elif pmax > 1e-3:
#         print(f"  (C) THE PREFILL/RELAY IS THE SEAM: the prompt's last row already differs by {pmax:.1e}, so the head is the")
#         print("      messenger.  Bisect by dumping the boundary x_state / pre_state columns of the LAST prefill pass and")
#         print("      comparing THOSE; the decode pass is downstream and innocent.")
#     elif pmax < 1e-5 and psame and dmax < 1e-4 and dsame:
#         print("  (D) NO DIVERGENCE IN THE HEAD AT ALL: the two runs agree at both dumps.  If the token files still differ,")
#         print("      the sampling path (not the state) is where they part; if the token files AGREE, this probe's geometry")
#         print("      did not reproduce the witness and N_TOK or the prompt has to move toward the failing case.")
#     else:
#         print(f"  MIXED: prefill {pmax:.1e} (argmax same: {psame}), decode {dmax:.1e} (argmax same: {dsame}).")
#         print("      Read the per-threshold counts: a few entries = late-layer float order; a whole row shifted = the payload.")
# EOF
# rc=$?
# echo
# echo "probe-p5.3b-dumpdiff: artifacts in $OUT.* (logs, token ids, logit dumps)"
# exit $rc
