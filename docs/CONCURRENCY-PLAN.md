# Implementation plan: vLLM-style concurrency and checkpoint-tree KV in Strata

This is the build plan for the phases in [CONCURRENCY-RADIX.md](CONCURRENCY-RADIX.md). It changes
no code. Each phase below is opt-in; the default path stays byte-identical to the last release
(AGENTS.md), every file a phase touches gets built for CUDA, HIP (incl. this fork's gfx906
wave64 target), and SYCL where applicable before review, and every phase names the test that must
witness it plus the numbers it must produce. PRs are split one-change-each, per AGENTS.md.

Working tree for all of it: this checkout (gfx906 fork: `/ai/engines/gfx906/worktrees/xolotl-merge`,
branch `gfx906/xolotl-merge`, gfx906 build dir `build-906`, patched ggml at
`/ai/engines/gfx906/llama.cpp-xolotl` carrying `tools/gfx906/ggml/hip-cmake.patch`).
Host RAM is 125 GiB with no swap: every build and boot on this box runs behind the standing
`MemAvailable < 2 GiB -> kill` watchdog, and boots that need tens of GiB wait for the incumbent
workload to release memory.

## Order and why

```
P0 (measure what ships)  -->  G8 decision (policy, ~free)
        |                              |
        +----> P1 (window rows + parity tightening) <----+
        |                                                |
        +----> P3 (batched recurrent-state kernel) ------+  (P1's extended parity harness feeds P3)
        |
        +----> P2 (arena + checkpoint-tree radix)   [only if P0's census/reuse numbers justify]
        |
        +----> P4 (ported engines' per-session caches) [last; GLM/DS4.1 first boots ride T0-T5]
```

P0 is cheap, needs almost no code, and gates the expensive phases: it locates this box between
`docs/BATCHING.md`'s two regimes (all-experts-in-VRAM, where slots multiply throughput: 4 GPUs,
1/2/4/8 concurrent -> 120/113/205/360 tok/s, engine PR #559) and the 12 GB CPU-expert-card side
(where slots buy latency but cost up to -24% aggregate). The route-overlap census from P0 decides
whether P2 earns its keep; the per-step host-loop timing decides whether P3 is worth building at
all.

## Ground rules (apply to every PR)

- Opt-in only; default path byte-identical. A new flag defaults to today's behavior.
- Parity before performance: correctness claims ride exact-token (and exact-recurrent-state where
  applicable) comparison against solo behavior with `--pcie-frac 0`, `--adapt-every 1000000`,
  `--no-prefill-borrow` (BATCHING.md's full recipe), and on CPU-IQ quants with
  `STRATA_IQ_MT_MIN=1` (G8).
- ctest gate every PR against the known baseline (4 known failures on this fork:
  `split_lend_complement_test`, `mmvq_il_parity`, `verify_parity`, `ple_parity` - no new ones).
- gfx906 ISA gate: `hipcc --offload-arch=gfx906 -Xclang -mcode-object --save-temps` piped through
  `llvm-objdump-22 | grep -c "v_wmma"` must print 0 for any touched kernel (gfx906-build skill).
- HIP ggml builds require `STRATA_GGML_DIR` to carry `tools/gfx906/ggml/hip-cmake.patch`
  (enforced by `tools/ds4/cmake/ggml_cuda.cmake`).
- Every measured number lands in `benchmarks/gfx906/receipts/` with its build receipt, per
  `docs/COMMUNITY_BENCHMARKS.md` practice, and is written into the doc that cites it.

---

## Phase P0 - measure the shipped batch mode on this hardware (~1 week, ~0 new code)

**Prerequisite P0.0 - artifacts.** The mainline engine needs a mainline model artifact and a serve
config. Neither exists on this box yet (only GLM/DS4.1 ported-engine artifacts are complete;
`serve/` holds no model config). Steps: download/convert the Qwen3.8-Flash-Next Strata artifact
per `docs/MODEL-DOWNLOADS.md` + `setup.py`'s flow, generate its serve config, and record its
identity (sha, quant, ctx). This is operator-gated (disk/RAM): 869 GB free on `/ai` as of
2026-10-18. **Status 2026-10-19: CLOSED.** Three fork gaps stood between the first attempt and
"All set", each fixed behind the opt-in (36f91961, 8add6741, 5b578ef7 - details in daily logs and
START-HERE): the PC check refused gfx906; the engine step configured the wave32 HIP backend, which
refuses gfx906 (the fix passes the witnessed wave64 configuration); and a build-hip folder of the
other backend made CMake re-configure without setup's -D flags (cache now dropped on a backend
switch). Witnessed end state on this box:
- engine: `engine/strata`, BUILD.json `local-hip / hip / 0.1.44.1 / archs [gfx906] / src
  a44013c50376ca60` - the first setup-built wave64 engine on gfx906 (its configure line is the
  manual `docs/AMD_HIP.md` recipe's flags, plus the cache-purge guard);
- model: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ3_XXS, 2 shards, 75,966,073,120 B on
  disk, fingerprint `qwen4exp / 1224 tensors / bpw 3.434 / experts_bpw 2.842`
  (data/gguf_fingerprints.json), landed in `/ai/models/hf.co/Swift/models/`;
- pack: `/ai/models/hf.co/Swift/packs/swift-iq3_xxs` (dense.bin 1,538,035,200 B + tokenizer);
  MTP draft: `/ai/models/hf.co/Swift/mtp/rt` (786 MB);
  (moved from `worktrees/Strata-data/` the same evening at the operator's request - all model data
  lives under `/ai/models` now; `~/.config/strata/settings.json` `data_dir` points at the new root
  and a full setup re-run witnessed "already downloaded / model prepared" at the new paths, exit 0.
  Note: setup's `--data-dir` migration would NOT have found the default `Strata-data` - it only
  merges from folders that contain a `setup.py` (setup.py:4310) - so a move like this must edit
  `data_dir` and the config paths together, which is what was done, 5 paths, verified present.);
- config: `strata-swift-iq3_xxs.json` - ctx 131072, `--kv int8 --kv-resident 32768`, `--spec 4
  --spec-min-p 0.5`, `--expert-cache auto`, `--prefill auto`, exe `engine/strata`; start script
  `run-swift-iq3_xxs.sh`; server NOT started (--no-start).
Because the setup-built engine and `build-906/strata` are both wave64 builds of this tree, P0.2 can
take `--exe engine/strata` (the production identity the config uses) with `build-906/strata` as the
cross-check; the exactness recipe and the GPU/RAM window still apply (setup says ~43 GB RAM for this
quant; the incumbent currently holds the box).

**P0.1 - no-GPU server-side suite. DONE 2026-10-19.**
`python3 -m serve.test_parallel` from the repo root (NOT `python3 serve/test_parallel.py` - the
suite imports `serve.frontend`/`serve.server`, which needs the repo root on `sys.path`; BATCHING.md
does not record this). 30 cases over the protocol: BYIELD re-admission order,
solo<->slot transitions, per-lane stop semantics, fit handshake, SSE stream integrity (one
`[DONE]` per lane), `finish=length|stop` exactness, the batched sampler ignoring penalties,
`--batch-mtp` with 12 slots through 8-row windows. Witness recorded: **Ran 30 tests ... OK,
53.9 s, this checkout at `4b4a5b35`**, scripted fake engine behind the real `StrataEngine` +
`Service`, no GPU used.

**P0.2 - engine parity on gfx906 (GPU, minutes per run).**
`python3 tools/batch_test.py --exe build-906/strata --config <flash-next.json> --batch 4 --n 4`
then `--batch 8 --n 8`, then with `--extra "--batch-mtp"` (exactness recipe flags appended).
`tools/batch_interleave_test.py` for the prefill/decode interleaving (default `--long 5000`).
Witness: exact native-token parity (or a recorded G8-class divergence with its `STRATA_IQ_MT_MIN=1`
resolution). These runs also witness `INFO batch_slots=fit slot_cache=..` - the capacity handshake
- on MI50/HIP for the first time.

**P0.3 - route census (mainline change, ~1 day, opt-in).**
Add to the mainline router (`src/core/layer.cpp` path, same shape as the ported engines'
`--dump-routes`/`--rank`, `tools/v41/engine/ds41_dense.cpp:599,724-725`): an env/flag-gated
per-layer route histogram + pairwise lane-overlap counter, off by default. This is the only real
code in P0. Output: overlap % per window over a fixed script, comparable across quant sizes -
the gate input for P2. Prior numbers to compare against: 4 lanes disjoint cost 54 ms vs 20 ms per
window on an RTX 5070, 24 vs ~8 CPU-experts/layer (BATCHING.md); gfx906 numbers do not exist yet.

> **SHIPPED (instrumentation) 2026-10-19, task #7.** `STRATA_ROUTE_CENSUS=FILE` opt-in,
> `strata::core::RouteCensus` (header-only) hooked into `drive_pool`/`drive_pool_multi` in
> `generate.cpp` beside the existing `--dump-routing` tap - the two functions every decode path
> (solo, session, verify window, batch `run_slot_rows`, layer-split) funnels through, ids already
> host-pinned by the doorbell, so zero extra GPU->CPU syncs. Emits H (per-layer/expert hits), U
> (read factor: window union vs lane sum), J (pairwise lane Jaccard). Unset = byte-identical.
> Gate witnessed: HIP wave64 build + `route_census_test` (8 cases) + full ctest where the
> change-fails are a strict subset of a clean-parent stash run (`comm -23` empty). Census OUTPUT and
> live batch lane semantics still need the P0.2 GPU window. Receipt
> `benchmarks/gfx906/receipts/2026-10-19-p0.3-route-census-instrumentation.json`. NOTE for later
> gates: this box's parity-test fail-set is not a fixed list under GPU contention - the sound gate
> is a parent-subset comparison, not a hardcoded name list.

**P0.4 - batched tok/s at several sizes (GPU, ~half day).**
`"parallel": 1/2/4/8` (and 12 with `--batch-mtp`) on one MI50, then on the 4-card layer split with
`--batch-groups` default auto. Record against BATCHING.md's two regimes; locate the box.

**P0.5 - recurrent-state step cost (GPU, ~half day).**
Time the host-side slot loop in `src/core/verify.cpp:1208-1217, 1242-1255` at 1/2/4/8 slots
(engine-side timing behind the same opt-in flag; or `STRATA_BATCH_DECODE_SHARE=0` variants to
separate GDN work from attention work). This number alone decides P3: if the serial loop at 8
slots is far below the per-step budget, there is no 3-6-week kernel to build.

**P0.6 - checkpoint/restore cost curve (GPU, ~half day).**
The engine already logs `session restored ... in %.1f ms` (`generate.cpp:10665`): script
save/restore at 16K/64K/256K/1M ctx (the state is ~118 MB regardless, but the KV bytes scale), and
parking-cache restores (`--conversation-cache-mib`, note its 2,560 MiB RAM floor and that HIP
coverage is unevidenced per DETAILS.md - this PR witnesses it). Feeds P2's economics with real H2D
numbers, replacing the assumed ~120 ms.

**Exit criteria:** a `benchmarks/gfx906/receipts/` entry per step; route-overlap % and per-step
loop cost written into CONCURRENCY-RADIX.md replacing "UNMEASURED"; a go/no-go on P2 and P3.

**P0 EXIT 2026-10-21:** met. Receipts P0.0-P0.5 in `benchmarks/gfx906/receipts/`; both
UNMEASURED markers replaced (P0.4 census, P0.5 profile); **P2 NO-GO and P3 NO-GO on measured
ground**, both recorded in this plan and CONCURRENCY-RADIX.md. P0.6 was skipped by decision, not
by failure: it fed P2's economics (now NO-GO); its residual value - witnessing HIP coverage of
`--conversation-cache-mib` parking restores - remains open as ordinary ops verification, not a
phase gate. What the data says the next lever is: batched decode on this box is CPU-expert-bound
(~32% of an 8-lane window is the GPU waiting on CPU expert service), so batching effort belongs
to expert service, not to lane-count kernel work.

## Phase G8 - the exactness decision (free, do first)

**RESOLVED 2026-10-21: option (b).** Recorded in `docs/BATCHING.md` ("The decision (G8, this
fork, 2026-10-21)" paragraph): batched serving is "same model, not bit-equal" by default, the
exactness recipe is the documented opt-in, generated `"parallel"` configs keep the default
(byte-identical-default rule), and the recipe is what the comparison tools and this fork's
benchmark scripts (`tools/gfx906/run-p0.*.sh`) pass. Witness basis: P0.2 (12/12 slots
token-identical on gfx906 under the recipe) and the pre-existing default-settings drift
measurement (RTX 5070, cited in BATCHING.md). The planned CI job (`batch_test.py --extra
"<recipe>"` green) waits on a GPU runner - this fork's CI (`.github/workflows/gfx906-cpu.yml`)
is CPU-only - stated openly instead of silently dropped.

Decide and document ONE of: (a) batched serving guarantees exact solo parity by default (pay
`STRATA_IQ_MT_MIN=1` + `--pcie-frac 0` + `--adapt-every 1000000` + `--no-prefill-borrow` in
`"parallel"` configs); or (b) batched output is "same model, not bit-equal" and the exactness
recipe is a documented opt-in for parity-sensitive users. The repo has already chosen (b) in
effect (BATCHING.md documents the divergence); this phase makes it a decision instead of drift:
wire the recipe into CI's batch parity jobs and into the generated `"parallel"` configs.
Witness: CI job runs `tools/batch_test.py --extra "<recipe>"` green.

## Phase P1 - window rows for prefill, parity tightening (2-4 weeks)

One PR per item; each opt-in.

**Phase verdict 2026-10-21: CLOSED WITH EXEMPTIONS.** The three BATCHING.md "does not batch"
paths: penalties are batched (item 1, #55/#56); prompt-as-window-rows is exempted with
numbers (item 2, NO-GO: 74.8 vs 182.5 tok/s on identical prompts); serial admissions are
exempted with the same arithmetic (their total already runs at the measured chunked-service
cap - item 3 below). Failed windows keep exit+restart with a counted fault ledger (item 3, arm A).
Acceptance parity witnessed (item 4). Parity harness green on this box: 12/12 (P0.2), 4/4 x 3
arms (P1.1), 8/8 + ACCEPTANCE OK (P1.4) - all on the same engine tree, unchanged since
(`git diff 795ac8fe..HEAD` touches no engine file). Exit receipt:
`benchmarks/gfx906/receipts/2026-10-21-p1-exit.json`.

1. **Penalties under batch** - two options, pick in the issue first: apply them in the batched
   sampler (`src/core/verify.cpp:3218`, where the comment pins the ignore), or reject them at the
   server the way `logit_bias` already returns HTTP 400 under `--batch` (`docs/LOGIT_BIAS.md`).
   The 400 precedent is cheaper and already users' expectation; sampler support is the feature.
   Tests: `serve/test_parallel.py` case 22 must flip from "ignored" to the chosen behavior.
   **RESOLVED 2026-10-21 (issue #55, PR #56):** sampler support, applied per row from each slot's
   own staged history. No penalty request = the old path, byte for byte. Witness on MI50: greedy
   4/4 and seeded sampled 4/4 IDENTICAL to solo, repetition 1.1 and 1.8
   (`benchmarks/gfx906/receipts/2026-10-21-p1.1-penalty-batch-parity.json`). The planned "case 22"
   did not exist in `serve/test_parallel.py` (citation stale) - a new test was added instead
   (`test_penalties_reach_the_batch_slot`), pinning the server's BGEN forwarding.
2. **Prefill chunks as window rows** - the last un-batched mainline path is the chunked-prefill
   GEN step. Citations re-pinned 2026-10-21 (the old `:6947-7000`/`:5679` point at KV chunk-mapping):
   the time-share is `STRATA_BATCH_DECODE_SHARE` + the interleave in `read_part`
   (`src/program/generate.cpp:11747-11750`, `:11825-11831`; one chunk per piece beside slots,
   `include/strata/program/batch_read.hpp:22-30`); the "BGEN reads as a GEN 1" pin is
   `rewrite_batch_admission` (`include/strata/program/native_trace_id.hpp:56-59`) plus the
   one-produced-token continuation (`generate.cpp:13183`).
   **Feasibility (source-mapped 2026-10-21): the layout already exists.** A slot owning N
   contiguous rows at p..p+N-1 is exactly what `batch_rows_error` validates for `--batch-mtp`
   (2-row groups), and GDN conv/recurrence, indexer, PLE chain and commit already execute
   same-slot groups (`src/core/verify.cpp:1208-1252`, `:2877-2899`, `:3167-3172` keep-all =
   teacher-forced). Short prompt parts already read through windows (`read_windows`,
   `generate.cpp:11537-11595`, `--short-read 64`). What a `--prefill-rows` build would add is a
   row class (prompt rows: commit-all, no BT/eos/penalty-history/nonfinite-slot-kill, per-slot
   sampling is per-slot), homing the prompt in a carved slot session, and a stable row layout
   (captures key on the layout, 20-30 MiB/pair).
   **Gate before building (this box's regime):** a window row costs what the row pays - the
   witnessed 8-row batch window is 131.24 ms (~16 ms/row, P0.2 b8 log; the engine's own comment:
   a window ~16 ms/token vs a chunked run ~300 ms however few tokens, `generate.cpp:11405-11409`)
   - so a prompt read as window rows advances <= 8 - (decodes) tokens per window and can lose the
   task's dual gate (slot AND prompt tok/s >= time-share) on the prompt side by arithmetic.
   `tools/gfx906/run-p1.2-economics.sh` measures both sides on this box (chunked vs `--short-read
   999999` windowed read of the same long prompt while slots decode); NO-GO with those numbers
   closes the item without building, within ~1.5x it justifies the build. Extends
   `tools/batch_test.py` / `batch_interleave_test.py` with a mixed chunk+decode scenario.
   **RESOLVED NO-GO 2026-10-21 by the probe** (same long prompts, same contention, exactness
   recipe): windowed read 74.2-74.8 tok/s vs chunked 182.5-184.9 = **2.5x slower** on the prompt
   side (13.5 vs 5.5 ms/token, matching the engine's own ~16 ms/token comment), while the
   time-share control already fed the decoding slots 398 tokens plus 8912 ms of whole windows
   between parts. A mixed window pays the same rows with fewer of them, so it cannot beat the
   prompt-side number; the gate needs both sides. Side finding: a windowed long read takes no
   BYIELD and no decode-share (those live only in the chunked path) - the feature would also
   have to port them. The layout stays expressible today for a regime where rows are cheap
   (VRAM-resident model, ~2.8 ms/row); this machine's CPU-expert regime is the exemption, with
   numbers. Receipt `benchmarks/gfx906/receipts/2026-10-21-p1.2-prefill-rows-nogo.json`.
3. **Failed-window blast radius** - **RESOLVED 2026-10-21 (arm A: keep exit+restart, documented with
   numbers + a counted fault ledger)**. Repair-in-place stays refused (upstream #24: after a CUDA/HIP
   submission the device state is uncertain - "never clear busy/live state"); the alternative is
   priced and parked: resuming one row needs a pre-window snapshot of every slot's recurrent state
   (118 MB per slot on a GDN model), ~2.7 ms of the measured 115 ms 8-row window - ~2.4% of decode
   in every window, healthy or not (computed from bytes over the box's HBM bandwidth, not yet
   measured), ~1 GB of shadow VRAM, and the harder problem of not trusting state captured after a
   sticky error. Restart cost is measured: 30-35 s to READY warm (the P1.4 arms each fit an engine
   boot plus a full test into 45-65 s), 1-3 min cold, plus 1.1-1.8 s to re-admit 2-4 slots (P0.2) -
   ~37 s per fault. The crossover `3600 x 2.4% / 37 s` is **2.4 faults/h**, so the decision is only
   valid while faults stay that rare - and that is now counted, not assumed: every non-deliberate
   engine death is appended to `engine-faults.jsonl` beside the engine log (its own reason, live
   streams, the restart it cost) and one server-log line carries the last-hour rate plus the
   row-recovery reminder once it crosses 2.4/h (`serve/faults.py`, `StrataEngine._fault_report`,
   test `test_a_window_fault_is_ledgered_with_the_crossover_reminder`; 32/32 in
   `python3 -m serve.test_parallel`). Fault rate on this box so far: 0/h across every P0/P1 run;
   the row-level class that does occur (#879 non-finite) never needed a restart. docs/BATCHING.md
   "Failed windows".
4. **`--batch-mtp` drafts path in the harness** - **tool shipped 2026-10-21** (`d2abbc35`):
   `tools/batch_test.py --acceptance` compares the tallies the engine already logs - the slots'
   drain tally (`generate.cpp:10283`) against the solo windows' FIRST proposal (spec-depth ladder
   depth 0, `:13354` with `STRATA_SPEC_DEPTH=1`, which the flag sets) - within `--acceptance-tol`
   pp or the 2-sigma counting band, whichever is wider; `--parse-check` proves the readers with no
   engine (11/11 on this box). **Witnessed 2026-10-21** (MI50, `--batch-mtp`, exactness recipe,
   batches 2 and 4): 8/8 slots IDENTICAL to solo; first-proposal acceptance solo 83.7%/87.1% vs
   batch 77.5%/77.5%, diffs -6.3/-9.6 pp inside the counting bands (+/-15.5/+/-10.7 pp) - both
   arms ACCEPTANCE OK, cross-checks exact (BT tokens == windows + accepted). Open observation:
   the batch rate sat below solo in both lanes; a `--max-new 256` run would decide whether the
   gap is real (candidate mechanism: the batched slot's private per-slot drafter K/V). Correctness
   is untouched - identical tokens, and a rejected proposal can only cost a slot speed, never a
   wrong token. Receipt `benchmarks/gfx906/receipts/2026-10-21-p1.4-batch-mtp-acceptance.json`.

Exit - **met 2026-10-21**: penalties batched (#56); the prompt-read and admission serialization
exempted with numbers (both are the same window-row economics, measured 2.5x slower as rows);
parity harness green on gfx906 (P0.2/P1.1/P1.4 receipts on one unchanged engine tree). Closed
with exemptions, no open P1 work; the surviving P1 observation (batch acceptance -6..-10 pp,
band-limited) is recorded in the P1.4 receipt with the run that would settle it.

## Phase P3 - batched recurrent-state kernel (3-6 weeks, ONLY if P0.5 says the loop costs)

**Gate resolved NO-GO 2026-10-21 by P0.5** (63 batch windows/arm, `STRATA_VERIFY_PROFILE` stage
stamps bracketing the per-slot loops; calibration: profile total within 0.5-1% of the host-run
ms/window at all three lanes): conv+rec at 8 lanes = 3.43 ms of a 115 ms window (3%), all GDN
recurrent stages together 7.8%, growth sublinear-to-linear in lanes; ~32% of the window is the
GPU waiting on CPU expert service (`waitCPU` 36.4 ms/window at b8), which this phase does not
touch. Receipt `benchmarks/gfx906/receipts/2026-10-21-p0.5-recurrent-step-cost-p3-nogo.json`.
The measured side observation that outranks P3 for GPU-side batching work: QSA attention grows
0.49 ms solo to 8.54 ms at 8 lanes per window.

1. Batched GDN/PLE/indexer variants taking a state-stride grid dim
   (`include/strata/kernels/verify_kernels.hpp:41-44` shows the single-state signature today);
   wave64-safe; per-state independent so ordering cannot couple lanes.
2. Runtime window length: replace compile-time `kVerifyMaxT = 8`
   (`include/strata/kernels/verify_kernels.hpp:20`) with runtime-sized windows ONLY in the batched
   path; the solo pipeline keeps its static shapes (blast radius: solo buffers, `kHistSlots`,
   exact-T instantiations, CPU `MAXT`).
3. Parity: reuse the abcd parity contract (`tools/abcd-parity-contract-v2.json`,
   `abcd_parity_test.py`) extended to N-lane drafted acceptance from P1.4.
Exit: per-step GDN cost at 8 lanes <= measured solo-equivalent; exact-state parity green.

## Phase P2 - arena + checkpoint-tree radix (1-2 months, ONLY if P0 census/reuse justifies)

**Gate resolved NO-GO 2026-10-20 with the first live census** (box: 1x MI50, Swift IQ3_XXS,
unrelated conversations, batches 2/4/8; receipts
`benchmarks/gfx906/receipts/2026-10-20-p0.4-census-sweep.json`): mean pairwise lane route Jaccard
0.7-1.9%, within-window read factor 1.04/1.31/1.44 at 2/4/8 lanes (b8 per-layer spread
1.14-1.58). There is no cross-lane route reuse to share at ~2% overlap, so the checkpoint-tree
radix does not start. Revisit only for shared-prefix-heavy traffic (same system prompt across
lanes) - there the win is prefill reuse, not decode route sharing.

Sequence so each step is independently shippable and opt-in:

1. **P2a - arena allocator** (`src/core/vmm.cpp` HIP branch + a new frame arena): one large
   per-stage `cudaMalloc` with offset arithmetic, refcounts, LRU free list. HIP has no VMM
   (`vmm.cpp:154-157`), and no VA contiguity is needed because kernels address pages only through
   `page_table` (`qsa.hpp:70-77`; permuted-table tests at page_size 1/4/512). Default off:
   today's per-session pool allocation stays byte-identical. Test: allocator unit tests +
   `--max-context x N` VRAM reduction with unchanged tokens on `bench` runs.
2. **P2b - frozen-prefix nodes** - generalize the existing `pin=N` machinery (`vmm.cpp:284`) to
   refcounted shared frames, WITH the write-semantics decision settled by a rewind-path audit
   (copy-on-write vs private-suffix; this is the correctness crux, not the allocator).
   Parity gate: KV-pinning parity tests + byte-parity against solo with `--pcie-frac 0`.
3. **P2c - checkpoint tree** - nodes = {refcounted frozen pages, sparse GDN snapshot at node
   depth, token-key edges}, inserted ONLY at depths the chain already checkpoints (root/turn/pin/
   every-16K - `conversation_state.cpp:230-236`, `state.cpp:30-38`). Not page-granular: at 4-token
   pages the 118 MB state is ~2,300x the page's 4 x 12,672 B. Reuse gate: a node may be reused
   only under a token-equality check (its own tokens, not a hash) and a VRAM headroom check - the
   `can_fit` + parking-cache RAM-floor pattern already models admission
   (`generate.cpp:4573-4611`, `--conversation-cache-min-free-mib`).
   Prior art to build on, with its caveats carried: `docs/MESSAGE_BOUNDARY_CACHE.md`
   (first-edit -33.1%, reuse 22,016 -> 32,831; excludes layer splits, chunk-geometry FP caveat).
4. **P2d - cross-session shared prefixes** (system prompts, few-shot sets): multi-pin, LRU across
   sessions, `serve` side announces hits. Exit: two-session same-prefix followups reuse tokens and
   produce byte-identical continuations under the exactness recipe.

## Phase P4a - GLM + DS4.1 first boots on gfx906 (the headline; operator-window gated)

Tasks #18-#24. Compilation is already a build gate, not a runtime claim - nothing has ever booted
here. T0's decisive gate (patched ggml as HIP/gfx906) is PASS (T0 table re-checked 2026-10-21:
`build-906/ggml/src/ggml-cuda/libggml-cuda.a` 74,266,036 B, `glm_generate` 81,171,976 B,
`ds41_generate` 77,374,200 B); T1/T2 artifacts and the GLM tokenizer are on disk. Each task ends
with a receipt; correctness, state and performance are recorded separately per AGENTS.md.

1. **P4.0 (#18) - bootable wrappers + local DS4.1 config + engram gate** (no GPU). WITNESSED
   blocker 2026-10-22: `systemd-run --user --scope` exits 1 here (root session, no user bus;
   `systemd-run --system --scope` exits 0) - both merged wrappers exec `--user`, so GLM/DS4.1
   serve would die before the engine starts; fix with an auto-probe that keeps the working
   `--user` path byte-identical upstream and falls back to `--system` here. Add
   `strata-ds41-906.json` (local V41-GSQ / `engram/l1:l14` / `rank.bin` / `tokenizer` paths) and
   `ds41-engine-906.sh`. The engram integrity gate is `check_engram_tables.py` (header + engine-
   expectation check), NOT `verify_engram.py` - the FP8 source shards were deleted, so the
   byte-vs-source verifier cannot re-run and the tables (95 GiB each) are the sole copy.
   **SHIPPED 2026-10-22** (receipt `2026-10-22-p4.0-prep-wrappers-engram-gate.json`): red control
   (pre-fix wrapper rc=66 at flock on this box) and fix witness (stub engine through the real
   wrappers: rc=0, `memory.max = 90194313216` = 84 GiB exactly, `swap.max = 0`); the engram gate
   runs PASS in 0.34 s (all 4 tensors exact-size, 16/16 reads); both smoke scripts refuse at rc=3
   against a stand-in engine holder. A second blocker surfaced and died here too: `flock` never
   creates its lock file's parent directory (rc=66 before the engine existed), and `glm_tok.load`
   needs `ST.PRE_PATTERNS.setdefault("glm4", ...)` first - exactly what `serve_glm.py:37` does,
   mirrored in the smoke script.
2. **P4.1 (#19) - RAM/arena budget** (no GPU) — **CLOSED 2026-10-22**
   (`docs/gfx906/P4-RAM-BUDGET.md`). Both engines route the host expert cache through
   `PinnedArena` + `cudaHostRegister` (HIP userptr) — the SAME driver budget class as the
   2026-10-20 reset (`kfd_svm.c:343` charges USERPTR against the resident-system-memory
   limit), so the witness `cudaHostRegister ... PORTABLE ok` is grepped in both smoke scripts
   (degraded pin = tested the fallback, not the pinned tier). Ledgers: GLM steady ≈ 69 GiB
   (arena 60 + dense 7.3 + ~2 est); DS4.1 steady ≈ 73.5, transient ≈ 81 during the
   Q8_0→Q6_K requant peak — 3 GiB under the 84 GiB scope cap (authors run the same numbers;
   if the scope OOM-kills, `ARENA_GIB=60` is a catchable exit, arena already resident).
   Refuse floors: arena + dense 8 + scratch 3 + 10 margin ⇒ GLM ≥ 81, DS4.1 ≥ 85; script
   defaults (arena+25 = 85/89) are stricter and stay. Smoke ctx: GLM 32768 explicit; DS4.1
   auto-4096 (`ds41_generate.cpp:286`). Both engines at once is arithmetically impossible
   (124 > 125 GiB of arenas). The exact driver limit is NOT readable here (partial /usr/src;
   only `no_system_mem_limit=N`); witnessed envelope 49 GiB single-engine OK vs 49+40
   two-engine crash — P4.2 EXTENDS it. 512K (P4b/P5): GLM 6.5 / DS4.1 1.25 GiB VRAM KV,
   RAM unaffected.
3. **P4.2 (#20) - GLM T4 direct-engine smoke** (operator-launched). 8 tokens, witnesses: 4-shard
   split geometry read (Xolotl tuned on 3 shards - the boundary arithmetic is untested here), no
   `has no kernel` error (IQ3_S gate/up / IQ4_XS down), no shard-layout error, tokens out.
   Failure ladder: graph-off (`GGML_CUDA_GRAPH_MAX_TOKENS=0`), `--experts cpu`, wider vram-margin.
   **RAN 2026-10-22, FAILED all three rungs at one deterministic point** (`HipVMM Failure: invalid
   argument`, ggml-cuda.cu:647, after `tier: gpu ... load 42.9 s`): root cause was OUR
   `hip-cmake.patch` - `common.cuh:264` honors only `GGML_HIP_NO_VMM` under HIP and the patch set the
   CUDA-side name, so the VMM pool compiled in; the ROCm 7.2 attribute probe reports `VMM: yes` but
   `hipMemSetAccess` aborts at first pool growth. Fixed (patch adds `GGML_HIP_NO_VMM`; both apply
   directions gated; ggml-cuda + engines rebuilt, def in 143 TUs). Same window also showed the
   cpuexp rung silently ran GPU experts (`${ARGS[@]/--experts gpu/...}` matches no element - pairs
   are separate array entries); fixed. Boot facts before the abort: 4-shard geometry read, 45 layers
   (34 KDA + 11 MLA - **11, not the 12 the budget doc assumed**), 288 experts top-8, vocab 154880,
   dense half 3.9-7.2 s, 2014 slots, arena 5522 experts, file tier 4560, tier load 35-43 s.
   **RE-RUN NEEDED** in a free window (vLLM must be down).
   **RUN 2 2026-10-22** (logs `p4.2-*-20261022-063849`): VMM fix witnessed (`VMM: no`) and the
   **pinned-tier witness CLOSED POSITIVE** - `arena: pinned (cudaHostRegister PORTABLE ok...)` on
   the 60 GiB arena (single-engine pinned capacity now witnessed at 60-64 GiB; the 2026-10-20 crash
   was two-engine). New failure, deterministic: `ds4_moe: c_stage: out of memory` - chunk-MMQ staging
   allocates TWO device halves of `max(1024 MiB, blob)` AFTER auto slots took 21.89 of 23.89 free
   (margin 2.0); 2.0 GiB staging + KV did not fit the ~1.6 GiB left. Fixed: smoke runs margin 4.0
   (slots 19.89, staging fits with headroom). cpuexp rung additionally OOM'd on the 6.8 GiB dense
   half over a not-yet-drained card (prior rung SIGABRT frees VRAM asynchronously) - both smokes now
   sleep 10 between rungs.
   **CLOSED PASS 2026-10-22** (log `p4.2-base-20261022-180307`, after the ODR fix `e13f1555`): base
   rung (CUDA graphs on, GPU experts) generated 14-token prefill + 64 decode tokens = **2.57 tok/s**
   (3.6x the CPU-experts rung); experts/token 23.3% VRAM + 22.6% PCIe + 54.1% cpu; VRAM free at end
   3.28 GiB (margin 4.0 held). Bring-up numbers, one MI50 of four, cold cache - not a perf claim.
   **RUN 3 2026-10-22** (logs `p4.2-*-20261022-073832`): margin 4.0 fixed the c_stage OOM and the
   **cpuexp rung GENERATED TOKENS** - prefill 14 tok + 64 decode at 0.71 tok/s, clean finish
   (PASS-CPU-ONLY ladder: engine, weights, tokenizer, file tier all work end-to-end; GPU expert
   path NOT witnessed). base+nograph moved to the SAME glibc heap-corruption abort DS4.1 shows
   (`malloc(): invalid size (unsorted)`) right after tier load - the failure is now common to BOTH
   ported engines on the GPU-experts path; CPU experts clean brackets the corruptor to that path.
4. **P4.3 (#21) - GLM serve** (operator-launched): the `--serve` scope + stdin/stdout protocol
   first live run; LoRA (`--lora-exps` on IQ3_S) first exercise; first GLM decode tok/s.
5. **P4.4 (#22) - DS4.1 engram smoke** (operator-launched): 128 tokens, engram integrity gate
   green first; NO `not the real model` warning; `STRATA_NO_LARGEPAGES=1` (pins the arena at
   ~1.2 GB/s, load ~50 s). Run plain (no `--prune`/`--route-bias`/`--build-mirror` - this box
   lacks the authors' profile/mirror files).
   **RUN 2 2026-10-22** (logs `p4.4-*-20261022-064342`): the new engram gate passed its first live
   use (both tables, exact rows×dim, 4/4 spot-reads) and the 64 GiB arena pinned `PORTABLE ok`.
   `loaded in 64.4 s: 40 layers, ctx 32768` then base crashed CPU-side: `malloc(): invalid size
   (unsorted)` (heap corruption) at the first prefill batch (`enc-skip: first batch B=14, cut=0`);
   cpuexp refused the tier run with no reason printed. Dispositions: added a `--no-enc-skip` rung
   to attribute the corruption; both refuse sites now print layer/token-count/path (rebuilt);
   `--ctx 32768` removed from the smoke (P4.1 verdict is the engine's auto-4096 - the script had
   contradicted the budget doc). Heap corruption class: NOT papered over - if the noenc rung also
   crashes the next triage is an ASAN/MALLOC_CHECK build of ds41 on the CPU path.
   **ROOT-CAUSED 2026-10-22 (ASan)**: one bug, both engines - `heap-buffer-overflow WRITE of
   size 1040, 0 bytes after 4336-byte region` at the same instruction (`mmq::Context::Context()`,
   `moe_mmq.cu:361`). strata_mmq TUs include ggml-cuda's `common.cuh` and inline-construct
   `ggml_backend_cuda_context` but lacked its layout-affecting defines (`GGML_CUDA_NO_VMM`,
   `GGML_HIP_NO_VMM`, `GGML_CUDA_USE_GRAPHS`): `new` sized moe_mmq's 4336 B view while the
   COMDAT-folded ctor and out-of-line dtor use ggml's 4400 B view (probe: 4336 vs 4400,
   `cuda_graphs` at 3256; with the defines, byte-identical). Fixed in the live CMake branch,
   gated `STRATA_HIP_GFX906` (CUDA-native byte-unchanged); witness: objdump `movl $0x1130`
   (=4400) in the rebuilt `ContextC2Ev`. Receipt `2026-10-22-p4-odr-rootcause-fix.json`.
   **CLOSED PASS 2026-10-22** (log `p4.4-base-20261022-180447`): engram gate 4/4, then base rung
   decode **128 tokens in 8.19 s = 15.50 tok/s** with **99.3% VRAM expert hits** (30261 hit / 68 cpu
   / 151 pcie / 17 file); the vram_lru lazy fill warmed during prefill exactly as P4.1 predicted.
   Both engines also re-ran clean under ASan (`rc=0`, no report) - the ODR fix removes the
   corruption at the instrumented level, not just the glibc symptom.
   **RUN 3 2026-10-22** (logs `p4.4-*-20261022-074337`): the `--no-enc-skip` rung ALSO crashed with
   `malloc(): invalid size (unsorted)` before any batch line - **enc-skip exonerated**; the
   corruption is on the common GPU-experts path both engines share. The new refuse-reason print
   caught the cpuexp rung's real problem: `(layer 0, 14 tokens, device path)` - with `--backend
   cuda` the layer-major path always hands device activations to `run_chunk_dev`, which refuses
   `cpu_only`: ds41 has NO GPU-dense/CPU-experts combo (GLM does). cpuexp rung now forces
   `--backend cpu -n 8`. Next triage tool shipped: `tools/gfx906/run-p4-asan.sh` (AddressSanitizer
   build `build-asan-906`, host TUs instrumented; names the corrupting store's stack instead of a
   later malloc's abort).
6. **P4.5 (#23) - DS4.1 serve** (operator-launched): exact-serve default (`STRATA_SERVE_STEP_MAX=
   96`), engram live, first DS4.1 decode tok/s + the NVMe read-factor if the file tier is the cap.
   **CLOSED PASS 2026-10-22**: through the `serve_ds41.py` shim (plain `serve.server` renders the
   comment-only chat template to an empty prompt - engine refuses loudly, `d27331de`), first request
   answered **391** with `finish_reason=stop`, decode **15.5 tok/s** in serve = the P4.4 smoke number
   exactly (serve layer adds no measurable B=1 decode overhead); prompt 6.3 tok/s cold.
   Receipt `2026-10-22-p4.5-ds41-serve-pass.json`.
7. **P4a exit (#24)**: XOLOTL-PORT.md port log + P4 verdict; every known-unknown dispositioned;
   START-HERE.md validation boundary updated; parity-vs-authors (250 prefill / 17-30 decode tok/s
   on 4090+90 GB) stated as a LATER, separate claim - this box is a different machine.

All of P4.2-P4.5 are one-engine-at-a-time operator windows (the standing crash rule) and refuse
below the P4.1 threshold.

## Phase P4b - ported engines' per-session caches (last; follows P4a)

- DS4.1 (`tools/v41/engine/`): its caches are positional, so per-session page tables + the
  P2b/P2c pattern port nearly directly. GLM (`tools/glm/engine/`): KDA state per session first
  (`glm_session.hpp:16-18` already holds S/F per session), shared prefixes last - state
  save/restore at node depth is the same economics as Part 2 of the analysis.
- `glm_generate`/`ds41_generate` are single-device today; multi-card layer split is P5 (do P5.2
  first, then P4b reads its must-not-hardcode device list from the design doc), and any
  replica-scale work (serve/`replicas.py`, whose per-process weights duplicate CPU RAM on a
  125 GiB box - a shared pinned arena is the same fix as P2a) comes after. 512K context lives here,
  not in P4a's smokes.

## Phase P5 - 4-GPU support for the ported engines (after P4a; tasks #25-#31)

The ported engines are single-device: one ggml backend per process (`im.backend`, 36 touchpoints in
`tools/glm/glm_dense.cpp`, 26 in `tools/v41/engine/ds41_dense.cpp`; the expert tiers use raw CUDA/HIP
calls with device 0 implied). The native engine already ships the pattern to follow: `--gpus 0,2`,
`--layer-split K1[,K2..]|auto`, `--split-device D1[,D2..]` (`docs/MULTI_GPU.md:38-39, 99-100`) with
per-stage `(cache, stream, device)` resolved by `locate(layer)` (`src/program/generate.cpp:284-285`,
`src/core/expert_source.cpp`, `src/core/peer_experts.cpp`). Layer split is the natural axis for MoE -
tensor parallel is not on the roadmap. One engine on four cards also stays inside the one-HIP-engine
crash rule; 4 replicas do not.

Why split pays here, arithmetic from file sizes (P5.4/P5.5 to witness): GLM's routed experts are
148.5 GB - one card's ~22 GB free after the dense 7.81 GB + KV caches ~15% of them; four stages
each cache ONLY their own ~37 GB of experts and can hold ~24-28 GB hot (~65-75%). DS4.1: 154.7 GB
experts, same shape per stage ~38.7 GB. Aggregate expert bandwidth 4 x ~1 TB/s HBM2. A stage
boundary costs one hidden-vector copy per token (KBs; PCIe Gen4 x16 ~25 GB/s) - the relay is not
the problem, cache residency is. P4b's per-session caches land AFTER P5.2 inventories the
structures so they can carry a device field from day one.

1. **P5.0 (#25) - source map + design (no GPU).** Inventory every device-choosing site in both
   engines: backend creation/`ggml_backend_alloc_ctx_tensors_from_buft` (`glm_dense.cpp:210, 883,
   898, 997`; `ds41_dense.cpp:318`), raw allocations/memcpys in the expert tiers and transcode,
   hipGraph capture, `ggml_backend_cuda_register_host_buffer` (`ds41_dense.cpp:1135-1144` - pinned
   registrations are per-device), arena/cache ownership. Output `docs/gfx906/P5-MULTI-GPU-DESIGN.md`:
   the change list, the flag surface (mirror the native names `--gpus/--layer-split/--split-device`),
   and what P4b must not hardcode. **CLOSED 2026-10-22** — `docs/gfx906/P5-MULTI-GPU-DESIGN.md`:
   topology measured on this box (4× Gen4 x16 at max, all pairs PCIE weight-40/2-hop, NO Infinity
   Fabric — risk-register assumption confirmed; BDF table in §0); one backend-creation chokepoint per
   engine (`glm_generate.cpp:208`, `ds41_generate.cpp:282`) and zero `cudaSetDevice` in ported code;
   36/26 `im.backend` touchpoints line-listed; tier raw-call counts 180/286 device-0-implicit (reuse
   native `On(dev)` RAII); the two quoted native traps (mmq-context segfault `ds41_moe.cpp:401-403`,
   wrong-owner cache read `expert_source.cpp`) carry into the design; relay sync-first (8/10 KiB per
   token); native `--layer-split K --split-device 0,0` single-card bit-exact check adopted as P5's
   G8-equivalent; hidden dims measured from artifacts (GLM 4096, DS4.1 5120).
2. **P5.1 (#26) - interconnect probe.** **CLOSED 2026-10-22, RUN WITNESSED** (`logs/p5.1-probe-20261022-055705.log`,
   receipt `2026-10-22-p5.1-interconnect-probe.json`): all 12 ordered pairs P2P-capable; peer-direct ~26.9 GiB/s
   only INSIDE a complex and ~7.3 GiB/s across complexes; **host-pinned staging uniform 25–26.7 GiB/s everywhere**
   ⇒ relay transport decision = **host-pinned bounce** (peer-direct buys nothing; cross-complex peer is 3.6×
   slower than bouncing); no Infinity Fabric (peer ≈ staged, all-PCIe signature); HIP enumerates 43,46,23,26
   (ordinals ≠ sysfs order — map ordinals→BDFs). Cross-complex bidir aggregate (~44 GiB/s vs 7.3 unidirectional)
   is unreconciled; the design takes unidirectional as the constraint.
3. **P5.2 (#27) - device plumbing (code; builds gate, no runtime claim).** Opt-in
   `--gpus D1,..` on `glm_generate`/`ds41_generate`: per-stage ggml backend + weight placement,
   expert tier device guards (set-device around raw calls), per-device pinned-buffer registration,
   expert-cache arena partitioned per stage. Default (flag absent) = today's code path, byte-identical.
4. **P5.3 (#28) - activation relay + pipeline (code).** First: synchronous boundary copies on
   per-stage streams with event handoff (correct, measurable). Only if that measures as a cost:
   the `--pipeline-windows` analogue (stage N+1 overlaps stage N's next token).
5. **P5.4 (#29) - split smoke (operator window).** Rerun the P4.2/P4.4 smokes with the split on,
   ladder 2 cards -> 4. Witnesses: stages built with per-card VRAM as `rocm-smi` shows it; greedy
   output IDENTICAL to the single-card run (this is the ported engines' G8 - same kernels per
   layer, exact copies, so equality is the claim to test); tier-fill lines show the per-stage
   hot fraction the header arithmetic predicts.
6. **P5.5 (#30) - perf characterization + serve (operator window).** Decode tok/s 1 -> 2 -> 4
   cards, same day, same configs (P4.3/P4.5 baselines are the control); CPU-service share per
   stage (the split equivalent of P0.5's waitCPU 32%); prefill side; serve round-trip.
7. **P5 exit (#31).** Receipt + `XOLOTL-PORT.md` port-log entry + START-HERE.md boundary update;
   scaling numbers with the regime stated (expert-streaming box, PCIe relay); P4b unblocked.

Exit criterion: a ported engine runs on 2 and 4 MI50s with token-identical greedy output vs its
single-card run, and the measured scaling is recorded whatever it turns out to be - a sublinear
result with numbers closes the phase too (the P1.2 pattern).

**P5 EXIT VERDICT (2026-10-26, #29/#30 closed, receipt `2026-10-26-p5-exit.json`): CONDITIONAL GO,
by engine and by claim class.** CORRECTNESS: the DS4.1 staged engine is BITWISE-equal to its unsplit
control - 64/64 at two cards (witness 20261025-190058), 128/128 at four cards, and 48/48 with the
state column moving MID-PREFILL at a 647-token prompt (witness 20261026-231105); the GLM staged
engine FAILS its calibrated bar (control pair 64/64 exact, staged diverges at token 16; the admission
confound is quantified and partial) and carries defect #40 - GLM staged is NO-GO until that closes.
STATE: every staged geometry claimed by its plan lines was built and witnessed, per-card VRAM spread
matches the placement design, and the predicted per-stage expert caches are REFUTED with the engine's
own counters - the tier is one card's by construction (that is #39's territory, not P5's). PERFORMANCE:
decode is faster on two cards in both paired boots (+85%, +41% - entirely DENSE spread, expert-path
counters identical across arms); prefill scales 3-5x NEGATIVE under staging with the limiter named
(4.3x file-tier read amplification; the relay itself costs 2.70 ms/pass warm and is never the limiter);
the 4-card decode mean is NOT separable from control at n=2 because within-arm boot variance is as
large as between-arm differences - so no scaling claim stronger than the paired comparison is made.
BOUNDARIES the checkpoint refuses by design: `--serve` and DSpark on a staged run, Engram boundaries
above layer 14, and (arithmetically) staged runs past ~26 state-column windows per stage (~1700 tokens
at cap 64) until the per-stage ggml graph pool grows or evicts. P4b is unblocked for the ported engines'
per-session caches with the device ownership from #25's inventory.

## Cross-cutting

- Docs: each phase updates CONCURRENCY-RADIX.md's gap table (gap -> shipped, with the receipt
  path), BATCHING.md's limits, and DETAILS.md; no claims without a measurement.
- PR split table (one change each, reviewable alone):
  PR1 route census -> PR2 G8 CI wiring -> PR3 penalties -> PR4 prefill-rows -> PR5 window
  recovery -> PR6 batched-GDN -> PR7 runtime window -> PR8 arena -> PR9 frozen-prefix ->
  PR10 checkpoint-tree -> PR11 cross-session -> PR12+ ported-engine caches.
- This fork's identity rules: each PR records source identity; experiment paths stay behind the
  gfx906 gates; upstream-facing work rebases onto `gfx906/v0.30.0` per the standing rebase-park
  note.

## Risk register

| Risk | Mitigation |
| --- | --- |
| P0 shows low route overlap on real traffic -> P2 shares nothing | P0.3 is the cheap gate; P2 simply does not start |
| P0.5 shows the serial loop is not the limiter -> no P3 | P3 is conditional on the measurement by design |
| batch + layer split interaction (windows, `--pipeline-windows` off under `--batch`) caps split gains | P0.4 measures `--batch-groups` directly before any split-side code |
| HIP paths of shipped-but-unvalidated features (parking, KV-prefetch, conversation cache) misbehave on gfx906 | P0.2/P0.6 are explicitly first-witness events; failures become issues, not assumptions |
| Write-semantics of shared frames break a rewind path nobody thought of | P2b's rewind-path audit precedes any sharing; exact-parity gate blocks merge |
| RAM watchdog kills builds/boots mid-phase on the shared 125 GiB box | standing watchdog + schedule heavy boots in incumbent gaps; operator decides |
| a boot maps past the amdgpu resident-system-memory limit: SVM refusal is NOT a catchable error - it wedged KFD and the IPMI watchdog hard-reset the box 2026-10-20 | one engine at a time (standing rule); P4.1 refuse-thresholds in every witness script (MemAvailable + engine-alive pre-flight); pinned arenas (`--arena-gib`) preferred over SVM paths |
| P5 relay assumed fast: MI50 Infinity Fabric may be absent on this workstation, making cross-stage copies ride PCIe (~25 GB/s) | P5.1 measures the real transport before any relay code; per-token hidden-vector copies are KBs, so even PCIe relay is small - but the claim is the measurement, not the assumption |

Effort summary: P0 ~1 week (mostly waiting on GPU/RAM windows), G8 ~0, P1 2-4 weeks, P3 3-6 weeks
(conditional), P2 1-2 months (conditional), P4a ~1-2 operator windows after the no-GPU prep,
P4b last and gated on P4a.
The honest current position (2026-10-22): P0 closed, G8 decided, P1 closed with exemptions, P2 and
P3 NO-GO with numbers - the open phases are P4a (GLM + DS4.1 first boots; no-GPU prep #18/#19 can
start any time, hardware steps #20-#23 are single-engine operator windows) and P5 (4-GPU layer
split for the ported engines, after P4a; its design step #25 and probe #26 can also start early).
