# gfx906 fork: start here

This is the entry point for the experimental Strata gfx906 (wave64) fork. It describes the public, checked-in
starting point. The exact build and limited GPU smoke evidence are indexed in [current state](CURRENT-STATE.md).

## Start with these pages

- [Current state](CURRENT-STATE.md): revisions, known evidence, and open validation.
- [Evidence index](EVIDENCE.md): pinned binary, historical throughput scope, and evidence boundaries.
- [Architecture](ARCHITECTURE.md): how the gfx906 compatibility path relates to upstream.
- [Known limitations](KNOWN-LIMITATIONS.md): explicit unsupported and unverified areas.
- [Fork scope decision](DECISIONS/0001-fork-scope.md): why this fork exists and what it will claim.
- [Upstream synchronization](UPSTREAM-SYNC.md): review upstream changes without duplicating a local patch.
- [Upstream gfx906 build notes](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source): inherited build instructions.

## Build entry point

The checked-in upstream build path is documented in `docs/AMD_HIP.md` under “gfx906 (Instinct MI50 / MI60,
Radeon VII): wave64, built from source.” It uses CMake with `-DSTRATA_HIP_GFX906=ON` and builds the `strata` target
with Ninja. Read that section for prerequisites and runtime device requirements. This is an inherited recipe, not a
rebuild result for the fork's present source revision.

The [container guide](CONTAINER.md) shows the archived digest-pinned runtime with external model mounts and
experimental AB/CD kept off by default.

The installer follows the same opt-in culture: `STRATA_EXPERIMENTAL_GFX906=1` lets `setup.py` accept gfx906
cards at its PC check (the same mechanism upstream uses for `STRATA_EXPERIMENTAL_GFX1103=1`); unset, setup
refuses gfx906 exactly as upstream does. Checked 2026-10-19 on this 4x MI50 box: unset refuses 4/4, set
accepts 4/4 (`tools/test_setup_amd.py`, 37 tests including a gfx906 case). This opens the mainline-model
install (pack, PLE shard, pinned MTP draft, run config). The engine remains the one this fork builds; setup's
ROCm-wheel fallback (`rocm.nightlies.amd.com/v2/gfx906/`, seen carrying 7.14.0a20260608 to ...20260612) is
reached only on machines without a system ROCm 7.

## Evidence boundary

The exact source base, archived native binary SHA-256, OCI index digest, and historical AB/CD throughput scope are
listed in [CURRENT-STATE.md](CURRENT-STATE.md) and [EVIDENCE.md](EVIDENCE.md). The archived figures are tied to that
binary and the public dual-MI50 32 GB / K8-V4 configuration; they do not validate the current source.

The short serial dual-MI50 GPU smoke with a cached continuation used a source-equivalent binary for pre-merge
main commit `64185a40962121b9adfc20e0e1a45b634cfeb472` (native SHA-256
`73a0f98105938850c749a3f0ed1dfe08e1753a97379aa9741c7fb73351c0137c`); see the [separate
receipt](../../benchmarks/gfx906/receipts/2026-10-16-current-main-gpu-smoke.json). It does not validate the
post-merge main at the 2026-10-17 checkpoint (audited checkpoint SHA
`73e01b3a5f1c577899e3f668e7e913759a1f7455`), for which no rebuild or GPU smoke is recorded. The archived AB/CD
opt-in delta is not integrated into current main, and token/state parity remains open. Use exact source and artifact
identities in any new report. Historical throughput measurements are not evidence for the present source.

### Ported engines: GLM-5.3-Flash and DeepSeek-V4.1-Flash (phase P4a, closed 2026-10-23)

Proven on this box, on one MI50 32 GB per session, ROCm 7.2.0, context 32768, engram tables present
and gated, so far: both ported engines boot and produce coherent tokens with experts on the GPU
([receipt](../../benchmarks/gfx906/receipts/2026-10-22-p4.2-p4.4-first-gpu-tokens.json)), DS4.1-Flash
answers an arithmetic question correctly through the OpenAI-compatible API with `finish_reason stop`
([receipt](../../benchmarks/gfx906/receipts/2026-10-22-p4.5-ds41-serve-pass.json)); GLM answers
through serve ([receipt](../../benchmarks/gfx906/receipts/2026-10-23-p4.3-vram-mechanism-solved.json)).
The fork's ctest set on this build is 51 tests with exactly the four pre-existing baseline
failures (split_lend, mmvq_il, verify_parity, ple_parity) against a baseline build of main at
e063ec6c.

Not proven, and not to be written as if it were: token equality with the authors' reference build
(never measured here), and state parity, which is not tested against the authors at all - no claim.
The decode rates are bring-up numbers on one MI50 of four (GLM 2.57 to 3.69 tok/s, DS4.1 15.50 tok/s,
serve adding no measurable overhead at batch 1) and are not throughput or cross-engine claims. No
draft models (DSpark off), no image input. The engines are single-device: four cards together is
phase P5 layer-split. GLM serve runs on the `--vram-margin 5.0` workaround, which costs 1.0 GiB of
expert slots; the proper fix is recorded as not shipped. Full log, with the two open items (why the
engine reported `length` at 55 of 64 tokens after a wrap-up prefill, and the serve stop-id gap
analysed in [P4.3-serve-stop-analysis.md](P4.3-serve-stop-analysis.md)):
[XOLOTL-PORT.md](XOLOTL-PORT.md).

### Layer split for the ported engines: phase P5 CLOSED 2026-10-26 (DS4.1 and GLM staged bitwise, both GO)

**P5 exit verdict** ([receipt](../../benchmarks/gfx906/receipts/2026-10-26-p5-exit.json)): **DS4.1** runs on two
and four MI50s with a greedy token stream **bitwise-equal** to its single-card run (64/64, 128/128, and 48/48
with the state column moving mid-prefill at 647 tokens), the staged boundaries costing 2.70 ms/pass warm; the
dense-spread decode gain measured +85%/+41% on two cards in both paired boots, while staged prefill runs 3-5x
SLOWER (limiter: 4.3x file-tier read amplification, not the relay) and the 4-card decode mean is not separable
from control at n=2. **GLM staged is also bitwise-GO**: it first looked to FAIL (staged diverging at token
16 while its control pair matched 64/64), but #40 closed the next day showing that was a WITNESS confound -
the P5.4 arms ran --pcie auto, whose share is an EMA of MEASURED per-path cost, so the staged shape's timing
difference converges the admission to a different cpu/pcie mix (and a CPU expert is not bit-equal to a GPU
one). With --pcie pinned, control / same-device relay / cross-card relay are EXACT 64/64 with identical path
counters (receipt 2026-10-27-p5.4c-glm-staged-parity.json). The rule: cross-topology parity witnesses pin
every timing-adaptive knob. Refused by design: `--serve`/DSpark on a staged run, Engram boundaries above layer 14,
and (arithmetically) more than ~26 state-column windows per stage until the per-stage graph pool grows or evicts.
The per-step ledger the rest of this section records stands as the path to that verdict.


The GLM engine now runs a STAGED pass opt-in (`--gpus 0,0`): the model is split into two stage
instances over their layer ranges, each owning its KDA/conv/MLA state, and the activation state at
each boundary is relayed through 8 MiB of pinned host memory (one synchronized copy out and one in,
64 KiB per token as F32 hc-wide state). Witness on this box: the staged run's 64 greedy token ids
are byte-identical to an unsplit run of the same prompt
([receipt](../../benchmarks/gfx906/receipts/2026-10-24-p5.3-glm-relay-handoff-pass.json),
logs/p5.3-*-20261024-102456). Without the flag nothing changes: the engine builds one instance and
runs as before.

Measured alongside, and it bounds what any parity claim on this engine can mean: two IDENTICAL
unsplit runs of this engine can diverge - that witness's control pair agreed on 39 of 64 tokens.
The path an expert takes (VRAM-resident, CPU arena, PCIe, file tier) changes float summation
order, and greedy argmax flips at knife-edge logits. Bit determinism across processes is not a
property this engine offers; the hand-off witness is therefore judged against the control pair's
own agreement, which the staged run matched (64/64 against one control, and parting from the other
exactly where the controls part from each other). Staged on one card costs about 0.78 GiB more
VRAM (per-stage state and graphs) and is a correctness mode, not a speed mode.

Not proven at this checkpoint: the DS4.1 engine under staging, staged serve mode (conversation
snapshot/restore is per-instance today), and any scaling number - the 1/2/4-card curve is P5.5 and
no throughput claim is made here.

#### DS4.1 staged run WITNESSED PASS (2026-10-25, P5.3b witness 20261025-190058)

The same capability now exists in the DeepSeek-V4.1 engine (`tools/v41`), built from the GLM pattern and
documented in [P5-RELAY-PATTERN.md section 6b](P5-RELAY-PATTERN.md).  What makes this engine different from GLM is
that **its caches are shared and a cache cannot cross a card**: a layer reads the compressed KV of the latest kv
source at or below it and the index keys of the latest index source at or below it, those rings are persistent GPU
state read by pointer inside one graph, and the relay moves only per-token state.  Hence **THE READER'S RULE** - a
boundary at `K` is legal only if **every layer at or above `K` finds both of its sources at or above `K`** (the stage
that runs a layer must run the layer whose ring it reads).  For the shipped 40-layer build (kv sources {2,8,14,20},
index sources {2,8,14,20,24,28,32,36}) the valid set is **{1, 2, 8, 14, 20}** - the layers that are both a kv and an
index source - and `--layer-split 20`, **the even halfway point of the model**, is one of them: stage 1 = [20,40)
runs the writer of ring 20 with every reader of it, the writers and readers of index rings 24/28/32/36, and the
candidate writer (20) with all of its readers (24/28/32/36).  **It is a set, not a floor.**  The first revision of
this code derived a floor of 37 ("last shared-cache layer + 1") by requiring the *writers* to sit below the
boundary; that direction is backwards, and 37 is a boundary that aborts at graph build - the tail [37,40) reads
layer 20's compressed KV and layer 36's index keys, both below it.  The cuts a GLM reader would guess (3/9/15/**21**)
stay refused, for the correct reason: `kv_src(21) == 20 < 21`, so ring 20 would remain on the stage below.
An out-of-set `--layer-split` **snaps down** to the nearest valid layer (21 boots as 20, and says so); a plan the
set cannot reach (an even 4-stage split, whose 4th boundary would sit past 20) is refused with the set named, not
silently run with fewer stages.
The payload is therefore **two** regions - `x_state` (80 KiB/token) and this model's explicit `pre_state` hc
pre-mix (16 B).  The third region the machinery implements, the 2048 candidate blocks layer 20 picks (8 KiB), is
**provably unreachable under the reader's rule** on any model whose caches are those two source lists: every reader
of that column is an index source, and validity at `K` already requires `cand_layer >= K` whenever a reader is left
at or above `K`, so producer and readers always share a stage.  The gate asserts that (`split_candidates_cross` is
false for every shape it can build) and the witness gates on the absence of a candidate region rather than its
presence.
Refused with a message, not a wrong answer: `--serve`, DSpark, a shared-weight stage on a foreign card, and a
`--layer-split` the geometry cannot reach.  One constraint the geometry does not carry: with `--engram` (layers 1 and
14 here) the **first** boundary must sit above the last Engram layer, because only stage 0 feeds the Engram token
ring - which leaves 20 as the only boundary this checkpoint can use while the Engram add is on.  The deeper legal
plan {8, 14, 20} (four cards) is a cache-legal split that the Engram rule refuses; it is reachable without
`--engram`.  The expert tier got device pinning
(`Ds41MoeConfig::device` + a `TierDev` guard at every CUDA seam); its seam is host-in/host-out so it
needs no cross-device transport.

Host-side verification on this box: full HIP build of `build-906`, `ctest -j4` at the fork's baseline
failure set, and the split gate (`tools/gfx906/gate-device-split.sh`, 31198 checks) which now includes
`tools/gfx906/test_split_geometry.cpp` - the reader's rule, the candidate rule, the payload size and the CLI plan gate
checked against the model's geometry with plain g++, no GPU.  Most of those checks are one invariant: for every 1-,
2- and 3-boundary cut of four geometries, the predicate must agree with a scan that **plays the runtime** (`split_
stages_own_their_reads` builds the stages and asks each of its layers whether the stage that runs it also runs the
layer whose ring it reads).  That scan is what a wrong direction cannot survive, and this code has had one.  The
gate's **third** step then cross-checks the source lists that copy against the engine's own reader
(`build-906/test_ds41_geometry <V41-GSQ.gguf>`, host-only): the full 40-pair `kv_src/idx_src` map must match
literally, and the valid set is re-derived **a third time, in awk, from the oracle's own map** - so a checkpoint that
moved a boundary cannot agree with a gate that did not.  Skipped, not failed, where no build or no checkpoint exists.
(A hand-rolled GGUF
header parse gave a different, plausible, **wrong** answer here - `cand L10`, `kv_src(l)==l`, "any split is valid" - and
it is the engine's reader that settles it, so the gate calls the reader rather than trusting a transcription.)  The
witness has since been run and PASSES: witness 20261025-190058, five runs, every staged variant (same-device
`--gpus 0,0` and both cross-device `--gpus 0,1` twins) byte-identical to an unsplit control for all 64 greedy token
ids, with the unsplit twins and the cross-device twins themselves agreeing 64/64 that session
([receipt](../../benchmarks/gfx906/receipts/2026-10-25-p5.3b-ds41-handoff-pass.json)).  Getting there took five
root causes, each found by hardware measurement and each given a host gate afterwards: the back-relay carried only
`x_state`/`pre_state`/`cand` and not the per-column run-scope carry state (1d18b7d2); the staged branch applied the
prune-only route bias while the decode-time bias is a residency lever (a6bfab87); the graph output cache was keyed
without the pass column (ce7e1116); a column transition did not carry the per-column carry state to the new window
(carry_forward, with d370bfe1); and `upload_pos_tables` wrote the window index table `i_widx` at byte `col0 * 4`
while the graph reads it at `col0 * SWA * 4` - a [SWA * NC] layout whose per-query stride of 128 hid under
`col0 == 0`, so every pass below the window agreed and the FIRST column-1 pass (position 64) attended rows the
table never held; that is why the failing witness matched its controls for exactly 51 tokens and left at the
column move (d370bfe1, with a byte-offset pass in `gate-relay-pos-tables.sh` that is red on the buggy stride in all
seven column-moving geometries it models, the witness's 14/64 among them, and clean on the fixed one).
One comparison discipline this witness established: the expert VRAM budget is derived from free VRAM and the shapes
start with different budgets (unsplit ~23.6 GiB, cross-device staged ~17.6 GiB on the card the budget reads), and
residency changes the tier path and with it the float summation order - at DEFAULT budgets a staged run is expected
to pick its own greedy tokens under exactly the knife-edge noise the GLM witness calibrated, and that is a
performance characteristic, not a state-seam property.  The witness therefore pins one identical ranked expert head
on every run (`--slots 1500`, controls included); with residency equalized the staged cross-device run matches its
control bitwise at the first decode row (supporting probe p5.3b-dumpdiff-20261025-184223: P row bitwise, midL19 x_state
0/20480 differ, 54/54 tokens), the only residual being sub-margin row diffs from the unsplit tier's prefetch timing.
A default-budget bit-parity claim is NOT made.  Its staged runs use `--layer-split 20` (the only boundary this
model's geometry and `--engram` both allow, a constant the gate keeps honest), gate that the stage really begins at
20, and FAIL if any staged run moves the candidate column across a boundary -
the reader's rule holding is what the witness documents.  Also unported, by decision: the layer-major prompt walk
(a staged run uses the sequential chunk path but keeps the batch's STATE columns as its cycling budget)
and staged serve/conversation reuse.

Step 2 landed the same day and is witnessed too: `--gpus 0,1` places the second stage - its layer
range, its KDA/conv/MLA state, and its own copy of its weights - on the SECOND card, and the expert
tier now pins the device it was given (it ran on the ambient device, always 0, because the dense
half was its only GPU user; a staged run is not). The expert seam needed no cross-device transport:
at that boundary the tier is host-in/host-out by design. Witness 20261024-123831:
[receipt](../../benchmarks/gfx906/receipts/2026-10-24-p5.3-cross-device-handoff-pass.json). A
cross-device staged run matched an unsplit control byte-identically for all 64 greedy tokens, the
cross-device twin pair agreed 39/64 - exactly the unsplit twin pair's agreement in the same
witness - and an earlier FAIL (61/64 against every reference, TS 20261024-110925) is thereby
settled as knife-edge noise: two identical cross-device runs do not even reproduce each other
bit-for-bit, so 61/64 is within this engine's own reproducibility and not a placement defect.

### Four cards, real context: P5.4 split smoke (2026-10-26, DS4.1 PASS, GLM defect filed)

[Receipt](../../benchmarks/gfx906/receipts/2026-10-26-p5.4-split-smoke.json), runner
`tools/gfx906/run-p5.4-split-smoke.sh`, three GPU sessions on the wrapper.

**DS4.1 across four cards is bitwise parity at real context.** `--gpus 0,1,2,3 --layer-split 8,14,20`
puts one stage on each card - `[0,8) [8,14) [14,20) [20,40)` - the legal four-stage plan the reader's rule
allows **only without `--engram`** (the Engram rule keeps the first boundary above layer 14). A 647-token
prompt runs 11 chunk passes and MOVES THE STATE COLUMN MID-PREFILL (columns 0..10, then decode inside
window 10); all three staged streams (two boots of the 4-stage arm) and the unsplit control produced the
SAME 48 greedy token ids, bitwise, and with the 14-token shape the same session pattern held at 128/128.
The P5.3b gate's column-moving PASS 3 - modeled, never booted - is now hardware-witnessed. Per-card VRAM
peaks (3.78 / 5.75 / 26.72 / 3.79 GiB) refute per-stage expert caches: the tier is ONE card's by design,
and spreading it is task #39, not this code.

**Two engine findings came out of the first boots.** (1) A staged run with no `--prefill-chunk` collapsed
its pass cap to `min(relay_chunk, max(1, 0)) = 1`, which gives every token its own state column and its own
cached graph - and a stage's ggml metadata pool (~205.7 MB, ~7.5 MB per 20-layer graph) dies at ~26 graphs
("needed 205986800, available 205676800", the pool's FULL size) after 14 prompt tokens and 12 decodes. Fixed
(9674c6d0): a staged run with no explicit chunk takes the relay unit and says so; the default path is
untouched. The arithmetic is a PRECONDITION for any long staged serve: distinct cached graphs per stage grow
with the column WINDOWS a run visits (~26 windows ≈ 1700 tokens at cap 64), so raising the pool or evicting
cold graphs must land before 4096-token staged serving. (2) A prompt of 521 REPEATED identical sentences
made both arms emit EOS as the first sampled token (the engine stops unconditionally at `G.eos`): long-prompt
witnesses need non-degenerate text; the runner now numbers the repetitions and asks for a long answer.

**GLM staged does NOT meet its calibrated bar and is a defect, not a refusal.** Its control twins matched
64/64 EXACTLY, while `--gpus 0,1` and `0,1,2,3` diverged from control at token 16 (staged twins 30/64 against
each other). Part of it is admission timing, which is the residency rule applied over TIME: the staged decode
loop issues no predicted prefetch (12.4% vs 20.6% VRAM hits), and which experts compute on GPU vs CPU changes
float summation order; a control with `--pf-b 0` moves the divergence 15 -> 26 but not to the bar. That also
rewrites witness #8's PASS honestly: it ran the same confound with no equalization, and its one 64/64 staged
match was knife-edge luck under a loose control pair. The residual is root-cause work on GLM's staged branch
(task #40, with the DS4.1 five-root-cause list as the checklist).

Numbers that are NOT claims: staged 4-card prefill of the 647-token prompt measured 6.7/4.6 tok/s against the
control's 34.1 - the staged MoE reads ~4x the file tier (8875 vs 2073 expert file reads) because it does not
reuse the owner's warm slots across stages, while the relay itself costs 2.93 ms/pass warm. Decode measured
+23% (session 2, 14-token prompt) and -10% (session 3, 647-token) against control on single greedy prompts.
A scaling curve needs repeated runs and is P5.5 (#31).

## Checked-in CLI

The tensor-split (model-parallel-by-shard) plan - distinct from P5's layer split, with its own
collective-latency kill gate before any engine surgery - is [P6-TENSOR-SPLIT.md](P6-TENSOR-SPLIT.md)
(tasks #35-#39).  Status as of 2026-10-26: P6.0 (transport probe: bounce-only, every GPU-direct
transport refused this board) and P6.1 (sharding-algebra host gate, 102 checks red-controlled) are
closed; the P6.2 same-device spike ran in four witnessed sessions and its kill gate fired - the
half-k mul_mat path costs +41% at 10 of 40 layers even in the minimal arm, so PREFILL TENSOR-TP is
NO-GO on this ggml/HIP path (receipts 2026-10-26-p6.2-tp-shard-same-device.json and
2026-10-26-p6.2-attribution-and-verdict.json; the seam itself is numerically certified, 128/128
identical tokens in every arm).  #38 (tensor-split-within-stage) inherits that NO-GO for its
prefill half; any decode-TP proposal must answer the w4 decode instability and cannot reuse any
same-device number as its benefit.

The checked-in gfx906 command entry point is [`tools/gfx906.py`](../../tools/gfx906.py). Run
`python tools/gfx906.py doctor` without a GPU. In a fresh Python environment, install the pinned schema-test
dependency with `python -m pip install -r tests/gfx906/requirements-schema.txt`, then run
`python tools/gfx906.py test-cpu`. The `build-gfx906` and
`bench-p2`/`bench-p3`/`bench-p4` commands print dry-run plans by default; their `--apply` modes require explicit
inputs. `bench-validate` checks the structure of a versioned evidence record, and `report` renders one. The HTTP
load commands do not establish native decode occupancy or a cached strict-overlap benchmark.
The [benchmark format guide](../../benchmarks/gfx906/README.md) explains the required raw-evidence review.
For a build dry run, provide the paths to your ROCm and llama.cpp installs, for example
`python tools/gfx906.py build-gfx906 --rocm-root /opt/rocm --ggml-dir /path/to/llama.cpp`.

## Conditional workflows

The checked-in [`.agents/skills/`](../../.agents/skills/) guides cover gfx906 builds, inference benchmarking, batching validation, and release
preparation. They apply only when their matching task is explicitly in scope. They do not install dependencies,
obtain hardware, publish releases, or add commands that are absent from the repository.
