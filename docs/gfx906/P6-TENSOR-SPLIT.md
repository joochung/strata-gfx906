# P6 - tensor split (model parallelism by shard, not by layer)

Status: **PLAN ONLY.** Nothing in this file is implemented or measured on hardware except the
numbers it cites, each of which carries the receipt it came from. P5 answers "can a staged,
layer-split run be exact?" (yes - witnessed 2026-10-25). P6 asks a different question: can a
LAYER be split across cards, so that one GEMM's work is shared instead of the model's depth.
This plan's first phase is a kill gate, exactly like P2/P3's: it predicts tensor-parallel decode
is net-negative on this board with the measured transports, and the fastest honest way to spend
the next GPU slot is to try to refute that prediction cheaply, before any engine surgery.

## 1. What the machine offers (measured, not assumed)

From the P5.1 interconnect probe (receipt 2026-10-22-p5.1-interconnect-probe.json):

- 4x MI50 32G on PCIe Gen4 x16, no downclock, **no Infinity Fabric** (peer never beats a
  host-pinned bounce: same-complex peer 27 GiB/s ~= staged 25-26.7; cross-complex peer 7.3 GiB/s,
  i.e. 3.6x SLOWER than bouncing through host-pinned memory).
- Peer access maps on all 12 pairs, so a future phase CAN do device->device flags/spin barriers;
  bandwidth is not the problem, small-message LATENCY and sync cost are.
- A full engine-grade cross-device hand-off of 80 KiB (one way, host-pinned, with sync) costs
  **0.66-0.69 ms per pass** as measured in the P5.3b witness (run4/run5 relay lines, TS
  20261025-190058). That is one boundary per pass - tensor parallel needs TWO per LAYER.
- **RCCL is installed** (`/opt/rocm/lib/librccl.so.1.0.702`), and the build does NOT currently
  find or link any collective library (configure says "Could NOT find NCCL"). A collective
  transport is available to evaluate; it is not yet evaluated, and gfx906+PCIe RCCL behaviour is
  exactly the kind of thing that must be measured here rather than quoted from elsewhere.

P5.1 measured 128 MiB bursts. A tensor-parallel step moves **10-1280 KiB collectives hundreds of
times per second**. No measurement of that regime exists on this box yet; P6.0 produces it.

## 2. The arithmetic that makes decode the hard case

DS4.1 (engine reader, not a transcription): `D 5120 layers 40 heads 64 d_head 512 rope 64
q_lora 1280 o 8x1024 swa 128 experts 384/6 ff 2304 hc 4 idx 32x128` (test_ds41_geometry, the
same oracle P5.3b's gate calls).

A 2-way tensor split of a layer needs one reduction per residual connection. The engine runs a
hyper-connection machine (`hc 4`), but the mixers consume D-wide rows, so the reduction points
are the same two per layer as a plain transformer: after the attention output projection, after
the FFN/down projection. Payload per reduction at batch n: `n x D x 4` bytes (fp32 partials,
summed once on either card).

| regime | reductions/token-pass | payload at n=1 | at n=64 |
|---|---|---|---|
| 40 layers x 2 | 80 per token, 80 per prefill pass | 1.6 MB/token | 102 MB/pass |

Payload sizes above are F32 (the engine's native activation dtype: `x_state` is F32 hc-wide and
the hc machine consumes F32 - section 3). **The reduction dtype is a per-run option, not a
constant**: F32 default, F16 and Q8_0 as opt-in modes (see 2b). They barely move the decode kill
gate (latency-dominated) but they matter wherever the wire stops being negligible: 4-way ring
hops (the payload crosses 3 links), large-batch prefill (1.28 MB F32 per hop at n=64), and any
cross-complex peer-direct leg (7.3 GiB/s: the F32 n=64 payload is 175 us of pure transfer per
hop; Q8_0 cuts that to ~44 us).

Two transport models, both anchored to measurements above:

- **host-pinned bounce, engine-grade sync (0.33-0.69 ms/hop):** decode at n=1 pays
  80 x ~0.33 ms = **~26-55 ms/token of pure communication** against a whole-token budget of
  57.7 ms with MTP or 74.6 ms without (receipt 2026-10-25-dspark-mtp-perf). Decode TP is
  dead on arrival with this transport. The same bounce at n=64 costs (80 x (0.33 ms + 51 us
  transfer))/64 = **~0.46 ms/token against ~25 ms/token of prefill compute** (39.2 tok/s
  single-card) - under 4% overhead, where the GEMM itself halves per card. **Prefill TP is
  plausibly a ~1.8x lever with the transport this repo already has.**
- **RCCL-class collective (to be measured):** IF decode reductions land in the 50-150 us range,
  80 reductions cost 4-12 ms/token - 7-20% of the decode step for a ~2x cut in dense-GEMM
  time, and TP decode becomes a live question with real expert-residency benefits on top (below).
  If reductions land at bounce-latency values, decode TP is NO-GO and the plan stops at prefill.

**The P6.0 kill gate is therefore: measure small-collective latency (10/20/40/320/1280 KiB,
2-way and 4-way, RCCL allreduce vs pinned-bounce barrier vs peer-direct spin-flag with a fused
fp32 add) on THIS box, and only decode-TP survives if the reduction p50 stays below ~150 us at
20 KiB.** Prefill-TP may proceed on the bounce transport even if decode dies; the two exit
criteria are separate.

### 2b. Reduction dtype: F32 default, F16 and Q8_0 opt-in

A reduction carries PARTIAL SUMS - activation-space values - so the checkpoint's weight quants are
not candidates; the axis is the activation dtype of the partials. Three modes, one flag
(`--tp-reduce f32|f16|q8`), default `f32` so the default path stays byte-identical to the plan's
baseline arithmetic:

- **f32 (default):** numerically free - the reduce is a plain fp32 add and the run's parity result
  attributes to SHARDING, not to the reduction. The correctness baseline every other mode is
  measured against.
- **f16:** halves the payload (20 -> 10 KiB/token, 1.28 -> 0.64 MB per n=64 hop). Each partial is
  rounded at ~5e-4 relative (11-bit mantissa) BEFORE the sum, at 80 sites per token. Industry-
  standard TP compromise - and the reason it is opt-in HERE is this engine's measured knife edge:
  state diffs of ~1e-1 move tokens on this model (P5.3b probes), so a systematic ~1e-3 per-site
  perturbation compounding through 40 layers must be calibrated, not assumed away.
- **q8 (Q8_0 partials):** blocks of 32 values + fp16 scale -> ~1.06 B/value, payload ~6 KiB/token,
  384 KiB per n=64 hop. Per-element error is `block_max / 254` - relative to the block's largest
  value, so small elements in a wide block are the noisy ones; partial-sum blocks (GEMM outputs,
  near-Gaussian) are the benign case for this format. Each card quantizes ITS partial; the fused
  add kernel dequantizes both to fp32 and sums, so the error is two quantization roundings plus an
  exact add. The fused dequant-add replaces the same kernel slot as the f32 add - no extra hop,
  no extra sync; the whole saving is wire bytes.

Rules binding the opt-ins (both are new features, so the repo's rule applies twice over): each
dtype mode is a SEPARATE measured configuration - it must pass the calibrated twin-pair parity
bar at equal residency with its own receipt, at BOTH n=1 decode and n=64 prefill shapes, before
any doc or serve config may recommend it; and a mode that passes parity may still be NO-GO on
speed if its wire-time saving is inside the latency floor (P6.0's size table settles which regimes
even have wire time to save: at 20 KiB on the bounce path, 0.8 us of transfer hides under a
330-690 us hop, so f16/q8 buy nothing for decode there and must not be sold on it).

## 3. What this model's geometry actually shards (and what it does not)

This is DS4.1-specific and the reason the plan starts at `tools/v41`, not at a generic TP
framework:

- **Attention is MLA-shaped, so there is no clean 64-head slice to hand out.** Heads are
  UP-PROJECTED from a 1280-wide compressed `q_lora`, and the KV side is a compressed latent the
  rings already store. The shardable seam is the up/down projections: column-shard `q_b` by head
  groups, row-shard `attn_output_b` along the SAME groups, and replicate the compressed latent
  read - every card reads the same kv latent rows (they are tiny relative to full-KV) and
  computes its 512/8-head subset's flash attention. The latent read is a gather from the owner's
  ring; it must arrive on each card per pass (one extra copy per pass, sized by SWA/compressed
  windows, not per layer) - P5.3b's `upload_pos_tables` machinery and its byte-offset discipline
  (the i_widx lesson: view and upload offsets must be modeled in BYTES over the full store, and
  the gate must simulate the runtime's own arithmetic, not restate the theory) are the pattern
  this copy inherits.
- **The output projection is low-rank in 8 groups (`o 8x1024`)** - 2-way and 4-way splits land
  exactly on group boundaries (8 = 4+4, 2+2+2+2). No straddled group, no fixup.
- **The indexer (`idx 32x128`, top-512 selection) is replicated, not sharded** - it scores whole
  keys; 32x128 is small, replicating it is cheaper than synchronizing it. Same decision as
  Engram: `i_eng` rows are host-uploaded per pass and shape-independent; each card computes its
  own engram add from the same uploaded rows (verify: engram table reads are file/VRAM reads of
  identical content; the P5.3b engram-row dumps were bitwise equal across shapes).
- **MoE FFN: do NOT tensor-shard the routed experts - the tier is the axis that wants splitting.**
  The expert tier already supports device pinning (`Ds41MoeConfig::device` + TierDev guards,
  P5.3 step 2) and is host-in/host-out, so an EXPERT-PARALLEL split (384 experts striped across
  cards, each card's ~9.5k-expert slice cached in its OWN VRAM) is a tier change, not a graph
  change. 2-card EP roughly doubles expert VRAM (the 2616-slot budget measured post-prompt,
  witness #13, doubles toward ~5200 per card at unchanged per-card VRAM) - which raises the VRAM
  hit rate (98.4-98.9% at decode now, worse at prefill: control prefill was 288000 PCIe loads and
  0 hits in the 1200-token DSpark-perf run) and cuts per-card arena/file traffic. The shared
  expert (ff 2304) is a plain small FFN: row/col-split like a standard MLP, or just replicate it.
- **The hyper-connection machine stays per-card and FULL-width.** hap/post_f/comb_f/shexp rows
  are D-wide state; the carry, carry_forward, and back-relay code P5.3b built all index them by
  COLUMN. Under TP every card needs the SAME post-mix row - so the allreduce output must be
  written into each card's hc pipeline before the mix, and the mix itself runs redundantly per
  card (it is ~4xD elementwise, negligible vs the GEMMs). The per-column state machinery does NOT
  get sharded; only the weights and the reduction points do. This keeps P5.3b's relay/staged work
  orthogonal: P6 shards a LAYER, P5 stages DEPTH; they compose as tensor-split-within-stage.

## 4. What "correct" will mean (no bit-parity claim, and residency is still the confounder)

- Splitting a matmul across cards and summing fp32 partials CHANGES SUMMATION ORDER. A TP run is
  the same model, not a bit-identical one - the same ruling as G8(b) for batching
  (decision 09eaef93). The bar is the calibrated one: **token agreement against controls must be
  at least the twin-pair agreement at equal residency**, measured in the same witness.
- **THE RESIDENCY RULE BINDS P6 HARDER THAN P5 DID**: a TP run frees ~half the dense VRAM per
  card, which changes the expert budget, which changes argmax knife-edge outcomes. Any P6 parity
  or performance comparison MUST equalize expert residency with `--slots` on every run it
  compares (memory: DEFAULT-BUDGET dumps are not comparable across shapes), and any P6 SPEED
  number must additionally be reported at default budgets with the freed-VRAM-into-expert-slots
  effect reported as part of the result, not hidden by the pin.
- Gates first, always: a red-controlled host model of the sharding algebra (which rows/cols of
  `q_b`/`attn_output_b`/down-projection each card owns, the o_groups boundary divisibility, the
  replicate sets, the partial-sum fp32 order) that a wrong direction cannot survive - the
  pattern that killed three ring theories and that PASS 3 of gate-relay-pos-tables.sh proves
  works when the gate simulates BYTE arithmetic instead of restating a theory.

## 5. Phases, each with its own kill gate

> Status 2026-10-26: P6.0, P6.1 closed (sections 7, 8). P6.2 witnessed and its gate FIRED - prefill
> tensor-TP is NO-GO (section 9, receipts 2026-10-26-p6.2-tp-shard-same-device.json and
> 2026-10-26-p6.2-attribution-and-verdict.json). P6.3 as scoped here is closed on that gate; the live
> axis that remains is P6.4 (expert-parallel tier caching, task #39).

- **P6.0 - collective probe (kills or feeds everything).** A `probe-collective` tool (pattern:
  probe-p2p.cu, refuses any live engine, no model, minutes not hours): allreduce latency and
  bandwidth at 10/20/40/320/1280 KiB, 2- and 4-way, three transports (RCCL linked directly;
  host-pinned bounce with a spin barrier; peer-direct copy + spin-flag + fused fp32 add kernel,
  same-complex AND cross-complex pairs). The size grid doubles as the dtype grid (section 2b): the
  F16 payload is the 10 KiB point, the Q8_0 payload ~6 KiB (probe 6 and 384 KiB too), so the
  receipt can say WHERE each dtype has wire time worth saving; the fused dequant-add variants
  (f16, q8) are measured as kernels standalone (add cost alone, without a collective) so their
  compute cost is on the table too. Exit: a receipt with the latency table, a chosen
  transport for prefill-TP, and a GO/NO-GO for decode-TP against the ~150 us bar in section 2.
  If NO-GO, decode stays single-card-per-token and P6 becomes a prefill-throughput project plus
  the EP axis (which needs no collective at all).
- **P6.1 - sharding algebra, host-only gate.** The section-3 partition written as a runnable
  model: per-card owned slices, replicated reads, partial-sum order, checked against DS4.1's
  read geometry; a wrong boundary (straddling an `o` group or a head-group) must turn the gate
  red, and the gate must re-derive the divisibility from the engine reader's own numbers, not a
  transcription. The gate also models the THREE reduction dtypes (section 2b): it computes the
  fp32 two-shard sum exactly, and each opt-in as (round/quantize partial -> dequant -> fp32 add),
  printing the induced error per site class so the f16/q8 calibration runs have a host-number to
  sanity-check the hardware result against - the same discipline as the relay byte-offset gate:
  simulate the arithmetic, don't restate the theory. No ggml, no GPU.
- **P6.2 - same-device shard spike (plumbing without a second card).** Split one layer's q_b/o
  into TWO ggml subgraphs with two sliced weight copies ON DEVICE 0 and an explicit fp32 reduce
  node between them; prove: (a) graph-build cache behaviour with sliced weights (the graph-cache
  column-key bug ce7e1116 is the cautionary precedent), (b) the reduce node composes with the hc
  machine and the per-column state (carry_forward still sees one D-wide row), (c) numerics land
  within the calibration rule, (d) overhead of the extra launch/reduce at n=1 and n=64. Kill
  gate: if the same-device two-shard overhead is already >10% at n=64, cross-card prefill TP
  cannot reach its predicted ~1.8x and P6.3 is refused.
- **P6.3 - two-card prefill TP.** Real second card, transport per P6.0. Scope: attention
  projections + shared expert sharded; rings/tier/engram/indexer stay owner-card (rings per
  section 3's per-pass latent copy). Measure against a same-day, same-session layer-split run and
  an unsplit control: prefill tok/s at 1200/4096-token prompts (the 39.2 tok/s single-card number
  is the reference), decode tok/s reported unchanged or explicitly refused-to-run, expert hit
  rates at DEFAULT and pinned budgets. Reduction dtype: `f32` first as the parity/speed baseline;
  `f16` and `q8` then as SEPARATE measured modes, each with its own twin-pair calibration at equal
  residency at both n=1 and n=64, each allowed to be NO-GO (section 2b's rules). This phase may
  ALSO test decode at 2 cards if P6.0 came
  back GO; if P6.0 said NO-GO it measures prefill only and says so.
- **P6.4 - expert parallel (the axis this engine is actually built for) and the 4-card
  question.** EP as a tier change only: expert slice per card by device pinning (exists), arena
  and rank.bin seeding per slice, route-to-remote-expert dispatch through the existing
  host-in/host-out seam (this is where the real work is: a routed expert on the wrong card
  currently just costs a PCIe read; EP makes remote fetch structural). Exit: the P5.5-shaped
  scaling curve, honest either way - including the result "the tier at 4 cards is 4 disjoint
  caches and the hit rate is what wins, allreduce was never needed".

## 6. Risks this plan cannot retire alone

- RCCL on gfx906/PCIe with wave64 builds is unproven here; P6.0 is its first measurement. If it
  refuses or underperforms, transports fall back to bounce/peer models P5.1 already bounded.
- The SVM one-engine-per-host crash rule is respected: TP stays ONE process on N devices (the
  staged witness already ran one process on two cards cleanly); never two engine processes.
- DSpark + staged is refused; DSpark + TP is a THIRD combination nobody has thought about -
  P6.3 must re-audit the draft-verify pass against TP reductions before enabling it, and will
  default to refusing MTP under TP until it has its own measurement.
- Serve-layer work (snapshot/restore for TP sessions) inherits the open staged-serve gap; P6
  does not close it and does not claim to.
- The cross-device slot-budget bug (budget read from the wrong card, ~6 GiB, P5.3b receipt
  limits) MUST be fixed before P6.3's default-budget numbers mean anything - a TP run's whole
  point is the freed VRAM, and the budget computation is currently shape-shifty.

## 7. P6.0 measured (2026-10-25): what this board delivers for small collectives

Witness logs: logs/p6.0-probe-20261025-214735.log (kern+bounce, complete), plus refused
transport attempts in p6.0-probe-20261025-{211652,222006,225139,230428,232407}.log. Tool:
tools/gfx906/probe-collective.cu (host build in the file header), runner run-p6.0-probe.sh.
Everything below is a measurement of the collective ALONE (plus its per-iteration host barrier
in the bounce case); engine-grade graph-segment overhead per seam is NOT in these numbers.

p50 microseconds, 300 iters at <= 40 KiB, 60 above:

| size KiB | bounce 2-way | bounce 4-way | note                       |
|---------:|-------------:|-------------:|:---------------------------|
|        4 |         41.1 |         88.0 | decode attn seam (f16 wire)|
|        6 |         39.0 |         86.9 |                            |
|       10 |         39.6 |         87.4 | decode seam f16            |
|       20 |         33.5 |         89.0 | decode seam f32 (D=5120)   |
|       40 |         35.4 |         89.0 |                            |
|      320 |         65.5 |        133.8 | prefill n=16               |
|     1280 |        176.6 |        359.3 | prefill chunk n=64         |

Sanity (bitwise sum-equal to host reference) OK on every bounce row. Single-device fused-add
kernel cost for reference: 2.6 us at 20 KiB, 9.6 us at 1280 KiB (dev-local, no wire).

Verdict against the section-2 kill gate: at decode seam sizes the measured host-synchronous
bounce is 33-41 us (2-way) and 87-89 us (4-way), both FAR below the 150 us bar. Multiplied to
the token (40 layers x 2 seams = 80 reductions): 2-way ~2.7-3.3 ms against the 57.7 ms/token
budget = 4.6-5.7 percent; 4-way ~7.1 ms = 12.2 percent. THE COLLECTIVE ITSELF DOES NOT KILL
DECODE TP. The plan's earlier bounce-based NO-GO used 0.33-0.69 ms per hop - but that figure
is a P5.3 ENGINE RELAY hop (full graph-segment completion + sync around the copy), not the
collective. Which one a P6.2 seam pays depends on whether the TP reduction rides INSIDE a
running graph segment or forces one to finish; that is exactly what P6.2 must measure, and
it, not this table, is where decode-TP lives or dies.

Transport refusals on this board (all witnessed, each a separate run, all with the four cards
idle and the engine stack stopped):
- Kernel-initiated peer stores (XFB): first collective never completed (run 211652, 900 s
  timeout; re-attempted protocol, same). 
- Copy-engine DMA to peer + local spin: hangs at first iteration (runs 222006, 232407) even
  after the two real protocol bugs found en route were fixed - legacy-default-stream implicit
  cross-device sync deadlock, and streams created on the wrong device context. 
- RCCL single process (ncclCommInitAll): init succeeds, first allreduce SEGFAULTS host-side
  on both default and SHM transports, in one-thread-per-rank AND sequential-enqueue shapes
  (225139, 230428).
- RCCL forked one-process-per-rank: children fail inside RCCL's own pinned host alloc
  ("host alloc 4 bytes", out of memory) on idle cards with MemAvailable ~100 GiB and
  RLIMIT_MEMLOCK 16 GiB (232407).
Consequence for the plan: the transport menu collapses to the host-pinned bounce - the same
transport P5.3's witnessed relay already uses. Every P6 numeric assumption must be computed
from the bounce table above, and no P6 phase may assume peer-direct or RCCL latency on this
box. Reduction dtypes: at decode seam sizes the bounce is latency-dominated (4 KiB 41.1 us vs
20 KiB 33.5 us - spread larger than the payload delta), so f16/q8 wire savings buy ~nothing
at decode; they only matter at prefill payload sizes.

## 8. P6.1 host gate (2026-10-26): the section-3 partition as runnable arithmetic

tools/gfx906/p6_tp_algebra.cpp + gate-p6.1-tp-algebra.sh (plain g++, no ggml, no GPU).
The ownership model is DATA (per-card ranges per tensor) and every invariant plays the
runtime's check over it; the numerics consume the SAME plan the invariants certified.
Clean: PASS, 102 checks. Red-controls, each alone: flip-qb-direction RED (units mismatch
named + induced reduce error 89 caught by the stability bars), straddle-ogroup RED (overlap
+ 4096 uncovered elements), drop-replicate RED (card-1 kv-latent read set), dtype-rounding
RED (truncation-vs-RNE probe). Usable TP widths re-derived from O_GROUPS=8: {1,2,4}; widths
{3,5,...} refuse. Induced per-site error vs fp64 reference (host numbers the f16/q8 hardware
calibration must sanity-check against): f32 max 1.1e-4-2.2e-4; f16 max 2.3e-2-7.0e-2;
q8 max 0.41-1.11 (attn site / shared-FFN site, 2-way and 4-way).

## 9. P6.2 same-device shard spike: the seam in the engine (2026-10-26, witnessed - prefill TP NO-GO)

`--tp-shards 2 [--tp-layers 0,1,...]` on ds41_generate (opt-in; default tp_shards=1 leaves the graph path
unchanged, AGENTS.md). One device, no transport: the seam exactly as section 3 and the P6.1 gate certify it -
q column-sharded by HEAD GROUP (heads [0,32)/[32,64) as a 4D view of the roped q, the same strides the default
branch's reshape+permute produces), per-shard flash-attn over the SAME K/V and mask (the replicated read),
per-shard sliced output-LoRA weights, and ONE ggml_add (f32) as the reduce node - the allreduce stand-in.
The sliced weights are real copies made at load, next to the STRATA_PRED_ROUTER copies and uploaded the same
way (extend convert_upload with an explicit source row stride): q_b rows [0, NH/2*DH) / [NH/2*DH, NH*DH),
output_a batches [0, OG/2) / [OG/2, OG), output_b the first/second HALF of each row's reduction axis
(block-aligned at 4096 for every quantized type; the upload dequantizes row-halves and requantizes). ~60 MB
of sliced copies per sharded layer. A staged run (--gpus) REFUSES the pair: the spike is same-device, and
tensor-split-within-stage is P6.3, after this kill gate.

Measured in code review, before any hardware: three offset/layout bugs the host algebra cannot see -
the q head-slice must use the HEAD stride (nb[1]), not the token stride; the flash-attn's 4D query must be a
view_4d (a ggml_reshape over a strided 3D slice asserts); and the reduction-half upload must carry the FILE's
full row stride (4352 B at f16 for output_b), not the sliced row's. Each one silently changes values or
crashes the build, and each survives a green compile.

The witness (tools/gfx906/run-p6.2-tp-shard.sh, launched through the GPU wrapper): same binary, same prompt,
same --slots pin BOTH runs (the residency rule), control default vs tp 10-of-40 layers; parses the engine's
own prompt/decode tok/s lines; kills on a wrong sliced-copy count so a flag that did not reach the graph
cannot be measured twice and called a no-op. Kill gate as planned: >10% overhead at n=64 with only the seam
plumbing on one device = cross-card prefill TP NO-GO.

Witnessed 2026-10-26 in four sessions (receipts 2026-10-26-p6.2-tp-shard-same-device.json and
2026-10-26-p6.2-attribution-and-verdict.json). Numerics: exit criteria (a)-(c) PASS outright - 128/128
identical greedy tokens, every arm, every session, through 8 chunked-prefill passes and 128 decodes; the
f32 reduce node composes with the hc machine and the per-column carry with zero knife-edge flips.
Performance: the kill gate fires with room to spare, and the attribution arms say where the cost lives.
The flash-attn split costs nothing (tp_nofa, one full-head FA but the same sliced GEMMs and add, matches
tp within noise). tp_b - full FA, full output_a, ONLY attn_output_b halved into two mul_mats plus the one
add - still costs +41% at 10 of 40 layers, so the half-k mul_mat path itself, not the seam plumbing, is
where prefill TP dies: halving the GEMM does not halve its time on this ggml/HIP q8 path, and a cross-card
split would run exactly that half-k op per card plus a bounce allreduce on top. PREFILL TENSOR-TP: NO-GO.
Decode side: the seam is free when stable (tp_nofa/tp_b decode +0.4% or less, both sessions they were
measured), but the FA-split tp arm decoded at 5.90 tok/s in w4 against 13.81 in w2 - same binary, same
args, MoE tier counters identical - an unexplained instability of that shape and a warning for any
decode-TP wiring that reuses it. Two hygiene facts the witnesses exposed, both now enforced by the runner:
a first boot after another tenant evicted the GGUF page cache runs 2.4x slow (throwaway warmup boot added),
and arm absolute rates are only comparable within a session.

The spike stays on the branch, opt-in and default-path byte-identical: if a future kernel path makes
half-k mul_mats pay their byte-bound share, w4 is the measurement to re-take with ARMS=tp,tp_nofa,tp_b.
