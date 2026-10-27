# P5 multi-GPU design for the ported engines (P5.0, task #25) — 2026-10-22

The change-list that turns `glm_generate` / `ds41_generate` from single-device engines into
opt-in layer-split engines. Design only — no code changes here. Vocabulary and failure
wisdom are reused from the native engine's shipped layer-split (`docs/MULTI_GPU.md`,
`src/core/expert_source.cpp`, `src/core/peer_experts.cpp`), because that path has already
been debugged on real hardware. Correctness/performance claims are separated throughout
(AGENTS.md): everything measured here is *topology and bytes*; nothing is *runtime*.

## 0. Device inventory — measured on THIS box 2026-10-22 (read-only, vLLM live)

| GPU | BDF | root complex | link |
| --- | --- | --- | --- |
| 0 | 0000:23:00.0 | pci0000:20 (bridge 20:01.1) | Gen4 x16, running at max |
| 1 | 0000:26:00.0 | pci0000:20 (bridge 20:03.1) | Gen4 x16, running at max |
| 2 | 0000:43:00.0 | pci0000:40 (bridge 40:01.1) | Gen4 x16, running at max |
| 3 | 0000:46:00.0 | pci0000:40 (bridge 40:03.1) | Gen4 x16, running at max |

`rocm-smi --showtopo`: every GPU pair weight **40**, hops **2**, link type **PCIE** — there is
**no Infinity Fabric between the GPUs on this workstation** (the P5 risk-register assumption,
CONFIRMED). Same-complex pairs (0↔1, 2↔3) are still 2 hops; they at least share a complex.
All four are NUMA node 0 (single-socket TR Pro 3975WX). P5.1 measured the interconnect
(2026-10-22, engines down, receipt `2026-10-22-p5.1-interconnect-probe.json`): every pair is
P2P-capable (`EnablePeerAccess` ok on all 12 ordered pairs), **peer-direct is fast only inside
a complex** (~26.9 GiB/s for 0↔1 and 2↔3 in HIP order) and **collapses to ~7.3 GiB/s across
complexes**, while **host-pinned staging is uniform at 25–26.7 GiB/s on every pair** — so the
bounce relay is not the fallback, it is the *best* transport, and it is the one #28 ships.
Peer-direct cross-complex is 3.6× slower than bouncing; peer mappings stay available for a
later KV phase but buy nothing for the relay.

**Enumeration order differs by API — map ordinals to BDFs, never trust the number.** sysfs
lists BDF order (23, 26, 43, 46); HIP and ggml enumerate **43, 46, 23, 26** — so in the engine
own order, devices 0,1 are the pci0000:40 complex and 2,3 are pci0000:20, and the natural chain
0→1→2→3 crosses complexes exactly once (1→2). With a uniform bounce transport, stage placement
is free.

Hidden-vector geometry (read from the artifacts 2026-10-22): GLM-5.3 `embedding_length =
4096` (GGUF u32 KV at shard-1 byte 369), DS4.1 `hidden_size = 5120` (embedded config JSON),
DS4.1 `n_routed_experts 384 / top-6 / moe_intermediate 2304`, 40 layers (both counts from
that JSON).

## 1. Backend creation + weight placement — the single chokepoint, then 62 touchpoints

- **Chokepoint (one line per engine)**: `glm_generate.cpp:208` and `ds41_generate.cpp:282`
  call `ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)` — first GPU wins,
  there is no device selection anywhere in the ported code. Verified: `cudaSetDevice` appears
  in NO ported file (only in the native `src/core/*` and `tools/h2d_bench.cu`).
  Per stage, create backends via `ggml_backend_dev_by_index(GGML_BACKEND_DEVICE_TYPE_GPU, k)`
  + `ggml_backend_init_by_device` instead. The device ordering must be pinned to the BDF
  table above (ggml enumerates by its own order — the design must map ordinal→BDF, not
  trust "GPU 0").
- `im.backend` touchpoints after creation: **36** in `tools/glm/glm_dense.cpp` (lines
  787-800, 883-1012, 1033-1073, 1168-1267) and **26** in `tools/v41/engine/ds41_dense.cpp`
  (318, 975-1162, 1430-1506, 1667-1715). These are tensor alloc, graph alloc, memcpy —
  they all follow whichever backend their layer belongs to, so the plumbing change is
  *per-layer routing of `im` to the owning stage's backend*, not 62 semantic changes.
- Weight placement sites: `ggml_backend_alloc_ctx_tensors_from_buft` at `glm_dense.cpp:210`
  region and `ds41_dense.cpp:318` — this is where a layer's tensors must go to the STAGE
  backend (split by layer index at the boundary vectors `--layer-split` gives).

## 2. Raw CUDA/HIP calls in the expert tiers (device-0 implicit)

- `tools/ds4/ds4_moe.cpp` (GLM's tier): **180** `cuda*` hits; `tools/v41/engine/ds41_moe.cpp`:
  **286**. All run on whatever device is current for the thread — today always device 0
  because nobody calls `cudaSetDevice`. Under HIP, `cuda*` resolves through
  `include/strata/platform/hip_compat/strata_hip.h` macros to `hip*`.
- Every raw call site must execute under the stage's device. The native engine's pattern is
  the RAII `On(dev)` guard (`src/core/peer_experts.cpp:40`: save current, set, restore) —
  reuse it verbatim at the tier entry points rather than sprinkling `set_device` per call.
- **The context trap, quoted** (`ds41_moe.cpp:401-403`): *"Created once and never destroyed:
  a binary that also links ggml-cuda resolves ~ggml_backend_cuda_context to ggml-cuda's real
  one, which segfaults on the context strata_mmq built."* With N stages there are N
  ggml-cuda contexts; the strata-mmq context (chunked prefill, `--chunk-mmq`) must be
  created once per device and bound per call, or chunk-mmq stays disabled under a split
  (opt-in default already keeps it off in the P4.4 config — the split design can start with
  `--chunk-mmq` refused when `--split-device` has >1 entry, and lift that later with a
  measurement).

## 3. Graph capture (per stage)

- Real GPU-graph calls exist only in `ds41_moe.cpp` (**12** `cudaGraph*` hits) and inside
  vendored `libggml-cuda.a` (`GGML_CUDA_GRAPH_MAX_TOKENS`, used in the P4 failure ladder);
  `ds4_moe.cpp` has zero. GLM's `GLM_GRAPH_REUSE` (`glm_dense.cpp:34-38`) is ggml's
  CPU-side cgraph reuse, not device graphs — no split interaction.
- Rule: capture must run with the stage's device current (same `On(dev)` guard). Graphs are
  per-context — a graph captured on device A cannot launch on device B; nothing in the
  design shares them across stages.

## 4. Pinned host registrations (PER-DEVICE — the subtle one)

- `ds41_dense.cpp:1135/1143/1144`: `ggml_backend_cuda_register_host_buffer` for `host_fn`,
  `host_ids`, `host_wts`. This maps a host pointer **into one device context** — it is
  currently called once, against the single device. Each stage that reads those buffers
  needs its own call, and the call must happen while that stage's backend exists.
- `PinnedArena` itself (`ds4_moe.cpp:953`, `ds41_moe.cpp:1487-88`) registers ONCE
  (`cudaHostRegister`, Portable) — that registration is process-global; per-device access
  maps come from the `register_host_buffer`-style calls above. Do not `cudaHostRegister`
  the arena once per device: it is one arena, N device mappings (P4.1's driver-budget
  accounting charges it once — keep it that way; `docs/gfx906/P4-RAM-BUDGET.md`).
- RAM budget consequence (feeds #27): with 4 stages the ONE arena is split into 4
  per-stage regions — the total stays ≤ ~110 GiB on this 125 GiB box (dense slices +
  scratch take the rest); the per-stage floors from P4.1 apply to the SUM, not per stage
  (one process, one MemAvailable check).

## 5. KV + recurrent state placement — what P4b must not hardcode

- Today all per-session structures ride `im.backend` (GLM: MLA KV + KDA state inside the 36
  GLM touchpoints; DS4.1: comp/idxk caches inside the 26). Under a split, a session's KV for
  layer L lives on the stage owning L: per-session structures must carry
  `(cache, stream, device)` per layer exactly like the native engine's
  `locate(layer) -> (cache, stream, device; -1 = CUDA0)` (`expert_source.cpp`;
  `generate.cpp:284-285` comments).
- **The failure the native engine already paid for, quoted** (`expert_source.cpp` comment):
  *"Reading a later stage's slot number out of CUDA0's cache copied another expert's bytes
  into RAM — the resident mode with a split then computed some experts from the wrong weights
  after the first adaptive swaps (garbage output)."* P4b's per-session cache entries MUST
  carry a device field from day one and never assume 0; a session touching layers on two
  stages is normal and every copy must name the owner's stream.
- Relay payload at a stage boundary per decode token: GLM one hidden vector = 4096×2 B =
  8 KiB; DS4.1 = 5120×2 B = 10 KiB (× hyper-connection stream count if `hc` keeps parallel
  streams across the boundary — check at #28). Prefill relays are chunk×hidden (e.g. 512×
  10 KiB = 5 MiB). Measured #26: every transport moves ≥7.3 GiB/s and the chosen host-pinned
  bounce 25–26.7 GiB/s uniformly, so a decode-token relay is ~0.4 µs against the ~50 ms/token
  tier-bound budget; **sync copies first**
  (one stream-sync per boundary), pipeline only if #28 measures
  a cost — the native engine ships sync relay for the same reason.

## 6. Flag surface (native vocabulary, byte-identical default)

- Port the native flags' semantics 1:1 (`docs/MULTI_GPU.md:38-39,99-100`;
  `src/program/generate.cpp:2130-2131`): `--layer-split K1[,K2..]|auto` and
  `--split-device D1[,D2..]` (stage devices, default: the next visible ones). The native
  `--gpus` stays a serve-layer concept that maps to these.
- **Bit-exact hand-off check reused from native**: `--layer-split K --split-device 0,0` runs
  both stages on ONE card sharing everything — "the bit-exact check of the hand-off, not a
  speed mode" (MULTI_GPU.md:99-100). That is P5's G8-equivalent gate before any multi-card
  boot, and it needs zero extra cards.
- No flag ⇒ single backend, single device: the exact code path shipped today. Every split
  behavior is opt-in (AGENTS.md rule).

## 7. Per-stage capacity table (measured file bytes, computed split; runtime hot-fraction NOT claimed)

GLM-5.3 Maya-L: routed experts **148.50 GB** = 138.3 GiB, dense 7.81 GB, nextn 0.03
(GGUF breakdown, 2026-10-18). DS4.1: routed **154.7 GB**, non-expert 8.25.

| per stage (÷4) | GLM | DS4.1 |
| --- | --- | --- |
| expert file bytes | 34.6 GiB | 36.0 GiB |
| per-stage arena (4 × 27.5 = 110 GiB sum) | 27.5 GiB ≈ 80% of its file bytes | 27.5 GiB ≈ 76% |
| dense slice (layers partition with the stage) | ~1.5 GiB + shared/head once | ~1.9 GiB + non-expert split |
| KV+state at smoke ctx, per stage (÷4 of P4.1 numbers) | ≤ 0.1 GiB each | ≤ 0.01 GiB |
| VRAM expert slots | (32 − stage dense − KV − margin) GiB per card, 4× the single-card cache | same shape |

The payoff claim stays an ESTIMATE until the ported engines run a route census: single-card
arena covers ~20% of GLM's expert bytes, 4 stages × per-stage arena+cache plausibly 65-75%
of *traffic*, but GLM/DS4.1 route distributions are NOT the native engine's (P0.4 census
numbers do not transfer). `glm_generate --profile routes.bin` + the P0.3 route-census
instrumentation are the measurement tools (#29).

## 8. Task mapping

#25 (this doc) → #27 device plumbing (+26 in parallel: P2P/bandwidth + peer-access probe).
#28 relay implementation is gated on #26's numbers (sync-first rule above). #29 smoke:
`--split-device 0,0` bit-exact, then 2-card, then 4-card with the census. #30/#31 exit:
token-identical greedy vs single-card (same seed) + measured 1/2/4 scaling curve —
sublinear-with-numbers closes P5 the way P1.2 closed (the P1.2 pattern).

## 9. Measured scaling (2026-10-26, DS4.1, [receipt](../../benchmarks/gfx906/receipts/2026-10-26-p5.5-split-scaling.json))

Six interleaved boots in one session (ctl, 2c, 4c, ctl, 2c, 4c), 647-token prompt + 64 greedy decode tokens,
slots pinned both sides. No predicted curve existed in this doc to sit beside, so the measurement stands alone
and names its own limiter.

| arm | prompt tok/s (boot1, boot2) | decode tok/s (boot1, boot2) | expert path (per run) |
|---|---|---|---|
| 1 card  | 38.5, 26.2 | 7.72, 6.07 | 14946 hit / 64 cpu / 110 pcie / 27 file (98.8%) |
| 2 cards | 12.7, 11.4 | 14.28, 8.57 | 14528 hit / 169 cpu / 423 pcie / 27 file (96.1%) |
| 4 cards | 7.6, 4.7   | 12.90, 5.86 | same as 2-card, per the engine's own counters |

What the numbers support and what they refuse:

* **Decode is faster on two cards in BOTH paired boots (+85%, +41%).** The gain is entirely from spreading the
  DENSE compute - the expert path counts are identical across arms because the tier is one card's by design;
  per-stage expert caches are task #39's prize, not this code's.
* **Prefill scales strongly negative: -3x at two cards, -5x at four, same sign both boots.** The limiter is
  named and it is not the relay (2.70 ms/pass warm): staging amplifies the file-tier reads 4.3x (8875 vs 2073
  expert reads) because the per-stage tier queries cycle the shared VRAM slots worse, and the file path is the
  ~6 GB/s one.
* **Within-arm boot variance (control 7.72 vs 6.07; 4-card 12.90 vs 5.86) is as large as the between-arm
  differences.** The 4-card decode mean is NOT distinguishable from control at n=2; a defensible curve needs
  n>=4 interleaved boots, and this is recorded as the honest close (the P1.2 pattern) rather than averaged into
  a claim.
* The serve round-trip this phase asked for is refused by design (a staged engine rejects `--serve`);
  GLM scaling is blocked by its staged parity defect (#40). Both are boundaries of the checkpoint, not gaps to
  average over.
