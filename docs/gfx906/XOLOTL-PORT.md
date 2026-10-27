# Xolotl port assessment: DeepSeek-V4.1-Flash and GLM-5.3-Flash on this fork

**Status: assessment only. Nothing here has been built or run for these two models; every number is
Xolotl's or upstream's, measured on hardware this fork does not have. No port, correctness, or
performance claim for gfx906 is made or implied.**

## What was investigated, and with what identities

- Subject: <https://github.com/Indras-Mirror/Xolotl>, a fork of Strata that adds serving for
  DeepSeek-V4.1-Flash and GLM-5.3-Flash.
- Studied as a blobless clone at `/ai/engines/gfx906/xolotl-study`, HEAD `f7c561b7`
  ("ds41: --tier-batch-pcie flag", 2026-10-18), with a `local` remote pointing at this checkout.
- Merge base with this fork's `main` (`e063ec6c`): `296b59d2` (2026-10-14, "Merge f-1441-perf ...
  into f-1441"). Since that base: Xolotl +1,389 commits (875 files changed), this fork's `main`
  +130 commits (81 files).
- `include/strata/hip_compat` and the `STRATA_HIP_GFX906` build option exist **at the merge base
  itself** — they are the shared gfx906 lineage, not Xolotl's work. Xolotl does not contain any of
  this fork's 130 later commits.

## What Xolotl actually is

Neither model is a Strata `ModelFamily`. `include/strata/artifact/model_family.hpp` is untouched
(it still knows only `qwen4exp` and `qwen35moe`); the GGUF architecture guard, the geometry
(`core/geometry.hpp`) and layout tables (`core/family_layout.hpp`), and the `strata` binary
(`src/program/generate.cpp`) never mention deepseek or glm. Both models are **standalone binaries
under `tools/` that link Strata's core libraries plus a patched llama.cpp/ggml**:

| Model | Engine | LOC | How it runs |
| --- | --- | ---: | --- |
| DeepSeek-V4.1-Flash | `tools/v41/engine/ds41_*` (main at `ds41_generate.cpp:242`) | ~12,600 | hybrid: the dense/attention half is a ggml-CUDA graph builder (`ds41_dense.cpp`); the MoE experts run on Strata's expert cache/streaming tier (`strata_engine`, `strata_kernels`, `strata_kernels_cpu`, `strata_artifact`) with Strata kernels |
| GLM-5.3-Flash | `tools/glm/glm_*` (main at `glm_generate.cpp:183`) | ~2,600 | ggml dense + the **legacy** `tools/ds4` expert tier unchanged, plus a baked rank-1 abliteration LoRA (`lora_bake.cpp`) and a soft-prune expert list (`reap_prune.py` -> `prune-ezct-0.25.txt`) |

Supporting pieces:

- **`tools/xolotl/ggml/xolotl-ggml.patch`** (1,201 lines, 48 hunks, auto-applied to the vendored
  llama.cpp by `CMakeLists.txt:1331-1365`): new ops `GGML_OP_DSV41_FAKE_QUANT` and
  `GGML_OP_DSV41_HC_MIXES` (CPU+CUDA implementations), fp4 (e2m1/NVFP4/MXFP4) quant/dequant and
  KV-fp4 helpers, lightning-indexer fixes, an MMQ rework (Q2_K/Q3_K/MXFP4, deterministic path) that
  **also matters for Strata's own `strata_mmq`/prefill build** (`CMakeLists.txt:1492-1516`), and
  `GGML_CUDA_GRAPH_MAX_TOKENS` graph gating (fixes GLM's `cudaGraphInstantiate` OOM on fresh long
  prompts; decode keeps its graphs).
- **Shared-core changes** (`src/`, `include/`): 63 files, +8,856/-251 versus the merge base. New
  quant formats for the GSQ files including the **PTQ1_0 ternary (type 143)** CPU kernels
  (`src/kernels/cpu/ptq1_0.cpp`); `src/kernels/cuda/fork/*.cuh` headers split out of
  `iq_kernels.cu`/`native_mmvq.cu`; per-model geometry/guard headers (`ds4_geometry.hpp`,
  `ds4_geometry_test.cpp`); `iq_kernels.cu` +148, `moe_mmq.cu` +116, `pool.cpp`,
  `native_expert.cpp`, `expert_cache.cpp` (+23) changes.
- **Not part of the DS/GLM path:** the large new `src/` files `dense_model.cpp`,
  `dense_main.cpp`, `dense_kernels.cu` are a Qwen3.8-27B **dense** engine (`option STRATA_DENSE`).
  A DS/GLM port does not need them.
- **Serve:** each model gets a clone of the Strata server pattern:
  `tools/{v41,glm}/serve/serve_{ds41,glm}.py` + `strata-*.json` configs + `ds41-engine.sh` /
  `glm-engine.sh` wrappers (memory guard, GPU lock, env defaults). These bypass `setup.py`'s
  catalog and `serve/server.py`'s `FAMILY_MODEL_NAMES`.
- **DeepSeek-V4.1-Flash's geometry** (from `docs/ds4/GGUF_INVENTORY.txt`): 43 layers, n_embd 4096,
  MLA-style attention (`q_lora_rank`/`output_lora_rank` 1024, 1 KV head, key_length 512), YaRN 16x
  over 64K -> 1M context, 256 experts top-6 + 1 shared, per-layer `compress_ratios`
  (`[0,0,4,128,...]`), `hash_layer_count = 3`. This is genuinely new architecture for Strata, not a
  Flash-Next shape.

## Merge mechanics against this fork: the easy part

- **Collision surface is 5 files** (of their 875 and our 81 changed files): `CMakeLists.txt`,
  `README.md`, `serve/server.py`, `src/core/verify.cpp`, `src/program/generate.cpp`. Our verify /
  generate edits (+208/-83 / +79) overlap theirs (+2 / +93): mechanical conflicts, not
  architectural ones.
- Xolotl's own `MERGE_SCOPE.md` maps their internal unification (~60 files, conflict hunk counts
  per file; `src/kernels/cuda/iq_kernels.cu` worst at 24 hunks). Their `tools/ds4/cmake/fork.cmake`
  already knows about this fork: it excludes the MMVQ test under `STRATA_HIP_GFX906`. They are
  aware of the gfx906 target but build only CUDA.
- A feature-branch `git merge` beats cherry-picking 1,389 commits. The new engines arrive behind
  their own CMake options, so the default `strata` path stays byte-identical to this fork's last
  release — the contributing rule is satisfied by construction, and a follow-up trim commit can
  drop the bulk of their `bench/` and `tools/` ecosystem without losing git's conflict
  resolution.

## The real work: CUDA -> HIP/gfx906

- **Engine side is mild.** Grep over `tools/v41/engine`, `tools/glm`, and the dense engine: no
  `cp.async`, no cooperative launches, no `__nanosleep`, no `cudaMallocAsync`. CUDA graphs appear
  in 2 files (`ds41_moe.cpp` tier staging), one dynamic-shared-memory attribute in 1 file, one
  `wmma` in `ds41_moe.cpp` (Vega needs an MFMA or scalar path). ROCm has graphs on the modern
  stack; whether they behave on gfx906 with this staging pattern is **untested here**.
- **The hard ~80% is ggml.** `xolotl-ggml.patch` is CUDA-only; its file list (fattn, mmq,
  cpy-utils, the two new ops' CUDA paths) must be mirrored into ggml's HIP backend — the same job
  this ecosystem already does for llama.cpp on gfx906, now extended with the new ops and fp4 KV.
  And because Strata compiles ggml MMQ sources for `strata_mmq`, the patch's MMQ instance list has
  to stay consistent with this fork's build.
- **De-risking precedent (their evidence, not ours):** HANDOFF S6 records
  `vincalkr/Strata-DSV4.1`, a gfx1100 (RX 7900 XTX) port of this code (+13-23% with the fused ops;
  graphs-off +6.6% on HIP), and antirez's DwarfStar ROCm discussion
  (`docs/ds4/RESEARCH_2026-10-10_DWARFSTAR.md`). A maintainer without gfx906 hardware would welcome
  a receipt from this box (`docs/COMMUNITY_BENCHMARKS.md`).

## Hardware and artifact budget (plan before building)

- Boot inputs per their docs: a GSQ GGUF (**96-152 GiB**; DeepSeek `IndrasMirror/DeepSeek-V4.1-Flash-GSQ(-v2)`,
  GLM `GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn` 113.6 GB or the ~96 GB Maya-S-v2), **Engram tables
  ~203 GB** on SSD (`tools/v41/extract_engram.py`, `--engram a:b`), `rank.bin`, the DSpark draft
  sidecar (`V41-MTP-Vision.Q2_K.gguf`), an exported tokenizer dir, and the serve JSON/wrapper.
  So: ~350 GB NVMe and roughly >=128 GB RAM for the expert tier. This box's role split (2x MI50
  32 GB VRAM + RAM/SSD tiers) matches the design, and since the expert tier *is* Strata's
  `expert_source`/`expert_cache`, that part is home ground.
- **All their numbers come from one PC** — RTX 4090 24 GB, Ryzen 5700X, 90 GB DDR4, NVMe
  (`docs/ds4/V41_START_HERE.md`). Per this fork's rules none of them transfer as gfx906 claims;
  the port ships with its own receipts.
- **License check before download:** the DeepSeek GSQ repos are labelled "Uncensored" variants and
  the GLM files come from individual quant publishers; each model's own license governs.

## Their maturity caveats (from their own docs)

- "Experimental, single-developer, measured on one PC" (`docs/DEEPSEEK_V41.md` header).
- DeepSeek-V4.1: vision is not in the engine; a prefix-cache unpark bug is open; keep-cache never
  engages under `vram_lru`; long-context real-document decode is unproven; the 1017 quant is mid
  work.
- GLM: **no speculative decode at all** (KDA state cannot rewind); the runtime LoRA is "not
  implemented in the engine yet" (`docs/glm/PLAN.md:14`); the long-prompt graph OOM is worked
  around with `GGML_CUDA_GRAPH_MAX_TOKENS=64`, not fixed; the legacy `tools/ds4` "slice" engine is
  slated for deletion but GLM still links it.

## Validation gates for a merge candidate

1. CUDA build of the merged tree reproduces their per-engine check (`docs/xolotl/BENCHMARKS.md`
   claims DS4.1 logits bit-identical across their merge; `tools/v41/engine/compare_ds41.py`).
2. `ds4_geometry_test` and the parity harnesses pass (`src/kernels/ds4_expert_parity.cpp`, 417
   lines; `iq`/`mmvq` parity binaries with the fork headers).
3. This fork's gfx906 receipts and parity-contract fixtures for the untouched Flash-Next path still
   hold byte-identical (`benchmarks/gfx906/`).
4. After that, HIP/gfx906 port proceeds kernel-family by kernel-family, each gated on parity
   against the CUDA result before any perf number is written down; correctness, state parity, and
   performance stay separate claims.

## Target hardware: this server (measured on the box, 2026-10-18)

4x AMD Instinct MI50 32 GB (gfx906, `rocm-smi` product info, device 0x66a1 rev 0x02; the 32 GB
SKU is the dual-HBM2-stack Vega 20 at **~1 TB/s per card**, twice the 16 GB SKU's 512 GB/s), AMD
Threadripper Pro 3975WX (64 threads), 125 GiB DDR4-3200 across 8 channels (~205 GB/s theoretical),
Samsung 990 Pro 4 TB (the `/ai` LVM volume: 2.0 TiB, 697 GiB free at assessment time), ROCm
**7.2.0** active (`/opt/rocm` -> 7.2.0, 6.3.3 also installed), `hipcc` on PATH. This fork's gfx906
HIP receipts live in `benchmarks/gfx906/receipts/` (latest 2026-10-18).

Tier mapping the port inherits from Strata (no new machinery needed for Disk -> RAM -> VRAM):

| Tier | DeepSeek-V4.1-Flash | GLM-5.3-Flash | This box |
| --- | --- | --- | --- |
| SSD | GSQ GGUF 140-152 GiB read in place below full-RAM (file tier), **Engram tables ~203 GB** row-reads (`--engram a:b`), DSpark Q2_K sidecar | GSQ-RCO 113.6 GB or Maya-S-v2 ~96 GB GGUF | 990 Pro: ~650 GB for both models; **697 GiB free now** - free space or pick one model first |
| RAM | all experts pinned, or budgeted via `--resident-budget-gib` / their `--budget-gib --ternary-cap-at`; load-time PTQ1_0 -> Q2_0 transcode (`ds41_transcode.cu`) | same; soft-prune list shrinks the resident set | 125 GiB total; 205 GB/s is ~4x the bandwidth of the 4090 box that produced their numbers - the main CPU-expert lever |
| VRAM | dense/attention/mixers, KV, MTP, expert cache fills the rest | same | ~1 TB/s HBM2 per card, 128 GB aggregate - the GPU tier outranks the CPU tier 5:1 on bytes, which is the argument for eventually multi-device; but the engine is **single-device** (`CUDA::cudart`, device 0) -> phase 1 = one MI50 (32 GB), three wait |

Three constraints this box imposes, verified:

1. **RAM contention is live, not hypothetical.** `free` showed 117 of 125 GiB in use at
   assessment time (the box's serving workload). The resident-expert tier needs nearly all 128
   GB at full pin; plan to stop the incumbent server or boot these models in the budgeted file
   tier.
2. **Single-device engines.** Multi-GPU Strata (`--gpus`, `--peer-device`, `peer_experts.cpp` /
   `remote_experts.cpp`) exists on the Flash-Next path of this lineage but Xolotl's ds41/glm tier
   does not use it. Driving 4x MI50 for these models is a fork-side feature after the HIP port,
   not part of it.
3. **ROCm 7.2 on gfx906.** This fork already builds its wave64 target on this box (receipts), but
   Xolotl's patched ggml is built and patched for CUDA; the HIP mirror of `xolotl-ggml.patch` must
   be validated against the ROCm version this fork's receipts name, not assumed.

## Port log (this box, 2026-10-18)

Branch `gfx906/xolotl-merge` (worktree `/ai/engines/gfx906/worktrees/xolotl-merge`):

- `f366d550` merge of `xolotl/main` @ `f7c561b7` (conflicts: CMakeLists union, README fork banner;
  server.py / verify.cpp / generate.cpp auto-merged; their engine changes are flag-gated,
  default-off).
- `d43935cc` + `35cef805` keep it building on gfx906: `STRATA_DENSE` default ON->OFF, v41/glm
  CUDA targets gated on `NOT STRATA_HIP_GFX906`, `moe_mmq.cu` fallback inlined under HIP (its
  pre-merge body), new `hip_compat/cuda_profiler_api.h` shim.
- Gates, ROCm 7.2.0 / clang 22 / llama.cpp `3cf03257` (`-DSTRATA_HIP_GFX906=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=/ai/engines/gfx906/llama.cpp-3cf03257`):
  all 192 targets build; `strata 0.1.44.1` links; ctest matches a pre-merge `main` baseline built
  on this box (same 4 pre-existing host failures: split_lend_complement, mmvq_il_parity,
  verify_parity, ple_parity; qsa/kv_stream pass serially). Logs: `logs/build-906-*.log`.
- Notable: the legacy `ds4_moe_cuda`/`ds4_generate` tier now COMPILES under HIP (only the
  profiler shim was missing). What is still gated: `ds41_transcode.cu`, and the patched-ggml
  CUDA dense halves (`ggml-cuda` mirror of `xolotl-ggml.patch`) - the remaining port.

## Suggested order

1. Read-only conflict probe (below), then the feature-branch merge, CUDA-gated.
2. Port the **GLM path first** (2.6k LOC, legacy tier, no spec-decode to validate), then the 12.6k
   LOC ds41 engine; the ggml HIP patch grows with each.
3. Decide later whether `deepseek4` should become a proper `ModelFamily` (registry guard +
   `layer_tensors` table for MLA/DSA geometry) instead of sidecar binaries.

Read-only probe — run from this checkout, changes nothing:

```bash
cd /ai/engines/gfx906/strata-gfx906
git fetch https://github.com/Indras-Mirror/Xolotl main:refs/remotes/xolotl/main --filter=blob:none
git merge-tree --write-tree --name-only main xolotl/main | head -40
```

Expected (from this assessment's same measurement): conflicts confined to `CMakeLists.txt`,
`README.md`, `serve/server.py`, `src/core/verify.cpp`, `src/program/generate.cpp`.

## Port log - 2026-10-18 (evening, while model downloads ran)

- `xolotl-ggml.patch` **applies cleanly** to the pinned llama.cpp clone at
  `/ai/engines/gfx906/llama.cpp-xolotl` (19 files, +720/-65); the **CPU** build of the patched
  ggml is green (exit 0, `-j2`, 6m16s, log `/tmp/ggmlcpu.log`). The **HIP/gfx906** `ggml-cuda`
  build is the one open decisive gate - held back by RAM only (needs `available >= ~10 GiB`;
  downloads + the incumbent workload leave 4.8 GiB right now).
- CUDA residue re-triage (line-level, added lines only): ggml patch has no hard CUDA blockers for
  HIP - masked shuffles exist in HIP, the NVIDIA-cc probes in `lightning-indexer.cu` evaluate
  false on gfx906 and take the existing fallback path. Engine side: `ds41_moe.cpp` is the only
  file with graph calls (12) and they are **already mapped** by `hip_compat/strata_hip.h`
  (`cudaGraph*` -> `hipGraph*`, both `cudaGraphInstantiate` overloads; ROCm 7.2's
  `hipGraphInstantiate` keeps the legacy 5-arg form the CUDA-12 `(NULL,NULL,0)` calls use).
  Syntax gates: `ds41_moe.cpp` (host, HIP defines) **exit 0**; `ds41_transcode.cu`
  (`hipcc -x hip --offload-arch=gfx906 -fsyntax-only`) **exit 0**.
- No engine `.cu/.cuh` under `tools/v41|glm|ds4` contains `wmma` or raw `cudaGraph` - the earlier
  CUDA-feature tally maps to: ds41_moe.cpp (shimmed), the ggml patch's NVIDIA-guarded wmma
  section (falls back on gfx906), and `dense_kernels.cu` (STRATA_DENSE, now OFF).
- **Maya-L kernel verdict for gfx906**: gate/up IQ3_S(=21) and down IQ4_XS(=23) are both in the
  AMD instantiation lists of `src/kernels/cuda/iq_kernels.cu` (gate/up {16,17,18,21,22,23,29,42,
  143,11}; down {20,23,42,143,11}); `is_iq` admits both; gate==up so the fork's mixed-pair
  {Q2_K,Q3_K} restriction does not apply; the AVX2 CPU fallback (this box is Zen2, no AVX-512)
  covers both types. Maya-S-v2's down types (IQ2_S=22, IQ3_XXS=18) are **not** in the gfx906
  down list - on this box the already-downloaded Maya-L is the better-supported file.
- New files: `docs/gfx906/GLM-FIRST-BOOT.md` (T0-T5 + six named first-boot unknowns),
  `tools/glm/serve/strata-glm-mayal-gfx906.json` (draft box config),
  `tools/glm/serve/glm-engine-906.sh` (`STRATA_BUILD_DIR -> build-906`).

## ggml-CUDA-as-HIP: BUILT 2026-10-18 19:25 - the decisive gate is green

`libggml-cuda.a` = 74 MB / 143 objects, 144/144 TUs, 8m01s at `-j2`, ROCm 7.2.0 clang 22, `--offload-arch=gfx906`.
Built through Strata's `build-906` with `-DSTRATA_GGML_CUDA=ON -DSTRATA_GGML_DIR=/ai/engines/gfx906/llama.cpp-xolotl`
(clone `3cf03257f-dirty` with `tools/xolotl/ggml/xolotl-ggml.patch` applied; ggml 0.24.0). The ggml-CPU build of the
same patched tree was green earlier (exit 0, log `/tmp/ggmlcpu.log`).

Two fork changes, all gated on `STRATA_HIP_GFX906`, captured in `tools/gfx906/ggml/hip-cmake.patch` (apply:
`git apply -p1 tools/gfx906/ggml/hip-cmake.patch` inside the ggml clone):
1. **CMakeLists, 4 hunks**: skip `find_package(CUDAToolkit)` and `enable_language(CUDA)`; define `GGML_USE_HIP`
   (that, not `__HIPCC__`, is what selects `vendors/hip.h` in `common.cuh`) and `GGML_CUDA_NO_VMM` (gfx906 has no
   VMM; also keeps the `CUDA::cuda_driver` link off); assign every globbed `.cu` `LANGUAGE HIP` - without that
   CMake silently produced a **0-object archive** and still "succeeded"; link `CUDA::cudart`/`CUDA::cublas`
   (non-static), which map to hip::host / roc::hipblas via Strata's imported targets.
2. **vendors/hip.h**: the pinned `__shfl_sync/__shfl_up_sync/__shfl_xor_sync` maps demanded all four arguments
   while newer llama.cpp code calls the 3-arg form (`width` defaults in CUDA; `argsort.cu:301` failed); the maps
   are now `##__VA_ARGS__`-tolerant.

No `.cu/.cuh` source edits were needed: `vendors/hip.h` already maps cublas names and float intrinsics, and the
NVIDIA-cc-gated FP4 code evaluates false on gfx906 and falls back as designed. The fork's `strata_mmq` target had
already proven ggml `.cu -> HIP` compiles for this arch; the CMake-level traps (0-object archive, header selector,
variadic warp maps) are recorded in daily memory 2026-10-18.

Every build on this box while the incumbent holds ~120 GiB runs behind a `MemAvailable < 2 GiB -> kill` watchdog;
there is no swap.

**Still open before any GLM/DS4.1 claim**: un-gate the `v41`/`glm` targets for HIP (currently excluded by
`NOT STRATA_HIP_GFX906`), link the dense half against `ggml` + `ggml-cuda` + `ggml-base` as one resolution set
(the fork's existing pattern, root CMakeLists ~line 494), then runtime validation T0-T5 of
`docs/gfx906/GLM-FIRST-BOOT.md`. Compilation is a build gate, not a correctness or speed result.

## Port log - P4a EXIT 2026-10-23: both ported engines boot and answer on one MI50

Phase P4a (GLM-5.3-Flash and DeepSeek-V4.1-Flash, first boot and first answer on this box) is
closed. The correctness, state-parity, performance and boundary claims below stay separate, and
each names the receipt or log it came from.

### Build and merge trail (no runtime claim in this section)

- Merge at `f366d550`; gfx906 build fixes at `d43935cc` (`STRATA_DENSE` default ON->OFF, the
  `v41`/`glm` CUDA targets gated, `moe_mmq.cu` inline fallback under HIP); HIP shim at `35cef805`.
- `libggml-cuda.a` as HIP: 74 MB, 144 of 144 translation units, 8 m 01 s (section above). Both
  engines un-gated for HIP at `322599c3`: `glm_generate` 81 MB, `ds41_generate` 77 MB. Full
  incremental HIP sweep, 77 targets, exit 0.
- `ctest` on this build: 51 tests, and the failures are exactly the four pre-existing baseline
  ones (`split_lend`, `mmvq_il`, `verify_parity`, `ple_parity`). The comparison build is
  `/ai/engines/gfx906/worktrees/baseline-main` at `e063ec6c`.
- P4.0 / P4.1 (`1af4a425`, `0fb56187`, then `3671549e`): bootable wrappers for both engines, plus an
  engram table gate (`tools/gfx906/check_engram_tables.py`) that checks the safetensors header
  against what the engine expects instead of re-deriving the tables. The RAM/arena budget was
  pinned and the pinned tier came back positive: `arena: pinned (cudaHostRegister PORTABLE ok)` in
  both engines - GLM 60 GiB and DS4.1 64 GiB arenas fully registered
  (`benchmarks/gfx906/receipts/2026-10-22-p4-run2-pinned-ok-new-fails.json`, log
  `logs/p4.4-base-20261022-180447.log` line 10).

### Two faults that had to be root-caused to get a single token

1. ROCm 7.2 reports virtual-memory support that the driver then refuses. The switch that works is
   `GGML_HIP_NO_VMM`, not `GGML_CUDA_NO_VMM`, because `common.cuh:264` honours only the HIP-specific
   name under HIP (`benchmarks/gfx906/receipts/2026-10-22-p4.2-run1-vmm-fail.json`).
2. A heap-buffer-overflow WRITE of 1040 bytes, caught identically in both engines by an
   AddressSanitizer build, root-caused as an one-definition-rule layout mismatch on
   `ggml_backend_cuda_context`: 4336 bytes as `strata_mmq` sees it against 4400 as `ggml-cuda.cu`
   sees it. Fixed at `e13f1555` by adding the three missing defines to the `strata_mmq` compile in
   the live CMake branch, gated on `STRATA_HIP_GFX906` so the default CUDA-native artifacts stay
   byte-unchanged (`benchmarks/gfx906/receipts/2026-10-22-p4-odr-rootcause-fix.json`).

### CORRECTNESS - what is proven, and what is not

Proven on this hardware: both ported engines produce coherent tokens on gfx906 with experts on the
GPU (`benchmarks/gfx906/receipts/2026-10-22-p4.2-p4.4-first-gpu-tokens.json`, `9ce2a434`;
logs `logs/p4.2-base-20261022-180307.log`, `logs/p4.4-base-20261022-180447.log`), and DS4.1-Flash
answers an arithmetic question correctly through the OpenAI-compatible API - 17 times 23 as 391,
with finish_reason `stop` (receipt `2026-10-22-p4.5-ds41-serve-pass.json`, `fa4af2ba`). GLM
through serve answered `Hello there, friend!`
(`logs/p4.3-vram-serve-20261023-083129.log.curl`, `199100f7`).

Not proven: token equality with the authors' reference build. Never measured on this box, and no
parity harness was run against their logits. One serve rule worth keeping (recorded at `d27331de`):
the DS4.1 serve entry MUST be `tools/v41/serve/serve_ds41.py`, because that tokenizer dir's chat
template is a comment-only stub, so the plain `serve.server` renders an empty prompt and the engine
refuses with `ERR empty prompt` at `ds41_generate.cpp:1125`. `tools/glm/serve/serve_glm.py` is the
analogous authors' shim and the recommended production entry for GLM.

### STATE PARITY - no claim

Not tested against the authors' build in either direction. Session-state equality for these two
engines is open, exactly as token and state parity are open for the mainline path in the
fork's `docs/gfx906/KNOWN-LIMITATIONS.md`.

### PERFORMANCE - bring-up rates on one MI50, each with its artifact

- GLM decode 2.57 tok/s in the direct smoke (`logs/p4.2-base-20261022-180307.log`); 3.69 and 3.38
  tok/s in serve, so 2.57 to 3.69 across the witnesses
  (`benchmarks/gfx906/receipts/2026-10-23-p4.3-vram-mechanism-solved.json`).
- DS4.1 decode 15.50 tok/s in the direct smoke (`logs/p4.4-base-20261022-180447.log`) and 15.5 tok/s
  in serve (`benchmarks/gfx906/receipts/2026-10-22-p4.5-ds41-serve-pass.json`) - the same number, so
  the serve layer adds no measurable decode overhead at batch 1.
- No cross-engine and no cross-hardware comparison is claimed, and none of these rates says
  anything about the fork's mainline model.

### BOUNDARY of everything above

One MI50 32 GB per session - the ported engines are single-device, and all four cards together is
phase P5 layer-split. No draft models (DSpark off), no image input, engram tables present and
gated, context 32768 in these runs, ROCm 7.2.0, and the four pre-existing `ctest` failures carried
from baseline. The P4.3 serve number is a workaround, not the fix: `--vram-margin 5.0` costs 1.0
GiB of expert slots (19.89 down to 18.89) to survive the 2.57 GiB first-chunk-MMQ burst; the proper
fix, allocating the MMQ set at load, is recorded as not shipped
(`docs/gfx906/P4.3-serve-stop-analysis.md`, `docs/gfx906/P4-RAM-BUDGET.md`).

### Open items after P4a

1. Why the engine itself reported `length` at 55 of 64 tokens after the wrap-up prefill: the
   finish_reason half of the P4.3 analysis is unresolved. The server's stop list holding two Qwen
   literals instead of this model's ids 154820 / 154827 / 154829
   (`serve/server.py:3220-3221`) is the documented serve-side blemish. Neither is an engine defect.
2. Pre-allocate the MMQ scratch set with `c_stage` at load, so the first chunk-MMQ execution is not
   a surprise 2.57 GiB request (`tools/ds4/ds4_moe.cpp:2153`, `:2190-2206`).
3. Token and state parity against the authors' reference build.
4. Phase P5: driving four MI50 for these engines (layer split), per `docs/gfx906/P5-MULTI-GPU-DESIGN.md`.

## Port log - P5 EXIT 2026-10-26 (corrected 2026-10-27): DS4.1 and GLM both run on multiple MI50s bitwise-equal to their single-card runs

This is the fork-side feature the top of this file predicted: after the HIP port made these engines run on
gfx906 at all, phase P5 gave them a layer split across the four MI50s. The full phase ledger is
`docs/CONCURRENCY-PLAN.md` P5; the exit verdict and the three claim classes are in the plan's **P5 EXIT
VERDICT** paragraph and `benchmarks/gfx906/receipts/2026-10-26-p5-exit.json`. What it took, in order, each
with its artifact:

* **Device plumbing (#27, opt-in `--gpus`, default path byte-identical)** and a build gate red-controlled
  against the pre-fix header (`gate-device-split.sh`, 105 checks).
* **Interconnect measured before relay code (#26)**: host-pinned bounce chosen over peer-direct at 25-26.7
  GiB/s, uniform across PCI complexes (witness 20261022-055705). That choice is why a staged run moves its
  boundary state through host memory and why the cross-device engines need `--relay-copy-weights`.
* **The relay itself (#28)**, proven same-device first (`--gpus 0,0`), then across two cards. Getting DS4.1
  to bitwise parity took five root causes, each found by hardware measurement and each given a host gate
  afterwards; they are listed with their commits in `docs/gfx906/START-HERE.md` (P5.3b). The last one - the
  window index table `i_widx` uploaded at `col0 * 4` while the graph reads it at `col0 * SWA * 4` - is the
  one that hid under column 0 and left at the first column move, which is why every witness now runs a
  prompt long enough to move a column.
* **Correctness witnesses (2026-10-25/26/27):** DS4.1 staged == unsplit at 64/64 (2 cards), 128/128 (4 cards),
  and 48/48 with a mid-prefill column move at 647 tokens. GLM staged first appeared to FAIL its calibrated bar
  (control pair 64/64 exact, staged diverging at token 16) - but that was a witness confound, not a defect
  (#40 closed 2026-10-27, see the P5.4c entry below): with the timing-adaptive `--pcie auto` pinned to a fixed
  share, GLM control / same-device relay / cross-card relay are all EXACT 64/64 with identical path counters.
* **Performance (measured, not claimed):** decode faster on two cards in both paired boots (+85%, +41%) from
  DENSE spread alone; prefill 3-5x NEGATIVE under staging, limiter named as 4.3x file-tier read amplification
  (8875 vs 2073 expert file reads), not the relay (2.70 ms/pass warm). The full table, and the honesty rule
  that within-arm boot variance bounds every stronger claim, are in `2026-10-26-p5.5-split-scaling.json`.

Two rules this phase forced on all future work here, because each was paid for: pin expert residency
(`--slots`) and boot order in any parity or perf comparison - free-VRAM-derived budgets and first-boot-after-
cache-eviction both move a control more than the design change under test; and never make a bit-parity claim
at default VRAM budgets across engine shapes. A third rule came from #40: pin every timing-adaptive admission
knob (`--pcie`, not just `--slots`) in any CROSS-TOPOLOGY parity comparison - `--pcie auto` sets the PCIe share
from an EMA of measured per-path cost, so a staged run's different wall time (by design) converges the
admission to a different cpu/pcie mix, and a CPU expert is not bit-equal to a GPU one. Auto is fine for
control-vs-control (same shape converges the same); it can never be used to compare a staged run to its control.

## Port log - P5.4c 2026-10-27 (#40 closed): the GLM staged "divergence" was a witness confound, not a relay bug

P5 EXIT had recorded GLM staged as NO-GO because its staged arms diverged from control at token 16 while the
unsplit twins matched 64/64. The `run-p5.4c-glm-pcie.sh` discriminator pinned every timing-adaptive knob
(`--pcie 0.25 --pf-b 0 --slots 1500` on all arms, the exact 14-token P5.4 prompt) and re-ran the three shapes.
All three - control, `--gpus 0,0` (same-device relay), and `--gpus 0,1` (cross-card) - came out EXACT 64/64,
with bit-IDENTICAL tier counters (hit 12.4%, cpu 75.9%, pcie 11.7%, file tier 8668 on every arm), so the
admission is now provably on the same path in all three, not just inferred from matching tokens.

What had gone wrong in P5.4: the GLM arms ran `--pcie auto`, which is FreeToken-style bandwidth balance - the
PCIe share is an EMA of the MEASURED cost of one expert (`tools/ds4/ds4_moe.cpp:337`). A staged run costs
~2.7 ms/pass more, so auto settled on a different cpu/pcie split (28.5% vs 23.2% witnessed), and a CPU-computed
expert is not bit-identical to a GPU-computed one, so a knife-edge argmax flipped. `--pf-b 0` alone moved the
divergence 15 -> 26 but could not remove it while auto was still timing-dependent. GLM and DS4.1 both default
`--pcie` to a fixed value (0.25 and 0.75); only the P5.4 witness had opted into auto, and the DS4.1 arms that
went 128/128 bitwise never passed `--pcie` at all - which is why they were comparable. The relay algebra was
sound all along; this is the same conclusion the DS4.1 five-root-cause work reached, now confirmed for GLM's copy.
Receipt `2026-10-27-p5.4c-glm-staged-parity.json`.
