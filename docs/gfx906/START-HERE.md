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

### Layer split for the ported engines: phase P5, step 1 hand-off proven (2026-10-24)

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

Not proven at this checkpoint: hand-off across DIFFERENT cards (the engine still refuses `--gpus
0,1`; the expert tier allocates its slots on one device and is the next work item), the DS4.1
engine under staging, staged serve mode (conversation snapshot/restore is per-instance today), and
any scaling number - the 1/2/4-card curve is P5.5 and no throughput claim is made here.

## Checked-in CLI

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
