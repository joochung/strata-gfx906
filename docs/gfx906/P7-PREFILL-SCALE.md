# P7 - making DS4.1-Flash prefill faster, and what four cards can (and cannot) buy

Question (operator, 2026-10-29): DS4.1-Flash runs at ~40 tok/s - how could TP4 improve it?
Status: INVESTIGATION CLOSED - anatomy measured, every surviving option priced. Nothing here is
implemented; the two GPU experiments (EXP1/EXP2) ran 2026-10-29 in `run-with-free-gpus.sh` windows
(runner `tools/gfx906/run-p7-prefill-anatomy.sh`, logs `logs/p7-anatomy-20261029-002015.*` and
`logs/p7-anatomy-20261029-004434.*`, window logs `logs/p7-exp1-anatomy.log` / `logs/p7-exp2-anatomy.log`).

## 1. What the 40 tok/s is

The 39.2-40 tok/s reference is the ONE-SHOT prefill of a 1200-token real-document prompt on one MI50
(the dspark-perf control, and reproduced in EXP1: 38.1-39.0 tok/s, split lines agreeing to 0.5 s):

    prompt: 1200 tokens in 30.75 s = 39.0 tok/s (experts 0 hit / 0 cpu / 288000 pcie / 5171 file)
    prompt split: dense 7.5 s, experts 23.1 s (file wait 7.7 s), logits 0.0 s

Same shape at 512 tokens (p6.2 control): dense 3.2 s + experts 15.5 s (file 6.2 s) = 27.2 tok/s.
Experts are 75-82% of prefill wall in every witness since P4.2. This box's serve numbers differ
again: the serve batch path measured here is 34-37 tok/s for a 1200 prompt (the authors'
"~16.8 s + n/250" crossover line is THEIR 4090 box - V41_PROGRESS records 219-227 tok/s there).

## 2. EXP1/EXP2 anatomy (this box, never measured before - MIMO_CHUNK_PROF had never printed)

The `MIMO_CHUNK_PROF=1` split of the run_chunk MMQ machinery over the whole 1200-token prompt:

    chunk prof: 40 calls, 75 launches: wall 1790 ms = waits on reads 497 + wave/copy waits 0
                + final sync 310 + other host 983; GPU grouped kernels 507 ms (resident 0);
                file tier 54.0 GB read in 7739 ms of reader time = 6.98 GB/s
    chunk prof MMQ: transcode 18 ms over 23 sub-runs, 15.7 GB of ternary = ~880 GB/s

Arms, 2 rounds each in ONE session (means over paired rounds; within-arm spread is this box's usual
+/-30%, the P5.5 lesson):

| arm (1200-tok prompt) | tok/s mean | dense s | experts s | file wait s | grouped kern ms |
|---|---|---|---|---|---|
| chunk-64  (today's deferred cache) | 35.5 | 7.4-9.6 | 23.6-26.7 | 7.7 | 507 |
| chunk-256 | 30.6 | 4.8-9.6 | 24.1-41.6 | 7.8 | 508 |
| chunk-1024 | 36.8 | 4.7-10.9 | 21.7-31.1 | 7.9 | 507 |
| batch-2048 | 35.6 | 4.5 | 27.5-30.8 | 7.5 | 510 |
| eager-ctl (`--prefill-chunk 1`, VRAM cache SEEDED before the prompt) | 11.2 | 69.5-79.1 | 25.2-41.8 | 7.7 | 509 |
| eager-t4 (same + `--tier-cards 4`) | 10.2 | 61.6-65.0 | 30.4-96.1 | 15.2 | 450 |

Five facts that decide the question:

1. **GPU compute is 0.5 s of a 31 s prompt.** The grouped expert kernels - the thing tensor-TP
   splits - are 1.6% of prefill wall. There is no GEMM time to win.
2. **Prefill runs with NO VRAM expert cache, by design.** `defer_vram = prefill_chunk > 1 ||
   prompt_batch > 1` (ds41_generate.cpp:1131) opens the VRAM slots only AFTER the prompt ("VRAM
   after the prompt: ... 1500 experts in VRAM (0.7 s)"), so every prefill lookup is `pcie` (arena ->
   card): 288000/288000, `resident 0`. Even the eager arm (cache seeded) showed 0 prompt hits - the
   prompt phase does not read the cache either. Prefill's expert cost is STREAMING, full stop.
3. **NVMe is 7.7 s of every prompt and is one disk.** 54.0 GB read at 6.98 GB/s warm (5171 blobs;
   46% arena residency: 7119 of 15360 experts). Cards do not add disks - eager-t4 made this WORSE
   (15.2 s: four arena seeds contending the same NVMe, file reads 5168 -> 9733).
4. **Chunk size does not move the rate** (64/256/1024 all 31-37 tok/s, variance-dominated): the
   staged MMQ classes are already kept across passes (`c_pin_tok kept across prompts`), so the
   "upload dedup per bigger pass" model is wrong - the ~20 s expert remainder is the per-lookup
   arena->card streaming the chunk path itself does NOT instrument (fd_tier - prof_wall - kern).
5. **The per-token pass path is 3x slower** (eager arms: dense 60-79 s at m=1), and fanning THAT
   out across shards (eager-t4, the P6.4 seam at token cadence) buys nothing: 10.2 vs 11.2 tok/s,
   exactly P6.4's serialized-submission tax, with dense (61 s) never leaving shard 0.

## 3. What TP4 means once the anatomy is known

Tensor parallel in the GEMM-splitting sense - what P6 planned and P6.2 tested - is dead for prefill
twice over: P6.2 measured the half-k `mul_mat` path +41% at 10/40 layers, and now we know the GEMMs
are 0.5 s of 31 s, so even a perfect 4-way split saves ~0.4 s. Decode TP is separately dead (P6.0
bounce latency, P6.4/#46: -25% at token cadence). Do not spend another window on GEMM shapes.

What four cards ACTUALLY offer this workload, priced against the facts above:

**Option A - expert-parallel chunk fan-out (the P6.4 seam, extended to `run_chunk`).**
Fan the chunk/batch tier call out by expert OWNERSHIP (`e % 4`), each shard staging and multiplying
ITS quarter of each chunk's entries through ITS OWN PCIe link and holding ITS OWN resident slots -
the same aggregated-VRAM prize P6.4b won on GLM (21.8 -> 78.7% hit, +66% decode) applied to DS4.1's
PREFILL phase, where today's hit rate is literally 0. No half-k GEMMs (fact 1 makes them pointless,
so P6.2's killer never applies), no RCCL (reductions at chunk cadence: 80 bounce hops/pass =
13-55 ms against a 1.6-26 s pass - under the latency floor, not over it).
- What it attacks: the ~20 s uninstrumented streaming remainder (fact 2/4) split over 4 links
  (25 GiB/s each, one is saturated today) and 4x residency; the 7.7 s NVMe floor STAYS (shared disk).
- Price: 31 s -> dense 7.5 + file 7.7 + stream ~20/3.5 ~= 21 s => **~1.5x (55 tok/s)** conservative,
  ~2x if the resident share also collapses the stream. Decode unchanged (P6.4 stands).
- Engine work (the #46 caveat, now the plan): `run_chunk` takes dense top_k rows per token, REFUSES
  ids < 0 and has no `n_ent` (ds41_moe.cpp:3961, 4875/4888); the fan-out needs an ownership-skip in
  the class-building (the shard `exclude` mask already exists) + partial-output fan-in (f32 add, the
  `shard_run` fixed-order pattern) + per-shard `c_*` chunk buffers. The serve batch path then gets
  it free; the #46 prompt-hit%% caveat disappears.
- Gate before building: the same-boot witness `chunk-ctl` vs `chunk-t4-fanout` must beat 1.25x on
  the 1200 AND 4096 shapes (the P5.5 variance rule, n>=2 paired rounds) - given facts 1-2 the win is
  a STREAMING bet, so a cheap pre-gate is to measure the streaming remainder directly (wrap the
  per-lookup upload calls in a timer inside gpu_run's prompt path; 20 lines, host-compilable).

**Option B - shrink the NVMe floor without cards.** 54 GB/prompt at 7 GB/s is 25% of wall and the
only cost all four cards SHARE. Arena coverage is 46% (64 GiB / 148 GB of experts, rank.bin hot-set);
96-110 GiB arena on a DS4.1-dedicated machine covers ~70-75% and cuts the per-prompt file reads
toward the serve steady state (decode already reaches 98.6% from the same arena - prefill's 1200x6
entries/layer simply touch a wider set). Zero new code; RAM is the constraint (vLLM co-tenancy holds
~50 GiB today). Also refuses to fight the wrong war: bigger chunks do nothing (fact 4).

**Option C - data-parallel serving (4 engines, 1 card each).** The engine's per-card dense footprint
is small (152 GB GGUF is ~96% experts), so four `ds41_generate --serve` instances could take requests
round-robin: ~4x CONCURRENT prefill+decode throughput, no fan-in, no sync. Against it: 4x arena
(64 -> ~15 GiB each under RAM pressure) re-creates the cache starvation the prize depends on, file
reads contend the one NVMe (eager-t4 already showed 2x file wait under seed contention), and it is a
serving-architecture change (router, park state, per-card configs), not an engine flag. Worth a
design note, not this quarter.

## 4. Not options (measured elsewhere, re-confirmed here)

- GEMM-split TP (any card count): P6.2 +41% half-k path; fact 1 says even perfect scaling saves 0.4 s.
- Layer staging for prefill: P5.5 -3x at 2 cards, -5x at 4 (4.3x file-read amplification).
- Per-token fan-out (`--tier-cards` as it exists today) for prefill: EXP2 eager-t4 = eager-ctl,
  file contention makes it worse at boot. For DS4.1 DECODE it is -25% (P6.4, re-measured under P4b
  concurrency for #46: 0.750/0.752) - the fan-out tax, and the reason Option A must ride CHUNK
  cadence, not token cadence.
- PTQ1_0 transcode: 18 ms GPU, 880 GB/s. Never was a cost.
- The per-token serve pass path as a prefill route: fact 5 (3x slower than chunked).

## 5. Order of work (if prefill is the goal)

1. Option B config knob (arena sizing per dedicated-machine profile) - no code, measurable today.
2. Option A pre-gate timer (20 lines in the tier prompt path) to name the 20 s precisely:
   if it is arena->card copy waits, Option A's bet is priced; if it is host entry-prep, four cards
   do not fix the host and the honest verdict is "prefill is capped at ~40 tok/s on this box until
   the RAM->VRAM pipeline for prompts is rebuilt".
3. Option A implementation + same-boot witness (gate 1.25x paired, 1200 and 4096, serve-shape too).
4. Serve wiring + the #46 batch-path caveat resolved by construction.
