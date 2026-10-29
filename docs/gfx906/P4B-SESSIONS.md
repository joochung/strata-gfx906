# P4b - concurrent sessions on the ported engines (time-slicing + N park slots)

Status: DESIGN (2026-10-28). Implementation order: P4b.1 DS4.1 engine slots, P4b.2 server slice
arbiter, P4b.3 DS4.1 witness, P4b.4 GLM mirror. Follow-on: task #46 (DS4.1 4-shard re-measure under
this concurrency) is the design's stress test, so it runs on the DS4.1 path first.

## 0. What today costs what (measured / line-referenced, this worktree)

- The server serializes EVERY request behind one lock: `Service.fifo` (`serve/server.py:3157`), acquired
  for the WHOLE generation (`serve/server.py:4316-4341`), with SSE `ping` keep-alives while queued. A
  128-token GLM request at the witnessed 2.7-4.2 tok/s blocks every other user 30-50 s. That is the
  problem P4b fixes.
- Both ported engines run ONE generation to completion (`glm_generate.cpp:1108-1310`,
  `ds41_generate.cpp:2190+`): hist/snap_len per conversation, ONE snapshot pair, and DS4.1 has ONE
  park slot (`STRATA_SERVE_PARK`, `ds41_dense.cpp:2961-3003`): `park(n)` host-copies the snapshot
  rings + live `comp/idxk` rows `[0, n]` + spare rows; `unpark()` writes them back. Written for the
  harness pattern (main conversation + side calls), it is exactly a session cache with N=1.
- A continuation GEN reuses at ZERO cost: `hist.push_back(tok)` during decode
  (`ds41_generate.cpp:2778,2804`) means prompt+generated-so-far is the exact LCP of the next slice,
  and the positional caches keep every row the slices wrote.
- Per-session bytes (docs/gfx906/P4-RAM-BUDGET.md 54,70 + code glm_dense.cpp:900,926-931,
  ds41_dense.cpp:1631,1660-1661): DS4.1 2,560 B/token + ctx-independent rings (RAW=144, doubled by
  raw_snap) ~= 0.08 GiB per 32K session; GLM 11,968 B/token + 139 MB KDA state + ~70 MB snapshot
  ~= 0.5-0.6 GiB per 32K session. Four sessions of each engine fit the card with the elastic expert
  slot budget giving back ~2 GiB (the mechanism exists: `glm_generate.cpp:1146-1190`).
- Park/unpark traffic: a 20K-position DS4.1 conversation is ~50 MB of rows + rings; host round-trip
  at pinned-DRAM speed is tens of ms - noise against a ~5 s slice.
- P2's census (receipt 2026-10-20-p0.4-census-sweep.json): lane route Jaccard 0.7-1.9%. Shared
  batched decode has NO reuse to harvest on this workload. P4b therefore claims FAIRNESS, not
  throughput: N sessions get N interleaved streams at ~the single-session aggregate rate; nobody
  waits 50 s to see a token. Do not promise 4x - nothing here delivers 4x.

## 1. Design decision: time-slice in the server, remember in the engine

Two changes, decoupled, each testable alone:

1. ENGINE (`ds41_generate`, then `glm_generate`): turn the single park slot into **N park slots**,
   self-managed - no session ids in the protocol. A GEN whose prompt best-matches a parked
   conversation (longest common prefix, existing partial rule `lp >= max(park_min, 3/4 x len)`)
   unparks that slot; an unrelated GEN parks the current conversation into a free slot (LRU evict).
   Flag `--park-slots N` (default 1 = today's single-slot code path, byte-identical). READY gains
   ` slots=N` ONLY when N > 1 (an old server keeps parsing the old line unchanged).
2. SERVER (`serve/server.py`): when the engine advertises slots (and has no native `batch`), run the
   generation in **slices of `slice_tokens` (default 16)** and between slices yield the fifo IF
   `status["queued"] > 0` (release + re-acquire). Round-robin emerges from the existing FIFO queue:
   A slices 16 tokens, yields, B (queued) gets its turn, yields, ... The engine never sees a mid-turn
   interrupt: each slice is a complete GEN ending in DONE; the parked conversation is exactly what
   the park slot already is. No STOP, no preemption, no new protocol verbs.

Why not the alternatives: BSlot-style kernel batching is P1/P2 territory and the census says it
buys nothing here; N engine processes (replicas.py) duplicates 156 GB of CPU RAM per replica on a
125 GiB box; token-level preemption with state save/restore costs a per-token 0.5 GiB copy - slice
boundaries (park once per handoff, tens of ms) are the cheap granularity the positional caches give
us for free.

## 2. Engine changes (DS4.1 first)

2a. `Ds41Dense` park interface: `park(positions)` -> `park(positions, int slot = 0)`,
`unpark(slot = 0)`, `drop_park(slot = 0)`, `parked(slot = 0)`; storage `parked`/`parked_pos` become
per-slot vectors. Slot 0 keeps today's exact buffers and semantics, so `--park-slots 1` is
byte-identical. (`ds41_dense.hpp:313-320`, `ds41_dense.cpp:2961-3003`.)

2b. `ds41_generate` serve loop: park bookkeeping (park_hist/park_logits/park_rows) becomes a small
per-slot table with last-used stamps. Arrival logic: scan slots, best lp wins (existing partial
rules per slot); unrelated GEN parks the live conversation into a free slot, else evicts the
least-recently-used slot. The park condition gains: with slots > 1, park conversations from
`--park-min 128` tokens (the default 2048 exists to avoid parking trivia when there is ONE slot;
with N slots, every active slice must be parkable or it re-prefills at 16.8 s+ - the batch-path
crossover, `STRATA_SERVE_STEP_MAX` note at `ds41_generate.cpp:2264`).

2c. Per-slice snapshot: with slots > 1, at each DONE re-arm the snapshot at the new
hist length (`dense.snapshot(hist.size())`, the call already exists at `:2636` taken pre-decode;
re-arm cost is the ring twin copy, ~ms). This is what makes slice continuation FREE: park then
covers prompt+generated, the next slice unparks at lp == full length, re-feeds nothing. Consequence
to document: a re-snapshot past the prompt-end moves `snap_len`, so the REGENERATE fast path
(same prompt again, samples from `snap_logits`) fires only at slice ends - a repeated prompt re-feeds
from the prompt-end LCP instead (correct, slower). Only with slots > 1.

2d. DSpark: the park/unpark path already resets `dsp_next` (acceptance drops after a swap, never
correctness, `ds41_generate.cpp:2255-2258` comment). Multi-slot keeps that property; no work.

2e. Refusals: `--park-slots N>1` with staged serve (already refused: reuse bookkeeping is
per-instance, `:2177-2188`). `--park-slots N>1` with `--tier-cards`: compatible BY CONSTRUCTION
(park holds KV/cache rows host-side; shards own experts, not caches) - DS4.1's tier+serve refusal is
task #46's wiring, not P4b's.

## 3. Server changes

- Parse the READY suffix into `StrataEngine.session_slots` (absent = 0, native engines unaffected).
- In the single-turn path (`serve/server.py:4316-4341` acquisition, engine.generate at `:1949`):
  when `session_slots > 0` and `slice_tokens > 0`, drive the GEN in slices: send
  `GEN min(remaining, slice_tokens) ... ids`, yield tokens, on DONE append them to ids and, if
  more remains, call a `yield_turn()` seam the Service injects (fifo release + re-acquire) only when
  `self.status["queued"] > 0`; then continue. Cancellation/timeout/STOP semantics ride the existing
  per-request cancel event: between slices the engine is idle, so STOP is never racing a slice.
- Sampling: the seed is re-sent per slice; at temperature 0 slices are bit-identical to one long
  GEN (greedy), at temperature > 0 a sliced generation samples a different stream than one GEN would
  (each slice re-seeds). Document, do not "fix" - per-slice determinism is the claim.
- Fake-engine tests: extend `serve/test_parallel.py`'s harness (run as
  `python3 -m serve.test_parallel` from the repo root; extra engine log lines BREAK its assertions -
  known trap). Cover: slice loop, yield only when queued > 0, DONE accounting across slices,
  READY-with-slots parsing, old-READY unchanged.

## 4. Budgets and limits

- Host RAM per parked slot: rings (~ctx-independent, per-layer RAW x DH x 2 B x 2 twins x 40 layers
  ~ 0.3 GiB DS4.1 / KDA state 139 MB + 70 MB GLM) + `park_rows` x 2,560 B (DS4.1) / ~13.6 KB (GLM).
  With `--park-slots 4` and 20K-position conversations: ~1-2.5 GB. Inside the 84 G serve scope next
  to the 60 G arena; the runner's MemAvailable gate (65 GiB) still applies.
- VRAM is NOT multiplied: caches are positional and shared; only parked (inactive) sessions' rows
  live host-side. N ACTIVE 32K contexts on the same positional cache do not exist here - one live
  conversation owns the rows; a second must park the first. (True N-resident caches = the BSlot
  rewrite, out of scope; the park design is the honest version of per-session caches for positional
  engines.)
- Fairness granularity: one slice = `slice_tokens` tokens at that session's rate; at DS4.1's ~2-4
  tok/s and slice 16, a user sees a 16-token burst every ~5-8 s x active sessions. Tune per engine.
- NOT claimed: throughput gain (section 0), shared-prefix radix (P2 NO-GO stands), native-engine
  changes (none), mid-slice preemption (never needed - slices end at DONE).

## 5. Witness protocol (P4b.3, DS4.1, GPU window)

Runner `tools/gfx906/run-p4b-ds41-sessions.sh`.

Stage A (engine-level, deterministic, no server): raw-protocol replay with `--park-slots 3` and
`STRATA_SERVE_PARK_MIN=16` - a resumed generation must equal a fresh prefill of the same conversation
(12/12-token match), 4-conversation pressure through 3 slots must answer every turn, and a strict-prefix
resubmission must NOT read the longer parked state (the P4b.1 guard: it prefills fresh).

Stage B (serve-level, ONE boot through serve_ds41.py with `--park-slots 4`): six conversations x two
turns run serially then concurrently (same boot - the engine is not bit-deterministic across processes,
the standing rule), then a fairness phase: one 256-token marathon with two short requests arriving at
+2 s and +4 s. Metrics: completion, errors, reused counters, TTFTs, per-slice park/unpark lines.

Gates (receipt verdict): stage-A exactness + fairness (short TTFT < 60% of the marathon's wall) + zero
request errors. Serve-level TEXT equality serial-vs-concurrent is INFORMATIONAL only: residency drift
reorders float sums and flips knife-edge argmax within a boot under traffic (the standing rule from
P6.4b); the exactness gate is the raw-protocol stage A.

Found by witness run 3 and fixed in cc31f914: releasing the FIFO at a slice boundary is not HANDING it
over - the releasing thread re-acquires before the woken waiter gets the GIL (Python locks are not
FIFO-fair). The trace was `queued=1, seam fired, 0 beats`, and the short request still waited the
marathon's remaining wall (28.7 s TTFT on the real box). beat() now sleeps 2 ms before each reacquire;
MarathonFairnessOverHttp in serve/test_parked_slices.py reproduces the starvation without the window
and passes with it.

RESULT (run 4, 2026-10-28 21:49, tip 0899e180, one boot through serve_ds41.py --park-slots 4):
GO. Stage A 5/5 PASS. Fairness: marathon wall 41.1 s; short requests arriving mid-marathon got first
token at 10.2 s and 15.7 s (unsliced baseline ~39 s) - gate is 60% of wall, cleared at 25%/38%.
All 25 requests completed, 0 errors; every concurrent request showed reused>0 through the real stack
(engine lines like `reuse lcp=292 ... reused=292` per slice: zero re-feed). Text match serial-vs-
concurrent 0/6, informational as designed (the concurrent phase runs on a much fuller arena than the
serial one did; residency differs, sums reorder, knife-edge flips). Receipt
`benchmarks/gfx906/receipts/2026-10-28-p4b3-ds41-sessions.json`.

#46 rides the same runner shape with the tier wired: DS4.1 ctl vs t4 under rotating sessions.

## 6. GLM mirror (P4b.4, last, per the plan's "KDA state per session first, shared prefixes last")

`glm_dense` has NO park today (snapshot pair only, `glm_dense.hpp:159-165`, `glm_dense.cpp:1074-1111`).
Mirror 2a-2c: `park(positions, slot)` = host-copy per-KDA-layer S+conv snap twins + MLA `kvc`
rows `[0, positions]` + `ipool` rows `[0, positions/4]`; `unpark(slot)` writes back and re-arms the
snapshot (the twins already exist, ~209 MB per session incl. state - P4-RAM-BUDGET.md 54-55). The
serve loop's hist/snap single-conversation bookkeeping (`glm_generate.cpp:1188-1239`) becomes the
same per-slot table. Slice arbiter: same server code, READY suffix, no further server work.

LANDED (P4b.4) and WITNESSED GO 2026-10-28 run 2 (receipt `benchmarks/gfx906/receipts/2026-10-28-p4b4-glm-sessions.json`,
runner `tools/gfx906/run-p4b-glm-sessions.sh`, window log `logs/p4b4-glm-witness-run2.log`): stage A 5/5 exact
(A2 resumed reused=37, D2 reused=28, the strict-prefix replay refused reused=0 and matched its resumed answer
12/12); stage B one boot through `serve_glm.py --park-slots 4`: the engine parked into slots 1/2 and unparked
slot 0 (`kda state + mla rows`) with `reused=92..189` per turn under concurrent replay, 15 requests 0 errors;
fairness shorts got their first streamed token at 16.1 s and 26.8 s against a 78.8 s marathon wall (20%/34%,
gate 60%). The hand-off is slower than DS4.1's (one slice + GLM's ~13.6 KB/position park copy + prefill), which
the TTFT numbers show honestly. Run 1 (log `logs/p4b46-window.log`) reported fairness false through a CLIENT bug,
not an engine one: it timed the first `delta.content`, GLM thinks inside every answer and the thinking streams as
`reasoning_content` (server.py:5022), so a 32-token short looked silent while its engine decoded it - the shorts
had finished at wall 20.3/30.2 s of a 75.3 s marathon all along. Fixed in 19e046e4 (first streamed token, either
channel; that IS the scheduling evidence). `--park-slots N` on `glm_generate --serve`, the same
1..16 clamp and the same `READY ... slots=N`-only-when-N>1 suffix; `GlmDense::park/unpark/drop_park/
parked(slot)` (`glm_dense.hpp`, `glm_dense.cpp` after `snapshot_pos`). Identity: GLM has no park at N=1,
so EVERY added line - the park calls, the scan, the READY suffix, the park scope - is inside a
`park_slots > 1` gate and the default run reaches none of it.

Four places the KDA state forced a different shape from DS4.1:
1. A park copies the LIVE state (live KDA `S` + `conv`, the MLA `itail` carry - the same tensor set
   `snapshot()` twins - plus `kvc` rows [0, positions) and `ipool` rows [0, ceil(positions/kpool))),
   not the snapshot twin at the last prompt end. That makes DS4.1's 2c re-arm unnecessary: the park
   already covers prompt + generated, so a slice resumes at zero re-feed without any extra copy.
   The `ipool` DUMP row is not saved (dead-slot/padding writes land there; attention reads [0, npc)).
2. A slot resumes on a FULL match that STRICTLY EXTENDS it, only. DS4.1's partial rule (resume at the
   longest common prefix >= 3/4) needs positional caches to overwrite a stale tail; a recurrent state
   cannot rewind to `lp` (attn_router_n refuses an out-of-order pass), and a same-length replay would
   decode from `last_logits` that belong to the outgoing conversation (no saved logits twin), so both
   prefill fresh. The strict-prefix guard of 5b4643df is ported verbatim ahead of that rule.
3. The OUTGOING conversation is parked on EVERY hand-off (also the one that resumes a slot), with the
   resuming slot protected from being the free/LRU victim. DS4.1's multi-slot scan parks only on an
   unrelated arrival, which drops the interrupted conversation; here dropping it costs a re-prefill.
4. `STRATA_SERVE_PARK_MAX` defaults to 32768 positions, not 65536: GLM's window is ~13.6 KB per
   position (section 4), ~5x DS4.1's, so one slot at DS4.1's ceiling would be ~0.9 GiB.
`--park-slots N>1` refuses a staged (--gpus) --serve, the same refusal as 2e. The engine log lines to grep (both
witnesses) are `parked the N-token conversation into slot K (slot=K ids=I positions=P, kda state + mla rows ...)`
and `unparked slot K's ... (kda state + mla rows back)`.
