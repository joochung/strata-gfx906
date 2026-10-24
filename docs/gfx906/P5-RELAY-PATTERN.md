

# Native layer-split relay: how activations cross a stage boundary

All paths relative to /ai/engines/gfx906/worktrees/xolotl-merge. Note upfront: the task premise for item 3 is off — `locate(layer)` does NOT live in src/core/expert_source.cpp; it is a callback (`Locate`/`home_of`) in src/program/generate.cpp. Details below. No file was modified.

---

## 1. THE RELAY ITSELF

There are TWO different relays, with different media and APIs:

### 1a. Decode (verify window) — mapped pinned host memory, copied by kernels inside CUDA graphs, sync by host stream-sync

Media. One hand-off buffer per stage boundary, allocated in src/program/generate.cpp:7883-7896 (inside the serve setup, guarded by `if (n_stages > 1)`):

    const size_t hb = (size_t) strata::kernels::kVerifyMaxT *
                      (size_t) strata::core::Verifier::handoff_floats(g) * sizeof(float);
    std::vector<float*> hand((size_t) n_stages - 1, nullptr);
    for (float*& h : hand) {
        float* hh = nullptr;
        if (cudaHostAlloc((void**) &hh, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &h, hh, 0) != cudaSuccess) { ... }
        std::memset(hh, 0, hb);
    }

So: host pinned memory, MAPPED (device-visible through its device pointer) and PORTABLE (readable from the second GPU), NOT device memory, NOT device-to-device. Layout per token: [R (hc x n_embd)] [bo (n_embd)] [inj (hc)], `Verifier::handoff_floats` = hc*n_embd + n_embd + hc (include/strata/core/verify.hpp:191). The doc contract is include/strata/core/verify.hpp:177-186 (`set_stage`): "Both pointers must be device-visible (mapped pinned memory, portable when the stages are on two devices)". Wiring: generate.cpp:7897-7907 (`stage_ver(st).set_stage(lb, le, hand[st-1], hand[st])` and `set_next(&stage_ver(st+1), &split_drive)` at :7936).

What is copied, decode (one hidden vector per token, T tokens per window). Both ends are kernel nodes captured inside the stage's per-T window graph (`Verifier::record_window` is the capture body, verify.cpp:905; `Verifier::capture` at :1925; the graph is replayed with cudaGraphLaunch on the stage's compute stream `cs_` at run(), verify.cpp:2326-2330).

Producer side (stage with `le_ < g.n_layers`), src/core/verify.cpp:1774-1780:

    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        float* hout = hand_out_ + (size_t) hrow0 * HB;
        copy_from_mapped(hout, R_,     (int64_t) T * HC * N, cs);
        copy_from_mapped(hout + (size_t) T * HC * N,     bo_,   (int64_t) T * N,   cs);
        copy_from_mapped(hout + (size_t) T * (HC + 1) * N, inj2_, (int64_t) T * HC, cs);
        return true;
    }

Consumer side (stage with `lb_ > 0`), verify.cpp:982-988:

    const int64_t HB = Verifier::handoff_floats(g);
    if (lb_ > 0) {
        const float* hin = hand_in_ + (size_t) hrow0 * HB;   // the group's rows: [R][bo][inj] contiguous
        copy_from_mapped(R_,   hin,                             (int64_t) T * HC * N, cs);
        copy_from_mapped(bo_,  hin + (size_t) T * HC * N,       (int64_t) T * N,   cs);
        copy_from_mapped(inj2_, hin + (size_t) T * (HC + 1) * N, (int64_t) T * HC, cs);
    }

`copy_from_mapped` is a float4 grid-stride copy KERNEL on the given compute stream (src/kernels/cuda/elementwise.cu:229-235 kernel, :299-310 launcher) — deliberately not a memcpy, because "a memcpy node is a copy-engine operation, which WDDM submits separately and which measured 67 flushes per token" and a memcpy node would not stay graph-capturable on the compute queue (include/strata/kernels/elementwise.hpp:100-104). It is direction-generic: used device->mapped (produce) and mapped->device (consume).

Streams and events. Everything (layers + hand-off copies) is on the ONE compute stream `cs`/`cs_` of each stage (the stage's own device, chosen at init: verify.cpp:606 `cudaGetDevice(&device_)`). No cross-stage CUDA events exist on the synchronous path.

The sync the next stage waits on is a HOST stream sync of the producing stage, then a host-side chain, verify.cpp:2464 and :2483-2485:

    const cudaError_t se = cudaStreamSynchronize(cs_);          // :2464  ("synced above")
    ...
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);
    }

So: stage A's graph (ending with the mapped-memory write) completes fully (host sync), THEN the caller launches stage B's graph, whose first nodes upload the hand-off into B's device buffers on B's stream. Correctness across PCIe is plain program order through the host; the buffers are reused only one window at a time in this path.

Pipelined variants (for completeness): with `--pipeline-windows` each parity gets its own hand-off (`hand_b`) and its own verifier, generate.cpp:7943-7963, and per-stage non-blocking streams created on-stage via `OnDevice`, generate.cpp:7872-7876; completion is polled via an event recorded after the graph, `cudaEventRecord(ev_done_, cs_)` verify.cpp:3513 with cudaEventQuery polling in done()/pl_finish (verify.cpp:3620-3638, "an earlier stage: the hand-off is written" at :3641). With `--batch-groups` the row ranges of window groups are bound into disjoint hand-off row ranges so in-flight groups never collide (include/strata/core/batch_rows.hpp:14-18; verify.cpp:962 `hrow0 = batch_rec_ ? row_base_ : 0`; verify.cpp:2982-2983).

### 1b. Prefill (chunk x hidden) — plain pinned host memory, real cudaMemcpyAsync, one host thread per stage

Media: NOT mapped — plain portable pinned, two double-buffered chunk buffers, allocated per stage in `Prefill::init`, src/prefill/prefill.cpp:1121-1127:

    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4,
                                        cudaHostAllocPortable) != cudaSuccess) {
            err = "prefill: the layer split's hand-off buffers"; ... }

`D = N * HC` (prefill.cpp:118: `constexpr int64_t N = 2560, HC = 4, D = N * HC, ...`) — the full hc-wide residual, and ONLY that (no bo/inj, unlike decode). The next stage's `hand_in_` is the HOST pointer (include/strata/prefill/prefill.hpp:204 "the previous stage's rows of the chunk being read (host, pinned)"); the next stage is the next `Prefill` object (`set_stage(lb, le, Prefill* next)`, prefill.hpp:161-162; wired in generate.cpp:7653 and :7669).

Producer (end of a chunk), prefill.cpp:4176-4195:

    if (next_ != nullptr) {
        // The current hand-off slot was used two chunks ago.
        float* h = m.hand[hand_buf_];
        if (cudaMemcpyAsync(h, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
            cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "prefill: the layer split's hand-off: " ...; }
        // Wait only for the DIRECT successor's previous chunk. That successor
        // may already have forwarded its older chunk to later GPUs.
        if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
        if (next_run_.valid() && !next_run_.get()) { err = next_err_; return false; }
        next_err_.clear();
        next_->hand_in_ = h;
        next_->single_chunk_ = single_chunk;
        next_run_ = std::async(std::launch::async, [this, tokens, c0, T, p0] {
            strata::aux_cpus::pin_current_thread();
            return next_->run_impl(tokens + c0, T, p0, next_err_); });
        hand_buf_ ^= 1;
        continue;   // the last stage reports the chunk (on_chunk)
    }

Consumer (start of its chunk loop, prefill.cpp:2196-2212):

    for (int64_t c0 = 0; c0 < n; c0 += m.T) {
        ...
        if (hand_in_ != nullptr) {
            if (cudaMemcpyAsync(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4,
                                cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                err = "prefill: the layer split's hand-off upload failed"; ... } }

So the prefill sync is again: real D2H memcpy on the producer's stream `m.cs` + `cudaStreamSynchronize(m.cs)` (:4179-4180), then the successor is started as a separate host task (`std::async`, :4189-4192) whose Prefill makes its own device current (`cudaGetDevice(&m.device)` at init :1115; `const core::OnDevice on_device(m.device)` per prompt call :2108). Double buffering (`hand_buf_ ^= 1`) plus `next_run_.get()` gives chunk-level pipelining across stages while keeping one host thread per stage; `on_stage_chunk` (callback set in generate.cpp:8638-8652) is the per-stage checkpoint hook, not a relay sync.

Summary for the port's mental model: there is no device-to-device path and no P2P requirement. Decode relays through MAPPED pinned memory with kernels (because the copies sit inside captured graphs and must be graph nodes on the compute queue); prefill relays through plain pinned memory with memcpyAsync (because it is eager, host-thread-driven). Both cross the boundary via the host; the "sync the next stage waits on" is always a producer-stream completion observed by the host (cudaStreamSynchronize / event poll), never a cross-device event.

---

## 2. WHERE ACTIVATIONS LIVE AT THE BOUNDARY (native engine)

Device memory, never host RAM between layers. The residual R is a float device buffer carved from a per-verifier device arena in the same sequence counted once and bumped-allocated:

- verify.cpp:713-715 (inside the "device arena: the same sequence counted, then carved" block): `ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);` (bo_ at :716, inj2_ at :717).
- prefill.cpp:1260-1266 `Prefill::carve`: "Every device buffer of a chunk of T tokens, from the Alloc alloc"; `m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok);`

There is no ggml ctx anywhere in the native engine — it is a hand-written kernel engine (WeightRef blobs, custom kernels). Between layers, R stays on the stage's GPU; it crosses to host memory ONLY at a stage boundary via the mapped/pinned staging above. (host_res in generate.cpp is the expert-residency slot table, not activations — see the trap documented at generate.cpp:283-286.)

---

## 3. locate(layer) -> (cache, stream, device)

Not in expert_source.cpp. It is:

- The struct: src/program/generate.cpp:287-291

      struct SwapHome {
          strata::core::ExpertCache* cache;
          cudaStream_t stream;
          int dev;
      };

  with the "why" comment at :283-286: with a layer split, `host_res` holds slot numbers of WHICHEVER cache owns the layer, so the copy back must read the owning card's cache on that card's stream: `locate(layer)` names them (cache, stream, device; -1 = CUDA0). Reading a later stage's slot out of CUDA0's cache once copied the wrong expert's bytes into RAM.

- The consumer: `resident_stage_swaps` template, generate.cpp:292-330; `const SwapHome home = locate(s.layer); const strata::core::OnDevice on(home.dev); cudaMemcpyAsync(src.exchange_buffer(q), home.cache->device_slot(slot), ..., cudaMemcpyDeviceToHost, home.stream);` (:307-313), then per-used-stream sync at :321-329.

- Where the mapping comes from: CLI parse `--layer-split` / `--split-device` at generate.cpp:2130-2131 (options at :618-623); validation (increasing split points, one device per boundary, `--split-device 0` = same-GPU A/B) at :2408-2425; the auto search sets `split_at` at :4319; the layer->stage table is the lambda `stage_of` at :4366-4371 (`while (st < split_at.size() && l >= split_at[st]) ++st;`), and per-stage ranges `stages[i]->lb/le` at :4384-4385. The actual `Locate` lambda passed at call time is `home_of`, generate.cpp:8930-8936:

      auto home_of = [&](int64_t layer) -> SwapHome {
          const int stn = multi_gpu ? stage_of(layer) : 0;
          if (stn > 0) {
              GpuStage* gs = stages[(size_t) stn - 1].get();
              return {&gs->cache, gs->adapt_stream, gs->dev};
          }
          return {&xcache, adapt_stream, -1};
      };

  (second call site at :13897).

- How callers use `dev`: as the `OnDevice` argument — every CUDA call touching the returned cache/stream is wrapped (`const strata::core::OnDevice on(home.dev)`, :308; sync at :323). The owning stage object is `GpuStage` (generate.cpp:1106-1130): its own WeightTable, NativeDense, SessionState, ExpertCache, Verifier, Prefill, streams, all created on `gs.dev` (e.g. stream creation wrapped in `OnDevice(gs.dev)` at :7873-7876; verifier init at :7925-7940).

- What IS in expert_source.cpp for a split: `stage_lend_regions` (include/strata/core/expert_source.hpp:581-583, src/core/expert_source.cpp:2797-2830) — generate hands the RAM copy each stage's lend region so prompt-borrowed experts are RAM-backed, not page-faulted from the shards. This is a RAM-planning table, not a (cache, stream, device) lookup.

---

## 4. The On(dev) RAII guard (src/core/peer_experts.cpp:37-42)

Exact code (peer_experts.cpp:37-42):

    /// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
    struct On {
        int prev = 0;
        explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
        ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
    };

Used at peer_experts.cpp:57, :113, :229, :261, :305, :324 (close, open, and every compute entry). The rule, stated for the portable twin `strata::core::OnDevice` (include/strata/core/on_device.hpp:3-4): "An object that owns CUDA streams, graphs and buffers on one device (a verify stage, the drafter) wraps its public calls in this, so the caller's thread may be on any device. A negative device, or the device already current, does nothing." So: every call that allocates, records/launches, or synchronizes objects belonging to another device must be wrapped — examples: generate.cpp:308 (copy-back), :7874 (stream create), :7925 (stage verifier init), prefill.cpp:1240, :1407, :2108 (Prefill per-call device pin). Plain host-mapped buffers and cudaHostAlloc don't need it; anything naming a stream/graph/device pointer does.

---

## 5. docs/MULTI_GPU.md :240-310 — --pipeline-windows vs the sync path

- Sync (non-pipelined) path, described at :266-267: "With a split the cards take turns on a verify window: the first card runs its layers and hands off, then waits while the last card runs the rest, the head and the draft." That is exactly the host-mediated relay of section 1a — one window in flight, the first card idle while the last card works.
- `--pipeline-windows 2` (:247-256, :266-273): lets the FIRST card start the NEXT window while the last card still verifies this one. The next window is a guess (window accepted whole + bonus token matches the draft layer's prediction); on a wrong guess the first card restores a copy of its recurrent state taken while the window ran, and the real tokens rebuild the next window; a guessed window starts only when the draft layer's estimate says it is likely kept. On gfx12/RDNA4 it is default-on for 2+ stages (:250-252); `--pipeline-windows 1` overlaps only short prompt reads through verify windows (:274-275).
- 3+ stages (:277-280): every stage but the last is a "front" stage; the guessed window follows the verified one through them one card behind, each front stage keeping its own pair of recurrent-state copies; only the last stage waits for the verdict.
- Cost/flags (:282-300): a second verifier per stage (~160 MiB kept out of each card's expert cache, plus ~3 MiB per GDN layer of state snapshots for level 2); off with `--batch`, `--peer-device`, helper caches, `--split-device 0`, no draft layer, or `--adapt-async 1` on 3+ stages; penalty/coupled-draft requests decode serially. Draft layer moved to the FIRST card with the pipeline on (:257-264).

---

## 6. What the ported engines have today at the equivalent site

### tools/glm (GLM-5.3-Flash)

- What `im` is: `im` is just the conventional local alias `Impl & im = *p_` for `GlmDense::Impl` — declared in tools/glm/glm_dense.hpp:124 (`std::unique_ptr<Impl> p_`), defined at tools/glm/glm_dense.cpp:242. It holds geometry, the WStore of weights, the backend pointer, and every persistent ggml tensor/state buffer.
- Per-token forward loop: tools/glm/glm_generate.cpp:500-534 — lambda `step`: `dense.begin_token(tid)` (:501), then `for (int l = 0; l < n_layer; ++l)` (:503): `dense.predict(l, ...)`/`tier.prefetch` (:506-510), `dense.attn_router_n(l, pos, 1, ids, w, &x)` (:513), `tier.run(l, ids, w, x, routed)` (:524), `dense.finish_layer_n(l, 1, routed)` (:531). Chunk loops: `--serve` short turns at :614-623 (`begin_tokens` + same trio with `m` tokens), prompt prefill at :775-801.
- Inside GlmDense, one layer = two ggml graphs per pass size: the attn+router graph (`attn_router_n`, glm_dense.cpp:1103-1206: build/reuse graph, one input upload, `ggml_backend_graph_compute_async(backend, var->gf)` at :1185, then ONE readback of the block [fn | ids | wts] with `ggml_backend_synchronize` at :1190-1192) and the finish graph (`finish_layer_n`, :1208-1226: uploads `routed_sum` async (:1213), computes `L.gf_finish[n]` async (:1219)).
- The activation at the boundary: `x_state`, an F32 ggml tensor [D, HC, NT] (glm_dense.cpp:830; HC==4 enforced at :120), living on ONE backend buffer `sbuf` allocated at :898 via the backend's default buft. The finish graph writes it in place as its last node — glm_dense.cpp:686:

      ggml_build_forward_expand(gf, ggml_cpy(gc, lout, ggml_view_3d(gc, im.x_state, D, HC, n,
                                        im.x_state->nb[1], im.x_state->nb[2], 0)));

  and the next layer's attn graph reads it via a view at :511. Between layers there is literally NO copy today: layers are chained by "finish(l) writes x_state, attn(l+1) reads x_state", ordered only by the host calling them in sequence (and by the per-layer readback sync inside attn_router_n; finish is issued async). Bytes per token: D * HC * 4 (floats), column stride `x_state->nb[2]` = D*HC*4, contiguous per token (see logits_rows using `slab = im.x_state->nb[2]`, :1249-1252). Host or device? Depends on the backend: `ggml_backend_is_cpu(backend)` (glm_dense.cpp:250-251, :265) — CPU backend (default in the header, glm_dense.hpp:60) means x_state is plain host RAM (the GGUF mmap path binds weights host-side, :176-198); CUDA backend means it is device VRAM. glm_generate defaults to cuda: options.backend = "cuda" (:46), `ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU ...)` (:208-210).
- Where a boundary copy would go: the natural seam is the generate loop, between `dense.finish_layer_n(K-1, ...)` (glm_generate.cpp:531/623/801) and `dense.attn_router_n(K, ...)` (:513/618/781). The payload is `x_state` columns [0,n) = n * D * HC * 4 bytes F32 (plus, for a real split, everything the later layers own: their KDA S + conv states, MLA latent cache rows, indexer pools — all allocated together with x_state in the single `sctx`/`sbuf`, :813-857/:898 — and their dense weights, in the single WStore buffer, :210). Today there is no second backend anywhere: graphs are allocated against one buft (`new_allo` = `ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))`, :347), so the mirror of the native pattern is (a) allocate x_state (or a second copy) per backend, or (b) after finish(K-1) read x_state back (host or D2D) and set it into the second engine's x_state before attn(K). Note the engines ALREADY make a host round-trip every layer for the expert tier (fn readback glm_dense.cpp:1190-1192; routed upload :1213) — a decode-style boundary copy through pinned host memory would ride the same seams.

### tools/v41 (DeepSeek-V4.1-Flash)

- `im` is again `Impl & im = *p_` for `Ds41Dense::Impl` (ds41_dense.hpp:60-61; implementation ds41_dense.cpp).
- Loops: decode lambda `step`, tools/v41/engine/ds41_generate.cpp:501-596 — `dense.begin(&tok,1,pos)` (:508), `for (int l = 0; l < n_layer; ++l)` (:512): `predict`/`predict_multi` prefetch (:513-537), `dense.attn_router(l, ids1, w1, &fn)` (:539), tier run (:540-560), then the fused `finish_predict(l, rsum, l+1, ...)` (:568/:573) or plain `dense.finish(l, rsum)` (:576). Multi-token chunk passes: `pass_m` :601-621 (attn_router :607, tier run_to_dev/run :611-617, finish :616), `pass_v` from :628 (loop :646, attn_router :689, finish :713); prompt prefill chunk loop :979-1004; layer-major batch path :894-939 (`attn_router_span(_kv)` :903/:908, `finish_span` :939, `upload_routed` :1438-1445).
- The activation at the boundary: `x_state` F32 [D, HC, NC] plus `pre_state` F32 [HC, NC] (the hc pre-mix that layer l hands to layer l+1, ds41_dense.hpp:11-13), allocated in `alloc_cols`, ds41_dense.cpp:937-938, buffer `im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend)` at :975. Finish writes x_state in-graph: `build_finish` — ds41_dense.cpp:843-851, notably :851 `ggml_build_forward_expand(gf, ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));`. Column size: "x_state [D, HC, NC]: a column is contiguous", :1655 — n * D * HC * 4 bytes F32 per token (plus HC*4 of pre_state). Per-layer persistent state (window ring `raw`, compressed KV `comp` possibly NVFP4, indexer keys `idxk`, ratio-2 compressor state) is in the single `sctx`/`sbuf` (:1071-1105), one buffer on one backend.
- Sync discipline today: ds41 computes synchronously — `Impl::compute` runs `ggml_backend_graph_compute` (blocking) at :446-459, and `attn_router` takes exactly one sync per layer for the three readbacks (fn/ids/wts), :1501-1506. A device-side activation path already exists on both edges: `attn_router(..., ggml_tensor** fn_dev)` returns the on-backend fn (:1509), and `routed_dev()` (ds41_dense.hpp, and used as `rdev1`/`tier.run_to_dev` in ds41_generate.cpp:611-617, :616) lets the expert tier write its routed sum straight into the device `routed` tensor, skipping the host.
- Where a boundary copy would go: between `finish`/`finish_predict`/`finish_span` for layer K-1 (ds41_generate.cpp:568-576 / :616 / :713 / :939) and `attn_router`/`attn_router_span` for layer K (:539 / :607 / :689 / :903-908). Payload: x_state (+ pre_state, since the hc pre-mix crosses layer boundaries in this model — ds41_dense.hpp:11-13) — n*(D*HC + HC)*4 bytes F32; everything from layer K up (weights, KV/indexer state, graphs) would have to live on backend B, since today one `im.backend` (:1043-1048) plus one sbuf/nbuf/weight buffer back the whole model.

---

## 6b. WHAT THE DS4.1 PORT ACTUALLY SHIPS (P5.3b, 2026-10-24)

Section 6 above is the survey of what the engines had BEFORE the split.  This is what tools/v41 does now, in the
terms that section asks about - the payload, the seam, and the three decisions that are DS4.1's own rather than GLM's.

**The seam.** `ds41_generate.cpp`'s `staged_pass()` drives one pass of at most `sr.unit` tokens: stage 0 takes the
token ids (`begin`), each stage runs its own `[lb, le)` layer loop against the SAME tier (the routed half is not split;
that is P5.4), and at every boundary the producer's finished columns go through one staging slot to the stage after
it.  The head runs on the LAST stage only, over its own state - so the logits are the same rows the unsplit path
reads, with no payload round trip.  `relay_staging.hpp` sizes the slots and hands out the double-buffered windows;
`Ds41Dense::relay_read_out` / `relay_write_in` are the two copies.  The sync is the native relay's rule, unchanged:
the producer's stream completion observed by the HOST before the successor is launched; there is no cross-device
CUDA event anywhere in this engine.

**THE ONE THING THAT MAKES DS4.1 DIFFERENT FROM GLM: ITS CACHES ARE SHARED, AND A CACHE CANNOT CROSS A CARD.**
Everything else about the port follows from that.  In GLM every layer keeps its own KV and reads only it, so a
boundary can sit almost anywhere.  In DS4.1 a layer reads the compressed KV of the LATEST kv source at or below it and
the index keys of the latest index source at or below it (`Ds41Geometry::kv_src / idx_src`) - so layer 39 reads layer
20's compressed KV and layer 36's index keys, and those rings are persistent GPU state allocated once, read by POINTER
inside a graph.  The relay moves per-token state; it has no way to move a cache, and a per-pass D2H/H2D of a visible
cache would be gigabytes a token.  Therefore:

> **THE READER'S RULE.**  A boundary at `K` is legal **iff every layer at or above `K` finds both of its sources at or
> above `K`** - the stage that runs a layer must also run the layer whose ring that layer reads
> (`Ds41Dense::split_boundary_valid`; `split_valid_boundaries()` enumerates the set, `split_boundaries_ok()` vets a whole
> plan).  The stage-0 side is free: `kv_src(m) <= m < K` always.  Because the sources are monotone in `m`, the scan can
> only bite at the boundary layer itself, so a valid boundary is a layer that is **both** a kv and an index source (plus
> the vacuous case of a boundary layer that runs no attention).  For the shipped 40-layer build that set is
> **{1, 2, 8, 14, 20}**, usable as {2, 8, 14, 20} - and **20 is the even halfway point of the model**: stage 1 =
> [20,40) runs the writer of ring 20 together with every reader of it, the writers and readers of index rings
> 24/28/32/36, and the candidate writer (20) with all of its readers (24/28/32/36).  **This is a set, not a floor.**  An
> out-of-set `--layer-split` **snaps down** to the nearest valid layer below it (21 boots as 20, and the run says so);
> a plan the set cannot reach is refused before allocation with the set named, rather than silently running with fewer
> stages.  `build_attn` still aborts if a stage is ever built by hand over a ring it does not own.
>
> **THE FIRST REVISION HAD THIS UPSIDE DOWN, AND THE GATE NOW TESTS THE DIRECTION.**  The version committed earlier
> today required the *writers* to sit below the boundary and called `LAST_SOURCE + 1 = 37` a floor.  A writer below the
> boundary is precisely what strands its ring: the tail [37,40) reads layer 20's compressed KV and layer 36's index
> keys, both below 37, so the witness would have died at graph build with the abort the floor claimed to prevent.  The
> cuts at 3/9/15/**21** are still refused, but for the true reason (`kv_src(21) == 20 < 21`), and the 40-pair source map
> plus the derived set are now checked three times over: in the engine's C++, in the host gate's C++ **by simulating the
> stages and their layers** (`split_stages_own_their_reads`, which every 1-, 2- and 3-boundary cut of four geometries
> must agree with), and in awk in `gate-device-split.sh` step 3, derived from the engine's own geometry reader
> (`test_ds41_geometry <file.gguf>`, host-only) rather than from this file's transcription.

**The payload - two regions on this model, and the third one is provably unreachable.** GLM hands on `x_state` alone:

| region | tensor | bytes / token | carried on this model? |
|---|---|---|---|
| residual | `x_state` F32 [D, HC, NC] | 5120 x 4 x 4 = 81 920 | yes - the successor's first layer reads it |
| hc pre-mix | `pre_state` F32 [HC, NC] | 16 | yes - this model's layer l hands layer l+1 an explicit pre-mix; GLM folds it into x_state |
| candidates | `cand` I32 [CB, NC] | 8 192 | **no** - see below |

The candidate column is per-token state, so it COULD ride along like the other two and the machinery is there
(`relay_cand_from` for the in-place read, a payload region for the copy).  It never does, and the reason follows from
the reader's rule rather than from this checkpoint: a layer that ranks inside the candidate blocks is an index source,
and validity at `K` requires `cand_layer >= K` whenever a reader is left at or above `K` - so no legal boundary can sit
strictly above the producer and at or below a reader, and the producer plus its readers always share a stage.
`test_split_geometry.cpp` asserts exactly that (`split_candidates_cross` is false for every shape it can build,
including three-stage plans and models whose sources are rearranged), which makes the candidate region code a model
with a cache OUTSIDE those two lists would need - a hybrid SSM state, or a cache named by a key the geometry does not
read yet - at 8 KiB a token.  Kept, sized, gated, and honest about not firing here.  The witness therefore gates on
`cand 0)` in the slab line: a non-zero value means the geometry moved underneath the witness, not that a feature turned
on.

Each region starts at byte 0 of its tensor (the pass's columns are the leading columns of a column-major tensor), so
one boundary is a small number of contiguous copies in a fixed order, and `relay_slab_bytes()` is the ONE number both
sides size their staging from.  It is deliberately a function of the GEOMETRY, never of a stage's range: two sides of a
boundary that each computed their own idea of where the copy ends is the worst possible failure mode, so a stage copies
the whole payload even when it will not read one of its regions.

**Decision 1 - a shared weight set cannot serve two cards.** ggml hands the host an iGPU device pointer, and these two
cards' copy engines cannot read each other's memory, so a stage that borrows the owner's tensors must run on the
owner's card.  A cross-card split therefore loads one weight set per card (`--relay-copy-weights`) - the reason the
witness's cross-device runs need the memory budget - and `ds41_generate` refuses the combination that would look
cheaper and cannot work.

**Decision 2 - the pass unit is decoupled from the state columns.** A staged run drops the layer-major WALK (the relay
has no seam for a sub-batch another stage's graphs wrote) but keeps its column budget: passes of `sr.unit` tokens cycle
through `NC` columns at `col0 = (pos / unit) % (NC / unit)`, and the position the graphs see is the run's own, not the
column's.  That is what lets a prompt longer than the window (and longer than the columns, for that matter) run at all,
and it is why `relay_write_in` takes the ABSOLUTE source position: producer and consumer derive the same column from
it, so the payload's offset is 0 on both sides.  `set_pass_cap()` narrows a stage's pass without touching its columns -
a stage that reads another's candidate tensor in place must keep the columns that tensor spans.

**What a staged run refuses**, each with a message rather than a wrong answer: `--serve` (conversation reuse -
snapshot / park / restore - is per-instance and a staged conversation spans several instances); DSpark (a rejected
draft overwrites positions a stage already handed over); a stage that shares weights but lands on a different card
(ggml hands the host an iGPU device pointer that the dGPU's copy engine cannot read, so use `--relay-copy-weights`);
a pass that would overrun a stage's rings (the per-stage cap is the LAST stage's - the only range that ever sees the
whole prefix - because an earlier stage's rings hold only its own range's output); a `--layer-split` the geometry
cannot reach (the reader's rule, above - the set it names is the answer); an `--engram` split whose **first** boundary
cuts above an Engram layer (see below); and a config that asks a stage to hold rings it does not run (`cfg.state_le != le`
aborts - the reader's rule made that shape unnecessary, and a stage holding a ring nobody reads is a stage the rule was
not understood by).

**Engram tables follow the range, and the token ring follows the stage.** The Engram tables are GBs of host memory and
only the range that holds an Engram layer reads them, so a stage opens them only if its own `[lb, le)` holds one
(`im.eng_slots` = the slots in range, passed to `rows`/`start` so a stage cannot even ask for a row of a layer it does
not run).  At the witness's boundary (20) the whole Engram pair sits in stage 0, and stage 1 maps none of it.  Note the
consequence for a split that DOES cut between Engram layers: the rolling token ring (`set_token`) is fed inside
`upload_pass`, which only stage 0 runs (`begin()` errors on a later stage by design), so stage 0's ring would be the
only one - and a stage that hashes its own window against a ring it never fed would read garbage rows.  The FIRST
boundary therefore has to sit above the LAST Engram layer (this model: 14), which `ds41_generate` enforces as its own
rule on top of the caches' set.  The interaction is worth knowing before anyone proposes a four-card plan: {8, 14, 20}
is a cache-legal split whose first boundary is 8, so with `--engram` it is refused and 20 is the ONLY boundary this
checkpoint has - two stages is the ceiling for a staged DS4.1 run that runs the real model.  Dropping `--engram` (an
ablation, not a serve config) lifts that limit to the four-stage {2, 8, 14, 20}.

**The tier needs no cross-device transport** and that is a design fact, not a simplification: at this engine's seam
the MoE tier is host-in / host-out (`attn_router` returns `fn` on the host, `finish` takes `routed_sum` on the host),
so it does not care which card ran the layers between its calls.  What it DID need is device pinning - it used to run
on the ambient device, always 0, because the dense half was its only GPU user - so `Ds41MoeConfig::device` plus a
`TierDev` RAII guard at every CUDA seam makes it name its own card even while another stage sets the ambient device.

**Where to read it**: the arithmetic gate is `tools/gfx906/test_split_geometry.{hpp,cpp}` (plain g++, no ggml, no GPU:
the reader's rule, the candidate rule, the payload size, stated against a copied geometry - and re-derived by scanning
the stages, for every 1-, 2- and 3-boundary cut), wired into
`tools/gfx906/gate-device-split.sh`; the runtime witness is `tools/gfx906/run-p5.3b-ds41-handoff.sh` (five runs: two
unsplit controls, one `--gpus 0,0` same-card split, two `--gpus 0,1 --relay-copy-weights` twins, verdict calibrated
against the twin pairs' own agreement, exactly as GLM's witness does).

**One limitation, stated where it can be seen: the caches allow a four-stage plan, the Engram add allows only two.**
The caches' set for this checkpoint is {2, 8, 14, 20}, and the even split at **20** puts 20 layers on each card - about
half the 3 GB of per-layer weights and half the compute, which is a real second stage rather than the 3-layer tail this
file's first revision believed was the only option.  The plan {8, 14, 20} is equally cache-legal and would drive four
cards, but its first boundary is 8, below the last Engram layer (14), and only stage 0 feeds the Engram token ring - so
with `--engram` (every real-model run) **20 is the only boundary there is**, and two stages is the ceiling.  Lifting
that means carrying the rolling token ring across a boundary (or feeding it on every stage), which is P5.4/P5.5 work
like the tier split, not a flag.  What is NOT on any roadmap and should stay off it: a boundary that cuts between a
cache writer and its readers.  The relay moves per-token state; feeding a successor's rings from a producer's compressed
KV / index keys is a per-pass copy of a growing cache, not 80 KiB a token, and the reader's rule refuses such a
boundary instead of faking one that happens to fall on a source layer.  The gate keeps the set honest: if a
checkpoint's source lists move, `split_valid_boundaries()` moves, the awk cross-check in `gate-device-split.sh` step 3
moves, and `test_plan_gate` fails until the witness's constant moves with them.

---

## 7. The ggml buffer-type allocation blocks — one ctx holds ALL dense layers

Yes for both: one ggml context per weight set, allocated into ONE buffer of the backend's DEFAULT buffer type, covering every decoder layer's dense tensors (routed experts skipped — the tier owns them). A layer split therefore needs either two ctx+WStore/weight sets (one per backend, which is what the native GpuStage does, generate.cpp:1106-1115) or per-tensor buffer placement (the manual ggml_backend_buft_alloc_buffer + ggml_backend_tensor_alloc pattern already used for the input/output spans).

tools/glm/glm_dense.cpp — the CPU branch binds every tensor onto the GGUF mmap (:176-197, `ggml_backend_cpu_buffer_from_ptr` + `ggml_backend_tensor_alloc` per tensor); the off-CPU branch (:199-229) is the one at ~line 210:

    for (size_t s = 0; s < model.size(); ++s)
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip_weight(ti.name, false, skip_experts, n_layer)) continue;
            ...
            ggml_tensor * t = ggml_new_tensor(w.ctx, ty, (int) ti.shape.size(), ne);
            ggml_set_name(t, ti.name.c_str());
            w.t[ti.name] = t;
        }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
            w.ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buf) { err = "cannot allocate the weight buffer on the backend"; return false; }
    w.bufs.push_back(buf);

The only registry/buft calls are `ggml_backend_get_default_buffer_type(backend)` (:210, and :347 for graph allocators, :883 LoRA, :898 state, :997 snapshot) and, for the hand-placed spans, `ggml_backend_buft_get_alignment` / `ggml_backend_buft_alloc_buffer` (:903-951). No per-tensor buft override exists. State: `im.sbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.sctx, ggml_backend_get_default_buffer_type(im.backend));` (:898) — one buffer for x_state + routed_sum + every layer's KDA S/conv/MLA cache/indexer pools (:813-857).

tools/v41/engine/ds41_dense.cpp — same shape inside the file-local load path, at :318:

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
            w.ctx, ggml_backend_get_default_buffer_type(im.backend));
    if (!buf) { err = "cannot allocate the dense weights on the backend"; return false; }
    w.bufs.push_back(buf);

(with per-tensor conversion/requant upload right after, :321-350). Persistent per-layer state uses the backend-taking variant: `im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);` (:1103-1105) covering all layers' raw/comp/idxk/compressor state (:1080-1099), and `im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend);` in `alloc_cols` (:975) for x_state/pre_state/hap/post_f/comb_f/shexp/o_*/routed/i_* . Graph allocators are again default-buft: `ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend))` (:996, :1128-1129, :1162).

Implication for the port (both engines): ggml_backend_alloc_ctx_tensors_from_buft places ALL tensors of one ctx in one buffer of one buft; graphs are then computed on that single backend. To run layers 0..K-1 on backend A and K..N-1 on backend B you must split at load time into two WStore contexts (filter by the block prefix, the way `skip_weight` already filters by layer, glm_dense.cpp:159-164) plus a second state ctx on B (glm sctx / ds41 sctx+nctx split), instantiate a second ggml_backend, second gallocr per backend, and insert the boundary copy in the generate loops at the seams named in item 6. The activation you copy is F32 hc-wide residual (D*HC floats/token in both engines; GLM also has only x_state across the seam since the attn/finish graphs are per-layer already; ds41 additionally needs pre_state and its per-layer KV/indexer state relocated, not copied, to B).

---

## Ambiguities flagged

- "sync the next stage waits on" has two true readings in the native code: strict serial path = host cudaStreamSynchronize on the producer's stream (verify.cpp:2464 -> :2485); pipelined path = producer records `ev_done_` (verify.cpp:3513) and the single host thread polls (verify.cpp:3620-3638) before launching the successor's graph — in both readings the dependency is brokered by the host, and the mapped/pinned buffer is the only shared object. I found no cudaEventRecord/cudaStreamWaitEvent pair that spans two stage streams.
- Decode and prefill relays differ in content: prefill hands on ONLY the residual R (T x D, D = HC*N, prefill.cpp:118/:4179), decode hands on R + bo + inj (verify.cpp:1776-1778). Reading: the decode mixer needs the "pending write folded into the first read" to be bit-exact across the boundary (verify.hpp:179-181); the prefill path evidently does not carry that state across the boundary. If the ported engines keep their per-layer finish self-contained (they do — finish writes complete x_state), the prefill pattern (residual only) is the closer model for them.
- `hand_in_` in prefill is used both ways in the literature: it is documented as host pinned (prefill.hpp:204) and the H2D memcpy at :2207 confirms host-pointer usage; in verify the equivalent pointer is always the device pointer of mapped memory (verify.hpp:183). Do not conflate the two when copying the pattern.