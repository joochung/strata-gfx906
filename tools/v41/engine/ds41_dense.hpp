// tools/v41/engine/ds41_dense.hpp - DeepSeek-V4.1-Flash (`deepseek41`): the dense (non-routed-expert) half.
//
// Math = DeepSeek's inference/model.py (tools/v41/engine/ds41_golden.py runs it on the GGUF's own weights; that is
// this file's gate).  The structure follows tools/ds4/ds4_dense (ggml graphs, CPU or CUDA backend, per-layer graphs
// built once and reused), but V4.1 differs enough that it is its own class:
//
//   - one PASS = n consecutive tokens at positions pos0..pos0+n-1 (n = 1: decode; n = a prompt chunk: prefill);
//     every graph is the same code for both (state rings are read as [ring | the pass's rows] and written back)
//   - hyper-connections single pass: a sublayer's hc_mixes give the NEXT sublayer's pre-mix (pre_state carries the
//     attention pre-mix from layer l's FFN to layer l+1)
//   - no per-head q norm; window K/V ring of 128; compressed KV (ratio 2 / 1) only on kv source layers, read by the
//     later layers of the same ratio; indexers on index source layers, their top-k reused by the layers after them;
//     layer `cand_layer` (20) picks 2048 candidate blocks of 8 that the later indexers rank inside
//   - attention = each query's own key set (its window + its top-k compressed rows) gathered and attended in one
//     flash-attn with the query as the batch axis (sinks added)
//   - Engram at layers 1 / 14: n-gram hash ids (host, from the token history) -> rows read from the HF shards
//     (Ds41Engram) -> wkv -> per-hc gated add, before the block
//
// Per layer per pass:  attn_router(l) -> [routed experts on fn] -> finish(l).  Then logits(...).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "ds41_geometry.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct Ds41DenseConfig {
    ggml_backend_t backend = nullptr;   ///< NULL = create and own a CPU backend
    int     n_threads = 8;
    int64_t ctx_max = 65536;            ///< positions the caches hold (compressed rows = ctx_max / ratio)
    int     max_pass = 1;               ///< tokens a pass may carry (>= 1; prefill chunk size = the sub-batch size S)
    /// Layer-major prompt batch (0 = off): the state tensors hold this many token columns and a prompt of up to
    /// `prompt_batch` tokens is walked layer by layer, one dense sub-batch of `max_pass` tokens at a time, with the
    /// experts run ONCE per layer over the whole batch (begin_batch / attn_router_span / finish_span).
    int     prompt_batch = 0;
    bool    skip_routed_experts = true; ///< do not load the routed experts (the tier owns them)
    /// Directories holding the HF shards with the Engram tables (model.safetensors.index.json in the first).
    /// Empty = no Engram (its add is skipped: a debugging mode, the model is then not the model).
    std::vector<std::string> engram_dirs;
    int     engram_threads = 32;        ///< parallel row reads (the disk's queue depth; cold peaks at 32-64)
    /// KV caches (window ring, compressed rows, indexer keys) element type: GGML_TYPE_F32 / BF16, or -1 = auto: BF16
    /// with act_quant (every cached value is then an FP8/FP4 grid point times a power-of-two / E4M3 scale: <= 7
    /// significant bits, exact in BF16 - half the VRAM of F32 at 512K), F32 without.
    int     kv_type = -1;
    /// model.py's activation rounding (ds41_ops.hpp): window K FP8, compressed KV FP4/E4M3, indexer q/k FP4.
    bool    act_quant = true;
    /// --kv-fp4 (needs act_quant, CUDA): the compressed KV as NVFP4 and the indexer keys as MXFP4 - their fake-quant
    /// grids exactly (e2m1 x E4M3 per 16 / x 2^k per 32), packed bit for bit by the vendored ggml-cuda's set_rows
    /// (cpy-utils.cuh *_ds41_block): 4.5 / 4.25 bits a value instead of BF16's 16 - ~3.5x less KV VRAM at long ctx.
    bool    kv_fp4 = false;
    /// --serve: start with max_pass (S) state columns and let set_columns() grow them for a layer-major batch and give
    /// them back after it (the prompt_batch columns are ~2.6 GiB at 8192 - expert slots while decoding).
    bool    elastic_cols = false;
    /// ---- P5.3b (mirrors GlmDenseConfig's P5.3 fields): the STAGED RUN.  A stage owns the layer range [lb, le) and
    /// nothing else.  Defaults: the whole model, which is exactly the single-stage construction of the last release,
    /// byte for byte.  With a range that does not start at 0 the token embeddings are not needed and their host rows
    /// are not mapped; with one that does not end at n_layer the LM head and output_norm are not loaded, so no stage
    /// but the last holds them.  Per-layer state (window ring, compressed KV, indexer keys, the ratio-2 compressor
    /// state, the routing bias) is allocated for this range's layers only, and the layer graphs run only over this
    /// stage's layers; the relay of tools/gfx906/relay_staging.hpp hands the state over at a boundary.
    int64_t lb = 0, le = 0;   ///< le 0 = n_layer (filled in from the GGUF when not given)
    /// P5.3b: the layer ONE PAST the persistent rings this stage owns - the boundary rule's state_end: this stage's
    /// range plus the source layer(s) its SUCCESSOR's first layer reads (a boundary at layer 21 leaves the successor
    /// reading layer 20's compressed KV / index keys, so stage 0 holds layer 20's rings).  Must be >= le; 0 = the
    /// stage's own range, which is correct only when the successor starts at its own kv AND index source, and for the
    /// last stage.  A driver that knows the whole split computes it with Ds41Dense::stage_state_end().
    int64_t state_le = 0;
    bool is_first = true, is_last = true;
    /// P5.3b (set by build_stages): this instance is one stage of a staged run.  A range that is not the whole model
    /// implies it, so callers rarely set this; it exists for the (rare) case of a stage whose range IS the model,
    /// which still needs the staged pass-column window and the state region the boundary rule gives it.
    bool staged = false;
    /// P5.3b (set by build_stages, not by callers): read the geometry and build the state and graphs but take NO weight
    /// tensors of our own; the owner hands its own over immediately before that init runs.
    bool share_weights = false;
};

class Ds41Dense {
public:
    struct Impl;
    Ds41Dense();
    ~Ds41Dense();
    Ds41Dense(const Ds41Dense &) = delete;
    Ds41Dense & operator=(const Ds41Dense &) = delete;

    bool init(const std::string & gguf, const Ds41DenseConfig & cfg, std::string & err);
    const strata::v41::Ds41Geometry & geom() const;
    /// Blocks present in the file (all 40 for the real model; a prefix for a partial test file).
    int64_t n_layer() const;
    ggml_backend_t backend() const;

    /// ---- P5.3b: the staged run.  The stage this instance runs (cfg.lb/cfg.le, or the whole model for the default
    /// config) and whether it owns the LM head.  With no range given these answer like a single-stage engine.
    int64_t stage_begin() const;
    int64_t stage_end() const;
    int64_t stage_layers() const;    ///< the layers THIS stage owns (the full model's count when unsplit)
    int64_t n_layer_total() const;   ///< the model's layer count, whatever the stage
    bool is_last_stage() const;
    /// Does this instance own its weights, or did build_stages share another's with it?  A shared instance must be
    /// destroyed BEFORE the instance that owns the weights.
    bool weights_shared() const;
    /// P5.3b: can a split with boundaries at `ends` (one-past-the-last-layer of every stage but the last, ascending)
    /// be run at all, given the model's kv / index source layers?  The rule build_stages enforces is THE READER'S
    /// SIDE of the shared caches: every ring a stage's layers read must have been written by a layer THAT STAGE runs,
    /// because a ring is persistent GPU state the successor reads by pointer inside one graph and the relay moves
    /// per-token state only.  Equivalently: every boundary is a layer of split_valid_boundaries().  Pure - geometry
    /// only, no allocation - so a caller can validate its --layer-split before loading anything.  On false, `err`
    /// says which boundary, why, and what the legal set is.
    static bool split_boundaries_ok(const strata::v41::Ds41Geometry & g, const std::vector<int64_t> & ends,
                                    std::string & err);
    /// Build the STAGES of a split run on this instance's weights instead of loading the model once per stage (the
    /// GLM engine's build_stages, tools/glm/glm_dense.cpp, mirrored): a stage whose cfg.backend IS this instance's
    /// borrows this instance's weight tensors (one load, one copy); a stage on another card cannot borrow tensors that
    /// live in the first card's memory, so it loads its own range onto its own backend.  The ranges must tile the
    /// model's layers in order and each cfg's is_first/is_last must match its position.  This instance must outlive
    /// every stage that borrowed from it.  A null entry is created; on failure `err` says which stage and why.
    bool build_stages(std::vector<std::unique_ptr<Ds41Dense>> & stage, const std::vector<Ds41DenseConfig> & cfg,
                      std::string & err);
    /// P5.3b: THE CACHE SPLIT - EVERY layer a stage boundary may sit at, ascending.  A layer reads the compressed KV /
    /// index keys of the LATEST source at or below it (Ds41Geometry::kv_src / idx_src), and those rings are persistent
    /// GPU state: the relay hands over x_state / pre_state / the candidate column PER TOKEN and has no way to move a
    /// cache, so a RING LIVES ON THE CARD OF THE STAGE THAT RUNS ITS WRITER - and therefore every READER of a ring has
    /// to sit in that same stage.  K is valid iff for every m >= K with ratio[m] > 0:
    ///     kv_src(m) >= K  AND  idx_src(m) >= K  AND  (m ranks inside cand_layer's blocks  =>  cand_layer >= K).
    /// The stage-0 side is free (kv_src(m) <= m < K always), and kv_src / idx_src are monotone, so the scan bites at
    /// the boundary layer itself: a valid boundary is a layer that is BOTH a kv and an index source (plus, vacuously,
    /// a boundary above a layer that runs no attention at all).  For the shipped 40-layer build the set is
    /// {1, 2, 8, 14, 20} - and the CLI's floor of 2 leaves {2, 8, 14, 20}, which INCLUDES the even split at 20.
    /// (A previous revision of this file derived a FLOOR of 37 from the writers instead of the readers; a boundary at
    /// 37 puts the tail [37,40) on a card whose layers read layer 20's KV and layer 36's index keys, and build_attn
    /// aborts on exactly that.  The per-token state crosses freely at any valid boundary; the caches never do.)
    /// Pure - geometry only, so a caller can check a --layer-split before allocating anything, and the witness can
    /// name the legal set without a model load.
    static std::vector<int64_t> split_valid_boundaries(const strata::v41::Ds41Geometry & g);
    /// One boundary of that set, as the predicate itself (the enumerator and the messages are both written from it).
    static bool split_boundary_valid(const strata::v41::Ds41Geometry & g, int64_t K);
    /// The same set as one line of text for a message: "2, 8, 14, 20", or "(none)" for a model that cannot split.
    static std::string split_valid_set_text(const strata::v41::Ds41Geometry & g);
    /// Snap a planned boundary list DOWN to the legal set: each planned boundary takes the nearest legal layer at or
    /// below it that still leaves its stage at least one layer.  `snapped` comes back ascending with positive stage
    /// widths, or false comes back with `err` naming the legal set (a stage that would collapse to zero layers has
    /// nowhere to snap to - the plan needs fewer stages, not a different layer).
    static bool split_snap_to_valid(const strata::v41::Ds41Geometry & g, const std::vector<int64_t> & planned,
                                    std::vector<int64_t> & snapped, std::string & err);
    /// P5.3b, the CANDIDATE exception (only the engine calls this, and only for a SAME-device split that puts
    /// cand_layer's producer and its readers in different stages).  Point this stage's graphs at ANOTHER stage's
    /// candidate tensor instead of its own: `src` is that stage's instance, its tensor is read in place (one card, so a
    /// graph may point at it - no copy, and the ids are the producer's own bits).  The stage then owns no candidate
    /// region of its own, so relay_slab_bytes() drops by relay_cand_bytes(geom()): call this BEFORE sizing the
    /// staging.  Set it on EVERY stage that reads the producer's column, not just the next one.  `src` must be a stage
    /// of the same model on the same backend, and must outlive this stage.
    void relay_cand_from(Ds41Dense * src);
    /// The bytes per token of the relay payload.  A boundary carries the state a successor's first layer reads: the
    /// x_state token column (D * HC * 4) and the pre_state column (HC * 4), plus - when THIS stage produced the
    /// candidate blocks the successor's indexers rank inside (its range holds geom().cand_layer) - the cand column
    /// (CB * 4).  The same number for every stage that produced (or did not produce) the candidates, which is what
    /// makes a boundary a small number of contiguous copies.
    size_t relay_slab_bytes() const;
    /// The boundary hand-off OUT of a producer stage / IN to a consumer stage.  Each payload tensor's pass columns
    /// [0, n) start at byte 0, so each region is ONE contiguous copy; the regions are laid out in
    /// relay_slab_bytes()'s order (x_state, pre_state, cand when relay_carries_cand()).  relay_read_out() copies the
    /// producer's finished state into the staging slot ~dst~, synchronizing the backend first (the native relay's
    /// rule: the producer's stream completion observed by the HOST before the successor is launched - no
    /// cross-device event anywhere).  relay_write_in() uploads the slot straight into this stage's state region.
    bool relay_read_out(int n, void * dst, size_t dst_bytes);
    /// src_pos0: the ABSOLUTE position the run is feeding (the graphs read it for RoPE and the index maps; the columns
    /// the payload covered are derived from it and this stage's window).  0 = the first unit of a prompt.
    bool relay_write_in(int n, const void * src, size_t src_bytes, int64_t src_pos0 = 0);
    /// P5.3b: does this instance's relay payload include the candidate column?  True unless it was told the
    /// candidates come from another stage's tensor in place (relay_cand_from).  The ENGINE owns this decision, because
    /// only it knows which stages share a card: it must match relay_slab_bytes() on both sides of a boundary, which
    /// build_stages checks.  A stage that produced the candidates writes the region; a stage that will be handed them
    /// reads the region; a stage that reads another instance's tensor in place has no region at all.
    bool relay_owns_cand_region() const;
    /// Does this stage's relay payload carry cand_layer's candidate blocks?  True iff the stage's range owns
    /// geom().cand_layer (the only layer that writes them) and a later layer reads them (it does, in this model).
    /// A property of the GEOMETRY, not of the range, so it is the same answer on every stage - which is what lets a
    /// boundary be one fixed-size copy.  Also readable before any stage exists: relay_cand_bytes().
    bool relay_carries_cand() const;
    /// The same question of a geometry alone (no instance), for an engine that sizes the staging before it builds the
    /// stages: the cand bytes a token's payload carries (0 when the model has no candidate blocks).
    static size_t relay_cand_bytes(const strata::v41::Ds41Geometry & g);
    /// P5.3b: the Engram slots this stage's layer range holds (its share of geom().engram_layers), in file order - the
    /// order Ds41Engram::rows() returns them in.  Empty when the range holds none (a stage on a second card then reads
    /// NO Engram rows at all) and every slot when unsplit.
    std::vector<int64_t> engram_slots() const;
    /// P5.3b: CAN a legal stage boundary separate cand_layer from a layer that ranks inside its blocks?  That, and not
    /// "does the model have candidates", is what decides whether the payload carries the column: it asks whether any
    /// member of split_valid_boundaries() sits strictly above the producer and at or below a reader.  On this model it
    /// cannot answer yes, and not by accident: validity already requires cand_layer >= K for every boundary K that
    /// leaves a reader above it, so producer and readers are in one stage at EVERY legal boundary (20, here: the writer
    /// and its readers 24/28/32/36 all land in stage 1).  A model whose cache lives OUTSIDE the kv / index source lists
    /// is the one that could separate them, and its payload carries the 8 KiB a token.  Pure - geometry only - because
    /// the engine sizes the staging before any stage exists.
    static bool cand_can_cross(const strata::v41::Ds41Geometry & g);

    /// Start a pass: tokens tids[0..n) at positions pos0.., n <= max_pass.  Embeddings + Engram rows (the hash
    /// needs the 3 tokens before pos0: the history is kept by position, so a re-decoded position overwrites it).
    bool begin(const int * tids, int n, int pos0);
    /// Layer-major prompt batch (cfg.prompt_batch > 0): the whole batch's inputs are prepared once - embeddings,
    /// positions, window / compressed / indexer index maps, Engram rows and the initial x/pre state - into the state
    /// tensors' columns 0..B-1; then attn_router_span / finish_span walk layer il's sub-batches in order.  B <=
    /// cfg.prompt_batch (the state columns) and B <= ctx_max - pos0.  Every cross-layer dependency of the sequential
    /// chunk path (the per-layer KV ring / compressed rows / indexer state advances in sub-batch order; layer l's
    /// residual and pre-mix are the columns the sub-batch used) is preserved, so the outputs match --prefill-chunk.
    bool begin_batch(const int * tids, int B, int pos0);
    /// Dense half of layer il for the batch's sub-batch starting at column col0 (n tokens, col0 % max_pass == 0):
    /// attention, hc, router, shared expert + the Engram add.  ids / wts / fn land in the batch buffers (batch_ids /
    /// batch_wts / batch_fn) at column col0.  Uses (and advances) layer il's own KV state in sub-batch order.
    bool attn_router_span(int il, int col0, int n);
    /// Encoder-only skip (enc-skip): build and run ONLY the persistent-KV half of layer il for the batch's sub-batch
    /// at column col0 - hc_pre / attn_norm, the raw window K/V ring write and, on a kv source layer, the compressed-KV
    /// row + indexer-key row + ratio-2 compressor state (all the state a LATER query can read across positions).  No
    /// attention output, no router, no shared expert, no hc_post / finish, no indexer selection / candidates: nothing
    /// that only ever flows into this layer's own output columns.  For prompt tokens whose layers past the replay
    /// window are never read (the model's cross-position reach is the 128-token window per layer, so the last
    /// `enc_replay` tokens are all that matters).  Writes no host ids/wts/fn, so batch_ids()/batch_wts()/batch_fn()
    /// are left untouched at these columns.
    bool attn_router_span_kv(int il, int col0, int n);
    /// The batch's host buffers filled by attn_router_span: ids / wts are n_used * B (token-major), fn is n_embd * B.
    const int32_t * batch_ids() const;
    const float * batch_wts() const;
    const float * batch_fn() const;
    int batch_size() const;
    /// finish for the batch's sub-batch at column col0: x_state[.., col0+n) = hc_post(shexp + routed, hap, post_f,
    /// comb_f).  `routed_host` is n_embd * B (token-major); call once per layer, before the sub-batches' finishes.
    bool upload_routed(const float * routed_host, int B);
    /// The routed-sum input tensor's device memory (columns of n_embd floats; null on the CPU backend): the tier can
    /// write a layer-major batch's routed sums straight into it (run_chunk_dev) instead of upload_routed's host trip.
    float * routed_dev();
    bool finish_span(int il, int col0, int n);
    /// STRATA_DENSE_PROF=1: print + reset the batch dense split (build / alloc / compute / Engram / readback) - diagnostic
    void prof_report(const char * tag);
    /// Give the prompt passes' compute buffer back to the device (rebuilt by the next multi-token pass): the deferred
    /// VRAM expert cache opens into it after the prompt.
    void release_pass();
    /// Called when a graph's buffer cannot be allocated; return true if it freed device memory (the allocation is
    /// then retried once).  E.g. the expert tier's prestage region.
    void set_oom_hook(std::function<bool()> hook);
    /// Selection-only routing bias for layer il (n_expert floats): added to the top-k score, not to the mixing weights.
    /// Soft pruning (-penalty on cold experts) / cache-aware routing (+bonus on VRAM-resident ones).  Default 0.
    void set_route_bias(int il, const float * bias);
    /// Layer il of the current pass: Engram (layers 1/14), attention, hc, ffn_norm, router, shared expert.
    /// ids / wts: n * n_used (token-major); *fn_host: n * n_embd floats (the routed experts' input, valid until the
    /// next call).  `fn_dev` (optional): the same activations on the backend.
    bool attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev = nullptr);
    /// Prefetch hint for layer il of a one-token pass (call before attn_router(il)): the layer's router applied to
    /// rms_norm(hc_pre(state, pre)) * ffn_norm.weight, + the selection bias, ranked; top_ids gets n_top (<= 16).
    /// A hint only: the result does not depend on it.
    bool predict(int il, int * top_ids, int n_top);
    /// predict() for EVERY column of the current pass (call before attn_router(il)): top_ids[t * 16 + i] = token t's
    /// i-th predicted expert of layer il (16 per token).  For the multi-token verify pass's expert prefetch; a hint only.
    bool predict_multi(int il, int * top_ids);
    /// routed_sum: n * n_embd floats, the routed experts' weighted sum (host).
    bool finish(int il, const float * routed_sum);
    /// Decode (n == 1) fusion: finish(il) and predict(next_il) in ONE cached graph - the router prediction for the next
    /// layer reads the state finish just wrote, so the result is identical to finish() then predict(), with one graph
    /// launch / sync / readback less per layer.  top_ids gets the next layer's top n_top (<= 16) predicted experts.
    bool finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top);
    /// finish_predict + the routing predictions of the `ahead` layers after next_il, from the same state, in the same
    /// graph: ahead_ids[a * 16 ..] = layer next_il + 1 + a's top 16 (-1 past the last layer).  For speculative
    /// file-tier reads; a hint only.
    bool finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                              int * ahead_ids);
    /// Logits of rows row0..row0+nrows-1 of the pass after the last layer's finish: nrows * n_vocab floats.
    bool logits(int row0, int nrows, float * out);
    /// Conversation reuse: every cache is position-indexed except the window ring and the ratio-2 compressor state
    /// ring, so a snapshot is those rings (~11 MB) + its position; restore() rewinds to it (decoding continues at
    /// snapshot_pos()).  Positions past a rewind point are overwritten when decoded again (rows past a query's
    /// visible count are never attended).
    bool snapshot(int pos);
    bool restore();
    int snapshot_pos() const;
    /// Serve: keep a conversation through an UNRELATED request.  An unrelated request writes its own positions from 0, so
    /// it overwrites the start of every position-indexed cache (compressed KV, indexer keys) and the snapshot rings.
    /// park(n) copies the snapshot rings + the first n positions' rows of those caches (and their spare rows) to host
    /// memory; unpark() writes them back and re-arms the snapshot, so restore() resumes the parked conversation.  Valid
    /// only while no request has written past position n since park (the caller checks).
    bool park(int64_t positions);
    bool unpark();
    void drop_park();
    bool parked() const;
    /// Debug / gate taps: the hc state after the last finish (n * hc * n_embd floats, token-major) and the pre mix.
    bool read_state(std::vector<float> & x, std::vector<float> & pre);
    /// The hc state of state columns [c0, c0 + n) (token-major, n * hc * n_embd floats) - a layer-major batch's columns
    /// after finish_span (the serve DSpark seed reads its main_hidden there instead of re-running the tail as passes).
    bool read_state_cols(int c0, int n, std::vector<float> & x);
    /// Give this stage a pass cap of its own, after init.  P5.3b's STAGING limits a pass (one pass's payload goes
    /// through one slot), not the state tensors: the state columns stay the run's size S, so a stage that reads another
    /// stage's tensor in place spans the same columns, and a staged run does not allocate the prefill chunk's scratch.
    /// Every cached graph and allocator is dropped (they remember the old width) and rebuilt lazily.  `n` <= the
    /// instance's column count, or it is ignored.
    void set_pass_cap(int n);
    /// Resize the state columns to max(S, min(nc, prompt_batch)) between passes (drops every cached graph); columns()
    /// is the current count.  A no-op when unchanged.
    bool set_columns(int nc);
    int columns() const;
    /// Gate tap: an index source layer's top-k rows of the current pass (n * K ints, token-major; K = min(cap, top-k))
    /// and their visibility mask; empty for other layers.
    bool read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K);
    /// The x state after begin (+ the Engram add of the layer `il`, when il is an Engram layer) - gate tap.
    const std::string & last_error() const;
    /// A loaded weight on the backend (nullptr if absent) - for sharing the trunk head with the DSpark draft stack.
    ggml_tensor * weight(const std::string & name) const;

private:
    std::unique_ptr<Impl> p_;
};
