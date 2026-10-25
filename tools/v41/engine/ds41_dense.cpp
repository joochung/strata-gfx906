// tools/v41/engine/ds41_dense.cpp - see ds41_dense.hpp.  Line references are to DeepSeek's inference/model.py.
#include "ds41_dense.hpp"

#include "ds41_engram.hpp"
#include "ds41_ops.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#endif   // ggml_cgraph::uid (graph reuse)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <thread>

#include <strings.h>

using namespace strata;
using namespace strata::v41;

namespace {

constexpr float NEG_INF = -INFINITY;

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

bool env_on(const char * k, bool dflt) {
    const char * e = std::getenv(k);
    return e ? std::atoi(e) != 0 : dflt;
}

// The decode graphs are built once and reused; stamping a uid lets ggml-cuda reuse its captured CUDA graph.  One
// allocator serves every cached graph (they run one at a time); when it has to grow, every graph allocated before
// is re-allocated on its next use (the epoch), as tools/ds4/ds4_dense.cpp alloc_graph does.
struct Allo {
    ggml_gallocr_t a = nullptr;
    uint64_t epoch = 0;
    std::map<const ggml_cgraph *, uint64_t> graph_epoch;
    std::set<const void *> compute_bufs;
    ~Allo() { if (a) ggml_gallocr_free(a); }
    bool alloc(ggml_cgraph * g, bool reuse) {
        auto it = graph_epoch.find(g);
        if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch) return true;
        if (it != graph_epoch.end() && it->second != epoch) {
            auto computed = [&](const ggml_tensor * t) { return t && t->buffer && compute_bufs.count(t->buffer); };
            auto unbind = [&](ggml_tensor * t) {
                if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
            };
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                ggml_tensor * t = ggml_graph_node(g, i);
                for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
                unbind(t);
            }
            for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
        }
        const size_t before = ggml_gallocr_get_buffer_size(a, 0);
        if (!ggml_gallocr_alloc_graph(a, g)) return false;
        if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) ++epoch;
        auto note = [&](const ggml_tensor * t) {
            if (!t || !t->buffer) return;
            if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) compute_bufs.insert(t->buffer);
            else compute_bufs.erase(t->buffer);
        };
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            note(t);
            for (int j = 0; j < GGML_MAX_SRC; ++j) note(t->src[j]);
        }
        for (int i = 0; i < g->n_leafs; ++i) note(g->leafs[i]);
        graph_epoch[g] = epoch;
        if (reuse) g->uid = ggml_graph_next_uid();
        return true;
    }
};

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    /// P5.3b: false when the tensors in `t` and the buffers in `bufs` belong to another Impl (build_stages shared
    /// them).  The owner frees; a borrower empties its own view of them and frees nothing.
    bool owns = true;
    ~WStore() {
        if (!owns) { ctx = nullptr; bufs.clear(); t.clear(); return; }
        for (auto b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "[ds41_dense] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
};

bool is_routed(const std::string & n) {
    return n.find("_exps.") != std::string::npos || n.find("ffn_exps_") != std::string::npos;
}

// P5.3b (mirrors skip_weight in tools/glm/glm_dense.cpp): what a STAGE does not load.  Everything outside the range
// it runs, and nothing of the head unless it is the last stage.  `wl`/`wle` are the stage's WEIGHT bounds, one block
// wider than its layer range on the top side (see weight_end): the layer AFTER a stage's last layer reads that
// block's ffn_norm / router / routing-bias weights in the fused finish+predict graph, so a producer loads it too.
bool skip_weight_stage(const std::string & n, bool skip_experts, int64_t n_layer, int64_t wl, int64_t wle,
                      bool last_stage) {
    // P5.3b: the boundary rule may make a stage RUN a layer above its range (the producer of the rings its successor
    // reads).  Those layers' weights are part of the stage's work, so the driver widens wle and this stays consistent.
    if (skip_experts && is_routed(n)) return true;
    if (n.rfind("blk.", 0) == 0) {
        const int64_t il = std::atoll(n.c_str() + 4);
        if (il >= n_layer) return true;                        // a block past the decoder (MTP): as before
        if (wle > 0 && (il < wl || il >= wle)) return true;    // another stage's block
    }
    if (!last_stage && (n == "output.weight" || n == "output_norm.weight")) return true;
    return false;
}

// Where a stage's block loading stops: the range it RUNS plus the one block its last layer reads ahead into,
// unless that range already ends the model.
static int64_t weight_end(int64_t n_layer, int64_t le) { return le >= n_layer ? n_layer : le + 1; }

}  // namespace

// ================================================================== Impl

struct Ds41Dense::Impl {
    std::function<bool()> oom_hook;       // set_oom_hook: frees device memory when a graph buffer does not fit
    Ds41DenseConfig cfg;
    Ds41Geometry g;
    std::unique_ptr<GgufModel> model;
    std::string model_path;               ///< P5.3b: build_stages re-opens the same model for a stage
    std::vector<int64_t> layers;          // blocks in the file (0..nl-1)
    int64_t nl = 0;
    ggml_backend_t backend = nullptr;
    bool own_backend = false, cpu = true;
    WStore w;
    /// P5.3b: whose weight tensors we borrowed (build_stages; nullptr = ours to free).  Set before any shared tensor
    /// pointer is copied in, so a half-shared Impl is either empty or fully owned - never dangling.
    Impl * w_owner = nullptr;
    bool share_weights = false;      ///< cfg.share_weights: init() takes no weight tensors of its own
    /// P5.3b: this instance's layer range [lb, le) and the WEIGHT bounds it loads (its range plus the one block its
    /// last layer reads ahead into - build_finish_predict of layer le-1 predicts layer le, and the next stage's
    /// attn_router(il) builds layer il's graph on demand, so the tensors are needed even though the RUN never
    /// executes block le).  Defaults: the whole model.
    int64_t lb = 0, le = 0, wl = 0, wle = 0;
    bool is_first = true, is_last = true;
    /// The layer range the PERSISTENT state was allocated for: [lb, state_end()) == the stage's own range [lb, le).
    /// THE READER'S RULE (Ds41Dense::split_boundary_valid) says why it can be that simple: a layer reads the compressed
    /// KV / index keys of the LATEST source at or below it, those rings are persistent GPU state, and the relay moves
    /// per-token state only - so a boundary has to leave every READER in the stage that runs its ring's writer, and
    /// every read of a ring stays inside the stage that runs it.  A stage on another card therefore never reaches a ring
    /// it does not hold, and a ring a stage does not run is simply not allocated (null), which is what makes
    /// attn_router(il) and build_attn fail loudly for another stage's layer instead of building a graph over NULL.
    int64_t state_end() const { return st_end; }
    /// Compute the state region from the stage's range and cache it (init()'s allocation loops read it many times).
    /// With the boundary rule it is exactly the range the stage runs; see the check inside.
    void compute_state_end() {
        // With the reader's rule (Ds41Dense::split_boundaries_ok) in force, every cache a stage's layers read was
        // produced by a layer of THIS stage: a boundary is a layer of the valid set, so a successor never reads across
        // one.  A stage's persistent rings are therefore exactly the layers it runs, and a config that claims otherwise
        // is a bug in the driver, not a shape to support.
        if (cfg_state_le > 0 && cfg_state_le != le) {
            std::fprintf(stderr, "[ds41_dense] stage [%lld,%lld): the run asked for state to layer %lld; a stage's "
                         "persistent rings are exactly the layers it runs\n", (long long) lb, (long long) le,
                         (long long) cfg_state_le);
            std::abort();
        }
        // A boundary that is not in the valid set would need a cache this stage does not run.  init() is where the rings
        // get allocated, so the rule is re-checked here rather than trusted from the caller - the abort names the legal
        // set, which is the message a user with a hand-built config needs.
        if (staged && le < nl && !Ds41Dense::split_boundary_valid(g, le)) {
            std::fprintf(stderr, "[ds41_dense] stage [%lld,%lld): a boundary at %lld is not a valid boundary for this "
                         "model (valid: %s); its layers would read rings the stage below them wrote\n",
                         (long long) lb, (long long) le, (long long) le,
                         Ds41Dense::split_valid_set_text(g).c_str());
            std::abort();
        }
        st_end = le;
    }
    int64_t cfg_state_le = 0; ///< cfg.state_le, kept for the consistency check in compute_state_end()
    /// The layer one-past-the-last layer that allocates persistent rings.  With the reader's rule this is the
    /// state region, which is the stage's own range; allocation loops and the weight bounds read it through this name
    /// so a future rule that widens the region has one place to do it.
    int64_t alloc_end() const { return std::max(st_end, le); }
    int64_t st_end = 0;      ///< the cached result of compute_state_end(); set before any allocation loop reads it
    /// P5.3b: does THIS stage run layer `il`?  Every per-layer entry point says so or fails loudly - a layer run
    /// twice would double-advance the state and a layer nobody runs would leave a relayed column unwritten.  The
    /// stage's run is its range, and a valid boundary guarantees no layer of a later stage reads a ring this
    /// stage's layers wrote (or the reverse), so range and rings are the same set of layers.
    bool owns_layer(int64_t il) const { return il >= lb && il < alloc_end(); }
    /// P5.3b: true once this instance is part of a staged run (build_stages sets it on every stage).  It is what makes
    /// the graph builders take the pass's column window from the impl instead of the caller's argument, and what
    /// enables window_col0(); a single-instance run never reads either.
    bool staged = false;
    /// P5.3b: the column a relayed piece of n tokens starts in.  A pass of N tokens is run piecewise by each stage
    /// inside its own window (see set_pass_cap and staged_pass), and the boundary carries the producer's WHOLE N-column
    /// block, laid out contiguously.  So the consumer's columns for the producer's columns [k, k+m) are [k, k+m):
    /// `src_col` is the offset the producer's block starts at, and a piece never crosses a consumer window because the
    /// consumer's window is the smaller one and every piece is sized to it.
    /// The column a staged pass of n tokens starting at absolute position pos runs in.  Passes of a window's worth
    /// cycle the NC columns (NC/S windows); a shorter pass (a decode step, a prompt's tail) still starts at a window
    /// boundary, chosen by the same position math, so the producer and the consumer of a boundary agree.
    int64_t stage_col0(int64_t pos, int n) const {
        const int64_t s = std::max<int64_t>(1, S);
        const int64_t nwin = std::max<int64_t>(1, NC / s);
        const int64_t c = (pos / s) % nwin * s;   // the window this position's pass runs in
        // A short pass (a prompt's tail, a decode step) starts a window but does not fill it; it may not run past the
        // allocation either, so clamp it inside the columns.  Both bounds hold because n <= S <= NC.
        return std::min(c, std::max<int64_t>(0, NC - n));
    }
    Ds41Engram engram;
    std::string err;

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, RQ = 0, SWA = 0, RAW = 0;
    int64_t IH = 0, ID = 0, ITOP = 0, NE = 0, NU = 0, NFF = 0, NC = 0, CB = 0, CBS = 0, EC = 0, ED = 0;
    int64_t cmax[3] = { 0, 0, 0 };        // compressed rows per ratio (index 1 / 2)

    // ---- persistent state (sctx / sbuf)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor * x_state = nullptr;      // [D, HC, NC]
    ggml_tensor * pre_state = nullptr;    // [HC, NC]   the next attention's pre-mix
    ggml_tensor * hap = nullptr;          // [D, HC, NC] the FFN residual (hc_post of the attention)
    ggml_tensor * post_f = nullptr;       // [HC, NC]
    ggml_tensor * comb_f = nullptr;       // [HC, HC, NC]
    ggml_tensor * shexp = nullptr;        // [D, NC]
    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
    ggml_tensor * o_ids = nullptr;        // I32 [NU, NC]
    ggml_tensor * o_wts = nullptr;        // [NU, NC]
    ggml_tensor * routed = nullptr;       // [D, NC]
    ggml_tensor * cand = nullptr;         // I32 [CB, NC]  candidate blocks of cand_layer
    /// P5.3b: the CANDIDATE relay - the boundary copy of cand_layer's blocks, used ONLY when the candidate WRITER and a
    /// candidate READER live on different cards.  `cand` is then this stage's own tensor, which the successor fills
    /// from the producer's via relay_read_out / relay_write_in; `cand_src_prev` is the PRODUCER's tensor the graphs read
    /// instead, on the stage that receives the relay.  The engine's KV rule normally keeps producer and readers in one
    /// stage (a boundary must follow a kv AND index source, and cand_layer 20 is one), so this path exists for the case
    /// where it cannot: it is a per-token top-k selection crossing cards, and reading the producer's own tensor keeps
    /// the gathered ids bit-identical to the unsplit run.
    ggml_tensor * cand_src_prev = nullptr;
    /// P5.3b: the CANDIDATE PRODUCER publishes its own tensor here (in place of publishing a relayed copy) so a stage
    /// that shares its backend can read it without a copy.  The producer ALWAYS publishes: the decision of who reads
    /// it in place is taken after init(), by whoever wires the stages, and a tensor that was never published cannot
    /// be read by a stage whose own region nobody wrote.  Valid only while the producer outlives the readers.
    ggml_tensor * cand_published = nullptr;
    /// P5.3b: the PRODUCER's pass window when its candidate column is read in place (relay_cand_from).  The writer
    /// aligns its column to its own window; this stage reads it at ITS offset, so the two must be the same size.
    /// The engine's staged run keeps every stage at one pass cap (build_stages requires the configs to agree and
    /// ds41_generate sets the same cap on all of them), so this is a tripwire, not a mode.
    int64_t cand_src_cap = 0;
    struct Layer {
        int64_t ratio = 0;
        bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false;
        int64_t kvl = -1, idxl = -1;      // the layers whose compressed KV / top-k this one reads
        int64_t eslot = -1;               // Engram slot
        ggml_tensor * raw = nullptr;      // [DH, RAW]       window K (= V), post-RoPE
        ggml_tensor * comp = nullptr;     // [DH, cmax + 1]  compressed KV (kv sources); row cmax = spare
        ggml_tensor * idxk = nullptr;     // [ID, cmax + 1]  indexer keys (kv sources)
        ggml_tensor * st_kv = nullptr;    // [DH, 2]         ratio-2 compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [DH, 2]         ... (scores)
        ggml_tensor * sel = nullptr;      // I32 [ITOP, NC]  top-k rows (index sources)
        ggml_tensor * selm = nullptr;     // F32 [ITOP, NC]  0 / -inf (row visible to the query)
        ggml_tensor * raw_snap = nullptr, * st_kv_snap = nullptr, * st_sc_snap = nullptr;   // snapshot()
        ggml_tensor * rbias = nullptr;    // F32 [NE]        selection-only routing bias (set_route_bias), 0 = model routing
    };
    int snap_pos = -1;
    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
    std::vector<ParkedRows> parked;     // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites
    int parked_pos = -1;
    std::vector<Layer> ly;

    // ---- per-pass inputs (ictx / ibuf), uploaded by begin()
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    // ---- the NC-column tensors (state columns + their per-pass inputs) in their own context / buffer, so
    // set_columns() can resize them: --serve keeps S columns while it decodes and grows to the prompt batch only for a
    // layer-major batch (~0.31 MiB a column: 8192 columns = 2.6 GiB that decode gives to expert slots instead)
    ggml_context * nctx = nullptr;
    ggml_backend_buffer_t nbuf = nullptr;
    int64_t NCmax = 0;
    ggml_tensor * i_emb = nullptr;        // [D, NC]
    ggml_tensor * i_pos = nullptr;        // I32 [NC]
    ggml_tensor * i_widx = nullptr;       // I32 [SWA * NC]   per query: rows of [ring | pass rows]
    ggml_tensor * i_wmask = nullptr;      // F32 [SWA, NC]
    ggml_tensor * i_wslot = nullptr;      // I32 [min(NC, RAW)]  ring slots of the pass's last rows
    ggml_tensor * i_nvis[3] = {};         // F32 [NC]   visible compressed rows per query (ratio 1 / 2)
    ggml_tensor * i_cslot[3] = {};        // I32 [NC]   compressed row each token writes (spare if its group is open)
    ggml_tensor * i_cpos[3] = {};         // I32 [NC]   RoPE position of that row (group start)
    ggml_tensor * i_sidx = nullptr;       // I32 [2 * NC] ratio-2 state rows per token, of [state ring | pass states]
    ggml_tensor * i_sslot = nullptr;      // I32 [min(NC, 2)]
    // upload_span's last (pos0, col0, n): its four index tensors depend only on these, not on the layer, so the 39
    // layers after the first skip identical re-uploads.  Those tiny H2D copies queued behind the chunk prestage's
    // ~1.5 GiB DMA of the next layer's experts and cost ~65 ms a layer (~2.5 s a turn).  Any other writer of the four
    // tensors (upload_inputs) or their reallocation (alloc_state) resets it.
    int64_t span_memo_pos0 = -1;
    int span_memo_col0 = -1, span_memo_n = -1;
    ggml_tensor * i_lastblk = nullptr;    // F32 [NC]   (nvis - 1) / block size: the pinned candidate block (ratio 1)
    ggml_tensor * i_eng[2] = {};          // F32 [EC * ED, NC]  Engram rows
    ggml_tensor * neg_ramp = nullptr;     // F32 [cmax1]  -i     (constant)
    ggml_tensor * blk_ramp = nullptr;     // F32 [cmax1 / CBS + 1]  i (constant)

    // ---- the pass
    int pos0 = 0, n = 0;
    int64_t pass_cap[3] = { 0, 0, 0 };
    // ---- the layer-major prompt batch (begin_batch): S = the dense sub-batch size, batch = B, span_cap per sub-batch
    int64_t S = 1, batch = 0, col0 = 0;
    int batch_pos0 = 0;                             // the batch's first position
    std::vector<std::array<int64_t, 3>> span_cap;   // [sub-batch][ratio] pass_cap
    std::vector<float> host_fn, host_wts, host_logits, host_emb;
    std::vector<int32_t> host_ids;
    const uint8_t * embd = nullptr;       // token_embd F16 rows on the mmap
    std::vector<std::vector<float>> eng_rows;
    std::vector<int64_t> eng_slots;      ///< P5.3b: the Engram slots THIS stage's range holds (its engram_layers), in
                                         /// file order - the order Ds41Engram::rows()/start() return rows in
    // The pass's Engram reads, started (async) by upload_inputs and waited at the first Engram layer's span /
    // attn_router: the whole pass's rows are read from disk while the embeddings, the init graph and the layers below
    // the first Engram layer (1) run.  Null once uploaded (or when Engram is off) - see engram_ready().
    std::shared_ptr<Ds41Engram::Batch> eng_batch;

    // ---- graphs
    ggml_context * gctx = nullptr;        // node structs of the cached (n = 1) graphs
    std::vector<uint8_t> pf_meta;         // node structs of a pass graph with n > 1 (built per pass)
    Allo allo_dec, allo_pass;
    struct GKey { int il, kind; int64_t cap, n; bool operator<(const GKey & o) const {
        return std::tie(il, kind, cap, n) < std::tie(o.il, o.kind, o.cap, o.n); } };
    std::map<GKey, ggml_cgraph *> graphs;
    std::map<int, ggml_tensor *> logits_out;   // decode head graph output, by nrows
    std::map<int, ggml_tensor *> predict_out;  // predict graph output, by layer
    std::map<std::pair<int, int>, ggml_tensor *> predict_out_n;   // predict_multi output, by (layer, pass width)
    std::map<int, ggml_tensor *> fpredict_out; // finish_predict graph output, by the predicted layer
    std::map<std::pair<int, int>, std::vector<ggml_tensor *>> fpredict_ahead; // finish_predict_ahead extra outputs

    ~Impl() {
        if (nbuf) ggml_backend_buffer_free(nbuf);
        if (nctx) ggml_free(nctx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (sctx) ggml_free(sctx);
        if (ictx) ggml_free(ictx);
        if (gctx) ggml_free(gctx);
        if (own_backend && backend) ggml_backend_free(backend);
    }
    /// Does THIS stage run the layer that writes the candidate blocks?  Its own graphs produce the column, so it owns
    /// and publishes it whichever way the readers get it.
    bool owns_cand_producer() const { return lb <= g.cand_layer && g.cand_layer < le; }
    /// Does this stage's OWN candidate column get filled from the boundary payload?  True unless the geometry never
    /// separates the column from its readers (cand_can_cross false, the shipped case) or its graphs read the producer's
    /// tensor in place (relay_cand_from, a same-card option).  The PAYLOAD number stays a function of the GEOMETRY alone
    /// so the two sides of a boundary can never disagree about where a copy ends.
    bool cand_region_owned() const { return Impl::cand_can_cross(g) && cand_src_prev == nullptr; }
    /// Can a LEGAL boundary separate cand_layer from a layer that ranks inside its blocks?  The question is now the
    /// enumerator's: a boundary at K separates them iff K is VALID, K sits strictly above the producer, and some reader
    /// sits at or above K.  A valid K always leaves the producer with its readers (validity requires cand_layer >= K for
    /// any K that leaves a reader at or above it), so for a model whose caches are the kv / index source lists this is
    /// false for EVERY plan - which is why the shipped run's slab line says `cand 0`.  Consequence, and it is worth
    /// stating where the code is read: under this rule a staged run always relays the TWO state columns and nothing
    /// else; the candidate region below is for a model with a cache OUTSIDE these lists (a hybrid SSM state, a cache
    /// keyed by something Ds41Geometry does not read yet), whose valid boundaries can put a reader above the producer.
    static bool cand_can_cross(const strata::v41::Ds41Geometry & g) {
        if (g.cand_layer < 0 || g.cand_layer >= (int64_t) g.ratio.size()) return false;
        int64_t last_reader = -1;      // a reader = an INDEX source strictly above the producer (L.uses_cand)
        for (int64_t k : g.index_sources) if (k > g.cand_layer && k > last_reader) last_reader = k;
        if (last_reader < 0) return false;
        for (int64_t K : Ds41Dense::split_valid_boundaries(g))
            if (g.cand_layer < K && K <= last_reader) return true;
        return false;
    }
    /// Does the model's routing consult cand_layer's column at all - is there an INDEXER above the producer?  (This
    /// model: yes, 24/28/32/36.)  That alone does not make the column relayable (see cand_can_cross), but it decides
    /// whether holding the column is worth any state at all.
    static bool cand_has_readers(const strata::v41::Ds41Geometry & g) {
        if (g.cand_layer < 0 || g.cand_layer >= (int64_t) g.ratio.size()) return false;
        for (int64_t k : g.index_sources) if (k > g.cand_layer) return true;
        return false;
    }

    ggml_tensor * W(const std::string & n) const { return w.get(n); }
    ggml_tensor * BL(int64_t il, const std::string & s) const { return w.get("blk." + std::to_string(il) + "." + s); }
    // the router as the prediction graphs read it: the STRATA_PRED_ROUTER copy when the loader made one, else the F32
    ggml_tensor * BLpred(int64_t il) const {
        const auto it = w.t.find("blk." + std::to_string(il) + ".ffn_gate_inp_pred.weight");
        return it != w.t.end() ? it->second : BL(il, "ffn_gate_inp.weight");
    }
    bool compute(ggml_cgraph * gf, bool cached);
    bool upload_inputs(const int * tids, int n, int pos0);
};

// ================================================================== weights

static bool load_weights(Ds41Dense::Impl & im, std::string & err) {
    WStore & w = im.w;
    const GgufModel & model = *im.model;
    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
    w.ctx = ggml_init(ip);
    auto make = [&](const TensorInfo & ti, int type = -1) {   // type: the in-memory type (default the file's)
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
        ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) (type >= 0 ? type : (int) ti.type), (int) ti.shape.size(), ne);
        ggml_set_name(t, ti.name.c_str());
        return t;
    };
    // routed experts never reach ggml (PTQ1_0 = 143 is past GGML_TYPE_COUNT); the I32 maps are read by the geometry.
    // P5.3b: on a staged run a stage also skips another stage's blocks and (unless it is the last one) the head.
    // wl/wle are the stage's WEIGHT bounds (its range plus the one block its last layer reads ahead into).
    const int64_t wl = im.wl, wle = im.wle;
    auto skip = [&](const std::string & n) {
        return skip_weight_stage(n, true, im.nl, wl, wle, im.is_last);
    };
    if (im.cpu) {
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & ts = sh.tensors();
            if (ts.empty()) continue;
            uint8_t * base = const_cast<uint8_t *>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            w.bufs.push_back(buf);
            for (const auto & ti : ts) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = make(ti);
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[ti.name] = t;
            }
        }
    } else {
        // STRATA_DENSE_REQUANT=<ggml type name> (q6_K, q5_K, q4_K, ...; quality experiment, OFF by default): every
        // Q8_0 dense matrix (attention, shared expert; output.weight unless STRATA_DENSE_REQUANT_HEAD=0) is
        // re-quantized in memory at load - rows dequantized, quantized to that type.  The file is not touched.
        ggml_type rq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_DENSE_REQUANT"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) rq = (ggml_type) t;
        const bool rq_head = !(std::getenv("STRATA_DENSE_REQUANT_HEAD") && std::getenv("STRATA_DENSE_REQUANT_HEAD")[0] == '0');
        auto requant = [&](const TensorInfo & ti) {
            return rq != GGML_TYPE_COUNT && (ggml_type) ti.type == GGML_TYPE_Q8_0 && ti.shape.size() == 2 &&
                   (int64_t) ti.shape[0] % ggml_blck_size(rq) == 0 && (rq_head || ti.name != "output.weight");
        };
        // STRATA_PRED_ROUTER=<ggml type> (q8_0, f16; OFF by default): the PREDICTION graphs (the prefetch top-16 and the
        // speculative reads' rows) read a copy of each layer's F32 router in that type (q8_0: 7.9 -> 2.1 MB a read); the
        // routing itself (build_attn) keeps the file's F32.  Only which experts are prefetched can change.
        ggml_type pq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_PRED_ROUTER"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) pq = (ggml_type) t;
        auto pred_name = [&](const TensorInfo & ti) -> std::string {   // "" = no prediction copy for this tensor
            static const std::string suf = "ffn_gate_inp.weight";
            const std::string & n = ti.name;
            if (pq == GGML_TYPE_COUNT || (ggml_type) ti.type != GGML_TYPE_F32 || ti.shape.size() != 2 ||
                (int64_t) ti.shape[0] % ggml_blck_size(pq) != 0 || n.size() <= suf.size() ||
                n.compare(n.size() - suf.size(), suf.size(), suf) != 0) return std::string();
            return n.substr(0, n.size() - suf.size()) + "ffn_gate_inp_pred.weight";
        };
        size_t rq_n = 0, rq_from = 0, rq_to = 0, pq_n = 0;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                w.t[ti.name] = make(ti, requant(ti) ? (int) rq : -1);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    ggml_tensor * tp = make(ti, (int) pq);
                    ggml_set_name(tp, pn.c_str());
                    w.t[pn] = tp;
                }
            }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!buf) { err = "cannot allocate the dense weights on the backend"; return false; }
        w.bufs.push_back(buf);
        // rows of the file's tensor `ti` (bytes `src`) converted to t->type and uploaded; returns the bytes uploaded
        auto convert_upload = [&](ggml_tensor * t, const TensorInfo & ti, const uint8_t * src) -> size_t {
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        if ((ggml_type) ti.type == GGML_TYPE_F32) std::memcpy(f.data(), src + (size_t) r * src_row, (size_t) n0 * 4);
                        else ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
            return dst.size();
        };
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = w.t[ti.name];
                const uint8_t * src = model.shard(s).tensor_data(ti);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    convert_upload(w.t[pn], ti, src);
                    ++pq_n;
                }
                if ((int) t->type == (int) ti.type) {
                    ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
                    continue;
                }
                rq_to += convert_upload(t, ti, src);
                ++rq_n;
                rq_from += (size_t) ggml_nrows(t) * ggml_row_size((ggml_type) ti.type, t->ne[0]);
            }
        if (rq_n) std::fprintf(stderr, "dense requant: %zu Q8_0 matrices -> %s, %.2f -> %.2f GB\n", rq_n, ggml_type_name(rq),
                               rq_from / 1e9, rq_to / 1e9);
        if (pq_n) std::fprintf(stderr, "prediction routers: %zu F32 copies as %s\n", pq_n, ggml_type_name(pq));
    }
    // The token embeddings are read from the GGUF mmap on the HOST (upload_inputs), never uploaded: only the first
    // stage looks tokens up, so only it maps the rows.  share_weights (a stage that borrowed the whole load) keeps the
    // owner's pointer, which is why it is not cleared here.
    if (im.is_first || im.share_weights) {
        size_t si = 0;
        const TensorInfo * te = model.find("token_embd.weight", &si);
        im.embd = model.shard(si).tensor_data(*te);
    }
    return true;
}

// ================================================================== graph helpers

namespace {

struct B {
    Ds41Dense::Impl * im;
    ggml_context * c;
    const Ds41Geometry * g;

    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * cols2(ggml_tensor * t, int64_t n0, int64_t n1, int64_t c0 = 0) const {   // columns c0.. of a [n0, *] tensor
        return ggml_view_2d(c, t, n0, n1, t->nb[1], (size_t) c0 * t->nb[1]);
    }
    ggml_tensor * cols3(ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t c0 = 0) const {
        return ggml_view_3d(c, t, n0, n1, n2, t->nb[1], t->nb[2], (size_t) c0 * t->nb[2]);
    }
    ggml_tensor * flat2(ggml_tensor * t, int64_t ne0, int64_t ne1) const {
        return ggml_is_contiguous(t) ? ggml_reshape_2d(c, t, ne0, ne1) : ggml_cont_2d(c, t, ne0, ne1);
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wt);
    }
    // hc_mixes (model.py Block.hc_mixes + kernel hc_split_sinkhorn): x [D, HC, n] -> pre [HC, n], post [HC, n],
    // comb [HC(dst), HC(src), n]
    void hc_mixes(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                  ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t HC = g->hc, nt = x->ne[2], D = g->n_embd;
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), (float) g->rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);                                  // [24, n]
        ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
#if defined(STRATA_DSV41_HC_MIX)
        // -DSTRATA_DSV41_HC_MIX=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fuse.patch: the two mixes as
        // ONE op (GGML_OP_DSV41_HC_MIXES), the same mul/add/sigmoid/scale steps bit for bit (test_ds41_hcmix).  It
        // replaces 8 of the layer graph's ~82 tiny kernels per call.  STRATA_HC_MIX_COMPOSITE=1: the composite.
        static const bool composite = std::getenv("STRATA_HC_MIX_COMPOSITE") != nullptr;
        if (!composite) {
            ggml_tensor * mix = ggml_dsv41_hc_mixes(c, mixes, v1(sc, 2, 0), v1(bs, 2 * HC, 0), (float) g->hc_eps);
            *pre  = ggml_view_2d(c, mix, HC, nt, mix->nb[1], 0);
            *post = ggml_view_2d(c, mix, HC, nt, mix->nb[1], (size_t) nt * mix->nb[1]);
            *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
            return;
        }
#endif
        ggml_tensor * m_pre = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        *pre = ggml_add(c, ggml_mul(c, m_pre, v1(sc, 1, 0)), v1(bs, HC, 0));
        *pre = ggml_scale_bias(c, ggml_sigmoid(c, *pre), 1.0f, (float) g->hc_eps);
        *post = ggml_add(c, ggml_mul(c, m_post, v1(sc, 1, sizeof(float))), v1(bs, HC, (size_t) HC * sizeof(float)));
        *post = ggml_scale(c, ggml_sigmoid(c, *post), 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
    }
    // RoPE over the last `nd` dims of each `ne0` row (model.py apply_rotary_emb: adjacent pairs = NORMAL mode).
    // ratio > 0: compress_rope_theta with YaRN; ratio 0: rope_theta, no YaRN (model.py Attention.__init__)
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t ratio, bool back = false) const {
        const bool yarn = ratio != 0;
        const float base = (float) (yarn ? g->compress_rope_base : g->rope_base);
        const float fscale = yarn ? (float) (1.0 / g->rope_factor) : 1.0f;
        const float ext = yarn ? 1.0f : 0.0f;
        // ggml multiplies by mscale = attn_factor * (1 + 0.1 ln(1/fscale)) under YaRN; model.py has no such scale
        const float attn = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / fscale)) : 1.0f;
        const float bf = yarn ? (float) g->beta_fast : 0.0f, bsl = yarn ? (float) g->beta_slow : 0.0f;
        const int n_ctx = yarn ? (int) g->orig_ctx : 0;
        const int nd = (int) g->d_rope;
        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    }
};

}  // namespace

// STRATA_DENSE_PROF=1 (diagnostic, off by default): where a layer-major batch's dense time goes - graph build, gallocr
// allocation, GPU compute, Engram wait, readback - summed per prompt and printed by prof_report()
static const bool g_dprof_on = std::getenv("STRATA_DENSE_PROF") != nullptr;
static double g_dprof[8] = {};   // build, alloc, compute, engram, readback, finish build, upload, calls
static double dprof_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_dprof_slot = -1;    // which of build/alloc/compute buckets compute() charges: set by the span callers

bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached) {
    Allo & a = cached ? allo_dec : allo_pass;
    const double tq0 = g_dprof_on ? dprof_now() : 0;
    bool ok_alloc = a.alloc(gf, cached && env_on("DS41_GRAPH_REUSE", true));
    const double tq1 = g_dprof_on ? dprof_now() : 0;
    if (!ok_alloc && oom_hook && oom_hook()) {   // something gave VRAM back: once more
        std::fprintf(stderr, "ds41_dense: graph buffer did not fit - freed the tier's prestage region, retrying\n");
        ok_alloc = a.alloc(gf, false);
    }
    if (!ok_alloc) { err = "graph allocation failed"; return false; }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
    if (g_dprof_on && g_dprof_slot >= 0) { g_dprof[1] += tq1 - tq0; g_dprof[2] += dprof_now() - tq1; }
    return true;
}

// ================================================================== the attention graph (+ router + shared expert)

// the model's activation rounding: ds41_ops.hpp's stock-op composite (~45 ops per call), or - built with
// -DSTRATA_DSV41_FQ_OP=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fake-quant.patch - the fused
// GGML_OP_DSV41_FAKE_QUANT, one kernel per call (bit-identical on CUDA and CPU: test_ds41_fq; ~3,000 fewer tiny
// kernels per decode token).  STRATA_FQ_COMPOSITE=1 in such a build: the composite.
static ggml_tensor * fake_quant(ggml_context * gc, ggml_tensor * x, Ds41Quant mode) {
#if defined(STRATA_DSV41_FQ_OP)
    static const bool composite = std::getenv("STRATA_FQ_COMPOSITE") != nullptr;
    return composite ? ds41_fake_quant(gc, x, mode) : ggml_dsv41_fake_quant(gc, x, (int) mode);
#else
    return ds41_fake_quant(gc, x, mode);
#endif
}

// The column a graph's pass reads and writes.  Unsplit, a pass starts at column 0 and col0 is the caller's; a STAGED
// stage's pass lives in its own window (relay_write_in set im.col0 when the payload landed), so the builders read the
// window from the impl rather than a parameter - a stale call site cannot then silently run a staged pass at column 0.
static inline int64_t pass_col0(const Ds41Dense::Impl & im, int64_t col0) { return im.staged ? im.col0 : col0; }

static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0_p = 0) {
    const int64_t col0 = pass_col0(im, col0_p);
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    // P5.3b: the compressed KV / index keys a layer reads belong to the LATEST source at or below it, which may be a
    // layer no stage in this run allocated state for (the model's first kv/index source is layer 2, so a split at
    // layer 2 leaves stage 1 with no producer).  Null is a bug we want loud, not a null into a graph.
    // A LAYER READS ACROSS STAGES, AND NO BOUNDARY FIXES THAT.  The compressed KV of layer kvl is a persistent cache
    // that lives on the card of the stage which RUNS kvl, and layer `il` reads it by pointer inside its graph; the
    // relay hands over per-token state (x_state / pre_state) and has no way to move a cache.  So a split's stage set has
    // to be cut where no layer reads across - which means EVERY stage's layers find their sources inside that stage, and
    // therefore every boundary is a layer of split_valid_boundaries().  build_stages refuses plans that are not; reaching
    // this line means a stage was built by hand, so it is fatal, not recoverable.  (The GLM engine's layers keep their
    // own caches, so the same code shape there is a legal split; see docs/gfx906/P5-RELAY-PATTERN.md section 6b.)
    if (L.ratio && !im.owns_layer(L.kvl)) {
        std::fprintf(stderr, "[ds41_dense] build_attn: layer %lld reads the compressed KV of layer %lld, which this "
                     "stage [%lld,%lld) does not run; its boundary %lld is not valid for this model (valid: %s)\n",
                     (long long) il, (long long) L.kvl, (long long) im.lb, (long long) im.le, (long long) im.le,
                     Ds41Dense::split_valid_set_text(im.g).c_str());
        std::abort();
    }
    if (L.ratio && !im.owns_layer(L.idxl)) {
        std::fprintf(stderr, "[ds41_dense] build_attn: layer %lld reads the index keys of layer %lld, which this "
                     "stage [%lld,%lld) does not run; its boundary %lld is not valid for this model (valid: %s)\n",
                     (long long) il, (long long) L.idxl, (long long) im.lb, (long long) im.le, (long long) im.le,
                     Ds41Dense::split_valid_set_text(im.g).c_str());
        std::abort();
    }
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH, NH = im.NH, SWA = im.SWA, RAW = im.RAW;
    const int64_t r = L.ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);          // [(HC+1) D, n]
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));   // [D, HC]
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn));    // [1, HC, n]
        dot = ggml_scale(gc, dot, 1.0f / std::sqrt((float) D));
        // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))  (signed sqrt, as the training kernel)
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);
        x = ggml_add(gc, x, gv);
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));   // the block input (residual) = state + Engram
    }

    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr;
    b.hc_mixes(x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
               &a_pre, &a_post, &a_comb);
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));             // [D, n]
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- q (no per-head norm in V4.1)
    ggml_tensor * qr = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn), im.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_reshape_3d(gc, ggml_mul_mat(gc, im.BL(il, "attn_q_b.weight"), qr), DH, NH, n);
    q = b.rope(q, pos, r);

    // ---- window K (= V): [ring | this pass's rows] -> each query's SWA rows; the ring keeps the last RAW
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);   // model.py _window_kv
    ggml_tensor * wsrc = ggml_concat(gc, L.raw->type == GGML_TYPE_F32 ? L.raw : ggml_cast(gc, L.raw, GGML_TYPE_F32), kv2, 1);     // [DH, RAW + n]
    ggml_tensor * Kw = ggml_reshape_3d(gc, ggml_get_rows(gc, wsrc, b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4)), DH, SWA, n);
    ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, n, col0);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV (kv sources write it; every ratio > 0 layer reads its source's)
    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;   // this pass's writes (later reads depend on them)
    if (r) {
        if (L.kv_src) {
            ggml_tensor * latent = nullptr;                                                  // [DH, n]
            if (r == 1) {   // model.py Compressor.forward: ratio 1 = a plain projection, no gate
                latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            } else {        // ratio 2: softmax-gated pooling over the group's 2 states, per dim (fp32)
                ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
                ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
                ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);                     // [DH, 2 + n]
                ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
                ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
                ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
                ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
                vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));                       // [2, DH, n]
                vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
                const int64_t k = std::min<int64_t>(n, 2);
                late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
                late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
            }
            latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
            ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
            ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
            // indexer keys from the UNROTATED latent (model.py Indexer.forward, owns_k)
            ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
            ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
            if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
            idxk_w = ggml_set_rows(gc, L.idxk, ik, cslot);
            outs.push_back(idxk_w);
            ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
            if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
            comp_w = ggml_set_rows(gc, L.comp, lr, cslot);
            outs.push_back(comp_w);
        }
        if (cap > 0) {
            const L_t & KS = im.ly[(size_t) L.kvl];
            const int64_t K = std::min<int64_t>(cap, im.ITOP);
            ggml_tensor * comp = comp_w ? comp_w : KS.comp;
            ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk;
            ggml_tensor * sel = nullptr, * selm = nullptr;
            if (L.idx_src) {
                // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                ggml_tensor * iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                iq = b.rope(ggml_reshape_3d(gc, iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) iq = fake_quant(gc, ggml_cont(gc, iq), Ds41Quant::FP4_POW2_32);
                ggml_tensor * iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));         // softmax_scale * heads^-0.5
                ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, cap, idxk->nb[1], idxk->nb[1], 0);
                // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], n, (size_t) col0 * 4), 1, n);
                ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, cap, 1, (size_t) cap * 4, 0);
                ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, n)), nvs)));
                ggml_tensor * sc = ggml_lightning_indexer(gc, iq, ksrc, iw, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, n]
                sc = ggml_reshape_2d(gc, sc, cap, n);
                if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                    // P5.3b: the tensor whose columns are read is the PRODUCER's when the producer lives on another
                    // card (cand_src_prev, read in place - same-card stages) or this stage's own copy of the relayed
                    // bytes (cross-card).  Either way it is I32 [CB, NC] with the same strides, so one expression.
                    ggml_tensor * cnd = im.cand_src_prev ? im.cand_src_prev : im.cand;
                    const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                    ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, n), NEG_INF);
                    ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, n), 0.0f);
                    m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, cnd, ncb, n, cnd->nb[1], (size_t) col0 * cnd->nb[1])));
                    m = ggml_view_2d(gc, m, cap, n, m->nb[2], 0);
                    sc = ggml_add(gc, sc, m);
                }
                if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                    const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                    GGML_ASSERT(cap % im.CBS == 0);
                    ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, n]
                    // pin: + 1e30 at block (nvis - 1) / CBS of each query
                    ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                    ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, n, (size_t) col0 * 4), 1, n);
                    ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                    bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                    ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));                     // I32 [ncb, n]
                    outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, n, im.cand->nb[1], (size_t) col0 * im.cand->nb[1])));
                }
                sel = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                                     // I32 [K, n]
                // Canonicalise the top-k order.  ggml_cuda_op_top_k asks CUB's DeviceTopK for
                // output_ordering::unsorted, and CUB hands the k ids back in an order its atomically assigned
                // output slots produce - run-to-run unstable, while the SET is stable (measured: order-only on
                // every differing row).  The ids drive the gather of the compressed rows, so that order is the
                // flash-attn's accumulation order; left alone the whole run stops being reproducible from the
                // first query whose compressed rows are attended (the SWA window's 128).  Sort ascending.
                sel = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, K, n),
                                                        ggml_argsort(gc, ggml_cast(gc, sel, GGML_TYPE_F32),
                                                                     GGML_SORT_ORDER_ASC)), K, n);
                selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, n), sel), K, n);   // vis at sel
                outs.push_back(ggml_cpy(gc, sel, ggml_view_2d(gc, L.sel, K, n, L.sel->nb[1], (size_t) col0 * L.sel->nb[1])));
                outs.push_back(ggml_cpy(gc, selm, ggml_view_2d(gc, L.selm, K, n, L.selm->nb[1], (size_t) col0 * L.selm->nb[1])));
            } else {
                const L_t & IS = im.ly[(size_t) L.idxl];
                sel = ggml_view_2d(gc, IS.sel, K, n, IS.sel->nb[1], (size_t) col0 * IS.sel->nb[1]);
                selm = ggml_view_2d(gc, IS.selm, K, n, IS.selm->nb[1], (size_t) col0 * IS.selm->nb[1]);
            }
            Kc = ggml_reshape_3d(gc, ggml_get_rows(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * n))), DH, K, n);
            Mc = ggml_cont(gc, selm);
            // the source's rows must be read AFTER this pass wrote its own (a kv source attends to its new row)
        }
    }

    // ---- attention: per query its window rows + its top-k compressed rows, one flash-attn with the query as batch
    ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                                 // [DH, M, n]
    ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;                                 // [M, n]
    int64_t M = Kall->ne[1];
    if (!im.cpu && M % 256) {   // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256
        const int64_t pad = 256 - M % 256;
        Kall = ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        M += pad;
    }
    ggml_tensor * k4 = ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, n), GGML_TYPE_F16);
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);    // [DH, 1, NH, n]
    ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
    ggml_tensor * o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
    o = ggml_reshape_3d(gc, o, DH, NH, n);
    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    // grouped output LoRA (model.py: einsum over groups) then wo_b
    o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);               // [OGD, n, OG]
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);              // [OL, OG, n]
    ggml_tensor * attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]

    // ---- hc_post(attention) -> FFN hc: its mixes give the next layer's pre-mix; its input uses the attention's
    ggml_tensor * h = ggml_dsv4_hc_post(gc, attn, x, a_post, a_comb);                       // [D, HC, n]
    ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    b.hc_mixes(h, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
               &f_pre, &f_post, &f_comb);
    ggml_tensor * fn = b.rms_w(ggml_dsv4_hc_pre(gc, h, a_pre), im.BL(il, "ffn_norm.weight"));   // [D, n]

    // ---- router (model.py Gate): sqrt(softplus), top-k on score + bias, weights from the raw scores
    ggml_tensor * rl = ggml_mul_mat(gc, im.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_sqrt(gc, ggml_softplus(gc, rl));                                  // [NE, n]
    // selection = score + exp_probs_b + the selection-only rbias (soft prune / cache bonus); the weights below come
    // from the unbiased scores, as model.py's Gate - all-zero rbias (the default) = the model's routing bit for bit
    ggml_tensor * s0 = ggml_add(gc, sc, im.BL(il, "exp_probs_b.bias"));                        // the model's own selection score
    ggml_tensor * rb = ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1);
    ggml_tensor * sel = nullptr;
    // SMART route bias (opt-in; the flat bias changes ~every choice it can: -14 pt API top-1, repetition loops):
    //   STRATA_RB_PROTECT=P  the exact top-P experts are always kept (+1e4); the bias only competes for the other slots
    //   STRATA_RB_MARGIN=d   the bias only applies to experts whose exact score is within d of the cutoff (midpoint of the
    //                        exact NU-th / NU+1-th scores); experts confidently above it (> cut + d) are always kept -
    //                        only near-ties can flip
    // Both may be set.  Weights still come from the unbiased scores.  Unset = the flat bias, bit for bit as before.
    static const int rb_protect = std::getenv("STRATA_RB_PROTECT") ? std::atoi(std::getenv("STRATA_RB_PROTECT")) : 0;
    static const float rb_margin = std::getenv("STRATA_RB_MARGIN") ? (float) std::atof(std::getenv("STRATA_RB_MARGIN")) : -1.0f;
    if (rb_protect > 0 || rb_margin >= 0.0f) {
        const int kt = (int) im.NU + 1;
        ggml_tensor * ti = ggml_argsort_top_k(gc, s0, kt);                                         // [kt, n]
        ggml_tensor * tv = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s0, 1, im.NE, n), ti), kt, n);   // [kt, n] desc
        auto row = [&](int r) { return ggml_view_2d(gc, tv, 1, n, tv->nb[1], (size_t) r * tv->nb[0]); };   // [1, n]
        ggml_tensor * like = ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.NE, n);
        ggml_tensor * protect = nullptr;
        if (rb_protect > 0) {   // s0 >= the P-th largest exact score
            ggml_tensor * vp = ggml_repeat(gc, ggml_cont(gc, row(std::min(rb_protect, (int) im.NU) - 1)), like);
            protect = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, vp), 1.0f, 1e-6f));
        }
        ggml_tensor * gate = nullptr;
        if (rb_margin >= 0.0f) {
            ggml_tensor * cut = ggml_repeat(gc, ggml_scale(gc, ggml_add(gc, ggml_cont(gc, row((int) im.NU - 1)), ggml_cont(gc, row((int) im.NU))), 0.5f), like);
            ggml_tensor * dv = ggml_sub(gc, s0, cut);                                              // [NE, n]
            ggml_tensor * above = ggml_step(gc, ggml_scale_bias(gc, dv, 1.0f, -rb_margin));          // dv > d
            ggml_tensor * below = ggml_step(gc, ggml_scale_bias(gc, dv, -1.0f, -rb_margin));         // dv < -d
            protect = protect ? ggml_clamp(gc, ggml_add(gc, protect, above), 0.0f, 1.0f) : above;
            gate = ggml_scale_bias(gc, ggml_clamp(gc, ggml_add(gc, above, below), 0.0f, 1.0f), -1.0f, 1.0f);   // in the band
        }
        ggml_tensor * unprot = ggml_scale_bias(gc, protect, -1.0f, 1.0f);
        if (gate) unprot = ggml_mul(gc, unprot, gate);
        sel = ggml_add(gc, ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, rb, like), unprot)), ggml_scale(gc, protect, 1e4f));
    } else {
        sel = ggml_add(gc, s0, rb);
    }
    ggml_tensor * ids = ggml_argsort_top_k(gc, sel, (int) im.NU);   // [NU, n]
    ggml_tensor * wts = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), ids), im.NU, n);
    if (G.norm_topk) wts = ggml_div(gc, wts, ggml_scale_bias(gc, ggml_sum_rows(gc, wts), 1.0f, 1e-20f));
    wts = ggml_scale(gc, wts, (float) G.weights_scale);

    // ---- shared expert (model.py Expert, clamped SwiGLU)
    const float lim = (float) G.swiglu_limit;
    ggml_tensor * up = ggml_mul_mat(gc, im.BL(il, "ffn_up_shexp.weight"), fn);
    ggml_tensor * gt = ggml_mul_mat(gc, im.BL(il, "ffn_gate_shexp.weight"), fn);
    if (lim > 0) {
        up = ggml_clamp(gc, up, -lim, lim);
        gt = ggml_clamp(gc, gt, NEG_INF, lim);
    }
    ggml_tensor * se = ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_swiglu_split(gc, gt, up));

    outs.push_back(ggml_cpy(gc, h, b.cols3(im.hap, D, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, f_post, b.cols2(im.post_f, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, ggml_reshape_3d(gc, f_comb, HC, HC, n), b.cols3(im.comb_f, HC, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, se, b.cols2(im.shexp, D, n, col0)));
    outs.push_back(ggml_cpy(gc, fn, b.cols2(im.o_fn, D, n, col0)));
    outs.push_back(ggml_cpy(gc, ids, b.cols2(im.o_ids, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, wts, b.cols2(im.o_wts, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, f_pre, b.cols2(im.pre_state, HC, n, col0)));   // reads of pre_state happen above
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// enc-skip: the persistent-KV half of build_attn for one span.  Runs hc_pre / attn_norm, writes the raw window K/V
// ring, and on a kv source layer writes the compressed-KV row + indexer-key row (+ the ratio-2 compressor state).
// Everything else build_attn does (q, the attention + output LoRA, hc_post, the router, the shared expert, the indexer
// selection / candidate blocks, and the pre_state / hap / shexp / o_* writes) only ever flows into THIS layer's own
// output columns, so a prompt token whose later layers are never read does not need them.  The state it does write is
// exactly what build_attn writes for those positions, so a later query's window / compressed reads see the same
// values as the full path (verified by the enc-skip gate).
static ggml_cgraph * build_attn_kv_only(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0_p = 0) {
    const int64_t col0 = pass_col0(im, col0_p);
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    // P5.3b: as in build_attn - a layer whose KV / index-key source this stage does not own means the stage's boundary
    // is not one of the model's valid boundaries (Ds41Dense::split_boundary_valid), which build_stages refuses.
    if (L.ratio && (!im.owns_layer(L.kvl) || !im.owns_layer(L.idxl))) {
        std::fprintf(stderr, "[ds41_dense] build_attn_kv_only: layer %lld reads the KV of layer %lld / the index keys "
                     "of layer %lld, outside this stage [%lld,%lld) (valid boundaries: %s)\n", (long long) il,
                     (long long) L.kvl, (long long) L.idxl, (long long) im.lb, (long long) im.le,
                     Ds41Dense::split_valid_set_text(im.g).c_str());
        std::abort();
    }
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH;
    const int64_t r = L.ratio;
    std::vector<ggml_tensor *> outs, late;   // late: the ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {   // the Engram add still feeds the K/V + compressor projections
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_scale(gc, ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn)), 1.0f / std::sqrt((float) D));
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        x = ggml_add(gc, x, ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate));
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    }
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- window K (= V): the same post-RoPE raw row build_attn writes into the ring
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV + indexer keys (kv source layers; the only other persistent state a later query reads)
    if (r && L.kv_src) {
        ggml_tensor * latent = nullptr;
        if (r == 1) {
            latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
        } else {
            ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
            ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
            ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);
            ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
            ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
            ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
            ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
            vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));
            vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
            latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
            const int64_t k = std::min<int64_t>(n, 2);
            late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
            late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
        }
        latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
        ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
        ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
        ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
        ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
        if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
        outs.push_back(ggml_set_rows(gc, L.idxk, ik, cslot));
        ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
        if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
        outs.push_back(ggml_set_rows(gc, L.comp, lr, cslot));
    }
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// predict: router(rms_norm(hc_pre(state, pre)) * ffn_norm) + bias, top 16 of token 0
static ggml_cgraph * build_predict(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor ** out, int64_t n = 1) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_dsv4_hc_pre(gc, b.cols3(im.x_state, D, HC, n), b.cols2(im.pre_state, HC, n));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(x, im.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
    *out = ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*out);
    ggml_build_forward_expand(gf, *out);
    return gf;
}

// finish: x_state = hc_post(shexp + routed, hap, post_f, comb_f).  col0: the batch column (0 on the single-pass paths).
static ggml_cgraph * build_finish(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0_p = 0) {
    const int64_t col0 = pass_col0(im, col0_p);
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                                        b.cols3(im.comb_f, HC, HC, n, col0));
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    return gf;
}

// finish (n = 1) + the next layer's routing prediction on the state it writes (build_predict's ops on the cpy result)
static ggml_cgraph * build_finish_predict(Ds41Dense::Impl & im, ggml_context * gc, int next_il, ggml_tensor ** pout,
                                          int ahead = 0, std::vector<ggml_tensor *> * aout = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 128, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, 1, 0), b.cols2(im.routed, D, 1, 0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, 1, 0), b.cols2(im.post_f, HC, 1, 0),
                                        b.cols3(im.comb_f, HC, HC, 1, 0));
    ggml_tensor * st = ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, 1, 0));
    ggml_build_forward_expand(gf, st);
    ggml_tensor * xp = ggml_dsv4_hc_pre(gc, st, b.cols2(im.pre_state, HC, 1));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(next_il), b.rms_w(xp, im.BL(next_il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(next_il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) next_il].rbias, im.NE, 1));
    // the layers after next_il on the same state (speculative reads): their scores are stacked under next_il's so ONE
    // argsort gives all the rows and ONE readback fetches them ([16, 1 + ahead] I32)
    for (int a = 1; a <= ahead && aout; ++a) {
        const int il = next_il + a;
        // P5.3b: stop at the model's last layer AND at this stage's own range: the fused graph reads block il's
        // ffn_norm / router / routing-bias weights, and a stage only loads the ONE block past its range (the one
        // its finish_predict(next_il = le-1) needs).  A caller that asked for more gets fewer rows, which the
        // readback loop already fills with -1 (never a prefetch hint for a layer this card cannot see).
        if (il >= (int) im.ly.size() || il >= im.wle) break;
        ggml_tensor * ra = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(xp, im.BL(il, "ffn_norm.weight")));
        ggml_mul_mat_set_prec(ra, GGML_PREC_F32);
        ggml_tensor * sa = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, ra)), im.BL(il, "exp_probs_b.bias")),
                                    ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
        sc = ggml_concat(gc, sc, sa, 1);
        aout->push_back(nullptr);   // a row of *pout
    }
    *pout = ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*pout);
    ggml_build_forward_expand(gf, *pout);
    return gf;
}

// head: hc_pre(state, pre_state) -> output_norm -> output, rows row0..row0+nr-1
static ggml_cgraph * build_head(Ds41Dense::Impl & im, ggml_context * gc, int64_t row0, int64_t nr, ggml_tensor ** lg) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = b.cols3(im.x_state, D, HC, nr, row0);
    ggml_tensor * p = b.cols2(im.pre_state, HC, nr, row0);
    ggml_tensor * h = b.rms_w(ggml_dsv4_hc_pre(gc, x, p), im.W("output_norm.weight"));
    *lg = ggml_mul_mat(gc, im.W("output.weight"), h);
    ggml_set_output(*lg);
    ggml_build_forward_expand(gf, *lg);
    return gf;
}

// begin: x_state = repeat(embd), pre_state = one-hot (model.py make_identity_pre_mix).  col0: the batch column the
// pass's n tokens live at (0 on the single-pass paths).
static ggml_cgraph * build_init(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0_p = 0) {
    const int64_t col0 = pass_col0(im, col0_p);   // stage 0 always: begin() set col0 = 0
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * e = ggml_reshape_3d(gc, b.cols2(im.i_emb, D, n, col0), D, 1, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_repeat_4d(gc, e, D, HC, n, 1), b.cols3(im.x_state, D, HC, n, col0)));
    ggml_tensor * one = ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, n), 1.0f);
    ggml_tensor * oh = ggml_pad(gc, one, (int) (HC - 1), 0, 0, 0);                            // [1, 0, 0, 0] per token
    ggml_build_forward_expand(gf, ggml_cpy(gc, oh, b.cols2(im.pre_state, HC, n, col0)));
    return gf;
}

// ================================================================== public API

Ds41Dense::Ds41Dense() : p_(new Impl()) {}
Ds41Dense::~Ds41Dense() = default;
const Ds41Geometry & Ds41Dense::geom() const { return p_->g; }
ggml_tensor * Ds41Dense::weight(const std::string & name) const { return p_->w.get(name); }
int64_t Ds41Dense::n_layer() const { return p_->nl; }
ggml_backend_t Ds41Dense::backend() const { return p_->backend; }
const std::string & Ds41Dense::last_error() const { return p_->err; }

// The NC-column tensors (see Impl::nctx): (re)created for `NC` columns, zeroed.  Every one is per-pass scratch (a pass
// writes the columns it reads), so a resize between passes loses nothing.
static bool alloc_cols(Ds41Dense::Impl & im, int64_t NC, std::string & err) {
    if (im.nbuf) { ggml_backend_buffer_free(im.nbuf); im.nbuf = nullptr; }
    if (im.nctx) { ggml_free(im.nctx); im.nctx = nullptr; }
    im.NC = NC;
    ggml_init_params cp = { ggml_tensor_overhead() * 256, nullptr, true };
    im.nctx = ggml_init(cp);
    ggml_context * c = im.nctx;
    const int64_t D = im.D, HC = im.HC;
    im.x_state = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.pre_state = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.hap = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.post_f = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.comb_f = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, NC);
    im.shexp = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_fn = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
    im.o_wts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    im.routed = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.cand = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.CB, NC);
    for (auto & L : im.ly)
        if (L.idx_src) { L.sel = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.ITOP, NC); L.selm = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.ITOP, NC); }
    im.i_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.i_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    im.i_widx = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.SWA * NC);
    im.span_memo_pos0 = -1;   // new tensors: nothing uploaded yet
    im.i_wmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.SWA, NC);
    im.i_wslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));
    for (int r = 1; r <= 2; ++r) {
        im.i_nvis[r] = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
        im.i_cslot[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
        im.i_cpos[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    }
    im.i_sidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2 * NC);
    im.i_sslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, 2));
    im.i_lastblk = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    for (int e = 0; e < 2; ++e) im.i_eng[e] = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.EC * im.ED, NC);
    // STRATA_VRAM_TRACE: WHICH state tensor owns the columns' VRAM (s13c: a batch's state alloc was 2.45-2.76 GiB and
    // a 2193.81 MiB allocation later failed with 6.83 GiB "free" - name the owner before sizing anything).
    if (std::getenv("STRATA_VRAM_TRACE")) {
        double tot = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) tot += (double) ggml_nbytes(t);
        std::fprintf(stderr, "vram trace: state NC=%d wants %.2f GiB (i_eng %lld x %d = %.2f GiB each, i_wmask %.1f MiB, "
                             "i_widx %.1f MiB)\n", NC, tot / 1073741824.0, (long long) im.EC, (int) im.ED,
                     (double) ggml_nbytes(im.i_eng[0]) / 1073741824.0, (double) ggml_nbytes(im.i_wmask) / 1048576.0,
                     (double) ggml_nbytes(im.i_widx) / 1048576.0);
    }
    im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend);
    if (!im.nbuf) { err = "cannot allocate " + std::to_string(NC) + " state columns"; return false; }
    ggml_backend_buffer_clear(im.nbuf, 0);
    return true;
}

bool Ds41Dense::set_columns(int nc) {
    Impl & im = *p_;
    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC) return true;
    ggml_backend_synchronize(im.backend);
    // every cached graph holds views of the old columns (their addresses are fixed at the first allocation), and the
    // allocators remember graphs by address - all of it goes, and is rebuilt lazily for the new tensors
    im.graphs.clear();
    im.logits_out.clear();
    im.predict_out.clear();
    im.predict_out_n.clear();
    im.fpredict_out.clear();
    im.fpredict_ahead.clear();
    for (Allo * al : { &im.allo_dec, &im.allo_pass }) {
        if (al->a) ggml_gallocr_free(al->a);
        al->a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        ++al->epoch;
    }
    if (im.gctx) ggml_free(im.gctx);
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    return alloc_cols(im, want, im.err);
}

int Ds41Dense::columns() const { return (int) p_->NC; }

// P5.3b: the pass cap a STAGED run enforces (see the header).  Deliberately NOT a set_columns(): the state columns
// stay at their allocation, only the largest pass the graphs are built for gets smaller, so a stage's per-pass scratch
// (the Engram rows, the window masks, the chunk's compute buffer) follows the staging while the columns another stage
// reads in place do not move.
void Ds41Dense::set_pass_cap(int n) {
    Impl & im = *p_;
    if (n < 1 || n > (int) im.NC) return;
    im.S = n;
    im.cfg.max_pass = n;
    im.graphs.clear();
    im.logits_out.clear();
    im.predict_out.clear();
    im.predict_out_n.clear();
    im.fpredict_out.clear();
    im.fpredict_ahead.clear();
    for (Allo * al : { &im.allo_dec, &im.allo_pass }) {
        if (al->a) ggml_gallocr_free(al->a);
        al->a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        ++al->epoch;
    }
}

bool Ds41Dense::init(const std::string & path, const Ds41DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    try {
        im.model = std::make_unique<GgufModel>(GgufModel::open(path));
        im.model_path = path;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (!(err = ds41_read_geometry(*im.model, im.g)).empty()) return false;
    im.layers = ds41_file_layers(*im.model, im.g);
    if (!(err = ds41_check_tensors(*im.model, im.g, im.layers)).empty()) return false;
    for (size_t i = 0; i < im.layers.size(); ++i)
        if (im.layers[i] != (int64_t) i) { err = "the file's blocks must be 0..k"; return false; }
    im.nl = (int64_t) im.layers.size();
    // ---- P5.3b: the stage's range.  Defaults (lb 0, le 0 = nl, is_first/is_last true) give the single-stage run of
    // the last release.  The checks are the ones build_stages also makes, kept here so a hand-built config that lies
    // about its position fails at init rather than producing a run with no embeddings or no head.
    im.share_weights = cfg.share_weights;
    im.lb = cfg.lb;
    im.le = cfg.le > 0 ? cfg.le : im.nl;
    im.is_first = cfg.is_first;
    im.is_last = cfg.is_last;
    // A hand-built config with a real range is staged too, so the state region follows the boundary rule here and not
    // only inside build_stages: the two must agree, and init() is the one place that both paths pass through.
    im.staged = cfg.staged || (im.lb != 0 || im.le != im.nl);
    im.cfg_state_le = cfg.state_le;
    im.compute_state_end();
    if (im.lb < 0 || im.le > im.nl || im.lb >= im.le) {
        err = "stage layer range [" + std::to_string(im.lb) + "," + std::to_string(im.le) +
              ") is not inside the model's [0," + std::to_string(im.nl) + ")";
        return false;
    }
    if (!cfg.is_first && im.lb == 0) { err = "stage [" + std::to_string(im.lb) + "," + std::to_string(im.le) +
                                           ") starts at 0 but says it is not the first stage"; return false; }
    if (!cfg.is_last && im.le == im.nl) { err = "stage [" + std::to_string(im.lb) + "," + std::to_string(im.le) +
                                               ") ends the model but says it is not the last stage"; return false; }
    if (im.lb > 0 && !cfg.share_weights && im.is_first) { err = "internal: is_first with lb > 0"; return false; }
    // The weight bounds: the range it RUNS plus the one block its last layer reads ahead into (see Impl::wle).
    // A stage that borrows the whole load (share_weights) loads nothing, so its bounds do not matter; leaving them
    // full-width keeps them honest if a borrowed Impl is ever asked what the model has.
    im.wl = im.share_weights ? 0 : im.lb;
    // The weight bounds: the layers it RUNS plus the one block its last layer reads ahead into (see Impl::wle).
    im.wle = im.share_weights ? im.nl
                              : (im.alloc_end() >= im.nl ? im.nl : weight_end(im.nl, im.alloc_end()));
    const Ds41Geometry & G = im.g;
    im.D = G.n_embd; im.HC = G.hc; im.DH = G.d_head; im.DHR = G.d_rope; im.NH = G.n_head; im.OG = G.o_groups;
    im.OL = G.o_lora; im.OGD = G.n_head * G.d_head / G.o_groups; im.RQ = G.r_q; im.SWA = G.n_swa;
    im.IH = G.idx_heads; im.ID = G.idx_dim; im.ITOP = G.idx_topk; im.NE = G.n_expert; im.NU = G.n_used; im.NFF = G.n_ff;
    im.CB = G.cand_blocks; im.CBS = G.cand_bsize; im.EC = (G.engram_ngram - 1) * G.engram_heads; im.ED = G.engram_dim;
    // gate-only overrides (the golden takes the same): small top-k / candidate counts make the selection logic bite
    // on a short prompt
    if (const char * e = std::getenv("DS41_TEST_TOPK")) im.ITOP = std::atoi(e);
    if (const char * e = std::getenv("DS41_TEST_CAND_BLOCKS")) im.CB = std::atoi(e);
    im.S = std::max(1, cfg.max_pass);                       // the dense sub-batch size of the layer-major walk
    im.NCmax = std::max<int64_t>(im.S, std::max(1, cfg.prompt_batch));   // state columns: the whole prompt batch, or the pass
    im.NC = cfg.elastic_cols ? im.S : im.NCmax;                           // elastic: S until a batch asks for more
    // the window ring: every row is gathered BY POSITION (slot = pos % RAW), so its size never changes results.  DSpark
    // verify passes write up to k rejected draft positions; a rejected q evicts q - RAW, which a later query q' (>= the
    // last accepted position, re-fed) needs iff q - RAW >= q' - SWA + 1  ->  safe for k <= RAW - SWA = 16 drafts.
    im.RAW = im.SWA + 16;
    im.cmax[1] = cfg.ctx_max; im.cmax[2] = cfg.ctx_max / 2;
    if (im.ITOP > im.cmax[2] || im.CBS <= 0 || cfg.ctx_max % 256) { err = "ctx_max must be a multiple of 256 and >= 1024"; return false; }

    if (cfg.backend) { im.backend = cfg.backend; }
    else {
        im.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(im.backend, cfg.n_threads);
        im.own_backend = true;
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    // share_weights (set by build_stages): this stage takes NO weight tensors of its own - it adopted the owner's map
    // BEFORE this init ran (build_stages adopts first, because init builds graphs that look weights up by name).
    if (!im.share_weights && !load_weights(im, err)) return false;
    // P5.3b SHAPE NOTE: the token ring that feeds these reads is filled in upload_pass, i.e. only on the stage that
    // runs begin() (stage 0).  So engram layers must live in ONE stage - and the CACHES' RULE does not give that for
    // free: {8, 14, 20} is a cache-legal plan that cuts between this model's pair (1, 14), which is why ds41_generate
    // enforces the Engram constraint itself (the first boundary above the last Engram layer).  A boundary between them
    // would leave a stage hashing against a ring it never fed.
    // The Engram tables are ~GBs of host memory and only the layer range that holds an engram layer reads them, so a
    // stage that owns none does not open them (its graphs never ask: no layer in its range has an eslot).
    const int64_t sen = im.state_end();
    bool stage_engram = false;
    for (int64_t l = im.lb; l < sen; ++l)
        if (G.engram_slot(l) >= 0) { stage_engram = true; im.eng_slots.push_back(G.engram_slot(l)); }
    if (!cfg.engram_dirs.empty() && stage_engram && !im.engram.open(G, cfg.engram_dirs, cfg.engram_threads, err)) return false;

    // layers
    im.ly.resize((size_t) im.nl);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        L.ratio = G.ratio[(size_t) l];
        L.kv_src = G.is_kv_source(l);
        L.idx_src = G.is_index_source(l);
        L.cand_src = l == G.cand_layer;
        L.uses_cand = G.cand_layer >= 0 && l > G.cand_layer && L.idx_src;
        L.kvl = G.kv_src(l);
        L.idxl = G.idx_src(l);
        L.eslot = G.engram_slot(l);
    }
    // persistent state
    if (cfg.kv_fp4 && (!cfg.act_quant || im.cpu || im.DH % 64 || im.ID % 32)) {
        err = "--kv-fp4 needs act_quant (the FP4 grids), the CUDA backend, d_head % 64 == 0 and idx_dim % 32 == 0";
        return false;
    }
    const ggml_type kvt = cfg.kv_type >= 0 ? (ggml_type) cfg.kv_type : cfg.act_quant ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    ggml_init_params sp = { ggml_tensor_overhead() * 512, nullptr, true };
    im.sctx = ggml_init(sp);
    auto T = [&](ggml_type t, std::initializer_list<int64_t> ne) {
        const int64_t * d = ne.begin();
        switch (ne.size()) {
            case 1: return ggml_new_tensor_1d(im.sctx, t, d[0]);
            case 2: return ggml_new_tensor_2d(im.sctx, t, d[0], d[1]);
            default: return ggml_new_tensor_3d(im.sctx, t, d[0], d[1], d[2]);
        }
    };
    const int64_t D = im.D, NCM = im.NCmax;
    // P5.3b: the persistent state covers the layers THIS stage writes and its successor reads: [lb, state_end()).  A
    // stage's LAST layer is the producer of the compressed KV / index keys every later layer of its ratio reads (the
    // geometry's kv_src/idx_src always resolve to the latest source <= l, which for a successor layer is at or below
    // this stage's end), so its rings belong here and the successor's layers - which only READ them - allocate none.
    // A stage that ends the model allocates to nl (its own range, unchanged).  Layers outside the range keep null
    // state tensors, which is also what makes attn_router(il) fail loudly for another stage's layer.
    const int64_t s_end = im.alloc_end();
    for (int64_t l = im.lb; l < s_end; ++l) {
        auto & L = im.ly[(size_t) l];
        L.raw = T(kvt, { im.DH, im.RAW });
        L.raw_snap = T(kvt, { im.DH, im.RAW });
        L.rbias = T(GGML_TYPE_F32, { im.NE });   // zeroed with the buffer
        if (L.kv_src) {
            // STRATA_KV_FP4_ONLY=comp|idx (bisect tap): pack only one of the two caches
            static const char * only = std::getenv("STRATA_KV_FP4_ONLY");
            const bool fc = cfg.kv_fp4 && !(only && std::string(only) == "idx");
            const bool fi = cfg.kv_fp4 && !(only && std::string(only) == "comp");
            L.comp = T(fc ? GGML_TYPE_NVFP4 : kvt, { im.DH, im.cmax[L.ratio] + 1 });
            L.idxk = T(fi ? GGML_TYPE_MXFP4 : kvt, { im.ID, im.cmax[L.ratio] + 1 });
            if (L.ratio == 2) {
                L.st_kv = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_sc = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_kv_snap = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_sc_snap = T(GGML_TYPE_F32, { im.DH, 2 });
            }
        }
    }
    im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
    if (!im.sbuf) { err = "cannot allocate the decode state"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);
    // P5.3b: a staged producer of the candidate blocks publishes the tensor its graphs just built.  See
    // cand_published above; a single-instance run leaves it null and nothing reads it.
    if (im.staged && im.owns_cand_producer()) im.cand_published = im.cand;
    // the ratio-2 state scores start at -1e30 (an unwritten state gets softmax weight 0); sel = 0 (a valid row)
    for (auto & L : im.ly)
        if (L.st_sc) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); ggml_backend_tensor_set(L.st_sc, v.data(), 0, v.size() * 4); }
    // P5.3b: the candidate blocks live in the NC-column group (im.cand, alloc_cols).  Every stage allocates the
    // tensor, because the relay WRITES the producer's candidates into the successor's tensor at a boundary; on a
    // stage that neither writes nor reads them it is CB * NC * 4 bytes of never-touched memory (CB=2048, one
    // column: 8 KiB) - cheap enough that one rule for every stage beats a conditional payload.
    // inputs
    ggml_init_params ipp = { ggml_tensor_overhead() * 64, nullptr, true };
    im.ictx = ggml_init(ipp);
    auto I = [&](ggml_type t, int64_t a, int64_t b2 = 1) { return b2 == 1 ? ggml_new_tensor_1d(im.ictx, t, a) : ggml_new_tensor_2d(im.ictx, t, a, b2); };
    im.neg_ramp = I(GGML_TYPE_F32, im.cmax[1] + 1);
    im.blk_ramp = I(GGML_TYPE_F32, im.cmax[1] / im.CBS + 2);
    im.ibuf = ggml_backend_alloc_ctx_tensors(im.ictx, im.backend);
    if (!im.ibuf) { err = "cannot allocate the inputs"; return false; }
    ggml_backend_buffer_clear(im.ibuf, 0);
    {
        std::vector<float> v((size_t) im.neg_ramp->ne[0]);
        for (size_t i = 0; i < v.size(); ++i) v[i] = -(float) i;
        ggml_backend_tensor_set(im.neg_ramp, v.data(), 0, v.size() * 4);
        v.assign((size_t) im.blk_ramp->ne[0], 0);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (float) i;
        ggml_backend_tensor_set(im.blk_ramp, v.data(), 0, v.size() * 4);
    }
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    im.allo_dec.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    if (!alloc_cols(im, im.NC, err)) return false;
    // the host mirrors are sized for the largest batch once (pinned registrations must not move)
    im.host_fn.resize((size_t) (D * NCM));
#ifdef DS41_HAVE_CUDA
    // pinned: the fn rows go device -> host -> the tier's device input every layer (pageable copies ran at ~2 GB/s)
    if (!im.cpu && !im.host_fn.empty()) ggml_backend_cuda_register_host_buffer(im.host_fn.data(), im.host_fn.size() * 4);
#endif
    im.host_ids.resize((size_t) (im.NU * NCM));
    im.host_wts.resize((size_t) (im.NU * NCM));
#ifdef DS41_HAVE_CUDA
    // pinned too: attn_router reads fn / ids / wts back with three async copies and one sync (truly async only from
    // pinned memory)
    if (!im.cpu) {
        ggml_backend_cuda_register_host_buffer(im.host_ids.data(), im.host_ids.size() * 4);
        ggml_backend_cuda_register_host_buffer(im.host_wts.data(), im.host_wts.size() * 4);
    }
#endif
    return true;
}

// ================================================================== P5.3b: the staged run

int64_t Ds41Dense::stage_begin() const { return p_->lb; }
int64_t Ds41Dense::stage_end() const { return p_->le; }
int64_t Ds41Dense::stage_layers() const { return p_->le - p_->lb; }
int64_t Ds41Dense::n_layer_total() const { return p_->nl; }
bool Ds41Dense::is_last_stage() const { return p_->is_last; }
bool Ds41Dense::weights_shared() const { return p_->w_owner != nullptr; }
bool Ds41Dense::relay_carries_cand() const { return cand_can_cross(p_->g); }
// P5.3b: is there a layer AFTER cand_layer that ranks inside its candidate blocks (the same test the relay payload
// uses), and where would a boundary have to fall to separate the producer from a reader?
size_t Ds41Dense::relay_cand_bytes(const strata::v41::Ds41Geometry & g) {
    return cand_can_cross(g) ? (size_t) (g.cand_blocks * 4) : 0;
}
/// Can a LEGAL boundary separate cand_layer from a layer that ranks inside its blocks?  It asks the question the way
/// the rule now states it: does any member of split_valid_boundaries() sit strictly above the producer and at or below
/// a reader?  For a model whose caches ARE the kv / index source lists the answer is no by construction - validity at K
/// already requires cand_layer >= K whenever a reader is left at or above K - so a staged run of this model relays the
/// two state columns and nothing else (20, the witness's boundary, keeps writer 20 with its readers 24/28/32/36).
/// A model with a cache OUTSIDE those lists is the one that can separate them, and its payload carries the 8 KiB a token.
bool Ds41Dense::cand_can_cross(const strata::v41::Ds41Geometry & g) { return Impl::cand_can_cross(g); }
std::vector<int64_t> Ds41Dense::engram_slots() const { return p_->eng_slots; }

/// P5.3b: ONE BOUNDARY, TESTED DIRECTLY - the single source of truth for where a DeepSeek-V4.1 stage boundary may sit.
///
/// A layer m reads the compressed KV of kv_src(m) and the index keys of idx_src(m): the LATEST source at or below m.
/// Those rings are PERSISTENT GPU state, allocated once and written as the run advances - not per-pass activations.
/// The relay moves x_state, pre_state and the candidate column, all sized PER TOKEN, and there is no mechanism to move
/// a cache: a stage reads a ring by POINTER inside its own graphs.  So A RING LIVES ON THE CARD OF THE STAGE THAT RUNS
/// ITS WRITER, and the constraint is on the READERS:
///
///     K is valid  iff  for every m in [K, n_layer) with ratio[m] > 0:
///         kv_src(m) >= K   AND   idx_src(m) >= K
///         AND (m ranks inside cand_layer's blocks  =>  cand_layer >= K)
///
/// i.e. every layer at or above the boundary finds its rings - and the candidate column - at or above the boundary too,
/// in the stage that runs it.  The stage-0 side is free: kv_src(m) <= m < K always, so stage 0 never reaches over its
/// own boundary.  A ratio-0 layer runs no attention and reads no ring, so it is exempt (that is why layer 1, sliding-
/// window like layer 0, is valid by this rule although split_plan refuses to type a boundary below 2).
///
/// WHY THE DIRECTION IS THIS WAY ROUND.  A previous revision of this file required the WRITERS to sit below the
/// boundary and called LAST_SOURCE + 1 = 37 a floor.  That read the constraint upside down: a writer below the boundary
/// is precisely what strands its ring.  The tail stage [37,40) runs layers 37/38/39, whose kv_src is 20 and idx_src is
/// 36 - both below 37 - so it would build graphs over rings it never allocated, and build_attn's own check aborts on
/// exactly that (the abort the floor was written to prevent).  A legal boundary's tail stage contains the WRITERS its
/// layers read: the even split at 20 puts stage 1 = [20,40) over the writer of ring 20 AND every reader of it (20..39),
/// the writers and readers of index rings 24/28/32/36, and the candidate writer (20) with all of its readers
/// (24/28/32/36).
///
/// Because kv_src / idx_src are monotone in m ("the latest source at or below m"), the scan can only ever bite at the
/// boundary layer itself: kv_src(K) >= K means K IS its own kv source.  So a valid boundary is a layer that is BOTH a
/// kv and an index source - plus the vacuous case of a boundary layer that runs no attention.  That is the same answer
/// build_stages' cross-card check reaches from the other side, and the host gate re-derives the set by scanning stages
/// and their layers rather than trusting this function.
bool Ds41Dense::split_boundary_valid(const strata::v41::Ds41Geometry & g, int64_t K) {
    const int64_t n_layer = (int64_t) g.ratio.size();
    if (K <= 0 || K >= n_layer) return false;              // a boundary is inside the model, and never at layer 0
    for (int64_t m = K; m < n_layer; ++m) {
        if (g.ratio[(size_t) m] == 0) continue;            // a sliding-window layer runs no attention, reads no ring
        if (g.kv_src(m) < K) return false;                 // its compressed KV was written below the boundary
        if (g.idx_src(m) < K) return false;                // ... and so were its index keys
        // The candidate column: L.uses_cand is "above the producer and runs an indexer" (init()'s layer table).  The
        // column is per-token state and COULD be relayed, but the rule keeps a producer with its readers, so no legal
        // boundary separates them and the payload stays at the two state columns (see Impl::cand_can_cross).
        if (g.cand_layer >= 0 && m > g.cand_layer && g.is_index_source(m) && g.cand_layer < K) return false;
    }
    return true;
}

/// The legal layers, ascending.  For the shipped 40-layer build that is {1, 2, 8, 14, 20}: the four layers that are both
/// a kv and an index source, plus the vacuous 1 (ratio[1] == 0).  split_plan's floor of 2 leaves {2, 8, 14, 20} as the
/// set a caller can name - which contains the EVEN split of a 40-layer model in two stages.
std::vector<int64_t> Ds41Dense::split_valid_boundaries(const strata::v41::Ds41Geometry & g) {
    std::vector<int64_t> v;
    for (int64_t K = 1; K < (int64_t) g.ratio.size(); ++K) if (split_boundary_valid(g, K)) v.push_back(K);
    return v;
}

std::string Ds41Dense::split_valid_set_text(const strata::v41::Ds41Geometry & g) {
    const std::vector<int64_t> v = split_valid_boundaries(g);
    if (v.empty()) return "(none: this model has no boundary a stage split may use)";
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s;
}

/// P5.3b: THE BOUNDARY CHECK, as a pure function of the geometry, so a caller can vet a planned --layer-split before
/// allocating anything (ds41_generate does exactly that) and build_stages can enforce the same thing.  Every boundary is
/// a member of split_valid_boundaries() and the stage widths are positive and ascending.  That is EQUIVALENT to the
/// per-stage statement of the rule (a plan is legal iff each stage's layers find their rings inside that stage),
/// because the stage begins rise: if every boundary is valid, a layer of stage j reads sources at or above stage j's
/// own begin, since those sources are at or above the boundary that opened its stage.  `ends` = the end of every stage
/// but the last.
bool Ds41Dense::split_boundaries_ok(const strata::v41::Ds41Geometry & g, const std::vector<int64_t> & ends,
                                    std::string & err) {
    const int64_t n_layer = (int64_t) g.ratio.size();
    int64_t prev = 0;
    for (size_t i = 0; i < ends.size(); ++i) {
        const int64_t e = ends[i];
        if (e <= prev || e >= n_layer) {
            err = "stage boundary " + std::to_string(e) + " is not inside (0," + std::to_string(n_layer) +
                  ") and above the previous boundary " + std::to_string(prev);
            return false;
        }
        if (!split_boundary_valid(g, e)) {
            // Name the layer that strands what, so a reader can see WHY their number is not on the list: the first
            // layer at or above the boundary whose rings reach below it.
            std::string who;
            for (int64_t m = e; m < n_layer; ++m) {
                if (g.ratio[(size_t) m] == 0) continue;
                if (g.kv_src(m) < e) {
                    who = "layer " + std::to_string(m) + " reads the compressed KV of layer " + std::to_string(g.kv_src(m));
                    break;
                }
                if (g.idx_src(m) < e) {
                    who = "layer " + std::to_string(m) + " reads the index keys of layer " + std::to_string(g.idx_src(m));
                    break;
                }
                if (g.cand_layer >= 0 && m > g.cand_layer && g.is_index_source(m) && g.cand_layer < e) {
                    who = "layer " + std::to_string(m) + " ranks inside layer " + std::to_string(g.cand_layer) +
                          "'s candidate blocks";
                    break;
                }
            }
            err = "a boundary at layer " + std::to_string(e) + " leaves the next stage reading state written below it (" +
                  (who.empty() ? "no layer above it names a source" : who) + "); those rings live on this stage's card "
                  "and the relay moves per-token state, not caches.  This model's valid boundaries are " +
                  split_valid_set_text(g);
            return false;
        }
        prev = e;
    }
    return true;
}

/// P5.3b: snap a planned plan DOWN to the legal set, so an EVEN split (--layer-split unset) gets a working answer
/// instead of a refusal: 40 layers over 3 stages plans 13 and 26, and this turns them into 8 and 20.  A boundary with
/// nowhere to go (no valid layer at or below it and above the previous boundary) is a refusal, not a stage of zero
/// layers - such a plan needs fewer stages, not a different layer.
bool Ds41Dense::split_snap_to_valid(const strata::v41::Ds41Geometry & g, const std::vector<int64_t> & planned,
                                    std::vector<int64_t> & snapped, std::string & err) {
    const std::vector<int64_t> valid = split_valid_boundaries(g);
    snapped.clear();
    int64_t prev = 0;
    for (size_t i = 0; i < planned.size(); ++i) {
        int64_t pick = -1;
        for (int64_t v : valid) { if (v > planned[i]) break; if (v > prev) pick = v; }
        if (pick < 0) {
            err = "boundary " + std::to_string(i + 1) + " (planned at layer " + std::to_string(planned[i]) +
                  ") has no valid layer at or below it above the previous boundary " + std::to_string(prev) +
                  "; this model's valid boundaries are " + split_valid_set_text(g) + ", so a plan of " +
                  std::to_string(planned.size() + 1) + " stages cannot be snapped - it needs fewer stages, or a "
                  "--layer-split drawn from that set";
            return false;
        }
        snapped.push_back(pick);
        prev = pick;
    }
    return true;
}

// The bytes per token of a boundary payload: the two state columns every successor reads, plus the candidate column
// when any stage produces candidates a later stage consumes.  x_state [D, HC, NC] and pre_state [HC, NC] are laid
// out column-major, so each tensor's NC columns are contiguous from byte 0 and their strides are exactly these
// numbers - a region is one contiguous copy, and the relay's staging is in this order.
size_t Ds41Dense::relay_slab_bytes() const {
    const Impl & im = *p_;
    const size_t xs = (size_t) (im.D * im.HC * 4), ps = (size_t) (im.HC * 4);
    // THE PAYLOAD IS THE SAME NUMBER ON EVERY STAGE, and it includes the candidate column whenever ANY stage of the
    // run produces it (i.e. whenever the model has candidates at all and the run is staged).  The producer always has
    // a region of its own, so it always sends the region; whether a given CONSUMER copies it or ignores it is decided
    // by relay_write_in from that consumer's own relay_owns_cand_region().  Making the number depend on the reader
    // would make the two sides of a boundary disagree on where the copy ends, which is worse than copying 8 KiB/token
    // that one stage will not use - and that is the whole cost: CB * 4 = 8 KiB on top of 80 KiB of state.
    size_t s = xs + ps;
    if (im.staged && cand_can_cross(im.g)) s += (size_t) (im.CB * 4);   // im.cand is I32 [CB, NC]
    return s;
}

// Take over the owner's weight tensors.  The order is the point (identical reasoning to GLM's adopt_weights): the
// store is emptied first (a share_weights stage never allocated its own, so there is nothing to free), the owner
// pointer is set BEFORE the live tensor pointers are copied in, so any failure between the two leaves a store that
// is either empty (WStore::get aborts loudly) or fully owned - never a dangling pointer, and never a borrowed Impl
// that frees what it borrowed.
static void adopt_weights_into(Ds41Dense::Impl & st, Ds41Dense::Impl * owner) {
    st.w.ctx = nullptr;
    st.w.bufs.clear();
    st.w.t.clear();
    st.w.owns = false;
    st.w_owner = owner;
    st.w.t = owner->w.t;   // the live tensor pointers (the map itself stays ours)
    st.w.ctx = owner->w.ctx;
    st.w.bufs = owner->w.bufs;
    // The token embeddings are read from the owner's GGUF mmap (the file it opened is the same model), so a borrowed
    // stage's host lookups read the owner's mapping - it is not freed until the owner is.
    st.embd = owner->embd;
}

// P5.3b (mirrors GlmDense::build_stages): build the stages of a split run on THIS instance's weights.  A stage whose
// cfg.backend is this instance's borrows the tensors (one load, one copy: stage 0 also needs the embeddings and the
// last stage also needs the head, so n separate loads would hold more than the whole model on the card the split is
// meant to help); a stage on another card cannot borrow tensors that live in the first card's memory, so it loads
// its OWN range onto its own backend and its destructor frees exactly that.
bool Ds41Dense::build_stages(std::vector<std::unique_ptr<Ds41Dense>> & stage, const std::vector<Ds41DenseConfig> & cfg,
                             std::string & err) {
    Impl & im = *p_;
    if (im.w_owner) { err = "build_stages: this instance shares another's weights; it cannot hand them on"; return false; }
    if (stage.size() != cfg.size()) { err = "build_stages: " + std::to_string(stage.size()) + " stages, " +
                                            std::to_string(cfg.size()) + " configs"; return false; }
    if (stage.empty()) { err = "build_stages: no stages"; return false; }
    const int64_t n_layer = im.nl;
    // Validate the tiling BEFORE loading anything: an overlap would run a layer twice and a gap would leave a column
    // of the relay payload unwritten, so both are refused here rather than seen in the numbers.
    int64_t next_lb = 0;
    for (size_t s = 0; s < stage.size(); ++s) {
        const Ds41DenseConfig & c = cfg[s];
        const int64_t le = c.le > 0 ? c.le : n_layer;
        if (c.lb != next_lb) {
            err = "build_stages: stage " + std::to_string(s) + " starts at layer " + std::to_string(c.lb) +
                  ", but the stage before it ends at " + std::to_string(next_lb) +
                  " (the ranges must tile the model with no gap and no overlap)";
            return false;
        }
        if (le <= c.lb || le > n_layer) {
            err = "build_stages: stage " + std::to_string(s) + " range [" + std::to_string(c.lb) + "," +
                  std::to_string(le) + ") is outside the model's [0," + std::to_string(n_layer) + ")";
            return false;
        }
        if (c.is_first != (s == 0) || c.is_last != (s + 1 == stage.size())) {
            err = "build_stages: stage " + std::to_string(s) + " of " + std::to_string(stage.size()) +
                  " says is_first=" + std::to_string(c.is_first) + " is_last=" + std::to_string(c.is_last) +
                  "; the first stage owns the token embeddings and the last one owns the head";
            return false;
        }
        if (c.max_pass != im.cfg.max_pass || c.ctx_max != im.cfg.ctx_max) {
            err = "build_stages: stage " + std::to_string(s) + " has max_pass " + std::to_string(c.max_pass) +
                  " / ctx_max " + std::to_string(c.ctx_max) + ", but the run's are " + std::to_string(im.cfg.max_pass) +
                  " / " + std::to_string(im.cfg.ctx_max) + " (a relayed column is only the same tensor on both sides"
                  " if both sides sized it the same way)";
            return false;
        }
        // P5.3b THE READER'S RULE, and the reason a boundary is legal at all: a stage's layers may read only rings that
        // THIS stage's layers write, because a ring is persistent state and the successor runs on another card and reads
        // it by POINTER inside one graph (a per-token KV relay would be a D2H/H2D of the whole visible cache, every
        // pass).  Since kv_src / idx_src resolve to the latest source <= l and are monotone, that is the same thing as
        // requiring the boundary layer itself to be BOTH a kv and an index source - which is what the geometry's valid
        // set contains ({2, 8, 14, 20} on this model, including the even split at 20).  A boundary whose layers reach
        // below it is refused here, not discovered as a null tensor mid-graph.
        if (le < n_layer) {
            std::string berr;
            if (!split_boundaries_ok(im.g, { le }, berr)) {
                err = "build_stages: stage " + std::to_string(s) + ": " + berr;
                return false;
            }
        }
        // P5.3b: a stage on a FOREIGN backend cannot be in shape (b) - its successor would read a ring that lives on
        // this stage's card, and there is no way to hand over persistent state (the payload is sized per token and a
        // KV ring is not).  A caller may still describe such a plan; build_stages refuses it here, where the config's
        // backend is known.  Read at the boundary layer, this is the same rule as above (its own sources, nothing below
        // them), and it is the stricter wording: it also refuses a boundary layer that runs no attention at all, whose
        // successor would therefore start a stage with no rings of its own.
        if (le < n_layer && cfg[s].backend && cfg[s].backend != im.backend) {
            const int64_t kv = im.g.kv_src(le), ix = im.g.idx_src(le);
            const bool own_sources = im.g.is_kv_source(le) && im.g.is_index_source(le);
            const bool reaches_below = (kv >= 0 && kv < le) || (ix >= 0 && ix < le);
            if (!own_sources || reaches_below) {
                err = "build_stages: stage " + std::to_string(s) + " is on another backend and its successor reads the "
                      "rings of layer " + std::to_string(std::max(kv, ix)) + ", which live on this stage's card; only a "
                      "boundary at the successor's own kv AND index source layer may cross a card";
                return false;
            }
        }
        next_lb = le;
    }
    if (next_lb != n_layer) {
        err = "build_stages: the stages cover " + std::to_string(next_lb) + " of the model's " +
              std::to_string(n_layer) + " layers";
        return false;
    }
    for (size_t s = 0; s < stage.size(); ++s) {
        if (!stage[s]) stage[s] = std::make_unique<Ds41Dense>();   // a caller may pass the vector holding nothing
        Ds41DenseConfig sc = cfg[s];
        const bool same_be = !sc.backend || sc.backend == im.backend;
        if (same_be) {
            sc.share_weights = true;   // init(): geometry, checks and graphs - but no weight tensors of its own
            sc.backend = im.backend;   // the owner's backend; the stage neither creates nor frees one
            // Adopt BEFORE init: init ends by building graphs, and a graph build looks weights up by name.  Borrowing
            // up front is safe under the same invariant as the borrow after it: w_owner is set first and WStore
            // checks `owns`, so a borrowed Impl frees nothing it borrowed.
            adopt_weights_into(*stage[s]->p_, &im);
        }
        if (!stage[s]->init(im.model_path, sc, err)) {
            err = "build_stages: stage " + std::to_string(s) + ": " + err;
            return false;
        }
        Impl & st = *stage[s]->p_;
        if (st.g.n_layer != im.g.n_layer || st.g.n_embd != im.g.n_embd || st.g.hc != im.g.hc ||
            st.g.cand_blocks != im.g.cand_blocks || st.g.cand_layer != im.g.cand_layer) {
            err = "build_stages: stage " + std::to_string(s) + "'s geometry does not match the model this instance loaded";
            return false;
        }
        // A relayed column is only the same tensor on both sides if the stages sized it the same way: every stage gets
        // the OWNER's context limit (the compressed-row counts and every index map derive from it) whatever its config
        // said.  The state COLUMNS stay at the run's width on every stage, including one on another card - a later
        // stage may share the producer's card and read the producer's candidate tensor in place, and a graph that reads
        // it spans the same columns.  What the staging limits is the PASS, which the engine sets per stage with
        // set_pass_cap() after this returns.
        if (same_be) adopt_weights_into(st, &im);   // idempotent re-adopt after the geometry check
        st.staged = true;   // the graph builders read the pass's column window from the impl (see pass_col0)
    }
    return true;
}

// The relay's two halves.  Each region starts at byte 0 of its tensor (the pass's columns are the leading columns of
// a column-major state tensor), so a boundary is relay_slab_bytes()/token contiguous bytes in relay_slab_bytes()'s
// region order; the sync is the native relay's rule - the producer's stream completion observed by the HOST before
// the successor is launched (docs/gfx906/P5-RELAY-PATTERN.md section 1a), no cross-device event anywhere.
bool Ds41Dense::relay_read_out(int n, void * dst, size_t dst_bytes) {
    Impl & im = *p_;
    if (n < 1 || n > (int) im.NC) { im.err = "relay_read_out: pass size out of range"; return false; }
    if (!dst) { im.err = "relay_read_out: no destination"; return false; }
    const size_t xs = (size_t) (im.D * im.HC * 4), ps = (size_t) (im.HC * 4);
    // The producer publishes the candidate column when the geometry can separate it from a reader (cs > 0); the byte
    // count is a function of the GEOMETRY, never of a stage's range, which is the property that makes the two sides of a
    // boundary agree on where the copy ends.  cs is 0 on any model whose caches are its KV / index source lists.
    const size_t cs = relay_slab_bytes() - xs - ps;
    const size_t bytes = ((xs + ps + cs) * (size_t) n);
    if (dst_bytes < bytes) {
        im.err = "relay_read_out: the staging slot holds " + std::to_string(dst_bytes) + " bytes, the pass needs " +
                 std::to_string(bytes);
        return false;
    }
    // P5.3b: the columns the pass just ran in (see relay_write_in's window).  Unsplit, col0 is 0 and this is byte 0.
    const int64_t c0 = im.staged ? im.col0 : 0;
    uint8_t * d = (uint8_t *) dst;
    ggml_backend_synchronize(im.backend);   // finish() is issued async; the payload must be complete
    ggml_backend_tensor_get(im.x_state, d, (size_t) c0 * xs, xs * (size_t) n);
    ggml_backend_tensor_get(im.pre_state, d + xs * (size_t) n, (size_t) c0 * ps, ps * (size_t) n);
    if (cs)   // this stage's own candidate column
        ggml_backend_tensor_get(im.cand, d + (xs + ps) * (size_t) n, (size_t) c0 * cs, cs * (size_t) n);
    return true;
}

// defined below with the span helpers; declared here so relay_write_in can set the pass's ranking span.
static void span_pass_caps(const Ds41Dense::Impl & im, int pos0, int n, int64_t out[3]);

bool Ds41Dense::relay_write_in(int n, const void * src, size_t src_bytes, int64_t src_pos0) {
    Impl & im = *p_;
    if (n < 1 || n > (int) im.NC) { im.err = "relay_write_in: pass size out of range"; return false; }
    if (!src) { im.err = "relay_write_in: no payload"; return false; }
    const size_t xs = (size_t) (im.D * im.HC * 4), ps = (size_t) (im.HC * 4);
    // The whole payload, always: this stage owns a column of its own unless it reads the producer's tensor in place,
    // and even then copying the region it received into the column it will not read is harmless (the region is 8 KiB a
    // token).  Skipping it would make a stage's copy depend on a choice its PRODUCER cannot see, and the two sides'
    // idea of the payload's end is the one thing that must never disagree.
    const size_t cs = relay_slab_bytes() - xs - ps;
    const size_t bytes = ((xs + ps + cs) * (size_t) n);
    if (src_bytes < bytes) {
        im.err = "relay_write_in: the staging slot holds " + std::to_string(src_bytes) + " bytes, the pass needs " +
                 std::to_string(bytes);
        return false;
    }
    // P5.3b: the payload lands in the columns this stage runs the pass in.  Stage 0 writes from column 0 (begin());
    // a relayed stage's window is its own pass cap, and the run's pass loop subdivides a longer pass so every stage
    // stays inside its window (see staged_pass), which keeps the token->column map one-to-one at the boundary.
    const int64_t c0 = im.stage_col0(src_pos0, n);
    if (c0 + n > im.NC || n > im.S) {
        im.err = "relay_write_in: " + std::to_string(n) + " tokens at column " + std::to_string(c0) +
                 " run past this stage's " + std::to_string(im.NC) + " columns / window " + std::to_string(im.S);
        return false;
    }
    const uint8_t * s = (const uint8_t *) src;
    ggml_backend_synchronize(im.backend);   // whatever was reading this stage's state (or the producer's graphs)
    ggml_backend_tensor_set(im.x_state, s, (size_t) c0 * xs, xs * (size_t) n);
    ggml_backend_tensor_set(im.pre_state, s + xs * (size_t) n, (size_t) c0 * ps, ps * (size_t) n);
    if (cs) ggml_backend_tensor_set(im.cand, s + (xs + ps) * (size_t) n, (size_t) c0 * cs, cs * (size_t) n);
    // P5.3b: `src_pos0` is the position the run is feeding (the staging says which COLUMNS the payload covered; the
    // position is what the graphs put into RoPE and the index maps).
    im.pos0 = (int) src_pos0;
    im.n = n;
    im.col0 = c0;
    // the pass's ranking span, by this stage's rule (span_pass_caps subtracts the rows below its range)
    span_pass_caps(im, (int) src_pos0, n, im.pass_cap);
    return true;
}

// P5.3b: read another stage's candidate blocks in place (same card).  See the header: this is only for a same-device
// split that puts cand_layer's producer and its readers on different stages, and it makes this stage's relay payload
// the state region alone (relay_slab_bytes() follows, so the staging is sized consistently).
void Ds41Dense::relay_cand_from(Ds41Dense * src) {
    Impl & im = *p_;
    // NULL = the opposite decision: this stage keeps a candidate region of its own (the relay's default).  An explicit
    // reset matters because build_stages applies the engine's decision to EVERY stage, including ones that never had it.
    if (!src) { im.cand_src_prev = nullptr; return; }
    if (!src->p_->cand) { im.err = "relay_cand_from: the source stage has no candidate tensor"; return; }
    if (src->p_->backend != im.backend) {
        im.err = "relay_cand_from: the source stage runs on another backend; a graph cannot read its tensor";
        return;
    }
    im.cand_src_prev = src->p_->cand;
    // P5.3b: the producer runs BEFORE this stage in every pass, so its candidate column is in the state this stage was
    // handed - EXCEPT when this stage's pass window is narrower than the producer's, which needs the producer's own
    // column count (set_pass_cap's cap; see the run's refusal of that combination).
    im.cand_src_cap = src->p_->S;
}
bool Ds41Dense::relay_owns_cand_region() const {
    return cand_can_cross(p_->g) && !p_->cand_src_prev;
}

void Ds41Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < im.lb || il >= im.state_end() || !im.ly[(size_t) il].rbias) return;   // another stage's layer
    ggml_backend_tensor_set(im.ly[(size_t) il].rbias, bias, 0, (size_t) im.NE * 4);
}

void Ds41Dense::set_oom_hook(std::function<bool()> hook) { p_->oom_hook = std::move(hook); }

void Ds41Dense::release_pass() {
    Impl & im = *p_;
    if (!im.allo_pass.a) return;
    ggml_gallocr_free(im.allo_pass.a);
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.graph_epoch.clear();
    im.allo_pass.compute_bufs.clear();
    ++im.allo_pass.epoch;
}

// pass_cap (per ratio) of a pass of n tokens ending at pos0+n: the visible compressed rows it ranks over.
// P5.3b: IDENTICAL to the unsplit rule, deliberately.  An earlier version subtracted floor(lb / r) here on the theory
// that a stage's ring "holds only its own range's rows" - that theory was wrong twice over.  (1) A compressed ring is
// re-derived by ITS OWN stage's source layer for EVERY pass (all tokens flow through all stages), at MODEL row
// coordinates, so stage 1's ring is as full as the unsplit one - there is nothing to subtract.  (2) The subtraction
// broke the divisibility the cand-block pooling requires: caps are next_pow2(...), hence multiples of CBS, and
// next_pow2(x) - lb/r is not (measured: cap 32 - 20 = 12 with CBS 8 tripped GGML_ASSERT(cap % im.CBS == 0) at
// ds41_dense.cpp:804 on the fourth decode token of witness run3, logs/p5.3b-run3-20261024-213054.log).  The split's
// real constraint is the READER'S RULE (which layer's ring, not how many rows), enforced at build_stages.
static void span_pass_caps(const Ds41Dense::Impl & im, int pos0, int n, int64_t out[3]) {
    out[0] = 0;
    for (int r = 1; r <= 2; ++r) {
        const int64_t nvis_last = (pos0 + n) / r;
        out[r] = nvis_last == 0 ? 0 : std::min<int64_t>(next_pow2(std::max<int64_t>(nvis_last, im.CBS)), im.cmax[r]);
    }
}

// Wait for the pass's Engram reads (started by upload_inputs) and push the rows into the state tensor's columns.  A
// no-op once done (eng_batch is null), so it is safe to call at every layer - but the caller gates it on L.eslot >= 0
// so the layers BELOW the first Engram layer run while the disk is still reading.  The whole pass's rows (both Engram
// slots, all n columns) go up in one shot, so the second Engram layer (14) is a plain read of the same tensor.
static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_batch.reset();
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
        ggml_backend_tensor_set(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
    return true;
}

// Per-token inputs of n tokens at positions pos0.. (embeddings, positions, the window / compressed / indexer index
// maps, the Engram rows and the per-pass ring slots).  Shared by begin() and begin_batch().
bool Ds41Dense::Impl::upload_inputs(const int * tids, int n, int pos0) {
    Impl & im = *this;
    im.span_memo_pos0 = -1;   // this writes i_widx / i_wslot / i_sidx / i_sslot: upload_span's memo is stale
    // P5.3b: a staged pass writes the tables of its own column window; it must not run past the allocation.
    if (col0 + n > NC) { err = "internal: a staged pass runs past the state columns"; return false; }
    const int64_t D = im.D, SWA = im.SWA, RAW = im.RAW;
    // embeddings (F16 rows on the mmap)
    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
        const uint16_t * row = reinterpret_cast<const uint16_t *>(im.embd) + (size_t) tids[t] * (size_t) D;
        for (int64_t i = 0; i < D; ++i) im.host_emb[(size_t) (t * D + i)] = ggml_fp16_to_fp32(row[i]);
    }
    // P5.3b: the same column offset for the embeddings and the positions (only stage 0 has embeddings at all, and a
    // staged piece of it lives at col0, so build_init's b.cols2(..., col0) must read the same slice).
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), (size_t) im.col0 * (size_t) D * 4, im.host_emb.size() * 4);
    std::vector<int32_t> pos(n);
    for (int t = 0; t < n; ++t) pos[t] = pos0 + t;
    ggml_backend_tensor_set(im.i_pos, pos.data(), (size_t) im.col0 * 4, (size_t) n * 4);
    // window: per query the SWA positions [p - SWA + 1, p] as rows of [ring (RAW) | pass rows]
    std::vector<int32_t> widx((size_t) (SWA * n));
    std::vector<float> wm((size_t) (SWA * n));
    for (int t = 0; t < n; ++t) {
        const int64_t p = pos0 + t;
        for (int64_t j = 0; j < SWA; ++j) {
            const int64_t q = p - SWA + 1 + j;
            int32_t row = 0;
            float m = 0.0f;
            if (q < 0) m = NEG_INF;
            else if (q >= pos0) row = (int32_t) (RAW + (q - pos0));
            else row = (int32_t) (q % RAW);
            widx[(size_t) (t * SWA + j)] = row;
            wm[(size_t) (t * SWA + j)] = m;
        }
    }
    // P5.3b: a staged piece writes its own slice of the window tables (offset = the piece's column); unsplit col0 is 0.
    const size_t woff = (size_t) im.col0;
    ggml_backend_tensor_set(im.i_widx, widx.data(), woff * 4, widx.size() * 4);
    ggml_backend_tensor_set(im.i_wmask, wm.data(), woff * (size_t) SWA * 4, wm.size() * 4);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pos0 + n - k + j) % RAW);
        ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    // compressed rows per ratio
    for (int r = 1; r <= 2; ++r) {
        std::vector<float> nv(n);
        std::vector<int32_t> cs(n), cp(n);
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t;
            nv[t] = (float) ((p + 1) / r);
            const bool done = (p + 1) % r == 0;
            cs[t] = (int32_t) (done ? p / r : im.cmax[r]);   // an open group writes the spare row
            cp[t] = (int32_t) ((p / r) * r);
        }
        ggml_backend_tensor_set(im.i_nvis[r], nv.data(), woff * 4, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_cslot[r], cs.data(), woff * 4, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_cpos[r], cp.data(), woff * 4, (size_t) n * 4);
        if (r == 1) {
            std::vector<float> lb(n);
            for (int t = 0; t < n; ++t) lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / im.CBS : -1);
            ggml_backend_tensor_set(im.i_lastblk, lb.data(), woff * 4, (size_t) n * 4);
        }
    }
    {   // ratio-2 state: rows of [ring (2) | pass states] for each token's group (2g, 2g+1); open groups repeat 2g
        std::vector<int32_t> si((size_t) (2 * n)), ss;
        auto row = [&](int64_t q) { return (int32_t) (q >= pos0 ? 2 + (q - pos0) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        const int64_t k = std::min<int64_t>(n, 2);
        for (int64_t j = 0; j < k; ++j) ss.push_back((int32_t) ((pos0 + n - k + j) % 2));
        ggml_backend_tensor_set(im.i_sidx, si.data(), woff * 2 * 4, si.size() * 4);
        ggml_backend_tensor_set(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    // Engram rows: hash the whole pass now and START the (dedup/sort/coalesce) reads; they are waited - and the rows
    // moved to the state tensor - by the first Engram layer's span (engram_ready), so the disk read overlaps the
    // embeddings upload, the init graph and the layers below it.  A tiny batch (decode) still runs on the caller.
    if (im.engram.enabled()) {
        if (!engram_ready(im)) return false;    // flush a still-pending batch from a previous pass first
        for (int t = 0; t < n; ++t) im.engram.set_token(pos0 + t, tids[t]);
        if (n <= 64) {   // decode / verify: the rows are needed at layer 1 - read them now (synchronously)
            static const bool eprof = std::getenv("STRATA_ENG_PROF") != nullptr;
            static double eng_ms = 0; static long eng_n = 0;
            const auto te = std::chrono::steady_clock::now();
            if (!im.engram.rows(pos0, n, im.eng_rows, im.err, im.eng_slots)) return false;
            if (eprof) {
                eng_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
                if (++eng_n % 64 == 0) std::fprintf(stderr, "engram decode reads: %.3f ms/call over %ld calls\n", eng_ms / eng_n, eng_n);
            }
            for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
                ggml_backend_tensor_set(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
        } else {
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err, im.eng_slots);
            if (!im.eng_batch) return false;
        }
    }
    return true;
}

bool Ds41Dense::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (im.lb > 0) { im.err = "begin: stage [" + std::to_string(im.lb) + "," + std::to_string(im.le) +
                                   ") has no token embeddings; only the first stage takes token ids";
                    return false; }
    // P5.3b: a STAGED run cycles the state columns: the run feeds a prompt in passes of at most the window (im.S
    // columns), pass k at column (pos0 / S) % (NC / S), and the token positions the graphs see are the run's own
    // (pos0 + t) - NOT the column.  The rings are position-indexed (slot = pos % RAW for the window ring, row =
    // pos / r for the compressed rows), so reusing a column overwrites the state of position pos - S * NC/S, a
    // position SWA tokens back and therefore outside every later query's reach.  Unsplit, this is pos0 at column 0.
    int64_t p0 = pos0;
    if (im.staged) {
        if (n < 1 || n > im.S || pos0 < 0) {
            im.err = "begin: a staged pass takes 1.." + std::to_string(im.S) + " tokens (got " +
                     std::to_string(n) + " at position " + std::to_string(pos0) + ")";
            return false;
        }
        if (im.cand_src_prev && im.cand_src_cap != im.S) {
            im.err = "stage reads another stage's candidate column, whose writer aligns to a window of " +
                     std::to_string(im.cand_src_cap) + " columns but this stage's window is " + std::to_string(im.S) +
                     ": the pass would read the producer's column at the wrong offset";
            return false;
        }
        im.col0 = im.stage_col0(pos0, n);
    }
    if (n < 1 || n > im.NC || p0 < 0 || p0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.pos0 = (int) p0;
    im.n = n;
    if (!im.upload_inputs(tids, n, (int) p0)) return false;
    // (span_pass_caps applies this stage's ring correction; see its comment.)
    span_pass_caps(im, pos0, n, im.pass_cap);
    // embeddings -> state
    ggml_context * gc = nullptr;
    ggml_cgraph * gf = nullptr;
    const bool cached = n == 1;
    std::vector<uint8_t> meta;
    if (cached) {
        auto it = im.graphs.find({ -1, 0, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Impl::GKey { -1, 0, 0, 1 }, build_init(im, im.gctx, 1)).first;
        gf = it->second;
    } else {
        meta.resize(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_init(im, gc, n);
    }
    const bool ok = im.compute(gf, cached);
    if (gc) ggml_free(gc);
    return ok;
}

// ================================================================== the layer-major prompt batch

bool Ds41Dense::begin_batch(const int * tids, int B, int pos0) {
    Impl & im = *p_;
    if (im.lb > 0) { im.err = "begin_batch: stage [" + std::to_string(im.lb) + "," + std::to_string(im.le) +
                                       ") has no token embeddings; only the first stage takes token ids";
                     return false; }
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) { im.err = "bad batch"; return false; }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.pos0 = pos0; im.batch_pos0 = pos0; im.batch = B; im.n = B; im.col0 = 0;
    // per-sub-batch pass_cap (the compressed rows each sub-batch ranks over) - depends only on (pos0, len)
    const int64_t nb = (B + im.S - 1) / im.S;
    im.span_cap.assign((size_t) nb, { 0, 0, 0 });
    for (int64_t i = 0; i < nb; ++i)
        span_pass_caps(im, pos0 + (int) (i * im.S), (int) std::min<int64_t>(im.S, (int64_t) B - i * im.S),
                       im.span_cap[(size_t) i].data());
    // the batch's initial state, one sub-batch at a time (x_state = embd, pre_state = one-hot)
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
    for (int64_t i = 0; i < nb; ++i) {
        const int64_t c0 = i * im.S, sn = std::min<int64_t>(im.S, (int64_t) B - c0);
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = build_init(im, gc, sn, c0);
        const bool ok = im.compute(gf, false);
        ggml_free(gc);
        if (!ok) { im.err = "batch init failed"; return false; }
    }
    return true;
}

// The per-span inputs of one layer-major sub-batch: the sub-batch's own pass_cap (the graphs read it) and the
// PASS-LOCAL row maps (the pass's own rows / states sit at the end of the [ring | pass rows] they index, so they are
// recomputed from the sub-batch's own position - upload_inputs laid them out for the whole batch, correct only at
// col0 = 0).  Shared by attn_router_span and attn_router_span_kv.
static bool upload_span(Ds41Dense::Impl & im, int col0, int n) {
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const size_t si = (size_t) (col0 / im.S);
    if (si >= im.span_cap.size()) { im.err = "bad span"; return false; }
    im.pass_cap[0] = im.span_cap[si][0];
    im.pass_cap[1] = im.span_cap[si][1];
    im.pass_cap[2] = im.span_cap[si][2];
    const int64_t pbase = (int64_t) im.batch_pos0 + col0;
    static const bool memo_on = !(std::getenv("STRATA_SPAN_MEMO") && std::getenv("STRATA_SPAN_MEMO")[0] == '0');
    if (memo_on && im.span_memo_pos0 == (int64_t) im.batch_pos0 && im.span_memo_col0 == col0 && im.span_memo_n == n) return true;
    im.span_memo_pos0 = (int64_t) im.batch_pos0;
    im.span_memo_col0 = col0;
    im.span_memo_n = n;
    {
        std::vector<int32_t> widx((size_t) (im.SWA * n));
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t;
            for (int64_t j = 0; j < im.SWA; ++j) {
                const int64_t q = p - im.SWA + 1 + j;
                widx[(size_t) (t * im.SWA + j)] = q < 0 ? 0 : (int32_t) (q >= pbase ? im.RAW + (q - pbase) : q % im.RAW);
            }
        }
        ggml_backend_tensor_set(im.i_widx, widx.data(), (size_t) col0 * im.SWA * 4, widx.size() * 4);
    }
    {
        std::vector<int32_t> si((size_t) (2 * n));
        auto row = [&](int64_t q) { return (int32_t) (q >= pbase ? 2 + (q - pbase) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        ggml_backend_tensor_set(im.i_sidx, si.data(), (size_t) col0 * 2 * 4, si.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pbase + n - k + j) % im.RAW);
        ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, 2);
        std::vector<int32_t> ss((size_t) k);
        for (int64_t j = 0; j < k; ++j) ss[(size_t) j] = (int32_t) ((pbase + n - k + j) % 2);
        ggml_backend_tensor_set(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    return true;
}

bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    if (!im.owns_layer(il)) { im.err = "attn_router_span_kv: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    if (!upload_span(im, col0, n)) return false;
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // the Engram add feeds this layer's K/V
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params pk = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gck = ggml_init(pk);
    ggml_cgraph * gk = build_attn_kv_only(im, gck, il, n, col0);
    const bool okk = im.compute(gk, false);
    ggml_free(gck);
    return okk;
}

bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    if (!im.owns_layer(il)) { im.err = "attn_router_span: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    double tp = g_dprof_on ? dprof_now() : 0;
    auto lap = [&](int i) { if (g_dprof_on) { const double t = dprof_now(); g_dprof[i] += t - tp; tp = t; } };
    if (!upload_span(im, col0, n)) return false;
    if (g_dprof_on && dprof_now() - tp > 50) std::fprintf(stderr, "dense prof: slow upload_span layer %d col0 %d n %d: %.0f ms\n", il, col0, n, dprof_now() - tp);
    lap(6);
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // wait the pass's Engram reads here
    lap(3);
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_attn(im, gc, il, n, col0);
    lap(0);
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    if (g_dprof_on) { tp = dprof_now(); g_dprof[7] += 1; }
    ggml_free(gc);
    if (!ok) return false;
    // the sub-batch's ids / wts / fn land at its own batch columns (token-major: token col0+t at t * NU / t * D)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data() + (size_t) col0 * im.D, (size_t) col0 * im.D * 4, (size_t) n * im.D * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ggml_backend_synchronize(im.backend);
    lap(4);
    return true;
}

bool Ds41Dense::upload_routed(const float * routed_host, int B) {
    Impl & im = *p_;
    if (!routed_host || B < 1 || B > im.batch) { im.err = "bad routed batch"; return false; }
    ggml_backend_tensor_set(im.routed, routed_host, 0, (size_t) B * im.D * 4);
    return true;
}

float * Ds41Dense::routed_dev() { return p_->cpu || !p_->routed ? nullptr : (float *) p_->routed->data; }

bool Ds41Dense::finish_span(int il, int col0, int n) {
    Impl & im = *p_;
    if (!im.owns_layer(il)) { im.err = "finish_span: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    (void) il;
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const double tf0 = g_dprof_on ? dprof_now() : 0;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    ggml_free(gc);
    return ok;
}

void Ds41Dense::prof_report(const char * tag) {
    if (!g_dprof_on) return;
    std::fprintf(stderr, "dense prof %s: %d attn spans: build %.0f ms, alloc %.0f, gpu compute %.0f, engram wait %.0f, "
                 "readback %.0f, upload %.0f, finish build %.0f\n", tag, (int) g_dprof[7], g_dprof[0], g_dprof[1], g_dprof[2],
                 g_dprof[3], g_dprof[4], g_dprof[6], g_dprof[5]);
    for (double & v : g_dprof) v = 0;
}

int Ds41Dense::batch_size() const { return (int) p_->batch; }
const int32_t * Ds41Dense::batch_ids() const { return p_->host_ids.data(); }
const float * Ds41Dense::batch_wts() const { return p_->host_wts.data(); }
const float * Ds41Dense::batch_fn() const { return p_->host_fn.data(); }

bool Ds41Dense::attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    Impl & im = *p_;
    if (!im.owns_layer(il)) { im.err = "attn_router: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    const int64_t n = im.n;
    // Engram layer: wait the pass's reads (started in upload_inputs) before the graph that reads i_eng runs.  For the
    // chunk path this is the first Engram layer too, so the layers below it ran while the disk was reading.
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    const bool cached = n == 1;
    if (cached) {
        const Impl::GKey key { il, 1, cap, 1 };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, 1)).first;
        gf = it->second;
    } else {
        im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
        ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
        gc = ggml_init(p);
        gf = build_attn(im, gc, il, n);
    }
    const bool ok = im.compute(gf, cached);
    if (gc) ggml_free(gc);
    if (!ok) return false;
    // one sync for the three readbacks (was one per tensor_get: 3 x 40 stream syncs per decode token)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_synchronize(im.backend);
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    return true;
}

bool Ds41Dense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (!im.owns_layer(il)) { im.err = "predict: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    if (im.n != 1 || n_top < 1 || n_top > 16) return false;
    const Impl::GKey key { il, 3, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out[il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    ggml_backend_tensor_get(im.predict_out[il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::predict_multi(int il, int * top_ids) {
    Impl & im = *p_;
    if (!im.owns_layer(il)) { im.err = "predict_multi: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    const int64_t n = im.n;
    if (n < 1) return false;
    if (n == 1) return predict(il, top_ids, 16);
    const Impl::GKey key { il, 3, 0, (int) n };   // one graph per (layer, pass width): verify passes reuse a few widths
    auto it = im.graphs.find(key);
    ggml_tensor * o = nullptr;
    if (it == im.graphs.end()) {
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o, n);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out_n[{ il, (int) n }] = o;
    }
    o = im.predict_out_n[{ il, (int) n }];
    if (!im.compute(it->second, true)) return false;
    ggml_backend_tensor_get(o, top_ids, 0, sizeof(int32_t) * 16 * (size_t) n);
    return true;
}

bool Ds41Dense::finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    (void) il;
    if (!im.owns_layer(il)) { im.err = "finish: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    const int64_t n = im.n;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) (im.D * n) * 4);
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    const bool cached = n == 1;
    std::vector<uint8_t> meta;
    if (cached) {
        auto it = im.graphs.find({ -1, 2, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Impl::GKey { -1, 2, 0, 1 }, build_finish(im, im.gctx, 1)).first;
        gf = it->second;
    } else {
        meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_finish(im, gc, n);
    }
    const bool ok = im.compute(gf, cached);
    if (gc) ggml_free(gc);
    return ok;
}

bool Ds41Dense::finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) {
    Impl & im = *p_;
    (void) il;
    if (!im.owns_layer(il)) { im.err = "finish_predict: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    if (im.n != 1 || n_top < 1 || n_top > 16) return false;
    // P5.3b: the fused graph runs on THIS stage's device with THIS stage's weight bounds: next_il must be the
    // layer the successor call will ask this stage to run, i.e. inside [lb, wle).  A staged loop's last layer of
    // a stage calls with next_il == le, which is exactly why wle is le + 1.
    if (next_il < im.lb || next_il >= im.wle) return false;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 5, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.fpredict_out[next_il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    ggml_backend_tensor_get(im.fpredict_out[next_il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                                     int * ahead_ids) {
    Impl & im = *p_;
    (void) il;
    if (!im.owns_layer(il)) { im.err = "finish_predict_ahead: layer " + std::to_string(il) + " is not this stage's [" + std::to_string(im.lb) + "," + std::to_string(im.le) + ")"; return false; }
    if (ahead <= 0 || !ahead_ids) return finish_predict(il, routed_sum, next_il, top_ids, n_top);
    if (im.n != 1 || n_top < 1 || n_top > 16) return false;
    if (next_il < im.lb || next_il >= im.wle) return false;   // P5.3b: as in finish_predict
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 6, ahead, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        std::vector<ggml_tensor *> ao;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o, ahead, &ao);
        it = im.graphs.emplace(key, gf).first;
        ao.insert(ao.begin(), o);   // [0] = next_il's prediction, [1 + a] = next_il + 1 + a's
        im.fpredict_ahead[{ next_il, ahead }] = ao;
    }
    if (!im.compute(it->second, true)) return false;
    const auto & ao = im.fpredict_ahead[{ next_il, ahead }];   // [0] = the output, then one entry per extra row
    // argsort_top_k is a VIEW of the full argsort: row r starts at r * nb[1] (NE ids apart), not r * 16 - one copy of
    // the span covering every row's first 16
    const int rows = (int) ao.size();
    const size_t rs = ao[0]->nb[1] / 4;
    std::vector<int32_t> buf(rs * (size_t) (rows - 1) + 16);
    ggml_backend_tensor_get(ao[0], buf.data(), 0, buf.size() * 4);
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[(size_t) i];
    for (int a = 0; a < ahead; ++a) {
        if (a + 1 < rows) std::copy(buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1)), buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1) + 16),
                                    ahead_ids + (size_t) a * 16);
        else std::fill(ahead_ids + (size_t) a * 16, ahead_ids + (size_t) (a + 1) * 16, -1);
    }
    return true;
}

bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    if (!im.is_last) { im.err = "logits: this stage is not the last one (no output head)"; return false; }
    if (row0 < 0 || nrows < 1 || row0 + nrows > im.n) { im.err = "bad logits rows"; return false; }
    // P5.3b: the caller counts rows from the start of the last pass; a staged pass's rows live at im.col0.
    const int64_t hrow0 = im.staged ? im.col0 + row0 : row0;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_tensor * lg = nullptr;
    ggml_cgraph * gf = build_head(im, gc, hrow0, nrows, &lg);
    const bool ok = im.compute(gf, false);
    if (ok) ggml_backend_tensor_get(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    return ok;
}

bool Ds41Dense::read_state(std::vector<float> & x, std::vector<float> & pre) {
    Impl & im = *p_;
    x.resize((size_t) (im.D * im.HC * im.n));
    pre.resize((size_t) (im.HC * im.n));
    ggml_backend_tensor_get(im.x_state, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_get(im.pre_state, pre.data(), 0, pre.size() * 4);
    return true;
}

bool Ds41Dense::read_state_cols(int c0, int n, std::vector<float> & x) {
    Impl & im = *p_;
    if (c0 < 0 || n < 1 || c0 + n > im.NC) { im.err = "read_state_cols: bad range"; return false; }
    const size_t col = (size_t) (im.D * im.HC) * sizeof(float);   // x_state [D, HC, NC]: a column is contiguous
    x.resize((size_t) (im.D * im.HC * n));
    ggml_backend_tensor_get(im.x_state, x.data(), (size_t) c0 * col, (size_t) n * col);
    return true;
}

bool Ds41Dense::snapshot(int pos) {
    Impl & im = *p_;
    // P5.3b: on a staged run the layers outside this stage's range own no state to snapshot.
    for (auto & L : im.ly) {
        if (!L.raw) continue;
        ggml_backend_tensor_copy(L.raw, L.raw_snap);
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv, L.st_kv_snap); ggml_backend_tensor_copy(L.st_sc, L.st_sc_snap); }
    }
    ggml_backend_synchronize(im.backend);
    im.snap_pos = pos;
    return true;
}

bool Ds41Dense::restore() {
    Impl & im = *p_;
    if (im.snap_pos < 0) return false;
    for (auto & L : im.ly) {
        if (!L.raw) continue;   // P5.3b: another stage's layer (see snapshot())
        ggml_backend_tensor_copy(L.raw_snap, L.raw);
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv_snap, L.st_kv); ggml_backend_tensor_copy(L.st_sc_snap, L.st_sc); }
    }
    ggml_backend_synchronize(im.backend);
    return true;
}

int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }

bool Ds41Dense::park(int64_t positions) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1) return false;
    ggml_backend_synchronize(im.backend);
    im.parked.clear();
    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n) };
        ggml_backend_tensor_get(t, r.b.data(), off, n);
        im.parked.push_back(std::move(r));
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) put(t, 0, ggml_nbytes(t));
        for (ggml_tensor * t : { L.comp, L.idxk }) {   // row r holds a position >= r (ratio 1 or 2): rows [0, n] cover n positions
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            put(t, 0, (size_t) n * t->nb[1]);
            put(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);   // the spare row (open compressor groups write it)
        }
    }
    im.parked_pos = im.snap_pos;
    return true;
}

bool Ds41Dense::unpark() {
    Impl & im = *p_;
    if (im.parked_pos < 0) return false;
    ggml_backend_synchronize(im.backend);
    for (auto & r : im.parked) ggml_backend_tensor_set(r.t, r.b.data(), r.off, r.b.size());
    ggml_backend_synchronize(im.backend);
    im.snap_pos = im.parked_pos;
    drop_park();
    return true;
}

void Ds41Dense::drop_park() {
    p_->parked.clear();
    p_->parked.shrink_to_fit();
    p_->parked_pos = -1;
}

bool Ds41Dense::parked() const { return p_->parked_pos >= 0; }

bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
    Impl & im = *p_;
    const auto & L = im.ly[(size_t) il];
    sel.clear();
    mask.clear();
    *K = 0;
    if (!im.owns_layer(il) || !L.idx_src || !L.ratio) return true;   // P5.3b: another stage's layer: the empty answer
    const int64_t cap = im.pass_cap[L.ratio];
    if (cap == 0) return true;
    const int64_t k = std::min<int64_t>(cap, im.ITOP);
    std::vector<int32_t> s((size_t) (im.ITOP * im.n));
    std::vector<float> m((size_t) (im.ITOP * im.n));
    ggml_backend_tensor_get(L.sel, s.data(), 0, s.size() * 4);
    ggml_backend_tensor_get(L.selm, m.data(), 0, m.size() * 4);
    for (int t = 0; t < im.n; ++t)
        for (int64_t j = 0; j < k; ++j) {
            sel.push_back(s[(size_t) (t * im.ITOP + j)]);
            mask.push_back(m[(size_t) (t * im.ITOP + j)]);
        }
    *K = (int) k;
    return true;
}
