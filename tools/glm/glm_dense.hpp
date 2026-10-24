// tools/glm/glm_dense.hpp - GLM-5.3-Flash (`glm5-next`) decode of the dense (non-expert) half.
//
// The DS4 engine's shape (tools/ds4/ds4_dense.hpp) for GLM's layers: mHC streams (identical to DeepSeek V4), KDA
// linear attention (34 layers, recurrent state), nope-MLA (11 layers, latent cache), dense FFN (layers 0-2) and the
// MoE router + shared expert (layers 3-44).  The routed experts are Ds4MoeTier's (tools/ds4/ds4_moe.hpp); this class
// hands out the ffn_norm activation they consume and takes back their weighted sum in `finish_layer`.
//
//   begin_tokens(tids, n)                 -> hc_init for n consecutive tokens
//   predict(l, top_ids, n_top)            -> prefetch hint (the layer's router on rms_norm(hc_attn_pre)*ffn_norm.w)
//   attn_router_n(l, pos0, n, ...)        -> attention + router ids/weights + shared expert (or the dense FFN on
//                                            layers 0-2: no routed experts, `routed()` is false)
//   finish_layer_n(l, n, routed_sum)      -> shared/dense FFN + routed sum, hc_post -> next layer state
//   logits_n(n, out)                      -> mean of the 4 streams, output_norm, output
//
// Formulas: docs/glm/ARCHITECTURE.md (from neurall/llama.cpp src/models/glm5-next.cpp; the graph mirrors it op for op).
//
// Phase 1 limits (checked, error otherwise):
//  - The lightning indexer is not built: MLA attends every earlier position, which is exactly the model while the
//    context holds <= indexer_top_k + kpool - 1 tokens (2051).  Longer contexts need the indexer (phase 2).
//  - KDA state is recurrent, not position-indexed: a position cannot be decoded twice (no speculative rollback yet;
//    ggml_gated_delta_net's K-snapshot slots are the planned mechanism).  Positions must arrive in order.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

namespace strata::glm { class LoraAdapter; }

/// What the GGUF says about the dense half (glm5-next.* keys).
struct GlmGeometry {
    int64_t n_layer = 0, n_embd = 0, n_head = 0, vocab = 0, n_ff = 0;
    int64_t hc = 4, hc_sinkhorn_iters = 20;
    double  hc_eps = 1e-6, rms_eps = 1e-5;
    int64_t kda_head_dim = 128, d_conv = 4;
    double  kda_gate_lower_bound = -5.0;
    int64_t q_lora = 1536, kv_lora = 512, k_mla = 256, v_mla = 256;
    int64_t idx_top_k = 2048, idx_kpool = 4, idx_heads = 32, idx_dim = 128;
    double  ln_eps = 1e-6;              ///< attention.layer_norm_epsilon (the indexer key LayerNorm)
    int64_t n_expert = 288, n_expert_used = 8, n_ff_exp = 2048, n_dense_lead = 3;
    double  expert_weights_scale = 2.5;
    bool    expert_weights_norm = true;
    std::vector<double> swiglu_clamp_exp, swiglu_clamp_shexp;
    std::vector<bool>   is_kda;   ///< per layer: head_count_kv == 0
    int64_t context_length = 0;

    bool kda(int64_t l) const { return is_kda[(size_t) l]; }
    bool routed(int64_t l) const { return l >= n_dense_lead; }
    /// Longest context the indexer-free attention is exact for.
    int64_t dense_attn_ctx() const { return idx_top_k + idx_kpool - 1; }
};

struct GlmDenseConfig {
    ggml_backend_t backend = nullptr;   ///< NULL: create and own a CPU backend
    int     n_threads = 8;
    int64_t ctx = 4096;                 ///< MLA latent-cache rows (positions) to allocate
    int     max_tokens = 4;             ///< tokens one pass may carry (decode 1; prompt chunks up to this, <= 4096)
    bool    skip_routed_experts = true; ///< the tier owns ffn_*_exps (never upload them)
    bool    gate_taps = false;          ///< keep host copies of l_out per layer (tap_l_out) for the gates
    bool    allow_long_ctx = false;     ///< dense MLA at any length (NOT the model's math past dense_attn_ctx())
    /// Run-time rank-1 adapter (tools/glm/glm_lora.hpp), or NULL.  The dense half applies the adapter's whole-module
    /// targets (`attn_output`, `ffn_down_shexp`) in the graph as y += B (A x); the routed experts' adapter entries
    /// are Ds4MoeTier's (set separately on the tier).  The adapter must outlive the GlmDense.
    const strata::glm::LoraAdapter * lora = nullptr;
    /// Requantize the dense half's Q6_K matrices (attention, shared experts, dense FFN, output head) to this ggml
    /// type at load (-1 = keep).  Maya-S-v2 -> Q4_K = Maya-S24's layout (its card: +14% decode on a 24 GB card,
    /// 97.7% vs 97.9% of FP8): ~1.9 GB less dense read per token and ~1.9 GB more VRAM for expert slots.
    int dense_requant = -1;
    /// P5.3 step 1 (set by build_stages, not by callers): read the geometry and build the state and graphs but take NO
    /// weight tensors of our own; the owner hands its own over immediately after init.
    bool share_weights = false;
    /// P5.3 step 1 (opt-in): this instance owns the layer range [lb, le) and nothing else.  Defaults: the whole
    /// model, which is exactly the single-stage construction of the last release, byte for byte.  With a range that
    /// does not start at 0 the token embeddings are not needed and are not looked up; with one that does not end at
    /// n_layer the LM head and output_norm are not loaded, so no stage but the last holds them.  Per-layer state
    /// (KDA S + conv, the MLA latent cache, the indexer pools) is allocated for this range's layers only, and the
    /// layer graphs run only over this stage's layers; the relay of tools/gfx906/relay_staging.hpp hands the state
    /// region over at the stage's end.
    int64_t lb = 0, le = 0;   ///< le 0 = n_layer (filled in from the GGUF when not given)
    bool is_first = true, is_last = true;
};

class GlmDense {
public:
    struct Impl;

    GlmDense();
    ~GlmDense();
    GlmDense(const GlmDense &) = delete;
    GlmDense & operator=(const GlmDense &) = delete;

    bool init(const std::string & model_path, const GlmDenseConfig & cfg, std::string & err);

    /// P5.3 step 1: the stage this instance runs (cfg.lb/cfg.le, or the whole model for the default config), and
    /// whether it is the one that owns the LM head.  With no range given these answer like a single-stage engine.
    int64_t stage_begin() const;
    int64_t stage_end() const;
    int64_t stage_layers() const;   ///< the layers THIS stage owns (the full model's count when unsplit)
    int64_t n_layer_total() const;  ///< the model's layer count, whatever the stage
    bool is_last_stage() const;
    /// P5.3 step 1 (opt-in): does this instance own its weights, or were they shared with it by build_stages()?  A
    /// shared instance must be destroyed BEFORE the instance that owns the weights.
    bool weights_shared() const;
    /// P5.3 step 1 (opt-in): build the STAGES of a split run on this instance's weights instead of loading the model
    /// once per stage.  Every `stage` is init'd with its own config (its layer range, its flags) and then uses the
    /// WEIGHT TENSORS of this instance rather than its own copies: on a split, stage 0 also holds the token embeddings
    /// and the last stage also holds the output head, so loading each stage separately costs MORE memory than the whole
    /// model, which defeats the point of a split on one card.  The weights a stage's range needs are a subset of what
    /// the whole model needs, so sharing is always safe; each stage still gets its own state (KDA/conv, MLA cache,
    /// indexer pools) and its own graphs.  This instance must outlive every stage.  The stage
    /// ranges must tile the model's layers in order and each config's is_first/is_last must match its position.
    /// On failure, `err` says which stage and why, and no stage is left half-built.
    /// Stages are held as unique_ptrs (GlmDense is pimpl'd and deliberately not movable); a null entry is created.
    bool build_stages(std::vector<std::unique_ptr<GlmDense>> & stage, const std::vector<GlmDenseConfig> & cfg,
                      std::string & err);

    const GlmGeometry & geom() const;
    int64_t n_layer() const;
    int max_tokens() const;
    ggml_backend_t backend() const;

    bool begin_tokens(const int * tids, int n);
    /// Frees the prompt chunks' shared compute buffer (re-acquired by the next chunk): its VRAM goes back to the
    /// device after a prompt, for the elastic expert cache.
    void release_big();
    /// The largest prompt chunk (<= want) starting at pos0 whose MLA cap x chunk stays within the VRAM-safe
    /// product (s28: the indexer's [pools x tokens] scratch grows with cap x n; 65536 x 4096 is the measured-safe size
    /// at margin 7.5).  Halves the chunk for every cap doubling past 64K - long prompts slow down, never OOM.
    int safe_chunk(int pos0, int want) const;
    bool begin_token(int tid) { return begin_tokens(&tid, 1); }
    /// Router hint for token 0 of the pass; false (no hint) on a dense layer.
    bool predict(int il, int * top_ids, int n_top);
    /// n_expert_used ids/weights per token (token-major); *ffn_norm_host = n * n_embd floats (token-major).  On a dense
    /// layer (il < n_dense_lead) no ids are written and the caller passes a zero routed sum to finish_layer_n.
    /// `il` is a GLOBAL layer index: on a stage whose range does not contain it, this fails.
    bool attn_router_n(int il, int pos0, int n, int * routed_ids, float * routed_w, const float ** ffn_norm_host);
    bool finish_layer_n(int il, int n, const float * routed_sum);
    bool logits_n(int n, const float ** out, int * n_vocab);
    /// Logits of rows [r0, r0+nr) of the last pass (nr <= 16) into out[nr * vocab]; works for any pass size (prompt
    /// chunks: their head is never built for all n rows - 2048 x 154880 floats would not fit).
    bool logits_rows(int r0, int nr, float * out);
    /// The bytes per token of the relay payload: x_state's token-column stride = n_embd * hc * sizeof(float).
    /// The same number on every stage of a split run, which is what makes a boundary a single contiguous copy.
    size_t relay_slab_bytes() const;
    /// P5.3 step 1 (opt-in): the two halves of a stage boundary, for the engine's loop (glm_generate.cpp).
    /// The pass's state is x_state's token columns [0,n) at byte 0 on EVERY stage, so a boundary is one contiguous
    /// copy of n * relay_slab_bytes() bytes.  relay_read_out() copies the producer's finished state out of its
    /// x_state into the staging slot ~dst~, synchronizing the backend first (the payload must be complete before
    /// the successor is launched - the native relay's rule).  relay_write_in() uploads the staging slot straight
    /// into this stage's state region, which its first layer then reads.  Any stage can do either.
    bool relay_read_out(int n, void * dst, size_t dst_bytes);
    bool relay_write_in(int n, const void * src, size_t src_bytes);
    /// P5.3 step 1: tell a LATER stage that the pass at positions [pos0, pos0+n) is the next one.  It owns no token
    /// ids, so it cannot see positions advance on its own, and its ordered-position rule (the KDA state cannot
    /// rewind) still has to hold.  Refused, with the same wording as the attention check, if pos0 is not where its
    /// recurrent state says the next pass starts.  The first stage tracks its own positions and refuses this call.
    bool advance_stage_pos(int pos0, int n);
    /// Selection-only bias per expert of layer il (cache-aware routing); zeros = the model's routing.
    void set_route_bias(int il, const float * bias);
    /// Forget every position (KDA state, conv state, MLA cache rows are simply overwritten from position 0 on).
    void reset();
    /// Conversation reuse (serve mode): save the recurrent state - KDA S + conv states, the indexer's tail carry -
    /// and the position.  MLA latent rows and complete indexer pools below that position are never rewritten, so
    /// restore() rewinds the model exactly to it (~70 MB device copies, not the whole latent cache).
    bool snapshot();
    bool restore();
    int snapshot_pos() const;   ///< -1: none

    const float * tap_l_out(int il) const;   ///< [n_embd * hc] of the last token of the last pass (gate_taps)
    const std::string & last_error() const;

private:
    std::unique_ptr<Impl> p_;
};
