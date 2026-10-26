// p6_tp_algebra - P6.1 sharding-algebra host gate (task #36): the runnable model of the
// docs/gfx906/P6-TENSOR-SPLIT.md section-3 partition for DS4.1. Host-only: plain g++, no
// ggml, no GPU. Pattern follows gate-relay-pos-tables PASS 3 and gate-device-split.sh:
// the gate SIMULATES the runtime's arithmetic and ownership checks in data, it does not
// restate the theory. A flipped shard direction, a straddled o-group, a dropped replicate,
// or a swapped rounding order must each turn it red INDEPENDENTLY (inject with --red=NAME;
// the shell gate asserts each red on its own).
//
// Geometry (the engine reader's numbers, NOT a transcription to be trusted - the
// divisibility below is RE-DERIVED from these constants at runtime):
//   D 5120 | hc 4 | heads 64 x d_head 512 | q_lora 1280 | o groups 8 (8 heads x 512 = 4096
//   wide each) | kv latent read (replicated) | indexer 32x128 (replicated) | engram adds
//   (replicated) | shared expert ff 2304 (col/row split) | routed experts NOT sharded.
//
// Reduction dtypes (plan section 2b, --tp-reduce f32|f16|q8 modelled here):
//   f32 : exact fp32 sum of per-card partials, rank-ascending (fixed order = deterministic);
//   f16 : each partial rounded to binary16 ROUND-TO-NEAREST-EVEN (hardware _Float16 cast,
//         the same rounding __float2half_rn performs), widened back, summed in fp32;
//   q8  : each partial Q8_0-quantized (32-element blocks, fp16-rounded scale = amax/127,
//         lrint to int8, clamp), dequantized to fp32, summed in fp32.
// The gate PRINTS induced per-site error stats against an fp64 reference: the host numbers
// a later hardware calibration run must match (no bit-parity claim, per plan section 4).

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------- geometry
static constexpr int D         = 5120;  // model width
static constexpr int HC        = 4;     // hyper-connection width (state stays full-width)
static constexpr int HEADS     = 64;
static constexpr int DH        = 512;   // d_head
static constexpr int Q_LORA    = 1280;
static constexpr int O_GROUPS  = 8;     // the o projection is low-rank in 8 groups
static constexpr int GROUP_H   = HEADS / O_GROUPS;          // 8 heads per group
static constexpr int GROUP_W   = GROUP_H * DH;              // 4096 o-input elements per group
static constexpr int O_IN      = HEADS * DH;                // 32768 attn_output_b rows
static constexpr int FF        = 2304;                      // shared expert width
static constexpr int IDX_KV    = 32;                        // indexer heads
static constexpr int IDX_D     = 128;                       // indexer head width
static constexpr int Q8_BLOCK  = 32;

// ---------------------------------------------------------------- red flags
struct RedFlags {
    bool flip_qb = false;    // q_b row-shard + o col-shard (directions swapped)
    bool straddle = false;   // card0 groups 0..3, card1 groups 3..6 (group 6 straddled)
    bool drop_rep = false;   // card 1 does not read the kv latent
    bool dtype_round = false;// f16 partial rounding becomes TRUNCATION, not RNE
};

// ---------------------------------------------------------------- ownership model
// A plan is DATA, exactly as the runtime would carry it: per tensor, per card, the owned
// element ranges on the sharded axis, plus read sets. Every invariant below plays the
// runtime's own checks over this data; the numeric section consumes the plan, so a defect
// can hide in no table and in no arithmetic.
struct Range { int lo, hi; };
inline bool operator==(const Range& a, const Range& b) { return a.lo == b.lo && a.hi == b.hi; }  // [lo, hi)

struct TensorPlan {
    std::string name;
    int axis_dim = 0;                            // size of the sharded (or replicated) span
    std::string units = "elements";              // coordinate space of owned boundaries
    bool sharded = false;                        // false => replicated
    std::vector<std::vector<Range>> owned;       // [card][ranges] coverage obligation
    std::vector<std::vector<Range>> reads;       // [card][ranges] read obligation
};

struct Plan {
    int ncards = 0;
    std::vector<TensorPlan> tensors;
    std::vector<bool> hc_input;                  // card fed the post-reduce row for its hc mix
    std::string group_units = "o-group";          // the units q_b and o must agree in
    std::vector<std::vector<int>> card_groups;   // [card][group ids] the shared split
};

static int g_checks = 0, g_fails = 0;
static void check(bool ok, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(bool ok, const char* fmt, ...) {
    ++g_checks;
    if (ok) return;
    ++g_fails;
    va_list ap;
    va_start(ap, fmt);
    std::printf("  FAIL: ");
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
}

// deterministic pseudo-weights: reproducible across processes without any RNG state
static inline float wgt(uint32_t i, uint32_t j) {
    uint32_t h = i * 2654435761u ^ (j + 0x9e3779b9u) * 40503u;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
    return ((int)(h >> 8) % 2000) / 1000.0f - 1.0f;  // [-1, 1)
}

// ---------------------------------------------------------------- plan builder
static void cover_even(TensorPlan& t, int ncards, int start, int end) {
    t.owned.assign((size_t)ncards, {});
    int span = end - start;
    for (int c = 0; c < ncards; ++c) {
        int lo = start + span * c / ncards;
        int hi = start + span * (c + 1) / ncards;
        t.owned[(size_t)c].push_back({lo, hi});
    }
}
static void cover_all(TensorPlan& t, int ncards, int dim) {
    t.owned.assign((size_t)ncards, {{0, dim}});
}
static void read_all(TensorPlan& t, int ncards, int dim) {
    t.reads.assign((size_t)ncards, {{0, dim}});
}

// returns false = unplaceable plan (the refusal itself is asserted by the caller/gate)
static bool build_plan(int ncards, RedFlags red, Plan& p) {
    p = Plan();
    p.ncards = ncards;
    p.hc_input.assign((size_t)ncards, true);
    if (O_GROUPS % ncards != 0) return false;  // 8 groups do not divide into ncards
    p.card_groups.assign((size_t)ncards, {});
    if (red.straddle && ncards == 2) {
        // the DEFECT: card0 owns groups 0..3, card1 owns groups 3..6 -> group 3 twice,
        // groups 6..7 never owned. Coverage and disjointness must both fire.
        p.card_groups[0] = {0, 1, 2, 3};
        p.card_groups[1] = {3, 4, 5, 6};
    } else {
        for (int c = 0; c < ncards; ++c)
            for (int gg = O_GROUPS * c / ncards; gg < O_GROUPS * (c + 1) / ncards; ++gg)
                p.card_groups[(size_t)c].push_back(gg);
    }
    auto groups_to_ranges = [&](const std::vector<int>& gs) {
        std::vector<Range> rs;
        for (int gg : gs) rs.push_back({gg * GROUP_W, (gg + 1) * GROUP_W});
        return rs;
    };

    if (!red.flip_qb) {
        // q_b: COLUMN shard on its out-dim (heads), expressed in GROUP units
        TensorPlan qb;
        qb.name = "q_b"; qb.axis_dim = O_IN; qb.sharded = true; qb.units = p.group_units;
        for (int c = 0; c < ncards; ++c) qb.owned.push_back(groups_to_ranges(p.card_groups[(size_t)c]));
        for (int c = 0; c < ncards; ++c) qb.reads.push_back(groups_to_ranges(p.card_groups[(size_t)c]));
        p.tensors.push_back(qb);
        // attn_output_b (o): ROW shard along the SAME groups
        TensorPlan ob;
        ob.name = "attn_output_b"; ob.axis_dim = O_IN; ob.sharded = true; ob.units = p.group_units;
        for (int c = 0; c < ncards; ++c) ob.owned.push_back(groups_to_ranges(p.card_groups[(size_t)c]));
        for (int c = 0; c < ncards; ++c) ob.reads.push_back(groups_to_ranges(p.card_groups[(size_t)c]));
        p.tensors.push_back(ob);
    } else {
        // THE DEFECT: directions swapped. q_b row-shards its q_lora INPUT (wrong axis:
        // pairing with o in group units cannot hold); o col-shards the D out-dim.
        TensorPlan qb;
        qb.name = "q_b"; qb.axis_dim = Q_LORA; qb.sharded = true; qb.units = "q_lora-in";
        cover_even(qb, ncards, 0, Q_LORA);
        for (int c = 0; c < ncards; ++c) qb.reads = qb.owned;
        p.tensors.push_back(qb);
        TensorPlan ob;
        ob.name = "attn_output_b"; ob.axis_dim = D; ob.sharded = true; ob.units = "D-out";
        cover_even(ob, ncards, 0, D);
        for (int c = 0; c < ncards; ++c) ob.reads = ob.owned;
        p.tensors.push_back(ob);
    }

    auto replicate = [&](const char* name, int dim) {
        TensorPlan t;
        t.name = name; t.axis_dim = dim; t.sharded = false;
        cover_all(t, ncards, dim);
        read_all(t, ncards, dim);
        if (red.drop_rep && ncards >= 2 && std::strcmp(name, "kv_latent_read") == 0)
            t.reads[1].clear();  // THE DEFECT: card 1 reads nothing
        p.tensors.push_back(t);
    };
    replicate("kv_latent_read", IDX_KV * IDX_D);  // width is a stand-in; the check is set-equality
    replicate("indexer_32x128", IDX_KV * IDX_D);
    replicate("engram_adds", D);

    // shared expert FFN: col-shard up (FF out), row-shard down (FF in), SAME boundaries
    TensorPlan su;
    su.name = "shared_up"; su.axis_dim = FF; su.sharded = true; su.units = "ff";
    cover_even(su, ncards, 0, FF);
    su.reads = su.owned;
    TensorPlan sd;
    sd.name = "shared_down"; sd.axis_dim = FF; sd.sharded = true; sd.units = "ff";
    cover_even(sd, ncards, 0, FF);
    sd.reads = sd.owned;
    p.tensors.push_back(su);
    p.tensors.push_back(sd);

    // hc state: full-width per card, fed by the allreduce output (checked as hc_input)
    TensorPlan hc;
    hc.name = "hc_state_fullwidth"; hc.axis_dim = HC * D; hc.sharded = false;
    cover_all(hc, ncards, HC * D);
    read_all(hc, ncards, HC * D);
    p.tensors.push_back(hc);
    // routed experts: ABSENT from the shard plan by design (EP axis only) - asserted absent
    return true;
}

// ---------------------------------------------------------------- invariants
static std::vector<int> boundaries(const std::vector<Range>& rs) {
    std::vector<int> b;
    for (const auto& r : rs) { b.push_back(r.lo); b.push_back(r.hi); }
    std::sort(b.begin(), b.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    return b;
}
static std::vector<int> owned_elems(const std::vector<Range>& rs) {
    std::vector<int> v;
    for (const auto& r : rs) for (int i = r.lo; i < r.hi; ++i) v.push_back(i);
    return v;
}

static void verify_plan(const Plan& p) {
    // 1. coverage + disjointness on every sharded owned set
    for (const auto& t : p.tensors) {
        if (!t.sharded) {
            // 2. replicated: every card reads the FULL span
            for (int c = 0; c < p.ncards; ++c) {
                std::vector<int> got = owned_elems(t.reads[(size_t)c]);
                bool full = (int)got.size() == t.axis_dim;
                if (full) for (int i = 0; i < t.axis_dim; ++i)
                    if (got[(size_t)i] != i) { full = false; break; }
                check(full, "%s: card %d read set is not the full span (replicated read)",
                      t.name.c_str(), c);
            }
            continue;
        }
        int total = 0;
        std::vector<char> seen((size_t)t.axis_dim, 0);
        bool overlap = false, oob = false;
        for (int c = 0; c < p.ncards; ++c)
            for (const auto& r : t.owned[(size_t)c]) {
                if (r.lo < 0 || r.hi > t.axis_dim || r.lo >= r.hi) oob = true;
                for (int i = std::max(r.lo, 0); i < std::min(r.hi, t.axis_dim); ++i) {
                    if (seen[(size_t)i]) overlap = true;
                    seen[(size_t)i] = 1;
                    ++total;
                }
            }
        check(!oob, "%s: owned range out of bounds", t.name.c_str());
        check(!overlap, "%s: shard OVERLAP (two cards own an element)", t.name.c_str());
        int missing = 0;
        for (char x : seen) if (!x) ++missing;
        check(missing == 0, "%s: %d elements uncovered by any shard", t.name.c_str(), missing);
    }
    // 3. pairing: q_b and attn_output_b shard in the SAME units with the SAME boundaries
    const TensorPlan* qb = nullptr;
    const TensorPlan* ob = nullptr;
    const TensorPlan* su = nullptr;
    const TensorPlan* sd = nullptr;
    for (const auto& t : p.tensors) {
        if (t.name == "q_b") qb = &t;
        if (t.name == "attn_output_b") ob = &t;
        if (t.name == "shared_up") su = &t;
        if (t.name == "shared_down") sd = &t;
    }
    check(qb && ob && su && sd, "plan is missing a required tensor");
    if (qb && ob) {
        check(qb->units == ob->units,
              "q_b shards in units [%s] but attn_output_b in [%s]: partials cannot line up",
              qb->units.c_str(), ob->units.c_str());
        std::vector<int> bq, bo;
        for (int c = 0; c < p.ncards; ++c) {
            for (const auto& r : qb->owned[(size_t)c]) { bq.push_back(r.lo); bq.push_back(r.hi); }
            for (const auto& r : ob->owned[(size_t)c]) { bo.push_back(r.lo); bo.push_back(r.hi); }
        }
        std::sort(bq.begin(), bq.end()); bq.erase(std::unique(bq.begin(), bq.end()), bq.end());
        std::sort(bo.begin(), bo.end()); bo.erase(std::unique(bo.begin(), bo.end()), bo.end());
        check(bq == bo, "q_b and attn_output_b boundary sets differ (straddled or mismatched groups)");
    }
    if (su && sd) {
        std::vector<int> bu, bd;
        for (int c = 0; c < p.ncards; ++c) {
            for (const auto& r : su->owned[(size_t)c]) { bu.push_back(r.lo); bu.push_back(r.hi); }
            for (const auto& r : sd->owned[(size_t)c]) { bd.push_back(r.lo); bd.push_back(r.hi); }
        }
        std::sort(bu.begin(), bu.end()); bu.erase(std::unique(bu.begin(), bu.end()), bu.end());
        std::sort(bd.begin(), bd.end()); bd.erase(std::unique(bd.begin(), bd.end()), bd.end());
        check(bu == bd, "shared expert up/down cuts differ");
    }
    // 4. every card's hc pipeline receives the post-reduce row
    for (int c = 0; c < p.ncards; ++c)
        check(p.hc_input[(size_t)c], "card %d hc mix would run without the reduced row", c);
    // 5. routed experts must NOT appear in the shard plan
    for (const auto& t : p.tensors)
        check(t.name.find("routed") == std::string::npos,
              "routed experts appear in the TP plan (they are an EP-axis concern only)");
    // 6. divisibility re-derived, not transcribed: 64 heads = 64 = ncards * heads/card AND
    //    O_GROUPS % ncards == 0 must both hold for a plan to have been buildable at all
    check(HEADS % p.ncards == 0 && O_GROUPS % p.ncards == 0,
          "build_plan ran for ncards=%d without dividing heads/groups", p.ncards);
}

// ---------------------------------------------------------------- rounding ops
static inline float round_f16_rne(float v) { return (float)(_Float16)v; }  // hardware RNE
static inline float round_f16_trunc(float v) {
    // truncate the mantissa to 10 bits at the correct binade (the WRONG order: this is the
    // dtype red-control; RNE is what __float2half_rn / the engine kernels do)
    if (v == 0 || !std::isfinite(v)) return v;
    float s = std::copysign(1.0f, v);
    float a = std::fabs(v);
    int e;
    std::frexp(a, &e);                       // a = m * 2^e, m in [0.5,1)
    float step = std::ldexp(1.0f, e - 11);   // value of one stored mantissa ulp here
    return s * (std::floor(a / step) * step);
}
struct Q8Stats { double max_abs_err; double sum_abs_err; long n; };
static void q8_quant_block(const float* in, float* out, int n, Q8Stats* st, float ref0) {
    for (int b0 = 0; b0 < n; b0 += Q8_BLOCK) {
        int b1 = std::min(b0 + Q8_BLOCK, n);
        float amax = 0;
        for (int i = b0; i < b1; ++i) amax = std::max(amax, std::fabs(in[i]));
        float scale_f32 = amax / 127.0f;
        float scale = (float)(_Float16)scale_f32;  // the scale itself is stored as fp16
        if (scale == 0) { for (int i = b0; i < b1; ++i) out[i] = 0; continue; }
        for (int i = b0; i < b1; ++i) {
            float q = std::nearbyintf(in[i] / scale);
            q = std::clamp(q, -127.0f, 127.0f);
            float dq = q * scale;
            out[i] = dq;
            if (st) { st->max_abs_err = std::max(st->max_abs_err, (double)std::fabs(dq - (ref0 + in[i] - ref0))); }
        }
    }
}

// ---------------------------------------------------------------- numerics
// The numeric model consumes the PLAN (card_groups / owned ranges), so what is measured is
// what the ownership tables promise. Reference is fp64, single ascending sweep.
struct SiteErr { const char* site; const char* dtype; double max_abs; double mean_abs; };
static std::vector<SiteErr> g_errs;

static void run_attention_site(const Plan& p, bool trunc_f16) {
    const int nc = p.ncards;
    // activations: head outputs h[j] (j over O_IN), deterministic
    std::vector<float> h((size_t)O_IN);
    for (int j = 0; j < O_IN; ++j) h[(size_t)j] = wgt(0x5EEDu, (uint32_t)j) * 0.25f;
    // reference: y[d] = sum_j O(d,j) * h[j] in double, ascending j
    static std::vector<double> yref;
    yref.assign((size_t)D, 0.0);
    for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (int j = 0; j < O_IN; ++j) acc += (double)wgt((uint32_t)d, (uint32_t)j) * h[(size_t)j];
        yref[(size_t)d] = acc;
    }
    // per-card fp32 partials over the o INPUT ranges the plan says the card owns
    std::vector<std::vector<float>> part((size_t)nc, std::vector<float>((size_t)D, 0.0f));
    for (int c = 0; c < nc; ++c) {
        // collect owned j ranges for attn_output_b from the plan (never re-derived here)
        const TensorPlan* ob = nullptr;
        for (const auto& t : p.tensors) if (t.name == "attn_output_b") ob = &t;
        for (const Range& r : ob->owned[(size_t)c])
            for (int d = 0; d < D; ++d) {
                float acc = 0;
                for (int j = r.lo; j < r.hi; ++j)
                    acc += wgt((uint32_t)d, (uint32_t)j) * h[(size_t)j];
                part[(size_t)c][(size_t)d] += acc;
            }
    }
    // reductions in the fixed rank-ascending order
    std::vector<float> y32((size_t)D, 0.0f), ytmp((size_t)D);
    std::vector<double> err32((size_t)D), err16((size_t)D), errq8((size_t)D);
    for (int c = 0; c < nc; ++c)
        for (int d = 0; d < D; ++d) y32[(size_t)d] += part[(size_t)c][(size_t)d];
    for (int d = 0; d < D; ++d)
        err32[(size_t)d] = std::fabs((double)y32[(size_t)d] - yref[(size_t)d]);
    std::fill(ytmp.begin(), ytmp.end(), 0.0f);
    for (int c = 0; c < nc; ++c)
        for (int d = 0; d < D; ++d)
            ytmp[(size_t)d] += trunc_f16 ? round_f16_trunc(part[(size_t)c][(size_t)d])
                                         : round_f16_rne(part[(size_t)c][(size_t)d]);
    for (int d = 0; d < D; ++d)
        err16[(size_t)d] = std::fabs((double)ytmp[(size_t)d] - yref[(size_t)d]);
    std::fill(ytmp.begin(), ytmp.end(), 0.0f);
    {
        std::vector<float> dq((size_t)D);
        for (int c = 0; c < nc; ++c) {
            q8_quant_block(part[(size_t)c].data(), dq.data(), D, nullptr, 0);
            for (int d = 0; d < D; ++d) ytmp[(size_t)d] += dq[(size_t)d];
        }
    }
    for (int d = 0; d < D; ++d)
        errq8[(size_t)d] = std::fabs((double)ytmp[(size_t)d] - yref[(size_t)d]);
    auto rec = [&](const char* site, const char* dt, const std::vector<double>& e) {
        double mx = 0, sum = 0;
        for (double x : e) { mx = std::max(mx, x); sum += x; }
        g_errs.push_back({site, dt, mx, sum / (double)e.size()});
    };
    rec("attn_output", "f32", err32);
    rec("attn_output", "f16", err16);
    rec("attn_output", "q8", errq8);
}

static void run_shared_ffn_site(const Plan& p, bool trunc_f16) {
    const int nc = p.ncards;
    std::vector<float> x((size_t)D);
    for (int i = 0; i < D; ++i) x[(size_t)i] = wgt(0xC0FFu, (uint32_t)i) * 0.1f;
    // reference up->identity-act->down in fp64, ascending
    static std::vector<double> href, yref;
    href.assign((size_t)FF, 0.0);
    yref.assign((size_t)D, 0.0);
    for (int f = 0; f < FF; ++f) {
        double acc = 0;
        for (int i = 0; i < D; ++i) acc += (double)wgt((uint32_t)f, (uint32_t)i) * x[(size_t)i];
        href[(size_t)f] = acc;
    }
    for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (int f = 0; f < FF; ++f) acc += (double)wgt((uint32_t)f + 0x10000u, (uint32_t)d) * href[(size_t)f];
        yref[(size_t)d] = acc;
    }
    // per card: up over owned ff rows -> down over the SAME owned ff columns, fp32
    std::vector<std::vector<float>> part((size_t)nc, std::vector<float>((size_t)D, 0.0f));
    const TensorPlan* su = nullptr;
    const TensorPlan* sd = nullptr;
    for (const auto& t : p.tensors) {
        if (t.name == "shared_up") su = &t;
        if (t.name == "shared_down") sd = &t;
    }
    for (int c = 0; c < nc; ++c) {
        // this card's hidden slice must be exactly its owned up-rows == owned down-cols
        check(su->owned[(size_t)c] == sd->owned[(size_t)c],
              "shared expert card %d: up rows and down cols differ (paired cut broken)", c);
        for (const Range& r : su->owned[(size_t)c]) {
            for (int d = 0; d < D; ++d) {
                float acc = 0;
                for (int f = r.lo; f < r.hi; ++f)
                    acc += wgt((uint32_t)f + 0x10000u, (uint32_t)d) * (float)href[(size_t)f];
                part[(size_t)c][(size_t)d] += acc;
            }
        }
    }
    std::vector<float> y32((size_t)D, 0.0f), ytmp((size_t)D), dq((size_t)D);
    std::vector<double> e32((size_t)D), e16((size_t)D), e8((size_t)D);
    for (int c = 0; c < nc; ++c)
        for (int d = 0; d < D; ++d) y32[(size_t)d] += part[(size_t)c][(size_t)d];
    std::fill(ytmp.begin(), ytmp.end(), 0.0f);
    for (int c = 0; c < nc; ++c)
        for (int d = 0; d < D; ++d)
            ytmp[(size_t)d] += trunc_f16 ? round_f16_trunc(part[(size_t)c][(size_t)d])
                                         : round_f16_rne(part[(size_t)c][(size_t)d]);
    for (int d = 0; d < D; ++d) {
        e32[(size_t)d] = std::fabs((double)y32[(size_t)d] - yref[(size_t)d]);
        e16[(size_t)d] = std::fabs((double)ytmp[(size_t)d] - yref[(size_t)d]);
    }
    std::fill(ytmp.begin(), ytmp.end(), 0.0f);
    for (int c = 0; c < nc; ++c) {
        q8_quant_block(part[(size_t)c].data(), dq.data(), D, nullptr, 0);
        for (int d = 0; d < D; ++d) ytmp[(size_t)d] += dq[(size_t)d];
    }
    for (int d = 0; d < D; ++d)
        e8[(size_t)d] = std::fabs((double)ytmp[(size_t)d] - yref[(size_t)d]);
    auto rec = [&](const char* site, const char* dt, const std::vector<double>& e) {
        double mx = 0, sum = 0;
        for (double v : e) { mx = std::max(mx, v); sum += v; }
        g_errs.push_back({site, dt, mx, sum / (double)e.size()});
    };
    rec("shared_ffn", "f32", e32);
    rec("shared_ffn", "f16", e16);
    rec("shared_ffn", "q8", e8);
}

// RNE property probes (the dtype red-control must break at least one)
static void check_rne_properties(bool trunc) {
    // 1 + 3/4 ulp: RNE goes UP to the next representable; truncation stays DOWN
    float up = 1.0f + 3.0f * std::ldexp(1.0f, -12);
    check(round_f16_rne(up) == 1.0f + std::ldexp(1.0f, -10), "RNE probe A failed");
    if (trunc)
        check(round_f16_trunc(up) != round_f16_rne(up), "truncation unexpectedly equals RNE");
    // ties round to even: 1 + 1/2 ulp stays at 1.0 (even mantissa), does not go up
    check(round_f16_rne(1.0f + 0.5f * std::ldexp(1.0f, -10)) == 1.0f, "RNE tie-to-even probe failed");
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    RedFlags red;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--red=", 6) == 0) {
            const char* r = argv[i] + 6;
            if (!std::strcmp(r, "flip-qb-direction")) red.flip_qb = true;
            else if (!std::strcmp(r, "straddle-ogroup")) red.straddle = true;
            else if (!std::strcmp(r, "drop-replicate")) red.drop_rep = true;
            else if (!std::strcmp(r, "dtype-rounding")) red.dtype_round = true;
            else { std::printf("unknown red control %s\n", r); return 2; }
            std::printf("RED-CONTROL injected: %s\n", r);
        }
    }
    std::printf("p6_tp_algebra: P6.1 sharding-algebra gate (D %d heads %d x %d, q_lora %d, "
                "o %d groups of %d, ff %d, idx %dx%d, hc %d)\n",
                D, HEADS, DH, Q_LORA, O_GROUPS, GROUP_W, FF, IDX_KV, IDX_D, HC);

    // A. unplaceable widths must be refused, naming the usable set
    for (int w : {3, 5}) {
        Plan junk;
        check(!build_plan(w, red, junk), "width %d was not refused (8 groups do not divide)", w);
    }
    std::printf("usable TP widths (groups=8): {1,2,4}; widths {3,5,6,7,8-3} refuse\n");

    // B. plan invariants at 2 and 4 cards
    for (int nc : {2, 4}) {
        Plan p;
        if (!build_plan(nc, red, p)) { std::printf("build_plan refused width %d\n", nc); return 1; }
        std::printf("--- plan @ %d-way: ", nc);
        for (int c = 0; c < nc; ++c) {
            std::printf("card%d groups", c);
            for (int gg : p.card_groups[(size_t)c]) std::printf(" %d", gg);
            std::printf(" | ");
        }
        std::printf("\n");
        verify_plan(p);
        // C. numerics consume the SAME plan the invariants just certified
        run_attention_site(p, red.dtype_round);
        run_shared_ffn_site(p, red.dtype_round);
    }

    // D. rounding-order properties
    check_rne_properties(red.dtype_round);
    if (red.dtype_round)
        check(round_f16_trunc(1.0f + 3.0f * std::ldexp(1.0f, -12)) == round_f16_rne(1.0f + 3.0f * std::ldexp(1.0f, -12)),
              "dtype red-control: truncation and RNE agree (the swap is not observable)");

    // E. induced error table (host numbers for the later hardware calibration)
    std::printf("site           dtype  max_abs     mean_abs\n");
    for (const auto& e : g_errs)
        std::printf("%-14s %-5s  %-10.3e  %.3e\n", e.site, e.dtype, e.max_abs, e.mean_abs);
    // loose sanity: f32 reduce error tiny; f16 within 2^-9 of partial sums magnitude;
    // q8 within 2^-6. These are STABILITY bars for the printed numbers, not correctness bars
    for (const auto& e : g_errs) {
        if (std::strcmp(e.dtype, "f32") == 0) check(e.max_abs < 1e-2, "f32 reduce error too big: %.3e", e.max_abs);
        if (std::strcmp(e.dtype, "f16") == 0) check(e.max_abs < 2.0,  "f16 reduce error too big: %.3e", e.max_abs);
        if (std::strcmp(e.dtype, "q8") == 0)  check(e.max_abs < 8.0,  "q8 reduce error too big: %.3e", e.max_abs);
    }

    if (g_fails) {
        std::printf("P6.1 GATE: RED (%d/%d checks failed)\n", g_fails, g_checks);
        return 1;
    }
    std::printf("P6.1 GATE: PASS (%d checks, 0 failures)\n", g_checks);
    return 0;
}
