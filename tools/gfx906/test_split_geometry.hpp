// tools/gfx906/test_split_geometry.hpp - P5.3b gate for the STAGE-BOUNDARY RULES a split must obey, expressed against
// a model's GEOMETRY so they can be checked with plain g++: no CUDA, no HIP, no ggml, no GPU, no model file.
//
// The engine enforces these inside Ds41Dense (which needs ggml to exist at all), so the same predicates are restated
// here - the RULE, not a copy of the code path - and the engine's own wording is checked by hand against them.  The
// things a DeepSeek-V4.1 split has to satisfy are:
//
//   1 THE READER'S RULE.  A layer reads the compressed KV of the latest kv source AT OR BELOW it and the index keys of
//     the latest index source at or below it (Ds41Geometry::kv_src / idx_src).  Those two rings are PERSISTENT GPU
//     state: allocated once, written as the run advances, read by pointer inside a graph.  The relay carries x_state,
//     pre_state and the candidate column - all sized PER TOKEN - and has no way to move a cache, so A RING LIVES ON THE
//     CARD OF THE STAGE THAT RUNS ITS WRITER, and every READER of a ring has to sit in that same stage:
//
//         K is valid  iff  for every m in [K, n_layer) with ratio[m] > 0:
//             kv_src(m) >= K  AND  idx_src(m) >= K  AND  (m uses cand_layer's blocks => cand_layer >= K)
//
//     The stage-0 side is free (kv_src(m) <= m < K always).  DeepSeek-V4.1-Flash: the valid set is {1, 2, 8, 14, 20} -
//     the layers that are both a kv and an index source, plus the vacuous 1, whose layer runs no attention.  THIS IS A
//     SET, NOT A FLOOR, and it contains the even split at 20.  An earlier revision of this file derived a floor of 37
//     ("last shared-cache layer + 1") by requiring the WRITERS to sit below the boundary; that boundary is the one that
//     aborts, because the tail [37,40) reads layer 20's compressed KV and layer 36's index keys.  The tests re-derive
//     the set by simulating the stages (split_stages_own_their_reads) rather than trusting the predicate.
//
//   2 THE CANDIDATE RULE.  Layer cand_layer picks the blocks the LATER indexers rank inside.  A boundary can only
//     separate them by sitting strictly ABOVE the producer and at or below a reader - and rule 1 forbids exactly that
//     (validity at K requires cand_layer >= K for any K that leaves a reader at or above it).  So on a model whose
//     caches ARE these two source lists the column never crosses a boundary, the payload is the two state columns, and
//     the candidate region exists for a model with a cache OUTSIDE them (a hybrid SSM state, a cache named by a key the
//     geometry does not read yet).
//
//   3 THE PAYLOAD RULE.  The bytes per token a boundary moves are fixed by the geometry - the two state columns plus
//     the candidate column only when rule 2 says it can cross - and both sides of one boundary must agree on that
//     number, so it is a function of the GEOMETRY and never of a stage's position.
//
// NOT RESTATED HERE, because the geometry does not carry it: ds41_generate additionally refuses a split whose FIRST
// boundary cuts above an Engram layer when --engram is given.  The rolling token ring the Engram hash reads is fed by
// upload_inputs, which only stage 0 runs (begin() rejects a later stage), so a later stage that runs an Engram layer
// would hash its window against a ring it never fed.  For the shipped model that leaves the first boundary above 14.
//
// DeepSeek-V4.1-Flash (the shipped 40-layer build): kv sources {2,8,14,20}, index sources {2,8,14,20,24,28,32,36},
// ratio[0]=ratio[1]=0 and ratio[l]=2 for 2<=l<=20 and 1 above, cand_layer 20, cand_blocks 2048, hc 4, n_embd 5120.
// The lists below are copied from the model's GGUF keys - and NOT taken on faith: step 3 of gate-device-split.sh asks
// the engine's own reader for the per-layer source map and FAILs if this copy no longer matches.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::gfx906 {

// A stand-in for Ds41Geometry with exactly the fields the rules read.
struct GeoShape {
    std::vector<int64_t> kv, index;   // source layer ids, ascending
    std::vector<int64_t> ratio;       // [n_layer] 0 = sliding-window layer, reads no ring
    int64_t n_layer = 0;
    int64_t cand_layer = -1, cand_blocks = 0;
    int64_t n_embd = 0, hc = 0;

    bool is_kv(int64_t l) const {
        for (int64_t k : kv) if (k == l) return true;
        return false;
    }
    bool is_index(int64_t l) const {
        for (int64_t k : index) if (k == l) return true;
        return false;
    }
    // The source whose rings layer l reads (-1 when l reads none, i.e. a sliding-window layer).
    int64_t kv_src(int64_t l) const {
        if (!ratio.empty() && ratio[(size_t) l] == 0) return -1;
        int64_t s = -1;
        for (int64_t k : kv) if (k <= l) s = k;
        return s;
    }
    int64_t idx_src(int64_t l) const {
        if (!ratio.empty() && ratio[(size_t) l] == 0) return -1;
        int64_t s = -1;
        for (int64_t k : index) if (k <= l) s = k;
        return s;
    }
    // Does layer l rank inside cand_layer's blocks?  Exactly the engine's L.uses_cand (ds41_dense.cpp's layer table):
    // above the producer AND running an indexer.
    bool uses_cand(int64_t l) const { return cand_layer >= 0 && l > cand_layer && is_index(l); }
    // Which stage runs layer l, given `ends` (the end of every stage but the last)?  0 = the first stage.
    int stage_of(int64_t l, const std::vector<int64_t> & ends) const {
        int s = 0;
        for (int64_t e : ends) if (l >= e) ++s;
        return s;
    }
    // Does any layer after cand_layer rank inside its blocks?  (The engine's Impl::cand_has_readers.)  "Rank inside"
    // means the layer runs an INDEXER - a layer's indexer scores over cand_layer's blocks, so it is the indexer layers
    // that read the column, not the source layers named next to them in the list.
    bool cand_has_readers() const {
        if (cand_layer < 0 || cand_layer >= n_layer) return false;
        for (int64_t k : index) if (k > cand_layer) return true;
        return false;
    }
    // The first / last layer that reads cand_layer's column (the index sources strictly above it).  n_layer / -1 when
    // nothing does.  cand_layer's own indexer runs inside the producer's pass and reads the column as written, so the
    // producer itself is not a "reader" in the sense the boundary rule cares about.
    int64_t first_cand_reader() const {
        for (int64_t k : index) if (k > cand_layer) return k;
        return n_layer;
    }
    int64_t last_cand_reader() const {
        if (!cand_has_readers()) return -1;
        int64_t k = cand_layer;
        for (int64_t x : index) if (x > k) k = x;
        return k;
    }
    // A copy of this shape with `l` added to the index sources, kept ASCENDING (an out-of-order list would silently
    // rewire which layer's keys a layer reads, which is the mistake these hypothetical shapes are easy to make).
    GeoShape with_index_source(int64_t l) const {
        GeoShape h = *this;
        h.index.push_back(l);
        std::sort(h.index.begin(), h.index.end());
        return h;
    }
};

// Rule 1, one boundary at a time: the predicate the engine's Ds41Dense::split_boundary_valid states.  A layer at or
// above the boundary must find its rings - and the candidate column - at or above it, in the stage that runs it.
inline bool split_boundary_valid(const GeoShape & g, int64_t K) {
    if (K <= 0 || K >= g.n_layer) return false;      // a boundary is inside the model, and never at layer 0
    for (int64_t m = K; m < g.n_layer; ++m) {
        if (!g.ratio.empty() && g.ratio[(size_t) m] == 0) continue;   // a sliding-window layer reads no ring
        if (g.kv_src(m) < K) return false;
        if (g.idx_src(m) < K) return false;
        if (g.uses_cand(m) && g.cand_layer < K) return false;
    }
    return true;
}

// Rule 1, the whole legal set, ascending.  {1, 2, 8, 14, 20} on the shipped 40-layer build; split_plan's floor of 2
// leaves {2, 8, 14, 20} as what a caller may name, and 20 of those is the EVEN split of the model in two stages.
inline std::vector<int64_t> split_valid_boundaries(const GeoShape & g) {
    std::vector<int64_t> v;
    for (int64_t K = 1; K < g.n_layer; ++K) if (split_boundary_valid(g, K)) v.push_back(K);
    return v;
}

// The same set as one line of text, the way a message prints it.
inline std::string split_valid_set_text(const GeoShape & g) {
    const std::vector<int64_t> v = split_valid_boundaries(g);
    if (v.empty()) return "(none: this model has no boundary a stage split may use)";
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s;
}

// Rule 1 as the engine checks a whole plan: `ends` = the end of every stage but the last, ascending, inside
// (0, n_layer), every one of them a member of the valid set.  Returns false with the reason - naming the legal set -
// when a boundary would strand a ring (or is out of order).
inline bool split_boundaries_ok(const GeoShape & g, const std::vector<int64_t> & ends, std::string & err) {
    int64_t prev = 0;
    for (size_t i = 0; i < ends.size(); ++i) {
        const int64_t e = ends[i];
        if (e <= prev || e >= g.n_layer) {
            err = "boundary " + std::to_string(e) + " is not inside (0," + std::to_string(g.n_layer) +
                  ") and above the previous boundary " + std::to_string(prev);
            return false;
        }
        if (!split_boundary_valid(g, e)) {
            // The SAME wording the engine uses (Ds41Dense::split_boundaries_ok), naming the first layer at or above the
            // boundary whose rings reach below it - so a reviewer can compare the two messages literally.
            std::string who;
            for (int64_t m = e; m < g.n_layer; ++m) {
                if (!g.ratio.empty() && g.ratio[(size_t) m] == 0) continue;
                if (g.kv_src(m) < e) {
                    who = "layer " + std::to_string(m) + " reads the compressed KV of layer " + std::to_string(g.kv_src(m));
                    break;
                }
                if (g.idx_src(m) < e) {
                    who = "layer " + std::to_string(m) + " reads the index keys of layer " + std::to_string(g.idx_src(m));
                    break;
                }
                if (g.uses_cand(m) && g.cand_layer < e) {
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

// THE INDEPENDENT RE-DERIVATION.  Where split_boundary_valid states the rule as a predicate on the geometry, this
// function PLAYS THE RUNTIME: it builds the stages `ends` describes and, for every layer of every stage, asks the
// question the engine's Impl::owns_layer asks - does the stage that runs this layer also run the layer whose ring it
// reads? - with the same two aborts build_attn has (the compressed-KV source, the index-key source) plus the candidate
// column's equivalent.  The tests assert split_boundaries_ok == this for EVERY single boundary and EVERY 2- and
// 3-boundary plan, so the DIRECTION of the rule cannot be flipped in one place without the gate screaming.  (A
// previous revision had the two disagree in exactly that direction, and the witness died at graph build.)
inline bool split_stages_own_their_reads(const GeoShape & g, const std::vector<int64_t> & ends) {
    std::vector<int64_t> b;
    b.push_back(0);
    for (int64_t e : ends) b.push_back(e);
    b.push_back(g.n_layer);
    for (size_t s = 0; s + 1 < b.size(); ++s) {
        const int64_t lb = b[s], le = b[s + 1];
        if (le <= lb) return false;                                   // a stage of zero layers is not a stage
        auto owns = [&](int64_t il) { return il >= lb && il < le; };   // == Ds41Dense::Impl::owns_layer
        for (int64_t m = lb; m < le; ++m) {
            if (g.ratio.empty() || g.ratio[(size_t) m] == 0) continue;   // build_attn's `if (L.ratio && ...)`
            if (!owns(g.kv_src(m))) return false;
            if (!owns(g.idx_src(m))) return false;
            if (g.uses_cand(m) && !owns(g.cand_layer)) return false;
        }
    }
    return true;
}

// Rule 1's default-plan behaviour: snap a planned boundary list DOWN to the legal set, so an EVEN split (--layer-split
// unset) gets a working plan instead of a refusal - 40 layers over 3 stages plans 13 and 26 and runs as 8 and 20.  A
// boundary with nowhere to go (no legal layer at or below it, above the previous boundary) is a refusal, not a stage of
// zero layers; `err` names the legal set.  This is the engine's Ds41Dense::split_snap_to_valid.
inline bool split_snap_to_valid(const GeoShape & g, const std::vector<int64_t> & planned,
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
                  std::to_string(planned.size() + 1) + " stages cannot be snapped";
            return false;
        }
        snapped.push_back(pick);
        prev = pick;
    }
    return true;
}

// Rule 2's question, in one line: can a LEGAL boundary separate cand_layer from a layer that ranks inside its blocks?
// It asks the enumerator, because that is what "legal" now means: some valid K strictly above the producer and at or
// below a reader.  Rule 1 forbids that for a model whose caches are the kv / index source lists - validity at K already
// requires cand_layer >= K whenever a reader is left at or above K - so the answer is false for every such model, this
// one included.  The consequence is worth stating plainly: a staged run relays the two state columns and nothing else,
// and the candidate region is code a model with a cache OUTSIDE these lists would pay 8 KiB a token for.  It is
// implemented, sized and gated; it does not fire here.
inline bool split_candidates_cross(const GeoShape & g) {
    if (!g.cand_has_readers()) return false;
    const int64_t last_reader = g.last_cand_reader();
    for (int64_t K : split_valid_boundaries(g))
        if (g.cand_layer < K && K <= last_reader) return true;
    return false;
}

// Rule 2, the shape where the in-place read is sound: true = nothing has to be relayed at all, so the caller need not
// carry the candidate region; false = it crosses and rides in the payload.  `owner_out` names the producer's stage,
// `adjacent_out` names the stage that may read the producer's OWN tensor instead of a copy.  There is exactly one such
// consumer - the producer's immediate successor, and only when it shares the producer's card, because ggml hands the
// host an iGPU device pointer the other card's copy engine cannot read.  A later consumer takes the copy out of the
// payload it was just handed, which is why a producer is never stranded however many stages sit between it and the
// layer whose indexer ranks inside its blocks.
inline bool split_keeps_candidates_local(const GeoShape & g, const std::vector<int64_t> & ends, int & owner_out,
                                         int & adjacent_out, std::string & err) {
    owner_out = -1;
    adjacent_out = -1;
    if (!split_candidates_cross(g)) return true;   // nothing a legal boundary can separate
    const int owner = g.stage_of(g.cand_layer, ends);
    bool separated = false;
    for (int64_t k : g.index)
        if (k > g.cand_layer && g.stage_of(k, ends) != owner) separated = true;
    if (!separated) return true;
    owner_out = owner;
    if (owner + 1 < (int) ends.size() + 1) adjacent_out = owner + 1;   // the one consumer that may read in place
    (void) err;   // never a refusal: the payload carries the column, so any legal split runs
    return false;
}

// Rule 3: the payload one token of a boundary carries: x_state's column, pre_state's column, plus the candidate column
// ONLY when a legal boundary can separate it from its producer (split_candidates_cross).  On DeepSeek-V4.1-Flash the two
// numbers below are EQUAL and that equality is the witness's gate: a slab line with a non-zero cand region means the
// model's geometry moved underneath this witness.  `relay_slab` is the number the engine and the staging both size from;
// a stage that could choose its own idea of where a copy ends is the worst possible failure mode, so there is exactly
// one function that answers that question.
inline uint64_t relay_slab_no_cand(const GeoShape & g) {
    return (uint64_t) (g.n_embd * g.hc * 4) + (uint64_t) (g.hc * 4);
}
inline uint64_t relay_slab(const GeoShape & g) {
    return relay_slab_no_cand(g) + (split_candidates_cross(g) ? (uint64_t) (g.cand_blocks * 4) : 0ull);
}

// Rule 4 (CLI gate): the boundary list a staged run of `n_stage` stages over `n_layer` layers WOULD use (the even
// chunks split_plan computes for an unset --layer-split), and what the engine turns it into.  This is the arithmetic
// ds41_generate does before it allocates anything - plan, then SNAP DOWN to the legal set, then refuse if a stage would
// collapse - reproduced here so the witness's `--layer-split $SPLIT` and the engine's answer are checked against the
// same code path a reviewer can read, on a host with no GPU.  `ends` excludes the last stage's end (that is not a
// boundary).
inline std::vector<int64_t> plan_even_chunks(int64_t n_layer, int n_stage) {
    std::vector<int64_t> ends;
    for (int i = 1; i < n_stage; ++i) ends.push_back((int64_t) (i * n_layer / n_stage));
    return ends;
}
inline bool plan_boundaries_ok(int64_t n_layer, int n_stage, const GeoShape & g, std::vector<int64_t> & ends_out,
                               std::string & err) {
    if (n_stage < 1 || n_layer <= 0) { err = "no layers or no stages"; return false; }
    const std::vector<int64_t> planned = plan_even_chunks(n_layer, n_stage);
    if (!split_snap_to_valid(g, planned, ends_out, err)) return false;   // ends_out keeps the snapped plan on success
    return split_boundaries_ok(g, ends_out, err);
}

}  // namespace strata::gfx906
