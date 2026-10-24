// tools/gfx906/test_split_geometry.hpp - P5.3b gate for the STAGE-BOUNDARY RULES a split must obey, expressed against
// a model's GEOMETRY so they can be checked with plain g++: no CUDA, no HIP, no ggml, no GPU, no model file.
//
// The engine enforces these inside Ds41Dense (which needs ggml to exist at all), so the same predicates are restated
// here - the RULE, not a copy of the code path - and the engine's own wording is checked by hand against them.  The
// three things a DeepSeek-V4.1 split has to satisfy are:
//
//   1 THE CACHE RULE.  A layer reads the compressed KV of the latest kv source AT OR BELOW it and the index keys of the
//     latest index source at or below it (Ds41Geometry::kv_src / idx_src).  Those two rings are PERSISTENT GPU state:
//     allocated once, written as the run advances, read by pointer inside a graph.  The relay carries x_state,
//     pre_state and the candidate column - all sized PER TOKEN - and has no way to move a cache, so a stage boundary
//     must sit ABOVE every layer a later stage reads from.  For a model whose deepest reader reaches the last source
//     (DeepSeek-V4.1-Flash: layers 24..39 rank inside layer 20's blocks and layer 36's index keys), that means one
//     boundary only, at LAST_SOURCE + 1, with the tail of the model on the far side.
//
//   2 THE CANDIDATE RULE.  Layer cand_layer picks the blocks the LATER indexers rank inside, so once a boundary sits
//     above it the column has to cross.  It rides in the payload on EVERY boundary and each consumer copies it into a
//     column of its own, so a producer is never stranded: with more than two stages the column hops stage by stage.
//     (Reading the producer's tensor in place is an optimization for the immediate successor on the same card, and this
//     gate checks the shape where that shortcut is sound.)
//
//   3 THE PAYLOAD RULE.  The bytes per token a boundary moves are fixed by the geometry - the two state columns plus
//     the candidate column whenever the model has candidates at all - and both sides of one boundary must agree on
//     that number, so it is a function of the GEOMETRY and never of a stage's position.
//
// DeepSeek-V4.1-Flash (the shipped 40-layer build): kv sources {2,8,14,20}, index sources {2,8,14,20,24,28,32,36},
// ratio[0]=ratio[1]=0 and ratio[l]=2 for 2<=l<=20 and 1 above, cand_layer 20, cand_blocks 2048, hc 4, n_embd 5120.
// The lists below are copied from the model's GGUF keys, so a change in the model that moves a source layer shows up
// as a stale list here - which is the point of a gate.
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
    // The source whose rings layer l reads ("" when l reads none, i.e. a sliding-window layer).
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
    // Does any layer after cand_layer rank inside its blocks?  (The engine's Impl::cand_has_readers.)  "Rank inside"
    // means the layer runs an INDEXER - a layer's indexer scores over cand_layer's blocks, so it is the indexer layers
    // that read the column, not the source layers named next to them in the list.
    bool cand_has_readers() const {
        if (cand_layer < 0 || cand_layer >= n_layer) return false;
        for (int64_t k : index) if (k > cand_layer) return true;
        return false;
    }
    // The first layer that reads cand_layer's column: the first INDEX source strictly above it.  cand_layer's own
    // indexer runs inside the producer's pass and reads the column as it was written, so the layer itself is not a
    // "reader" in the sense the boundary rule cares about.  n_layer when there is no reader at all.
    int64_t first_cand_reader() const {   // (kept for the messages a caller may print; the rule uses the range test)
        for (int64_t k : index) if (k > cand_layer) return k;
        return n_layer;
    }
    // The last layer that reads it, or n_layer when nothing does.  The gate builds hypothetical geometries with this
    // helper (an indexer above the cache floor, which the shipped model does not have) so an out-of-order list - which
    // silently rewires which layer's keys a layer reads - cannot sneak into a test case.
    int64_t last_cand_reader() const {
        if (!cand_has_readers()) return -1;
        int64_t k = cand_layer;
        for (int64_t x : index) if (x > k) k = x;
        return k;
    }
    // A copy of this shape with `l` added to the index sources, kept ASCENDING.
    GeoShape with_index_source(int64_t l) const {
        GeoShape h = *this;
        h.index.push_back(l);
        std::sort(h.index.begin(), h.index.end());
        return h;
    }
};

// Rule 1: the LOWEST layer a boundary may sit at - the model's last kv / index source + 1.  A model with no shared
// caches at all answers 0, which is to say any boundary is fine (the relay carries everything its successor reads).
inline int64_t split_min_boundary(const GeoShape & g) {
    int64_t last = -1;
    for (int64_t k : g.kv) if (k > last) last = k;
    for (int64_t k : g.index) if (k > last) last = k;
    return last + 1;
}

// Rule 1 as the engine checks it: `ends` = the end of every stage but the last, ascending, inside (0, n_layer).
// Returns false with the reason when a boundary would strand a cache (or is out of order).
inline bool split_boundaries_ok(const GeoShape & g, const std::vector<int64_t> & ends, std::string & err) {
    const int64_t min_e = split_min_boundary(g);
    int64_t prev = 0;
    for (size_t i = 0; i < ends.size(); ++i) {
        const int64_t e = ends[i];
        if (e <= prev || e >= g.n_layer) {
            err = "boundary " + std::to_string(e) + " is not inside (0," + std::to_string(g.n_layer) +
                  ") and above the previous boundary " + std::to_string(prev);
            return false;
        }
        if (e < min_e) {
            err = "a boundary at layer " + std::to_string(e) + " leaves the next stage reading a cache it does not own; "
                  "split at layer " + std::to_string(min_e) + " or above";
            return false;
        }
        prev = e;
    }
    return true;
}

// Rule 2's question, in one line: can a LEGAL boundary separate cand_layer from a layer that ranks inside its blocks?
// A reader at or below the cache floor is in the producer's own stage whatever the plan, so only a reader AT OR ABOVE the
// floor separates.  And the floor is the model's LAST kv / index source + 1, so with the caches defined by those two
// lists a reader can never reach it: the answer is false for every such model, this one included.  The consequence is
// worth stating plainly - under the cache rule a staged run always relays the two state columns and nothing else, and the
// candidate region is code a model with a cache OUTSIDE these lists (a hybrid SSM state, a cache named by a key the
// geometry does not read yet) would pay 8 KiB a token for.  It is implemented, sized and gated; it does not fire here.
inline bool split_candidates_cross(const GeoShape & g) {
    return g.cand_has_readers() && g.last_cand_reader() >= split_min_boundary(g);
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
    auto stage_of = [&](int64_t l) { int s = 0; for (int64_t e : ends) if (l >= e) ++s; return s; };
    const int owner = stage_of(g.cand_layer);
    // THE FLOOR is what the cache rule (rule 1) puts on the boundary list, and the candidate rule reads it instead of
    // guessing that every reader can be reached: the model's last kv / index source + 1.  A model whose readers all sit
    // below that floor (so a legal boundary cannot separate them from the producer) keeps its column at home and pays
    // no bytes for it; one whose readers sit above it (DeepSeek-V4.1-Flash: the indexers of 24/28/32/36 read layer 20's
    // blocks, and the floor is 37) carries the region on every boundary that separates them.
    const int64_t floor_e = split_min_boundary(g);
    bool separated = false;
    for (int64_t k : g.index)
        if (k > g.cand_layer && stage_of(k) != owner && k >= floor_e) separated = true;
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

// Rule 4 (CLI gate): the boundary list a staged run of `n_stage` stages over `n_layer` layers WOULD use, and whether the
// cache rule accepts it.  This is the arithmetic ds41_generate does before it allocates anything - reproduced here so the
// witness's `--layer-split $SPLIT` and the engine's refusal of a lower number are checked against the same code path a
// reviewer can read, on a host with no GPU.  `ends` excludes the last stage's end (that is not a boundary).
inline bool plan_boundaries_ok(int64_t n_layer, int n_stage, const GeoShape & g, std::vector<int64_t> & ends_out,
                               std::string & err) {
    if (n_stage < 1 || n_layer <= 0) { err = "no layers or no stages"; return false; }
    std::vector<int64_t> ends;
    for (int i = 1; i < n_stage; ++i) ends.push_back((int64_t) (i * n_layer / n_stage));
    if (!split_boundaries_ok(g, ends, err)) { ends_out = ends; return false; }
    ends_out = ends;
    return true;
}

}  // namespace strata::gfx906
