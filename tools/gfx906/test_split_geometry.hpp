// tools/gfx906/test_split_geometry.hpp - P5.3b gate for the STAGE-BOUNDARY RULES a split must obey, expressed against
// a model's GEOMETRY so they can be checked with plain g++: no CUDA, no HIP, no ggml, no GPU, no model file.
//
// The engine enforces these inside Ds41Dense (which needs ggml to exist at all), so the same predicates are restated
// here - the RULE, not a copy of the code path - and the engine's own wording is checked by hand against them.  The
// three rules a DeepSeek-V4.1 split has to satisfy are:
//
//   1 THE KV RULE.  A stage's LAST layer must produce the compressed KV and the index keys its successor reads,
//     because that is the only persistent state a boundary cannot leave in place: the successor runs elsewhere and
//     reads the producer's rings by pointer inside one graph.  The geometry resolves a source to "the latest source
//     <= l", so requiring the layer BEFORE a boundary to be BOTH a kv source and an index source makes every
//     successor read only this stage's rings.
//
//   2 THE CANDIDATE RULE.  Layer cand_layer picks the blocks the LATER indexers rank inside.  Once a boundary
//     separates the producer from a reader, the column has to cross: on one card the reader points at the producer's
//     tensor, across cards it rides in the relay payload.  A reader more than one stage after the producer can be
//     given nobody (stage s borrows from stage s-1), so such splits are refused outright.
//
//   3 THE PAYLOAD RULE.  The bytes per token a boundary moves are fixed by the geometry (the state columns, plus the
//     candidate column when the producer owns it), and both sides of one boundary must agree on that number.
//
// DeepSeek-V4.1-Flash (the shipped 40-layer build): kv sources {0,2,8,14,20}, index sources {0,2,8,14,20,24,28,32,36},
// cand_layer 20, cand_blocks 2048, hc 4, n_embd 5120.  The lists below are copied from the model's GGUF keys, so a
// change in the model that moves a source layer shows up as a stale list here - which is the point of a gate.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::gfx906 {

// A stand-in for Ds41Geometry with exactly the fields the rules read.
struct GeoShape {
    std::vector<int64_t> kv, index;   // source layer ids, ascending
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
    // Does any layer after cand_layer rank inside its blocks?  (The engine's Impl::cand_is_relayed.)
    bool cand_relayed() const {
        if (cand_layer < 0 || cand_layer >= n_layer) return false;
        for (int64_t k : index) if (k > cand_layer) return true;
        return false;
    }
};

// The end of every stage, given a boundary list: `ends` = end of each stage but the last, ascending, inside
// (0, n_layer).  Returns false with the reason when a boundary would leave its successor reading state the producer
// does not own (rule 1).
inline bool split_boundaries_ok(const GeoShape & g, const std::vector<int64_t> & ends, std::string & err) {
    int64_t prev = 0;
    for (size_t i = 0; i < ends.size(); ++i) {
        const int64_t e = ends[i];
        if (e <= prev || e >= g.n_layer) {
            err = "boundary " + std::to_string(e) + " is not inside (0," + std::to_string(g.n_layer) +
                  ") and above the previous boundary " + std::to_string(prev);
            return false;
        }
        const int64_t last = e - 1;
        if (!g.is_kv(last) || !g.is_index(last)) {
            err = "a boundary after layer " + std::to_string(last) + " leaves the next stage reading state the "
                  "previous stage does not own (the layer before a boundary must be a kv AND index source)";
            return false;
        }
        prev = e;
    }
    return true;
}

// Rule 2, as the engine's stages_keep_candidates_local: true = nothing crosses a boundary, so no stage has to be told
// anything; false = the split separates the producer from a reader and `owner_out` names the producer's stage.  A
// stranded producer (its readers two or more stages away) sets `stranded`.
inline bool split_keeps_candidates_local(const GeoShape & g, const std::vector<int64_t> & ends, int & owner_out,
                                         bool & stranded, std::string & err) {
    owner_out = -1;
    stranded = false;
    if (!g.cand_relayed()) return true;
    int64_t last_reader = -1;
    for (int64_t k : g.index) if (k > g.cand_layer) last_reader = k > last_reader ? k : last_reader;
    auto stage_of = [&](int64_t l) { int s = 0; for (int64_t e : ends) if (l >= e) ++s; return s; };
    const int owner = stage_of(g.cand_layer), reader = stage_of(last_reader);
    if (owner == reader) return true;
    owner_out = owner;
    // only the stage right after the producer can borrow (or receive) its column
    if (owner + 1 != reader) {
        stranded = true;
        err = "the candidate producer is " + std::to_string(reader - owner) + " stage(s) from a reader";
        return false;
    }
    return false;   // crosses one boundary; the caller relays (across cards) or borrows (on one card)
}

// Rule 3: the payload one token of a boundary carries, for a stage that does (or does not) own the candidate column.
inline uint64_t relay_slab(const GeoShape & g, bool with_cand) {
    return (uint64_t) (g.n_embd * g.hc * 4) + (uint64_t) (g.hc * 4) +
           (with_cand ? (uint64_t) (g.cand_blocks * 4) : 0ull);
}

}  // namespace strata::gfx906
