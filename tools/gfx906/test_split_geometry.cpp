// tools/gfx906/test_split_geometry.cpp - P5.3b gate for the stage-boundary rules of a STAGED run, stated against a
// model's geometry (tools/gfx906/test_split_geometry.hpp) so it runs with plain g++: no CUDA, no HIP, no ggml, no GPU
// and no model file.  It checks arithmetic and refusal logic only and makes NO runtime claim - the relay's execution
// is witnessed by the split-smoke task, which is the one with hardware access.
//
//   g++ -std=c++17 -O0 -Wall -Wextra -Werror -I tools/gfx906 -o /tmp/test_split_geometry
//       tools/gfx906/test_split_geometry.cpp  &&  /tmp/test_split_geometry
//
// Included by tools/gfx906/test_device_split.cpp so one binary and one gate script cover the whole split plumbing.
#include "test_split_geometry.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::gfx906::GeoShape;
using strata::gfx906::relay_slab;
using strata::gfx906::split_boundaries_ok;
using strata::gfx906::split_keeps_candidates_local;

extern int g_checks;   // the shared counters of the single gate binary (test_device_split.cpp defines them)
extern int g_fail;

// DeepSeek-V4.1-Flash, the shipped 40-layer build (see the header for why these numbers are copied, not derived).
static GeoShape ds41() {
    GeoShape g;
    g.n_layer = 40;
    g.kv = {0, 2, 8, 14, 20};
    g.index = {0, 2, 8, 14, 20, 24, 28, 32, 36};
    g.cand_layer = 20;
    g.cand_blocks = 2048;
    g.n_embd = 5120;
    g.hc = 4;
    return g;
}

// The legal boundaries of that model: a boundary sits AFTER a kv AND index source, i.e. at layer 3, 9, 15, 21 (the
// source + 1), or past the last index source, where no successor reads another stage's rings.
static void test_kv_rule() {
    const GeoShape g = ds41();
    std::string err;
    // the source layers themselves are NOT boundaries: a boundary at 20 puts layer 19 in the next stage, and 19's
    // kv source is 14, which the previous stage owns but did not hand over.
    CHECK(!split_boundaries_ok(g, {20}, err));
    CHECK(err.find("does not own") != std::string::npos);
    CHECK(!split_boundaries_ok(g, {2}, err));
    // Layer 0 is BOTH a kv and an index source, so a boundary at 1 is legal by the rule (the next stage reads layer
    // 0's rings, which the first stage produced and hands over).  It is a silly split - one layer on the first card -
    // but it is not the kind of thing the gate refuses.
    CHECK(split_boundaries_ok(g, {1}, err));
    CHECK(!split_boundaries_ok(g, {21, 22}, err));  // the second boundary (layer 21) is not a source
    // the four legal mid-model boundaries and the "everything after the candidates" one
    for (int64_t e : {3, 9, 15, 21}) CHECK(split_boundaries_ok(g, {e}, err));
}

static void test_kv_rule_rejects_index_only_sources() {
    const GeoShape g = ds41();
    std::string err;
    // 24/28/32/36 are index sources but not kv sources: a boundary there would relay keys but not KV.
    for (int64_t e : {25, 29, 33, 37}) CHECK(!split_boundaries_ok(g, {e}, err));
    // the only mid-model legal boundaries are the four source+1 values, and they chain:
    CHECK(split_boundaries_ok(g, {3, 9, 15, 21}, err));
    CHECK(!split_boundaries_ok(g, {9, 3}, err));   // must rise
    CHECK(!split_boundaries_ok(g, {40}, err));     // not a boundary; that is the last stage's end
    CHECK(!split_boundaries_ok(g, {0}, err));
}

static void test_candidate_rule() {
    const GeoShape g = ds41();
    std::string err;
    int owner = -1;
    bool stranded = false;
    // one stage = nothing crosses
    CHECK(split_keeps_candidates_local(g, {}, owner, stranded, err));
    CHECK_EQ(owner, -1);
    // a boundary inside the producer's own stage: cand_layer stays with its readers
    CHECK(split_keeps_candidates_local(g, {15}, owner, stranded, err));
    // a boundary at 21 (just after the producer): the next stage's indexers rank inside layer 20's blocks.  The
    // producer is stage 0 and the first reader is stage 1, so the column crosses exactly one boundary - borrowable
    // on one card, relatable across cards.
    CHECK(!split_keeps_candidates_local(g, {21}, owner, stranded, err));
    CHECK_EQ(owner, 0);
    CHECK(!stranded);
    // a boundary at 9 keeps the producer and every reader together: nothing crosses.
    CHECK(split_keeps_candidates_local(g, {9}, owner, stranded, err));
    // two boundaries, producer in stage 0 and readers in stage 2: stage 2 borrows from stage 1, which never wrote it.
    CHECK(!split_keeps_candidates_local(g, {21, 25}, owner, stranded, err));
    CHECK(stranded);
    CHECK_EQ(owner, 0);
    // and a split whose producer stage ends at cand_layer + 1 is fine at any stage number.
    CHECK(!split_keeps_candidates_local(g, {21}, owner, stranded, err));
    CHECK(!stranded);
}

static void test_payload_size() {
    const GeoShape g = ds41();
    // x_state: D * hc F32 = 5120 * 4 * 4 = 81920 bytes; pre_state: hc F32 = 16; cand: 2048 I32 = 8192.
    CHECK_EQ(relay_slab(g, false), 81920ull + 16ull);
    CHECK_EQ(relay_slab(g, true), 81920ull + 16ull + 8192ull);
    // the difference is exactly the candidate region, and a stage that borrows the producer's column pays it
    CHECK_EQ(relay_slab(g, true) - relay_slab(g, false), (uint64_t) (g.cand_blocks * 4));
    // 21 tokens of a boundary: the staging sizes its slot from these numbers (see relay_staging's own gate).
    CHECK_EQ(relay_slab(g, true) * 21ull, (81920ull + 16ull + 8192ull) * 21ull);
}

void test_split_geometry_run() {
    test_kv_rule();
    test_kv_rule_rejects_index_only_sources();
    test_candidate_rule();
    test_payload_size();
}
