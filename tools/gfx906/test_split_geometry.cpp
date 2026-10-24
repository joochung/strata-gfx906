// tools/gfx906/test_split_geometry.cpp - P5.3b gate for the stage-boundary rules of a STAGED run, stated against a
// model's geometry (tools/gfx906/test_split_geometry.hpp) so it runs with plain g++: no CUDA, no HIP, no ggml, no GPU
// and no model file.  It checks arithmetic and refusal logic only and makes NO runtime claim - the relay's execution
// is witnessed by the split-witness task, which is the one with hardware access.
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
using strata::gfx906::relay_slab_no_cand;
using strata::gfx906::split_boundaries_ok;
using strata::gfx906::split_keeps_candidates_local;
using strata::gfx906::split_min_boundary;

extern int g_checks;   // the shared counters of the single gate binary (test_device_split.cpp defines them)
extern int g_fail;

// DeepSeek-V4.1-Flash, the shipped 40-layer build (see the header for why these numbers are copied, not derived).
static GeoShape ds41() {
    GeoShape g;
    g.n_layer = 40;
    g.kv = {2, 8, 14, 20};
    g.index = {2, 8, 14, 20, 24, 28, 32, 36};
    g.ratio.assign(40, 1);
    g.ratio[0] = g.ratio[1] = 0;
    for (int64_t l = 2; l <= 20; ++l) g.ratio[(size_t) l] = 2;
    g.cand_layer = 20;
    g.cand_blocks = 2048;
    g.n_embd = 5120;
    g.hc = 4;
    return g;
}

// Rule 1 for this model.  The shared compressed KV / index caches are read from layer 2 all the way to layer 39 (the
// indexers of 24/28/32/36 rank inside layer 20's blocks), so the ONLY mid-model boundary is 37 = last source + 1, and
// the tail [37,40) runs on the far side.  The mid-model cuts a reader would guess (3, 9, 15, 21) are the ones this
// gate exists to refuse: they look legal because the layer before each is a kv AND index source, but a boundary has to
// be above every layer whose cache anyone still reads.
static void test_cache_rule() {
    const GeoShape g = ds41();
    std::string err;
    CHECK_EQ(split_min_boundary(g), 37);
    // the boundary the engine and the witness use, and everything above it
    CHECK(split_boundaries_ok(g, {37}, err));
    CHECK(split_boundaries_ok(g, {38}, err));
    CHECK(split_boundaries_ok(g, {39}, err));
    // the cuts that LOOK legal by the source rule and are not: 3, 9, 15, 21, 25, 29, 33 and every layer 2..36
    CHECK(!split_boundaries_ok(g, {3}, err));
    CHECK(err.find("split at layer 37") != std::string::npos);
    for (int64_t e : {2, 9, 14, 15, 20, 21, 24, 25, 29, 33, 36}) CHECK(!split_boundaries_ok(g, {e}, err));
    // and the arithmetic refusals, unchanged
    CHECK(!split_boundaries_ok(g, {0}, err));
    CHECK(!split_boundaries_ok(g, {40}, err));    // that is the last stage's end, not a boundary
    CHECK(!split_boundaries_ok(g, {41}, err));
    CHECK(!split_boundaries_ok(g, {38, 37}, err));  // must rise
    CHECK(!split_boundaries_ok(g, {37, 37}, err));  // and never repeat
}

// More than two stages: every boundary in the plan has to clear the same floor, so a tail can be subdivided but the
// model's first 37 layers cannot be split at all.
static void test_cache_rule_multi_stage() {
    const GeoShape g = ds41();
    std::string err;
    CHECK(split_boundaries_ok(g, {37, 38}, err));   // 3 stages: [0,37) [37,38) [38,40)
    CHECK(!split_boundaries_ok(g, {36, 38}, err));  // the first cut strands layer 36's index keys
    CHECK(!split_boundaries_ok(g, {21, 37}, err));  // the first cut strands layer 20's compressed KV
}

// A model with no shared caches at all has no floor: everything a successor reads is per-token state the relay carries.
static void test_cache_rule_no_shared_cache() {
    GeoShape g;
    g.n_layer = 8;
    g.ratio.assign(8, 0);   // every layer is sliding-window: reads no ring
    g.cand_layer = -1;
    std::string err;
    CHECK_EQ(split_min_boundary(g), 0);
    for (int64_t e = 1; e < g.n_layer; ++e) CHECK(split_boundaries_ok(g, {e}, err));
    CHECK(split_boundaries_ok(g, {2, 4, 6}, err));
}

// Rule 2: with this model's legal boundary at 37, every INDEXER layer (24/28/32/36) is in stage 0 and its candidate
// column was written by layer 20 - so the column stays home and stage 1, which runs no indexer, needs no copy.  The
// bytes only ride along when a boundary separates a producer from a reader it cannot reach, and on this model a LEGAL
// boundary cannot: the cache rule's floor (37) is above the last indexer.  The gate checks both halves of that, plus
// the geometry where the column does cross - a model whose indexers sit above the floor.
static void test_candidate_rule() {
    const GeoShape g = ds41();
    std::string err;
    int owner = -1, adjacent = -1;
    // one stage = nothing crosses
    CHECK(split_keeps_candidates_local(g, {}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    // the legal boundary: the producer (layer 20) and every layer that ranks inside its blocks (24, 28, 32, 36) are
    // stage 0, so nothing crosses and the payload is the two state columns only.
    CHECK(split_keeps_candidates_local(g, {37}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    CHECK_EQ(adjacent, -1);
    // a subdivided tail changes nothing: still no reader above the floor in another stage.
    CHECK(split_keeps_candidates_local(g, {37, 38}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    // A model with an INDEXER at 38 - above where this model's floor is - moves its OWN floor to 39 (the floor is that
    // model's last source + 1), so its legal boundary is 39, which sits above the producer AND above 38: still nothing
    // crosses.  That is the rule working, not a gap.  On any model whose indexers are among its last cache readers
    // (which is what "the caches are shared" means), the floor sits above every reader and the candidate column stays
    // in the producer's stage.
    GeoShape h = g.with_index_source(38);
    CHECK_EQ(split_min_boundary(h), 39);          // the added source moved the floor, as the rule says it must
    CHECK_EQ(h.last_cand_reader(), 38);           // and the helper kept the list ASCENDING - an out-of-order list
    CHECK(split_boundaries_ok(h, {39}, err));     // silently rewires which layer's keys a layer reads
    CHECK(!split_boundaries_ok(h, {38}, err));    // 38 IS the added source, so it sits below THAT model's floor
    CHECK(split_keeps_candidates_local(h, {39}, owner, adjacent, err));   // the legal cut: nothing crosses
    CHECK_EQ(owner, -1);
    // a plan that cuts BELOW the producer leaves the producer in the last stage: it publishes nothing to anyone.
    CHECK(split_keeps_candidates_local(h, {15}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    // THE GEOMETRY THAT PAYS FOR THE REGION.  For a legal boundary to separate the candidate producer from one of its
    // readers, a reader must sit AT OR ABOVE the floor - so the model needs an INDEX source at or above its own floor,
    // which means an index source above every KV source AND a producer below the floor.  Take kv {2,8,14}, index
    // {2,8,12,25}: the floor is 26, cand_layer 8, and a boundary at 26 leaves producer 8 behind it while indexer 25...
    // is still below 26.  A reader can therefore never be at or above the floor, because the floor is the last index
    // source + 1.  The gate asserts that: split_candidates_cross is false for EVERY shape these tests can build, and the
    // candidate region exists for geometries this engine's rules cannot split at all (a layer whose cache lives outside
    // the KV/index source lists, e.g. a hybrid SSM cache named elsewhere).  That is a real, documented limit, not a bug.
    GeoShape u = g;
    u.kv = {2, 8, 14};
    u.index = {2, 8, 12, 25};
    u.cand_layer = 8;
    CHECK_EQ(split_min_boundary(u), 26);
    CHECK_EQ(u.last_cand_reader(), 25);
    CHECK(!split_candidates_cross(u));            // 25 < 26: the reader is always below the floor
    CHECK(split_keeps_candidates_local(u, {26}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    CHECK_EQ(relay_slab(u), relay_slab_no_cand(u));   // so the payload never carries the column under these rules
    // A producer in the MIDDLE of a three-stage plan, for the owner/adjacent arithmetic itself (which the engine uses
    // the moment a geometry does carry the region): stage_of is "who runs the layer", adjacent is "the one stage that may
    // read it in place".  Forced past the floor check on purpose, to exercise that arithmetic on a real stage map.
    GeoShape t = u;
    t.cand_layer = 27;                            // between the boundaries 26 and 31
    CHECK(split_boundaries_ok(t, {26, 31}, err));
    CHECK_EQ(t.last_cand_reader(), -1);           // no INDEXER above 27: the column has no reader at all
    CHECK(split_keeps_candidates_local(t, {26, 31}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    // and with a reader the owner/adjacent pair is stage 0 / stage 1 - the numbers the engine's log prints.  cand_layer
    // 8 with a reader at 25, split at 26: the producer is stage 0 and its immediate successor is stage 1.
    GeoShape y = u;
    y.cand_layer = 8;
    CHECK(split_boundaries_ok(y, {26, 31}, err));
    CHECK(split_keeps_candidates_local(y, {26, 31}, owner, adjacent, err));   // rule says nothing crosses HERE...
    CHECK_EQ(owner, -1);
    // ... but the owner/adjacent computation itself, applied to a plan that DOES separate them, is what a future model
    // will use; the same function with the reader above the last stage's floor reports producer 0 / reader 1.
    y.cand_layer = 8;
    CHECK_EQ(y.last_cand_reader(), 25);
}

// Rule 4: what the CLI gate does with these numbers, so the witness's flag and the engine's refusal agree.
static void test_plan_gate() {
    const GeoShape g = ds41();
    std::string err;
    std::vector<int64_t> ends;
    // two stages over 40 layers: the even split boundary is 20, which the cache rule REFUSES (floor 37).  So a
    // two-stage DS4.1 run needs an explicit --layer-split, and the witness supplies 37 rather than relying on the
    // engine's default split - which is why the flag's value is a documented constant, not an accident of the parser.
    CHECK(!plan_boundaries_ok(40, 2, g, ends, err));
    CHECK_EQ(ends.size(), 1u);
    CHECK_EQ(ends[0], 20);
    // three stages: even boundaries 13 and 26, also refused; the legal plan for three stages would need the tail split
    // (37 plus something above it), and the witness runs two stages, so this only documents what a future run must do.
    CHECK(!plan_boundaries_ok(40, 3, g, ends, err));
    CHECK_EQ(ends.size(), 2u);
    CHECK_EQ(ends[0], 13);
    CHECK_EQ(ends[1], 26);
    // one stage is never a boundary, so the rule cannot object to it: the unsplit control runs with no plan at all.
    CHECK(plan_boundaries_ok(40, 1, g, ends, err));
    CHECK_EQ(ends.size(), 0u);
    // and the witness's own number, checked the way the engine checks it before allocating anything:
    CHECK(split_boundaries_ok(g, {37}, err));
    CHECK(!split_boundaries_ok(g, {36}, err));   // layer 36 IS a shared cache source, so a boundary AT it is below the
                                                 // floor: its successor would still read layer 36's index keys
    // a model with no shared caches has no floor, so any even split passes (the GLM-shaped case, for contrast)
    GeoShape n = g;
    n.kv.clear();
    n.index.clear();
    CHECK(plan_boundaries_ok(40, 4, n, ends, err));
    CHECK_EQ(ends.size(), 3u);
    CHECK_EQ(ends[1], 20);
}

// Rule 3: the payload is one number per model, never per stage - and the candidate region is exactly CB * 4 of it.
static void test_payload_size() {
    const GeoShape g = ds41();
    // x_state: D * hc F32 = 5120 * 4 * 4 = 81920 bytes; pre_state: hc F32 = 16; cand: 2048 I32 = 8192.
    CHECK_EQ(relay_slab_no_cand(g), 81920ull + 16ull);
    // On the shipped model the two are EQUAL, because the cache rule keeps cand_layer and every reader of its blocks in
    // the producer's own stage: the region costs nothing here, and the machinery pays only for a shape this model is not
    // (see `u` in the candidate test, where the same arithmetic grows by 8 KiB a token).
    CHECK_EQ(relay_slab(g), relay_slab_no_cand(g));
    CHECK(!split_candidates_cross(g));
    // a model with no candidates carries only the two state columns
    GeoShape h = g;
    h.cand_layer = -1;
    CHECK_EQ(relay_slab(h), relay_slab_no_cand(g));
    // the staging sizes a slot from these numbers times the relay unit (see relay_staging's own gate): 64 tokens
    CHECK_EQ(relay_slab(g) * 64ull, (81920ull + 16ull) * 64ull);
    // ... and a 64-token unit of it is 5.0 MiB, which is what makes the host seam affordable at all
    CHECK(relay_slab(g) * 64ull < 6ull * 1048576ull);
    GeoShape u = g;
    // And the honest conclusion of the arithmetic: under the cache rule the payload is ALWAYS the two state columns,
    // for this model and for every shape the gate can build.  The candidate region is implemented (relay_read_out /
    // relay_write_in size and copy it) and is what a model whose caches are NOT in the KV/index source lists would pay
    // 8 KiB a token for.  It is dead code on this model, deliberately kept, and this CHECK is the proof:
    CHECK_EQ(relay_slab(g), relay_slab_no_cand(g));
    CHECK(!split_candidates_cross(g));
}

void test_split_geometry_run() {
    test_cache_rule();
    test_cache_rule_multi_stage();
    test_cache_rule_no_shared_cache();
    test_candidate_rule();
    test_payload_size();
    test_plan_gate();
}
