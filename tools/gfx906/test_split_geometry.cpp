// tools/gfx906/test_split_geometry.cpp - P5.3b gate for the stage-boundary rules of a STAGED run, stated against a
// model's geometry (tools/gfx906/test_split_geometry.hpp) so it runs with plain g++: no CUDA, no HIP, no ggml, no GPU
// and no model file.  It checks arithmetic and refusal logic only and makes NO runtime claim - the relay's execution
// is witnessed by the split-witness task, which is the one with hardware access.
//
//   g++ -std=c++17 -O0 -Wall -Wextra -Werror -I tools/gfx906 -o /tmp/test_split_geometry
//       tools/gfx906/test_split_geometry.cpp  &&  /tmp/test_split_geometry
//
// Included by tools/gfx906/test_device_split.cpp so one binary and one gate script cover the whole split plumbing.
//
// THE SHAPE OF THESE TESTS IS THE POINT.  Rule 1 is asserted twice over, from two directions: a short list of
// hand-anchored expectations (the numbers a reader expects to see, and the refusal messages they print), and a DATA-
// DRIVEN invariant that re-derives the answer by simulating the stages and asking each of its layers the runtime's own
// question (split_stages_own_their_reads).  Every single boundary and every 2- and 3-boundary plan of every geometry
// here is checked against that scan.  A previous revision of this file asserted a FLOOR ("last shared-cache layer + 1" =
// 37) instead of the valid set, and the direction of the rule was wrong in every layer of the engine; the scan below is
// what would have caught it, so it is the first thing these tests do, not a footnote.
#include "test_split_geometry.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using strata::gfx906::GeoShape;
using strata::gfx906::plan_boundaries_ok;
using strata::gfx906::plan_even_chunks;
using strata::gfx906::relay_slab;
using strata::gfx906::relay_slab_no_cand;
using strata::gfx906::split_boundaries_ok;
using strata::gfx906::split_boundary_valid;
using strata::gfx906::split_candidates_cross;
using strata::gfx906::split_keeps_candidates_local;
using strata::gfx906::split_snap_to_valid;
using strata::gfx906::split_stages_own_their_reads;
using strata::gfx906::split_valid_boundaries;
using strata::gfx906::split_valid_set_text;

extern int g_checks;   // the shared counters of the single gate binary (test_device_split.cpp defines them)
extern int g_fail;

// DeepSeek-V4.1-Flash, the shipped 40-layer build (see the header for why these numbers are copied, not derived - and
// for the gate step that cross-checks the copy against the engine's own reader).
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

// THE INVARIANT.  For every plan of 1..3 boundaries over this geometry, the predicate and the runtime simulation agree.
// The scan is written from the engine's OWN checks (Impl::owns_layer, build_attn's two aborts, L.uses_cand), so a rule
// that drifts away from the runtime - in either direction - fails here rather than at graph build on a GPU.
static void test_matches_the_runtime_scan(const GeoShape & g, const char * what) {
    std::string err;
    const int64_t n = g.n_layer;
    long nplans = 0;
    // every single boundary, in both directions
    for (int64_t K = 1; K < n; ++K) {
        const std::vector<int64_t> ends{K};
        const bool scan = split_stages_own_their_reads(g, ends);
        CHECK(split_boundaries_ok(g, ends, err) == scan && split_boundary_valid(g, K) == scan);
        ++nplans;
    }
    // every pair and every triple: the multi-stage claim ("every boundary is valid") has to be the same claim as
    // "each stage's layers find their rings inside that stage", or the message a user gets is not the rule that runs
    for (int64_t a = 1; a < n; ++a)
        for (int64_t b = a + 1; b < n; ++b) {
            const std::vector<int64_t> ends{a, b};
            CHECK(split_boundaries_ok(g, ends, err) == split_stages_own_their_reads(g, ends));
            ++nplans;
            for (int64_t c = b + 1; c < n; ++c) {
                const std::vector<int64_t> e3{a, b, c};
                CHECK(split_boundaries_ok(g, e3, err) == split_stages_own_their_reads(g, e3));
                ++nplans;
            }
        }
    // and the enumerator is exactly the set the predicate accepts, ascending with no duplicates
    const std::vector<int64_t> v = split_valid_boundaries(g);
    for (size_t i = 0; i < v.size(); ++i) {
        CHECK(split_boundary_valid(g, v[i]));
        if (i) CHECK(v[i] > v[i - 1]);
    }
    for (int64_t K = 1; K < n; ++K)
        CHECK(split_boundary_valid(g, K) == (std::find(v.begin(), v.end(), K) != v.end()));
    std::printf("ok: %s: valid boundaries %s; %ld plans (all 1/2/3-boundary cuts of %lld layers) match the runtime scan\n",
                what, split_valid_set_text(g).c_str(), nplans, (long long) n);
}

// Rule 1 for this model, hand-anchored: the valid SET, and the refusals - including the one this file used to mandate.
static void test_cache_rule() {
    const GeoShape g = ds41();
    std::string err;
    // The layers that are BOTH a kv and an index source, plus 1 (ratio[1] == 0, so it reads no ring and nothing above it
    // reaches below it).  split_plan refuses a --layer-split below 2, so {2, 8, 14, 20} is what a caller can type.
    CHECK_EQ(split_valid_boundaries(g), (std::vector<int64_t>{1, 2, 8, 14, 20}));
    CHECK(split_valid_set_text(g) == "1, 2, 8, 14, 20");
    // THE EVEN SPLIT OF A 40-LAYER MODEL IN TWO STAGES IS LEGAL, and it is what the witness runs: stage 1 = [20,40) holds
    // the writer of ring 20 and every reader of it (20..39), the index rings 24/28/32/36 writers + readers, and the
    // candidate writer 20 with its readers 24/28/32/36.
    CHECK(split_boundaries_ok(g, {20}, err));
    CHECK(split_stages_own_their_reads(g, {20}));
    // every member of the set passes, and nothing else does
    for (int64_t e : {1, 2, 8, 14, 20}) CHECK(split_boundaries_ok(g, {e}, err));
    for (int64_t e : {3, 9, 15, 21, 24, 25, 29, 33, 36, 37, 38, 39}) {
        CHECK(!split_boundaries_ok(g, {e}, err));
        CHECK(!split_stages_own_their_reads(g, {e}));
    }
    // THE DEATH THIS GATE USED TO MANDATE.  37 = "last shared-cache layer + 1" was the previous revision's answer, and
    // it is the boundary that aborts: the tail [37,40) reads layer 20's compressed KV and layer 36's index keys, both of
    // which sit below it.  The refusal names the layer and prints the set - this CHECK is the message an operator with
    // the old --layer-split 37 sees.
    CHECK(!split_boundaries_ok(g, {37}, err));
    CHECK(err.find("layer 37 reads the compressed KV of layer 20") != std::string::npos);
    CHECK(err.find("valid boundaries are 1, 2, 8, 14, 20") != std::string::npos);
    // 21 was the other plausible guess (the layer before it is a kv AND index source): kv_src(21) == 20 < 21, so ring 20
    // stays on the producer's card.  Real abort, wrong generalization.
    CHECK(!split_boundaries_ok(g, {21}, err));
    CHECK(err.find("layer 21 reads the compressed KV of layer 20") != std::string::npos);
    // a boundary at an index source that is not a kv source reaches back to ring 20 the same way
    CHECK(!split_boundaries_ok(g, {36}, err));
    CHECK(err.find("layer 36 reads the compressed KV of layer 20") != std::string::npos);
    // the arithmetic refusals, unchanged
    CHECK(!split_boundaries_ok(g, {0}, err));
    CHECK(!split_boundaries_ok(g, {40}, err));    // that is the last stage's end, not a boundary
    CHECK(!split_boundaries_ok(g, {41}, err));
    CHECK(!split_boundaries_ok(g, {20, 8}, err));   // must rise
    CHECK(!split_boundaries_ok(g, {20, 20}, err));  // and never repeat
    CHECK(!split_boundaries_ok(g, {8, 8}, err));
    // an invalid boundary is refused wherever it sits in a plan, even behind a legal one
    CHECK(!split_boundaries_ok(g, {2, 21}, err));
    CHECK(!split_boundaries_ok(g, {21, 37}, err));
}

// More than two stages: the model CAN be cut in the middle, several times, as long as every cut is a member of the set.
// {8, 14, 20} is the four-card plan a --gpus 0,1,2,3 run needs; the even thirds (13, 26) are not members and snap.
static void test_cache_rule_multi_stage() {
    const GeoShape g = ds41();
    std::string err;
    CHECK(split_boundaries_ok(g, {8, 14, 20}, err));      // 4 stages: [0,8) [8,14) [14,20) [20,40)
    CHECK(split_stages_own_their_reads(g, {8, 14, 20}));
    CHECK(split_boundaries_ok(g, {2, 8, 14, 20}, err));   // 5 stages, the whole usable set
    CHECK(split_stages_own_their_reads(g, {2, 8, 14, 20}));
    CHECK(split_boundaries_ok(g, {8, 20}, err));          // 3 stages, the snapped even thirds
    CHECK(split_stages_own_their_reads(g, {8, 20}));
    CHECK(!split_boundaries_ok(g, {13, 26}, err));        // the UNSNAPPED even thirds
    CHECK(!split_boundaries_ok(g, {10, 20, 30}, err));    // the unsnapped even quarters: 30 is past the set
    CHECK(!split_boundaries_ok(g, {20, 37}, err));        // a legal cut followed by the old floor still dies
    CHECK(!split_stages_own_their_reads(g, {20, 37}));
}

// A model with no shared caches at all has no constraint: every layer's state is per-token state the relay carries.
static void test_cache_rule_no_shared_cache() {
    GeoShape g;
    g.n_layer = 8;
    g.ratio.assign(8, 0);   // every layer is sliding-window: reads no ring
    g.cand_layer = -1;
    std::string err;
    for (int64_t e = 1; e < g.n_layer; ++e) {
        CHECK(split_boundaries_ok(g, {e}, err));
        CHECK(split_stages_own_their_reads(g, {e}));
    }
    CHECK(split_boundaries_ok(g, {2, 4, 6}, err));
    CHECK_EQ(split_valid_boundaries(g).size(), 7u);   // every interior layer
}

// Rule 2: on this model the candidate column never crosses a boundary, because validity keeps a producer with its
// readers - and the geometry where it COULD cross is one this rule cannot build, which the checks below assert rather
// than assume.
static void test_candidate_rule() {
    const GeoShape g = ds41();
    std::string err;
    int owner = -1, adjacent = -1;
    // one stage = nothing crosses
    CHECK(split_keeps_candidates_local(g, {}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    // the witness's boundary: the producer (layer 20) and every layer that ranks inside its blocks (24, 28, 32, 36) are
    // in stage 1 together, so nothing crosses and the payload is the two state columns only.
    CHECK(split_keeps_candidates_local(g, {20}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    CHECK_EQ(adjacent, -1);
    // the four-card plan: 24/28/32/36 and the writer 20 all land in the LAST stage, so still nothing crosses.
    CHECK(split_keeps_candidates_local(g, {8, 14, 20}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    CHECK(!split_candidates_cross(g));
    // ADDING AN INDEX SOURCE at 38 (an indexer above every cache source) does NOT move the set: a boundary has to be a
    // kv source too, and 38 is not one.  The old rule's floor jumped to 39 on this shape; the corrected rule's set is
    // unchanged, and the column still stays home (the highest valid boundary is still 20, the producer itself).
    GeoShape h = g.with_index_source(38);
    CHECK_EQ(h.last_cand_reader(), 38);                 // and the helper kept the list ASCENDING
    CHECK_EQ(split_valid_boundaries(h), split_valid_boundaries(g));
    CHECK(split_boundaries_ok(h, {20}, err));
    CHECK(!split_boundaries_ok(h, {38}, err));          // kv_src(38) == 20 < 38: an index source is not a kv source
    CHECK(!split_candidates_cross(h));
    CHECK(split_keeps_candidates_local(h, {20}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    // A MODEL WHOSE Caches are sparser: kv {2,8,14}, index {2,8,12,25}, cand 8.  The set is {1, 2, 8} - 12 is an index
    // source but not a kv source (kv_src(12) == 8), and 25 reaches back to ring 14.  The producer 8 is the highest
    // member, so its readers 12 and 25 come with it and the column stays home again.
    GeoShape u = g;
    u.kv = {2, 8, 14};
    u.index = {2, 8, 12, 25};
    u.cand_layer = 8;
    CHECK_EQ(split_valid_boundaries(u), (std::vector<int64_t>{1, 2, 8}));
    CHECK_EQ(u.last_cand_reader(), 25);
    CHECK(!split_candidates_cross(u));
    CHECK(split_keeps_candidates_local(u, {8}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    CHECK_EQ(relay_slab(u), relay_slab_no_cand(u));       // so the payload never carries the column under these rules
    // A PRODUCER BETWEEN TWO LEGAL BOUNDARIES, for the owner/adjacent arithmetic itself (which the engine uses the
    // moment a geometry does carry the region).  cand_layer 11 in `u`, split at {2, 8}: the producer lands in stage 2 and
    // its readers (12, 24, 25, 28, 32, 36) with it, so `separated` is false and owner stays unnamed - the stage map,
    // not the wish, decides.
    GeoShape t = u;
    t.cand_layer = 11;                                    // ratio[11] > 0 and not a source of either kind
    CHECK(split_boundaries_ok(t, {2, 8}, err));
    CHECK_EQ(t.last_cand_reader(), 25);
    CHECK(split_keeps_candidates_local(t, {2, 8}, owner, adjacent, err));
    CHECK_EQ(owner, -1);
    // and the stage map itself: with boundaries {2, 8}, stage_of is who runs the layer - the numbers the engine's log
    // prints.  Producer 11 -> stage 2; its readers 12 and 25 -> stage 2 too, so the column never crosses.
    CHECK_EQ(t.stage_of(11, {2, 8}), 2);
    CHECK_EQ(t.stage_of(12, {2, 8}), 2);
    CHECK_EQ(t.stage_of(25, {2, 8}), 2);
    CHECK_EQ(t.stage_of(3, {2, 8}), 1);
    CHECK_EQ(t.stage_of(1, {2, 8}), 0);
    // THE HONEST CONCLUSION OF THE ARITHMETIC, asserted instead of asserted-about: for EVERY shape this gate can build,
    // no valid boundary separates the candidate writer from a reader.  Rule 1 requires cand_layer >= K for any K that
    // leaves a reader at or above it, so the region is dead code on every model whose caches are these two source lists,
    // and live only for a model with a cache OUTSIDE them (a hybrid SSM state, a cache named by a key the geometry does
    // not read yet).  It is implemented, sized and gated, and does not fire here.
    const GeoShape shapes[5] = { g, h, u, t, GeoShape() };
    for (const GeoShape & s : shapes) CHECK(!split_candidates_cross(s));
}

// Rule 3: the payload is one number per model, never per stage - and the candidate region is exactly CB * 4 of it.
static void test_payload_size() {
    const GeoShape g = ds41();
    // x_state: D * hc F32 = 5120 * 4 * 4 = 81920 bytes; pre_state: hc F32 = 16; cand: 2048 I32 = 8192.
    CHECK_EQ(relay_slab_no_cand(g), 81920ull + 16ull);
    // On the shipped model the two are EQUAL, because the reader's rule keeps cand_layer and every reader of its blocks
    // in one stage: the region costs nothing here, and the machinery pays only for a shape this model is not.
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
}

// Rule 4: what the CLI gate does with these numbers, so the witness's flag and the engine's answer agree.  The engine
// PLANS, then SNAPS DOWN, then refuses only what snapping cannot save.
static void test_plan_gate() {
    const GeoShape g = ds41();
    std::string err;
    std::vector<int64_t> ends;
    // two stages over 40 layers: the even split boundary is 20, which IS valid - no snap, no refusal, no flag needed.
    // (This is the whole point of the correction: the even split was the case the old rule refused.)
    CHECK_EQ(plan_even_chunks(40, 2).size(), 1u);
    CHECK_EQ(plan_even_chunks(40, 2)[0], 20);
    CHECK(plan_boundaries_ok(40, 2, g, ends, err));
    CHECK_EQ(ends, (std::vector<int64_t>{20}));
    // three stages: even boundaries 13 and 26 are not members, and snap DOWN to 8 and 20 - a real 3-stage plan whose
    // stage widths are 8 / 12 / 20 layers.
    CHECK_EQ(plan_even_chunks(40, 3), (std::vector<int64_t>{13, 26}));
    CHECK(plan_boundaries_ok(40, 3, g, ends, err));
    CHECK_EQ(ends, (std::vector<int64_t>{8, 20}));
    CHECK(split_stages_own_their_reads(g, ends));
    // four stages: even quarters are 10, 20 and 30.  10 snaps to 8, 20 stays, and 30 has nowhere to go ABOVE 20 - the
    // set stops at 20 - so the plan is refused with the set named, not run as a three-stage plan behind the user's back.
    CHECK_EQ(plan_even_chunks(40, 4), (std::vector<int64_t>{10, 20, 30}));
    CHECK(!plan_boundaries_ok(40, 4, g, ends, err));
    CHECK(err.find("valid boundaries are 1, 2, 8, 14, 20") != std::string::npos);
    CHECK(err.find("cannot be snapped") != std::string::npos);
    // ... and the SAME four cards work when the caller names the boundaries the geometry has: {8, 14, 20}.
    CHECK(split_boundaries_ok(g, {8, 14, 20}, err));
    // five stages cannot be snapped either (8, 16, 24, 32 -> 8, 14, 20, nothing): four stages is the most this
    // checkpoint's geometry has boundaries for.
    CHECK(!split_snap_to_valid(g, plan_even_chunks(40, 5), ends, err));
    CHECK(!plan_boundaries_ok(40, 5, g, ends, err));
    CHECK(split_snap_to_valid(g, plan_even_chunks(40, 4 - 1), ends, err));   // three stages: saves cleanly
    CHECK_EQ(ends, (std::vector<int64_t>{8, 20}));
    // an explicit --layer-split goes through the same snap.  13 is not a member and takes the nearest one below it;
    // 21 (the other plausible guess, one past the kv AND index source 20) snaps to 20 and the run SAYS so;
    // snapping is monotone and never repeats, so a plan whose later boundary forces an earlier one down moves the
    // earlier one DOWN ({13, 15} -> {8, 14}) rather than inventing a layer that does not exist.
    CHECK(split_snap_to_valid(g, {13}, ends, err));
    CHECK_EQ(ends, (std::vector<int64_t>{8}));
    CHECK(split_snap_to_valid(g, {21}, ends, err));
    CHECK_EQ(ends, (std::vector<int64_t>{20}));
    CHECK(split_snap_to_valid(g, {13, 15}, ends, err));
    CHECK_EQ(ends, (std::vector<int64_t>{8, 14}));
    // and a plan that asks for TWO boundaries above the set cannot be saved: 20 snaps to 20, and there is nothing valid
    // above 20 for the second one.
    CHECK(!split_snap_to_valid(g, {20, 26}, ends, err));
    CHECK(!split_snap_to_valid(g, {26, 33}, ends, err));    // 26->20, then nothing above 20
    CHECK(split_boundaries_ok(g, ends, err));               // ... but whatever snapping returns IS legal
    // one stage is never a boundary, so the rule cannot object to it: the unsplit control runs with no plan at all.
    CHECK(plan_boundaries_ok(40, 1, g, ends, err));
    CHECK_EQ(ends.size(), 0u);
    // and the witness's own number, checked the way the engine checks it before allocating anything:
    CHECK(split_boundaries_ok(g, {20}, err));
    CHECK(!split_boundaries_ok(g, {37}, err));
    // a model with no shared caches has no constraint, so any even split passes - the shape the borrowed rule was
    // actually written for.  "No shared caches" means no layer reads a ring, which is what ratio == 0 says.
    GeoShape n = g;
    n.kv.clear();
    n.index.clear();
    n.cand_layer = -1;
    n.ratio.assign(40, 0);
    CHECK(plan_boundaries_ok(40, 4, n, ends, err));
    CHECK_EQ(ends.size(), 3u);
    CHECK_EQ(ends, plan_even_chunks(40, 4));   // nothing to snap: 10, 20 and 30 are all legal here
    // and the same geometry with ratio > 0 but NO source lists is a corrupt model, not a free one: a layer that runs
    // attention with no named source reads nothing the geometry can vouch for, and the loader aborts on that shape.
    GeoShape bad = g;
    bad.kv.clear();
    bad.index.clear();
    bad.cand_layer = -1;
    CHECK_EQ(split_valid_boundaries(bad).size(), 0u);
}

void test_split_geometry_run() {
    // The invariant first: it is the test that makes the direction of the rule impossible to get wrong quietly.
    test_matches_the_runtime_scan(ds41(), "DeepSeek-V4.1-Flash 40L");
    GeoShape u = ds41();
    u.kv = {2, 8, 14};
    u.index = {2, 8, 12, 25};
    u.cand_layer = 8;
    test_matches_the_runtime_scan(u, "sparse-cache variant");
    test_matches_the_runtime_scan(ds41().with_index_source(38), "indexer above every cache source");
    {
        GeoShape n;
        n.n_layer = 8;
        n.ratio.assign(8, 0);
        n.cand_layer = -1;
        test_matches_the_runtime_scan(n, "no shared caches");
    }
    test_cache_rule();
    test_cache_rule_multi_stage();
    test_cache_rule_no_shared_cache();
    test_candidate_rule();
    test_payload_size();
    test_plan_gate();
}
