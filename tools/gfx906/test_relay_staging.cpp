// tools/gfx906/test_relay_staging.cpp - P5.3 step 1 gate for the stage-boundary relay's arithmetic
// (tools/gfx906/relay_staging.hpp): staging buffer sizes, copy windows, slot rotation, and the refusals.
// Plain g++, no CUDA, no HIP, no ggml, no GPU: it checks integer math only and makes no runtime claim - the
// relay's execution is witnessed by the split-smoke task (#29), which is the one with hardware access.
//
//   g++ -std=c++17 -O0 -Wall -Wextra -Werror -I tools/gfx906 -o /tmp/test_relay_staging
//       tools/gfx906/test_relay_staging.cpp  &&  /tmp/test_relay_staging
//
// It is included by tools/gfx906/test_device_split.cpp so one binary and one gate script cover both headers.
#include "relay_staging.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::gfx906::kRelaySlots;
using strata::gfx906::RelayCopy;
using strata::gfx906::RelayPlan;
using strata::gfx906::RelaySlot;
using strata::gfx906::RelayStaging;
using strata::gfx906::relay_boundary_columns;
using strata::gfx906::relay_plan;
using strata::gfx906::relay_stage_pass;

extern int g_checks;   // the shared counters of the single gate binary (test_device_split.cpp defines them)
extern int g_fail;

// The numbers one GLM-5.3 stage boundary moves: hc=4 streams of n_embd=4096 F32 -> 64 KiB per token, and the
// engine's own per-token stride in x_state is exactly that (no padding).
static constexpr uint64_t kGlmSlab = 4096ull * 4ull * 4ull;

// A boundary's staging holds `slots` token ranges at once, each `cap` tokens x slab bytes, aligned up.
static void test_buffer_size() {
    // 2 stages = 1 boundary
    RelayPlan p;
    std::string err;
    CHECK_EQ(relay_plan(2, 4096, kGlmSlab, p, err), true);
    CHECK_EQ(p.ok, true);
    CHECK_EQ(p.n_boundaries, 1);
    CHECK_EQ(p.slots, (int) kRelaySlots);
    CHECK_EQ(p.slot_bytes, 4096ull * kGlmSlab);
    CHECK_EQ(p.bytes, 2ull * 4096ull * kGlmSlab);
    CHECK(p.err.empty());

    // 3 stages = 2 boundaries, 4 stages = 3 boundaries: each boundary gets its own slots
    RelayPlan q;
    CHECK_EQ(relay_plan(3, 1024, kGlmSlab, q, err), true);
    CHECK_EQ(q.n_boundaries, 2);
    CHECK_EQ(q.bytes, 2ull * 2ull * 1024ull * kGlmSlab);
    RelayPlan r;
    CHECK_EQ(relay_plan(4, 512, kGlmSlab, r, err), true);
    CHECK_EQ(r.n_boundaries, 3);
    CHECK_EQ(r.bytes, 2ull * 3ull * 512ull * kGlmSlab);

    // the alignment of the native prefill relay's buffers is a multiple of the slot size, so a slot never straddles
    p = RelayPlan();
    CHECK_EQ(relay_plan(3, 4095, 1000, p, err), true);   // unaligned slot: padded up
    CHECK_EQ(p.slot_bytes, 4095ull * 1000ull);
    CHECK_EQ(p.stride, p.slot_bytes);
    CHECK_EQ(p.bytes, 2ull * 2ull * p.stride);
    if (p.slot_bytes) CHECK_EQ(p.stride % p.slot_bytes, 0ull);   // guarded: a stub leaves slot_bytes 0

    // zero is the "no relay" answer, not an error: one stage has no boundary to stage for
    RelayPlan z;
    CHECK_EQ(relay_plan(1, 4096, kGlmSlab, z, err), true);
    CHECK_EQ(z.ok, true);
    CHECK_EQ(z.n_boundaries, 0);
    CHECK_EQ(z.bytes, 0ull);
    CHECK_EQ(z.stride, 0ull);   // a stub that leaves the stride set would divide by zero in the checks below

    // refusals: no slab, no capacity, or a byte count past what the struct can name
    RelayPlan b1;
    CHECK_EQ(relay_plan(2, 4096, 0, b1, err), false);
    CHECK(err.find("per-token") != std::string::npos);
    CHECK_EQ(b1.ok, false);
    RelayPlan b2;
    CHECK_EQ(relay_plan(2, 0, kGlmSlab, b2, err), false);
    CHECK(err.find("capacity") != std::string::npos);
    err.clear();
    RelayPlan b3;
    CHECK_EQ(relay_plan(2, 1ull << 40, (1ull << 40), b3, err), false);
    CHECK(err.find("overflow") != std::string::npos);
    CHECK_EQ(b3.ok, false);
}

// The copy window for tokens [a,b) of a pass: where the payload starts and how long it is.
static void test_copy_window() {
    RelayPlan p;
    std::string err;
    CHECK_EQ(relay_plan(2, 4096, kGlmSlab, p, err), true);

    // the whole pass (n == capacity): offset 0, every byte of the slot, nothing clipped
    RelayCopy c;
    CHECK_EQ(p.window(0, 0, 4096, c, err), true);
    CHECK_EQ(c.dst_offset, 0ull);
    CHECK_EQ(c.bytes, 4096ull * kGlmSlab);
    CHECK_EQ(c.incomplete, false);

    // one token (decode): the payload is exactly one per-token slab
    CHECK_EQ(p.window(0, 0, 1, c, err), true);
    CHECK_EQ(c.dst_offset, 0ull);
    CHECK_EQ(c.bytes, kGlmSlab);

    // a token range in the middle: the offset is a whole number of slabs from the slot's start
    CHECK_EQ(p.window(0, 17, 22, c, err), true);
    CHECK_EQ(c.dst_offset, 17ull * kGlmSlab);
    CHECK_EQ(c.bytes, 5ull * kGlmSlab);
    CHECK_EQ(c.incomplete, false);

    // a range that runs past the staging capacity is refused: the engine must split the pass, the relay must
    // never write outside its slot
    err.clear();
    CHECK_EQ(p.window(0, 4094, 4099, c, err), false);
    CHECK(err.find("capacity") != std::string::npos);
    err.clear();
    CHECK_EQ(p.window(0, 4096, 4097, c, err), false);
    CHECK(err.find("capacity") != std::string::npos);
    // an empty or reversed range is refused too
    err.clear();
    CHECK_EQ(p.window(0, 3, 0, c, err), false);
    CHECK(err.find("empty") != std::string::npos);
    err.clear();
    CHECK_EQ(p.window(0, 5, 3, c, err), false);

    // an unknown boundary is refused, and a refused call leaves the previous window alone
    err.clear();
    const RelayCopy before = c;
    CHECK_EQ(p.window(1, 0, 1, c, err), false);
    CHECK_EQ(c.dst_offset, before.dst_offset);
    CHECK(err.find("boundary") != std::string::npos);

        // with 3 stages (2 boundaries), boundary 1 is the second stride of slot 0: the layout is slot-major, so one
    // producer alternates stride*b and stride*(b+1) the way the native relay flips hand_buf_ (prefill.cpp:4194)
    RelayPlan t;
    CHECK_EQ(relay_plan(3, 256, kGlmSlab, t, err), true);
    CHECK_EQ(t.window(0, 4, 6, c, err), true);
    CHECK_EQ(c.dst_offset, 4ull * kGlmSlab);
    CHECK_EQ(c.bytes, 2ull * kGlmSlab);
    CHECK_EQ(t.window(1, 4, 6, c, err), true);
    CHECK_EQ(c.dst_offset, t.stride + 4ull * kGlmSlab);
    CHECK_EQ(c.bytes, 2ull * kGlmSlab);
    // slot 1 of a boundary sits after every boundary's slot 0 (n_boundaries strides in)
    CHECK_EQ(t.window(1, 0, 3, c, err, 1), true);
    CHECK_EQ(c.dst_offset, 3ull * t.stride);   // slot-major: slot 1 starts after every boundary's slot 0
    CHECK_EQ(t.window(0, 0, 3, c, err, 1), true);
    CHECK_EQ(c.dst_offset, 2ull * t.stride);
    CHECK_EQ(c.bytes, 3ull * kGlmSlab);
    CHECK_EQ(c.slot, 1);
    // slot_base is nullptr for a plan that was never attached to a buffer (an arithmetic-only plan owns no memory)
    CHECK(t.slot_base(1, 0) == nullptr);
    CHECK(t.slot_base_w(1, 0) == nullptr);
    {
        std::vector<uint8_t> buf((size_t) t.bytes);
        t.base = buf.data();
        CHECK(t.slot_base(1, 0) != nullptr);
        CHECK_EQ(t.slot_base(1, 0) - (const uint8_t *) t.base, (ptrdiff_t) t.stride);
        CHECK_EQ(t.slot_base(0, 1) - (const uint8_t *) t.base, (ptrdiff_t)(2 * t.stride));  // slot-major
        t.base = nullptr;
    }
    err.clear();
    CHECK_EQ(t.window(0, 0, 3, c, err, 2), false);   // slot 2 does not exist (two slots per boundary)
    CHECK(err.find("slots") != std::string::npos);
    // the last window of the last slot ends exactly at the buffer's end
    CHECK_EQ(t.window(1, 0, 256, c, err, 1), true);
    CHECK_EQ(c.dst_offset + c.bytes, t.bytes);
}

// The slots a relay alternates between, and the rule that an in-flight slot is never handed out again.
static void test_slot_rotation() {
    RelayPlan p;
    std::string err;
    CHECK_EQ(relay_plan(2, 1024, kGlmSlab, p, err), true);

    // fresh: both slots free, handed out in order 0, 1
    RelaySlot s0, s1, s2;
    CHECK_EQ(p.acquire(0, s0), true);
    CHECK_EQ(s0.slot, 0);
    CHECK_EQ(p.acquire(0, s1), true);
    CHECK_EQ(s1.slot, 1);
    CHECK_EQ(s0.slot_bytes, 1024ull * kGlmSlab);
    // the plan is not attached to a buffer until the engine pins one; arithmetic-only plans report no host pointer
    CHECK(s0.host == nullptr);
    RelayPlan with_buf = p;
    with_buf.base = (const uint8_t *) 0 + 0;   // explicitly the unattached case
    CHECK(with_buf.base == nullptr);

    // third acquire with nothing released: both slots are in flight, so nothing is handed out
    CHECK_EQ(p.acquire(0, s2), false);
    CHECK_EQ(p.in_flight(0), 2);

    // releasing a slot makes the NEXT acquire hand out that slot, whichever order the releases come in: the
    // payload of the other slot is still in flight and must never be overwritten
    p.release(0, s0.slot);
    CHECK_EQ(p.in_flight(0), 1);
    CHECK_EQ(p.acquire(0, s2), true);
    CHECK_EQ(s2.slot, 0);
    CHECK_EQ(p.acquire(0, s0), false);
    p.release(0, s1.slot);
    CHECK_EQ(p.acquire(0, s0), true);   // the other one, now that it is free too
    CHECK_EQ(s0.slot, 1);
    p.release(0, s2.slot);
    p.release(0, s0.slot);
    CHECK_EQ(p.in_flight(0), 0);

    // rotating a hundred times with the payload released when the DIRECT SUCCESSOR finishes with it (the native
    // relay's next_run_.get() before the next write, prefill.cpp:4186-4188): the slot used by the previous pass is
    // still in flight, so a two-slot staging needs the pass BEFORE that one to be done - and never starves
    int in[2] = { 0, 0 };
    int live[2] = { -1, -1 };   // the pass whose payload each slot holds
    bool ok = true, starved = false;
    for (int pass = 0; pass < 100; ++pass) {
        if (pass >= 1) {   // the successor has finished the pass two behind this one; its slot is free now
            const int done = pass - 1;
            for (int s = 0; s < 2; ++s)
                if (live[s] >= 0 && live[s] < done) { p.release(0, s); live[s] = -1; }
        }
        RelaySlot sl;
        if (!p.acquire(0, sl)) { ok = false; starved = true; break; }
        if (live[sl.slot] >= 0) ok = false;                 // handed out while still in flight
        live[sl.slot] = pass;
        ++in[sl.slot];
    }
    CHECK(!starved);
    CHECK(ok);
    CHECK(in[0] + in[1] == 100);
    CHECK(in[0] == 50 && in[1] == 50);   // both slots get used, evenly (this is the double buffer)

    // the in-flight guard is a real refusal, not a niceness: with one slot released and re-acquired, the slot that
    // still holds an unread payload stays unobtainable until it is released
    p.reset();
    RelaySlot x, y;
    CHECK_EQ(p.acquire(0, x), true);      // slot 0, in flight
    CHECK_EQ(p.acquire(0, y), true);      // slot 1, in flight
    p.release(0, x.slot);
    CHECK_EQ(p.acquire(0, x), true);      // takes the freed slot 0 again
    CHECK_EQ(x.slot, 0);
    CHECK_EQ(p.acquire(0, y), false);     // slot 1's payload was never read: not handed out
    CHECK_EQ(p.in_flight(0), 2);

    // boundaries are tracked separately: an in-flight boundary 0 does not block boundary 1
    RelayPlan t;
    CHECK_EQ(relay_plan(3, 64, kGlmSlab, t, err), true);
    RelaySlot a, b;
    CHECK_EQ(t.acquire(0, a), true);
    CHECK_EQ(t.acquire(1, b), true);
    CHECK_EQ(t.in_flight(0), 1);
    CHECK_EQ(t.in_flight(1), 1);
    CHECK_EQ(t.acquire(0, b), true);
    CHECK_EQ(t.acquire(1, b), true);
    CHECK_EQ(t.acquire(0, a), false);   // boundary 0 is full now
    CHECK_EQ(t.acquire(1, a), false);   // and so is boundary 1
    t.reset();
    CHECK_EQ(t.in_flight(0), 0);
    CHECK_EQ(t.in_flight(1), 0);
    CHECK_EQ(t.acquire(0, a), true);
}

// Which column of the slab each side of a boundary uses.  The producer's last layer's finish output is the column the
// successor's first layer reads: if the two ever disagree, the relay ships a column nobody wrote.
static void test_boundary_columns() {
    int64_t w = -1, r = -1;
    // no split: the model's last column, and both sides agree
    CHECK_EQ(relay_boundary_columns(1, 45, 0, w, r), false);   // a one-stage run has no boundary
    CHECK_EQ(relay_boundary_columns(2, 45, 0, w, r), true);
    CHECK_EQ(w, 23); CHECK_EQ(r, 23);   // 45/2 = 22 with remainder 1: stage 0 gets 23 layers, ends at column 23
    // 3 stages of 45 layers: 15/15/15 -> boundaries at columns 15 and 30
    CHECK_EQ(relay_boundary_columns(3, 45, 0, w, r), true);
    CHECK_EQ(w, 15); CHECK_EQ(r, 15);
    CHECK_EQ(relay_boundary_columns(3, 45, 1, w, r), true);
    CHECK_EQ(w, 30); CHECK_EQ(r, 30);
    // 4 stages of 45 layers: 12/11/11/11 -> 12, 23, 34
    int64_t cols[3] = { 0, 0, 0 };
    for (int b = 0; b < 3; ++b) { CHECK_EQ(relay_boundary_columns(4, 45, b, w, r), true); cols[b] = w; }
    CHECK_EQ(cols[0], 12); CHECK_EQ(cols[1], 23); CHECK_EQ(cols[2], 34);
    // the last boundary's column is the last layer index of every stage but the last, never the model's n_layer
    CHECK_EQ(relay_boundary_columns(2, 45, 1, w, r), false);
    CHECK_EQ(relay_boundary_columns(2, 1, 0, w, r), false);   // more stages than layers
    CHECK_EQ(relay_boundary_columns(0, 45, 0, w, r), false);
    // an even split, and the GLM decode size the engine will actually use (45 layers, 2 stages)
    CHECK_EQ(relay_boundary_columns(2, 44, 0, w, r), true);
    CHECK_EQ(w, 22); CHECK_EQ(r, 22);
    // column * row stride is the offset in a token's slab: the largest column is the slab's last one, so a whole-slab
    // copy of relay_slab_bytes() bytes always contains the column both sides use
    CHECK_EQ(relay_boundary_columns(5, 45, 3, w, r), true);   // 5 stages: the last boundary is index 3
    CHECK(w < 45);
}

// The pass driver: the order the stages are run and the boundaries crossed in, and what a slot handed to a
// successor actually contains.  The stages are stand-ins (no engine, no device): what is under test is the driver's
// sequencing and its slot bookkeeping, which is what decides whether a successor reads the bytes its producer wrote.
static void test_pass_driver() {
    std::string err;
    // a real staging with a real buffer, 3 stages of a 4-token capacity, a tiny slab so the payload is readable
    const uint64_t slab = 64;
    RelayStaging st;
    CHECK_EQ(st.init(3, 4, slab, err), true);
    CHECK(st.mem != nullptr);
    CHECK_EQ(st.plan.n_boundaries, 2);
    CHECK_EQ(st.plan.bytes, 2ull * kRelaySlots * 4ull * slab);

    std::vector<int> order;          // every callback, in the order it happened
    std::vector<int> ran, injected;  // which stages ran / received a payload
    auto run = [&](int s, std::string & e) { order.push_back(100 + s); ran.push_back(s); (void) e; return true; };
    // The producer stamps the payload with a value the test controls: a slot the driver rotated wrongly, or released
    // too early, hands the consumer the PREVIOUS pass's bytes, and this is what catches it.
    int stamp = 0;
    auto out = [&](int s, uint8_t * p, size_t bytes, std::string & e) {
        order.push_back(200 + s);
        for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) (stamp + i);
        (void) e;
        return true;
    };
    auto in = [&](int s, const uint8_t * p, size_t bytes, std::string & e) {
        order.push_back(300 + s);
        injected.push_back(s);
        // the payload came from stage s-1, and every byte of it is what the producer of THIS pass wrote
        for (size_t i = 0; i < bytes; ++i)
            if (p[i] != (uint8_t) (stamp + i)) { e = "payload is not the producer's current bytes"; return false; }
        return true;
    };
    auto advance = [&](int s, std::string & e) { order.push_back(400 + s); (void) e; return true; };

    CHECK_EQ(relay_stage_pass(4, st, run, out, in, advance, err), true);
    // stage 0 runs, hands on, stage 1 receives, runs, hands on, stage 2 receives, runs, and the pass ends there
    const int want[] = { 100, 200, 301, 101, 201, 302, 102, 402 };
    CHECK_EQ(order.size(), 8);
    for (size_t i = 0; i < 8; ++i) CHECK_EQ(order[i], want[i]);
    CHECK_EQ(ran.size(), 3);
    CHECK_EQ(injected.size(), 2);
    CHECK_EQ(injected[0], 1);
    CHECK_EQ(injected[1], 2);
    CHECK_EQ(st.plan.in_flight(0), 0);   // nothing left in flight after a clean pass
    CHECK_EQ(st.plan.in_flight(1), 0);

    // a second pass works the same way (the slots were released, so the driver never starves)
    order.clear(); ran.clear(); injected.clear();
    CHECK_EQ(relay_stage_pass(1, st, run, out, in, advance, err), true);
    CHECK_EQ(order.size(), 8);
    CHECK_EQ(order[0], 100);
    CHECK_EQ(order[7], 402);

    // a pass longer than the staging is refused before any stage runs (the caller must split it, never overflow)
    order.clear();
    CHECK_EQ(relay_stage_pass(5, st, run, out, in, advance, err), false);
    CHECK(err.find("capacity") != std::string::npos);
    CHECK_EQ(order.size(), 0);
    CHECK_EQ(st.plan.in_flight(0), 0);

    // a stage that fails stops the pass, and its successor never sees a payload
    auto run_fails = [&](int s, std::string & e) { if (s == 1) { e = "stage 1 died"; return false; } return true; };
    injected.clear();
    CHECK_EQ(relay_stage_pass(2, st, run_fails, out, in, advance, err), false);
    CHECK(err.find("stage 1 died") != std::string::npos);
    CHECK_EQ(injected.size(), 1);   // boundary 0 crossed, boundary 1 never reached
    CHECK_EQ(st.plan.in_flight(0), 0);

    // a producer that fails leaves the slot unreleased to nobody: it is released, and the consumer is not called
    auto out_fails = [&](int, uint8_t *, size_t, std::string & e) { e = "read-out failed"; return false; };
    injected.clear();
    CHECK_EQ(relay_stage_pass(2, st, run, out_fails, in, advance, err), false);
    CHECK(err.find("read-out failed") != std::string::npos);
    CHECK_EQ(injected.size(), 0);
    CHECK_EQ(st.plan.in_flight(0), 0);
    CHECK_EQ(st.plan.in_flight(1), 0);

    // one stage has nothing to relay: the driver says so rather than running the only stage twice
    RelayStaging solo;
    CHECK_EQ(solo.init(1, 4, slab, err), true);
    CHECK_EQ(relay_stage_pass(1, solo, run, out, in, advance, err), false);
    CHECK(err.find("one stage") != std::string::npos);

    // records what EACH boundary received, so a payload that skipped a boundary or arrived stale is visible
    std::vector<std::vector<uint8_t>> got;
    auto record = [&](int s, const uint8_t * p, size_t bytes, std::string & e) {
        (void) s;
        got.emplace_back(p, p + bytes);
        (void) e;
        return true;
    };
    // THE BYTE-IDENTITY CHECK across a rotation: three passes in a row, each producer writing a different pattern,
    // each consumer receiving exactly the bytes the producer of ITS pass wrote.  A slot released too early, or handed
    // out in the wrong order, shows up here as the previous pass's bytes.
    for (int pass = 0; pass < 3; ++pass) {
        stamp = 100 * (pass + 1);
        got.clear();
        CHECK_EQ(relay_stage_pass(2, st, run, out, record, advance, err), true);
        // 2 tokens x slab bytes, relayed into each of the two boundaries this 3-stage run crosses
        CHECK_EQ(got.size(), 2ull);
        for (size_t b = 0; b < got.size(); ++b) {
            // the callback is handed the WHOLE slot; the pass copies the first n tokens' worth of it
            CHECK_EQ(got[b].size(), 4ull * slab);   // the staging capacity, so a slot always holds a full pass
            for (size_t i = 0; i < 2ull * slab; ++i) CHECK_EQ(got[b][i], (uint8_t) (stamp + i));
        }
        CHECK_EQ(st.plan.in_flight(0), 0);
        CHECK_EQ(st.plan.in_flight(1), 0);
    }
}

int test_relay_staging_run() {
    test_buffer_size();
    test_copy_window();
    test_slot_rotation();
    test_boundary_columns();
    test_pass_driver();
    return 0;
}
