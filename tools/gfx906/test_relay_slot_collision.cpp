// P5.3b gate (2): the ring write-back table is ONE tensor per instance, written at byte 0, sized by the LAST pass that ran.
//
// Where it comes from.  The window ring is the instance's RAW = swa + 16 rows of sliding-window history; every graph reads
// it and writes it back through one tensor, i_wslot (and the ratio-2 state ring through i_sslot):
//
//   ggml_tensor * i_wslot;          // I32 [min(NC, RAW)]  ring slots of the pass's last rows   (ds41_dense.cpp:297)
//   im.i_wslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));            (:1180)
//   ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);                            (:1985)
//   ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0))                                       (:736 / :1003)
//
// - so the table is per instance (not per column: unlike i_widx / i_wmask, which the graphs read at a column offset
// b.v1(..., n, col0 * stride), i_wslot is read at byte 0 by every graph), it is allocated at the RING's width, and
// upload_pos_tables() fills only min(n, RAW) entries of it with the ring slots of the pass's OWN last positions.  So the
// live table describes the LAST pass that ran, and a pass only ever ADDRESSES min(n, RAW) entries.  Which means:
//
//   * UNSPLIT, the first pass is the whole prompt: the table is min(prompt, RAW) wide, and a decode step overwrites entry 0
//     of a table whose remaining entries still describe rows the run has fed.  Nothing addresses past one entry, so the run
//     never notices.
//   * STAGED, passes are --relay-chunk (64) rows and then ONE row.  The narrowest pass leaves the table NARROW - and on the
//     geometries this gate can model (a prompt below the ring's width, so the run's own passes are the only writers) every
//     pass addresses only entries an earlier pass of the SAME run wrote.  The gate verifies that, for every geometry under
//     the ring, and fails if it is ever false.
//
// WHAT THIS GATE ACTUALLY GATES, and what it REFUTES.  It was written to prove a second root cause for the staged
// first-decode divergence: that a staged pass writes rows back through slots an OLDER, wider pass published, corrupting
// ring rows a live query reads.  It does not prove that.  On every modelled geometry - including the witness (14-token
// prompt, --relay-chunk 64, the first decode row) - a staged run's addresses stay inside what its own earlier passes wrote,
// an unsplit control's do too, and the two shapes make the SAME ring: the same rows, in the same slots.  So the "narrow
// table" theory, like the ring-reach theory before it (90230297, reverted by eb1fed5e), does NOT explain the witness, and
// the engine change that was built on it is NOT the fix and has been reverted.
//
// The gate is kept as a RED CONTROL, in the same style as gate-relay-ring-cover.sh.  It pins the arithmetic - the published
// width, the range a pass addresses, and the ring the two shapes leave behind - so that any future change to the tiling or
// to upload_pos_tables that WOULD introduce a write-back through an unpublished entry, or a staged ring that differs from
// the unsplit ring row for row, says so with numbers instead of showing up as a wrong token 455.
//
// The two rules the reverted fix proposed are still exercised, for COST: the model runs a "fixed" shape (published width may
// not shrink; a pass may be preceded by a re-feed) alongside the two real ones and the gate asserts the re-feed costs ZERO
// rows wherever a pass is no wider than what the run already published - true of every decode step, and of the witness's
// whole run.  If a future change makes the re-feed owed work somewhere, that is a real change in the arithmetic, and the
// count says where.
//
// LIMITATION, stated so nobody mistakes this for a device test: it is a host model of the table's arithmetic (no ggml, no
// GPU, no model).  It covers prompts and passes NARROWER than the ring - the geometry the witness runs in - and says
// nothing about a pass WIDER than the ring, where a run's own first pass leaves ring rows unwritten and the model would
// need the full i_widx mapping to say anything meaningful.  Those cases are skipped and counted.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static const int64_t SWA = 128;             // the sliding window the graphs read
static const int64_t RAW = SWA + 16;        // the ring's rows (ds41_dense.cpp:1333)

// The live table: the slots upload_pos_tables published, and how many entries the LAST pass wrote.
struct Tail {
    std::vector<int32_t> wslot = std::vector<int32_t>(RAW, -1);
    int64_t len = 0;                        // entries the live table HAS (what the last pass wrote)
};

struct State {
    std::vector<int64_t> row = std::vector<int64_t>(RAW, -1);   // ring row -> position it holds (-1 = never written)
    Tail tail;
};

struct Run {
    int stale_writes = 0;                  // write-backs through an entry no pass in this run ever published
    int refeeds = 0, refeed_rows_total = 0;
    int64_t max_refeed = 0;
    bool over_ring = false;                // a pass wider than the ring: outside this gate (see LIMITATION)
    std::vector<int64_t> ring;             // what the run leaves the ring: position per slot (-1 = never written)
};

// upload_pos_tables(): publish the ring slots of this pass's own last min(n, RAW) positions.  'fixed_width' = the fix's
// rule (a): the published width may only grow, never shrink to a narrower pass's own.
static void publish(Tail & t, int64_t pos0, int64_t n, bool fixed_width) {
    const int64_t k = std::min<int64_t>(n, RAW);
    for (int64_t j = 0; j < k; ++j) t.wslot[j] = (int32_t) ((pos0 + n - k + j) % RAW);
    t.len = fixed_width ? std::max(t.len, k) : k;
}

// The rule Ds41Dense::stage_refeed_rows implements: how many rows a pass of n tokens must be PRECEDED by so the live table
// is at least as wide as the pass's own write-back.  Zero whenever the run's widest pass so far already published that
// many - which is why an unsplit run never pays it, and why on the witness geometry (widest pass 64, a decode's write-back
// is 1 entry) the refresh is zero rows.  Pure arithmetic on the geometry, so the host gate and the engine ask the same
// question and cannot drift apart.
static int64_t stage_refeed_rows(int64_t widest_so_far, int64_t n) {
    return std::max<int64_t>(0, std::min<int64_t>(n, RAW) - std::min(widest_so_far, RAW));
}

// A pass's write-back: the graph's LAST k = min(n, RAW) rows, to the slots of b.v1(im.i_wslot, k, 0).  A ggml view's extent
// does not clip a ggml_get_rows / ggml_set_rows index (the file's own note at :749), so what a graph addresses is the
// TENSOR's range; what it finds there is the last publisher's.  An address past everything the run published is a stale
// write: in a real run it lands on whatever a previous prompt or a zeroed allocation left, in another position's ring row.
static void write_pass(State & s, Run & r, int64_t pos0, int64_t n, bool fixed_width, int64_t widest) {
    const int64_t k = std::min<int64_t>(n, RAW);
    const int64_t live = std::min(fixed_width ? std::max(s.tail.len, std::min(widest, RAW))
                                              : s.tail.len, RAW);
    publish(s.tail, pos0, n, fixed_width);
    for (int64_t j = 0; j < k; ++j) {
        const int64_t pos = pos0 + n - k + j;
        if (j >= live) { ++r.stale_writes; continue; }   // an entry no pass in this run ever wrote
        s.row[(size_t) s.tail.wslot[j]] = pos;
    }
}

/// The run.  'staged' walks the prompt in windows of 'window' and decodes one row at a time; 'refeed' turns rule (b)
/// on; 'fixed_width' turns rule (a) on.  A run's FIRST pass is modelled as self-consistent (its own upload published its own
/// tables), the same assumption both shapes run on.  The result carries the ring the run leaves behind, which is the thing
/// the two shapes have to agree on: the engine's reads are BY POSITION (i_widx, ds41_dense.cpp:1986: a row below a pass's
/// first position is read from slot q % RAW), so a ring row that holds position q is a correct row for every query that
/// reaches q - regardless of which pass put it there.
static Run run(int64_t prompt, int64_t window, int64_t ndec, bool staged, bool refeed, bool fixed_width) {
    Run r;
    State s;
    int64_t widest = 0, fed = 0;            // widest pass fed so far; highest position computed so far (the run's knowledge)
    const auto pass = [&](int64_t pos0, int64_t n, bool first) {
        const int64_t k = std::min<int64_t>(n, RAW);
        if (first) {
            for (int64_t j = 0; j < k; ++j) {
                const int64_t pos = pos0 + n - k + j;
                s.row[(size_t) (pos % RAW)] = pos;               // self-consistent: the table is its own
            }
            publish(s.tail, pos0, n, fixed_width);
        } else {
            write_pass(s, r, pos0, n, fixed_width, widest);
        }
        widest = std::max(widest, n);
        fed = std::max(fed, pos0 + n);
        if (n > RAW) r.over_ring = true;                         // outside the gate (see LIMITATION)
    };

    if (!staged) {
        // The unsplit reference: ONE pass over the whole prompt, then one row per decode step.
        pass(0, prompt, true);
        for (int64_t d = 0; d < ndec; ++d) pass(prompt + d, 1, false);
        r.ring = s.row;
        return r;
    }
    for (int64_t c0 = 0; c0 < prompt; ) {
        const int64_t m = std::min<int64_t>(window, prompt - c0);
        const int64_t rf = (refeed && fed > 0) ? std::min(stage_refeed_rows(widest, m), std::min(fed, c0)) : 0;
        if (rf > 0) {
            pass(c0 - rf, rf, false);
            ++r.refeeds; r.refeed_rows_total += rf; r.max_refeed = std::max(r.max_refeed, rf);
        }
        pass(c0, m, fed == 0);
        c0 += m;
    }
    for (int64_t d = 0; d < ndec; ++d) {
        const int64_t p = prompt + d;
        const int64_t rf = (refeed && fed > 0) ? std::min(stage_refeed_rows(widest, 1), fed) : 0;
        if (rf > 0) {
            pass(p - rf, rf, false);
            ++r.refeeds; r.refeed_rows_total += rf; r.max_refeed = std::max(r.max_refeed, rf);
        }
        pass(p, 1, false);
    }
    r.ring = s.row;
    return r;
}

// The witness geometry: the 14-token DS4.1 prompt, --relay-chunk 64, and the first decode row whose logits diverged.
// Pinned as a red control: whatever a future change does to the tiling or to upload_pos_tables, this geometry must keep
// (i) no write-back through an unpublished entry, and (ii) a staged ring identical to the unsplit ring, row for row.
static void check_witness(int & fails) {
    const int64_t prompt = 14, window = 64;
    Run staged = run(prompt, window, 1, true, false, false);
    Run ref = run(prompt, window, 1, false, false, true);
    for (const auto & t : { std::pair<const char *, const Run &>("staged", staged), { "unsplit", ref } }) {
        if (t.second.stale_writes != 0) {
            std::printf("FAIL witness (%s): %d write-back(s) through an entry no pass published\n",
                        t.first, t.second.stale_writes);
            ++fails;
        }
    }
    if (staged.ring != ref.ring) {
        int diff = 0;
        for (size_t q = 0; q < staged.ring.size(); ++q) if (staged.ring[q] != ref.ring[q]) ++diff;
        std::printf("FAIL witness: the staged run's ring differs from the unsplit run's at %d slot(s) - the narrow-table "
                    "difference this gate refuted has reappeared, or the model has drifted\n", diff);
        ++fails;
    }
    // The theory this gate was written for predicted damage HERE, at the geometry whose first decode token diverged.  It is
    // absent, and the gate says which way it came out so nobody re-reads it as a confirmation.
    std::printf("witness window %lld prompt %lld: staged %d stale write-back(s), unsplit %d, rings %s "
                "[narrow-table theory NOT reproduced]\n",
                (long long) window, (long long) prompt, staged.stale_writes, ref.stale_writes,
                staged.ring == ref.ring ? "identical" : "DIFFERENT");
}

// --------------------------------------------------------------------------------------- the engine's side, string-gated
// A host model can be self-consistent and still describe an engine that has moved on.  The gate therefore refuses to pass
// unless the ds41_dense.cpp it is reading still has the arithmetic it models - the same trick gate-relay-pos-tables.sh
// uses (it takes --real <path> and greps).  Each predicate below is the ONE line the model stands in for; if a line moves
// or changes shape, the gate says which assumption went stale instead of quietly proving a fiction.
static bool engine_has(const char * needle, const char * file, const char * what, int & hits) {
    FILE * f = fopen(file, "r");
    if (!f) { std::printf("GATE: cannot read %s\n", file); return false; }
    std::vector<char> buf(1 << 22);
    size_t n = fread(buf.data(), 1, buf.size() - 1, f);
    fclose(f);
    buf[n] = 0;
    // Ignore whitespace differences: the engine reflows these comments, the arithmetic is what matters.
    std::string hay, s(needle);
    for (size_t i = 0; i < n; ++i) if (!isspace((unsigned char) buf[i])) hay += (char) tolower(buf[i]);
    std::string w;
    for (size_t i = 0; i < s.size(); ++i) if (!isspace((unsigned char) s[i])) w += (char) tolower(s[i]);
    for (size_t at = hay.find(w); at != std::string::npos; at = hay.find(w, at + 1)) ++hits;
    if (hits == 0) {
        std::printf("FAIL engine drift: %s no longer says\n  %s\n", what, needle);
        return false;
    }
    return true;
}

// The engine-side checks, run over the same argv the C++ test gets.  Returns the number of failures to add.
static int engine_checks(int argc, char ** argv) {
    const char * file = "tools/v41/engine/ds41_dense.cpp";
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], "--real")) file = argv[i + 1];
    int f = 0, hits = 0;
    auto has = [&](const char * n, const char * w) { hits = 0; f += !engine_has(n, file, w, hits); };
    // The ring is swa + 16 rows, and the ring is addressed by position: the two facts the model's slots rest on.
    has("im.RAW = im.SWA + 16;",  "the ring's width");
    has("else row = (int32_t) (q % RAW);",  "the ring read by position");
    // The table is ONE tensor per instance, allocated at the ring's width, and written at byte 0 by every pass - the whole
    // reason a staged pass inherits a table sized by someone else.
    has("// I32 [min(NC, RAW)]  ring slots of the pass's last rows", "i_wslot is one table for the instance");
    // upload_pos_tables publishes min(n, RAW) slots of its own pass - the publish rule the model's publish() restates.
    has("const int64_t k = std::min<int64_t>(n, RAW);", "the publish width");
    has("ws[(size_t) j] = (int32_t) ((pos0 + n - k + j) % RAW);", "the published slot");
    has("ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);",
        "the table write, at byte 0 and this pass's width only");
    // The graph's write-back: the LAST min(n, RAW) rows, through the table, via a VIEW whose extent does not clip the index.
    has("ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0))", "the ring write-back, through a view of the table");
    // The table's allocation is the ring's width, not a pass's: this is why a pass can ADDRESS an entry nobody wrote.
    has("ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW))", "i_wslot's allocation");
    // The mask that keeps an unwritten row out of the logits: without it, a missing ring row would be a WRONG one.
    has("if (q < 0) m = NEG_INF;",  "the mask on rows the run does not have");
    // The relay seam refreshes the tables (bb4c6871): the fix the gate's geometry now sits on top of.
    has("if (!im.upload_pos_tables(n, (int) src_pos0)) return false;", 
                     "relay_write_in refreshing the position tables");
    return f;
}

int main(int argc, char ** argv) {
    int fails = 0, cases = 0, skipped = 0, stale_any = 0;
    // Every window at or below the ring's width; RAW itself is the boundary case where a pass covers the ring completely.
    const int64_t windows[] = {1, 2, 14, 16, 32, 63, 64, 65, 127, 128, 129, 143, 144};
    // Prompts BELOW the ring's width, so a run's own passes are the only writers the model has to reason about.  A prompt
    // at or over RAW is the chunk-path geometry a real run keeps above the ring and this gate cannot model.
    const int64_t prompts[] = {1, 2, 14, 63, 64, 65, 127, 128, 143};
    const int64_t decodes[] = {1, 8};

    for (int64_t window : windows) {
        for (int64_t prompt : prompts) {
            for (int64_t ndec : decodes) {
                if (prompt >= RAW) { ++skipped; continue; }
                Run ref = run(prompt, window, ndec, false, false, true);
                if (ref.over_ring) { ++skipped; continue; }              // outside the gate (see LIMITATION)
                ++cases;
                Run staged = run(prompt, window, ndec, true, false, false);
                Run fixed = run(prompt, window, ndec, true, true, true);

                // (1) A write-back may not address an entry the run never published.
                if (staged.stale_writes != 0) {
                    ++stale_any;
                    std::printf("FAIL window %lld prompt %lld decode %lld: the staged run writes back through %d "
                                "unpublished entr(y/ies)\n",
                                (long long) window, (long long) prompt, (long long) ndec, staged.stale_writes);
                    ++fails;
                }
                if (fixed.stale_writes != 0) {
                    std::printf("FAIL window %lld prompt %lld decode %lld: the fixed staged run does %d stale write-back(s)\n",
                                (long long) window, (long long) prompt, (long long) ndec, fixed.stale_writes);
                    ++fails;
                }
                // (2) THE property: the staged run's ring must equal the unsplit run's, slot for slot.  Reads are by
                // position, so an identical ring means every query of both shapes reads the same rows.
                if (staged.ring != ref.ring) {
                    int diff = 0;
                    for (size_t q = 0; q < staged.ring.size(); ++q) if (staged.ring[q] != ref.ring[q]) ++diff;
                    std::printf("FAIL window %lld prompt %lld decode %lld: the staged ring differs from the unsplit ring at "
                                "%d slot(s)\n",
                                (long long) window, (long long) prompt, (long long) ndec, diff);
                    ++fails;
                }
                                // (3) The fix's rules may not COST anything where the ring's own arithmetic already covers the run: a pass
                // no wider than everything the run published before it is owed no re-feed.
                const int64_t owed = stage_refeed_rows(std::min(prompt, window), std::min(window, prompt));
                if (owed == 0 && fixed.refeeds != 0) {
                    std::printf("FAIL window %lld prompt %lld decode %lld: the run's widest pass already covers every pass's "
                                "write-back, yet the fix re-feeds %d row(s)\n",
                                (long long) window, (long long) prompt, (long long) ndec, fixed.refeed_rows_total);
                    ++fails;
                }
                if (fixed.max_refeed > RAW) {
                    std::printf("FAIL window %lld prompt %lld decode %lld: a re-feed of %lld rows exceeds the ring's %lld\n",
                                (long long) window, (long long) prompt, (long long) ndec,
                                (long long) fixed.max_refeed, (long long) RAW);
                    ++fails;
                }
            }
        }
    }

    check_witness(fails);
    fails += engine_checks(argc, argv);
    std::printf("test_relay_slot_collision: %d case(s) (prompt < the ring's %lld), %d skipped (a pass at or over the ring), "
                "%d case(s) with a stale write-back, %d FAILURE(S)\n", cases, (long long) RAW, skipped, stale_any, fails);
    return fails ? 1 : 0;
}

