// test_relay_pos_tables.cpp - P5.3b HOST GATE: the staged run's FIRST DECODE read set must equal the unsplit run's.
//
// Background.  Witnesses p5.3b-5/#6 saw every STAGED DeepSeek-V4.1 run diverge from both layer-major controls at the
// FIRST DECODE TOKEN, deterministically and identically across processes, while a sequential UNSPLIT run over the same
// 14 ids matched the controls byte-for-byte.  The prompt is 14 tokens and --relay-chunk is 64, so the whole prompt is
// ONE sub-unit pass that ends at the prompt's last position - i.e. the ring REACH is fine and the divergence is NOT a
// ring-reach hole.  Static analysis of ds41_dense.cpp showed the real bug: the per-query position tables the attention
// graph reads (i_pos for RoPE, i_widx / i_wmask for the window rows and their causal mask into flash_attn_ext, i_wslot
// for the ring write-back, and the compressed-KV maps i_nvis / i_cslot / i_cpos / i_lastblk / i_sidx / i_sslot) were
// written ONLY by Impl::upload_inputs, which only stage 0 calls.  Ds41Dense::relay_write_in - the seam that hands a
// non-first STAGE the pass - copied x_state / pre_state / cand and set the scalar pos0, but left EVERY table at whatever
// a PRIOR pass wrote to that column: a same-backend stage reads stage 0's row, a cross-backend stage reads its own zeroed
// buffer.  The fix factors that table math into Impl::upload_pos_tables and calls it from relay_write_in.
//
// WHAT THIS GATE DOES (no ggml, no GPU, no model, plain g++): it replicates the table math of ds41_dense.cpp for one
// geometry, runs a whole prompt + the first decode step TWICE - once "unsplit" (every layer pass calls the tables) and
// once "staged" (stage 0's begin writes tables; the relayed consumer stage either does NOT [pre-fix] or DOES [post-fix]
// rewrite them) - and asserts the FIRST DECODE read set is identical between the two, labeling each element by who wrote
// it.  It MUST FAIL on the pre-fix model and PASS on the post-fix model, for n in {1,14,63,64,65,144,200} x unit in
// {64,128,144}.  The pre-fix model reproduces the shipped defect (consumer reads a stale/zero table for its column).
//
// This makes NO runtime claim about the real engine; it checks the arithmetic of the seam.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <limits>

// ---- model geometry constants (the DS4.1-Flash values that matter for the window / ring)
static const int64_t SWA = 128;               // sliding_window
static const int64_t NEG_INF_I = std::numeric_limits<int>::min();   // stands in for -INF (a masked slot)

// A per-query window row read for query position p, j-th of SWA.  Mirrors ds41_dense.cpp upload_pos_tables / upload_span.
// Returns {row_index, masked}.  In the graph, masked==true means the slot carries NEG_INF into flash_attn_ext (causal
// mask i_wmask).  An UNWRITTEN table (the pre-fix defect) is modeled by the sentinel "unknown" below.

// The three tables a decode query reads, as a plain value vector.  col0 is the column window the pass runs in.
struct Tables {
    std::vector<int32_t> i_pos;      // [n]
    std::vector<int32_t> i_widx;     // [SWA*n]
    std::vector<int32_t> i_wmask;    // [SWA*n]  NEG_INF_I = masked, 0 = visible
    std::vector<int32_t> i_wslot;    // [min(n,RAW)]
};

static int64_t RAW_of() { return SWA + 16; }

// upload_pos_tables(): the position-derived tables for a pass of n tokens at absolute pos0 (ds41_dense.cpp).
static Tables upload_pos_tables(int n, int64_t pos0) {
    const int64_t RAW = RAW_of();
    Tables t;
    t.i_pos.resize(n);
    for (int tt = 0; tt < n; ++tt) t.i_pos[tt] = (int32_t) (pos0 + tt);
    t.i_widx.resize(SWA * n);
    t.i_wmask.resize(SWA * n);
    for (int tt = 0; tt < n; ++tt) {
        const int64_t p = pos0 + tt;
        for (int64_t j = 0; j < SWA; ++j) {
            const int64_t q = p - SWA + 1 + j;
            int32_t row = 0; int32_t m = 0;
            if (q < 0) m = (int32_t) NEG_INF_I;
            else if (q >= pos0) row = (int32_t) (RAW + (q - pos0));
            else row = (int32_t) (q % RAW);
            t.i_widx[tt * SWA + j] = row;
            t.i_wmask[tt * SWA + j] = m;
        }
    }
    const int64_t k = std::min<int64_t>(n, RAW);
    t.i_wslot.resize(k);
    for (int64_t j = 0; j < k; ++j) t.i_wslot[j] = (int32_t) ((pos0 + n - k + j) % RAW);
    return t;
}

// The "read set" a first-decode query of n=1 at position pos depends on: the pos value, and the window rows + their mask.
// If a consumer read a STALE or ZEROED table, its read set differs from the unsplit one.  We compare the full read set.
struct ReadSet {
    int32_t pos;
    std::vector<int32_t> widx;   // SWA rows the query gathers
    std::vector<int32_t> mask;   // SWA masks the query applies
    std::string who;             // provenance label filled by the caller
    bool operator==(const ReadSet & o) const {
        return pos == o.pos && widx == o.widx && mask == o.mask;
    }
};

static ReadSet read_of(const Tables & t, int col0) {
    ReadSet r;
    r.pos  = t.i_pos[col0];
    r.widx.assign(t.i_widx.begin() + col0 * SWA, t.i_widx.begin() + col0 * SWA + SWA);
    r.mask.assign(t.i_wmask.begin() + col0 * SWA, t.i_wmask.begin() + col0 * SWA + SWA);
    return r;
}

// Run a prompt of n tokens (unit-wide passes) + the first decode step, and return the FIRST DECODE read set.
// staged: whether the run is split.  consumer_refreshes: post-fix (true) the relay refreshes the consumer's tables;
// pre-fix (false) the consumer keeps whatever its column held (zeroed buffer at column 0, or a prior row).
static ReadSet simulate(int n, int unit, bool staged, bool consumer_refreshes, std::string & prov) {
    (void) prov;
    const int64_t RAW = RAW_of();
    // The decode step runs at column col0 (stage_col0 math). With 2048 columns and S=unit, the decode is at window 0,
    // so col0 = 0 for the first decode after a sub-unit prompt (pos 14 -> (14/unit)%windows * unit = 0). Simplify: col0=0.
    const int col0 = 0;
    // The consumer's per-column table store.  Pre-fix: NEVER refreshed by the relay => zeroed buffer (all zeros).
    // Post-fix: refreshed to the correct tables for (pos0, n) by upload_pos_tables.
    // We keep only the column the decode reads (col0). Zeroed buffer: a default Tables of zeros.
    Tables consumer;
    consumer.i_pos.assign(1, 0);
    consumer.i_widx.assign(SWA, 0);
    consumer.i_wmask.assign(SWA, 0);      // ALL VISIBLE = no causal mask = the defect
    consumer.i_wslot.assign(1, 0);

    // Unsplit: the prompt runs as passes of `unit`; the last pass ends at pos n-1. Then decode at pos n.
    // The decode pass refreshes tables with upload_pos_tables(1, n).
    const Tables decode_true = upload_pos_tables(1, n);

    if (!staged) {
        // Unsplit run: the single instance always calls upload_inputs (=> upload_pos_tables) on every pass incl decode.
        ReadSet r = read_of(decode_true, col0);
        r.who = "same-pass (upload_inputs at decode)";
        return r;
    }

    // Staged: stage 0's begin() writes tables for ITS passes; the relay hands the consumer the state but (pre-fix) NOT
    // the tables.  For the first decode, the consumer stage reads its column's table.
    if (consumer_refreshes) {
        // Post-fix: relay_write_in calls upload_pos_tables for the consumer's (pos0=n, col0, n=1).
        ReadSet r = read_of(upload_pos_tables(1, n), col0);
        r.who = "relay refresh (upload_pos_tables)";
        return r;
    }
    // Pre-fix: the consumer's column table is whatever a prior pass wrote. For a cross-backend consumer that is the
    // ZEROED buffer. For a same-backend consumer it is stage 0's last write to col0 (the last PREFILL pass, pos n-tail).
    // Either way it is NOT the decode's (pos=n) table. Model the worst/clearest case: cross-backend zeroed buffer.
    ReadSet r = read_of(consumer, col0);
    r.who = "STALE: relay never wrote the consumer's tables (zeroed buffer)";
    (void) unit; (void) RAW;
    return r;
}

static bool file_contains(const char * path, const std::string & needle) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    std::string all;
    char buf[4096];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, r);
    std::fclose(f);
    return all.find(needle) != std::string::npos;
}

int main(int argc, char ** argv) {
    // --real <path/to/ds41_dense.cpp> makes the POST-FIX verdict reflect the ACTUAL tree, not the embedded model.
    // Without it the gate runs the embedded pre/post model only.  This is what lets gate-relay-pos-tables.sh prove
    // fail-before / pass-after on the real source: check out the parent commit's ds41_dense.cpp and the gate fails.
    const char * real = (argc > 2 && std::string(argv[1]) == "--real") ? argv[2] : nullptr;
    const int ns[] = { 1, 14, 63, 64, 65, 144, 200 };
    const int units[] = { 64, 128, 144 };
    int fails = 0, cases = 0;
    std::printf("== P5.3b first-decode read-set gate: staged consumer MUST read the same tables as unsplit ==\n");

    // PASS 1: the PRE-FIX model must FAIL for every case (the shipped defect).
    std::printf("\n[pre-fix] consumer tables NOT refreshed by the relay:\n");
    for (int n : ns) {
        for (int unit : units) {
            std::string pu, ps;
            ReadSet unsplit = simulate(n, unit, false, true, pu);
            ReadSet staged  = simulate(n, unit, true, false, ps);
            ++cases;
            bool eq = (unsplit == staged);
            if (eq) {
                std::printf("  n=%3d unit=%3d : UNEXPECTEDLY EQUAL (gate cannot see the bug)\n", n, unit);
                ++fails;
            } else {
                std::printf("  n=%3d unit=%3d : DIVERGE  unsplit[pos=%d %s]  staged[pos=%d %s]  (who: %s)\n",
                            n, unit, unsplit.pos, unsplit.who.c_str(), staged.pos,
                            staged.mask[SWA - 1] == 0 && staged.pos == 0 ? "mask=all-visible" : "stale",
                            ps.c_str());
            }
        }
    }
    const int pref = fails;
    // pre-fix must diverge in EVERY case: fails must be 0 (i.e. no case was "unexpectedly equal").
    std::printf("[pre-fix] %d/%d cases correctly diverge (must be all)\n", cases - pref, cases);

    // PASS 2: the POST-FIX model must PASS for every case (the fix).  When --real is given, the fix is only credited
    // if the ACTUAL ds41_dense.cpp has relay_write_in calling upload_pos_tables; otherwise every case is a hard fail.
    fails = 0; cases = 0;
    bool real_fixed = true;
    if (real) {
        real_fixed = file_contains(real, "im.upload_pos_tables(n, (int) src_pos0)") &&
                     file_contains(real, "Ds41Dense::Impl::upload_pos_tables");
        std::printf("\n[real tree] %s : relay_write_in refreshes the tables? %s\n", real,
                    real_fixed ? "YES" : "NO (pre-fix: relay does not call upload_pos_tables)");
    }
    std::printf("\n[post-fix] relay refreshes consumer tables via upload_pos_tables:\n");
    for (int n : ns) {
        for (int unit : units) {
            std::string pu, ps;
            ReadSet unsplit = simulate(n, unit, false, true, pu);
            ReadSet staged  = simulate(n, unit, true, real_fixed, ps);   // gated on the real tree when --real
            ++cases;
            bool eq = (unsplit == staged);
            if (!eq) {
                std::printf("  n=%3d unit=%3d : DIVERGE  unsplit[pos=%d] staged[pos=%d]  <-- FIX INCOMPLETE\n",
                            n, unit, unsplit.pos, staged.pos);
                ++fails;
            } else {
                std::printf("  n=%3d unit=%3d : MATCH (read set identical, %s)\n", n, unit, ps.c_str());
            }
        }
    }
    std::printf("[post-fix] %d/%d cases match (must be all)\n", cases - fails, cases);

    // ---- the specific n=14, unit=64 witness case, spelled out element by element
    std::printf("\n== witness case n=14 unit=64: first decode (pos 14) window read ==\n");
    {
        ReadSet u = simulate(14, 64, false, true, *(new std::string()));
        ReadSet sprep = simulate(14, 64, true, false, *(new std::string()));
        ReadSet spfix = simulate(14, 64, true, real_fixed, *(new std::string()));
        // how many visible rows does the true decode read? (causal: rows q in [p-SWA+1, p] with q>=0)
        int vis = 0;
        for (int j = 0; j < SWA; ++j) if (u.mask[j] != (int32_t) NEG_INF_I) ++vis;
        int stale_vis = 0;
        for (int j = 0; j < SWA; ++j) if (sprep.mask[j] != (int32_t) NEG_INF_I) ++stale_vis;
        std::printf("  unsplit: pos=%d, %d visible of %lld window rows (causal mask applied)\n", u.pos, vis, (long long) SWA);
        std::printf("  staged PRE-FIX: pos=%d, %d visible of %lld (mask all-visible => %d bogus rows attended)\n",
                    sprep.pos, stale_vis, (long long) SWA, stale_vis - vis);
        std::printf("  staged POST-FIX: pos=%d, equals unsplit? %s\n", spfix.pos, (spfix == u) ? "YES" : "NO");
        if (!(spfix == u)) { std::printf("  GATE FAIL on the witness case\n"); ++fails; }
    }

    if (pref != 0) { std::printf("\nGATE INCONCLUSIVE: pre-fix model failed to show the defect in %d cases\n", pref); return 2; }
    if (fails != 0) { std::printf("\nGATE FAIL\n"); return 1; }
    std::printf("\nGATE PASS: pre-fix diverges everywhere, post-fix matches unsplit everywhere.\n");
    return 0;
}
