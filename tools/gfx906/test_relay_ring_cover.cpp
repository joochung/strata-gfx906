// P5.3b GATE: THE WINDOW RING IS NOT THE STAGED TOKEN-2 DEFECT, AND THE "RING CARRY" MAKES IT WORSE.
//
// WHY THIS GATE EXISTS.  A staged (layer-split, host-relayed) DeepSeek-V4.1 run gets the whole PROMPT right and the FIRST
// DECODE token wrong, deterministically.  A plausible theory was a ring slot collision: a piece ending at a window edge
// leaves the edge row's slot written by nobody, so the first decode reads a row from NC/S windows back.  The fix that
// theory implies - a "ring carry": the piece that opens a window re-runs the row below as its row 0 (a pass one row wider
// than the tokens it feeds, placed by Ds41Dense::Impl::stage_col0 with carry = true) - was implemented.  This gate is the
// arithmetic that says it must not ship, and it is the red control for the revert.
//
// WHAT THE GRAPHS ACTUALLY DO, transcribed from tools/v41/engine/ds41_dense.cpp:
//
//   upload_pos_tables (~:2146)   per query at position p, per window column j:
//       q = p - SWA + 1 + j
//       i_widx = (q >= pos0) ? RAW + (q - pos0) : q % RAW
//   build_attn (~:848)           wsrc = [ring (RAW rows) | this pass's n rows]; row u of wsrc is ring slot u when u < RAW
//                                and the pass's OWN row u - RAW otherwise.  So a query reads a position FROM ITS OWN PASS
//                                if that pass runs it, and from the RING otherwise.
//   the write-back (~:858)       the pass stores its LAST k = min(n, RAW) rows into ring slots (pos0 + n - k + j) % RAW.
//
// WHAT FOLLOWS, and what is measured below.
//
// (1) THE RING CAN NEVER BE THE PREFILL'S PROBLEM, because of how i_widx is built: a piece that RUNS a position reads it
//     from its own rows, and the rows it does not run are the piece above's - which it reads from the ring, where the piece
//     above's write-back put them.  The only ring rows a staged pre-fill can genuinely lose are the ones NO piece ran, and
//     both tilings run the same set of positions.  The residue this gate reports is the ring's wrap depth, not a hole: with
//     NC/S passes through a slot of RAW slots, a slot written NC/S passes ago holds the same position the reader asks for
//     only when the geometry lets it, which is a property of RAW versus SWA and NOT of the tiling.
//
// (2) THE FIRST DECODE is the one 1-row pass whose whole window comes from the ring, and the row that decides its token is
//     the NEWEST one - the prompt's last position, published into slot (n-1) % RAW by the piece that ENDS on it.  The
//     release's tiling always ends a piece on the prompt's last position, so it never loses it.  The tilt/carry BREAKS
//     THIS CASE BY CASE: a carrying piece feeds one row less than it runs, so whenever the prompt's last row falls inside a
//     tipped piece that piece stops short of it, and that piece's successor - not it - is the pass whose write-back lands on
//     the slot; the newest row is then the row the decode's window is missing.  Measured below, per geometry: the release
//     loses it ZERO times, the carry loses it on a third of prompts.
//
// (3) THE CARRY RE-RUNS ROWS.  A carried row is RUN TWICE, once by each side of a boundary; for a recurrent model that means
//     a state advanced twice.  The release's tiling never does this.
//
// (4) SWA > RAW is a geometry no tiling can cover - the release's own tiling loses decode rows there too, so it is not a
//     staged defect and not what the carry was for.
//
// CONCLUSION, FOR THE RECORD: the ring-carry tiling was REVERTED.  The staged divergence must come from where a staged pass
// and an unsplit pass differ in the ROWS THEY COMPUTE - the relayed state and the position tables that index it (see
// gate-relay-pos-tables.sh, which is the fix that does hold) - not from the tiling of the prompt.
//
// Compiled and run with plain g++: no CUDA, no HIP, no GPU, no model.
//
// Usage: tools/gfx906/gate-relay-ring-cover.sh
#include <cstdio>
#include <algorithm>
#include <vector>
#include <map>

static long long S = 128, SWA = 128, RAW = 128, NC = 512;
struct Pass { long long pos0, n; };   // a pass RUNS positions [pos0, pos0 + n)

static long long room_at(long long slice) {
    return std::max<long long>(1, std::min(S - (slice % S), NC - ((slice / S) % (NC / S)) * S));
}

// THE RELEASE'S TILING: a piece fills the room its window gives it (Impl::pass_room).
static std::vector<Pass> tile_release(long long n, long long pos0) {
    std::vector<Pass> v;
    for (long long c = 0; c < n;) {
        const long long room = std::min(room_at(pos0 + c), n - c);
        v.push_back({pos0 + c, room});
        c += room;
    }
    return v;
}
// THE TILT/CARRY THIS BRANCH TRIED: the piece that opens a window re-runs the row below, feeding one row less.
static std::vector<Pass> tile_carry(long long n, long long pos0) {
    std::vector<Pass> v;
    long long c = 0;
    bool head = false;
    while (c < n) {
        const long long room = room_at(pos0 + c);
        const bool tips = head && room >= 2 && n - c > 1;
        const long long own = tips ? room - 1 : std::min(room, n - c);
        v.push_back({tips ? pos0 + c - 1 : pos0 + c, own + (tips ? 1 : 0)});
        c += own;
        head = ((pos0 + c) % S) == 0;
    }
    return v;
}

// THE RING A TILING LEAVES, exactly as the graph writes it (i_wslot: the pass's LAST min(n, RAW) positions).
static std::map<long long, long long> ring_after(const std::vector<Pass>& v) {
    std::map<long long, long long> ring;
    for (size_t i = 0; i < v.size(); ++i) {
        const long long k = std::min(v[i].n, RAW);
        for (long long j = 0; j < k; ++j) { const long long pos = v[i].pos0 + v[i].n - k + j; ring[pos % RAW] = pos; }
    }
    return ring;
}

// THE ROW THAT DECIDES THE FIRST DECODE.  A decode is a 1-row pass at position last + 1, so its whole window comes from the
// ring, and i_wslot publishes the LAST min(n, RAW) rows of each pass.  The newest window row - the prompt's last position -
// is therefore in the ring if and only if the piece that RUNS it ENDS on it.  That is the predicate this gate measures, and
// it is exactly what the tilt/carry breaks.
static bool newest_row_published(const std::vector<Pass>& v, long long last) {
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i].pos0 <= last && last <= v[i].pos0 + v[i].n - 1)
            return (v[i].pos0 + v[i].n - 1) == last && v[i].n >= 1;
    return false;   // nobody ran it: a worse failure, reported as a loss
}

// Rows run twice: pieces starting at or below where the piece above ended (a recurrent state advanced twice).
static long long rows_run_twice(const std::vector<Pass>& v) {
    long long d = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i].pos0 <= v[i - 1].pos0 + v[i - 1].n - 1) ++d;
    return d;
}

// The ring's coverage of the decode's window, for reporting (NOT a pass criterion: see fact (1)).
static long long decode_misses(const std::map<long long, long long>& ring, long long last) {
    long long m = 0;
    for (long long j = 0; j < SWA - 1; ++j) {
        const long long r = last - SWA + 1 + j;
        if (r < 0) continue;
        std::map<long long, long long>::const_iterator it = ring.find(r % RAW);
        if (it == ring.end() || it->second != r) ++m;
    }
    return m;
}

int main() {
    const long long G[][4] = {{512,128,128,128},{512,128,144,144},{288,128,128,144},
                              {1024,256,256,256},{512,256,128,128},{290,145,144,145},{512,128,128,144},{640,128,128,128}};
    const long long G2[][4] = {{512,128,144,128},{512,128,256,128}};   // SWA > RAW: no tiling can cover

    long long prompts = 0, lost_rel = 0, lost_carry = 0, twice_carry = 0, twice_rel = 0;

    printf("=== SWA <= RAW (the shipped shape): does the FIRST DECODE find its NEWEST window row? ===\n");
    for (int gi = 0; gi < 8; ++gi) {
        NC = G[gi][0]; S = G[gi][1]; SWA = G[gi][2]; RAW = G[gi][3];
        long long lr = 0, lc = 0;
        for (long long pos0 : {0LL, 1LL, 63LL, 127LL, 128LL, 129LL, 300LL, 511LL, 1000LL})
            for (long long n = 1; n <= 1200; ++n) {
                ++prompts;
                const std::vector<Pass> r = tile_release(n, pos0), c = tile_carry(n, pos0);
                const long long last = pos0 + n - 1;
                if (!newest_row_published(r, last)) ++lr;
                if (!newest_row_published(c, last)) ++lc;
                twice_carry += rows_run_twice(c);
                twice_rel   += rows_run_twice(r);
            }
        lost_rel += lr; lost_carry += lc;
        printf("NC=%4lld S=%4lld SWA=%4lld RAW=%4lld : newest decode row NOT published - release %5lld, tilt/carry %5lld (of %5lld prompts)\n",
               NC, S, SWA, RAW, lr, lc, (long long) (9 * 1200));
    }

    printf("\n=== SWA > RAW: the ring cannot hold a full window for ANY tiling (the release loses decode rows too) ===\n");
    for (int gi = 0; gi < 2; ++gi) {
        NC = G2[gi][0]; S = G2[gi][1]; SWA = G2[gi][2]; RAW = G2[gi][3];
        long long hr = 0, hc = 0, tot = 0;
        for (long long pos0 : {0LL, 1LL, 128LL})
            for (long long n = 1; n <= 600; ++n) {
                ++tot;
                const std::vector<Pass> r = tile_release(n, pos0), c = tile_carry(n, pos0);
                if (decode_misses(ring_after(r), pos0 + n - 1)) ++hr;
                if (decode_misses(ring_after(c), pos0 + n - 1)) ++hc;
            }
        printf("  S=%lld SWA=%lld RAW=%lld : %lld of %lld prompts lose a decode row under BOTH tilings - not staged-specific\n",
               S, SWA, RAW, hr, tot);
    }

    printf("\nprompts=%lld\n"
           "  release tiling : newest decode row NOT published %lld times; rows run twice %lld\n"
           "  tilt/carry     : newest decode row NOT published %lld times; rows run twice %lld\n",
           prompts, lost_rel, twice_rel, lost_carry, twice_carry);
    printf("\nVERDICT:\n"
           "  * The release's tiling ALWAYS publishes the prompt's last row into the ring - the row the first decode reads\n"
           "    first - for every prompt length, resume position and geometry (%lld failures in %lld prompts).\n"
           "  * The tilt/carry loses it %lld times, because a carrying piece feeds one row less than it runs and therefore\n"
           "    stops short of the position whose slot the decode reads first.  This is a NEW defect it introduces, on the\n"
           "    single quantity that decides the first decode token - which is the token the witnesses saw change.\n"
           "  * The carry also re-runs %lld rows, advancing a recurrent state twice.\n"
           "  So the ring is NOT the staged token-2 defect and the tiling is NOT where the fix belongs.  The fix that holds\n"
           "  is the position-table refresh in relay_write_in (gate-relay-pos-tables.sh).\n",
           lost_rel, prompts, lost_carry, twice_carry);

    const bool ok_release = (lost_rel == 0 && twice_rel == 0);
    const bool ok_carry   = (lost_carry > 0 && twice_carry > 0);
    if (!ok_release) printf("GATE FAIL: the release's tiling is not clean (newest row lost %lld, rows run twice %lld)\n", lost_rel, twice_rel);
    if (!ok_carry)   printf("GATE FAIL: the tilt/carry did not reproduce as worse (newest row lost %lld, rows run twice %lld)\n", lost_carry, twice_carry);
    return (ok_release && ok_carry) ? 0 : 1;
}
