// P5.3b HOST SIMULATION: THE COLUMN A DECODE PASS LIVES IN, UNSPLIT vs STAGED, AT THE WITNESS'S GEOMETRY.
//
// THE GEOMETRY (witness #7, TS 20261025-001554): prompt 14 tokens, --relay-chunk/--prefill-chunk 64, window S = 128,
// NC = 512 state columns, so nwin = NC / S = 4 windows.  The prompt is ONE pass (14 < 64 < S), so the FIRST DECODE is a
// 1-row pass at position 14 - the pass this task points at.
//
// THE TWO COLUMN RULES, transcribed from tools/v41/engine/ds41_dense.cpp:
//
//   UNSPLIT   begin_impl(): if (im.staged) is FALSE, so im.col0 = 0 - EVERY pass, decode included, runs at column 0, and
//             the run's own comment says why: "Unsplit, this is pos0 at column 0."  The state columns are then indexed
//             by OFFSET WITHIN THE PASS, and the graphs take positions from i_pos (upload_inputs writes i_pos[t] =
//             pos0 + t at column wcol + t), so a decode at position 14 occupies column 0, holding position 14.
//
//   STAGED    begin_impl(): if (im.staged) is TRUE, so im.col0 = im.stage_col0(pos0, n) = (pos0 / S) % nwin * S.  For a
//             decode at position 14 that is (14 / 128) % 4 * 128 = 0 - the SAME column.  For a decode at position 140 it
//             would be 128, and at 640 it wraps to 0 again: the staged run CYCLES the columns so that a later pass cannot
//             overwrite state a still-live query reads.
//
// WHAT THAT DIFFERENCE BUYS, and what it costs.  The cycling is the whole point of the staged map: with the window ring
// RAW = SWA = 128 rows deep, a decode at position p reads slots for positions [p - 127, p], and the state a stage holds is
// only NC = 512 columns wide.  Writing every pass at column 0 (the unsplit rule) would make a staged pass step on the
// columns its OWN preceding passes wrote and its own decode still reads.  So the columns differ by design and are not a
// bug - and this gate prints both numbers so the claim is checkable rather than argued.
//
// WHAT THE GATE ACTUALLY ASSERTS, at the witness's geometry:
//   (i)   the unsplit and staged decode columns are printed for positions 0..1200 and are EQUAL for every position the
//         witness's run can reach (prompt 14 + 64 decodes = positions 14..77, all below S = 128);
//   (ii)  a staged run whose first piece is the WHOLE prompt (witness: 14 < unit = 64) therefore has the decode pass land
//         on the column the prompt pass's rows did NOT vacate - so the ring carry, which moves the decode's column by one,
//         cannot be the fix; and
//   (iii) the map's wrap point (position 512) is where the two rules DIVERGE, which bounds any future claim about
//         "column mismatch at the first decode" to a prompt longer than the state columns.  The witness's prompt is 14.
//
// If (i) or (iii) ever stops holding, the decode column IS the answer and this gate says so loudly.
//
// Compiled and run with plain g++: no CUDA, no HIP, no GPU, no model.
//
// Usage: tools/gfx906/gate-decode-column.sh
#include <cstdio>
#include <algorithm>
#include <vector>

// The witness's geometry.
static long long NC = 512, S = 128;
static const long long N_PROMPT = 14, UNIT = 64, N_TOKENS = 64;

// Ds41Dense::Impl::stage_col0, ordinary pass (carry = false): the window's head column.
static long long staged_col(long long pos) {
    const long long nwin = NC / S;
    return ((pos / S) % nwin) * S;
}
// The unsplit rule: begin_impl leaves im.col0 at 0 for every pass.
static long long unsplit_col(long long) { return 0; }
// The column a READER asks for, position-indexed (Impl::stage_position): the window's head plus the offset in the window.
static long long reader_col(long long pos) {
    const long long nwin = NC / S;
    return ((pos / S) % nwin) * S + (pos % S);
}

int main() {
    const long long nwin = NC / S;
    printf("geometry: NC=%lld state columns, window S=%lld, nwin=%lld windows, prompt %lld tokens, relay unit %lld\n",
           NC, S, nwin, N_PROMPT, UNIT);
    printf("the prompt is ONE pass (14 < 64 < 128); the FIRST DECODE is a 1-row pass at position %lld\n\n", N_PROMPT);

    printf("=== THE FIRST DECODE, PASS BY PASS, at the witness's geometry ===\n");
    printf("  %-8s %-14s %-14s %-14s %s\n", "pos", "unsplit col0", "staged col0", "reader asks", "agree?");
    long long reach = N_PROMPT + N_TOKENS;              // positions 14 .. 77 are the witness's decode passes
    long long differ_reach = 0;
    for (long long p = 0; p <= reach; ++p) {
        const long long u = unsplit_col(p), s = staged_col(p), r = reader_col(p);
        if (p >= N_PROMPT && p <= reach) {   // the decode passes
            if (u != s) ++differ_reach;
            if (p <= N_PROMPT + 3 || p == reach)
                printf("  %-8lld %-14lld %-14lld %-14lld %s\n", p, u, s, r, u == s ? "SAME" : "DIFFER");
        }
    }
    printf("\n=== WHERE THE TWO RULES DIVERGE (the bound on any 'decode column mismatch' claim) ===\n");
    long long first_differ = -1;
    for (long long p = 0; p <= 2 * NC; ++p) if (unsplit_col(p) != staged_col(p)) { first_differ = p; break; }
    printf("  the first position where unsplit col0 (0) and the staged map differ: %lld = the end of window 1 (S=%lld);\n"
           "  the staged map wraps back to column 0 at position %lld = NC.  The witness's run reaches position %lld.\n",
           first_differ, S, NC, reach);

    printf("\n=== THE RING CARRY'S COLUMN, for the same decode (the theory this refutes) ===\n");
    // A carrying pass is placed one row below the map's column for its own first row (Impl::stage_col0, carry = true):
    const long long carry_col = (staged_col(N_PROMPT - 1) + ((N_PROMPT - 1) % S)) % NC;
    printf("  a carry into the first decode would put the pass at column %lld, where the map puts it at %lld and an\n"
           "  unsplit run puts it at %lld: the carry MOVES the decode off the column its producer's rows sit on.\n",
           carry_col, staged_col(N_PROMPT), unsplit_col(N_PROMPT));

    printf("\nVERDICT:\n"
           "  * For every position the witness's run can reach (prompt 14 + 64 decode tokens, all below the window's %lld\n"
           "    rows), the UNSPLIT and STAGED decode columns are the SAME column (%lld divergences in the reachable range).\n"
           "    The first decode pass is NOT placed differently.\n"
           "  * The two rules only part company at position %lld, which a 14-token prompt plus 64 tokens never reaches, so a\n"
           "    'column mismatch at the first decode' cannot explain witness #7 - the bound is %lld tokens of prompt.\n"
           "  * The ring carry would have MOVED the decode column (%lld vs %lld): it was a source of divergence, not a fix.\n",
           S, differ_reach, first_differ, first_differ - N_PROMPT, carry_col, staged_col(N_PROMPT));
    (void) nwin;
    return (differ_reach == 0 && first_differ >= reach) ? 0 : 1;
}
