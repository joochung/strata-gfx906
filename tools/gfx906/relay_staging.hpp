// tools/gfx906/relay_staging.hpp - P5.3 step 1: the arithmetic of the stage-boundary activation relay of the PORTED
// engines (glm_generate first).  Header-only and free of ggml/HIP includes on purpose, like device_split.hpp, so
// tools/gfx906/test_relay_staging.cpp compiles it with plain g++ in a second, with no CUDA, HIP or GPU present.
//
// WHAT IT IS.  When the layers are split into stages (tools/gfx906/device_split.hpp), the activations cross each
// boundary through host pinned memory - never device-to-device.  This header answers the three questions that
// arithmetic has to answer before any byte moves: how big the staging buffer is, which bytes of it a given pass
// writes or reads, and which slot of the double buffer is safe to write now.  It copies nothing, allocates nothing
// and names no device: the copies go through the ggml backend API in the engine (glm_generate.cpp), and the pinning
// is the engine's own.
//
// THE PATTERN IT COPIES, from the native engine's shipped prefill relay (docs/gfx906/P5-RELAY-PATTERN.md section 1b):
// src/prefill/prefill.cpp:1121-1127 allocates TWO double-buffered hand-off buffers per boundary, each chunk x D x 4
// bytes (D = n_embd * hc, the full hc-wide residual and only that); the producer does a real D2H memcpy and a stream
// sync (:4176-4180) into the slot whose previous payload was consumed two chunks ago (hand_buf_ ^= 1 at :4194), and
// the consumer uploads its chunk from the same pointer (memcpyAsync H2D, :2205-2212).  The sync the successor waits
// on is the producer's stream completion observed by the host, never a cross-device event - which is why a
// host-pinned staging works unchanged whether the stages share a card or not, and why this header can be complete
// while the multi-device expert tier (P5.3 step 2) is still missing.
//
// WHAT THIS DOES NOT DO: no pinned allocation (the engine pins), no copies (the engine calls ggml_backend_tensor_*),
// no device or backend knowledge, no pipelining across passes beyond the two-slot rotation the native relay ships.

#if !defined(STRATA_GFX906_RELAY_STAGING_HPP)
#define STRATA_GFX906_RELAY_STAGING_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata {
namespace gfx906 {

/// Slots per boundary: the native prefill relay's two double-buffered hand-off buffers (prefill.cpp:1121).
constexpr int kRelaySlots = 2;

/// One byte range of a staging slot: the [a,b) token window of one pass, at its offset in the buffer.
struct RelayCopy {
    uint64_t src_offset = 0;   ///< the byte offset in the token-major payload the pass already holds
    uint64_t dst_offset = 0;   ///< the byte offset in the staging buffer (boundary base + slot stride + window)
    uint64_t bytes = 0;        ///< (b - a) * slab_bytes
    int boundary = 0;          ///< the boundary this window belongs to
    int slot = 0;              ///< which slot of that boundary's double buffer
    uint64_t a = 0, b = 0;     ///< the token range, for the caller's own indexing
    bool incomplete = false;   ///< reserved: a partial window is never produced (the relay refuses instead)
};

/// A slot handed out by RelayPlan::acquire: which one it is, and where its pinned memory is.
struct RelaySlot {
    int slot = -1;
    uint64_t slot_bytes = 0;
    void * host = nullptr;   ///< the slot's start in the pinned staging buffer (NULL when none was free)
};

/// The staging layout for one engine run: (n_stage - 1) boundaries x kRelaySlots token ranges of `capacity` tokens.
/// A default-constructed plan is "no relay", which is what a one-stage run uses.
struct RelayPlan {
    bool ok = false;
    int n_stage = 1;
    int n_boundaries = 0;      ///< n_stage - 1
    int slots = kRelaySlots;
    uint64_t capacity = 0;     ///< tokens one slot can hold (the engine's max pass size)
    uint64_t slab_bytes = 0;   ///< the per-token payload (x_state's column stride: D * hc * sizeof(float))
    uint64_t slot_bytes = 0;   ///< capacity * slab_bytes
    uint64_t stride = 0;       ///< the distance between two slots of one boundary (a multiple of slot_bytes)
    uint64_t bytes = 0;        ///< the whole staging buffer the engine must pin
    const void * base = nullptr;   ///< the pinned buffer the engine attaches (never allocated here)
    std::string err;

/// Where a boundary's slot lives.  The layout is slot-major, then boundary: the native relay keeps one pair of
/// buffers PER STAGE (prefill.cpp:1121 allocates m.hand[2] inside one stage's Prefill::init; the stage writes its
/// own pair and its successor reads that same pair), so the stage producing across boundary `b` alternates between
/// stride*b and stride*(b+1) exactly the way hand_buf_ ^= 1 alternates between hand[0] and hand[1].
    const uint8_t * slot_base(int boundary, int slot) const {
        return (const uint8_t *) base + ((uint64_t) slot * (uint64_t) n_boundaries + (uint64_t) boundary) * stride;
    }
    uint8_t * slot_base_w(int boundary, int slot) const {
        return (uint8_t *) base + ((uint64_t) slot * (uint64_t) n_boundaries + (uint64_t) boundary) * stride;
    }

    /// The copy window for tokens [a,b) of a pass through `boundary`.  With no slot given it is the boundary's
    /// first slot (the pure arithmetic question); a caller that rotates slots passes the one it acquired.
    /// Fails - and leaves `out` untouched - when the range is empty, reversed, or does not fit the slot: a pass too
    /// long for the staging is the caller's problem to split, the relay never writes outside its slot.
    bool window(int boundary, uint64_t a, uint64_t b, RelayCopy & out, std::string & err, int slot = 0) const {
        err.clear();
        if (!ok) { err = "relay: the staging plan is not set"; return false; }
        if (boundary < 0 || boundary >= n_boundaries) {
            err = "relay: boundary " + std::to_string(boundary) + " of " + std::to_string(n_boundaries) +
                  " (stage 0 consumes no relay)";
            return false;
        }
        if (slot_check(boundary, slot) < 0) {
            err = "relay: slot " + std::to_string(slot) + " is not one of boundary " + std::to_string(boundary) +
                  "'s " + std::to_string(slots) + " slots";
            return false;
        }
        if (b <= a) { err = "relay: empty token range [" + std::to_string((long long) a) + "," +
                             std::to_string((long long) b) + ")"; return false; }
        if (b > capacity) {
            err = "relay: token " + std::to_string((long long) b) + " is past the staging capacity of " +
                  std::to_string((long long) capacity) + " tokens/slot (" + std::to_string((long long) slab_bytes) +
                  " B per token); split the pass";
            return false;
        }
        out.src_offset = a * slab_bytes;
        out.dst_offset = ((uint64_t) slot * (uint64_t) n_boundaries + (uint64_t) boundary) * stride + a * slab_bytes;
        out.bytes = (b - a) * slab_bytes;
        out.boundary = boundary;
        out.slot = slot;
        out.a = a;
        out.b = b;
        out.incomplete = false;
        if (out.dst_offset + out.bytes > bytes) { err = "relay: the window would leave the staging buffer"; return false; }
        return true;
    }

    /// The slot to write this pass: the oldest released one, or the next free one.  False when every slot of that
    /// boundary still holds a payload a later stage has not read yet - the caller must let the successor consume
    /// before producing again (the native relay's next_run_.get() at prefill.cpp:4186-4188, in synchronous form).
    bool acquire(int boundary, RelaySlot & out) {
        if (!ok || boundary < 0 || boundary >= n_boundaries) return false;
        // round-robin over the free slots, so a plain produce/consume alternation reuses the slot the successor
        // released two passes ago rather than the one it just freed (the native relay's hand_buf_ ^= 1)
        for (int k = 0; k < slots; ++k) {
            const int s = (next_[boundary] + k) % slots;
            const int i = idx(boundary, s);
            if (inuse_[i]) continue;
            inuse_[i] = true;
            ++live_[boundary];
            next_[boundary] = (s + 1) % slots;
            out.slot = s;
            out.slot_bytes = slot_bytes;
            out.host = (void *) slot_base(boundary, s);
            return true;
        }
        out.slot = -1;
        out.host = nullptr;
        return false;
    }

    /// The successor has read this slot: it may be written again.
    void release(int boundary, int slot) {
        if (slot_check(boundary, slot) < 0 || !inuse_[idx(boundary, slot)]) return;
        inuse_[idx(boundary, slot)] = false;
        --live_[boundary];
    }

    /// How many slots of a boundary still hold an unread payload.
    int in_flight(int boundary) const { return boundary >= 0 && boundary < n_boundaries ? live_[boundary] : 0; }

    /// Forget every in-flight slot (the start of a new pass sequence, or after an error path).
    void reset() {
        for (int i = 0; i < kMaxBoundaries * kRelaySlots; ++i) inuse_[i] = false;
        for (int i = 0; i < kMaxBoundaries; ++i) { live_[i] = 0; next_[i] = 0; }
    }

    /// Slot indices are validated, not trusted: a caller that passes the wrong boundary must not touch a
    /// neighbouring boundary's bookkeeping.
    int slot_check(int boundary, int slot) const {
        if (boundary < 0 || boundary >= n_boundaries || boundary >= kMaxBoundaries) return -1;
        if (slot < 0 || slot >= slots || slot >= kRelaySlots) return -1;
        return idx(boundary, slot);
    }
    static int idx(int boundary, int slot) { return boundary * kRelaySlots + slot; }

    /// A plan is built for at most this many boundaries (device_split.hpp's kSplitMaxStages minus one).
    static constexpr int kMaxBoundaries = 7;

private:
    bool inuse_[kMaxBoundaries * kRelaySlots] = {};
    int live_[kMaxBoundaries] = {};
    int next_[kMaxBoundaries] = {};   ///< the slot the next acquire of that boundary tries first
};

/// The staging math for a run: `n_stage` stages, each pass at most `capacity` tokens, `slab_bytes` per token.
/// Pure: no allocation, no device query, so it is testable without a GPU.  One stage is the byte-identical default
/// and yields an empty plan (ok = true, bytes = 0).  Returns false with the reason in err on a nonsense request.
inline bool relay_plan(int n_stage, uint64_t capacity, uint64_t slab_bytes, RelayPlan & p, std::string & err) {
    p = RelayPlan();
    if (n_stage < 1) { err = "relay: " + std::to_string(n_stage) + " stages"; return false; }
    if (n_stage - 1 > RelayPlan::kMaxBoundaries) {
        err = "relay: " + std::to_string(n_stage) + " stages is past the " +
              std::to_string(RelayPlan::kMaxBoundaries + 1) + " stages the staging supports";
        return false;
    }
    if (n_stage == 1) {   // no boundary, no bytes: today's single-stage run never calls the relay
        p.ok = true;
        p.n_stage = 1;
        err.clear();
        return true;
    }
    if (slab_bytes < 1) { err = "relay: the per-token payload is 0 bytes"; return false; }
    if (capacity < 1) { err = "relay: the staging capacity is 0 tokens"; return false; }
    if (capacity > ~0ull / slab_bytes) {
        err = "relay: " + std::to_string((long long) capacity) + " tokens x " + std::to_string((long long) slab_bytes) +
              " bytes overflows the staging size";
        return false;
    }
    const int nb = n_stage - 1;
    p.slot_bytes = capacity * slab_bytes;
    p.stride = p.slot_bytes;   // one slot after another, no padding: the native relay's buffers are chunk*D*4 exactly
    if (nb > 0 && ~0ull / ((uint64_t) nb * kRelaySlots) < p.stride) {
        err = "relay: " + std::to_string(nb) + " boundaries x " + std::to_string(kRelaySlots) + " slots x " +
              std::to_string((long long) p.stride) + " bytes overflows the staging size";
        return false;
    }
    p.n_stage = n_stage;
    p.n_boundaries = nb;
    p.slots = kRelaySlots;
    p.capacity = capacity;
    p.slab_bytes = slab_bytes;
    p.bytes = (uint64_t) nb * (uint64_t) kRelaySlots * p.stride;
    p.ok = true;
    err.clear();
    return true;
}

}  // namespace gfx906
}  // namespace strata

#endif  // STRATA_GFX906_RELAY_STAGING_HPP
