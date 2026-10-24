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
// WHAT THIS DOES NOT DO: it does not allocate pinned memory (RelayStaging below allocates ORDINARY host memory, the
// engine pins if it wants DMA-speed copies), it copies nothing (the engine calls ggml_backend_tensor_get/set), it
// knows no device or backend, and it pipelines nothing beyond the two-slot rotation the native relay ships.

#if !defined(STRATA_GFX906_RELAY_STAGING_HPP)
#define STRATA_GFX906_RELAY_STAGING_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>

namespace strata {
namespace gfx906 {

/// Slots per boundary: the native prefill relay's two double-buffered hand-off buffers (prefill.cpp:1121).
constexpr int kRelaySlots = 2;

/// Which x_state column the stage that PRODUCES across `boundary` wrote, and which column the stage that CONSUMES
/// across it reads.  With no split (n_stage 1) the answer is the unsplit layout: the model's last column, n_layer.
/// The two are the same column - the producer's last layer's finish output - and the engine checks that they agree
/// before it copies anything (a split that does not tile the layers contiguously would otherwise read a column
/// nobody wrote).  The column index times the row stride (D * hc * 4 bytes) is the offset inside a token's slab.
inline bool relay_boundary_columns(int n_stage, int64_t n_layer, int boundary, int64_t & written_by_producer,
                                   int64_t & read_by_consumer) {
    if (n_stage < 2 || boundary < 0 || boundary >= n_stage - 1) return false;
    if (n_layer < n_stage) return false;   // more stages than layers: a stage would own no layer
    // Stages are contiguous and cover the model; the k-th stage gets floor/ceil of n_layer/n_stage in order, which
    // is what device_split's split_plan hands out when the caller does not pass explicit cut points.
    const int64_t base = n_layer / n_stage, rem = n_layer % n_stage;
    int64_t lb = 0, le = 0;
    for (int s = 0; s <= boundary; ++s) { lb = le; le = lb + base + (s < rem ? 1 : 0); }
    written_by_producer = read_by_consumer = le;   // the producer's last finish output; the successor's input column
    return true;
}

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
/// What kind of host memory a RelayStaging holds (the definition of the allocators is below the namespace, and the
/// plain-host one is here for translation units that include this header without RELAY_STAGING_IMPL - the standalone
/// gate compiles it with plain g++ and no CUDA/HIP headers).  Declared here so RelayStaging can use them.
enum class RelayBacking { PlainHost, MappedPinned };
inline uint8_t * relay_alloc(uint64_t bytes, bool want_device_visible, RelayBacking & backing, std::string & note);
inline void relay_free(uint8_t * p, uint64_t bytes, RelayBacking backing);

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
    /// Where a slot's bytes live.  A plan that was never attached to a buffer (base == nullptr - the arithmetic-only
    /// plans the tests and any pre-flight size check use) reports no pointer rather than inventing one, so a caller
    /// cannot mistake a plan's slot index for memory it owns.
    const uint8_t * slot_base(int boundary, int slot) const {
        if (base == nullptr) return nullptr;
        return (const uint8_t *) base + ((uint64_t) slot * (uint64_t) n_boundaries + (uint64_t) boundary) * stride;
    }
    uint8_t * slot_base_w(int boundary, int slot) const {
        return (uint8_t *) slot_base(boundary, slot);
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
    /// The slot bookkeeping is const-callable: it guards a staging buffer a caller may hold by const reference (the
    /// pass driver below does).  Nothing observable outside the plan changes - it is which slot is next, not its bytes.
    bool acquire(int boundary, RelaySlot & out) const {
        if (!ok || boundary < 0 || boundary >= n_boundaries) return false;
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
    void release(int boundary, int slot) const {
        if (slot_check(boundary, slot) < 0 || !inuse_[idx(boundary, slot)]) return;
        inuse_[idx(boundary, slot)] = false;
        --live_[boundary];
    }

    /// Forget every in-flight slot (the start of a new pass sequence, or after an error path).
    void reset() {
        for (int i = 0; i < kMaxBoundaries * kRelaySlots; ++i) inuse_[i] = false;
        for (int i = 0; i < kMaxBoundaries; ++i) { live_[i] = 0; next_[i] = 0; }
    }

    /// How many slots of a boundary still hold an unread payload.
    int in_flight(int boundary) const { return boundary >= 0 && boundary < n_boundaries ? live_[boundary] : 0; }

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
    // mutable: acquire()/release() are const-callable (see above), so a caller holding a const plan can drive a pass
    mutable bool inuse_[kMaxBoundaries * kRelaySlots] = {};
    mutable int live_[kMaxBoundaries] = {};
    mutable int next_[kMaxBoundaries] = {};   ///< the slot the next acquire of that boundary tries first
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

/// The staging buffer itself: the plan's bytes, in host memory (plain, or mapped + portable when the runtime gives it,
struct RelayStaging {
    RelayPlan plan;                 ///< ok = false until init(); `base` points into `mem`
    uint8_t * mem = nullptr;
    RelayBacking backing = RelayBacking::PlainHost;
    std::string note;               ///< what the runtime gave us, for the engine's startup line

    RelayStaging() = default;
    RelayStaging(const RelayStaging &) = delete;
    RelayStaging & operator=(const RelayStaging &) = delete;
    RelayStaging(RelayStaging && o) noexcept : plan(o.plan), mem(o.mem), backing(o.backing), note(std::move(o.note)) {
        o.plan = RelayPlan();
        o.mem = nullptr;
    }
    ~RelayStaging() { free(); }

    /// `want_device_visible`: ask the runtime for mapped + portable host memory (the native relay's choice); on "no"
    /// or on a refusal this is ordinary host memory.  A one-stage plan allocates nothing and succeeds.
    bool init(int n_stage, uint64_t capacity, uint64_t slab_bytes, std::string & err, bool want_device_visible = true) {
        if (!relay_plan(n_stage, capacity, slab_bytes, plan, err)) return false;
        if (plan.bytes == 0) return true;   // one stage: no boundary, no buffer (today's run)
        mem = relay_alloc(plan.bytes, want_device_visible, backing, note);
        if (!mem) {
            err = "relay: cannot allocate the staging of " + std::to_string((long long) plan.bytes) + " bytes (" + note + ")";
            free();
            return false;
        }
        std::memset(mem, 0, plan.bytes);   // a slot nobody wrote reads as zeros, not as garbage
        plan.base = mem;
        return true;
    }
    void free() {
        if (mem) relay_free(mem, plan.bytes, backing);
        mem = nullptr;
        plan = RelayPlan();
    }
    /// The payload of tokens [a,b) of one boundary's slot as a host pointer the engine can hand straight to
    /// ggml_backend_tensor_set/get (a D2H writes it, an H2D reads it).  False when the window does not fit.
    bool payload(int boundary, int slot, uint64_t a, uint64_t b, uint8_t ** out, std::string & err) const {
        RelayCopy c;
        if (!plan.window(boundary, a, b, c, err, slot)) return false;
        *out = mem + c.dst_offset;
        return true;
    }
};


/// Run one pass of `n` tokens across EVERY stage of the split, in order, with the staging `st` attached.
///
/// The pass walks the stages forward: stage s runs its layers, copies the column its last layer wrote into a slot
/// of the boundary that follows, and the next stage uploads that slot and copies it into its own columns before
/// running its own layers.  A stage that is not the last produces across exactly one boundary, so the payload
/// goes straight to its successor and no bytes are copied twice.
///
/// `run(s, err)`       runs stage s's layers (the engine's own loop over its range, expert tier included).
/// `stage_out(s, payload, bytes, err)`
///                     copies the producer's finished column range out of its x_state into the slot.
/// `stage_in(s, payload, bytes, err)`  uploads the slot into stage s's landing pad and copies it into the columns
///                     its layers read.  The offset is each stage's own business: the producer knows the column
///                     its range ends at, the consumer the column its range starts at.
/// `advance(s, err)`   called ONCE per pass, on the LAST stage, after its layers ran.  A later stage owns no token
///                     ids, so the engine advances every stage after the first from this one call; the first stage
///                     tracks its own positions and is never passed to it.
///
/// Everything waits on the host, which is the native relay's rule: the producer's stream is synced inside the
/// read-out before the successor is launched (docs/gfx906/P5-RELAY-PATTERN.md section 1a).
template <typename Runner, typename Producer, typename Consumer, typename Advancer>
bool relay_stage_pass(int n, const RelayStaging & st, Runner && run, Producer && stage_out, Consumer && stage_in,
                      Advancer && advance, std::string & err) {
    err.clear();
    if (!st.plan.ok) { err = "relay: the staging plan is not set"; return false; }
    if (st.plan.n_boundaries == 0) { err = "relay: nothing to relay (one stage)"; return false; }
    if (n < 1) { err = "relay: a pass of " + std::to_string(n) + " tokens"; return false; }
    if ((uint64_t) n > st.plan.capacity) {
        err = "relay: a pass of " + std::to_string(n) + " tokens is past the staging capacity of " +
              std::to_string((long long) st.plan.capacity) + "; split the pass";
        return false;
    }
    for (int s = 0; s < st.plan.n_stage; ++s) {
        if (!run(s, err)) return false;
        if (s == st.plan.n_boundaries) {   // the last stage runs the head and hands on nothing
            if (s > 0 && !advance(s, err)) return false;
            return true;
        }
        uint8_t * payload = nullptr;
        RelaySlot slot;
        if (!st.plan.acquire(s, slot)) {
            err = "relay: every slot of boundary " + std::to_string(s) + " still holds an unread payload";
            return false;
        }
        if (!st.payload(s, slot.slot, 0, (uint64_t) n, &payload, err)) { st.plan.release(s, slot.slot); return false; }
        // the producer writes the slot, its successor reads it, and only then may the slot be written again
        bool bad = false;
        if (!stage_out(s, payload, slot.slot_bytes, err)) bad = true;
        if (!bad && !stage_in(s + 1, payload, slot.slot_bytes, err)) bad = true;
        st.plan.release(s, slot.slot);
        if (bad) return false;
    }
    return true;   // unreachable: the loop returns at the last stage
}

}  // namespace gfx906
}  // namespace strata

/// What kind of host memory a RelayStaging holds: ordinary host memory, or the mapped + portable host allocation the
/// native relay uses for the same job (docs/gfx906/P5-RELAY-PATTERN.md section 1a: src/program/generate.cpp:7883-7896
/// calls cudaHostAlloc(..., cudaHostAllocMapped | cudaHostAllocPortable) per boundary).  The engine asks for
/// MappedPinned and takes whatever the runtime gives it; failing to pin costs bandwidth, never correctness, and is
/// reported in RelayStaging::note.

/// Allocate `bytes` of staging memory, falling back to plain host memory when the runtime refuses (no CUDA/HIP
/// context, no budget).  Returns nullptr only when plain host memory cannot be allocated either.  The plain-host
/// definition below applies to a translation unit that includes this header WITHOUT RELAY_STAGING_IMPL - the standalone
/// gate, which has no CUDA/HIP headers; the engine defines the macro and gets the mapped + portable one after the
/// namespace closes.
namespace strata {
namespace gfx906 {
#if !defined(RELAY_STAGING_IMPL)
inline uint8_t * relay_alloc(uint64_t bytes, bool, RelayBacking & backing, std::string & note) {
    backing = RelayBacking::PlainHost;
    note = "pageable host memory (built without RELAY_STAGING_IMPL: no CUDA/HIP runtime in this translation unit)";
    return (uint8_t *) ::operator new((size_t) bytes, std::nothrow);
}
inline void relay_free(uint8_t * p, uint64_t, RelayBacking) { ::operator delete((void *) p); }
#endif  // !RELAY_STAGING_IMPL
}  // namespace gfx906
}  // namespace strata



// The runtime include, at FILE SCOPE (never inside this file's namespace: a CUDA/HIP header parsed inside a user
// namespace re-declares std and the global ::hip* names inside it, which breaks every later std:: use - the first cut
// of this file did that and produced a wall of "no member named hipGetSymbolAddress" and "no template named 'forward'
// in namespace strata::gfx906::std").  Only an engine translation unit gets it: the standalone gate compiles this
// header with plain g++ and no CUDA/HIP headers, and takes the plain-host definitions above.  A translation unit may
// also set RELAY_STAGING_HAS_RUNTIME itself (with RELAY_STAGING_IMPL) when it has already included the runtime, which
// is how the engine's own include order is respected.
#if defined(RELAY_STAGING_IMPL) && !defined(RELAY_STAGING_HAS_RUNTIME)
#include <cuda_runtime.h>       // CUDA names -> HIP via this repo's hip_compat shim on the AMD build
#define RELAY_STAGING_HAS_RUNTIME 1
#endif

#if defined(RELAY_STAGING_IMPL)
namespace strata {
namespace gfx906 {


inline uint8_t * relay_alloc(uint64_t bytes, bool want_device_visible, RelayBacking & backing, std::string & note) {
#if defined(RELAY_STAGING_HAS_RUNTIME)
    if (want_device_visible) {
        void * p = nullptr;
        // MAPPED: device-visible through a device pointer; PORTABLE: readable from a second device too.  Exactly the
        // flags the native layer-split relay uses (src/program/generate.cpp:7883-7896).
        const int flag = cudaHostAllocMapped | cudaHostAllocPortable;
        if (cudaHostAlloc(&p, (size_t) bytes, flag) == cudaSuccess && p) {
            backing = RelayBacking::MappedPinned;
            note = "mapped + portable host memory (cudaHostAlloc)";
            return (uint8_t *) p;
        }
        note = "cudaHostAlloc(mapped|portable) refused for " + std::to_string((long long) bytes) +
               " bytes - using pageable host memory instead (correct, slower copies)";
    } else {
        note = "pageable host memory (requested)";
    }
#endif
    backing = RelayBacking::PlainHost;
    if (note.empty()) note = "pageable host memory";
    uint8_t * p = (uint8_t *) ::operator new((size_t) bytes, std::nothrow);
    if (!p && note.find("pageable host memory (requested)") == std::string::npos)
        note += "; plain allocation failed too";
    return p;
}

inline void relay_free(uint8_t * p, uint64_t bytes, RelayBacking backing) {
#if defined(RELAY_STAGING_HAS_RUNTIME)
    if (backing == RelayBacking::MappedPinned) {   // the runtime owns these pages; ::operator delete would corrupt them
        (void) cudaFreeHost((void *) p);
        (void) bytes;
        return;
    }
#else
    (void) backing;
#endif
    ::operator delete((void *) p);
}

}  // namespace gfx906
}  // namespace strata
#endif  // RELAY_STAGING_IMPL

#endif  // STRATA_GFX906_RELAY_STAGING_HPP
