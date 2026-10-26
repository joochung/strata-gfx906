// probe-collective - P6.0 small-collective latency probe (task #35): the kill gate for decode
// tensor-parallelism, per docs/gfx906/P6-TENSOR-SPLIT.md sections 2, 2b and 5.
//
// P5.1 (probe-p2p) measured 64-128 MB bursts. A tensor-parallel step moves 4-1280 KiB
// collectives hundreds of times per second: a regime with NO measurement on this box yet.
// This probe produces it. Design input, not a speed claim (AGENTS.md): the numbers decide
// whether decode-TP is a question at all, and which reduction dtype has wire time worth saving.
//
// Build (gfx906 is wave64 natively; RCCL is linked HERE only - the engine build does not link it):
//   /opt/rocm/bin/hipcc --offload-arch=gfx906 -O2 -pthread \
//     -I/opt/rocm/include -L/opt/rocm/lib -lrccl \
//     -o build-906/probe-collective tools/gfx906/probe-collective.cu
// Run via tools/gfx906/run-p6.0-probe.sh (engine-alive guard). One PROCESS on all cards -
// the standing one-HIP-engine rule, and the same process model a TP engine would use.
//
// Modes (comma list in argv, default kern,bounce,peer,rccl):
//   kern   fused add kernels standalone (f32 add, f16-dequant-add, q8-dequant-add) at the two
//          real payload element counts (5120 and 327680) - the COMPUTE cost of the reduction
//          dtypes, without any transport.
//   bounce host-pinned bounce barrier: D2H own partial -> host barrier -> H2D peers' partials ->
//          fused add on each card; host-driven, per-iteration barrier (the transport the engine
//          relay already has, measured in collective shape). 2-way and 4-way.
//   peer   device-driven peer-direct allreduce with NO host in the loop: kernel 1 grid-strides
//          the partial into every peer's inbox over P2P, last block fence+flags; kernel 2 spins
//          on flags, fences, sums. Host only launches and syncs per iteration. Pairs (0,1)
//          same-complex, (1,2) cross-complex, and 4-way.
//   rccl   RCCL ring allreduce, single process (ncclCommInitAll), per-iteration
//          stream-sync (the regime the engine's synchronous graph compute would use). 2 and 4.
//
// Payload size grid (plan 2b): 4/6/10/20/40/320/384/640/1280 KiB.
//   n=1:   f32 = 20 KiB, f16 = 10 KiB, Q8_0 = 5.3 KiB (probe 6 as the rounded q8 point)
//   n=64:  f32 = 1280 KiB, f16 = 640 KiB, Q8_0 = 345 KiB (probe 384)
// The GO bar: decode-TP needs ~150 us p50 at 20 KiB, 2-way. The transport/dtype table decides
// where each reduction dtype has wire time worth saving (it does not at 20 KiB on bounce:
// 0.8 us of transfer under a 330-690 us hop).

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                \
    do {                                                                     \
        hipError_t e_ = (x);                                                 \
        if (e_ != hipSuccess) {                                              \
            std::fprintf(stderr, "FATAL %s -> %s (%s:%d)\n", #x,             \
                         hipGetErrorString(e_), __FILE__, __LINE__);         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define NK(x)                                                                \
    do {                                                                     \
        ncclResult_t r_ = (x);                                               \
        if (r_ != ncclSuccess) {                                             \
            std::fprintf(stderr, "FATAL %s -> %s (%s:%d)\n", #x,             \
                         ncclGetErrorString(r_), __FILE__, __LINE__);        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

typedef unsigned long long u64;
static const size_t KIB = 1024ull;
static const size_t MAX_S = 1280ull * KIB;                 // biggest payload, bytes
static const size_t MAX_ELEMS = MAX_S / 4;                 // as f32
static const size_t SIZES_KIB[] = {4, 6, 10, 20, 40, 320, 384, 640, 1280};
static const int NSIZES = (int)(sizeof(SIZES_KIB) / sizeof(SIZES_KIB[0]));

// ---------------------------------------------------------------- utilities

static double percentile(std::vector<double> v, double q) {  // q in [0,1]
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)(q * (double)(v.size() - 1) + 0.5);
    return v[i];
}

static void report(const char* mode, const char* shape, size_t kib, int iters,
                   const std::vector<double>& us, int sanity) {
    double p50 = percentile(us, 0.50), p90 = percentile(us, 0.90);
    double mean = 0;
    for (double u : us) mean += u;
    mean /= (double)(us.empty() ? 1 : us.size());
    double gb = (double)(kib * KIB) / (p50 * 1e3 / 1e9) / (1024.0 * 1024.0);  // MiB/s of payload
    std::printf("PROBE mode=%-6s shape=%-10s size_KiB=%-5zu iters=%-4d p50_us=%9.1f p90_us=%9.1f "
                "mean_us=%9.1f payload_MiBs=%8.1f sanity=%s\n",
                mode, shape, kib, iters, p50, p90, mean, gb,
                sanity > 0 ? "OK" : sanity == 0 ? "MISMATCH" : "-");
    std::fflush(stdout);
}

static int timed_iters(int size_kib) { return size_kib < 64 ? 300 : 60; }
static const int WARMUP = 10;

// ---------------------------------------------------------------- kernels

// fused sum of own partial + up to 3 peer partials -> out (bounce transport's reduce node).
// The peer pointers travel IN A STRUCT BY VALUE: the first version passed a HOST array of
// device pointers and the kernel dereferenced a host address -> illegal memory access (witnessed
// on hardware, witness log p6.0-probe-20261025-210301). Kernel params are the fix and the lesson.
struct Peers3 {
    const float* p[3];
    int n;
};
__global__ void k_add(const float* __restrict__ own, Peers3 peers, float* __restrict__ out,
                      size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        float s = own[i];
        for (int j = 0; j < peers.n; ++j) s += peers.p[j][i];
        out[i] = s;
    }
}

// standalone dtype kernels (payload == what crosses the wire; out is f32)
__global__ void k_add_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
        out[i] = a[i] + b[i];
}
__global__ void k_add_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          float* __restrict__ out, size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
        out[i] = __half2float(a[i]) + __half2float(b[i]);
}
struct q8blk {  // ggml Q8_0 layout: fp16 block scale + 32 int8 (34 B / 32 values)
    __half d;
    signed char qs[32];
};
__global__ void k_add_q8(const q8blk* __restrict__ a, const q8blk* __restrict__ b,
                         float* __restrict__ out, size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        size_t blk = i / 32, off = i % 32;
        out[i] = __half2float(a[blk].d) * (float)a[blk].qs[off]
               + __half2float(b[blk].d) * (float)b[blk].qs[off];
    }
}

// peer transport (copy-engine variant, NON-BLOCKING STREAMS - see the deadlock lesson below).
// r2 used kernel stores into peer memory (XFB), r3 used copy-engine DMA on the DEFAULT stream;
// both hung at the first spin. STRONGEST SUSPECT (consistent with both hangs): the LEGACY
// default stream is an implicit GLOBAL sync point ACROSS peer-access-linked devices, so rank
// 0's spin (enqueued first) blocks rank 1's later-enqueued inbound DMA - a self-made deadlock
// that looks exactly like broken XFB. The XFB verdict is therefore WITHDRAWN: this variant puts
// every rank's work on its own hipStreamNonBlocking, where no cross-device implicit sync exists:
//   r: for each peer p: DMA partial -> p's inbox slot r; DMA seq -> p's flags[r];
//   r: local spin+sum kernel polls its OWN flags (local VRAM, written by peer DMA) and sums.
// If THIS variant still hangs at the first iteration, only then is kernel-path P2P (or inbound
// DMA under bidirectional load) really dead on this board - and RCCL P2P protocols share that
// risk. The runner also measures NCCL_P2P_DISABLE=1 as the fallback.
struct PeerLocal {
    const float* inbox[3];  // LOCAL device ptrs: inbox slot p = data FROM peer p
    int n;
};
__global__ void pk_spin_sum(const float* __restrict__ partial, PeerLocal loc, int self,
                            const u64* flags, int nd, u64 seq, float* __restrict__ out,
                            size_t count) {
    for (int f = 0; f < nd; ++f) {
        if (f == self) continue;
        // POLL VIA ATOMIC: an atomicAdd of 0 is an L2-coherent read. A plain volatile spin is
        // the prime suspect for BOTH peer hangs (r2 XFB, r3 copy-engine): the polling CU can
        // keep serving a stale line while peer DMA writes land in L2 - livelock, no fault.
        while (atomicAdd((u64*)&flags[f], 0ull) < seq) { }
    }
    __threadfence_system();
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        float s = partial[i];
        for (int j = 0; j < loc.n; ++j) s += loc.inbox[j][i];
        out[i] = s;
    }
}

// ---------------------------------------------------------------- kern mode

static void run_kern() {
    CK(hipSetDevice(0));
    const size_t counts[2] = {5120, 327680};  // n=1 (20 KiB f32) and n=64 (1280 KiB f32)
    float *a32, *b32, *o32;
    __half *a16, *b16;
    q8blk *a8, *b8;
    size_t max_blocks = (MAX_ELEMS + 31) / 32;
    CK(hipMalloc(&a32, MAX_ELEMS * 4)); CK(hipMalloc(&b32, MAX_ELEMS * 4));
    CK(hipMalloc(&o32, MAX_ELEMS * 4));
    CK(hipMalloc(&a16, MAX_ELEMS * 2)); CK(hipMalloc(&b16, MAX_ELEMS * 2));
    CK(hipMalloc(&a8, max_blocks * sizeof(q8blk))); CK(hipMalloc(&b8, max_blocks * sizeof(q8blk)));
    CK(hipMemset(a32, 0x3c, MAX_ELEMS * 4)); CK(hipMemset(b32, 0x3c, MAX_ELEMS * 4));
    CK(hipMemset(a16, 0x3c, MAX_ELEMS * 2)); CK(hipMemset(b16, 0x3c, MAX_ELEMS * 2));
    CK(hipMemset(a8, 0x3c, max_blocks * sizeof(q8blk))); CK(hipMemset(b8, 0x3c, max_blocks * sizeof(q8blk)));

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int ITERS = 500;
    for (size_t c : counts) {
        size_t kib = c * 4 / KIB;
        struct { const char* name; int kind; } kinds[] = {
            {"add-f32", 0}, {"add-f16", 1}, {"add-q8", 2}};
        for (auto& k : kinds) {
            for (int it = -WARMUP; it < ITERS; ++it) {
                if (it == 0) CK(hipEventRecord(e0));
                if (k.kind == 0)
                    k_add<<<128, 256>>>(a32, Peers3{{b32, b32, b32}, 1}, o32, c);
                else if (k.kind == 1)
                    k_add_f16<<<128, 256>>>(a16, b16, o32, c);
                else
                    k_add_q8<<<128, 256>>>(a8, b8, o32, c);
            }
            CK(hipEventRecord(e1));
            CK(hipEventSynchronize(e1));
            float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
            // one clean timed pass was 500 iters between e0 and e1; warmup launched before e0
            // re-measure cleanly per-iteration only via total/ITERS (kernel is serial per stream)
            std::printf("PROBE mode=kern   shape=dev0-%s size_KiB=%-5zu iters=%-4d "
                        "p50_us=%9.1f p90_us=%9.1f mean_us=%9.1f payload_MiBs=%8.1f sanity=-\n",
                        k.name, kib, ITERS, ms * 1000.0 / ITERS, ms * 1000.0 / ITERS,
                        ms * 1000.0 / ITERS,
                        (double)(c * 4) / (ms / ITERS * 1e-3) / (1024.0 * 1024.0));
            std::fflush(stdout);
        }
        (void)0;
    }
    CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
    CK(hipFree(a32)); CK(hipFree(b32)); CK(hipFree(o32));
    CK(hipFree(a16)); CK(hipFree(b16)); CK(hipFree(a8)); CK(hipFree(b8));
}

// ---------------------------------------------------------------- bounce mode

static void run_bounce(int ndev, const int* devs) {
    char shape[32];
    std::snprintf(shape, sizeof(shape), "%dway", ndev);
    float** partial = new float*[ndev];
    float** out = new float*[ndev];
    float** scratch = new float*[ndev];  // (ndev-1) slots of MAX_S
    for (int r = 0; r < ndev; ++r) {
        CK(hipSetDevice(devs[r]));
        CK(hipMalloc(&partial[r], MAX_S));
        CK(hipMalloc(&out[r], MAX_S));
        CK(hipMalloc(&scratch[r], (size_t)(ndev - 1) * MAX_S));
        std::vector<float> h(MAX_S / 4, (float)r + 0.5f);
        CK(hipMemcpy(partial[r], h.data(), MAX_S, hipMemcpyHostToDevice));
    }
    char* pin = nullptr;  // host-pinned slots, one per rank
    CK(hipHostMalloc(&pin, (size_t)ndev * MAX_S, hipHostMallocDefault));
    float* pinslots[4];
    for (int r = 0; r < ndev; ++r) pinslots[r] = (float*)(pin + (size_t)r * MAX_S);

    for (int s = 0; s < NSIZES; ++s) {
        size_t kib = SIZES_KIB[s], bytes = kib * KIB, count = bytes / 4;
        int iters = timed_iters((int)kib);
        // rank r's scratch has one slot per peer p != r, in peer-ordinal order; peers[r].p[slot]
        // is the device ptr rank r's add kernel reads for peer p, and rslot[r][p] is the slot the
        // H2D leg must write for peer p. ONE mapping, computed once - the two legs share it.
        Peers3 peers[4];
        for (int i = 0; i < 4; ++i) peers[i] = Peers3{{nullptr, nullptr, nullptr}, 0};
        std::vector<std::vector<int>> rslot((size_t)ndev, std::vector<int>((size_t)ndev, -1));
        for (int r = 0; r < ndev; ++r) {
            int slot = 0;
            for (int p = 0; p < ndev; ++p)
                if (p != r) {
                    peers[r].p[slot] = (const float*)((char*)scratch[(size_t)r] + (size_t)slot * MAX_S);
                    rslot[(size_t)r][(size_t)p] = slot;
                    ++slot;
                }
            peers[r].n = ndev - 1;
        }

        std::vector<double> us;
        int sanity = -1;
        for (int it = -WARMUP; it < iters; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < ndev; ++r) {  // 1. D2H own partial
                CK(hipSetDevice(devs[r]));
                CK(hipMemcpyAsync(pinslots[r], partial[r], bytes, hipMemcpyDeviceToHost));
            }
            for (int r = 0; r < ndev; ++r) { CK(hipSetDevice(devs[r])); CK(hipDeviceSynchronize()); }
            for (int r = 0; r < ndev; ++r) {  // 2. H2D peers' partials + fused add
                CK(hipSetDevice(devs[r]));
                for (int p = 0; p < ndev; ++p)
                    if (p != r)
                        CK(hipMemcpyAsync((char*)scratch[(size_t)r] + (size_t)rslot[(size_t)r][(size_t)p] * MAX_S,
                                          pinslots[p], bytes, hipMemcpyHostToDevice));
                k_add<<<128, 256>>>(partial[r], peers[r], out[r], count);
            }
            for (int r = 0; r < ndev; ++r) { CK(hipSetDevice(devs[r])); CK(hipDeviceSynchronize()); }
            auto t1 = std::chrono::steady_clock::now();
            if (it >= 0)
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        {  // sanity: every rank's out equals sum of (rank + 0.5)
            float expect = 0;
            for (int r = 0; r < ndev; ++r) expect += (float)r + 0.5f;
            std::vector<float> h(count);
            sanity = 1;
            for (int r = 0; r < ndev && sanity; ++r) {
                CK(hipSetDevice(devs[r]));
                CK(hipMemcpy(h.data(), out[r], bytes, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < count; i += count / 8 + 1)
                    if (h[i] != expect) { sanity = 0; break; }
            }
        }
        report("bounce", shape, kib, iters, us, sanity);
    }
    CK(hipHostFree(pin));
    for (int r = 0; r < ndev; ++r) {
        CK(hipSetDevice(devs[r]));
        CK(hipFree(partial[r])); CK(hipFree(out[r])); CK(hipFree(scratch[r]));
    }
    delete[] partial; delete[] out; delete[] scratch;
}

// ---------------------------------------------------------------- peer mode

static void run_peer(int ndev, const int* devs) {
    char shape[32];
    std::snprintf(shape, sizeof(shape), "%dway-%d-%d", ndev, devs[0], devs[ndev - 1]);
    // per-rank device buffers: inbox[ndev][MAX_S] (slot p = data FROM peer p), flags[ndev]
    float** inbox = new float*[ndev];
    u64** flags = new u64*[ndev];
    u64** seqbuf = new u64*[ndev];  // device u64 holding this rank's current seq (flag payload)
    float** partial = new float*[ndev];
    float** out = new float*[ndev];
    for (int r = 0; r < ndev; ++r) {
        CK(hipSetDevice(devs[r]));
        CK(hipMalloc(&inbox[r], (size_t)ndev * MAX_S));
        CK(hipMemset(inbox[r], 0, (size_t)ndev * MAX_S));
        CK(hipMalloc(&flags[r], (size_t)ndev * sizeof(u64)));
        CK(hipMemset(flags[r], 0, (size_t)ndev * sizeof(u64)));
        CK(hipMalloc(&seqbuf[r], sizeof(u64)));
        CK(hipMemset(seqbuf[r], 0, sizeof(u64)));
        CK(hipMalloc(&partial[r], MAX_S));
        CK(hipMalloc(&out[r], MAX_S));
        std::vector<float> h(MAX_S / 4, (float)r + 0.5f);
        CK(hipMemcpy(partial[r], h.data(), MAX_S, hipMemcpyHostToDevice));
    }
    u64* seqpin = nullptr;  // pinned seq staging (stable until the next iteration's barrier)
    CK(hipHostMalloc(&seqpin, (size_t)ndev * sizeof(u64), hipHostMallocDefault));
    hipStream_t* strm = new hipStream_t[ndev];  // NON-BLOCKING per-rank streams (no legacy-
    for (int r = 0; r < ndev; ++r) {
        // streams belong to the CURRENT device's context: creating them all without SetDevice
        // bound every stream to the LAST device (r6 witness: "invalid resource handle" when
        // rank 0 launched on a device-3-owned stream). One SetDevice per create, always.
        CK(hipSetDevice(devs[r]));
        CK(hipStreamCreateWithFlags(&strm[r], hipStreamNonBlocking));
    }
    u64 seq = 0;
    for (int s = 0; s < NSIZES; ++s) {
        size_t kib = SIZES_KIB[s], bytes = kib * KIB, count = bytes / 4;
        std::printf("MARK peer %s: size %zu KiB\n", shape, kib); std::fflush(stdout);
        int iters = timed_iters((int)kib);
        // local inbox pointers for this size, passed to the kernel BY VALUE
        PeerLocal loc[4];
        for (int i = 0; i < 4; ++i) loc[i] = PeerLocal{{nullptr, nullptr, nullptr}, 0};
        for (int r = 0; r < ndev; ++r) {
            int slot = 0;
            for (int p = 0; p < ndev; ++p) {
                if (p == r) continue;
                loc[r].inbox[slot] = (const float*)((char*)inbox[(size_t)r] + (size_t)p * MAX_S);
                ++slot;
            }
            loc[r].n = slot;
        }
        std::vector<double> us;
        int sanity = -1;
        for (int it = -WARMUP; it < iters; ++it) {
            ++seq;  // one seq per iteration, shared by all ranks (monotonic, no resets)
            auto t0 = std::chrono::steady_clock::now();
            int grid = (int)std::min<size_t>(128, (count + 255) / 256);
            seqpin[0] = seq;  // same value on every rank (pinned staging slot 0)
            for (int r = 0; r < ndev; ++r) {  // stream-ordered on r's NON-BLOCKING stream
                CK(hipSetDevice(devs[r]));
                CK(hipMemcpyAsync(seqbuf[(size_t)r], seqpin, sizeof(u64), hipMemcpyHostToDevice,
                                  strm[r]));
                for (int p = 0; p < ndev; ++p)
                    if (p != r) {
                        // Explicit peer copy: r6's generic D2D on the owning stream still
                        // surfaced "invalid resource handle"; hipMemcpyPeerAsync names both
                        // devices and is the documented cross-device call.
                        CK(hipMemcpyPeerAsync((char*)inbox[p] + (size_t)r * MAX_S, devs[p],
                                              partial[r], devs[r], bytes, strm[r]));
                        CK(hipMemcpyPeerAsync(flags[p] + r, devs[p], seqbuf[(size_t)r], devs[r],
                                              sizeof(u64), strm[r]));
                    }
                CK(hipGetLastError());  // async-error split: a DMA failure surfaces HERE,
                // not misattributed to the kernel launch on the next line
                pk_spin_sum<<<grid, 256, 0, strm[r]>>>(partial[r], loc[r], r,
                                                       flags[(size_t)r], ndev, seq,
                                                       out[r], count);
                CK(hipGetLastError());  // catch launch failures here, not three syncs later
            }
            for (int r = 0; r < ndev; ++r) { CK(hipSetDevice(devs[r])); CK(hipStreamSynchronize(strm[r])); }
            auto t1 = std::chrono::steady_clock::now();
            if (it >= 0)
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        {
            float expect = 0;
            for (int r = 0; r < ndev; ++r) expect += (float)r + 0.5f;
            std::vector<float> h(count);
            sanity = 1;
            for (int r = 0; r < ndev && sanity; ++r) {
                CK(hipSetDevice(devs[r]));
                CK(hipMemcpy(h.data(), out[r], bytes, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < count; i += count / 8 + 1)
                    if (h[i] != expect) { sanity = 0; break; }
            }
        }
        report("peer", shape, kib, iters, us, sanity);
    }
    CK(hipHostFree(seqpin));
    for (int r = 0; r < ndev; ++r) CK(hipStreamDestroy(strm[r]));
    delete[] strm;
    for (int r = 0; r < ndev; ++r) {
        CK(hipSetDevice(devs[r]));
        CK(hipFree(inbox[r])); CK(hipFree(flags[r])); CK(hipFree(seqbuf[r]));
        CK(hipFree(partial[r])); CK(hipFree(out[r]));
    }
    delete[] inbox; delete[] flags; delete[] seqbuf; delete[] partial; delete[] out;
}

// ---------------------------------------------------------------- rccl mode

static void run_rccl(int ndev, const int* devs) {
    char shape[32];
    std::snprintf(shape, sizeof(shape), "%dway", ndev);
    // ncclCommInitAll: the single-process, multi-GPU init. The PREVIOUS version called
    // ncclCommInitRank rank-by-rank in one thread - CommInitRank blocks until ALL ranks join,
    // so rank 0 waited forever for rank 1 (r4 witness: MARK CommInitRank r0 was the last line,
    // and the rccl-shm fallback hung on the SAME line - it was never the transport).
    std::printf("MARK rccl %s: CommInitAll\n", shape); std::fflush(stdout);
    std::vector<ncclComm_t> comms((size_t)ndev);
    NK(ncclCommInitAll(comms.data(), ndev, devs));
    std::printf("MARK rccl %s: CommInitAll done\n", shape); std::fflush(stdout);
    std::vector<float*> bufs((size_t)ndev), outs((size_t)ndev);
    std::vector<hipStream_t> streams((size_t)ndev);
    for (int r = 0; r < ndev; ++r) {
        CK(hipSetDevice(devs[r]));
        CK(hipStreamCreateWithFlags(&streams[(size_t)r], hipStreamNonBlocking));
        CK(hipMalloc(&bufs[(size_t)r], MAX_S));
        CK(hipMalloc(&outs[(size_t)r], MAX_S));
        std::vector<float> h(MAX_S / 4, (float)r + 0.5f);
        CK(hipMemcpy(bufs[(size_t)r], h.data(), MAX_S, hipMemcpyHostToDevice));
    }
    for (int s = 0; s < NSIZES; ++s) {
        std::printf("MARK rccl %s: size %zu KiB\n", shape, SIZES_KIB[s]); std::fflush(stdout);
        size_t kib = SIZES_KIB[s], bytes = kib * KIB, count = bytes / 4;
        int iters = timed_iters((int)kib);
        // SEQUENTIAL ENQUEUE, then sync all: ncclAllReduce's host call only enqueues, and the
        // ring kernels spin on each other device-side, so both ranks' enqueues landing before
        // any sync is all the collective concurrency requires. (r5: the one-thread-per-rank
        // variant SEGFAULTED inside RCCL on the first allreduce, both default and SHM - a
        // single-process multi-thread call pattern RCCL 2.27 does not tolerate here.)
        std::vector<double> us;
        int sanity = -1;
        for (int it = -WARMUP; it < iters; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < ndev; ++r) {
                CK(hipSetDevice(devs[r]));
                NK(ncclAllReduce(bufs[(size_t)r], outs[(size_t)r], count, ncclFloat, ncclSum,
                                 comms[(size_t)r], streams[(size_t)r]));
            }
            for (int r = 0; r < ndev; ++r) {
                CK(hipSetDevice(devs[r]));
                CK(hipStreamSynchronize(streams[(size_t)r]));
            }
            auto t1 = std::chrono::steady_clock::now();
            if (it >= 0)
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        {
            float expect = 0;
            for (int q = 0; q < ndev; ++q) expect += (float)q + 0.5f;
            std::vector<float> h(count);
            sanity = 1;
            for (int r = 0; r < ndev && sanity; ++r) {
                CK(hipSetDevice(devs[r]));
                CK(hipMemcpy(h.data(), outs[(size_t)r], bytes, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < count; i += count / 8 + 1)
                    if (h[i] != expect) { sanity = 0; break; }
            }
        }
        report("rccl", shape, kib, iters, us, sanity);
    }
    for (int r = 0; r < ndev; ++r) {
        NK(ncclCommDestroy(comms[(size_t)r]));
        CK(hipSetDevice(devs[r]));
        CK(hipStreamDestroy(streams[(size_t)r]));
        CK(hipFree(bufs[(size_t)r])); CK(hipFree(outs[(size_t)r]));
    }
}

// ---------------------------------------------------------------- main

// rccl multiprocess transport: fork one process per rank (the classic NCCL deployment shape).
// WHY (r5+r6): the SINGLE-PROCESS paths are where RCCL 2.27 on this ROCm 7.2 box SEGFAULTS -
// one-thread-per-rank on the first allreduce, and sequential-enqueue from one thread at the
// same place (both after "post-adjustment ... nc:1", host-side, default AND SHM transports).
// ncclCommInitAll itself succeeded both times, so the crash is in RCCL's single-process
// collective-enqueue path, not the transport. Multi-process is also how a serving engine would
// run RCCL ranks, so this is the honest ceiling measurement for GPU-direct collectives here.
// UniqueId is created in the parent and inherited by fork; each child completes CommInitRank
// (blocking init is fine ACROSS processes) and times its own loop, reporting row "rcclmp r<r>".
static int run_rcclmp(int ndev, const int* devs) {
    ncclUniqueId id;
    NK(ncclGetUniqueId(&id));
    std::printf("MARK rcclmp %dway: id acquired, forking %d rank processes\n", ndev, ndev);
    std::fflush(stdout);
    std::vector<pid_t> kids;
    for (int r = 0; r < ndev; ++r) {
        pid_t pid = fork();
        if (pid < 0) { std::perror("fork"); return 1; }
        if (pid == 0) {
            CK(hipSetDevice(devs[r]));
            ncclComm_t comm;
            std::printf("MARK rcclmp r%d: CommInitRank\n", r); std::fflush(stdout);
            NK(ncclCommInitRank(&comm, ndev, id, r));
            std::printf("MARK rcclmp r%d: CommInitRank done\n", r); std::fflush(stdout);
            hipStream_t strm;
            CK(hipStreamCreateWithFlags(&strm, hipStreamNonBlocking));
            float *buf = nullptr, *out = nullptr;
            CK(hipMalloc(&buf, MAX_S));
            CK(hipMalloc(&out, MAX_S));
            std::vector<float> h(MAX_S / 4, (float)r + 0.5f);
            CK(hipMemcpy(buf, h.data(), MAX_S, hipMemcpyHostToDevice));
            char shape[16];
            std::snprintf(shape, sizeof(shape), "r%d", r);
            for (int s = 0; s < NSIZES; ++s) {
                size_t kib = SIZES_KIB[s], bytes = kib * KIB, count = bytes / 4;
                int iters = timed_iters((int)kib);
                std::vector<double> us;
                for (int it = -WARMUP; it < iters; ++it) {
                    auto t0 = std::chrono::steady_clock::now();
                    NK(ncclAllReduce(buf, out, count, ncclFloat, ncclSum, comm, strm));
                    CK(hipStreamSynchronize(strm));
                    auto t1 = std::chrono::steady_clock::now();
                    if (it >= 0)
                        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
                float expect = 0;
                for (int q = 0; q < ndev; ++q) expect += (float)q + 0.5f;
                std::vector<float> hh(count);
                int sanity = 1;
                CK(hipMemcpy(hh.data(), out, bytes, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < count; i += count / 8 + 1)
                    if (hh[i] != expect) { sanity = 0; break; }
                report("rcclmp", shape, kib, iters, us, sanity);
            }
            NK(ncclCommDestroy(comm));
            _exit(0);
        }
        kids.push_back(pid);
    }
    int rc = 0;
    for (pid_t p : kids) {
        int st = 0;
        waitpid(p, &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            std::fprintf(stderr, "rcclmp pid %d status %d (signal=%d)\n",
                         (int)p, st, WIFSIGNALED(st) ? WTERMSIG(st) : 0);
            rc = 1;
        }
    }
    return rc;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);  // hang location survives a SIGTERM (pipe would
                                          // block-buffer, and a killed run tells you NOTHING)
    std::string modes = argc > 1 ? argv[1] : "kern,bounce,peer,rccl";
    std::printf("probe-collective: P6.0 small-collective latency (design input, not a speed claim)\n");
    std::printf("modes=%s  sizes_KiB:", modes.c_str());
    for (int i = 0; i < NSIZES; ++i) std::printf(" %zu", SIZES_KIB[i]);
    std::printf("\n");

    int ndev = 0;
    CK(hipGetDeviceCount(&ndev));
    if (ndev < 2) { std::fprintf(stderr, "need >= 2 GPUs, found %d\n", ndev); return 1; }

    // rcclmp forks rank children: dispatch HERE, before ANY call that creates a HIP context
    // in this process (a child inheriting a parent-initialized ROCm context is undefined).
    if (modes.find("rcclmp") != std::string::npos) {
        std::vector<int> mdevs;
        if (argc > 2) {
            size_t pos = 0;
            std::string sl = argv[2];
            while (pos < sl.size()) {
                size_t c = sl.find(',', pos);
                mdevs.push_back(std::atoi(sl.substr(pos, c - pos).c_str()));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        } else {
            for (int i = 0; i < ndev; ++i) mdevs.push_back(i);
        }
        if ((int)mdevs.size() < 2) { std::fprintf(stderr, "rcclmp: need >= 2 devices\n"); return 1; }
        return run_rcclmp((int)mdevs.size(), mdevs.data());
    }
    for (int i = 0; i < ndev; ++i) {
        hipDeviceProp_t p;
        CK(hipGetDeviceProperties(&p, i));
        size_t free_b = 0, total_b = 0;
        CK(hipSetDevice(i));
        CK(hipMemGetInfo(&free_b, &total_b));
        std::printf("dev %d: %s  vram %.1f/%.1f GiB free/total  pci %04x:%02x:%02x\n",
                    i, p.name, free_b / 1073741824.0, total_b / 1073741824.0,
                    p.pciDomainID, p.pciBusID, p.pciDeviceID);
    }

    // enable peer access on every pair, keep it enabled (probe-p2p proved mapping works)
    for (int i = 0; i < ndev; ++i) {
        hipDevice_t di;
        CK(hipDeviceGet(&di, i));
        for (int j = 0; j < ndev; ++j) {
            if (i == j) continue;
            CK(hipSetDevice(j));
            hipError_t en = hipDeviceEnablePeerAccess(di, 0);
            if (en != hipSuccess && en != hipErrorPeerAccessAlreadyEnabled)
                std::fprintf(stderr, "warn: peer %d->%d enable failed: %s (peer mode will show MISMATCH)\n",
                             i, j, hipGetErrorString(en));
        }
    }

    if (modes.find("kern") != std::string::npos) run_kern();
    if (modes.find("bounce") != std::string::npos) {
        int d2[2] = {0, 1};
        run_bounce(2, d2);
        if (ndev >= 4) { int d4[4] = {0, 1, 2, 3}; run_bounce(4, d4); }
    }
    if (modes.find("peer") != std::string::npos) {
        int d2a[2] = {0, 1};  // same PCIe complex (P5.1: peer 27 GiB/s)
        std::printf("MARK peer 2way 0,1 start\n");
        run_peer(2, d2a);
        if (ndev >= 3) { int d2b[2] = {1, 2};
                         std::printf("MARK peer 2way 1,2 (cross-complex) start\n");
                         run_peer(2, d2b); }
        if (ndev >= 4) { int d4[4] = {0, 1, 2, 3};
                         std::printf("MARK peer 4way start\n");
                         run_peer(4, d4); }
    }
    if (modes.find("rccl") != std::string::npos) {
        int d2[2] = {0, 1};
        std::printf("MARK rccl 2way start\n");
        run_rccl(2, d2);
        if (ndev >= 4) { int d4[4] = {0, 1, 2, 3};
                         std::printf("MARK rccl 4way start\n");
                         run_rccl(4, d4); }
    }
    std::printf("probe-collective: done, exit 0\n");
    return 0;
}
