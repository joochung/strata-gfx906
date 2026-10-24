// tools/gfx906/device_split.hpp - P5.2: the opt-in device plumbing (flag values + stage math) for the PORTED engines,
// glm_generate and ds41_generate.  Header-only and free of ggml/HIP includes on purpose, so the standalone gate
// tools/gfx906/test_device_split.cpp compiles it with plain g++ in a second, with no CUDA, HIP or GPU present.
//
// WHY THIS IS OPT-IN (AGENTS.md: a new feature is opt-in and the default path stays byte-identical):
// without --gpus, nothing in this file runs.  Each engine's parse() calls split_args() ONLY when it sees --gpus, and
// the engine's single-device block is reached unchanged in every other case.  The stage count defaults to 1 so an
// existing call site that asks for it behaves as it does today.
//
// VOCABULARY, copied from the native engine rather than invented (src/program/generate.cpp:2130-2131 and
// :2379-2438, docs/MULTI_GPU.md:38-41 and :99-101):
//   --gpus D1,D2,..      the cards, as the runtime numbers them (setup.py's --gpus; the first is the main one).
//                        N names = N stages, so this is the one flag a caller has to type.
//   --layer-split K[,K2..]   the FIRST LAYER of each later stage, one rising number per stage after the first -
//                        NOT a count of layers per card.  Default: equal layer chunks (auto).
//   --split-device D1,D2..   the LATER stages' devices; stage 0 is always --gpus' first card.  Default: the next
//                        visible ones after it.  "0" with one K puts both stages on that one card - the bit-exact
//                        check of the hand-off, not a speed mode (docs/MULTI_GPU.md:100-101).
//   --split-skip-if-fits   accepted, and ignored: the ported engines have no expert profile to price a fit against,
//                        so there is nothing to decide the "fits" question with.  Accepted so a native-engine config
//                        line can be handed to a ported engine without the unknown-argument exit.
//
// WHAT THIS DOES NOT DO (the rest of P5, deliberately): it creates no backend, calls no CUDA/HIP, moves no tensor,
// and relays nothing.  Per-stage backend placement (docs/gfx906/P5-MULTI-GPU-DESIGN.md section 1) and the per-device pinned
// registrations (section 4) are the follow-on tasks, as is the cross-device activation relay.

#if !defined(STRATA_GFX906_DEVICE_SPLIT_HPP)
#define STRATA_GFX906_DEVICE_SPLIT_HPP

#include <cstdint>
#include <cstdlib>   // std::atoi in split_ints (the header stands alone: the gate includes only this)
#include <string>
#include <vector>

namespace strata {
namespace gfx906 {

/// The stage caps of P5 for now: the native engine's SplitDrive::kMax (src/program/generate.cpp:1084).
constexpr int kSplitMaxStages = 8;

/// One parsed --gpus/--layer-split/--split-device request plus the stage math derived from it.
/// A default-constructed Split is today's single-device run: one stage, one device (0), every layer.
struct Split {
    int n_stage = 1;                                       ///< 1 = no split (the byte-identical default)
    std::vector<int> dev;                                  ///< dev[i] = the device stage i runs on; size n_stage
    std::vector<int64_t> end;                              ///< end[i] = one past the last layer of stage i (layer count once planned)
    bool auto_split = true;                                ///< --layer-split absent/short: chunks placed equally
    bool same_device = false;                              ///< every stage on one device: the bit-exact hand-off check
    std::string gpus, layer_split, split_device;           ///< the raw flag values as given

    int64_t begin(int i) const { return i <= 0 ? 0 : end[(size_t) i - 1]; }
    int stage_of(int64_t layer) const {
        int st = 0;
        while (st + 1 < n_stage && layer >= end[(size_t) st]) ++st;
        return st;
    }
    int dev_of(int64_t layer) const { return dev[(size_t) stage_of(layer)]; }
    int64_t stage_layers(int i) const { return end[(size_t) i] - begin(i); }
};

/// "D1,D2,.." as integers, the native engine's own ints() lambda (src/program/generate.cpp:2394-2405): digits only,
/// empty fields are an error, and an empty list is an error.  Trailing comma allowed there, so allowed here.
inline bool split_ints(const std::string & s, std::vector<int> & out) {
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        const std::string t = s.substr(a, b - a);
        if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
        out.push_back(std::atoi(t.c_str()));
        a = b + 1;
    }
    return !out.empty();
}

/// The stage math, given the parsed flags and the model's layer count.  Pure: no device query, so it is testable
/// without a GPU.  Fills s.dev / s.end / s.auto_split / s.same_device and returns false with the reason in err.
///
/// Defaults mirror the native engine: --layer-split unset = equal chunks (auto), --split-device unset = the next
/// visible devices after the first, and one K with --split-device 0 = both stages on that card (same_device).
inline bool split_plan(Split & s, int64_t n_layer, std::string & err) {
    std::vector<int> gpus;
    if (!split_ints(s.gpus, gpus)) { err = "--gpus needs device numbers, e.g. --gpus 0,1 (got \"" + s.gpus + "\")"; return false; }
    // A card named in --gpus is a stage on that card: "--gpus 1,1" names two stages on one card, the bit-exact
    // hand-off check with no extra hardware (what the native engine reaches through
    // --layer-split K --split-device 0, docs/MULTI_GPU.md:100-101).  "--gpus 0" names one stage, which is today's
    // single-device run with the split flags off.
    if (n_layer < 1) { err = "the model reports " + std::to_string((long long) n_layer) + " layers"; return false; }
    const int n_stage = (int) gpus.size();
    if (n_stage < 1) { err = "--gpus names no device"; return false; }
    if (n_stage > kSplitMaxStages) {
        err = "--gpus names " + std::to_string(n_stage) + " devices; at most " + std::to_string(kSplitMaxStages) +
              " stages are supported";
        return false;
    }

    // --layer-split: "auto" / empty = equal chunks, else the first layer of each later stage, rising from 2
    std::vector<int64_t> at;
    s.auto_split = s.layer_split.empty() || s.layer_split == "auto";
    if (n_stage == 1 && !s.auto_split) {
        err = "--gpus 0 names one stage, so there is nothing to split: drop --layer-split, or name one card per stage";
        return false;
    }
    if (!s.auto_split) {
        std::vector<int> raw;
        if (!split_ints(s.layer_split, raw)) {
            err = "--layer-split wants K[,K2,..] or auto; K is the FIRST LAYER of each later stage, not a count of "
                  "layers per card - \"24,36,42\" for 4 stages, not \"24,12,6,6\" (got \"" + s.layer_split + "\")";
            return false;
        }
        for (const int v : raw) at.push_back((int64_t) v);
        for (size_t i = 0; i < at.size(); ++i)
            if (at[i] < 2 || (i && at[i] <= at[i - 1])) {
                err = "--layer-split must rise from 2 (the first layer of each later stage): got \"" + s.layer_split + "\"";
                return false;
            }
        if (at.size() + 1 != (size_t) n_stage) {
            err = "--layer-split gives " + std::to_string(at.size()) + " split point(s) for " + std::to_string(n_stage) +
                  " stage(s) from --gpus; it needs one rising number per stage after the first ("
                  + std::to_string(n_stage - 1) + " here)";
            return false;
        }
    }
    for (const int64_t k : at)
        if (k >= n_layer) {
            err = "--layer-split: layer " + std::to_string((long long) k) + " is past the last (" +
                  std::to_string((long long) n_layer - 1) + ")";
            return false;
        }

    // The stage devices: stage 0 is --gpus[0]; the later ones are --split-device, or the next --gpus entries.
    s.dev.assign((size_t) n_stage, 0);
    for (int i = 0; i < n_stage; ++i) s.dev[(size_t) i] = gpus[(size_t) i];   // stage 0 = --gpus[0], the main card
    if (!s.split_device.empty()) {
        if (n_stage == 1) {
            err = "--gpus 0 names one stage, so --split-device has no later stage to place";
            return false;
        }
        std::vector<int> d;
        if (!split_ints(s.split_device, d)) { err = "--split-device wants device numbers (got \"" + s.split_device + "\")"; return false; }
        if (d.size() + 1 != (size_t) n_stage) {
            err = "--split-device gives " + std::to_string(d.size()) + " device(s) for " + std::to_string(n_stage) +
                  " stages; one per stage after the first (" + std::to_string(n_stage - 1) + " here)";
            return false;
        }
        for (size_t i = 0; i < d.size(); ++i) s.dev[(size_t) i + 1] = d[i];
    }
    s.same_device = true;
    for (size_t i = 1; i < s.dev.size(); ++i) {
        if (s.dev[i] != s.dev[0]) s.same_device = false;
        // The native engine wants the later stages on DISTINCT devices, with the one exception above (all of them
        // equal to the first stage's device: the hand-off A/B).  Anything else is a typo.
        for (size_t j = 0; j < i; ++j)
            if (s.dev[i] == s.dev[j] && s.dev[i] != s.dev[0]) {
                err = "--split-device lists device " + std::to_string(s.dev[i]) + " for two stages (stages must be on "
                      "distinct devices, or all of them on the first stage's device for the bit-exact check)";
                return false;
            }
    }

    // The layer ranges: explicit split points, or equal chunks (each stage at least one layer; with fewer layers
    // than stages there is no split to make, which is the same answer the native engine gives for one visible GPU).
    s.end.assign((size_t) n_stage, n_layer);
    if (s.auto_split) {
        if (n_stage > (int) n_layer) {
            s.n_stage = 1;
            s.dev.assign(1, gpus[0]);
            s.end.assign(1, n_layer);
            s.same_device = true;
            return true;
        }
        for (int i = 0; i < n_stage - 1; ++i) s.end[(size_t) i] = n_layer * (i + 1) / n_stage;
    } else {
        for (int i = 0; i < n_stage - 1; ++i) s.end[(size_t) i] = at[(size_t) i];
    }
    s.n_stage = n_stage;
    return true;
}

/// The flags this plumbing knows, taken straight from argv.  Returns false with the reason in err on a bad value.
/// Called ONLY when argv contains --gpus, so the absence of that flag leaves the engine's own parse loop, and every
/// default, exactly as it is today.
inline bool split_args(Split & s, int argc, char ** argv, std::string & err) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { err = k + " needs a value"; return std::string(); }
            return argv[++i];
        };
        if (k == "--gpus") s.gpus = next();
        else if (k == "--layer-split") s.layer_split = next();
        else if (k == "--split-device") s.split_device = next();
        else if (k == "--split-skip-if-fits") { /* accepted, ignored: no expert profile to price a fit against */ }
    }
    if (!err.empty()) return false;
    if (s.gpus.empty()) { err = "no --gpus on the command line"; return false; }
    // A --gpus list must name at least one device: split_ints rejects junk and an empty list here as well, so a
    // mistyped value fails at the parse, before the engine loads anything.
    std::vector<int> gpus;
    if (!split_ints(s.gpus, gpus)) { err = "--gpus needs device numbers, e.g. --gpus 0,1 (got \"" + s.gpus + "\")"; return false; }
    return true;
}

/// One log line naming the plan, in the native engine's shape (generate.cpp:2536-2537, MULTI_GPU.md:186).
inline std::string split_describe(const Split & s) {
    std::string r = "layer split: " + std::to_string(s.n_stage) + " stage(s) on device";
    r += s.n_stage > 1 ? "s " : " ";
    for (int i = 0; i < s.n_stage; ++i) {
        r += (i ? ", " : "") + std::to_string(s.dev[(size_t) i]) + " layers " + std::to_string((long long) s.begin(i)) +
             "-" + std::to_string((long long) (s.end[(size_t) i] - 1));
    }
    if (s.same_device && s.n_stage > 1) r += " (all on one device: the bit-exact hand-off check, not a speed mode)";
    r += s.auto_split ? " (equal chunks)" : " (--layer-split)";
    return r;
}

}  // namespace gfx906
}  // namespace strata

#endif  // STRATA_GFX906_DEVICE_SPLIT_HPP
