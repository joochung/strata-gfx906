// tools/gfx906/test_device_split.cpp - P5.2 gate: the opt-in --gpus/--layer-split/--split-device plumbing of the
// ported engines (tools/gfx906/device_split.hpp), with no CUDA, no HIP, no ggml and no GPU: plain g++ runs it in a
// second.
//
//   g++ -std=c++17 -O0 -Wall -Wextra -Werror -I tools/gfx906 -o /tmp/test_device_split
//       tools/gfx906/test_device_split.cpp  &&  /tmp/test_device_split
//
// It checks the semantics the native engine ships (docs/MULTI_GPU.md, src/program/generate.cpp:2379-2438), the
// default that must stay byte-identical, and the errors a user hits.  It makes no runtime claim: nothing here
// touches a device, and per-stage backend placement plus the relay are later tasks.
#include "device_split.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::gfx906::Split;
using strata::gfx906::split_args;
using strata::gfx906::split_describe;
using strata::gfx906::split_plan;

// The counters the CHECK macros below use; test_relay_staging.cpp (included at the end of this file) shares them
// through the g_checks/g_fail declarations it makes there, so ONE binary reports both headers' checks.
int g_fail = 0, g_checks = 0;

#define CHECK(cond)                                                                                              \
    do {                                                                                                         \
        ++g_checks;                                                                                              \
        if (!(cond)) {                                                                                           \
            ++g_fail;                                                                                            \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                          \
        }                                                                                                        \
    } while (0)

#define CHECK_EQ(a, b)                                                                                           \
    do {                                                                                                         \
        ++g_checks;                                                                                              \
        if (!((a) == (b))) {                                                                                     \
            ++g_fail;                                                                                            \
            std::printf("FAIL %s:%d  %s == %s\n", __FILE__, __LINE__, #a, #b);                                   \
        }                                                                                                        \
    } while (0)

// A command line from a list of words, so a test reads like the flag it checks.  The words are copied into owned
// buffers: argv() hands out real char* the way a process's argv does.
struct Cli {
    std::vector<std::string> words;
    std::vector<char *> v;
    void build(const char * prog, std::initializer_list<const char *> args) {
        words.assign({prog});
        for (const char * a : args) words.emplace_back(a);
        for (auto & w : words) v.push_back(w.data());
    }
    int argc() const { return (int) v.size(); }
    char ** argv() { return v.data(); }
};

// Plan a split from flags; on a rejected plan returns false and puts the reason in err.  The Split the caller passes
// in must be the default one: a plan either fills it or leaves it as today's single-device run.
static bool plan_of(Split & s, std::initializer_list<const char *> args, int64_t n_layer, std::string & err,
                    const char * prog = "engine") {
    err.clear();
    Cli cli;
    cli.build(prog, args);
    if (!split_args(s, cli.argc(), cli.argv(), err)) return false;
    return split_plan(s, n_layer, err);
}

// The default path: no --gpus on the command line, so the plumbing is never consulted and a default Split is
// exactly today's single-device run (one stage, device 0, all layers, nothing marked as a split).
static void test_default_is_single_device() {
    Split d;
    CHECK_EQ(d.n_stage, 1);
    CHECK(d.dev.empty());
    CHECK(d.end.empty());
    CHECK(!d.same_device);
    CHECK(d.gpus.empty());
    // A CLI with no --gpus: split_args refuses, and the engine's own parse loop runs untouched.
    Split s;
    std::string err;
    CHECK(!plan_of(s, {"-m", "model.gguf", "--layer-split", "20"}, 40, err));
    CHECK(err.find("no --gpus") != std::string::npos);
    // stage_of / dev_of on the default Split answer like a one-stage plan once its range is known
    d.dev = {0};
    d.end = {39};
    d.n_stage = 1;
    CHECK_EQ(d.stage_of(0), 0);
    CHECK_EQ(d.stage_of(38), 0);
    CHECK_EQ(d.dev_of(38), 0);
}

// --gpus alone: N stages on the named cards, equal layer chunks, first-layer-of-each-later-stage boundaries.
static void test_default_equal_chunks() {
    Split s;
    std::string err;
    CHECK(plan_of(s, {"--gpus", "0,1,2,3"}, 40, err));
    CHECK_EQ(s.n_stage, 4);
    CHECK(s.dev == std::vector<int>({0, 1, 2, 3}));
    // equal chunks of 40 layers over 4 stages: 10, 20, 30, 40 (first layers 10, 20, 30 - not counts)
    CHECK(s.end == std::vector<int64_t>({10, 20, 30, 40}));
    CHECK(s.auto_split);
    CHECK(!s.same_device);
    CHECK_EQ(s.begin(1), 10);
    CHECK_EQ(s.stage_layers(3), 10);
    CHECK_EQ(s.stage_of(9), 0);
    CHECK_EQ(s.stage_of(10), 1);
    CHECK_EQ(s.stage_of(39), 3);
    CHECK_EQ(s.dev_of(10), 1);
    CHECK_EQ(s.dev_of(39), 3);

    // 3 stages over 48 layers: 16 / 32 / 48
    Split t;
    CHECK(plan_of(t, {"--gpus", "2,0,3"}, 48, err));
    CHECK_EQ(t.n_stage, 3);
    CHECK(t.dev == std::vector<int>({2, 0, 3}));
    CHECK(t.end == std::vector<int64_t>({16, 32, 48}));

    // uneven: 7 stages... no, 3 stages over 10 layers -> 3, 6, 10 (later stages get the remainder)
    Split u;
    CHECK(plan_of(u, {"--gpus", "0,1,2"}, 10, err));
    CHECK(u.end == std::vector<int64_t>({3, 6, 10}));
    CHECK_EQ(u.stage_layers(0), 3);
    CHECK_EQ(u.stage_layers(2), 4);
}

// --layer-split K[,K2,..] means the FIRST LAYER of each later stage, not layers per card.
static void test_explicit_layer_split() {
    Split s;
    std::string err;
    CHECK(plan_of(s, {"--gpus", "0,1,2,3", "--layer-split", "24,36,42"}, 48, err));
    CHECK_EQ(s.n_stage, 4);
    CHECK(s.end == std::vector<int64_t>({24, 36, 42, 48}));
    CHECK(!s.auto_split);
    CHECK_EQ(s.stage_layers(0), 24);
    CHECK_EQ(s.stage_layers(1), 12);
    CHECK_EQ(s.stage_layers(3), 6);
    CHECK_EQ(s.stage_of(23), 0);
    CHECK_EQ(s.stage_of(24), 1);
    CHECK_EQ(s.stage_of(47), 3);

    // one split point, two stages
    Split t;
    CHECK(plan_of(t, {"--gpus", "0,1", "--layer-split", "18"}, 40, err));
    CHECK(t.end == std::vector<int64_t>({18, 40}));

    // the per-card-count mistake the native engine's message calls out: counts are not first-layer indexes
    Split w;
    CHECK(!plan_of(w, {"--gpus", "0,1,2,3", "--layer-split", "24,12,6,6"}, 48, err));
    CHECK(err.find("rise from 2") != std::string::npos);
}

// The bit-exact hand-off check: stages on one card (--split-device 0 with one K, or a repeated card in --gpus).
static void test_same_device_handoff() {
    Split s;
    std::string err;
    CHECK(plan_of(s, {"--gpus", "0,1", "--layer-split", "20", "--split-device", "0"}, 40, err));
    CHECK_EQ(s.n_stage, 2);
    CHECK(s.dev == std::vector<int>({0, 0}));
    CHECK(s.same_device);
    CHECK(s.end == std::vector<int64_t>({20, 40}));

    // The same thing named the --gpus way: the same card twice is the hand-off check, no error.
    Split t;
    CHECK(plan_of(t, {"--gpus", "1,1", "--layer-split", "20"}, 40, err));
    CHECK(t.dev == std::vector<int>({1, 1}));
    CHECK(t.same_device);
}

// The plan step must fill what a default Split leaves empty, and a plan that fails must leave them empty.
static void test_plan_fills_defaults_and_leaves_them_on_error() {
    Split s;
    std::string err;
    CHECK(plan_of(s, {"--gpus", "0,1", "--layer-split", "20"}, 40, err));
    CHECK_EQ(s.n_stage, 2);
    CHECK(s.dev == std::vector<int>({0, 1}));       // default Split: dev empty -> RED until plan() fills it
    CHECK(s.end == std::vector<int64_t>({20, 40}));
    CHECK_EQ(s.begin(0), 0);
    CHECK_EQ(s.stage_layers(1), 20);

    Split bad;
    CHECK(!plan_of(bad, {"--gpus", "0,1", "--layer-split", "99"}, 40, err));
    CHECK_EQ(bad.n_stage, 1);                        // a refused plan leaves today's single-device defaults
    CHECK(bad.dev.empty());
    CHECK(bad.end.empty());
}

// Validation errors: a short --layer-split list, a non-monotonic split, a split past the last layer, a short
// --split-device list, junk values, too many stages, and a split with fewer stages than layers.
static void test_validation_errors() {
    Split s;
    std::string err;

    // too few split points for the stages --gpus names
    CHECK(!plan_of(s, {"--gpus", "0,1,2", "--layer-split", "20"}, 40, err));
    CHECK(err.find("1 split point") != std::string::npos);

    // too many split points
    CHECK(!plan_of(s, {"--gpus", "0,1", "--layer-split", "10,20,30"}, 40, err));
    CHECK(err.find("3 split point") != std::string::npos);

    // non-monotonic
    CHECK(!plan_of(s, {"--gpus", "0,1,2", "--layer-split", "30,20"}, 40, err));
    CHECK(err.find("rise from 2") != std::string::npos);
    CHECK(!plan_of(s, {"--gpus", "0,1,2", "--layer-split", "20,20"}, 40, err));
    CHECK(err.find("rise from 2") != std::string::npos);

    // K below 2 (a stage with no layers)
    CHECK(!plan_of(s, {"--gpus", "0,1", "--layer-split", "1"}, 40, err));
    CHECK(err.find("rise from 2") != std::string::npos);

    // a split index beyond the layer count (name the layer, like the native engine does at :4361)
    Split pl;
    CHECK(!plan_of(pl, {"--gpus", "0,1", "--layer-split", "40"}, 40, err));
    CHECK(err.find("past the last") != std::string::npos);
    CHECK(err.find("39") != std::string::npos);

    // junk in a list (caught at the flag read, so before any model load)
    Split j0;
    CHECK(!plan_of(j0, {"--gpus", "0,x"}, 40, err));
    CHECK(j0.gpus.empty() == false);   // the value was read; the plan step never ran
    Split j1;
    CHECK(!plan_of(j1, {"--gpus", "0,x"}, 40, err));
    CHECK(err.find("--gpus") != std::string::npos);
    Split j2;
    CHECK(!plan_of(j2, {"--gpus", "0,1", "--layer-split", "1o"}, 40, err));
    CHECK(err.find("FIRST LAYER") != std::string::npos);
    Split j3;
    CHECK(!plan_of(j3, {"--gpus", "0,,1"}, 40, err));

    // --split-device with the wrong number of devices
    Split sd;
    CHECK(!plan_of(sd, {"--gpus", "0,1,2", "--layer-split", "10,20", "--split-device", "1"}, 40, err));
    CHECK(err.find("--split-device") != std::string::npos);

    // a flag with no value at the end of the command line
    Split nv;
    CHECK(!plan_of(nv, {"--gpus"}, 40, err));
    CHECK(err.find("needs a value") != std::string::npos);

    // more stages than the stage cap (the native engine's SplitDrive::kMax is 8)
    Split cap;
    CHECK(!plan_of(cap, {"--gpus", "0,1,2,3,4,5,6,7,8"}, 48, err));
    CHECK(err.find("stages") != std::string::npos);

    // a device named twice in --gpus is the one-device hand-off check (the bit-exact A/B), not an error;
    // two LATER stages sharing one card that is not the first stage's card is a mistake, and says so
    Split dup;
    CHECK(!plan_of(dup, {"--gpus", "0,1,2", "--layer-split", "10,20", "--split-device", "2,2"}, 40, err));
    CHECK(err.find("two stages") != std::string::npos);
}

// --gpus with fewer cards than the model has layers: auto has no split to make and says so by keeping one stage.
static void test_more_stages_than_layers() {
    Split s;
    std::string err;
    CHECK(plan_of(s, {"--gpus", "0,1,2,3"}, 2, err));
    CHECK_EQ(s.n_stage, 1);
    CHECK(s.dev == std::vector<int>({0}));
    CHECK(s.end == std::vector<int64_t>({2}));
}

// The log line a maintainer sees, naming every stage's device and layer range.
static void test_describe() {
    Split s;
    std::string err;
    CHECK(plan_of(s, {"--gpus", "0,1", "--layer-split", "20"}, 40, err));
    const std::string line = split_describe(s);
    CHECK(line.find("0 layers 0-19") != std::string::npos);
    CHECK(line.find("1 layers 20-39") != std::string::npos);
    CHECK(line.find("bit-exact") == std::string::npos);

    Split t;
    CHECK(plan_of(t, {"--gpus", "0,1", "--layer-split", "20", "--split-device", "0"}, 40, err));
    CHECK(split_describe(t).find("bit-exact") != std::string::npos);

    // a single named card is today's run: one stage, and no hand-off claim in its line
    Split u;
    CHECK(plan_of(u, {"--gpus", "2"}, 40, err));
    CHECK_EQ(u.n_stage, 1);
    CHECK_EQ(u.dev_of(0), 2);
    CHECK(split_describe(u).find("bit-exact") == std::string::npos);
}

// P5.3 step 1: the stage-boundary relay's arithmetic (tools/gfx906/relay_staging.hpp).  Included rather than linked
// so the gate stays one g++ command with no objects and no makefile.
#include "test_relay_staging.cpp"

// P5.3b: the stage-boundary RULES of a staged run, stated against a model geometry (tools/gfx906/test_split_geometry.hpp).
void test_split_geometry_run();
#include "test_split_geometry.cpp"

int main() {
    test_default_is_single_device();
    test_default_equal_chunks();
    test_explicit_layer_split();
    test_same_device_handoff();
    test_plan_fills_defaults_and_leaves_them_on_error();
    test_validation_errors();
    test_more_stages_than_layers();
    test_describe();
    test_relay_staging_run();
    test_split_geometry_run();
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAILED" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
