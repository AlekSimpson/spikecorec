#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include <unordered_set>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/topologies.h"

using namespace std;
using namespace spikecorec;

namespace {

// ── topology smoke tests ─────────────────────────────────────────────────────

void test_square_torus() {
    auto net = square_torus(4);
    assert(net.size() == 16);
    for (s64 i = 0; i < (s64)net.size(); ++i) {
        assert(net[i].size() == 4);
        unordered_set<s32> seen;
        for (s32 child : net[i]) {
            assert(child >= 0 && child < 16);
            assert(child != (s32)i && "square_torus must not contain self-loops");
            assert(seen.insert(child).second && "square_torus must not contain duplicate edges");
        }
    }
    printf("  square_torus: ok\n");
}

void test_small_world_torus() {
    auto net = small_world_torus(4, /*random_fanout=*/4, /*seed=*/42);
    assert(net.size() == 16);
    for (s64 i = 0; i < (s64)net.size(); ++i) {
        assert(net[i].size() == 8); // 4 torus neighbors + 4 random shortcuts
        unordered_set<s32> seen;
        for (s32 child : net[i]) {
            assert(child >= 0 && child < 16);
            assert(child != (s32)i && "small_world_torus must not contain self-loops");
            assert(seen.insert(child).second && "small_world_torus must not contain duplicate edges");
        }
    }
    printf("  small_world_torus: ok\n");
}

void test_random_fixed_outdegree() {
    auto net = random_fixed_outdegree(4, /*fanout=*/6, /*seed=*/42);
    assert(net.size() == 16);
    for (s64 i = 0; i < (s64)net.size(); ++i) {
        assert(net[i].size() == 6);
        unordered_set<s32> seen;
        for (s32 child : net[i]) {
            assert(child >= 0 && child < 16);
            assert(child != (s32)i && "random_fixed_outdegree must not contain self-loops");
            assert(seen.insert(child).second && "random_fixed_outdegree must not contain duplicate edges");
        }
    }
    printf("  random_fixed_outdegree: ok\n");
}

// ── engine smoke tests ───────────────────────────────────────────────────────

void test_construction() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4});
    assert(engine.neuron_count == 16);
    assert(engine.is_alive());
    engine.shutdown();
    assert(!engine.is_alive());
    printf("  construction: ok\n");
}

void test_step_loop() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);

    engine.set_input_neurons({0, 1, 2});
    assert(engine.input_neuron_count == 3);

    bool moved_from_resting = false;
    for (s64 tick = 0; tick < 10; ++tick) {
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

        const f32 *mp = engine.membrane_potentials.get_contents();
        for (s64 i = 0; i < engine.neuron_count; ++i) {
            assert(std::isfinite(mp[i]));
            if (std::fabs(mp[i] - engine.resting_membrane_potential) > 1e-6f)
                moved_from_resting = true;
        }

        s32 active_count = engine.active_neuron_count.get_contents()[0];
        assert(active_count >= 0 && active_count <= engine.neuron_count);
        const s32 *active_indices = engine.active_neuron_indices.get_contents();
        for (s32 i = 0; i < active_count; ++i) {
            assert(active_indices[i] >= 0 && active_indices[i] < engine.neuron_count);
        }
    }
    assert(moved_from_resting && "membrane potentials should evolve away from resting_mp under nonzero input");

    engine.shutdown();
    printf("  step_loop: ok\n");
}

void test_reset_state() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);

    engine.set_input_neurons({0, 1, 2});
    for (s64 tick = 0; tick < 10; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

    engine.reset_state();

    const f32 *mp = engine.membrane_potentials.get_contents();
    for (s64 i = 0; i < engine.neuron_count; ++i)
        assert(mp[i] == engine.resting_membrane_potential);
    assert(engine.active_neuron_count.get_contents()[0] == 0);

    engine.shutdown();
    printf("  reset_state: ok\n");
}

void test_reservoir_features() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);

    engine.set_input_neurons({0, 1, 2});
    for (s64 tick = 0; tick < 5; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

    s64 feature_count = 2 * engine.neuron_count + 1;
    GpuPointer<f32> output = allocate<f32>((usize)feature_count * sizeof(f32));
    f32 *raw = output.get_contents();

    // get_reservoir_features_vector takes its GpuPointer by value but only ever
    // calls .get_contents() on it (engine.cpp) — it neither stores nor frees the
    // handle. GpuPointer is move-only, so hand it a borrowed duplicate of the raw
    // handle and keep `output` as the sole owner responsible for deallocate().
    GpuPointer<f32> borrowed;
#ifdef SPIKECOREC_CUDA
    borrowed.pointer = output.pointer;
#elif defined(SPIKECOREC_METAL)
    borrowed.buffer = output.buffer;
#endif
    engine.get_reservoir_features_vector(5, /*spike_tau=*/10.0f, /*voltage_scale=*/1.0f, std::move(borrowed));

    for (s64 i = 0; i < feature_count; ++i)
        assert(std::isfinite(raw[i]));
    assert(raw[feature_count - 1] == 1.0f && "bias slot must be 1.0");

    deallocate(std::move(output));

    engine.shutdown();
    printf("  reservoir_features: ok\n");
}

void test_merge_input_neurons() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);

    engine.set_input_neurons({0});

    // gpu_merge_input_neurons appends override neurons into the *current* active
    // set, which gpu_step then consumes (and replaces with the next-tick active
    // set via swap) within the same step_simulation call — so the override
    // neurons themselves won't necessarily be in the post-swap active set.
    // What's directly observable is that gpu_step actually visited them this
    // tick: every processed active neuron gets last_tick_updated[n] = tick.
    s64 tick = 0;
    vector<s64> override_neurons = {5, 9};
    engine.step_simulation({2.0f}, tick, override_neurons);

    const s64 *last_tick_updated = engine.last_tick_updated.get_contents();
    assert(last_tick_updated[5] == tick && "override neuron 5 must be merged into the active set and processed by gpu_step");
    assert(last_tick_updated[9] == tick && "override neuron 9 must be merged into the active set and processed by gpu_step");

    engine.shutdown();
    printf("  merge_input_neurons: ok\n");
}

} // namespace

int main() {
    static_assert(sizeof(spikecorec::f32) == 4);
    static_assert(sizeof(spikecorec::f64) == 8);

    initialize_gpu_context();

    printf("topology tests:\n");
    test_square_torus();
    test_small_world_torus();
    test_random_fixed_outdegree();

    printf("engine tests:\n");
    test_construction();
    test_step_loop();
    test_reset_state();
    test_reservoir_features();
    test_merge_input_neurons();

    printf("core tests passed\n");
    return 0;
}
