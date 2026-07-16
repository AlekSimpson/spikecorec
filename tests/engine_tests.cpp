#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <memory>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/topologies.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

namespace {

bool approx(f32 first, f32 second, f32 epsilon = 1e-3f) {
    return std::fabs(first - second) <= epsilon * (1.0f + std::fabs(second));
}

void seed_never_spiked(SpikeEngine &engine, s64 sentinel = -1000) {
    s64 *last_spiked = engine.last_spiked.get_contents();
    for (s64 index = 0; index < engine.neuron_count; ++index) last_spiked[index] = sentinel;
}

void configure_deterministic(SpikeEngine &engine, f32 weight = 2.0f) {
    engine.spike_threshold = 1.0f;
    engine.spike_period = 1;
    engine.learning_rate = 0.0f;
    engine.use_constant_weight = true;
    engine.weights.set_constant_weight(weight);
    seed_never_spiked(engine);
}

} // namespace

// ── backend / types ────────────────────────────────────────────────────────────

TEST(Backend, types_layout) {
    EXPECT_EQ(sizeof(u8), 1u);
    EXPECT_EQ(sizeof(u16), 2u);
    EXPECT_EQ(sizeof(u32), 4u);
    EXPECT_EQ(sizeof(u64), 8u);
    EXPECT_EQ(sizeof(s32), 4u);
    EXPECT_EQ(sizeof(s64), 8u);
    EXPECT_EQ(sizeof(f32), 4u);
    EXPECT_EQ(sizeof(f64), 8u);
    EXPECT_EQ(sizeof(spikecorec::float4), 16u);
    EXPECT_EQ(alignof(spikecorec::float4), 16u);
}

TEST(Backend, gpu_pointer_alloc) {
    const usize neuron_count = 32;
    GpuPointer<f32> buffer = allocate<f32>(neuron_count * sizeof(f32));
    f32 *data = buffer.get_contents();
    ASSERT_NE(data, nullptr);

    for (usize index = 0; index < neuron_count; ++index) data[index] = (f32)index * 1.5f;
    for (usize index = 0; index < neuron_count; ++index) EXPECT_EQ(data[index], (f32)index * 1.5f);

    GpuPointer<f32> moved = std::move(buffer);
    EXPECT_EQ(moved.get_contents(), data);
    EXPECT_EQ(buffer.pointer, nullptr);

    deallocate(std::move(moved));
}

// ── construction / lifecycle ──────────────────────────────────────────────────

TEST(SpikeEngine, validate_construction) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4});

    EXPECT_EQ(engine.neuron_count, 16);
    EXPECT_TRUE(engine.alive);
    engine.shutdown();
    EXPECT_FALSE(engine.alive);
}

class SpikeEngineTest : public ::testing::Test {
protected:
    vector<vector<s32>> network;
    unique_ptr<SpikeEngine> engine;

    void SetUp() override {
        network = square_torus(4);
        const vector<s64> shape = {4, 4};
        engine = make_unique<SpikeEngine>(&network, shape, /*rank=*/4);
        engine->set_input_neurons({0, 1, 2});
    }

    void TearDown() override {
        if (engine) engine->shutdown();
    }
};

TEST_F(SpikeEngineTest, step_loop) {
    EXPECT_EQ(engine->input_neuron_count, 3);

    bool moved_from_resting = false;
    for (s64 tick = 0; tick < 10; ++tick) {
        engine->step_simulation({2.0f, 2.0f, 2.0f}, tick);

        const f32 *membrane_potentials = engine->membrane_potentials.get_contents();
        for (s64 index = 0; index < engine->neuron_count; ++index) {
            EXPECT_TRUE(std::isfinite(membrane_potentials[index]));
            if (std::fabs(membrane_potentials[index] - engine->resting_membrane_potential) > 1e-6f)
                moved_from_resting = true;
        }

        s32 active_count = engine->active_neuron_count.get_contents()[0];
        EXPECT_GE(active_count, 0);
        EXPECT_LE(active_count, (s32)engine->neuron_count);
        const s32 *active_indices = engine->active_neuron_indices.get_contents();
        for (s32 index = 0; index < active_count; ++index) {
            EXPECT_GE(active_indices[index], 0);
            EXPECT_LT(active_indices[index], (s32)engine->neuron_count);
        }
    }
    EXPECT_TRUE(moved_from_resting);
}

TEST_F(SpikeEngineTest, reset_state) {
    for (s64 tick = 0; tick < 10; ++tick)
        engine->step_simulation({2.0f, 2.0f, 2.0f}, tick);

    engine->reset_state();

    const f32 *membrane_potentials = engine->membrane_potentials.get_contents();
    for (s64 index = 0; index < engine->neuron_count; ++index)
        EXPECT_EQ(membrane_potentials[index], engine->resting_membrane_potential);
    EXPECT_EQ(engine->active_neuron_count.get_contents()[0], 0);
}

TEST_F(SpikeEngineTest, reservoir_features) {
    for (s64 tick = 0; tick < 5; ++tick)
        engine->step_simulation({2.0f, 2.0f, 2.0f}, tick);

    s64 feature_count = 2 * engine->neuron_count + 1;
    GpuPointer<f32> output = allocate<f32>((usize)feature_count * sizeof(f32));
    f32 *raw = output.get_contents();

    GpuPointer<f32> borrowed;
    borrowed.pointer = output.pointer;
    engine->get_reservoir_features_vector(5, /*spike_tau=*/10.0f, /*voltage_scale=*/1.0f, std::move(borrowed));

    for (s64 index = 0; index < feature_count; ++index)
        EXPECT_TRUE(std::isfinite(raw[index]));
    EXPECT_EQ(raw[feature_count - 1], 1.0f);

    deallocate(std::move(output));
}

TEST_F(SpikeEngineTest, merge_input_neurons) {
    engine->set_input_neurons({0});

    s64 tick = 0;
    vector<s64> override_neurons = {5, 9};
    engine->step_simulation({2.0f}, tick, override_neurons);

    const s64 *last_tick_updated = engine->last_tick_updated.get_contents();
    EXPECT_EQ(last_tick_updated[5], tick);
    EXPECT_EQ(last_tick_updated[9], tick);
}

// ── bifurcation / scaling ─────────────────────────────────────────────────────

TEST(SpikeEngine, estimate_bifurcation_weight) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/8);
    for (s32 period : {1, 2, 5}) {
        auto [weight_accum, weight_instant] = engine.estimate_bifurcation_weight(period);
        f32 decay_factor = std::pow(1.0f - engine.decay_rate, (f32)period);
        f32 expected_accum = (engine.spike_threshold - engine.resting_membrane_potential) * (1.0f - decay_factor);
        f32 expected_instant = engine.spike_threshold - engine.resting_membrane_potential;
        EXPECT_TRUE(approx(weight_accum, expected_accum, 1e-5f));
        EXPECT_TRUE(approx(weight_instant, expected_instant, 1e-5f));
    }
    engine.shutdown();
}

TEST(SpikeEngine, scale_uniform_near_bifurcation) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/8);

    f32 target = 0.0f, weight_accum = 0.0f, weight_instant = 0.0f;
    engine.scale_uniform_weights_near_bifurcation(&target, &weight_accum, &weight_instant,
                                                  /*input_period=*/1, /*scale=*/1.2f,
                                                  /*freeze_learning=*/true);
    auto [expected_accum, expected_instant] = engine.estimate_bifurcation_weight(1);
    EXPECT_TRUE(approx(weight_accum, expected_accum, 1e-5f));
    EXPECT_TRUE(approx(target, expected_accum * 1.2f, 1e-5f));
    EXPECT_EQ(engine.learning_rate, 0.0f);
    EXPECT_TRUE(engine.use_constant_weight);
    EXPECT_TRUE(approx(engine.weights.constant_weight, target, 1e-5f));
    EXPECT_TRUE(approx(engine.weights.get(0, 1), target, 1e-3f));

    engine.shutdown();
}

TEST(SpikeEngine, scale_randomized_near_bifurcation) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/8);

    ScaledReservoirResult result =
        engine.scale_randomized_weights_near_bifurcation(/*input_period=*/1, /*scale=*/1.2f,
                                                         /*freeze_learning=*/true);

    f32 expected_target = std::fabs(result.w_accum * 1.2f);
    EXPECT_TRUE(approx(result.weight_scale_result.target_root_mean_square, expected_target, 1e-5f));
    EXPECT_TRUE(approx(result.weight_scale_result.after.root_mean_square, expected_target, 1e-2f));
    EXPECT_FALSE(engine.use_constant_weight);
    EXPECT_EQ(engine.learning_rate, 0.0f);

    engine.shutdown();
}

// ── setup_lifetime ────────────────────────────────────────────────────────────

TEST(SpikeEngine, setup_lifetime) {
    auto network = square_torus(4);

    {
        SpikeEngine engine(&network, {4, 4}, /*rank=*/8);
        engine.setup_lifetime(/*lifetime=*/10, /*allocate_logs=*/true);
        EXPECT_EQ(engine.lifetime, 10);
        EXPECT_NE(engine.cell_state_logs, nullptr);
        engine.shutdown();
    }
    {
        SpikeEngine engine(&network, {4, 4}, /*rank=*/8);
        EXPECT_THROW(
            engine.setup_lifetime(/*lifetime=*/1000, /*allocate_logs=*/true, /*max_log_bytes=*/16),
            std::runtime_error
        );
        engine.shutdown();
    }
    {
        SpikeEngine engine(&network, {4, 4}, /*rank=*/8);
        engine.setup_lifetime(/*lifetime=*/10, /*allocate_logs=*/false);
        EXPECT_EQ(engine.cell_state_logs, nullptr);
        engine.shutdown();
    }
}

// ── guards ────────────────────────────────────────────────────────────────────

TEST(SpikeEngine, input_and_step_guards) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/8);

    engine.set_input_neurons({});
    EXPECT_EQ(engine.input_neuron_count, 0);
    engine.set_input_neurons({0, 1, 2});
    EXPECT_EQ(engine.input_neuron_count, 3);

    EXPECT_THROW(engine.step_simulation({}, /*tick=*/0), std::runtime_error);

    engine.shutdown();
}

TEST(SpikeEngine, reset_state_generations) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/8);
    engine.set_input_neurons({0, 1, 2});
    for (s64 tick = 0; tick < 5; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

    engine.reset_state();
    const s32 *active_generation = engine.active_generation.get_contents();
    const s64 *last_spiked = engine.last_spiked.get_contents();
    for (s64 index = 0; index < engine.neuron_count; ++index) {
        EXPECT_EQ(active_generation[index], -1);
        EXPECT_EQ(last_spiked[index], 0);
    }

    engine.shutdown();
}

TEST(SpikeEngine, reservoir_features_guard) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/8);
    engine.set_input_neurons({0, 1, 2});
    engine.step_simulation({2.0f, 2.0f, 2.0f}, 0);

    s64 feature_count = 2 * engine.neuron_count + 1;
    GpuPointer<f32> output = allocate<f32>((usize)feature_count * sizeof(f32));
    f32 *raw = output.get_contents();
    for (s64 index = 0; index < feature_count; ++index) raw[index] = -12345.0f;

    GpuPointer<f32> borrowed;
    borrowed.pointer = output.pointer;
    engine.get_reservoir_features_vector(1, /*spike_tau=*/-1.0f, /*voltage_scale=*/1.0f, std::move(borrowed));
    for (s64 index = 0; index < feature_count; ++index)
        EXPECT_EQ(raw[index], -12345.0f);

    deallocate(std::move(output));
    engine.shutdown();
}

// ── step_simulation paths ─────────────────────────────────────────────────────

TEST(SpikeEngine, step_simulation_decay_path) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/8);
    engine.set_input_neurons({0, 1, 2});

    const s64 last_tick = 5;
    for (s64 tick = 0; tick <= last_tick; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick, /*override_input_neurons=*/{},
                               /*decay_all_neurons=*/true);

    const f32 *membrane_potentials = engine.membrane_potentials.get_contents();
    const s64 *last_updated = engine.last_tick_updated.get_contents();
    for (s64 index = 0; index < engine.neuron_count; ++index) {
        EXPECT_TRUE(std::isfinite(membrane_potentials[index]));
        EXPECT_EQ(last_updated[index], last_tick);
    }

    engine.shutdown();
}

// ── deterministic spike propagation ──────────────────────────────────────────

TEST(SpikeEngine, spike_single_hop) {
    vector<vector<s32>> network = {{1}, {}};
    SpikeEngine engine(&network, {2, 1}, /*rank=*/4);
    configure_deterministic(engine, /*weight=*/2.0f);
    engine.set_input_neurons({0});

    engine.step_simulation({2.0f}, /*tick=*/0, /*override_input_neurons=*/{0});

    const s64 *last_spiked = engine.last_spiked.get_contents();
    EXPECT_EQ(last_spiked[0], 0);

    const f32 *network_inputs = engine.network_inputs.get_contents();
    EXPECT_TRUE(approx(network_inputs[1], 2.0f));

    s32 active_count = engine.active_neuron_count.get_contents()[0];
    const s32 *active = engine.active_neuron_indices.get_contents();
    bool neighbor_scheduled = false;
    for (s32 index = 0; index < active_count; ++index)
        if (active[index] == 1) neighbor_scheduled = true;
    EXPECT_TRUE(neighbor_scheduled);

    engine.shutdown();
}

TEST(SpikeEngine, spike_subthreshold_no_fire) {
    vector<vector<s32>> network = {{1}, {}};
    SpikeEngine engine(&network, {2, 1}, /*rank=*/4);
    configure_deterministic(engine, /*weight=*/2.0f);
    engine.set_input_neurons({0});

    engine.step_simulation({0.5f}, /*tick=*/0, /*override_input_neurons=*/{0});

    const s64 *last_spiked = engine.last_spiked.get_contents();
    for (s64 index = 0; index < engine.neuron_count; ++index)
        EXPECT_EQ(last_spiked[index], -1000);

    const f32 *network_inputs = engine.network_inputs.get_contents();
    for (s64 index = 0; index < engine.neuron_count; ++index)
        EXPECT_EQ(network_inputs[index], 0.0f);

    engine.shutdown();
}

TEST(SpikeEngine, spike_refractory_period) {
    vector<vector<s32>> network = {{1}, {}};
    SpikeEngine engine(&network, {2, 1}, /*rank=*/4);
    configure_deterministic(engine, /*weight=*/2.0f);
    engine.set_input_neurons({0});

    engine.step_simulation({2.0f}, /*tick=*/0, /*override_input_neurons=*/{0});
    EXPECT_EQ(engine.last_spiked.get_contents()[0], 0);

    engine.step_simulation({0.0f}, /*tick=*/1);
    const s64 *last_spiked = engine.last_spiked.get_contents();
    const f32 *membrane_potentials = engine.membrane_potentials.get_contents();
    EXPECT_EQ(last_spiked[0], 0);
    EXPECT_TRUE(approx(membrane_potentials[0], engine.resting_membrane_potential));

    engine.shutdown();
}

TEST(SpikeEngine, spike_fanout) {
    vector<vector<s32>> network = {{1, 2, 3}, {}, {}, {}};
    SpikeEngine engine(&network, {4, 1}, /*rank=*/4);
    configure_deterministic(engine, /*weight=*/2.0f);
    engine.set_input_neurons({0});

    engine.step_simulation({2.0f}, /*tick=*/0, /*override_input_neurons=*/{0});
    const f32 *network_inputs = engine.network_inputs.get_contents();
    for (s32 neighbor : {1, 2, 3})
        EXPECT_TRUE(approx(network_inputs[neighbor], 2.0f));

    engine.step_simulation({0.0f}, /*tick=*/1);
    const s64 *last_spiked = engine.last_spiked.get_contents();
    for (s32 neighbor : {1, 2, 3})
        EXPECT_EQ(last_spiked[neighbor], 1);
    EXPECT_EQ(last_spiked[0], 0);

    engine.shutdown();
}

TEST(SpikeEngine, spike_propagation_chain_end_to_end) {
    const s32 chain_length = 5;
    vector<vector<s32>> network(chain_length);
    for (s32 index = 0; index + 1 < chain_length; ++index) network[index] = {index + 1};
    network[chain_length - 1] = {};

    SpikeEngine engine(&network, {chain_length, 1}, /*rank=*/4);
    configure_deterministic(engine, /*weight=*/2.0f);
    engine.set_input_neurons({0});

    for (s64 tick = 0; tick < chain_length; ++tick) {
        vector<f32> input = (tick == 0) ? vector<f32>{2.0f} : vector<f32>{0.0f};
        vector<s64> override_neurons = (tick == 0) ? vector<s64>{0} : vector<s64>{};
        engine.step_simulation(input, tick, override_neurons);

        const f32 *membrane_potentials = engine.membrane_potentials.get_contents();
        for (s64 index = 0; index < engine.neuron_count; ++index)
            EXPECT_TRUE(std::isfinite(membrane_potentials[index]));
    }

    const s64 *last_spiked = engine.last_spiked.get_contents();
    for (s32 k = 0; k < chain_length; ++k)
        EXPECT_EQ(last_spiked[k], k);

    engine.shutdown();
}

TEST(SpikeEngine, plasticity_end_to_end) {
    auto run_edge_update = [](f32 learning_rate) -> pair<f32, f32> {
        vector<vector<s32>> network = {{1}, {}};
        SpikeEngine engine(&network, {2, 1}, /*rank=*/8);
        engine.spike_threshold = 1.0f;
        engine.spike_period = 1;
        engine.learning_rate = learning_rate;
        engine.use_constant_weight = false;
        engine.set_input_neurons({0});

        s64 *last_spiked = engine.last_spiked.get_contents();
        last_spiked[0] = -1000;
        last_spiked[1] = 1;

        f32 before = engine.weights.get(0, 1);
        engine.step_simulation({5.0f}, /*tick=*/3, /*override_input_neurons=*/{0});
        f32 after = engine.weights.get(0, 1);

        engine.shutdown();
        return {before, after};
    };

    auto [frozen_before, frozen_after] = run_edge_update(/*learning_rate=*/0.0f);
    EXPECT_TRUE(approx(frozen_before, frozen_after, 1e-6f));

    auto [live_before, live_after] = run_edge_update(/*learning_rate=*/0.5f);
    EXPECT_GT(std::fabs(live_after - live_before), 1e-6f);
}

// ── active-set optimization toggle ────────────────────────────────────────────

TEST(SpikeEngine, active_set_optimization_constructor_flag) {
    auto network = square_torus(4);

    {
        SpikeEngine engine_with_optimization(&network, {4, 4}, /*rank=*/4,
                                             /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                                             /*learning_rate=*/0.0f, /*plasticity_enabled=*/true,
                                             /*active_set_optimization_enabled=*/true);
        EXPECT_TRUE(engine_with_optimization.active_set_optimization_enabled);
        engine_with_optimization.shutdown();
    }
    {
        SpikeEngine engine_without_optimization(&network, {4, 4}, /*rank=*/4,
                                                /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                                                /*learning_rate=*/0.0f, /*plasticity_enabled=*/true,
                                                /*active_set_optimization_enabled=*/false);
        EXPECT_FALSE(engine_without_optimization.active_set_optimization_enabled);
        engine_without_optimization.shutdown();
    }
}

TEST(SpikeEngine, active_set_optimization_disabled_step_loop) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/4,
                       /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                       /*learning_rate=*/0.0f, /*plasticity_enabled=*/true,
                       /*active_set_optimization_enabled=*/false);
    engine.set_input_neurons({0, 1, 2});

    bool moved_from_resting = false;
    for (s64 tick = 0; tick < 10; ++tick) {
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

        const f32 *membrane_potentials = engine.membrane_potentials.get_contents();
        for (s64 index = 0; index < engine.neuron_count; ++index) {
            EXPECT_TRUE(std::isfinite(membrane_potentials[index]));
            if (std::fabs(membrane_potentials[index] - engine.resting_membrane_potential) > 1e-6f)
                moved_from_resting = true;
        }
    }
    EXPECT_TRUE(moved_from_resting);

    engine.shutdown();
}

TEST(SpikeEngine, active_set_optimization_disabled_active_count_not_tracked) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/4,
                       /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                       /*learning_rate=*/0.0f, /*plasticity_enabled=*/true,
                       /*active_set_optimization_enabled=*/false);
    engine.set_input_neurons({0, 1, 2});

    for (s64 tick = 0; tick < 5; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

    // The no-active-optimization kernel does not write to next_active_neuron_count,
    // so after each step the count (reset to 0 before gpu_step) stays 0.
    EXPECT_EQ(engine.active_neuron_count.get_contents()[0], 0);

    engine.shutdown();
}

TEST(SpikeEngine, active_set_optimization_disabled_spike_propagation_equivalence) {
    // Chain: 0 -> 1 -> 2 -> 3 -> 4. Fire neuron 0 at tick 0, expect each
    // subsequent neuron to fire one tick later. Verify both code paths produce
    // the same last_spiked vector.
    const s32 chain_length = 5;
    vector<vector<s32>> network(chain_length);
    for (s32 index = 0; index + 1 < chain_length; ++index) network[index] = {index + 1};
    network[chain_length - 1] = {};

    auto run_chain = [&](bool active_set_optimization_enabled) -> vector<s64> {
        SpikeEngine engine(&network, {chain_length, 1}, /*rank=*/4,
                           /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                           /*learning_rate=*/0.0f, /*plasticity_enabled=*/true,
                           active_set_optimization_enabled);
        configure_deterministic(engine, /*weight=*/2.0f);
        engine.set_input_neurons({0});

        for (s64 tick = 0; tick < chain_length; ++tick) {
            vector<f32> input_values = (tick == 0) ? vector<f32>{2.0f} : vector<f32>{0.0f};
            vector<s64> override_neurons = (tick == 0) ? vector<s64>{0} : vector<s64>{};
            engine.step_simulation(input_values, tick, override_neurons);
        }

        const s64 *last_spiked_ptr = engine.last_spiked.get_contents();
        vector<s64> result(last_spiked_ptr, last_spiked_ptr + chain_length);
        engine.shutdown();
        return result;
    };

    vector<s64> with_optimization    = run_chain(/*active_set_optimization_enabled=*/true);
    vector<s64> without_optimization = run_chain(/*active_set_optimization_enabled=*/false);

    ASSERT_EQ(with_optimization.size(), without_optimization.size());
    for (s32 index = 0; index < chain_length; ++index) {
        EXPECT_EQ(with_optimization[index], without_optimization[index])
            << "last_spiked mismatch at neuron " << index;
        EXPECT_EQ(with_optimization[index], (s64)index)
            << "expected neuron " << index << " to spike at tick " << index;
    }
}

// ── ticket #62 [F1]: active-set x nonlinear-dynamics correctness rule -- the LINEAR side ─────────
//
// The hardcoded engine's own leaky-integrate-and-fire dynamics (kernels.metal/kernels.cu's
// `apply_decay`) are exactly the "all of GLIF (linear)" case arch §0.5 describes -- untouched by
// this ticket. This test is a regression safety net demonstrating that side of the ticket's own
// acceptance criterion (the SEPARATE "nonlinear population never gets a closed-form multi-tick
// jump" side lives in master_kernel_tests.cpp's own
// `nonlinear_population_never_receives_a_closed_form_multi_tick_jump`, since no notion of "cell
// type"/IR exists on this hardcoded path): a neuron that falls out of the active set (no edges, so
// nothing ever scatters into its own `network_inputs`; subthreshold, so it never spikes and is
// never self-re-enqueued -- see kernels.metal's own `step`, whose re-enqueue code only runs past
// the spike branch) and is skipped for several ticks, then reactivated with no new stimulus, must
// land on EXACTLY the membrane potential produced by decaying it one tick at a time the whole way
// -- i.e. `apply_decay`'s closed-form multi-tick jump is mathematically equivalent to N
// single-tick decays, not an approximation of them. Reuses this file's own
// `active_set_optimization_disabled_spike_propagation_equivalence` precedent immediately above
// (a `run(active_set_optimization_enabled)` closure comparing the SAME two dispatch paths --
// `step`'s active-set-gated skip vs. `step_no_active_optimization`'s always-dispatch-every-neuron
// path), applied to membrane-potential decay instead of spike timing.

TEST(SpikeEngine, active_set_skip_then_revisit_matches_per_tick_decay_equivalence) {
    vector<vector<s32>> network = {{}, {}}; // two isolated neurons, no edges at all
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const s64 quiet_tick_count = 4; // ticks 1..quiet_tick_count: neuron 0 receives nothing at all
    const s64 revisit_tick = quiet_tick_count + 1;

    auto run = [&](bool active_set_optimization_enabled) -> f32 {
        SpikeEngine engine(&network, {2, 1}, /*rank=*/1, resting_mp, decay_rate,
                           /*learning_rate=*/0.0f, /*plasticity_enabled=*/false,
                           active_set_optimization_enabled);
        engine.spike_threshold = 1000.0f; // never spikes -- isolates pure decay from reset/emit
        // Both `step`/`step_no_active_optimization` also force membrane_potentials back to
        // resting_mp whenever `tick - last_spiked == spike_period` (an orthogonal "periodic forced
        // reset" quirk, see master_kernel_tests.cpp's own comment on it) -- last_spiked defaults to
        // 0, which would spuriously fire that branch at tick == spike_period even though this
        // neuron never actually spikes. seed_never_spiked pushes it far enough into the past that
        // this test's own short tick horizon can never coincide, isolating pure decay.
        seed_never_spiked(engine);
        engine.set_input_neurons({0});

        // Tick 0: a subthreshold external pulse forces neuron 0 into the active set (both dispatch
        // paths process it identically here).
        engine.step_simulation({0.5f}, /*tick=*/0, /*override_input_neurons=*/{0});

        // Ticks 1..quiet_tick_count: no external input, no override. With the optimization enabled,
        // neuron 0 was never re-enqueued after tick 0 (subthreshold, no edges), so `step` never
        // dispatches to it at all here -- it is genuinely skipped, not merely "processed with zero
        // input." With the optimization disabled, `step_no_active_optimization` keeps dispatching to
        // (and re-decaying) it every tick regardless.
        for (s64 tick = 1; tick <= quiet_tick_count; ++tick) {
            engine.step_simulation({0.0f}, tick);
        }

        // Revisit tick: force neuron 0 back into the active set (a no-op under
        // active_set_optimization_enabled=false, which never stopped dispatching to it) with no
        // additional external stimulus -- this is where the optimization-enabled path's own
        // closed-form apply_decay bridges the whole quiet_tick_count-tick gap in one shot.
        engine.step_simulation({0.0f}, revisit_tick, /*override_input_neurons=*/{0});

        f32 result = engine.membrane_potentials.get_contents()[0];
        engine.shutdown();
        return result;
    };

    f32 with_optimization = run(/*active_set_optimization_enabled=*/true);
    f32 without_optimization = run(/*active_set_optimization_enabled=*/false);

    EXPECT_NEAR(with_optimization, without_optimization, 1e-4f);

    // Sanity: this is a genuine decay-over-time comparison, not a vacuous match at some
    // already-settled fixed point.
    EXPECT_LT(with_optimization, 0.5f);
    EXPECT_GT(with_optimization, resting_mp);
}
