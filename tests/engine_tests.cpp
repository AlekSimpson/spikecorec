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
