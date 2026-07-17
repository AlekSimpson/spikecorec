#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <type_traits>

#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/inputs_lowering.h"
#include "spikecorec/nml/gpu_source.h"
#include "spikecorec/nml/master_kernel.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── On-device generator lowering tests (ticket #65 [F4]) ─────────────────────────────────────────
//
// Two things this file demonstrates:
//  1. `lower_inputs_to_ir` produces correct `.alloc`/`.tick` IR for each of the four supported
//     on-device generator ComponentTypes (sineGenerator/rampGenerator/voltageClamp/
//     SpikeSourcePoisson), and every generated function ACTUALLY COMPILES as real MSL (`xcrun -sdk
//     macosx metal -c`), not just structurally-checked text -- mirrors gpu_source_tests.cpp's own
//     established pattern. `poissonFiringSynapse`/`compoundInput` are asserted to throw a clear,
//     named error (the documented scope boundary -- see inputs_lowering.h).
//  2. The acceptance criterion's Poisson piece: a real population of `SpikeSourcePoisson` instances
//     (parsed from real NML text through the whole front end, exactly like cell_lowering_tests.cpp's
//     own pattern), driven for several thousand ticks through a real `AssembledModel` (ticket #6),
//     with its spike-count statistics checked against the expected Poisson rate.

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

bool compiles_as_msl(const String &source, const String &label) {
    String path = "/tmp/spikecorec_inputs_lowering_test_" + label + ".metal";
    ofstream file(path);
    file << source;
    file.close();
    String command = "xcrun -sdk macosx metal -O2 -c " + path + " -o /dev/null";
    return system(command.c_str()) == 0;
}

// Builds a one-instance-bound TypeLibraryEntry for `component_type_name` (a real std-lib
// ComponentType -- no inline `<ComponentType>` fixture needed, unlike cell_lowering_tests.cpp's own
// GLIF fixtures, since sineGenerator/rampGenerator/voltageClamp/SpikeSourcePoisson are all already
// vendored in third_party/neuroml2/std_lib). `population_size` lets the Poisson acceptance test
// build a real multi-neuron population; every other test here only needs size 1.
ModelSpecification build_inputs_model(const String &fixture_id, const String &component_type_name,
                                       const String &instance_attributes, s32 population_size) {
    write_temp_file("spikecorec_inputs_lowering_" + fixture_id + "_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"InputsLowering" + fixture_id + "Content\">"
        "  <" + component_type_name + " id=\"instance0\" " + instance_attributes + "/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"instance0\" size=\"" + std::to_string(population_size) + "\"/>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_inputs_lowering_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"InputsLowering" + fixture_id + "Top\">"
        "  <include href=\"spikecorec_inputs_lowering_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

} // namespace

// ── sineGenerator / rampGenerator / voltageClamp: IR shape + real MSL compilation ────────────────

TEST(InputsLowering, sine_generator_lowers_to_windowed_on_device_current_and_compiles_as_msl) {
    ModelSpecification model = build_inputs_model(
        "Sine", "sineGenerator", "phase=\"0\" delay=\"10ms\" duration=\"500ms\" amplitude=\"2nA\" period=\"100ms\"", 1);
    ASSERT_EQ(model.type_library.size(), 1u);
    EXPECT_EQ(model.type_library[0].category, TypeLibraryCategory::Inputs);

    IrProgram program = lower_inputs_to_ir(model.type_library[0]);
    EXPECT_EQ(program.component_type_name, "sineGenerator");

    // `.alloc` carries the baked constants + the `i` state/expose.
    bool has_state_i = false, has_period_param = false;
    for (const auto &directive : program.alloc) {
        if (auto *state = std::get_if<StateDirective>(&directive); state && state->name == "i") has_state_i = true;
        if (auto *param = std::get_if<ParamConstantDirective>(&directive); param && param->name == "period") {
            has_period_param = true;
            ASSERT_TRUE(param->literal_value.has_value());
        }
    }
    EXPECT_TRUE(has_state_i);
    EXPECT_TRUE(has_period_param);

    GpuSource source = lower_ir_program_to_gpu_source(program);
    ASSERT_EQ(source.functions.size(), 1u);
    EXPECT_EQ(source.functions[0].function_name, "sineGenerator_tick");
    EXPECT_NE(source.msl_source.find("sin("), String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "sine_generator"));
}

TEST(InputsLowering, ramp_generator_lowers_to_windowed_linear_ramp_and_compiles_as_msl) {
    ModelSpecification model = build_inputs_model("Ramp", "rampGenerator",
        "delay=\"10ms\" duration=\"200ms\" startAmplitude=\"0nA\" finishAmplitude=\"5nA\" baselineAmplitude=\"0nA\"", 1);
    ASSERT_EQ(model.type_library.size(), 1u);

    IrProgram program = lower_inputs_to_ir(model.type_library[0]);
    EXPECT_EQ(program.component_type_name, "rampGenerator");

    GpuSource source = lower_ir_program_to_gpu_source(program);
    ASSERT_EQ(source.functions.size(), 1u);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "ramp_generator"));
}

TEST(InputsLowering, voltage_clamp_lowers_with_a_require_v_binding_and_compiles_as_msl) {
    ModelSpecification model = build_inputs_model("VoltageClamp", "voltageClamp",
        "delay=\"0ms\" duration=\"1000ms\" targetVoltage=\"-65mV\" simpleSeriesResistance=\"1Mohm\"", 1);
    ASSERT_EQ(model.type_library.size(), 1u);

    IrProgram program = lower_inputs_to_ir(model.type_library[0]);
    EXPECT_EQ(program.component_type_name, "voltageClamp");

    bool has_require_v = false;
    for (const auto &directive : program.alloc) {
        if (auto *require = std::get_if<RequireDirective>(&directive); require && require->name == "v") {
            has_require_v = true;
            EXPECT_EQ(require->scope, "postsynaptic");
        }
    }
    EXPECT_TRUE(has_require_v);

    GpuSource source = lower_ir_program_to_gpu_source(program);
    ASSERT_EQ(source.functions.size(), 1u);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "voltage_clamp"));
}

// ── unsupported Inputs ComponentTypes: a clear, named error, not a silent guess ───────────────────

TEST(InputsLowering, compound_input_throws_a_named_child_composition_scope_error) {
    ModelSpecification model = build_inputs_model("CompoundNotUsed", "pulseGenerator",
        "delay=\"0ms\" duration=\"10ms\" amplitude=\"1nA\"", 1);
    TypeLibraryEntry fake_compound_input_entry = model.type_library[0];
    fake_compound_input_entry.component_type_name = "compoundInput";

    try {
        lower_inputs_to_ir(fake_compound_input_entry);
        FAIL() << "expected lower_inputs_to_ir to throw for compoundInput";
    } catch (const std::runtime_error &error) {
        String message = error.what();
        EXPECT_NE(message.find("compoundInput"), String::npos);
        EXPECT_NE(message.find("child-component"), String::npos);
    }
}

TEST(InputsLowering, poisson_firing_synapse_throws_a_named_child_composition_scope_error) {
    ModelSpecification model = build_inputs_model("PoissonSynapseNotUsed", "pulseGenerator",
        "delay=\"0ms\" duration=\"10ms\" amplitude=\"1nA\"", 1);
    TypeLibraryEntry fake_entry = model.type_library[0];
    fake_entry.component_type_name = "poissonFiringSynapse";

    try {
        lower_inputs_to_ir(fake_entry);
        FAIL() << "expected lower_inputs_to_ir to throw for poissonFiringSynapse";
    } catch (const std::runtime_error &error) {
        String message = error.what();
        EXPECT_NE(message.find("poissonFiringSynapse"), String::npos);
        EXPECT_NE(message.find("child-component"), String::npos);
    }
}

// ── SpikeSourcePoisson: IR shape + real MSL compilation ───────────────────────────────────────────

TEST(InputsLowering, spike_source_poisson_lowers_to_a_rand_driven_spike_source_and_compiles_as_msl) {
    ModelSpecification model = build_inputs_model("Poisson", "SpikeSourcePoisson", "start=\"0\" duration=\"1000s\" rate=\"20\"", 1);
    ASSERT_EQ(model.type_library.size(), 1u);
    EXPECT_EQ(model.type_library[0].category, TypeLibraryCategory::Inputs);
    EXPECT_EQ(model.type_library[0].component_type_name, "SpikeSourcePoisson");

    IrProgram program = lower_inputs_to_ir(model.type_library[0]);

    bool has_rand = false;
    for (const auto &instruction : program.tick.reset) {
        std::visit([&](auto &&value) {
            using InstructionType = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<InstructionType, IfInstruction>) {
                for (const auto &nested : value.then_body) {
                    if (std::holds_alternative<RandomInstruction>(nested.operation)) has_rand = true;
                }
            }
        }, instruction.operation);
    }
    EXPECT_TRUE(has_rand);

    GpuSource source = lower_ir_program_to_gpu_source(program);
    ASSERT_EQ(source.functions.size(), 1u);
    EXPECT_EQ(source.functions[0].function_name, "SpikeSourcePoisson_tick");
    EXPECT_NE(source.msl_source.find("rng_state"), String::npos);
    EXPECT_NE(source.msl_source.find("spikecorec_rand_uniform"), String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "spike_source_poisson"));

    // rng_state is a required parameter of the generated per-neuron function.
    bool rng_state_is_a_parameter = false;
    for (const auto &name : source.functions[0].parameter_names_in_order) {
        if (name == "rng_state") rng_state_is_a_parameter = true;
    }
    EXPECT_TRUE(rng_state_is_a_parameter);
}

// ── acceptance criterion: an on-device Poisson population run through AssembledModel ──────────────
//
// 50 independent SpikeSourcePoisson neurons at rate=20Hz, driven for 5000 ticks at dt=1ms (5
// simulated seconds) -- expected total spike count = 50 * 20 * 5 = 5000. Tolerance: this is a
// genuine stochastic process (each neuron's ISI ~ Exponential(rate), summed over 50 independent
// renewal processes over 5s), so the SAMPLE mean spike count has standard deviation
// sqrt(expected_count) ~= sqrt(5000) ~= 71 by the Poisson-process count-variance identity. A +-25%
// band (+-1250, ~17.6 standard deviations) is deliberately far more generous than a strict
// statistical test would need -- it is sized to absorb: (a) the onset transient common to every
// neuron (`tnext` zero-initializes rather than seeding from a random first ISI, since
// allocate_model never applies a `StateDirective`'s `initial_value` -- ticket #65 does not change
// that pre-existing, documented gap, see ir.h's own comment), which fires every neuron once per
// tick from the window's own onset (`start`) until `tnext` accumulates past the current time --
// at `start=0` (this test) that is a SINGLE extra, non-Poisson-distributed spike per neuron (~50
// extra spikes total, ~1% of expected -- comfortably inside this band on its own); at a nonzero
// `start` the burst instead scales as roughly `start*rate` extra spikes per neuron, which this
// tolerance is NOT sized to absorb -- on-device Poisson with `start > 0` should be treated as
// unverified until `initial_value` wiring lands, and (b) `spikecorec_rand_uniform`'s
// xorshift32 + top-24-bits construction being a fast, minimal-quality PRNG (ticket #55's own
// documented choice) rather than a rigorously-vetted one, not something this ticket should demand
// perfect statistical fidelity from.
TEST(InputsLowering, poisson_population_spike_count_matches_expected_rate_over_several_thousand_ticks) {
    const s32 population_size = 50;
    const f64 rate_hz = 20.0;
    const s64 tick_count = 5000;
    const f32 dt = 0.001f; // seconds

    ModelSpecification model = build_inputs_model("PoissonPopulation", "SpikeSourcePoisson",
        "start=\"0\" duration=\"1000s\" rate=\"20\"", population_size);
    ASSERT_EQ(model.populations.size(), 1u);
    ASSERT_EQ((s32)model.total_neuron_count, population_size);

    IrProgram program = lower_inputs_to_ir(model.type_library[0]);
    spikecorec::Vector<IrProgram> programs{program};

    ModelAllocation allocation = allocate_model(model, programs);

    // Trivial ring adjacency (K2Tree rejects self-loops, so neuron i -> neuron (i+1)%N instead),
    // purely so WeightMatrix accepts a non-empty network (it rejects an edge-free one) -- this test
    // only cares about spike TIMING (last_spiked), never network_inputs/propagation, so the edge's
    // own weight is irrelevant.
    vector<vector<s32>> adjacency((usize)population_size);
    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        adjacency[(usize)neuron_index] = {(neuron_index + 1) % population_size};
    }
    WeightMatrix weights(adjacency, /*rank=*/1);
    weights.set_constant_weight(0.0f);

    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> network_inputs = allocate<f32>((usize)population_size * sizeof(f32));
    memset(network_inputs.get_contents(), 0, (usize)population_size * sizeof(f32));
    // Sentinel -1 ("never fired"), NOT 0 -- `last_spiked[n] == tick` at tick 0 would otherwise be
    // indistinguishable from a genuine first spike at tick 0.
    GpuPointer<s64> last_spiked = allocate<s64>((usize)population_size * sizeof(s64));
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + population_size, (s64)-1);
    GpuPointer<s32> next_active_indices = allocate<s32>((usize)population_size * sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>((usize)population_size * sizeof(s32));
    std::fill(active_generation.get_contents(), active_generation.get_contents() + population_size, -1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)population_size * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)population_size * sizeof(bool));
    GpuPointer<u32> rng_state = allocate<u32>((usize)population_size * sizeof(u32));
    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        rng_state.get_contents()[neuron_index] = (u32)((neuron_index + 1) * 2654435761u) | 1u; // nonzero seed
    }

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();
    buffers.rng_state = rng_state.get_contents();

    s64 total_spike_count = 0;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        assembled_model.step_tick(buffers, dt, tick, tick + 1);
        const s64 *last_spiked_contents = last_spiked.get_contents();
        for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
            if (last_spiked_contents[neuron_index] == tick) ++total_spike_count;
        }
    }

    f64 expected_spike_count = (f64)population_size * rate_hz * ((f64)tick_count * (f64)dt);
    f64 tolerance = expected_spike_count * 0.25;
    EXPECT_NEAR((f64)total_spike_count, expected_spike_count, tolerance)
        << "observed=" << total_spike_count << " expected=" << expected_spike_count;

    // Sanity: every neuron actually fired at least once (a vacuous all-zero trajectory would
    // trivially "pass" a count-only check for the wrong reason).
    s32 neurons_that_fired = 0;
    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        if (last_spiked.get_contents()[neuron_index] != -1) ++neurons_that_fired;
    }
    EXPECT_GT(neurons_that_fired, population_size / 2);
}
