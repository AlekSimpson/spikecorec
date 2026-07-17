#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cstring>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/master_kernel.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Within-population parameter heterogeneity tests (ticket #65 [F4]) ────────────────────────────
//
// Three things this file demonstrates, matching arch §3.1's heterogeneous branch of the
// bake-vs-parameterize rule:
//  1. cell_lowering.cpp's alternative path: a Parameter name present in
//     `TypeLibraryEntry::heterogeneous_parameter_values` emits a `param : dyn` array
//     (`ParamDynamicDirective`), not a baked literal, even though Phase-1's default rule would
//     otherwise bake it.
//  2. allocator.cpp fills the resulting per-neuron array from those exact values, at the right
//     offset within the model-wide `dynamic_parameter_arrays` buffer.
//  3. The acceptance criterion: a population of otherwise-identical LIF cells with per-neuron `vth`
//     values, driven by IDENTICAL stimulus through a real AssembledModel, spikes at genuinely
//     different times -- each neuron's own `vth` value, not just "doesn't crash".
//
// Real NML syntax for genuine population-level heterogeneity (per-instance distinct bound
// components, `inhomogeneousParameter`/`inhomogeneousValue`) is NOT implemented by this ticket --
// see model_specification.h's own doc comment on `heterogeneous_parameter_values` for why. This
// file's fixtures build an ordinary UNIFORM population through the real front end, then set
// `heterogeneous_parameter_values` directly on the resulting TypeLibraryEntry -- the same
// hand-built-fixture pattern allocator_tests.cpp/master_kernel_tests.cpp already use to exercise
// engine-level data the NML front end has no syntax for yet.

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

// A regime-less LIF cell -- textually identical in shape to cell_lowering_tests.cpp's own
// "PlainLifCell" fixture (dv/dt = (network_inputs + gL*(EL-v))/C, spike + reset on v > vth) --
// kept as its own self-contained copy here (matching this codebase's own per-file fixture
// convention; see master_kernel_tests.cpp's header comment) rather than shared across translation
// units. `vth_placeholder` seeds an ordinary uniform-population baked value that every test below
// then overrides via `heterogeneous_parameter_values` -- its exact value is never actually used.
ModelSpecification build_lif_population_model(s32 population_size, const String &vth_placeholder) {
    write_temp_file("spikecorec_heterogeneous_parameters_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"HeterogeneousParametersContent\">"
        "  <ComponentType name=\"HeteroLifCell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <TimeDerivative variable=\"v\" value=\"(network_inputs + gL * (EL - v)) / C\"/>"
        "      <OnCondition test=\"v .gt. vth\">"
        "        <EventOut port=\"spike\"/>"
        "        <StateAssignment variable=\"v\" value=\"vreset\"/>"
        "      </OnCondition>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <HeteroLifCell id=\"instance0\" C=\"1.0\" gL=\"0.1\" EL=\"0.0\" vth=\"" + vth_placeholder + "\" vreset=\"0.0\"/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"instance0\" size=\"" + std::to_string(population_size) + "\"/>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_heterogeneous_parameters_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"HeterogeneousParametersTop\">"
        "  <include href=\"spikecorec_heterogeneous_parameters_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

} // namespace

// ── 1. cell_lowering.cpp's bake-vs-parameterize alternative path ─────────────────────────────────

TEST(HeterogeneousParameters, heterogeneous_vth_emits_param_dynamic_directive_instead_of_baking) {
    ModelSpecification model = build_lif_population_model(3, "-1.0");
    ASSERT_EQ(model.type_library.size(), 1u);

    TypeLibraryEntry &entry = model.type_library[0];
    ASSERT_TRUE(entry.baked_constants.count("vth")); // the uniform-population default still baked it
    entry.heterogeneous_parameter_values["vth"] = {1.0, 2.0, 3.0};

    IrProgram program = lower_cell_to_ir(entry);

    bool vth_is_dynamic = false;
    bool vth_is_baked = false;
    for (const auto &directive : program.alloc) {
        if (auto *param_dynamic = std::get_if<ParamDynamicDirective>(&directive); param_dynamic && param_dynamic->name == "vth") {
            vth_is_dynamic = true;
            EXPECT_EQ(param_dynamic->dtype, "f32");
        }
        if (auto *param_constant = std::get_if<ParamConstantDirective>(&directive); param_constant && param_constant->name == "vth") {
            vth_is_baked = true;
        }
    }
    EXPECT_TRUE(vth_is_dynamic);
    EXPECT_FALSE(vth_is_baked);

    // Every OTHER parameter (gL/EL/C/vreset) stays baked -- heterogeneity is per-parameter, not
    // whole-type.
    bool gL_is_baked = false;
    for (const auto &directive : program.alloc) {
        if (auto *param_constant = std::get_if<ParamConstantDirective>(&directive); param_constant && param_constant->name == "gL") {
            gL_is_baked = true;
            EXPECT_TRUE(param_constant->literal_value.has_value());
        }
    }
    EXPECT_TRUE(gL_is_baked);
}

// ── 2. allocator.cpp fills the per-neuron array from the real values ─────────────────────────────

TEST(HeterogeneousParameters, allocator_fills_dynamic_parameter_array_from_heterogeneous_values) {
    ModelSpecification model = build_lif_population_model(3, "-1.0");
    model.type_library[0].heterogeneous_parameter_values["vth"] = {1.5, 2.5, 3.5};

    IrProgram program = lower_cell_to_ir(model.type_library[0]);
    spikecorec::Vector<IrProgram> programs{program};

    ModelAllocation allocation = allocate_model(model, programs);

    auto found = allocation.dynamic_parameter_arrays.find(type_scoped_key(0, "vth"));
    ASSERT_NE(found, allocation.dynamic_parameter_arrays.end());
    const f32 *values = found->second.get_contents();
    EXPECT_FLOAT_EQ(values[0], 1.5f);
    EXPECT_FLOAT_EQ(values[1], 2.5f);
    EXPECT_FLOAT_EQ(values[2], 3.5f);
}

TEST(HeterogeneousParameters, allocator_throws_when_heterogeneous_value_count_mismatches_population_size) {
    ModelSpecification model = build_lif_population_model(3, "-1.0");
    model.type_library[0].heterogeneous_parameter_values["vth"] = {1.5, 2.5}; // only 2, population size 3

    IrProgram program = lower_cell_to_ir(model.type_library[0]);
    spikecorec::Vector<IrProgram> programs{program};

    EXPECT_THROW({ ModelAllocation allocation = allocate_model(model, programs); }, std::runtime_error);
}

// ── 3. acceptance criterion: identical stimulus, genuinely different per-neuron spike timing ─────
//
// Four neurons, one population, identical gL/EL/C/vreset and identical injected current every tick
// -- only `vth` differs (1.0 < 2.0 < 3.0 < 4.0). Under dv/dt = (I + gL*(EL-v))/C with I=0.5,
// gL=0.1, EL=0, C=1, v settles toward I/gL = 5.0, so every one of these thresholds is reachable and
// a strictly higher vth strictly delays first-spike time (analytically
// t = -tau*ln(1 - vth/(I/gL)), tau=1/gL=10: ~2.2/5.1/9.2/16.1 ticks respectively) -- this test only
// asserts the qualitative, discretization-robust consequence (strictly increasing first-spike tick
// with strictly increasing vth), not the exact analytic tick.
TEST(HeterogeneousParameters, population_with_heterogeneous_vth_spikes_at_genuinely_different_times) {
    const s32 population_size = 4;
    const Vector<f64> vth_values = {1.0, 2.0, 3.0, 4.0};
    const f32 injected_current = 0.5f;
    const f32 dt = 1.0f;
    const s64 tick_count = 40;

    ModelSpecification model = build_lif_population_model(population_size, "-1.0");
    model.type_library[0].heterogeneous_parameter_values["vth"] = vth_values;

    IrProgram program = lower_cell_to_ir(model.type_library[0]);
    spikecorec::Vector<IrProgram> programs{program};

    ModelAllocation allocation = allocate_model(model, programs);

    // Trivial ring adjacency (K2Tree rejects self-loops, so neuron i -> neuron (i+1)%N instead),
    // purely so WeightMatrix accepts a non-empty network (see inputs_lowering_tests.cpp's own
    // Poisson acceptance test for the identical precedent) -- this test injects its own constant
    // current directly into network_inputs every tick and does not depend on propagation between
    // neurons.
    vector<vector<s32>> adjacency((usize)population_size);
    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        adjacency[(usize)neuron_index] = {(neuron_index + 1) % population_size};
    }
    WeightMatrix weights(adjacency, /*rank=*/1);
    weights.set_constant_weight(0.0f);

    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> network_inputs = allocate<f32>((usize)population_size * sizeof(f32));
    // Sentinel -1 ("never fired"), NOT 0 -- `last_spiked[n] == tick` at tick 0 would otherwise be
    // indistinguishable from a genuine first spike at tick 0 (matches active_generation's own -1
    // convention below).
    GpuPointer<s64> last_spiked = allocate<s64>((usize)population_size * sizeof(s64));
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + population_size, (s64)-1);
    GpuPointer<s32> next_active_indices = allocate<s32>((usize)population_size * sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>((usize)population_size * sizeof(s32));
    std::fill(active_generation.get_contents(), active_generation.get_contents() + population_size, -1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)population_size * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)population_size * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();

    Vector<s64> first_spike_tick((usize)population_size, -1);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
            network_inputs.get_contents()[neuron_index] = injected_current; // identical stimulus, every tick
        }
        assembled_model.step_tick(buffers, dt, tick, tick + 1);
        const s64 *last_spiked_contents = last_spiked.get_contents();
        for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
            if (first_spike_tick[(usize)neuron_index] < 0 && last_spiked_contents[neuron_index] == tick) {
                first_spike_tick[(usize)neuron_index] = tick;
            }
        }
    }

    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        ASSERT_GE(first_spike_tick[(usize)neuron_index], 0)
            << "neuron " << neuron_index << " (vth=" << vth_values[(usize)neuron_index] << ") never spiked "
            << "within " << tick_count << " ticks";
    }
    for (s32 neuron_index = 1; neuron_index < population_size; ++neuron_index) {
        EXPECT_GT(first_spike_tick[(usize)neuron_index], first_spike_tick[(usize)neuron_index - 1])
            << "neuron " << neuron_index << " (vth=" << vth_values[(usize)neuron_index]
            << ") should spike strictly later than neuron " << (neuron_index - 1) << " (vth="
            << vth_values[(usize)(neuron_index - 1)] << ")";
    }
}
