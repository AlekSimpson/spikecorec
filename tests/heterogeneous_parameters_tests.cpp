#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/synapse_lowering.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/stimulus_schedule.h"
#include "spikecorec/nml/discrete_spike_input.h"

#include "nml_network_generator.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::nml::network_generation;

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
//     values, driven by IDENTICAL stimulus through a real SpikeEngine, spikes at genuinely
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
    const spikecorec::Vector<f64> vth_values = {1.0, 2.0, 3.0, 4.0};
    const f32 injected_current = 0.5f;
    const f32 dt = 1.0f;
    const s64 tick_count = 40;

    ModelSpecification model = build_lif_population_model(population_size, "-1.0");
    model.type_library[0].heterogeneous_parameter_values["vth"] = vth_values;

    IrProgram program = lower_cell_to_ir(model.type_library[0]);
    spikecorec::Vector<IrProgram> programs{program};

    // SpikeEngine builds its own ModelAllocation + WeightMatrix internally. This model has no real
    // projections, so its own auto-built weight matrix has zero edges -- an explicit, harmless no-op
    // stating the same "this test injects its own constant current directly into network_inputs every
    // tick and does not depend on propagation between neurons" intent the old manually-built ring
    // adjacency (see inputs_lowering_tests.cpp's own Poisson acceptance test for the identical
    // precedent) used to document.
    SpikeEngine engine(model, programs, dt);
    engine.weights.set_constant_weight(0.0f);

    spikecorec::Vector<s64> first_spike_tick((usize)population_size, -1);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
            engine.network_inputs.get_contents()[neuron_index] = injected_current; // identical stimulus, every tick
        }
        engine.step_tick(dt, tick, tick + 1);
        const s64 *last_spiked_contents = engine.last_spiked.get_contents();
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

// ── 4. ticket #130 [T9]: within-population heterogeneity on a REAL GLIF population, wired into a
// connected ring network, with spiking activity + heterogeneity checked at neurons multiple synaptic
// hops from the only directly-driven neuron ──────────────────────────────────────────────────────────
//
// The acceptance test just above (ticket #65) proves the heterogeneous_parameter_values mechanism on
// an ISOLATED, effectively-unconnected population of LIF cells, each fed the SAME hand-injected
// current directly -- no real synaptic hop is involved (its own comment: "this test injects its own
// constant current directly into network_inputs every tick and does not depend on propagation between
// neurons"). GLIF cells were never exercised this way either (LIF only). This section closes both
// gaps: a real GLIF5 population (chosen per this ticket's own instruction -- GLIF3/GLIF5 both carry
// after-spike-current state alongside a spike threshold), wired into a REAL connected ring topology
// (nml_network_generator.h's own out_degree=1 wiring, ticket #100 -- neuron i scatters onto neuron
// (i+1) % N), with spiking activity and heterogeneity explicitly checked and reported at neurons many
// synaptic hops deep, not merely the directly-driven neuron or its immediate neighbor.
//
// TWO parameters are set heterogeneously per neuron (build_heterogeneous_glif5_ring, below): `thetaInf`
// (GLIF5's own asymptotic/instantaneous spike threshold -- this fixture's "th_inf/vth"-equivalent) and
// `ascAdd1` (the first after-spike-current's amplitude). Every OTHER parameter (C/gL/EL/vreset/t_ref/
// tauTheta/thetaSpikeAdd/tauAsc1/tauAsc2/ascAdd2) stays uniform -- heterogeneity is per-parameter, not
// whole-type, matching test 1's own established convention above.
//
// ONLY neuron 0 is ever directly driven -- continuous pulseGenerator current injection in one test,
// a discrete host-provided 0/1 spike array in the other (tests/end_to_end_network_tests.cpp's own two
// established driving mechanisms, ticket #100). Every OTHER neuron's activity is caused entirely by
// real propagation through the ring's own WeightMatrix/k^2-tree (arch's stage 6 `network_inputs`
// scatter) -- no neuron but Pop[0] ever receives host-injected current. Both tests below explicitly
// report (via stdout) and assert on neurons at hop distances 4, 8, 12, and 15 out of this 16-neuron
// ring -- multiple synaptic hops from Pop[0], not merely it or its immediate neighbor.
//
// Heterogeneity signature used at EVERY checked neuron, near or many hops deep alike: each GLIF5
// neuron's own `theta`/`asc1` state immediately after ITS OWN first spike is a DETERMINISTIC function
// of that neuron's OWN heterogeneous `thetaInf`/`ascAdd1` -- `theta` stays pinned at `thetaInf` until a
// spike (`TimeDerivative theta' = (thetaInf-theta)/tauTheta` is exactly zero once theta==thetaInf, its
// own OnStart value, and nothing perturbs it before that neuron's own first spike), so a first spike's
// own post-reset `theta == thetaInf_own + thetaSpikeAdd`; likewise `asc1` starts at exactly 0 (OnStart)
// and a first spike's own post-reset `asc1 == ascAdd1_own` (`-asc1/tauAsc1` integrates to ~0 from a
// starting value of exactly 0). This is a STRONGER, timing-independent proof of genuine per-neuron
// parameter heterogeneity than comparing first-spike TICKS (test 3's own LIF mechanism above): it
// holds regardless of how many synaptic hops away a neuron is or how many ticks it took to reach its
// own threshold via real chain propagation, which a raw tick-ordering comparison could not cleanly
// attribute to parameter differences alone once genuine multi-hop chain dynamics are involved (unlike
// test 3's isolated population, every neuron here except Pop[0] is activated purely by upstream
// neurons' own spikes, at a real, neuron-dependent, non-uniform delay).
//
// The ring's own `WeightMatrix` constant-weight placeholder is deliberately large (30nA-equivalent, one
// forward-Euler dt step already carries ~30mV -- comfortably more than any of this fixture's own
// heterogeneous threshold gaps) so a single upstream spike reliably carries the wave to the very next
// ring neuron regardless of that neuron's own thetaInf -- purely a ROUTING vehicle (this tree's own
// established "not a real per-edge synapse-derived magnitude" convention, tests/end_to_end_network_
// tests.cpp's own header comment finding #1), empirically confirmed (a throwaway exploration harness,
// not checked in) to carry the wave across this fixture's full 16-neuron ring within ~20 ticks. The
// heterogeneity claim above never depends on that wave's own timing, only on each neuron's own
// post-first-spike state.

namespace {

const s32 HETEROGENEOUS_RING_SIZE = 16;
const String HETEROGENEOUS_RING_SYNAPSE_XML =
    "  <expOneSynapse id=\"synA\" gbase=\"100nS\" erev=\"0mV\" tauDecay=\"3ms\" weight=\"1\"/>";

spikecorec::Vector<IrProgram> build_heterogeneous_ring_type_library_ir_programs(const ModelSpecification &model) {
    spikecorec::Vector<IrProgram> programs;
    programs.reserve(model.type_library.size());
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) {
            programs.push_back(lower_cell_to_ir(entry));
        } else if (entry.category == TypeLibraryCategory::Synapse) {
            programs.push_back(lower_synapse_to_ir(entry));
        } else {
            IrProgram placeholder;
            placeholder.component_type_name = entry.component_type_name;
            programs.push_back(std::move(placeholder));
        }
    }
    return programs;
}

s32 heterogeneous_ring_population_index(const ModelSpecification &model) {
    for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
        if (model.populations[(usize)population_index].id == "Pop") return population_index;
    }
    throw std::runtime_error("heterogeneous_parameters_tests: no population named 'Pop'");
}

s32 heterogeneous_ring_global_neuron_index(const ModelSpecification &model, s32 local_index) {
    return model.populations[(usize)heterogeneous_ring_population_index(model)].neuron_index_begin + local_index;
}

// A population's cell_state chunk is structure-of-arrays: state slot `state_slot_index`'s own
// [population_size] row sits at `cell_type_boundaries[population_index] + state_slot_index *
// population_size` (allocator.h's own doc comment; same convention tests/end_to_end_network_tests.cpp
// uses for its own population_state_value_index).
s64 heterogeneous_ring_state_value_index(const ModelAllocation &allocation, s32 population_index, s32 state_slot_index,
                                   s32 local_index, s32 population_size) {
    return allocation.cell_type_boundaries.get_contents()[population_index] +
           (s64)state_slot_index * population_size + local_index;
}

f32 read_heterogeneous_ring_glif5_state(const ModelAllocation &allocation, const ModelSpecification &model,
                                  const String &state_variable_name, s32 local_index) {
    s32 population_index = heterogeneous_ring_population_index(model);
    const PopulationEntry &population = model.populations[(usize)population_index];
    s32 state_slot_index = glif_state_variable_slot(GlifVariant::Glif5, state_variable_name);
    return allocation.cell_state.get_contents()[heterogeneous_ring_state_value_index(
        allocation, population_index, state_slot_index, local_index, population.size)];
}

bool heterogeneous_ring_all_finite(const ModelAllocation &allocation) {
    const f32 *cell_state = allocation.cell_state.get_contents();
    for (s64 index = 0; index < allocation.cell_state_element_count; ++index) {
        if (!std::isfinite(cell_state[index])) return false;
    }
    return true;
}

struct HeterogeneousGlifRingFixture {
    ModelSpecification model;
    spikecorec::Vector<IrProgram> programs;
    SpikeEngine engine;
    spikecorec::Vector<f64> theta_inf_values;
    spikecorec::Vector<f64> asc_add1_values;
};

// Builds the shared GLIF5 ring fixture described in this section's own header comment above.
// `stimuli` is forwarded verbatim to generate_network_nml -- empty for the discrete-spike-array test
// (which drives Pop[0] by writing straight into network_inputs itself, DiscreteSpikeInputSchedule's
// own established convention), one pulseGenerator/explicitInput targeting Pop[0] for the
// continuous-current-injection test.
HeterogeneousGlifRingFixture build_heterogeneous_glif5_ring(const String &fixture_id,
                                                             const spikecorec::Vector<GeneratedStimulus> &stimuli) {
    spikecorec::Vector<f64> theta_inf_values((usize)HETEROGENEOUS_RING_SIZE), asc_add1_values((usize)HETEROGENEOUS_RING_SIZE);
    for (s32 local_index = 0; local_index < HETEROGENEOUS_RING_SIZE; ++local_index) {
        // -52mV .. -43.6mV across the ring (thetaInf), -80pA .. -192pA across the ring (ascAdd1) --
        // spaced widely enough apart (0.6mV / 8pA per neuron) that neither float32 rounding nor the
        // tiny forward-Euler step taken between OnStart and a neuron's own first spike could
        // plausibly be mistaken for genuine cross-neuron heterogeneity.
        theta_inf_values[(usize)local_index] = -0.052 + local_index * 0.0006;
        asc_add1_values[(usize)local_index] = -(80.0 + local_index * 8.0) * 1e-12;
    }

    GeneratedPopulation population;
    population.population_id = "Pop";
    population.variant = GlifVariant::Glif5;
    population.bound_instance_id = "glif5HeterogeneousRingInstance";
    // thetaInf/ascAdd1 attribute values here are an ordinary uniform-population placeholder -- never
    // actually used (this file's own established `vth_placeholder` convention, header comment above);
    // every neuron's real value comes from heterogeneous_parameter_values below instead.
    population.cell_instance_attributes =
        "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vreset=\"-70mV\" t_ref=\"5ms\" thetaInf=\"-50mV\" tauTheta=\"50ms\" "
        "thetaSpikeAdd=\"5mV\" tauAsc1=\"100ms\" tauAsc2=\"10ms\" ascAdd1=\"-100pA\" ascAdd2=\"-200pA\"";
    population.neuron_count = HETEROGENEOUS_RING_SIZE;

    GeneratedProjection ring_projection;
    ring_projection.projection_id = "HeterogeneousRingProj";
    ring_projection.presynaptic_population_id = "Pop";
    ring_projection.postsynaptic_population_id = "Pop";
    ring_projection.synapse_instance_id = "synA";
    ring_projection.out_degree = 1;

    String content_xml =
        generate_network_nml("HeterogeneousGlif5Ring", {population}, HETEROGENEOUS_RING_SYNAPSE_XML, {ring_projection}, stimuli);

    write_temp_file("spikecorec_" + fixture_id + "_content.nml", content_xml);
    String top_path = write_temp_file("spikecorec_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"" + fixture_id + "Top\">"
        "  <include href=\"spikecorec_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    ModelSpecification model = build_model_specification(resolved);
    if (model.total_neuron_count != HETEROGENEOUS_RING_SIZE) {
        throw std::runtime_error("build_heterogeneous_glif5_ring: unexpected neuron count");
    }

    s32 cell_type_index = -1;
    for (s32 index = 0; index < (s32)model.type_library.size(); ++index) {
        if (model.type_library[(usize)index].category == TypeLibraryCategory::Cell) cell_type_index = index;
    }
    if (cell_type_index < 0) throw std::runtime_error("build_heterogeneous_glif5_ring: no Cell type-library entry");
    model.type_library[(usize)cell_type_index].heterogeneous_parameter_values["thetaInf"] = theta_inf_values;
    model.type_library[(usize)cell_type_index].heterogeneous_parameter_values["ascAdd1"] = asc_add1_values;

    spikecorec::Vector<IrProgram> programs = build_heterogeneous_ring_type_library_ir_programs(model);

    // SpikeEngine builds its own ModelAllocation + WeightMatrix internally -- the latter from
    // model.projections, the exact same adjacency this file's own former (now-removed)
    // build_heterogeneous_ring_weight_matrix helper built by hand.
    SpikeEngine engine(model, programs);

    // allocate_model (called internally by SpikeEngine's constructor) zero-initializes cell_state
    // without applying OnStart (this tree's own established convention, tests/end_to_end_network_
    // tests.cpp's own seed_glif_initial_state) -- seed v to EL (uniform) and theta to EACH neuron's
    // own heterogeneous thetaInf (NOT the uniform baked placeholder above -- reading that here instead
    // would silently defeat the entire point of this fixture: a first spike's own post-reset theta
    // would no longer trace back to that neuron's OWN configured value). asc1/asc2 legitimately stay
    // at their own zero-initialized default, matching GLIF5's own OnStart `asc1=0`/`asc2=0`.
    {
        s32 population_index = heterogeneous_ring_population_index(model);
        const PopulationEntry &population_entry = model.populations[(usize)population_index];
        f64 resting_potential =
            model.type_library[(usize)population_entry.type_library_index].baked_constants.at("EL");
        s32 v_slot = glif_state_variable_slot(GlifVariant::Glif5, "v");
        s32 theta_slot = glif_state_variable_slot(GlifVariant::Glif5, "theta");
        for (s32 local_index = 0; local_index < population_entry.size; ++local_index) {
            engine.nml_allocation_.cell_state.get_contents()[heterogeneous_ring_state_value_index(
                engine.nml_allocation_, population_index, v_slot, local_index, population_entry.size)] =
                (f32)resting_potential;
            engine.nml_allocation_.cell_state.get_contents()[heterogeneous_ring_state_value_index(
                engine.nml_allocation_, population_index, theta_slot, local_index, population_entry.size)] =
                (f32)theta_inf_values[(usize)local_index];
        }
    }

    // ticket #131: SpikeEngine now dispatches HETEROGENEOUS_RING_SYNAPSE_XML's own real
    // expOneSynapse per-edge dynamics for this model (it has real projections), so this
    // constant-weight placeholder is inert (its own scalar propagate contribution is forced to zero
    // whenever real per-edge synapse dispatch is active -- see master_kernel.h) -- left set anyway
    // as an explicit, harmless no-op rather than deleting the call, matching this tree's established
    // "state the placeholder was here" convention. HETEROGENEOUS_RING_SYNAPSE_XML's own `gbase` is
    // tuned strong enough (100nS) that a single upstream spike's real, decaying conductance still
    // reliably carries the wave across all 16 ring hops within this section's own tick budgets,
    // regardless of each hop's own heterogeneous thetaInf. Purely a ROUTING vehicle -- the
    // heterogeneity claim in both tests below never depends on the propagated current's own
    // magnitude or the wave's own timing, only on each neuron's own post-first-spike state (see this
    // section's own header comment).
    engine.weights.set_constant_weight(3.0e-8f);

    return {std::move(model), std::move(programs), std::move(engine),
            std::move(theta_inf_values), std::move(asc_add1_values)};
}

// Runs `tick_count` ticks of `fixture.engine`, invoking `apply_stimulus_for_tick`
// once per tick immediately before step_tick (the caller's own driving mechanism -- continuous
// injection or a discrete spike array), and returns each of the ring's own neurons' first-spike tick
// plus its `theta`/`asc1` state captured AT THE MOMENT of that first spike (NOT at the end of the run
// -- theta/asc1 both continue evolving after a neuron's first spike, so reading them only once the
// whole run is done would no longer reflect that neuron's own first-spike-time value).
struct HeterogeneousRingRunResult {
    spikecorec::Vector<s64> first_spike_tick;
    spikecorec::Vector<f32> theta_after_first_spike;
    spikecorec::Vector<f32> asc1_after_first_spike;
};

HeterogeneousRingRunResult run_heterogeneous_ring(HeterogeneousGlifRingFixture &fixture, s64 tick_count,
                                     f32 dt_seconds, const std::function<void(s64)> &apply_stimulus_for_tick) {
    HeterogeneousRingRunResult result;
    result.first_spike_tick.assign((usize)HETEROGENEOUS_RING_SIZE, -1);
    result.theta_after_first_spike.assign((usize)HETEROGENEOUS_RING_SIZE, 0.0f);
    result.asc1_after_first_spike.assign((usize)HETEROGENEOUS_RING_SIZE, 0.0f);

    for (s64 tick = 0; tick < tick_count; ++tick) {
        apply_stimulus_for_tick(tick);
        fixture.engine.step_tick(dt_seconds, tick, tick + 1);
        EXPECT_TRUE(heterogeneous_ring_all_finite(fixture.engine.nml_allocation_)) << "tick=" << tick;

        for (s32 local_index = 0; local_index < HETEROGENEOUS_RING_SIZE; ++local_index) {
            s32 global_index = heterogeneous_ring_global_neuron_index(fixture.model, local_index);
            if (result.first_spike_tick[(usize)local_index] < 0 &&
                fixture.engine.last_spiked.get_contents()[global_index] == tick) {
                result.first_spike_tick[(usize)local_index] = tick;
                result.theta_after_first_spike[(usize)local_index] =
                    read_heterogeneous_ring_glif5_state(fixture.engine.nml_allocation_, fixture.model, "theta", local_index);
                result.asc1_after_first_spike[(usize)local_index] =
                    read_heterogeneous_ring_glif5_state(fixture.engine.nml_allocation_, fixture.model, "asc1", local_index);
            }
        }
    }
    return result;
}

// Shared assertions for both driving mechanisms below: spiking activity + heterogeneity, explicitly
// checked and reported at hop distances 4/8/12/15 out of this 16-neuron ring -- ticket #130's own
// explicit requirement that multi-hop neurons (not merely the directly-driven neuron or its immediate
// neighbor) be covered.
void assert_heterogeneous_ring_multi_hop_heterogeneity(const HeterogeneousGlifRingFixture &fixture,
                                                 const HeterogeneousRingRunResult &result, s64 tick_count,
                                                 const String &driving_mechanism_label) {
    ASSERT_GE(result.first_spike_tick[0], 0) << "the directly-driven neuron (Pop[0]) never spiked at all";

    const spikecorec::Vector<s32> hop_distances_to_check = {4, 8, 12, 15};
    for (s32 hop_distance : hop_distances_to_check) {
        std::cout << "[HeterogeneousGlifNetworkParameters/" << driving_mechanism_label
                  << "] hop_distance=" << hop_distance
                  << " first_spike_tick=" << result.first_spike_tick[(usize)hop_distance]
                  << " theta_after_first_spike=" << result.theta_after_first_spike[(usize)hop_distance]
                  << " asc1_after_first_spike=" << result.asc1_after_first_spike[(usize)hop_distance]
                  << " own_thetaInf=" << fixture.theta_inf_values[(usize)hop_distance]
                  << " own_ascAdd1=" << fixture.asc_add1_values[(usize)hop_distance] << "\n";

        ASSERT_GE(result.first_spike_tick[(usize)hop_distance], 0)
            << "neuron at hop distance " << hop_distance << " (" << driving_mechanism_label
            << " driving) from the only directly-driven neuron (Pop[0]) never spiked within " << tick_count
            << " ticks -- propagation through the ring failed to reach it";

        f32 expected_theta = (f32)fixture.theta_inf_values[(usize)hop_distance] + 0.005f; // + thetaSpikeAdd
        f32 expected_asc1 = (f32)fixture.asc_add1_values[(usize)hop_distance];

        EXPECT_NEAR(result.theta_after_first_spike[(usize)hop_distance], expected_theta, 1e-5f)
            << "hop " << hop_distance << " (" << driving_mechanism_label << "): theta right after this "
               "neuron's OWN first spike should equal ITS OWN heterogeneous thetaInf + thetaSpikeAdd, not "
               "some other neuron's (or a uniform baked placeholder)";
        EXPECT_NEAR(result.asc1_after_first_spike[(usize)hop_distance], expected_asc1, 1e-12f)
            << "hop " << hop_distance << " (" << driving_mechanism_label << "): asc1 right after this "
               "neuron's OWN first spike should equal ITS OWN heterogeneous ascAdd1, not some other "
               "neuron's (or a uniform baked placeholder)";
    }

    // Cross-neuron check: two DIFFERENT multi-hop neurons must show MEASURABLY different behavior, not
    // merely each coincidentally matching a near-identical config.
    EXPECT_GT(result.theta_after_first_spike[12], result.theta_after_first_spike[4] + 0.001f)
        << "(" << driving_mechanism_label << ") hop 12 and hop 4 should show measurably different "
           "post-spike theta, reflecting their different heterogeneous thetaInf values";
    EXPECT_LT(result.asc1_after_first_spike[12], result.asc1_after_first_spike[4] - 1e-11f)
        << "(" << driving_mechanism_label << ") hop 12 and hop 4 should show measurably different "
           "post-spike asc1, reflecting their different heterogeneous ascAdd1 values";
}

} // namespace

TEST(HeterogeneousGlifNetworkParameters, glif5_ring_network_continuous_current_injection_multi_hop_heterogeneity) {
    const s64 tick_count = 250;
    const f32 dt_seconds = 1e-4f;

    GeneratedStimulus stimulus;
    stimulus.population_id = "Pop";
    stimulus.local_index = 0;
    stimulus.pulse_generator_id = "heterogeneousRingPulseGen";
    stimulus.delay = "0ms";
    stimulus.duration = "25ms"; // the whole run (tick_count * dt_seconds)
    stimulus.amplitude = "3nA";

    HeterogeneousGlifRingFixture fixture =
        build_heterogeneous_glif5_ring("heterogeneous_glif5_ring_continuous", {stimulus});

    StimulusSchedule schedule = build_stimulus_schedule(fixture.model, (f64)dt_seconds);
    s32 driven_neuron_index = heterogeneous_ring_global_neuron_index(fixture.model, 0);

    HeterogeneousRingRunResult result = run_heterogeneous_ring(fixture, tick_count, dt_seconds,
        [&](s64 tick) {
            fixture.engine.network_inputs.get_contents()[driven_neuron_index] +=
                (f32)schedule.current_at(driven_neuron_index, tick);
        });

    assert_heterogeneous_ring_multi_hop_heterogeneity(fixture, result, tick_count, "continuous_current_injection");
}

TEST(HeterogeneousGlifNetworkParameters, glif5_ring_network_discrete_spike_array_multi_hop_heterogeneity) {
    const s64 tick_count = 250;
    const f32 dt_seconds = 1e-4f;

    HeterogeneousGlifRingFixture fixture = build_heterogeneous_glif5_ring("heterogeneous_glif5_ring_discrete", {});

    s32 driven_neuron_index = heterogeneous_ring_global_neuron_index(fixture.model, 0);

    // A literal, host-provided 0/1 array -- one value per tick -- held on for the whole run, driving
    // Pop[0] directly (tests/end_to_end_network_tests.cpp's own second established driving mechanism,
    // ticket #100; DiscreteSpikeInputSchedule, ticket #100/#101).
    DiscreteSpikeInputSchedule discrete_schedule;
    discrete_schedule.target_neuron_indices = {driven_neuron_index};
    discrete_schedule.current_amplitude_amperes = 3.0e-9f; // 3nA
    discrete_schedule.spike_bits.assign((usize)tick_count, spikecorec::Vector<u8>{1});

    HeterogeneousRingRunResult result = run_heterogeneous_ring(fixture, tick_count, dt_seconds,
        [&](s64 tick) {
            discrete_schedule.apply_to_network_inputs(fixture.engine.network_inputs.get_contents(), tick);
        });

    assert_heterogeneous_ring_multi_hop_heterogeneity(fixture, result, tick_count, "discrete_spike_array");
}
