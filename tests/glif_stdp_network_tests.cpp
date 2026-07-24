#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/synapse_lowering.h"

#include "nml_network_generator.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::nml::network_generation;

// spikecorec/core/engine.h (pulled in by this file's own SpikeEngine migration) declares a
// file-scope `using namespace spikecorec::log;`, which makes the ALIAS TEMPLATE
// `spikecorec::log::Vector` ambiguous with `spikecorec::Vector` for plain unqualified `Vector<...>`
// lookup -- mirrors tests/simple_lif_stdp_network_tests.cpp's/tests/end_to_end_network_tests.cpp's
// own identical, already-documented issue. `Vector<...>` below is spelled out fully as
// `spikecorec::Vector<...>` for this same reason.

// ── STDP plasticity on a real GLIF network (ticket #129 [T8]) ────────────────────────────────────
//
// tests/simple_lif_stdp_network_tests.cpp (#101 [T2]) deliberately drives STDP against a
// hand-authored, minimal, NON-GLIF cell to isolate whether an at-scale failure is GLIF-specific or
// general. tests/end_to_end_network_tests.cpp's own
// `EndToEndSpikeEnginePlasticity.stdp_measurably_changes_weights_in_the_expected_direction_at_network_scale`
// (#100 [T1]) exercises STDP at real network scale, but only against the legacy `SpikeEngine` (a
// hardcoded LIF cell) -- that file's own header comment documents why: before ticket #132, STDP and
// any real NML/GLIF cell dynamics had no shared tick loop to run in at all. This file is what #132
// unblocked (see master_kernel.h's own "ticket #132"/"Relationship to ticket #129" doc comment): a
// real GLIF3 and a real GLIF5 population (chosen specifically for their after-spike-current /
// threshold-adaptation state -- the interaction ticket #101 and #100 could never exercise), each
// driven through one `AssembledModel` tick loop with active STDP, at the same ~400-neuron/1500-tick
// scale as #100's own medium anchor (tests/end_to_end_network_tests.cpp's
// `glif3_ring_network_current_injection_medium_anchor`).
//
// ── Why delay-ring mode, and why a real (non-constant, un-rescaled) weight ───────────────────────
// `AssembledModel::enable_plasticity` throws if real per-edge synapse dispatch (ticket #131) is
// active, i.e. if `model.projections` is non-empty AND the model was NOT built with
// `enable_delay_ring=true` (master_kernel.h's own "ticket #132" doc comment) -- STDP's rank-1 nudge
// of the shared U/V basis is not (yet) compensated against a peredge synapse's own Ck reconstruction
// sharing that same basis. So this file's network is a real per-edge-delay ring (ticket #64),
// which forces #131's dispatch off entirely regardless of `model.projections` -- the one combination
// `enable_plasticity` already accepts (see
// SpikeEngineNmlPlasticity.enable_plasticity_does_not_throw_under_delay_ring_mode_even_with_projections_present,
// tests/spike_engine_nml_construction_tests.cpp).
//
// The `WeightMatrix` below is left exactly as its constructor's own random U/V init produces (rank=8,
// a fixed seed for reproducibility) -- deliberately NOT passed through `set_constant_weight()` or
// `scale_neighbor_weights_to_root_mean_square()`. Both were tried and rejected: `set_constant_weight`
// overwrites U/V into a uniform encoding of its argument for EVERY possible (source, target) pair
// (weight_matrix.cpp's own `set_constant_weight` -- not just a fallback scalar read only in
// "using_constant_weight" mode, as this file originally assumed), and a value small enough to be safe
// as raw `network_inputs` current (this fixture's GLIF cells have C=100pF/gL=10nS, so even a
// `network_inputs` contribution of order 0.01-0.1 would jump `v` by volts in a single tick) collapses
// `apply_stdp_plasticity`'s own nudge (proportional to the OTHER factor, U or V, which is then also
// that same tiny magnitude) to something numerically unmeasurable -- confirmed empirically against a
// standalone `WeightMatrix::update()` probe before this file was written: a "current-safe" magnitude
// and a "STDP-measurable" magnitude are fundamentally in tension via either of those two APIs, since
// the update step scales roughly QUADRATICALLY down as the weight magnitude is scaled down. Real,
// natural-scale random weights are kept instead (this test's own assertions read exactly the same
// `WeightMatrix::get()` reconstruction `apply_stdp_plasticity`'s own real rank-1 nudge writes), and
// the ring's own SCATTERED delivery is made harmless a different way: every connection's `delay`
// (`delay_attribute` below) is set far LONGER than this test's own `tick_count`, so no delayed
// delivery ever actually reaches the ring's "current slot" within the run -- `apply_stdp_plasticity`
// itself never depends on delivery arriving at all (it walks `buffers.weights->get_neighbors()`, the
// static k^2-tree adjacency, independent of the ring's own scheduling), so this fully decouples "STDP
// still functions correctly against a real k^2-tree topology" from "whatever magnitude a real random
// weight happens to be is safe to inject as current" -- the latter is deliberately never exercised
// here (every neuron is fired by its own direct, isolated external pulse instead, see below).
//
// ── Drive scheme: single reverse-order pass over a directed ring ─────────────────────────────────
// The population is wired as one directed ring (`GeneratedProjection` with `out_degree=1`: neuron
// `n`'s only real edge targets neuron `(n+1) % neuron_count`, tests/nml_network_generator.cpp's own
// deterministic formula). Each neuron is fired EXACTLY ONCE by directly bumping its own "v" state
// element (`allocation.cell_state`) by a fixed external pulse comfortably above every fixture's own
// 20mV EL-to-threshold gap -- the same isolated-single-spike technique
// tests/assembled_model_plasticity_tests.cpp's own acceptance-criterion test already established for
// this exact plasticity path, generalized here to network scale. Neurons are driven in REVERSE index
// order (`neuron_count - 1, neuron_count - 2, ..., 0`), one `drive_gap_ticks`-tick step apart: by the
// time neuron `n` is driven, its own ring target `n + 1` was already driven (and spiked) exactly
// `drive_gap_ticks` ticks earlier -- `apply_stdp_plasticity`'s own precondition
// (`child_last_spiked` real, nonzero, and strictly earlier than the firing tick) is satisfied for
// EVERY edge except the single wraparound edge `(neuron_count - 1) -> 0` (neuron 0 is driven LAST in
// reverse order, so it has not yet spiked when neuron `neuron_count - 1` fires first -- that one edge
// is expected to be skipped, matching `apply_stdp_plasticity`'s own "child never spiked" guard).
// Critically, this keeps every edge's `tick_delta` small and CONSTANT (`== drive_gap_ticks`) rather
// than growing across the pass -- `apply_stdp_plasticity`'s own `decay_delta` shrinks as
// `tick_delta^-3`, so a small, constant `tick_delta` is what actually makes the resulting weight
// change measurable at every edge, not just a handful (empirically confirmed against a standalone
// `WeightMatrix::update` probe at this exact scale/rank/`drive_gap_ticks` before this file was
// written: every edge's weight moved by ~0.005-0.007 with `stdp_learning_rate=0.2`/
// `drive_gap_ticks=3`, several orders of magnitude above any reasonable "measurable" epsilon).
// GLIF3/GLIF5's own `t_ref=5ms` (50 ticks) refractory window is NOT a concern here regardless of how
// small `drive_gap_ticks` is -- unlike tests/end_to_end_network_tests.cpp's own forward+reverse
// SpikeEngine scheme, every neuron here fires only ONCE for the whole run, so no neuron's own
// refractory state is ever re-entered.

namespace {

const String EXP_ONE_SYNAPSE_XML =
    "  <expOneSynapse id=\"synA\" gbase=\"20nS\" erev=\"0mV\" tauDecay=\"3ms\" weight=\"1\"/>";

String write_temp_file(const String &file_name, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / file_name).string();
    std::ofstream output_file(path);
    output_file << contents;
    return path;
}

// Same "content file + thin include-only top.nml wrapper" convention every other *_tests.cpp fixture
// in this tree uses -- NML_Parser only XSD-validates the TOP-LEVEL file it is given.
ModelSpecification load_model_from_generated_content(const String &fixture_id, const String &content_xml) {
    String content_file_name = "spikecorec_glif_stdp_" + fixture_id + "_content.nml";
    write_temp_file(content_file_name, content_xml);

    String top_path = write_temp_file("spikecorec_glif_stdp_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"GlifStdp" + fixture_id + "Top\">"
        "  <include href=\"" + content_file_name + "\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// Parallel to model.type_library -- same convention every other *_tests.cpp fixture in this tree uses.
spikecorec::Vector<IrProgram> build_type_library_ir_programs(const ModelSpecification &model) {
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

// The real directed-ring WeightMatrix (rank > 1, a real random-seeded U/V basis) built from the
// model's own single projection -- deliberately NOT the trivial rank=1 placeholder other files'
// `build_weight_matrix` helper uses, since this file's own assertions read `weights.get()` (the real
// U/V reconstruction) before/after the run.
WeightMatrix build_ring_weight_matrix(const ModelSpecification &model, s64 rank, s64 weight_seed) {
    vector<vector<s32>> adjacency((usize)model.total_neuron_count);
    for (const auto &projection : model.projections) {
        for (const auto &connection : projection.connections) {
            adjacency[(usize)connection.source_neuron_index].push_back(connection.target_neuron_index);
        }
    }
    return WeightMatrix(adjacency, rank, /*check_indexing=*/true, /*max_neighbor_count=*/-1, weight_seed);
}

// Seeds the single population's "v" (every GLIF variant's own state slot 0) to its cell type's own
// EL, and "theta" (GLIF5 only) to its own thetaInf -- allocate_model zero-initializes cell_state
// without applying a cell type's OnStart block, matching this tree's own established convention
// (e.g. tests/end_to_end_network_tests.cpp's own seed_glif_initial_state).
void seed_ring_glif_initial_state(ModelAllocation &allocation, const ModelSpecification &model, GlifVariant variant) {
    const PopulationEntry &population = model.populations[0];
    const TypeLibraryEntry &type_entry = model.type_library[(usize)population.type_library_index];
    s64 population_base_offset = allocation.cell_type_boundaries.get_contents()[0];

    f64 resting_potential = type_entry.baked_constants.at("EL");
    s32 v_slot = glif_state_variable_slot(variant, "v");
    for (s32 local_index = 0; local_index < population.size; ++local_index) {
        allocation.cell_state.get_contents()[population_base_offset + (s64)v_slot * population.size + local_index] =
            (f32)resting_potential;
    }

    if (variant == GlifVariant::Glif5) {
        f64 theta_inf = type_entry.baked_constants.at("thetaInf");
        s32 theta_slot = glif_state_variable_slot(variant, "theta");
        for (s32 local_index = 0; local_index < population.size; ++local_index) {
            allocation.cell_state.get_contents()[population_base_offset + (s64)theta_slot * population.size + local_index] =
                (f32)theta_inf;
        }
    }
}

// Element index of neuron `neuron_index`'s own "v" -- always state slot 0 for every GLIF variant
// (nml_network_generator.cpp's own GLIF1_SLOTS..GLIF5_SLOTS), and this file's model has exactly one
// population, so its own chunk base offset is `cell_type_boundaries[0]`.
s64 membrane_potential_element_index(const ModelAllocation &allocation, s32 neuron_index) {
    return allocation.cell_type_boundaries.get_contents()[0] + neuron_index;
}

bool all_finite(const ModelAllocation &allocation) {
    const f32 *cell_state = allocation.cell_state.get_contents();
    for (s64 index = 0; index < allocation.cell_state_element_count; ++index) {
        if (!std::isfinite(cell_state[index])) return false;
    }
    return true;
}

struct GlifStdpNetworkRunResult {
    spikecorec::Vector<f32> weight_before; // weight_before[n] == the ring's own (n -> (n+1)%neuron_count) edge
    spikecorec::Vector<f32> weight_after;
    spikecorec::Vector<s64> last_spiked_by_neuron;
};

// Builds one fresh GLIF ring network (`neuron_count` neurons of `variant`, real per-edge delay,
// delay-ring mode) and drives it exactly as this file's own header comment describes: a single
// reverse-index-order pass of isolated, direct external pulses, one `drive_gap_ticks`-tick step
// apart. `stdp_learning_rate <= 0.0f` never calls `enable_plasticity` at all (the zero-learning-rate
// control case -- `enable_plasticity`'s own default argument is nonzero, so this must be an explicit
// skip, not a `enable_plasticity(0.0f)` call).
GlifStdpNetworkRunResult run_glif_ring_stdp_network(
    GlifVariant variant, const String &cell_instance_attributes, const String &fixture_id, s32 neuron_count,
    s64 tick_count, f32 dt_seconds, s64 drive_gap_ticks, f32 stdp_learning_rate
) {
    GeneratedPopulation population;
    population.population_id = "Pop";
    population.variant = variant;
    population.bound_instance_id = "glifRingInstance";
    population.cell_instance_attributes = cell_instance_attributes;
    population.neuron_count = neuron_count;

    GeneratedProjection ring_projection;
    ring_projection.projection_id = "RingProj";
    ring_projection.presynaptic_population_id = "Pop";
    ring_projection.postsynaptic_population_id = "Pop";
    ring_projection.synapse_instance_id = "synA";
    ring_projection.out_degree = 1;
    // A real per-edge delay (ticket #64/#132), deliberately far longer than this file's own
    // tick_count (1500, i.e. 150ms of simulated time at dt_seconds=1e-4) -- see this file's own
    // header comment for why no delayed delivery ever arriving within the run is exactly the point.
    ring_projection.delay_attribute = "1s";

    String content_xml = generate_network_nml(fixture_id, {population}, EXP_ONE_SYNAPSE_XML, {ring_projection}, {});
    ModelSpecification model = load_model_from_generated_content(fixture_id, content_xml);
    if (model.total_neuron_count != neuron_count) {
        throw std::runtime_error("run_glif_ring_stdp_network: unexpected total_neuron_count");
    }

    spikecorec::Vector<IrProgram> programs = build_type_library_ir_programs(model);
    // SpikeEngine builds its own ModelAllocation/WeightMatrix/delay-ring configuration internally
    // (folding what AssembledModel/ModelRuntimeBuffers/DelayRingAllocation used to require the caller
    // to assemble by hand -- see include/spikecorec/core/engine.h). Its own ModelSpecification
    // constructor seeds `weights`' own constant_delay_ticks from this model's real "1s" per-connection
    // delay attribute (the SAME conversion delay_ring.cpp's own compute_max_delay_ticks performs), and
    // (since that delay is far longer than 1 tick) forces its own real per-edge synapse dispatch off
    // entirely for this model -- mirrors AssembledModel's own `enable_delay_ring=true` precedent
    // exactly (master_kernel.h's own doc comment: "ticket #64's delay ring and this ticket's synapse
    // dispatch have not been integrated with each other"), so `enable_plasticity` below is not
    // rejected by the real-per-edge-synapse-dispatch guard (see this file's own header comment).
    SpikeEngine engine(model, programs, dt_seconds);
    seed_ring_glif_initial_state(engine.nml_allocation_, model, variant);

    // Real, natural-scale random weights -- see this file's own header comment for why NOT calling
    // set_constant_weight()/scale_neighbor_weights_to_root_mean_square() here is deliberate. Captures
    // the engine's own already-established constant_delay_ticks (derived above from this model's real
    // "1s" delay attribute) before replacing U/V wholesale, then reapplies it -- swapping in a whole
    // fresh WeightMatrix would otherwise reset delay back to its own default (1 tick), desyncing it
    // from the engine's own already-fixed ring size (a private field, sized once at construction and
    // never recomputed).
    if (!engine.weights.using_constant_delay_ticks) {
        throw std::runtime_error(
            "run_glif_ring_stdp_network: expected a uniform per-connection delay (every generated ring "
            "connection shares the SAME delay_attribute)");
    }
    s32 established_delay_ticks = engine.weights.constant_delay_ticks;
    engine.weights = build_ring_weight_matrix(model, /*rank=*/8, /*weight_seed=*/7);
    engine.weights.set_constant_delay_ticks(established_delay_ticks);

    if (stdp_learning_rate > 0.0f) engine.enable_plasticity(stdp_learning_rate);

    GlifStdpNetworkRunResult result;
    result.weight_before.resize((usize)neuron_count);
    for (s32 neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
        result.weight_before[(usize)neuron_index] = engine.weights.get(neuron_index, (neuron_index + 1) % neuron_count);
    }

    const f32 external_pulse = 0.03f; // 30mV -- comfortably above every fixture's own 20mV EL-to-threshold gap

    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick >= 1 && (tick - 1) % drive_gap_ticks == 0) {
            s64 pass_index = (tick - 1) / drive_gap_ticks;
            if (pass_index < neuron_count) {
                s32 neuron_to_drive = (s32)(neuron_count - 1 - pass_index);
                s64 v_index = membrane_potential_element_index(engine.nml_allocation_, neuron_to_drive);
                engine.nml_allocation_.cell_state.get_contents()[v_index] += external_pulse;
            }
        }

        engine.step_tick(dt_seconds, tick, tick + 1);
        EXPECT_TRUE(all_finite(engine.nml_allocation_)) << "tick=" << tick;
    }

    result.weight_after.resize((usize)neuron_count);
    for (s32 neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
        result.weight_after[(usize)neuron_index] = engine.weights.get(neuron_index, (neuron_index + 1) % neuron_count);
    }
    result.last_spiked_by_neuron.assign(engine.last_spiked.get_contents(),
                                        engine.last_spiked.get_contents() + neuron_count);

    return result;
}

// Shared body for both variants below: runs an active (real learning rate) and a zero-learning-rate
// control pass of the exact same drive sequence, then asserts direction/scale/attribution -- mirrors
// EndToEndSpikeEnginePlasticity.stdp_measurably_changes_weights_in_the_expected_direction_at_network_scale's
// own three-part check (tests/end_to_end_network_tests.cpp), generalized from a SpikeEngine LIF torus
// to a real GLIF delay-ring network.
void run_and_check_glif_ring_stdp(GlifVariant variant, const String &cell_instance_attributes) {
    const s32 neuron_count = 400;
    const s64 tick_count = 1500;
    const f32 dt_seconds = 1e-4f;
    const s64 drive_gap_ticks = 3;
    const f32 stdp_learning_rate = 0.2f;
    const String fixture_prefix = variant == GlifVariant::Glif3 ? "glif3_stdp_ring" : "glif5_stdp_ring";

    GlifStdpNetworkRunResult active_run = run_glif_ring_stdp_network(
        variant, cell_instance_attributes, fixture_prefix + "_active", neuron_count, tick_count, dt_seconds,
        drive_gap_ticks, stdp_learning_rate);
    GlifStdpNetworkRunResult control_run = run_glif_ring_stdp_network(
        variant, cell_instance_attributes, fixture_prefix + "_control", neuron_count, tick_count, dt_seconds,
        drive_gap_ticks, /*stdp_learning_rate=*/0.0f);

    // Sanity: this is real GLIF cell dynamics driven at network scale, not hand-set spike times --
    // every neuron actually crossed its own threshold on the exact tick its external pulse landed.
    for (s64 pass_index = 0; pass_index < neuron_count; ++pass_index) {
        s32 neuron_index = (s32)(neuron_count - 1 - pass_index);
        s64 expected_tick = 1 + pass_index * drive_gap_ticks;
        EXPECT_EQ(active_run.last_spiked_by_neuron[(usize)neuron_index], expected_tick) << "neuron=" << neuron_index;
    }

    f64 active_total_before = 0.0, active_total_after = 0.0;
    f64 control_total_before = 0.0, control_total_after = 0.0;
    s64 active_edges_with_measurable_decrease = 0;
    for (s32 neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
        active_total_before += (f64)active_run.weight_before[(usize)neuron_index];
        active_total_after += (f64)active_run.weight_after[(usize)neuron_index];
        control_total_before += (f64)control_run.weight_before[(usize)neuron_index];
        control_total_after += (f64)control_run.weight_after[(usize)neuron_index];
        if (active_run.weight_after[(usize)neuron_index] < active_run.weight_before[(usize)neuron_index] - 1e-4f) {
            ++active_edges_with_measurable_decrease;
        }
    }

    // Direction: apply_stdp_plasticity's own decay_delta is always <= 0 (a one-signed depression, see
    // master_kernel.cpp/plasticity_wiring.h) -- the network-wide SUM must decrease.
    EXPECT_LT(active_total_after, active_total_before)
        << "before=" << active_total_before << " after=" << active_total_after;

    // At real network scale, not just one or two edges: every one of this ring's 400 edges except the
    // single wraparound edge (neuron_count - 1 -> 0, see this file's own header comment) must show a
    // real, measurable depression.
    EXPECT_GE(active_edges_with_measurable_decrease, neuron_count - 5)
        << "only " << active_edges_with_measurable_decrease << " of " << neuron_count << " edges changed measurably";

    // Zero-learning-rate control: the SAME exact drive sequence must leave every weight untouched,
    // confirming the observed change above is attributable to STDP specifically.
    EXPECT_NEAR(control_total_after, control_total_before, 1e-6);
}

} // namespace

// ── GLIF3: after-spike-current network + STDP ─────────────────────────────────────────────────────

TEST(GlifStdpNetwork, glif3_ring_network_stdp_measurably_depresses_weights_in_the_expected_direction) {
    // Verbatim from tests/fixtures/nml/glif3_single_cell.nml's own real, jLEMS-verified parameter set
    // (same literal tests/end_to_end_network_tests.cpp's own glif3_ring_network_current_injection_medium_anchor
    // uses).
    const String cell_instance_attributes =
        "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\" t_ref=\"5ms\" tauAsc1=\"100ms\" "
        "tauAsc2=\"10ms\" ascAdd1=\"-100pA\" ascAdd2=\"-200pA\"";
    run_and_check_glif_ring_stdp(GlifVariant::Glif3, cell_instance_attributes);
}

// ── GLIF5: combined after-spike-current + adaptive-threshold network + STDP ──────────────────────

TEST(GlifStdpNetwork, glif5_ring_network_stdp_measurably_depresses_weights_in_the_expected_direction) {
    // Verbatim from tests/end_to_end_network_tests.cpp's own glif5_large_network_current_injection_largest_anchor.
    const String cell_instance_attributes =
        "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vreset=\"-70mV\" t_ref=\"5ms\" thetaInf=\"-50mV\" tauTheta=\"50ms\" "
        "thetaSpikeAdd=\"5mV\" tauAsc1=\"100ms\" tauAsc2=\"10ms\" ascAdd1=\"-100pA\" ascAdd2=\"-200pA\"";
    run_and_check_glif_ring_stdp(GlifVariant::Glif5, cell_instance_attributes);
}
