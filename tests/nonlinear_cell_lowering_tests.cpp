#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>
#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/stimulus_schedule.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Nonlinear point-cell lowering (ticket #63 [F2]) ──────────────────────────────────────────────
//
// Lowers the real vendored `izhikevich2007Cell`/`adExIaFCell`/`fitzHughNagumoCell`/
// `hindmarshRose1984Cell` ComponentTypes (third_party/neuroml2/std_lib/Cells.xml -- read directly,
// not reproduced/simplified here) through the SAME pipeline cell_lowering_tests.cpp's GLIF fixtures
// already exercise, and additionally drives each one through a real SpikeEngine (ticket #6) with
// a pulseGenerator current step (ticket #58's stimulus_schedule.h), verifying it reproduces that
// cell's own well-known textbook firing pattern.
//
// Front-end gaps this ticket's real-cell lowering exposed and closed (all narrowly scoped, all
// documented at their own definition site -- expression_lowering.h/.cpp, cell_lowering.cpp,
// nml.cpp/resolve.cpp -- not re-derived here):
//   - `^` (pow) and single-argument function-call syntax (`exp(...)`) -- neither existed in the
//     minimal expression parser before this ticket; both map onto ir.h ops that already existed
//     (BinaryOpcode::Pow, UnaryOpcode::Exp), so no IR/gpu_source.cpp change was needed.
//   - `<Constant>` elements (e.g. `fitzHughNagumoCell`'s `SEC`, `hindmarshRose1984Cell`'s `MSEC`)
//     were not parsed at all before this ticket -- now cataloged as an ordinary Parameter + a
//     synthesized Fixed pin.
//   - `t` (LEMS's own "current simulation time") -- needed by `adExIaFCell`'s real refractory regime
//     (`lastSpikeTime <- t`, `t .gt. lastSpikeTime + refract`) -- resolved to `tick * dt` inline.
//   - `.and.`/`.or.` compound OnCondition tests (`hindmarshRose1984Cell`'s own
//     `test="v .gt. 0 .and. spiking .lt. 0.5"` edge-detector idiom, also used by several other real
//     vendored cell types) -- the condition parser only supported one bare comparison before.
//   - The F1 active-set classifier (`cell_dynamics_are_closed_form_advanceable`) now transitively
//     inlines a plain-value DerivedVariable's own definition when checking affine-ness -- every one
//     of these four real cells hides its actual nonlinear term behind exactly this indirection
//     (`izhikevich2007Cell`'s `v` TimeDerivative is just `iMemb / C`; `hindmarshRose1984Cell` nests
//     three levels deep) -- ticket #62's own classifier only treated a DerivedVariable's name as an
//     always-affine opaque leaf, which its own header comment flagged as a gap "no Phase-1 GLIF
//     fixture" exercised at the time; this ticket's real cells are exactly that gap.

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

const TypeLibraryEntry &type_library_entry_for(const ModelSpecification &specification, const String &bound_instance_id) {
    for (const auto &entry : specification.type_library) {
        if (entry.bound_instance_id == bound_instance_id) return entry;
    }
    throw std::runtime_error("no type library entry for '" + bound_instance_id + "'");
}

// Recomputes the active-set x nonlinear-dynamics classification (ticket #62 [F1]) directly via
// cell_dynamics_are_closed_form_advanceable/gather_regime_info (cell_lowering.h), the SAME classifier
// lower_cell_to_ir itself calls internally -- IrProgram no longer carries a stored
// closed_form_advanceable tag (this cleanup).
bool is_closed_form_advanceable(const TypeLibraryEntry &entry) {
    const CellType &cell = std::get<CellType>(entry.dynamics.flattened);
    return cell_dynamics_are_closed_form_advanceable(cell, gather_regime_info(cell));
}

// Builds a single-neuron model around one REAL vendored Cells.xml ComponentType (included directly
// via NML_Parser::STANDARD_LIBRARY_PATH -- nml_tests.cpp's own
// `does_not_schema_gate_included_files` precedent for pulling in the real std-lib bundle), optionally
// driven by a `pulseGenerator` current step wired via `explicitInput` (ticket #58's own
// stimulus_schedule_tests.cpp fixture pattern). `stimulus_xml` is a pair of
// (top-level pulseGenerator element, network-body explicitInput element), both empty for the
// lowering-only tests that don't need a live stimulus.
ModelSpecification build_real_cell_model(
    const String &fixture_id, const String &tag_name, const String &instance_attributes,
    const String &pulse_generator_xml = "", const String &explicit_input_xml = ""
) {
    NML_Parser path_helper;
    String cells_file = path_helper.STANDARD_LIBRARY_PATH + "/Cells.xml";

    write_temp_file("spikecorec_nonlinear_cell_" + fixture_id + "_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"NonlinearCell" + fixture_id + "Content\">"
        "  <include href=\"" + cells_file + "\"/>"
        + pulse_generator_xml +
        "  <" + tag_name + " id=\"cellInstance\" " + instance_attributes + "/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"cellInstance\" size=\"1\"/>"
        + explicit_input_xml +
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_nonlinear_cell_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"NonlinearCell" + fixture_id + "Top\">"
        "  <include href=\"spikecorec_nonlinear_cell_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// One `IrProgram` per `model.type_library` entry, in the same index order (allocator.h's own
// required shape) -- a Cell-category entry gets its real `lower_cell_to_ir` program; the
// pulseGenerator's own Inputs-category entry gets an empty placeholder (allocator.h's own documented
// convention: "an empty `.alloc` correctly contributes nothing" -- Phase-1 stimulus is host-precomputed
// via ticket #58's stimulus_schedule.h, not IR-lowered at all).
spikecorec::Vector<IrProgram> build_ir_programs_for_model(const ModelSpecification &model) {
    spikecorec::Vector<IrProgram> programs;
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) {
            programs.push_back(lower_cell_to_ir(entry));
        } else {
            programs.push_back(IrProgram{});
        }
    }
    return programs;
}

// ── SpikeEngine harness shared by every acceptance test below ───────────────────────────────────
//
// A single-neuron, single-population, no-edges SpikeEngine, mirroring
// master_kernel_tests.cpp's own established fixture-construction pattern (ModelAllocation +
// WeightMatrix-with-no-edges + the fixed set of ModelRuntimeBuffers GpuPointers) but through
// SpikeEngine's own unified ModelSpecification constructor, which now owns all of that internally --
// generalized to whichever `IrProgram program` this ticket's own lower_cell_to_ir produced for a
// real cell instead of that file's own hand-built GLIF1-equivalent IrProgram.
struct SingleNeuronHarness {
    // `model` is NOT owned/copied here (ModelSpecification isn't copyable -- it holds a WeightMatrix
    // inside `adjacency`) -- callers keep their own local ModelSpecification alive for at least this
    // harness's own lifetime (every test below does) and may keep using it afterward (e.g. to build a
    // StimulusSchedule from the same model).
    SpikeEngine engine;

    explicit SingleNeuronHarness(ModelSpecification &model)
        : engine(model, build_ir_programs_for_model(model)) {}

    f32 &state(s32 offset) { return engine.nml_allocation_.cell_state.get_contents()[offset]; }

    // ModelAllocation::cell_state is always zero-initialized (allocate_model's own
    // allocate_zeroed_f32) -- the allocator does not (yet) interpret a StateDirective's own
    // `initial_value` (OnStart's seed, cell_lowering.cpp's own doc comment on StateDirective) to seed
    // it, matching master_kernel_tests.cpp's own established workaround for this exact gap
    // (`std::fill(allocation.cell_state.get_contents(), ..., resting_mp)` right after allocate_model).
    // `initial_values` is this cell's own state-vector seed, in the SAME state-offset order
    // `state(offset)` uses (allocator.cpp/master_kernel.cpp's own `StateDirective` declaration order).
    void seed_state(std::initializer_list<f32> initial_values) {
        s32 offset = 0;
        for (f32 value : initial_values) state(offset++) = value;
    }
};

} // namespace

// ── izhikevich2007Cell (v, u) ─────────────────────────────────────────────────────────────────────
//
// Parameters: Izhikevich (2007), Dynamical Systems in Neuroscience (MIT Press), Ch. 8's "simple
// model" cortical Regular Spiking (RS) pyramidal-neuron parameter set (the same widely-reproduced
// table cited by e.g. Brian2's/NEURON's own izhikevich2007Cell-equivalent example models):
// C=100pF, k=0.7nS/mV, vr=-60mV, vt=-40mV, a=0.03/ms, b=-2nS, c=-50mV, d=100pA, vpeak=35mV.
// Canonical pattern: Regular Spiking -- a sustained, roughly periodic tonic spike train under a
// constant current step (Izhikevich 2003/2007's own hallmark RS trace), typically with a brief
// initial-ISI transient that settles into an essentially constant firing rate.

TEST(NonlinearCellLowering, izhikevich2007Cell_lowers_and_is_classified_nonlinear) {
    ModelSpecification model = build_real_cell_model(
        "izhikevich_lowering", "izhikevich2007Cell",
        "v0=\"-60mV\" C=\"100pF\" k=\"0.7nS_per_mV\" vr=\"-60mV\" vt=\"-40mV\" vpeak=\"35mV\" "
        "a=\"0.03per_ms\" b=\"-2nS\" c=\"-50mV\" d=\"100pA\"");
    const TypeLibraryEntry &entry = type_library_entry_for(model, "cellInstance");
    ASSERT_EQ(entry.category, TypeLibraryCategory::Cell);

    IrProgram program = lower_cell_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "izhikevich2007Cell");

    // ticket #62 [F1]: the nonlinear `k*(v-vr)*(v-vt)` product hides behind `iMemb`'s own
    // DerivedVariable indirection -- must still be tagged nonlinear (ticket #63's own transitive
    // DerivedVariable-inlining extension to the classifier).
    EXPECT_FALSE(is_closed_form_advanceable(entry));

    String printed = print_ir_program(program);
    EXPECT_NE(printed.find("mul"), String::npos);
    EXPECT_NE(printed.find("expose v"), String::npos);
    EXPECT_NE(printed.find("expose u"), String::npos);
}

TEST(NonlinearCellAcceptance, izhikevich2007Cell_regular_spiking_pattern_under_current_step) {
    const f32 dt = 1e-4f;             // 0.1ms
    const s64 delay_ticks = 100;      // 10ms
    const s64 duration_ticks = 2000;  // 200ms
    const s64 tick_count = delay_ticks + duration_ticks + 200;

    ModelSpecification model = build_real_cell_model(
        "izhikevich_acceptance", "izhikevich2007Cell",
        "v0=\"-60mV\" C=\"100pF\" k=\"0.7nS_per_mV\" vr=\"-60mV\" vt=\"-40mV\" vpeak=\"35mV\" "
        "a=\"0.03per_ms\" b=\"-2nS\" c=\"-50mV\" d=\"100pA\"",
        "  <pulseGenerator id=\"pulseGen1\" delay=\"10ms\" duration=\"200ms\" amplitude=\"150pA\"/>",
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGen1\"/>");

    SingleNeuronHarness harness(model);
    harness.seed_state({-0.06f, 0.0f}); // OnStart: v=v0=-60mV, u=0 (allocate_model doesn't seed OnStart yet)
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)dt);

    spikecorec::Vector<f32> voltage_trace((usize)tick_count);
    spikecorec::Vector<s64> spike_ticks;

    for (s64 tick = 0; tick < tick_count; ++tick) {
        // ticket #58 machinery: this cell's own `iSyn` DerivedVariable (select/reduce="add" over
        // attached synapses) aliases straight to `network_inputs` (arch §3.2) -- exactly the same
        // channel a real synaptic current would scatter into, and exactly what this cell's own
        // `iMemb = k*(v-vr)*(v-vt) + iSyn - u` equation expects an external current step to enter
        // through. Writing it directly here (rather than through the k^2-tree propagate stage) is
        // deliberate: this single-neuron model has no edges at all, and this is the same "host
        // supplies the per-tick value directly" contract stimulus_schedule.h's own header comment
        // documents for the not-yet-wired (ticket #61) splice into a generated master kernel.
        harness.engine.network_inputs.get_contents()[0] = (f32)schedule.current_at(0, tick);
        harness.engine.step_tick(dt, tick, tick + 1);

        voltage_trace[(usize)tick] = harness.state(0); // state offset 0 == "v" (first declared StateVariable)
        // ticket #6's own documented contract (master_kernel.h): the fixed propagate stage this same
        // step_tick call already ran reads-then-CLEARS emit_port_flags["spike"] before returning, so
        // checking it here would always see false -- last_spiked[neuron_index]==tick (set by that
        // same propagate stage) is the correct post-step_tick spike signal, matching
        // master_kernel_tests.cpp's own established LIF-equivalence test precedent.
        if (harness.engine.last_spiked.get_contents()[0] == tick) spike_ticks.push_back(tick);
    }

    ASSERT_GE(spike_ticks.size(), 3u) << "expected several spikes during the current step (regular spiking)";

    // Regular spiking's own signature: after settling past the first (adaptation-influenced) ISI,
    // successive inter-spike intervals stay roughly constant -- compare the LAST two ISIs.
    usize spike_count = spike_ticks.size();
    s64 last_isi = spike_ticks[spike_count - 1] - spike_ticks[spike_count - 2];
    s64 second_last_isi = spike_ticks[spike_count - 2] - spike_ticks[spike_count - 3];
    f64 isi_relative_difference = std::fabs((f64)(last_isi - second_last_isi)) / (f64)std::max(last_isi, second_last_isi);
    EXPECT_LT(isi_relative_difference, 0.25)
        << "last_isi=" << last_isi << " second_last_isi=" << second_last_isi
        << " -- regular spiking's own settled firing rate should be roughly constant tick-to-tick";

    // Membrane potential resets below vpeak on every spike (arch: OnCondition v>vpeak -> v<-c).
    for (s64 spike_tick : spike_ticks) {
        EXPECT_LT(voltage_trace[(usize)spike_tick], 0.036f) << "tick=" << spike_tick;
    }
}

// ── adExIaFCell (v, w) ────────────────────────────────────────────────────────────────────────────
//
// Parameters: Brette & Gerstner (2005), "Adaptive Exponential Integrate-and-Fire Model as an
// Effective Description of Neuronal Activity", J Neurophysiol 94:3637-3642 -- their own Figure 4a
// "regular spiking (adapting)" parameter set (also the parameter set Brian2's own AdEx documentation
// reproduces): C=281pF, gL=30nS, EL=-70.6mV, VT=-50.4mV, delT=2mV, a=4nS, tauw=144ms, b=80.5pA,
// reset=-70.6mV. `refract` is set to (near-)0 -- the original Brette-Gerstner model has no explicit
// refractory clamp at all; `thresh` (numerical spike-detection cutoff, distinct from the exponential
// term's own soft threshold VT) is set to 0mV, matching the paper's own spike-detection convention.
// Canonical pattern: spike-frequency ADAPTATION -- successive inter-spike intervals under a constant
// current step LENGTHEN over time (the adaptation current `w` builds up with each spike), settling
// toward a slower asymptotic rate -- this is the feature that qualitatively distinguishes AdEx's own
// "regular spiking (adapting)" mode from Izhikevich's own non-adapting regular spiking above.

TEST(NonlinearCellLowering, adExIaFCell_lowers_and_is_classified_nonlinear) {
    ModelSpecification model = build_real_cell_model(
        "adex_lowering", "adExIaFCell",
        "C=\"281pF\" gL=\"30nS\" EL=\"-70.6mV\" VT=\"-50.4mV\" thresh=\"0mV\" reset=\"-70.6mV\" "
        "delT=\"2mV\" tauw=\"144ms\" refract=\"0ms\" a=\"4nS\" b=\"80.5pA\"");
    const TypeLibraryEntry &entry = type_library_entry_for(model, "cellInstance");
    ASSERT_EQ(entry.category, TypeLibraryCategory::Cell);

    IrProgram program = lower_cell_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "adExIaFCell");

    // ticket #62 [F1]: the `exp((v-VT)/delT)` term hides behind `iMemb`'s own DerivedVariable
    // indirection, and is only reachable via `emit_expression`'s new function-call support.
    EXPECT_FALSE(is_closed_form_advanceable(entry));

    String printed = print_ir_program(program);
    EXPECT_NE(printed.find("exp "), String::npos) << printed;
    EXPECT_NE(printed.find("expose v"), String::npos);
    EXPECT_NE(printed.find("expose w"), String::npos);
    // ticket #63's own `t` support (`lastSpikeTime <- t`) -- lowers to `mul <dest>, tick, dt`.
    EXPECT_NE(printed.find("mul"), String::npos);
}

TEST(NonlinearCellAcceptance, adExIaFCell_spike_frequency_adaptation_under_current_step) {
    const f32 dt = 5e-5f; // 0.05ms
    const s64 delay_ticks = 1000;    // 50ms
    const s64 duration_ticks = 16000; // 800ms
    const s64 tick_count = delay_ticks + duration_ticks + 1000;

    ModelSpecification model = build_real_cell_model(
        "adex_acceptance", "adExIaFCell",
        "C=\"281pF\" gL=\"30nS\" EL=\"-70.6mV\" VT=\"-50.4mV\" thresh=\"0mV\" reset=\"-70.6mV\" "
        "delT=\"2mV\" tauw=\"144ms\" refract=\"0ms\" a=\"4nS\" b=\"80.5pA\"",
        "  <pulseGenerator id=\"pulseGen1\" delay=\"50ms\" duration=\"800ms\" amplitude=\"1000pA\"/>",
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGen1\"/>");

    SingleNeuronHarness harness(model);
    harness.seed_state({-0.0706f, 0.0f, 0.0f}); // OnStart: v=EL=-70.6mV, w=0 (lastSpikeTime defaults to 0)
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)dt);

    spikecorec::Vector<s64> spike_ticks;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        harness.engine.network_inputs.get_contents()[0] = (f32)schedule.current_at(0, tick);
        harness.engine.step_tick(dt, tick, tick + 1);
        // ticket #6's own documented contract (master_kernel.h): the fixed propagate stage this same
        // step_tick call already ran reads-then-CLEARS emit_port_flags["spike"] before returning, so
        // checking it here would always see false -- last_spiked[neuron_index]==tick (set by that
        // same propagate stage) is the correct post-step_tick spike signal, matching
        // master_kernel_tests.cpp's own established LIF-equivalence test precedent.
        if (harness.engine.last_spiked.get_contents()[0] == tick) spike_ticks.push_back(tick);
    }

    ASSERT_GE(spike_ticks.size(), 4u) << "expected several spikes during the current step (adapting regular spiking)";

    spikecorec::Vector<s64> inter_spike_intervals;
    for (usize index = 1; index < spike_ticks.size(); ++index) {
        inter_spike_intervals.push_back(spike_ticks[index] - spike_ticks[index - 1]);
    }

    // Spike-frequency adaptation's own signature: the LAST ISI is noticeably longer than the FIRST
    // ISI (successive intervals lengthen as the adaptation current `w` builds up).
    s64 first_isi = inter_spike_intervals.front();
    s64 last_isi = inter_spike_intervals.back();
    EXPECT_GT(last_isi, first_isi) << "first_isi=" << first_isi << " last_isi=" << last_isi
        << " -- AdEx regular-spiking-adapting mode should show successively lengthening ISIs";
    EXPECT_GT((f64)last_isi, 1.15 * (f64)first_isi)
        << "first_isi=" << first_isi << " last_isi=" << last_isi << " -- adaptation should be clearly visible";
}

// ── fitzHughNagumoCell (V, W) ─────────────────────────────────────────────────────────────────────
//
// The vendored `fitzHughNagumoCell` bakes its own equations' numeric coefficients directly into the
// ComponentType (`dV/dt = V - V^3/3 - W + I`, `dW/dt = 0.08*(V+0.7-0.8*W)`) -- the classic
// FitzHugh-Nagumo parameterization (FitzHugh 1961; see also Izhikevich 2007's own FHN chapter),
// a=0.7, b=0.8, eps=0.08 -- leaving only `I` (this ComponentType's sole real Parameter) free. This
// cell has no Attachments/iSyn (unlike the other three) and no OnStart, so its stimulus is driven by
// directly perturbing its own primary state variable `V` (mirroring master_kernel_tests.cpp's own
// established "external stimulus added straight into the membrane-potential-like state, arch §0.2"
// convention -- the one channel every one of this ticket's cells has, unlike network_inputs/iSyn,
// which this ComponentType alone doesn't declare); FHN_CURRENT_TO_DIMENSIONLESS_SCALE below is this
// test's own explicit, documented bridge from a genuine SI-current pulseGenerator (ticket #58
// machinery) to FHN's own dimensionless forcing convention.
// Canonical pattern: for I above roughly 0.3-0.4 (this exact a/b/eps parameterization's own Hopf
// bifurcation threshold), the system's single fixed point loses stability and FHN enters a SUSTAINED,
// PERIODIC (limit-cycle) oscillation -- repetitive "spiking" -- rather than settling back to rest
// after at most one excitable pulse. I=0.5 (used here) is well inside that oscillatory regime.

namespace {
// FHN's own `I` is LEMS dimension="none" (a bare dimensionless number); this test still drives it
// through a genuine SI-unit pulseGenerator (amplitude dimension="current", ticket #58 machinery) for
// a stimulus-protocol shape consistent with the other three cells, so this is this test's own
// explicit, chosen scale factor bridging real current (amperes) back to FHN's own dimensionless
// convention: amplitude="0.5nA" combined with this factor evaluates to exactly the dimensionless I=0.5
// cited above.
const f64 FHN_CURRENT_TO_DIMENSIONLESS_SCALE = 1e9; // 1 / 1nA
} // namespace

TEST(NonlinearCellLowering, fitzHughNagumoCell_lowers_and_is_classified_nonlinear) {
    ModelSpecification model = build_real_cell_model("fhn_lowering", "fitzHughNagumoCell", "I=\"0.5\"");
    const TypeLibraryEntry &entry = type_library_entry_for(model, "cellInstance");
    ASSERT_EQ(entry.category, TypeLibraryCategory::Cell);

    IrProgram program = lower_cell_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "fitzHughNagumoCell");

    // ticket #62 [F1]: `V^3` is a directly-embedded nonlinear term (no DerivedVariable indirection
    // needed here) -- only reachable via ticket #63's own `^` parser support.
    EXPECT_FALSE(is_closed_form_advanceable(entry));

    String printed = print_ir_program(program);
    EXPECT_NE(printed.find("pow"), String::npos) << printed;
    EXPECT_NE(printed.find("expose V"), String::npos);
    EXPECT_NE(printed.find("expose W"), String::npos);
}

TEST(NonlinearCellAcceptance, fitzHughNagumoCell_sustained_oscillation_under_current_step) {
    const f32 dt = 0.01f; // 10ms == 0.01 non-dimensional time units (SEC == 1s, ticket #63's own Constant support)
    const s64 delay_ticks = 1000;   // 10s
    const s64 duration_ticks = 20000; // 200s -- several oscillation periods (~30-40 non-dim units each)
    const s64 tick_count = delay_ticks + duration_ticks + 1000;

    ModelSpecification model = build_real_cell_model(
        "fhn_acceptance", "fitzHughNagumoCell", "I=\"0\"",
        "  <pulseGenerator id=\"pulseGen1\" delay=\"10s\" duration=\"200s\" amplitude=\"0.5nA\"/>",
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGen1\"/>");

    SingleNeuronHarness harness(model);
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)dt);

    spikecorec::Vector<f32> voltage_trace((usize)tick_count);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        harness.engine.step_tick(dt, tick, tick + 1);

        // Direct-state stimulus injection (see this file's own header comment above): this cell has
        // no network_inputs/iSyn channel at all, so the current step is added directly to `V`
        // (state offset 0), scaled by dt (SEC == 1s, so dt/SEC == dt numerically) -- applied AFTER
        // this tick's own step_tick call so the intrinsic cubic term is evaluated at the pre-tick V,
        // exactly matching a genuine combined `dV/dt = (intrinsic + I)/SEC` Euler step algebraically
        // (both additions distribute over the same dt/SEC scale).
        f64 injected_current = schedule.current_at(0, tick) * FHN_CURRENT_TO_DIMENSIONLESS_SCALE;
        harness.state(0) += (f32)(dt * injected_current);

        voltage_trace[(usize)tick] = harness.state(0);
    }

    // Count upward threshold-crossings of V (this ComponentType has no native spike/EventOut --
    // FHN is a continuous relaxation oscillator, arch §3.3 D1 -- so "spike count" here means the
    // number of times V crosses a fixed threshold from below, the standard way to count cycles of a
    // relaxation oscillator's fast upstroke).
    const f32 crossing_threshold = 1.0f;
    s32 crossing_count = 0;
    for (s64 tick = delay_ticks + 1; tick < tick_count; ++tick) {
        if (voltage_trace[(usize)(tick - 1)] < crossing_threshold && voltage_trace[(usize)tick] >= crossing_threshold) {
            ++crossing_count;
        }
    }

    EXPECT_GE(crossing_count, 3)
        << "expected several oscillation cycles during the current step (sustained periodic firing)";
}

// ── hindmarshRose1984Cell (v, y, z, spiking) ─────────────────────────────────────────────────────
//
// Parameters: Hindmarsh, J.L. and Rose, R.M. (1984), "A model of neuronal bursting using three
// coupled first order differential equations", Proc. R. Soc. Lond. B 221:87-102 -- their own
// canonical bursting parameter set: a=1, b=3, c=1, d=5, s=4, x1=-1.6 (x0=x1, the model's own
// "start at rest" initialization, with y0/z0 read off the y/z nullclines at x0). `r` (the
// slow/fast timescale-separation parameter, which the paper requires small for bursting) is set to
// 0.005 here rather than the paper's own more commonly cited r=0.001 -- a deliberate, documented
// judgment call trading a ~5x shorter (but still well within the "r small" bursting regime) burst
// period for a computationally tractable tick count, since burst period scales roughly as 1/r.
// `v_scaling` bridges the model's own dimensionless `x` into a physically-dimensioned `v` state
// variable (this ComponentType's own NeuroML-specific addition, arch §3.2's DerivedVariable
// mechanism) -- chosen here purely so `v`'s numeric range looks like a plausible membrane potential.
// The applied current (`iSyn`, entering the `x` equation as `iSyn/(C*v_scaling/MSEC)` once the unit
// bridge is unwound) is this ComponentType's own addition beyond the paper's original 3-equation
// autonomous system -- 60nA (with C=1nF, v_scaling=20mV) corresponds to a dimensionless forcing of
// ~3, inside the range the standard a/b/c/d/r/s/x1 parameterization above is well known to burst for.
// Canonical pattern: periodic BURSTING -- an initial fast transient settling into a robust, clearly
// REPEATING short-ISI/long-ISI rhythm (a period-2 "doublet" burst here: pairs of closely-spaced
// spikes separated by a longer, also-regular gap) -- Hindmarsh & Rose's own defining qualitative
// behavior (clustered spikes, not the roughly-uniform ISIs of plain regular spiking, and not a single
// decaying transient either -- the doublet rhythm persists for as long as the current step does).

TEST(NonlinearCellLowering, hindmarshRose1984Cell_lowers_and_is_classified_nonlinear) {
    ModelSpecification model = build_real_cell_model(
        "hindmarsh_rose_lowering", "hindmarshRose1984Cell",
        "a=\"1.0\" b=\"3.0\" c=\"1.0\" d=\"5.0\" r=\"0.005\" s=\"4.0\" x1=\"-1.6\" "
        "v_scaling=\"20mV\" x0=\"-1.6\" y0=\"-11.8\" z0=\"0.0\" C=\"1nF\"");
    const TypeLibraryEntry &entry = type_library_entry_for(model, "cellInstance");
    ASSERT_EQ(entry.category, TypeLibraryCategory::Cell);

    IrProgram program = lower_cell_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "hindmarshRose1984Cell");

    // ticket #62 [F1]: the `x^2`/`x^3` terms hide three DerivedVariable levels deep (`iMemb` reads
    // `phi`/`z`; `phi` reads `x`; `x` reads `v`) -- exactly the transitive-inlining gap this ticket
    // closes in cell_lowering.cpp's classifier.
    EXPECT_FALSE(is_closed_form_advanceable(entry));

    String printed = print_ir_program(program);
    EXPECT_NE(printed.find("pow"), String::npos) << printed;
    // ticket #63's own `.and.` compound-condition support (`v .gt. 0 .and. spiking .lt. 0.5`).
    EXPECT_NE(printed.find("and "), String::npos) << printed;
    EXPECT_NE(printed.find("expose v"), String::npos);
}

TEST(NonlinearCellAcceptance, hindmarshRose1984Cell_bursting_pattern_under_current_step) {
    const f32 dt = 5e-5f;          // 0.05ms
    const s64 delay_ticks = 400;      // 20ms
    const s64 duration_ticks = 24000;  // 1200ms
    const s64 tick_count = delay_ticks + duration_ticks + 400;

    ModelSpecification model = build_real_cell_model(
        "hindmarsh_rose_acceptance", "hindmarshRose1984Cell",
        "a=\"1.0\" b=\"3.0\" c=\"1.0\" d=\"5.0\" r=\"0.005\" s=\"4.0\" x1=\"-1.6\" "
        "v_scaling=\"20mV\" x0=\"-1.6\" y0=\"-11.8\" z0=\"0.0\" C=\"1nF\"",
        "  <pulseGenerator id=\"pulseGen1\" delay=\"20ms\" duration=\"1200ms\" amplitude=\"60nA\"/>",
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGen1\"/>");

    SingleNeuronHarness harness(model);
    // OnStart: v=x0*v_scaling=-1.6*20mV=-32mV, y=y0=-11.8, z=z0=0, spiking=0 (allocate_model doesn't
    // seed OnStart yet -- see SingleNeuronHarness::seed_state's own doc comment).
    harness.seed_state({-0.032f, -11.8f, 0.0f, 0.0f});
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)dt);

    spikecorec::Vector<s64> spike_ticks;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        harness.engine.network_inputs.get_contents()[0] = (f32)schedule.current_at(0, tick);
        harness.engine.step_tick(dt, tick, tick + 1);
        // ticket #6's own documented contract (master_kernel.h): the fixed propagate stage this same
        // step_tick call already ran reads-then-CLEARS emit_port_flags["spike"] before returning, so
        // checking it here would always see false -- last_spiked[neuron_index]==tick (set by that
        // same propagate stage) is the correct post-step_tick spike signal, matching
        // master_kernel_tests.cpp's own established LIF-equivalence test precedent.
        if (harness.engine.last_spiked.get_contents()[0] == tick) spike_ticks.push_back(tick);
    }

    ASSERT_GE(spike_ticks.size(), 20u) << "expected many spikes across the current step (a sustained bursting rhythm)";

    spikecorec::Vector<s64> inter_spike_intervals;
    for (usize index = 1; index < spike_ticks.size(); ++index) {
        inter_spike_intervals.push_back(spike_ticks[index] - spike_ticks[index - 1]);
    }

    // Bursting's own signature: a bimodal ISI distribution -- some SHORT (within-burst) intervals and
    // some much LONGER (between-burst) intervals -- unlike regular spiking's roughly-uniform ISIs.
    // Checked over the SECOND HALF of the spike train specifically (not just the whole-trace min/max),
    // so this asserts the settled, REPEATING doublet-burst rhythm itself (this file's own header
    // comment above) -- not merely the initial fast-then-decelerating transient every run starts with.
    usize second_half_start = inter_spike_intervals.size() / 2;
    s64 min_isi = inter_spike_intervals[second_half_start];
    s64 max_isi = inter_spike_intervals[second_half_start];
    for (usize index = second_half_start; index < inter_spike_intervals.size(); ++index) {
        min_isi = std::min(min_isi, inter_spike_intervals[index]);
        max_isi = std::max(max_isi, inter_spike_intervals[index]);
    }
    EXPECT_GT((f64)max_isi, 1.3 * (f64)min_isi)
        << "min_isi=" << min_isi << " max_isi=" << max_isi
        << " -- the settled bursting rhythm should keep alternating between a short (within-burst) "
           "and a longer (between-burst) interval, not settle into uniform tonic-rate ISIs";
}
