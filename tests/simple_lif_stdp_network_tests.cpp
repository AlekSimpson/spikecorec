#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/master_kernel.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/plasticity_wiring.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/synapse_lowering.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// plasticity_wiring.h pulls in core/engine.h, whose own `using namespace spikecorec::log;` (a
// pre-existing, file-scope directive in that header, not something this ticket's own file should
// touch) makes the ALIAS TEMPLATE `spikecorec::log::Vector` ambiguous with `spikecorec::Vector` for
// plain unqualified `Vector<...>` lookup (two distinct `using namespace` directives at the same scope,
// each making a same-named alias TEMPLATE visible) -- unlike plain type aliases such as `String`,
// where two aliases naming the identical underlying type are never ambiguous, two alias TEMPLATE
// declarations are treated as distinct entities regardless of what they expand to, and neither a
// using-declaration nor an ordinary re-declaration for one of them outranks the other's
// using-directive at the same scope (tried both; both still left it ambiguous). So this file spells
// out `spikecorec::Vector<...>` fully everywhere below, rather than the bare `Vector<...>` most other
// *_tests.cpp files in this tree use -- the one file-wide style deviation this particular header
// clash forces.

// ── Ticket #101 [T2]: simple single-variable LIF + STDP network tests at scale ─────────────────────
//
// Complements ticket #100's GLIF-focused coverage (#61/#67) with the SIMPLEST possible spiking-neuron
// model -- a hand-authored, minimal, non-GLIF NeuroML ComponentType -- run at real network scale (up
// to 2000 neurons / 10000 ticks), to isolate whether failures at scale are GLIF-specific or
// fundamental to the shared engine/codegen machinery. As with tickets #61/#67, this file's validation
// is INTERNAL-CONSISTENCY ONLY (no jLEMS reference) -- confirmed with the user for this ticket.
//
// ── The cell model (SimpleAccumulatorCell) ──────────────────────────────────────────────────────────
// - One state variable `v`.
// - No leak/decay: `v[t] = v[t-1] + network_inputs[t]`, exactly (not scaled by dt).
// - One global (population-wide, baked) `spikeThreshold` parameter; fires when `v` exceeds it.
// - On fire: a standard `EventOut` spike, scattered through the normal weighted per-edge k^2-tree
//   propagate mechanism to every downstream neighbor (every neighbor receives its OWN edge weight).
// - Reset-on-fire: `v <- restingPotential`, a SECOND global parameter, distinct from the threshold.
//
// ── The dt-cancellation trick (the "how do you get a literal, unscaled accumulator out of a
// forward-Euler `v += dt * TimeDerivative` integrator" question this ticket's own body poses) ────────
// This engine's forward-Euler fallback (expression_lowering.cpp's `lower_time_derivative`, the path
// any TimeDerivative that isn't a recognized pure-exponential-decay shape takes) always lowers
// `TimeDerivative variable="v" value="RHS"` to exactly:
//     t0 = RHS
//     t0 = t0 * dt
//     v  = v + t0
// So writing `value="network_inputs / dt"` for RHS makes the emitted `* dt` step exactly CANCEL the
// division this ComponentType's own TimeDerivative wrote: `t0 = network_inputs/dt; t0 = t0*dt` puts
// `t0` back to (within ordinary f32 round-trip rounding) `network_inputs`, so `v += network_inputs`
// literally, for ANY dt -- not just dt=1. `dt` is a reserved engine name resolvable inside a
// TimeDerivative expression the exact same way `network_inputs`/`tick` already are
// (expression_lowering.cpp's own `RESERVED_ENGINE_NAMES = {"dt", "tick", "network_inputs"}`,
// `LoweringContext::resolve_identifier`) -- this is not a new capability this ticket had to add, only
// a new (documented) way of USING an existing one. `SimpleAccumulatorCellIsolation.
// ir_lowering_shows_the_dt_cancelling_network_inputs_over_dt_expression` below asserts the exact
// emitted `.tick` instructions directly; `SimpleAccumulatorCellIsolation.
// accumulate_threshold_reset_cycle_matches_hand_computed_trajectory` confirms the NUMERIC consequence
// against a hand-computed trajectory.
// Precision note (documented honestly, per this ticket's own instruction, mirroring how ticket #66
// documented its own real-formalism gap): `(x/dt)*dt` is NOT bit-exact-identical to `x` in IEEE 754
// float32 in general (division then multiplication by the same nonzero value can differ by ~1 ULP due
// to double rounding) -- this is a real, tiny discrepancy, not silently glossed over. Every numeric
// check in this file uses a tolerance (>= 1e-5 in absolute terms, against values of order 0.1-1.0)
// many orders of magnitude looser than that rounding, so it is never what any assertion here actually
// depends on.
//
// ── Discrete spike-array driving mechanism (this file's own self-contained plumbing; ticket #100 is
// being worked in parallel and had zero commits beyond `nightly` at the time this ticket's own
// implementation started -- see this ticket's own coordination note -- so this is built independently
// here, not shared) ──────────────────────────────────────────────────────────────────────────────────
// A host-provided `spikecorec::Vector<s32>` of 0s/1s, one entry per tick, drives a DESIGNATED input neuron
// directly: on a tick where the array holds 1, the test driver writes
// `buffers.emit_port_flags["spike"][input_neuron_index] = true` BEFORE calling
// `AssembledModel::step_tick` for that tick. Master-kernel.cpp's own generated `_tick` kernels only
// ever WRITE `emit_<port>[neuron] = true` inside their own `if spiked { emit spike }` block (never an
// unconditional `= false` -- see gpu_source.cpp's `emit_emit`), and the FIXED propagate stage is what
// clears it back to false, immediately after consuming it (master_kernel.cpp's
// `spikecorec_master_propagate`/`..._ring`: `if (!emit_spike[neuron_index]) return; emit_spike[...] =
// false;`) -- so a host-side pre-set `true` survives untouched into that same tick's real propagate
// dispatch (real k^2-tree scatter, real `last_spiked` bookkeeping, real delay-ring interaction if
// enabled) exactly as if the neuron's OWN dynamics had spiked it, with ZERO new engine code. The
// forced neuron's own `v`/reset cycle is untouched by this (propagate never reads/writes cell state) --
// documented and expected, matching how a real external spike-source input is a synthetic drive, not
// itself simulated through this cell's own accumulate/threshold dynamics.
//
// ── Known limitations / scope boundaries a reviewer should scrutinize closely ───────────────────────
// 1. AssembledModel's own fixed propagate stage still does not invoke a real per-edge SYNAPSE
//    ComponentType's own `.tick`/`.deliver_<port>` dynamics (master_kernel.h's own long-documented
//    scope boundary, unchanged since ticket #6, carried through #61/#64/#66/#67 unchanged). This
//    file's own `SimpleStdpSynapse` ComponentType is included in every generated network purely so
//    `find_stdp_spec` (ticket #66) can structurally detect its baked `tauPlus`/`tauMinus`/`aPlus`/
//    `aMinus` parameter VALUES -- its own Dynamics (`g`, `OnEvent`, ...) are compiled (via
//    `lower_synapse_to_ir`) but never dispatched by AssembledModel, exactly as documented.
// 2. A DEEPER, newly-surfaced architecture gap this ticket's own implementation discovered while
//    trying to satisfy "combine real per-edge storage + delays + ACTIVE STDP together": ticket #66's
//    own `apply_stdp_wiring` drives `SpikeEngine::enable_plasticity` -- the LEGACY, pre-NML hardcoded
//    LIF engine (`src/core/engine.cpp`, the ONLY place `step_apply_hebbian_update` is ever actually
//    invoked, from `step`/`step_no_active_optimization`) -- NOT `AssembledModel` (the NML-codegen
//    master-kernel path this ticket's own custom cell/delay-ring/real-per-edge-storage all run
//    through). Grepping `src/core/`, `src/metal/`, `src/cuda/`, and `master_kernel.cpp` confirms: the
//    legacy engine has NO delay-ring support at all (delay_ring.h/.cpp is exclusively an
//    AssembledModel-side subsystem), and AssembledModel has NO automatic STDP/plasticity dispatch at
//    all (master_kernel.h's own documented "STDP/plasticity (stage 7) is not part of the fixed
//    propagate stage" -- ticket #66's own scope was wiring the LEGACY engine, not this one). So there
//    is currently NO single simulation driver in this codebase that runs BOTH the delay ring AND the
//    automatic Hebbian kernel. `SimpleAccumulatorNetwork.
//    anchor_b_combines_real_per_edge_synapse_storage_spike_delays_and_active_stdp_together` below
//    resolves this the same way this codebase already resolves "the fixed stage doesn't do X yet" in
//    other places (e.g. run_delayed_coupling_network's own manual ring-slot stimulus injection): it
//    reuses `find_stdp_spec`/`map_stdp_spec_to_learning_rate` (ticket #66, unchanged) to get a
//    genuinely model-specified learning rate, then reproduces kernels.metal's OWN
//    `step_apply_hebbian_update` call-site formula EXACTLY (`decay_delta = -learning_rate *
//    pow(tick_delta, -3)`, then `WeightMatrix::update(source, target, decay_delta, 0.5f, 1.0f, 1)` --
//    the same internal 0.5/1.0 constants the real kernel bakes in) from the TEST'S OWN per-tick driver
//    loop, calling the REAL, already-implemented, standalone `WeightMatrix::update()` API
//    (weight_matrix.h's own documented "a standalone, directly-callable rank-1 Hebbian nudge ... a
//    manually-invokable API", NOT invented math. This is not "AssembledModel automatically running
//    STDP" -- it is a genuine, honest, host-driven reproduction of the same real weight-changing
//    machinery, clearly flagged as such rather than silently assumed to already be wired.
// 3. "Real per-edge synapse storage" (tickets #52-54) for the SAME reason needs a deliberate choice,
//    including a genuine, newly-surfaced finding from building this ticket: AssembledModel's own fixed
//    propagate kernel (`spikecorec_master_propagate`/`..._ring`, master_kernel.cpp) scatters a raw
//    `dot(U[source], V[target])` (or `constant_weight`) -- it takes no `coefficient_vectors`
//    parameter at all, so it implicitly assumes DEFAULT_MATRIX_INDEX's own coefficient vector `Ck`
//    stays the all-ones default forever (ticket #52's own "bit-compatible with pre-shared-basis
//    dot(U,V)" invariant). `WeightMatrix::refit()` (ticket #54) legitimately re-fits EVERY registered
//    matrix's `Ck`, INCLUDING DEFAULT_MATRIX_INDEX's (weight_matrix.h's own documented behavior) --
//    this ticket's own first attempt at this combined test seeded real per-edge weights via
//    `accumulate_edge_delta` (Sk) then called `refit()`, which converged `WeightMatrix::get()` to
//    within 0.14% relative RMS error of the intended values, yet the live cascade below stopped
//    propagating past neuron 1 -- because `refit()` moved DEFAULT_MATRIX_INDEX's `Ck` away from
//    all-ones, so `get()`'s own Ck-aware reconstruction and the propagate kernel's Ck-unaware raw
//    dot(U,V) had silently diverged. **This is a real, previously-undocumented gap in how ticket #54
//    (refit) interacts with ticket #6/#64's own fixed propagate kernel, surfaced by this ticket's own
//    implementation -- flagged here for task_master/the user, not silently worked around, and not
//    something this ticket's own scope owns fixing.** The combined test below instead seeds each
//    edge's U/V rows DIRECTLY (bypassing accumulate_edge_delta/refit entirely for the LIVE-kernel-
//    facing weight -- see the test's own inline comment for the exact construction), which keeps
//    `Ck` untouched and so keeps `get()` and the live propagate kernel computing the IDENTICAL value;
//    `WeightMatrix::update()` (the STDP path) also only ever touches raw U[source]/V[target], never
//    `Ck`, so it stays consistent with this too. tests/weight_matrix_tests.cpp already exhaustively
//    covers accumulate_edge_delta/refit correctness in isolation (e.g.
//    `refit_recovers_a_known_low_rank_fixture_within_tolerance`) -- this file does not re-prove that.
//    Every OTHER network test in this file (the four deterministic-timing tests below) instead uses
//    `WeightMatrix::set_constant_weight()` (the SAME placeholder
//    `run_delayed_coupling_network`/`run_glif_ei_network` already establish as precedent) for a
//    bit-exact scattered value, since their own job is to verify ACCUMULATE/THRESHOLD/RESET/PROPAGATE
//    TIMING exactly -- not per-edge storage fidelity. Only the combined network below exercises real,
//    distinct per-edge storage, a deliberate reading of this ticket's own acceptance criterion ("AT
//    LEAST one network combines...") rather than an oversight.
// 4. `WeightMatrix::update()` dispatches and SYNCHRONIZES a real GPU kernel per call
//    (weight_matrix.cpp's own `gpu_weight_update` + `synchronize_gpu_work()`) -- calling it once per
//    (firing source, downstream target) pair, every tick, at network scale, would mean tens of
//    thousands to millions of blocking round trips. The combined test below deliberately drives its
//    designated input neuron with a SPARSE discrete spike array (occasional forced fires, not a dense
//    one) specifically to bound the total `WeightMatrix::update()` call count to a tractable number
//    (reported below) -- this bounds STDP-triggering ACTIVITY, not the network's own SIZE or TICK
//    COUNT (both stay at this ticket's own anchor-point (b) values; the network genuinely runs the
//    full tick count either way, it is simply quiescent between forced input events).
// 5. Programmatic NML/LEMS generation (`build_network_content_xml` below) is this file's own,
//    self-contained helper -- no shared helper exists with ticket #100 as of this ticket's own start
//    (checked: `SC-100_EndToEndGlifNetworkTests` had zero commits beyond `nightly` at that point).
//    task_master reconciles any duplicate/divergent approach at merge time, per this session's own
//    coordination note.

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

// ── the hand-authored ComponentTypes (this ticket's own deliverable) ────────────────────────────────

const String SIMPLE_ACCUMULATOR_CELL_COMPONENT_TYPE =
    "  <ComponentType name=\"SimpleAccumulatorCell\" extends=\"baseCell\">"
    "    <Parameter name=\"spikeThreshold\" dimension=\"voltage\"/>"
    "    <Parameter name=\"restingPotential\" dimension=\"voltage\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <TimeDerivative variable=\"v\" value=\"network_inputs / dt\"/>"
    "      <OnCondition test=\"v .gt. spikeThreshold\">"
    "        <EventOut port=\"spike\"/>"
    "        <StateAssignment variable=\"v\" value=\"restingPotential\"/>"
    "      </OnCondition>"
    "    </Dynamics>"
    "  </ComponentType>";

// A structural stand-in for a real STDP synapse (exactly mirroring plasticity_wiring_tests.cpp's own
// established `TestStdpSynapse` fixture/rationale -- the vendored `stdpSynapse` is explicitly marked
// "NOT YET WORKING" and declares no tauPlus/tauMinus/aPlus/aMinus at all). Its own Dynamics are never
// dispatched by AssembledModel (see this file's own header comment #1) -- only its baked parameter
// VALUES are read, via `find_stdp_spec`.
const String SIMPLE_STDP_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"SimpleStdpSynapse\" extends=\"baseConductanceBasedSynapse\">"
    "    <Property name=\"weight\" dimension=\"none\" defaultValue=\"1\"/>"
    "    <Parameter name=\"tau\" dimension=\"time\"/>"
    "    <Parameter name=\"tauPlus\" dimension=\"time\"/>"
    "    <Parameter name=\"tauMinus\" dimension=\"time\"/>"
    "    <Parameter name=\"aPlus\" dimension=\"none\"/>"
    "    <Parameter name=\"aMinus\" dimension=\"none\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"g\" dimension=\"conductance\" exposure=\"g\"/>"
    "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g * (erev - v)\"/>"
    "      <TimeDerivative variable=\"g\" value=\"-g / tau\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"g\" value=\"0\"/>"
    "      </OnStart>"
    "      <OnEvent port=\"in\">"
    "        <StateAssignment variable=\"g\" value=\"g + weight\"/>"
    "      </OnEvent>"
    "    </Dynamics>"
    "  </ComponentType>";

String format_quantity(f64 value, const String &unit_suffix) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return String(buffer) + unit_suffix;
}

struct GeneratedConnection {
    s32 source_neuron_index = -1;
    s32 target_neuron_index = -1;
    f64 weight = 1.0;
    f64 delay_seconds = 0.0;
};

// A plain feed-forward chain 0 -> 1 -> ... -> (population_size - 1) (no cycles, no fan-out > 1) --
// the deliberately simple, exactly-hand-traceable topology every deterministic-timing test below
// (isolation aside) is built from. `uniform_weight` is written into the generated NML/ConnectionEntry
// for realism, but the four `set_constant_weight`-based tests below override the LIVE scattered value
// anyway (see this file's own header comment #3).
spikecorec::Vector<GeneratedConnection> build_chain_connections(s32 population_size, f64 uniform_weight, f64 uniform_delay_seconds) {
    spikecorec::Vector<GeneratedConnection> connections;
    connections.reserve((usize)std::max(0, population_size - 1));
    for (s32 index = 0; index < population_size - 1; ++index) {
        connections.push_back(GeneratedConnection{index, index + 1, uniform_weight, uniform_delay_seconds});
    }
    return connections;
}

// Builds one self-contained NeuroML/LEMS content document: both ComponentTypes above, one bound
// SimpleAccumulatorCell instance, one bound SimpleStdpSynapse instance, and (if `connections` is
// non-empty) one population + one projection wiring them together -- the programmatic
// arbitrary-population-size XML generator this ticket's own body calls for (no existing helper did
// this before this ticket; see this file's own header comment #5).
String build_network_content_xml(const String &fixture_id, s32 population_size, f64 spike_threshold_volts,
                                  f64 resting_potential_volts, const spikecorec::Vector<GeneratedConnection> &connections,
                                  f64 a_minus_stdp_amplitude) {
    std::ostringstream xml;
    xml << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SimpleAccumulator" << fixture_id
        << "Content\">";
    xml << SIMPLE_ACCUMULATOR_CELL_COMPONENT_TYPE;
    xml << SIMPLE_STDP_SYNAPSE_COMPONENT_TYPE;
    xml << "<SimpleAccumulatorCell id=\"cellInstance\" spikeThreshold=\""
        << format_quantity(spike_threshold_volts, "V") << "\" restingPotential=\""
        << format_quantity(resting_potential_volts, "V") << "\"/>";
    xml << "<SimpleStdpSynapse id=\"synapseInstance\" gbase=\"1nS\" erev=\"0mV\" tau=\"5ms\" tauPlus=\"20ms\" "
           "tauMinus=\"20ms\" aPlus=\"0.01\" aMinus=\""
        << format_quantity(a_minus_stdp_amplitude, "") << "\"/>";
    xml << "<network id=\"Net\">";
    xml << "<population id=\"Pop\" component=\"cellInstance\" size=\"" << population_size << "\"/>";
    if (!connections.empty()) {
        xml << "<projection id=\"Proj\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" "
               "synapse=\"synapseInstance\">";
        for (usize edge_index = 0; edge_index < connections.size(); ++edge_index) {
            const GeneratedConnection &connection = connections[edge_index];
            xml << "<connectionWD id=\"" << edge_index << "\" preCellId=\"Pop/" << connection.source_neuron_index
                << "/cellInstance\" postCellId=\"Pop/" << connection.target_neuron_index << "/cellInstance\" weight=\""
                << format_quantity(connection.weight, "") << "\" delay=\""
                << format_quantity(connection.delay_seconds, "s") << "\"/>";
        }
        xml << "</projection>";
    }
    xml << "</network>";
    xml << "</neuroml>";
    return xml.str();
}

ModelSpecification load_generated_network_model(const String &fixture_id, const String &content_xml) {
    write_temp_file("spikecorec_simple_lif_stdp_" + fixture_id + "_content.nml", content_xml);
    String top_path = write_temp_file(
        "spikecorec_simple_lif_stdp_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SimpleAccumulator" + fixture_id + "Top\">"
        "  <include href=\"spikecorec_simple_lif_stdp_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// Same convention as exit_model_validation_tests.cpp's own build_type_library_ir_programs.
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

// Given a set of "head" fire ticks (the chain's own neuron 0), the exact expected fire tick of a
// downstream neuron at `neuron_index` hops away, given a UNIFORM per-hop `delay_ticks`: every hop
// scatters a weight comfortably above threshold (see this file's own header comment #3), so an
// arrival ALWAYS triggers an immediate same-tick fire -- neuron j's own fire tick is therefore exactly
// `head_tick + j * delay_ticks`, for every head fire tick that still lands within `tick_count`.
spikecorec::Vector<s64> expected_downstream_fire_ticks(const spikecorec::Vector<s64> &head_fire_ticks, s32 neuron_index, s64 delay_ticks,
                                            s64 tick_count) {
    spikecorec::Vector<s64> expected;
    for (s64 head_tick : head_fire_ticks) {
        s64 arrival_tick = head_tick + (s64)neuron_index * delay_ticks;
        if (arrival_tick < tick_count) expected.push_back(arrival_tick);
    }
    return expected;
}

// The exact fire-tick pattern of a SimpleAccumulatorCell driven by a constant `injected_current_per_tick`
// every tick, given `spike_threshold`/`resting_potential` (both from resting_potential=0 -- the
// accumulation is exact modular arithmetic on the tick index): fires the first tick the running sum
// exceeds threshold, then resets and repeats with the same period.
spikecorec::Vector<s64> expected_continuous_injection_fire_ticks(f64 injected_current_per_tick, f64 spike_threshold,
                                                       s64 tick_count) {
    spikecorec::Vector<s64> expected;
    f64 running_sum = 0.0;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        running_sum += injected_current_per_tick;
        if (running_sum > spike_threshold) {
            expected.push_back(tick);
            running_sum = 0.0;
        }
    }
    return expected;
}

// ── shared network-run harness for the four deterministic-timing tests (bit-exact constant weight,
// this file's own header comment #3) ────────────────────────────────────────────────────────────────

struct ChainNetworkRunResult {
    UnorderedMap<s32, spikecorec::Vector<s64>> spike_ticks_by_watched_neuron;
    s64 total_spike_count = 0;
    bool all_values_finite = true;
    double wall_clock_seconds = 0.0;
};

// Builds and runs a feed-forward chain network end to end through the real master-kernel path
// (AssembledModel, ticket #6, with the delay ring enabled -- ticket #64) for `tick_count` ticks.
// `drive_tick` is invoked once per tick, BEFORE step_tick, so the caller can inject that tick's own
// stimulus (continuous current into the ring's current slot, or a forced `emit_port_flags` write for
// the discrete spike-array mechanism -- see this file's own header comment). Only
// `watched_neuron_indices`' own spike-tick histories are recorded (network scale can make recording
// EVERY neuron's full history needlessly heavy); every neuron's `v` is still checked for
// finiteness, and `total_spike_count` tallies across the WHOLE population, every tick.
ChainNetworkRunResult run_chain_network(
    const String &fixture_id, s32 population_size, s64 tick_count, s64 delay_ticks, f32 constant_weight,
    f64 spike_threshold_volts, f64 resting_potential_volts, f32 dt_seconds, const spikecorec::Vector<s32> &watched_neuron_indices,
    const std::function<void(ModelRuntimeBuffers &, DelayRingAllocation &, s64)> &drive_tick) {
    spikecorec::Vector<GeneratedConnection> connections =
        build_chain_connections(population_size, (f64)constant_weight, (f64)delay_ticks * (f64)dt_seconds);
    ModelSpecification model = load_generated_network_model(
        fixture_id, build_network_content_xml(fixture_id, population_size, spike_threshold_volts,
                                               resting_potential_volts, connections, /*a_minus_stdp_amplitude=*/0.0));

    spikecorec::Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ModelAllocation allocation = allocate_model(model, programs);

    vector<vector<s32>> adjacency((usize)population_size);
    for (const auto &connection : connections) {
        adjacency[(usize)connection.source_neuron_index].push_back(connection.target_neuron_index);
    }
    WeightMatrix weights(adjacency, /*rank=*/1);
    weights.set_constant_weight(constant_weight);

    DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);
    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);

    GpuPointer<s64> last_spiked = allocate<s64>((usize)population_size * sizeof(s64));
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + population_size, (s64)-1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)population_size * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)population_size * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.last_spiked = last_spiked.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();
    buffers.delay_ring = &ring;

    ChainNetworkRunResult result;
    for (s32 neuron_index : watched_neuron_indices) result.spike_ticks_by_watched_neuron[neuron_index] = {};

    auto start_time = std::chrono::steady_clock::now();
    for (s64 tick = 0; tick < tick_count; ++tick) {
        drive_tick(buffers, ring, tick);
        assembled_model.step_tick(buffers, dt_seconds, tick, tick + 1);

        for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
            if (!std::isfinite(allocation.cell_state.get_contents()[neuron_index])) result.all_values_finite = false;
            if (buffers.last_spiked[neuron_index] == tick) {
                ++result.total_spike_count;
                auto watched = result.spike_ticks_by_watched_neuron.find(neuron_index);
                if (watched != result.spike_ticks_by_watched_neuron.end()) watched->second.push_back(tick);
            }
        }
    }
    auto end_time = std::chrono::steady_clock::now();
    result.wall_clock_seconds = std::chrono::duration<double>(end_time - start_time).count();
    return result;
}

// Reproduces kernels.metal's own `step_apply_hebbian_update` call-site formula exactly (see this
// file's own header comment #2 for why this is invoked manually here, and exactly which existing,
// already-tested API it reuses -- WeightMatrix::update(), not new math): for every connection whose
// SOURCE just fired THIS tick, if the TARGET has a genuine, strictly-earlier last_spiked tick (this
// codebase's own "-1 = never fired" sentinel for the master-kernel path, the adaptation of the
// original kernel's own "0 = never fired" sentinel -- see kernels.metal), apply the same
// `decay_delta = -learning_rate * pow(tick_delta, -3)` depression the real kernel computes, via a real
// WeightMatrix::update() call with the SAME baked-in 0.5f/1.0f internal constants. Returns how many
// update() calls were actually made this tick (for reporting the total call count, this file's own
// header comment #4).
s64 apply_manual_stdp_after_tick(WeightMatrix &weights, const spikecorec::Vector<GeneratedConnection> &connections,
                                  const s64 *last_spiked, s64 tick, f32 mapped_learning_rate) {
    if (mapped_learning_rate == 0.0f) return 0;
    s64 update_calls_made = 0;
    for (const auto &connection : connections) {
        if (last_spiked[connection.source_neuron_index] != tick) continue;
        s64 target_last_spiked = last_spiked[connection.target_neuron_index];
        if (target_last_spiked == -1 || target_last_spiked == tick) continue;

        f32 tick_delta = (f32)std::abs(tick - target_last_spiked);
        f32 decay_delta = -mapped_learning_rate * std::pow(tick_delta, -3.0f);
        weights.update(connection.source_neuron_index, connection.target_neuron_index, decay_delta,
                       /*learning_rate=*/0.5f, /*l2_regularization=*/1.0f, /*iterations=*/1);
        ++update_calls_made;
    }
    return update_calls_made;
}

} // namespace

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// ── Deliverable #1: the cell's own accumulate/threshold/reset/emit cycle, verified in isolation FIRST
// ══════════════════════════════════════════════════════════════════════════════════════════════════

TEST(SimpleAccumulatorCellIsolation, ir_lowering_shows_the_dt_cancelling_network_inputs_over_dt_expression) {
    ModelSpecification model = load_generated_network_model(
        "ir_check", build_network_content_xml("IrCheck", /*population_size=*/1, /*spike_threshold_volts=*/0.5,
                                               /*resting_potential_volts=*/0.0, /*connections=*/{},
                                               /*a_minus_stdp_amplitude=*/0.0));
    const TypeLibraryEntry &cell_entry = type_library_entry_for(model, "cellInstance");
    IrProgram program = lower_cell_to_ir(cell_entry);
    String printed = print_ir_program(program);

    // The forward-Euler fallback (see this file's own header comment): `t0 = network_inputs/dt; t0 *=
    // dt; v += t0` -- the middle step exactly cancels the division this ComponentType's own
    // TimeDerivative wrote.
    EXPECT_NE(printed.find("div t0, network_inputs, dt"), String::npos) << printed;
    EXPECT_NE(printed.find("mul t0, t0, dt"), String::npos) << printed;
    EXPECT_NE(printed.find("add v, v, t0"), String::npos) << printed;

    // `network_inputs / dt` does not reference `v` at all, so it is (trivially) affine in `v` --
    // cell_lowering.cpp's own classifier (ticket #62) correctly tags this closed_form_advanceable.
    EXPECT_TRUE(program.closed_form_advanceable);
}

TEST(SimpleAccumulatorCellIsolation, accumulate_threshold_reset_cycle_matches_hand_computed_trajectory) {
    const f32 dt_seconds = 1e-4f;
    const f64 spike_threshold_volts = 0.5;
    const f64 resting_potential_volts = 0.0;
    const f32 injected_current_per_tick = 0.2f;
    const s64 tick_count = 20;

    ModelSpecification model = load_generated_network_model(
        "isolation", build_network_content_xml("Isolation", /*population_size=*/1, spike_threshold_volts,
                                                resting_potential_volts, /*connections=*/{},
                                                /*a_minus_stdp_amplitude=*/0.0));
    ASSERT_EQ(model.total_neuron_count, 1);

    spikecorec::Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ModelAllocation allocation = allocate_model(model, programs);
    AssembledModel assembled_model(model, programs);

    vector<vector<s32>> adjacency(1); // one edge-free neuron -- same convention
                                       // exit_model_validation_tests.cpp's own build_weight_matrix uses
                                       // for a single-cell model.
    WeightMatrix weights(adjacency, /*rank=*/1);

    GpuPointer<f32> network_inputs = allocate<f32>(sizeof(f32));
    network_inputs.get_contents()[0] = 0.0f;
    GpuPointer<s64> last_spiked = allocate<s64>(sizeof(s64));
    last_spiked.get_contents()[0] = -1;
    GpuPointer<s32> next_active_indices = allocate<s32>(sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>(sizeof(s32));
    active_generation.get_contents()[0] = -1;
    GpuPointer<bool> emit_spike = allocate<bool>(sizeof(bool));
    emit_spike.get_contents()[0] = false;

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();

    spikecorec::Vector<s64> observed_spike_ticks;
    f64 hand_computed_v = 0.0;

    for (s64 tick = 0; tick < tick_count; ++tick) {
        buffers.network_inputs[0] += injected_current_per_tick;
        assembled_model.step_tick(buffers, dt_seconds, tick, tick + 1);

        f32 observed_v = allocation.cell_state.get_contents()[0];
        if (buffers.last_spiked[0] == tick) observed_spike_ticks.push_back(tick);

        // Hand-computed reference: accumulate, then reset to resting_potential EXACTLY when the sum
        // first exceeds threshold -- the accumulate/threshold/reset/emit cycle this deliverable exists
        // to pin down, verified BEFORE any network/scale is introduced.
        hand_computed_v += injected_current_per_tick;
        if (hand_computed_v > spike_threshold_volts) hand_computed_v = resting_potential_volts;
        EXPECT_NEAR(observed_v, (f32)hand_computed_v, 1e-5f) << "tick=" << tick;
    }

    spikecorec::Vector<s64> expected_spike_ticks =
        expected_continuous_injection_fire_ticks(injected_current_per_tick, spike_threshold_volts, tick_count);
    ASSERT_EQ(expected_spike_ticks, (spikecorec::Vector<s64>{2, 5, 8, 11, 14, 17})); // sanity on the reference formula itself
    EXPECT_EQ(observed_spike_ticks, expected_spike_ticks);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// ── Deliverable #2: anchor-point (a) -- 50 neurons / 50 ticks -- both driving mechanisms ────────────
// ══════════════════════════════════════════════════════════════════════════════════════════════════

TEST(SimpleAccumulatorNetwork, anchor_a_50_neurons_50_ticks_continuous_current_injection) {
    const f32 dt_seconds = 1e-4f;
    const s32 population_size = 50;
    const s64 tick_count = 50;
    const s64 delay_ticks = 2;
    const f32 constant_weight = 0.6f; // > spikeThreshold -- a single incoming spike always fires its
                                       // target immediately on arrival.
    const f64 spike_threshold_volts = 0.5;
    const f64 resting_potential_volts = 0.0;
    const f32 injected_current_per_tick = 0.2f;

    spikecorec::Vector<s32> watched_neuron_indices;
    for (s32 index = 0; index < population_size; ++index) watched_neuron_indices.push_back(index);

    auto drive_continuous = [injected_current_per_tick](ModelRuntimeBuffers &, DelayRingAllocation &ring, s64 tick) {
        s64 current_slot = tick % ring.ring_slot_count;
        ring.input_ring.get_contents()[current_slot * ring.neuron_count + 0] += injected_current_per_tick;
    };

    ChainNetworkRunResult result =
        run_chain_network("anchor_a_continuous", population_size, tick_count, delay_ticks, constant_weight,
                           spike_threshold_volts, resting_potential_volts, dt_seconds, watched_neuron_indices,
                           drive_continuous);

    EXPECT_TRUE(result.all_values_finite);

    spikecorec::Vector<s64> expected_head_ticks =
        expected_continuous_injection_fire_ticks(injected_current_per_tick, spike_threshold_volts, tick_count);
    ASSERT_EQ(result.spike_ticks_by_watched_neuron[0], expected_head_ticks);

    for (s32 neuron_index = 1; neuron_index < population_size; ++neuron_index) {
        spikecorec::Vector<s64> expected_ticks =
            expected_downstream_fire_ticks(expected_head_ticks, neuron_index, delay_ticks, tick_count);
        EXPECT_EQ(result.spike_ticks_by_watched_neuron[neuron_index], expected_ticks) << "neuron_index=" << neuron_index;
    }
}

TEST(SimpleAccumulatorNetwork, anchor_a_50_neurons_50_ticks_discrete_spike_array_drives_input_neuron) {
    const f32 dt_seconds = 1e-4f;
    const s32 population_size = 50;
    const s64 tick_count = 50;
    const s64 delay_ticks = 3;
    const f32 constant_weight = 0.6f;
    const f64 spike_threshold_volts = 0.5;
    const f64 resting_potential_volts = 0.0;

    // A literal, host-provided array of 0s/1s, one entry per tick (this file's own header comment on
    // the discrete-spike-array driving mechanism) -- two closely-spaced forced ticks and one isolated
    // one, to also show the mechanism handles back-to-back forced ticks correctly.
    spikecorec::Vector<s32> input_spike_array(tick_count, 0);
    input_spike_array[5] = 1;
    input_spike_array[6] = 1;
    input_spike_array[30] = 1;

    spikecorec::Vector<s32> watched_neuron_indices;
    for (s32 index = 0; index < population_size; ++index) watched_neuron_indices.push_back(index);

    auto drive_discrete = [&input_spike_array](ModelRuntimeBuffers &buffers, DelayRingAllocation &, s64 tick) {
        if (input_spike_array[tick] == 1) buffers.emit_port_flags.at("spike")[0] = true;
    };

    ChainNetworkRunResult result =
        run_chain_network("anchor_a_discrete", population_size, tick_count, delay_ticks, constant_weight,
                           spike_threshold_volts, resting_potential_volts, dt_seconds, watched_neuron_indices,
                           drive_discrete);

    EXPECT_TRUE(result.all_values_finite);

    spikecorec::Vector<s64> expected_head_ticks = {5, 6, 30};
    ASSERT_EQ(result.spike_ticks_by_watched_neuron[0], expected_head_ticks);

    for (s32 neuron_index = 1; neuron_index < population_size; ++neuron_index) {
        spikecorec::Vector<s64> expected_ticks =
            expected_downstream_fire_ticks(expected_head_ticks, neuron_index, delay_ticks, tick_count);
        EXPECT_EQ(result.spike_ticks_by_watched_neuron[neuron_index], expected_ticks) << "neuron_index=" << neuron_index;
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// ── Deliverable #3: anchor-point (b) -- ~300-500 neurons / 1000-2000 ticks ──────────────────────────
// ══════════════════════════════════════════════════════════════════════════════════════════════════

TEST(SimpleAccumulatorNetwork, anchor_b_400_neurons_1500_ticks_continuous_current_injection_scaling_sanity) {
    const f32 dt_seconds = 1e-4f;
    const s32 population_size = 400;
    const s64 tick_count = 1500;
    const s64 delay_ticks = 2;
    const f32 constant_weight = 0.6f;
    const f64 spike_threshold_volts = 0.5;
    const f64 resting_potential_volts = 0.0;
    const f32 injected_current_per_tick = 0.2f;

    spikecorec::Vector<s32> watched_neuron_indices = {0, 1, 5, 20, 100, 200, 399};

    auto drive_continuous = [injected_current_per_tick](ModelRuntimeBuffers &, DelayRingAllocation &ring, s64 tick) {
        s64 current_slot = tick % ring.ring_slot_count;
        ring.input_ring.get_contents()[current_slot * ring.neuron_count + 0] += injected_current_per_tick;
    };

    ChainNetworkRunResult result =
        run_chain_network("anchor_b_continuous", population_size, tick_count, delay_ticks, constant_weight,
                           spike_threshold_volts, resting_potential_volts, dt_seconds, watched_neuron_indices,
                           drive_continuous);

    std::cout << "[SimpleAccumulatorNetwork anchor_b_continuous] " << population_size << " neurons / " << tick_count
              << " ticks, " << result.total_spike_count << " total spikes, wall-clock: "
              << result.wall_clock_seconds << "s\n";

    EXPECT_TRUE(result.all_values_finite);
    EXPECT_GT(result.total_spike_count, 0);

    spikecorec::Vector<s64> expected_head_ticks =
        expected_continuous_injection_fire_ticks(injected_current_per_tick, spike_threshold_volts, tick_count);
    ASSERT_EQ(result.spike_ticks_by_watched_neuron[0], expected_head_ticks);
    for (s32 neuron_index : watched_neuron_indices) {
        if (neuron_index == 0) continue;
        spikecorec::Vector<s64> expected_ticks =
            expected_downstream_fire_ticks(expected_head_ticks, neuron_index, delay_ticks, tick_count);
        EXPECT_EQ(result.spike_ticks_by_watched_neuron[neuron_index], expected_ticks) << "neuron_index=" << neuron_index;
    }
}

TEST(SimpleAccumulatorNetwork,
     anchor_b_combines_real_per_edge_synapse_storage_spike_delays_and_active_stdp_together) {
    const f32 dt_seconds = 1e-4f;
    const s32 chain_length = 350;
    const s32 population_size = chain_length + 2; // + one isolated, never-firing control pair
    const s64 tick_count = 1500;
    const f64 spike_threshold_volts = 0.5;
    const f64 resting_potential_volts = 0.0;
    const f64 a_minus_stdp_amplitude = 0.05;

    // Real, DISTINCT per-edge weights (0.8-1.2V, well above threshold with margin) and non-trivial,
    // VARYING per-edge delays (1-3 ticks) -- this file's own header comment #3.
    spikecorec::Vector<GeneratedConnection> connections;
    spikecorec::Vector<f64> intended_weight;
    connections.reserve((usize)chain_length);
    intended_weight.reserve((usize)chain_length);
    for (s32 index = 0; index < chain_length - 1; ++index) {
        f64 weight = 0.8 + 0.1 * (f64)(index % 5);
        f64 delay_seconds = (f64)(1 + (index % 3)) * (f64)dt_seconds;
        connections.push_back(GeneratedConnection{index, index + 1, weight, delay_seconds});
        intended_weight.push_back(weight);
    }
    const s32 control_source = chain_length;
    const s32 control_target = chain_length + 1;
    const f64 control_weight = 0.95;
    connections.push_back(GeneratedConnection{control_source, control_target, control_weight, 2.0 * (f64)dt_seconds});
    intended_weight.push_back(control_weight);

    ModelSpecification model = load_generated_network_model(
        "anchor_b_combined",
        build_network_content_xml("AnchorBCombined", population_size, spike_threshold_volts, resting_potential_volts,
                                   connections, a_minus_stdp_amplitude));
    ASSERT_EQ(model.total_neuron_count, population_size);

    const TypeLibraryEntry &synapse_entry = type_library_entry_for(model, "synapseInstance");
    std::optional<StdpSpec> spec = find_stdp_spec(synapse_entry);
    ASSERT_TRUE(spec.has_value());
    f32 mapped_learning_rate = map_stdp_spec_to_learning_rate(*spec);
    EXPECT_NEAR(mapped_learning_rate, (f32)a_minus_stdp_amplitude, 1e-6f);

    spikecorec::Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ModelAllocation allocation = allocate_model(model, programs);

    vector<vector<s32>> adjacency((usize)population_size);
    for (const auto &connection : connections) {
        adjacency[(usize)connection.source_neuron_index].push_back(connection.target_neuron_index);
    }
    WeightMatrix weights(adjacency, /*rank=*/-1);

    // ── real per-edge synapse storage (tickets #52-54; this file's own header comment #3) ──────────
    // Seeds each edge's own U/V rows DIRECTLY (a single float4 lane per row: U[source]=(sqrt(w),0,0,0),
    // V[target]=(sqrt(w),0,0,0), so dot(U[source],V[target]) = w exactly) rather than via
    // `accumulate_edge_delta`(Sk)+`refit()`. This is a real, deliberate finding from this ticket's own
    // implementation, not the originally-planned approach: `refit()` legitimately re-fits EVERY
      // registered matrix's coefficient vector Ck, INCLUDING DEFAULT_MATRIX_INDEX's (weight_matrix.h's
    // own documented behavior) -- but master_kernel.cpp's fixed propagate kernel
    // (`spikecorec_master_propagate`/`..._ring`) was never updated to read `coefficient_vectors` at
    // all; it hardcodes a raw `dot(U[source], V[target])`, implicitly assuming DEFAULT_MATRIX_INDEX's
    // Ck stays the all-ones default forever. Calling `refit()` on a model ALSO driven through
    // AssembledModel therefore silently desyncs the LIVE scattered weight from what
    // `WeightMatrix::get()` reports (verified empirically while building this test: refit converged
    // `get()` to within a 0.14% relative RMS error of the intended values, yet the live cascade
    // stopped propagating past neuron 1 -- `get()`'s own Ck-aware reconstruction and the propagate
    // kernel's Ck-unaware one had diverged). Directly setting U/V rows (each edge exclusive to its own
    // node pair in this chain topology, so no cross-edge interference) keeps DEFAULT_MATRIX_INDEX's Ck
    // untouched, making `get()` and the live propagate kernel compute the IDENTICAL raw dot(U,V) --
    // and `WeightMatrix::update()` (the STDP path below) also only ever touches U[source]/V[target]
    // directly, never Ck, so it stays consistent with this too. tests/weight_matrix_tests.cpp already
    // exhaustively covers accumulate_edge_delta/refit correctness in isolation (e.g.
    // `refit_recovers_a_known_low_rank_fixture_within_tolerance`); this file does not re-prove that,
    // and reports this refit/master-kernel interaction as a genuine, newly-surfaced gap for
    // task_master/the user to weigh -- not something this ticket's own scope owns fixing.
    for (usize edge_index = 0; edge_index < connections.size(); ++edge_index) {
        const GeneratedConnection &connection = connections[edge_index];
        f32 root = std::sqrt((f32)intended_weight[edge_index]);
        spikecorec::float4 *u_row = weights.U_matrix.get_contents() + (s64)connection.source_neuron_index * weights.rank_float4_stride;
        spikecorec::float4 *v_row = weights.V_matrix.get_contents() + (s64)connection.target_neuron_index * weights.rank_float4_stride;
        u_row[0] = spikecorec::float4{root, 0.0f, 0.0f, 0.0f};
        v_row[0] = spikecorec::float4{root, 0.0f, 0.0f, 0.0f};
        for (s64 lane = 1; lane < weights.rank_float4_stride; ++lane) {
            u_row[lane] = spikecorec::float4{0.0f, 0.0f, 0.0f, 0.0f};
            v_row[lane] = spikecorec::float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    spikecorec::Vector<f32> seeded_edge_weight(connections.size());
    for (usize edge_index = 0; edge_index < connections.size(); ++edge_index) {
        const GeneratedConnection &connection = connections[edge_index];
        f32 reconstructed = weights.get(connection.source_neuron_index, connection.target_neuron_index);
        seeded_edge_weight[edge_index] = reconstructed;
        EXPECT_NEAR(reconstructed, (f32)intended_weight[edge_index], 1e-3f) << "edge_index=" << edge_index;
        ASSERT_GT(reconstructed, (f32)spike_threshold_volts + 0.1f)
            << "edge_index=" << edge_index
            << " seeded weight too close to spikeThreshold for this test's own cascade/causality checks "
               "below to remain valid";
    }
    std::cout << "[SimpleAccumulatorNetwork combined test] " << connections.size()
              << " edges seeded with real, distinct per-edge U/V weights (see this file's own header "
                 "comment #3 on why direct U/V seeding, not accumulate_edge_delta+refit, is what reaches "
                 "the live propagate kernel)\n";

    DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);
    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);

    GpuPointer<s64> last_spiked = allocate<s64>((usize)population_size * sizeof(s64));
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + population_size, (s64)-1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)population_size * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)population_size * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.last_spiked = last_spiked.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();
    buffers.delay_ring = &ring;

    // ── discrete spike-array driving, deliberately SPARSE (this file's own header comment #4): 30
    // forced fires of neuron 0, spaced 5 ticks apart (comfortably above the largest single-edge delay
    // of 3 ticks, so every wave retriggers STDP for every edge -- see this file's own header comment). ──
    spikecorec::Vector<s32> input_spike_array(tick_count, 0);
    const s64 head_period_ticks = 5;
    const s32 head_fire_count = 30;
    for (s32 wave_index = 0; wave_index < head_fire_count; ++wave_index) {
        input_spike_array[10 + wave_index * head_period_ticks] = 1;
    }

    s64 total_stdp_update_calls = 0;
    spikecorec::Vector<s64> chain_first_fire_tick(population_size, -1);
    bool all_finite = true;
    s64 control_pair_spike_count = 0;

    auto run_start_time = std::chrono::steady_clock::now();
    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (input_spike_array[tick] == 1) buffers.emit_port_flags.at("spike")[0] = true;

        assembled_model.step_tick(buffers, dt_seconds, tick, tick + 1);

        for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
            if (!std::isfinite(allocation.cell_state.get_contents()[neuron_index])) all_finite = false;
            if (buffers.last_spiked[neuron_index] == tick && chain_first_fire_tick[neuron_index] == -1) {
                chain_first_fire_tick[neuron_index] = tick;
            }
        }
        if (buffers.last_spiked[control_source] == tick || buffers.last_spiked[control_target] == tick) {
            ++control_pair_spike_count;
        }

        total_stdp_update_calls +=
            apply_manual_stdp_after_tick(weights, connections, buffers.last_spiked, tick, mapped_learning_rate);
    }
    auto run_end_time = std::chrono::steady_clock::now();
    double run_wall_clock_seconds = std::chrono::duration<double>(run_end_time - run_start_time).count();
    std::cout << "[SimpleAccumulatorNetwork combined test] " << population_size << " neurons / " << tick_count
              << " ticks, " << total_stdp_update_calls
              << " manual STDP WeightMatrix::update() calls, run wall-clock: " << run_wall_clock_seconds << "s\n";

    // ── internal-consistency checks ──────────────────────────────────────────────────────────────────
    EXPECT_TRUE(all_finite);
    EXPECT_GT(total_stdp_update_calls, 0);
    EXPECT_EQ(control_pair_spike_count, 0); // the isolated control pair never receives any input at all

    for (s32 neuron_index = 0; neuron_index < chain_length; ++neuron_index) {
        EXPECT_NE(chain_first_fire_tick[neuron_index], -1)
            << "neuron_index=" << neuron_index << " never fired within " << tick_count << " ticks";
    }
    // Causality/monotonicity (weight-approximation-robust, unlike an exact-tick formula): each
    // neuron's FIRST fire can only be caused by its immediate predecessor's own prior fire plus that
    // edge's own real per-edge delay (>= 1 tick), so first-fire ticks strictly increase down the chain.
    for (s32 neuron_index = 1; neuron_index < chain_length; ++neuron_index) {
        EXPECT_GT(chain_first_fire_tick[neuron_index], chain_first_fire_tick[neuron_index - 1])
            << "neuron_index=" << neuron_index;
    }

    // ── STDP genuinely changed weights, at network scale, in the kernel-defined (depression-only)
    // direction (this file's own header comment #2) ────────────────────────────────────────────────
    s32 edges_with_measurable_depression = 0;
    for (usize edge_index = 0; edge_index < (usize)(chain_length - 1); ++edge_index) {
        const GeneratedConnection &connection = connections[edge_index];
        f32 final_weight = weights.get(connection.source_neuron_index, connection.target_neuron_index);
        EXPECT_LE(final_weight, seeded_edge_weight[edge_index] + 1e-6f) << "edge_index=" << edge_index;
        if (seeded_edge_weight[edge_index] - final_weight > 1e-4f) ++edges_with_measurable_depression;
    }
    EXPECT_GT(edges_with_measurable_depression, (chain_length - 1) / 2)
        << "expected the majority of the chain's own edges to show a measurable STDP depression";

    // Specificity: the isolated control edge never fires, so it never receives a single
    // WeightMatrix::update() call -- its weight must be untouched.
    f32 control_final_weight = weights.get(control_source, control_target);
    EXPECT_NEAR(control_final_weight, seeded_edge_weight.back(), 1e-6f);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// ── Deliverable #4: anchor-point (c) -- 2000 neurons / 10000 ticks -- scaling sanity ────────────────
// ══════════════════════════════════════════════════════════════════════════════════════════════════

TEST(SimpleAccumulatorNetwork, anchor_c_2000_neurons_10000_ticks_scaling_sanity) {
    const f32 dt_seconds = 1e-4f;
    const s32 population_size = 2000;
    const s64 tick_count = 10000;
    const s64 delay_ticks = 2;
    const f32 constant_weight = 0.6f;
    const f64 spike_threshold_volts = 0.5;
    const f64 resting_potential_volts = 0.0;
    const f32 injected_current_per_tick = 0.2f;

    spikecorec::Vector<s32> watched_neuron_indices = {0, 1, 2, 10, 100, 500, 1000, 1500, 1999};

    auto drive_continuous = [injected_current_per_tick](ModelRuntimeBuffers &, DelayRingAllocation &ring, s64 tick) {
        s64 current_slot = tick % ring.ring_slot_count;
        ring.input_ring.get_contents()[current_slot * ring.neuron_count + 0] += injected_current_per_tick;
    };

    ChainNetworkRunResult result =
        run_chain_network("anchor_c_continuous", population_size, tick_count, delay_ticks, constant_weight,
                           spike_threshold_volts, resting_potential_volts, dt_seconds, watched_neuron_indices,
                           drive_continuous);

    std::cout << "[SimpleAccumulatorNetwork anchor_c_continuous] " << population_size << " neurons / " << tick_count
              << " ticks, " << result.total_spike_count << " total spikes, wall-clock: "
              << result.wall_clock_seconds << "s\n";

    EXPECT_TRUE(result.all_values_finite);
    EXPECT_GT(result.total_spike_count, 0);

    spikecorec::Vector<s64> expected_head_ticks =
        expected_continuous_injection_fire_ticks(injected_current_per_tick, spike_threshold_volts, tick_count);
    ASSERT_EQ(result.spike_ticks_by_watched_neuron[0], expected_head_ticks);
    for (s32 neuron_index : watched_neuron_indices) {
        if (neuron_index == 0) continue;
        spikecorec::Vector<s64> expected_ticks =
            expected_downstream_fire_ticks(expected_head_ticks, neuron_index, delay_ticks, tick_count);
        EXPECT_EQ(result.spike_ticks_by_watched_neuron[neuron_index], expected_ticks) << "neuron_index=" << neuron_index;
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// ── Deliverable #5: a concrete, generated NML/LEMS XML sample, captured for inspection ──────────────
// ══════════════════════════════════════════════════════════════════════════════════════════════════

TEST(SimpleAccumulatorNetwork, generated_nml_lems_sample_is_captured_for_inspection) {
    spikecorec::Vector<GeneratedConnection> connections = build_chain_connections(/*population_size=*/5, /*uniform_weight=*/0.6,
                                                                        /*uniform_delay_seconds=*/2.0 * 1e-4);
    String content_xml = build_network_content_xml("SampleForInspection", /*population_size=*/5,
                                                     /*spike_threshold_volts=*/0.5, /*resting_potential_volts=*/0.0,
                                                     connections, /*a_minus_stdp_amplitude=*/0.02);

    std::cout << "\n"
                 "==================== SimpleAccumulatorCell -- generated NML/LEMS sample "
                 "(ticket #101 [T2], acceptance criterion #4) ====================\n"
              << content_xml
              << "\n"
                 "======================================================================================"
                 "==============\n\n";

    // Also a real sanity check, not just a print: this exact generated string genuinely parses,
    // resolves, and lowers to the expected shape.
    ModelSpecification model = load_generated_network_model("captured_sample", content_xml);
    EXPECT_EQ(model.total_neuron_count, 5);
    ASSERT_EQ(model.populations.size(), 1u);
    ASSERT_EQ(model.projections.size(), 1u);
    EXPECT_EQ(model.projections[0].connections.size(), 4u);
}
