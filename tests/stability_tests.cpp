#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/nml/kernel_codegen.h"

#include "nml_network_generator.h"

using namespace std;
using namespace spikecorec;

// ── Long-run stability regression suite ──────────────────────────────────────────────────────
//
// Every other suite in this tree checks that a run of a few hundred ticks produces the numbers
// we expect it to. This one checks the property that holds over an ARBITRARILY long run: that a
// leak-controlled simulation settles and then stays settled. Cell state does not inflate tick
// over tick, the synaptic input accumulator does not accumulate, and the firing rate neither
// runs away nor dies out.
//
// ── WHY THIS EXISTS ─────────────────────────────────────────────────────────────────────────
// This branch shipped exactly that failure. `network_inputs` was never drained, so a constant
// injected current reached its target as a linear ramp -- 5e-10, 1e-9, 1.5e-9, 2e-9, ... -- and
// every steady-state voltage and every spike time in every network model was silently wrong.
// It compiled. 196 tests passed. Nothing in the tree asserted that a per-tick accumulator is
// bounded, so nothing noticed.
//
// The defect is instructive beyond its own fix, and the shape of this file follows from it:
//
//   * It produced plausible numbers, not a crash. Voltages stayed inside their physical range
//     the whole time, because a spiking cell is pinned between its reset and its threshold no
//     matter how hard it is driven. A bounded-state assertion alone would NOT have caught it.
//     The assertion that catches it is the one on the accumulator itself
//     (`synaptic_accumulator_is_drained_every_tick`), which is why that one is exact and has no
//     tolerance at all.
//   * It grew without bound, so it was visible in any statistic compared across time, and
//     invisible in any statistic taken at one instant. Every verdict below is therefore a
//     comparison of the run against ITSELF at a different time -- with the one exception noted
//     next, which is exact and needs no window at all.
//   * A defect of the same shape that SETTLES rather than growing defeats every comparison
//     across time, because a run at a fixed multiple of its intended drive is stationary at
//     every multiple. That is why the accumulator carries two exact verdicts rather than one:
//     `synaptic_accumulator_is_drained_every_tick` says the ring empties, and
//     `no_arrival_appears_in_the_ring_without_an_emission` says nothing enters it that no spike
//     put there. The first alone accepts a defect delivering a hundred times the intended
//     current on every one of 8000 ticks.
//
// ── HOW THE ASSERTIONS ARE BUILT ────────────────────────────────────────────────────────────
// The hard part is asserting something an accumulating defect cannot satisfy without being so
// loose it catches nothing or so tight it fails on a legitimate transient. Three rules:
//
//   1. Measure only settled behaviour. All four models here settle inside a few hundred ticks;
//      the earliest window any verdict opens is at a quarter of the run, which is two thousand.
//   2. Compare later against earlier, not against a constant. A drift bound survives a
//      parameter change; a magic number pins today's behaviour and has to be re-measured every
//      time a fixture moves.
//   3. Where an absolute bound is genuinely needed, derive it from the MODEL -- from `vth`,
//      `EL`, `t_ref` as the .nml declares them -- never from a measured value. A bound read off
//      today's run is not a test, it is a snapshot.
//
// Two window choices, and the difference between them is deliberate:
//
//   * The RATE and ACCUMULATOR comparisons use the SECOND and FOURTH quarters, because the
//     reason is arithmetic. A quantity growing linearly from tick zero has a window mean set by
//     the window's midpoint, so two adjacent settled halves (midpoints 5N/8 and 7N/8) differ by
//     a*N/4 while the second and fourth quarters (midpoints 3N/8 and 7N/8) differ by a*N/2 --
//     twice the signal against the same band. On the firing rate that is the difference between
//     rejecting a ramp once it has grown 1.6x across the run and letting it reach 3.7x, measured
//     (see a_linearly_ramping_firing_rate_is_rejected, which the settled-halves windows read at
//     a ratio of 1.197 and accepted).
//   * The STATE comparison keeps the two halves of the settled run, because its tightest real
//     margin belongs to glif5's asc1, which is still converging by a few parts in 10^4 at the
//     half-way point. Opening its early window at a quarter of the run would measure that
//     convergence rather than any drift, and would spend the margin on it.
//
// The statistic every state-variable verdict uses is the (min, max) PAIR over each half, not
// the mean. The mean is unusable here: a window holding three spikes and a window holding two
// have visibly different means, purely because the window boundary and the inter-spike interval
// beat against each other, and that beat is 1.4% of the variable's own magnitude on the driven
// single cell -- the same order as the drift a real defect would produce. The extremes have no
// such sensitivity: a settled periodic trajectory reaches the same floor and the same ceiling in
// every half whether that half holds two cycles or three.
//
// ── WHERE THE STATE-VARIABLE VERDICT ACTUALLY DISCRIMINATES ─────────────────────────────────
// The (min, max) statistic needs the extremes to be FREE. Where the model pins an extreme, the
// verdict measures the pin and not the trajectory, and its reported margin is not sensitivity.
// The distinction is not cosmetic: only 4 of this file's 36 state slots have both extremes free,
// so they are listed here rather than left to be rediscovered.
//
//   LOAD-BEARING (extremes free, a defect can move them) -- 4 slots:
//     driven single cell   theta, asc1, asc2 -- continuous, no reset touches them. asc1 is the
//                          tightest real margin in the file at 11.6x, and it is tight because it
//                          is still settling by a few parts in 10^4, not because it drifts. asc2
//                          has converged to within f32 rounding by the settle point and so reads
//                          bit-identical across the halves, but nothing about the model pins it:
//                          a defect can still move it, and the drift falsification confirms the
//                          verdict rejects one when it does.
//     delayed coupling     the target cell's v, over its [-67.4mV, -64.7mV] settled span. It
//                          never reaches threshold, so both ends are set by the leak against the
//                          arriving current. The only membrane potential here of which that is
//                          true.
//
//   PINNED BY CONSTRUCTION (no engine defect can move them) -- the other 32 slots:
//     every spiking cell's v         10 slots. The window minimum is bit-identically `vreset`,
//                                    because the reset assignment writes exactly that value --
//                                    measured, it does not move by one bit in either half of any
//                                    model here. The maximum is whichever sample first landed
//                                    past `vth`, so it moves only by where in the tick grid the
//                                    crossing fell. That is sub-tick crossing jitter, and the
//                                    recurrent network's worst reported margin of 20x is 20x the
//                                    jitter -- not a safety factor, and it does not become one by
//                                    being large.
//     refractoryTimeElapsed          11 slots, cycling [0, t_ref] by the regime machinery's own
//                                    definition, or frozen at 0 for a cell that never fires.
//     the regime index               11 slots, cycling {0, 1} for the same reason.
//
// The recurrent network's state test is kept anyway, because the pinning is itself worth
// asserting -- a `vreset` that moved is a real change -- but it is NOT counted as coverage of
// state drift, and its name says what it checks.
//
// One further caution on every margin quoted in this file: they are drawn from a chaotic
// quantity and are not monotone in anything. Scaling the recurrent fixture's drive moves the
// worst state-stationarity ratio to 0.976 of its allowance at 4x, then back to 0.257 at 5x and
// 0.310 at 6x -- on a correct engine, every time. Headroom here is the current draw of a chaotic
// system, not a safety factor, and a fixture amplitude change can push a ratio past 1.0 with
// nothing wrong. Re-measure after any fixture edit; do not reason from these numbers.
//
// ── WHAT THIS SUITE DOES NOT COVER ──────────────────────────────────────────────────────────
// Stated because a suite whose claimed coverage exceeds its real coverage is worse than no suite:
// it converts absence of evidence into apparent evidence of absence, which is the exact failure
// mode the section above describes.
//
//   * A CONSTANT-FACTOR ERROR IN DELIVERED CURRENT. Not covered, and not coverable here.
//     Scaling every pulseGenerator amplitude on the recurrent model gives 696 spikes at 1.0x,
//     773 at 1.1x, 1253 at 2.0x and 1795 at 4.0x, and every verdict in this file passes at every
//     one of them. That is correct behaviour on this suite's part: a network at 1.1x its
//     intended drive is genuinely stable -- it settles, it stays settled, nothing accumulates --
//     so there is no stability property left to violate. Delivering the wrong AMOUNT is a
//     correctness property with no stability signature.
//     The 1.1x row is not hypothetical: the pulseGenerator defect fixed alongside this file's own
//     first version delivered 10% too much charge. This suite would not have caught it. What did
//     catch it is reference validation -- tests/exit_model_validation_tests.cpp, comparing whole
//     trajectories against recorded jLEMS output, where it read as a 600uV deviation. Anything
//     about the SIZE of a delivered quantity belongs there, not here.
//     The one part of the level this file can reach is the saturated end, where the physics
//     itself provides a bound: see evaluate_firing_rate_stays_below_the_refractory_ceiling.
//
//   * THE SIZE OF A SYNAPTIC ARRIVAL. Same reason. The accumulator verdicts establish that
//     charge appears in network_inputs only where a spike put it and leaves on schedule; they
//     say nothing about how much of it there is. An edge weight scaled wrongly is invisible here.
//
//   * STATE DRIFT ON A SPIKING CELL'S MEMBRANE POTENTIAL. Pinned by reset and threshold, as
//     above. Runaway on such a cell appears in the RATE, which is where it is checked.
//
// ── FALSIFICATION ───────────────────────────────────────────────────────────────────────────
// A stability suite that passes on a broken engine is worse than none. Every verdict function
// below has at least one test in the StabilityFalsification suite that feeds it a run the
// verdict MUST reject, and the falsifications are engine-level wherever the defect can be
// produced by an engine-level mutation:
//
//   * `undrained_synaptic_accumulator_is_rejected` re-injects the running total of everything
//     delivered so far into the ring row the engine just cleared, after every tick -- the
//     shipped defect, reconstructed, driving a real simulation.
//   * `input_carried_forward_into_the_next_ring_row_is_rejected` carries a fraction of each
//     tick's input into the NEXT row instead of dropping it, so every row still drains to
//     exactly zero while the network runs at a hundred times its intended current. It is the
//     accumulator defect the drained-row check cannot see, and it is why that check has a
//     whole-ring counterpart next to it.
//   * `an_arrival_scheduled_deep_in_the_delay_ring_is_rejected` puts an arrival no spike asked
//     for 50 rows ahead in the 101-row delay ring -- the furthest a defect can get from the one
//     row the drained-row check inspects on any given tick, and still drained on schedule.
//   * `arithmetic_overflow_is_rejected` drives a real cell with a current large enough to
//     overflow f32, so the non-finite samples the finiteness verdict rejects are ones the
//     engine really produced.
//   * `silenced_network_is_rejected` runs a real model whose drive stops between the two
//     measurement windows, against both the rate comparison's falling direction and the growth
//     comparison's shrinking one.
//   * `an_unclosed_output_block_is_refused_rather_than_left_in_place` covers a fixture-
//     preparation step rather than a verdict, and is here because it used to fail silently and
//     the consequence was a recording written into the shared working directory of this binary.
//   * the raster falsifications perturb a REAL recorded raster -- thinned into a linear rate
//     ramp, or replaced by the maximal train its own t_ref permits -- so what a verdict rejects
//     is a shape it would meet in practice, and so the perturbation cannot violate a DIFFERENT
//     verdict and be rejected for the wrong reason. Each one asserts the verdicts it must not
//     trip, which is what attributes the rejection to the verdict under test.
//   * the remaining two perturb a real recorded trajectory (a linear ramp on one state variable;
//     a one-tick shift) for the same reason.
//
// ── MODELS ──────────────────────────────────────────────────────────────────────────────────
//   driven single cell   tests/fixtures/nml/glif5_single_cell.nml, lengthened. The richest
//                        single-cell state in the tree: v, an adapting threshold theta, and two
//                        after-spike currents asc1/asc2 -- four continuous state variables with
//                        four different time constants, which is four independent chances for
//                        something to inflate.
//   recurrent network    generated by tests/nml_network_generator.cpp: 8 GLIF1 cells wired
//                        ExcPop -> ExcPop at out-degree 2 through a current-based
//                        alphaCurrentSynapse (2 per-edge state variables), each cell on its own
//                        pulseGenerator at a slightly different amplitude so they do not lock
//                        into one synchronous population. Recurrence is what makes a rate
//                        assertion worth having: a feedback loop is the thing that can actually
//                        run away.
//   delayed coupling     tests/fixtures/nml/delayed_coupling_network.nml, lengthened. Its
//                        connection carries a real 10ms (100-tick) delay, so the network_inputs
//                        ring is 101 rows deep and 100 of them hold not-yet-due arrivals at any
//                        instant. That is the accumulator's hardest case, and its target cell
//                        stays sub-threshold, so it is also the one cell here whose voltage is
//                        bounded by the LEAK rather than by a reset.
//
// The checked-in E/I fixture (glif_ei_network.nml) is deliberately NOT used: it declares
// expOneSynapse and an NMDA type, both conductance-based, which kernel_codegen refuses by name
// today. The generated network above is the current-based recurrent model this suite needs.
//
// ── COST ────────────────────────────────────────────────────────────────────────────────────
// Four 8000-tick runs plus five short falsification runs, about 12s wall clock for 35 tests. A
// tick is one Metal command buffer committed and waited on, ~0.27ms, independent of neuron count
// -- so duration is bought in ticks and networks are nearly free. LONG_RUN_TICK_COUNT below is
// the one knob; every tolerance is relative, so lowering it costs sensitivity to slow drift (the
// detectable per-tick drift scales as 1/LONG_RUN_TICK_COUNT) and nothing else.
//
// The whole-ring statistic adds one pass over ring_depth * neuron_count floats per tick, which is
// 202 on the deepest model here and does not register against the command buffer.

namespace {

// engine.h pulls in both `spikecorec` and `spikecorec::log`, and each declares its own `Vector`
// alias, so the unqualified name is ambiguous in any translation unit that includes it. This
// file-local alias picks the engine's one for the whole file.
template <typename SomeType>
using Vector = spikecorec::Vector<SomeType>;

// 800ms at the 0.1ms step every fixture here declares. See "COST" above.
constexpr s64 LONG_RUN_TICK_COUNT = 8000;

// Fraction of a run discarded before any verdict measures anything. Half is far more than any
// of these models needs -- the slowest constant in play is glif5's second after-spike current,
// which is inside f32 rounding of its fixed point by tick ~4700 of an 8000-tick run.
constexpr f64 SETTLE_FRACTION = 0.5;

// ── how much a state variable is allowed to move between the two halves of the settled run ───
// Applies to evaluate_state_is_stationary only; the rate and accumulator comparisons use the
// second and fourth quarters and their own bands. See "HOW THE ASSERTIONS ARE BUILT" above.
// Both are fractions, of the variable's OWN settled dynamic range and of its OWN largest
// magnitude respectively, so neither pins a number and neither needs revisiting when a fixture
// parameter changes. The range term carries the bound for an oscillating variable; the
// magnitude term is what keeps the bound meaningful for one that barely moves.
//
// Sized against measurement from both directions. The tightest real margin across all four
// models is 11.6x (the driven cell's asc1, still settling by a few parts in 10^4 at the half-way
// point); a variable ramping linearly across the settled run -- the defect shape -- shifts by
// half its range and so overshoots this bound by 15x. Tightening further would start to bite on
// asc1; loosening further would start to let a slow ramp through.
constexpr f64 STATIONARITY_RANGE_FRACTION = 0.03;
constexpr f64 STATIONARITY_MAGNITUDE_FRACTION = 0.003;

String fixture_path(const String &relative_path) {
    return String(SPIKECOREC_TEST_FIXTURES_DIR) + "/" + relative_path;
}

// The standard library is checked in under third_party/neuroml2/std_lib, not fetched, so its
// absence is a broken checkout rather than a configuration this suite may skip over.
bool standard_library_available() {
    nml::NML_Parser parser;
    return !parser.STANDARD_LIBRARY_PATH.empty() &&
           filesystem::exists(parser.STANDARD_LIBRARY_PATH);
}

// Fixtures are written to disk because the engine's unit of work is a file path, and into a
// temporary directory addressed by ABSOLUTE path because nothing here may depend on the
// process's working directory: this binary's working directory is shared with every other
// suite in it.
class FixtureDirectory {
public:
    explicit FixtureDirectory(const String &run_name) {
        root_ = filesystem::temp_directory_path() / "spikecorec_stability_tests" / run_name;
        filesystem::remove_all(root_);
        filesystem::create_directories(root_);
    }

    ~FixtureDirectory() {
        std::error_code ignored;
        filesystem::remove_all(root_, ignored);
    }

    FixtureDirectory(const FixtureDirectory &) = delete;
    FixtureDirectory &operator=(const FixtureDirectory &) = delete;

    // Returns the absolute path of the written file.
    String write(const String &relative_name, const String &contents) const {
        const filesystem::path destination = root_ / relative_name;
        ofstream file(destination);
        file << contents;
        file.close();
        return destination.string();
    }

private:
    filesystem::path root_;
};

String read_text_file(const String &path) {
    ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("stability_tests: cannot open '" + path + "'");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Throws rather than silently doing nothing when the text is absent, so that editing a
// checked-in fixture's simulation length or stimulus duration breaks this suite loudly instead
// of leaving it running a model it no longer describes.
String substitute_once(const String &text, const String &original, const String &replacement) {
    const usize position = text.find(original);
    if (position == String::npos) {
        throw std::runtime_error("stability_tests: fixture no longer contains '" + original +
                                 "' -- this suite's copy of it needs updating");
    }
    String result = text;
    result.replace(position, original.size(), replacement);
    return result;
}

// Drops every <OutputFile>/<EventOutputFile> block. The checked-in fixtures name their outputs
// RELATIVELY, so leaving them in would have the engine's recorders write into whatever working
// directory this binary happens to be in -- which is the shared working directory of every other
// suite in this binary, and two suites writing the same relative .dat name is a collision, not a
// diagnostic. Nothing here reads a recording -- every verdict reads engine buffers directly -- so
// the whole recording path is simply removed.
//
// Throws, like substitute_once and for the same reason, when an opening tag is found with no
// matching close: a self-closing <OutputFile .../> or a reformatted fixture would otherwise leave
// the block in place and silently re-enable the very recording this exists to remove. Failing to
// find the element at all is NOT an error -- a fixture is allowed to declare no outputs.
String remove_element_blocks(const String &text, const String &element_name) {
    String result = text;
    const String open_tag = "<" + element_name;
    const String close_tag = "</" + element_name + ">";
    for (;;) {
        const usize start = result.find(open_tag);
        if (start == String::npos) break;
        const usize end = result.find(close_tag, start);
        if (end == String::npos) {
            throw std::runtime_error("stability_tests: fixture opens '" + open_tag +
                                     "' with no matching '" + close_tag +
                                     "' -- this suite cannot strip the block, and leaving it in "
                                     "would have the engine write a recording into the shared "
                                     "working directory of the test binary");
        }
        result.erase(start, end + close_tag.size() - start);
    }
    return result;
}

// ── one recorded run ─────────────────────────────────────────────────────────────────────────

// Where one state variable of one neuron lives in the flat cell_state buffer. Resolved once at
// construction out of cell_state_base and the type's own declared order, rather than re-derived
// per tick.
struct StateSlotDescription {
    s64 neuron_index = -1;
    s64 flat_state_index = -1;
    String variable_name;

    // The regime index the generated kernel appends after every StateVariable (see
    // nml::cell_state_slot_count). Sampled like every other slot -- its window extremes say
    // whether the cell is still cycling between its regimes -- but named so a failure report
    // does not present a regime number as a membrane potential.
    bool is_regime_slot = false;
};

struct StabilityRun {
    String label;
    s64 tick_count = 0;
    s64 neuron_count = 0;
    s32 ring_depth = 1;
    f64 step_dt = 0.0;

    // How many per-edge synapse state variables the model carries. Zero means the lazy and
    // eager synapse-update paths have nothing to disagree about, which is what an equivalence
    // test needs to know before it can claim to have checked anything.
    s64 per_edge_synapse_variable_count = 0;

    Vector<StateSlotDescription> state_slots;

    // [tick][slot_index], parallel to state_slots.
    Vector<Vector<f32>> state_samples;

    // [tick] largest |network_inputs| anywhere in the ring once the tick completed. Non-zero
    // entries are arrivals scheduled into rows that are not due yet.
    Vector<f32> largest_ring_magnitude;

    // [tick] largest |network_inputs| left in the row THIS tick's cells read, after the engine's
    // end-of-tick ring clear ran. The engine's own invariant says this is exactly zero.
    Vector<f32> drained_row_residual;

    // [tick] largest change, anywhere in the ring OUTSIDE the row this tick drained, between the
    // end of the previous tick and the end of this one. This is the whole-ring statistic: every
    // element of every not-yet-due row is differenced, not just the one row `drained_row_residual`
    // reads. A non-zero entry means this tick scheduled a new arrival, and the engine schedules an
    // arrival only from the propagate stage, which runs only for a neuron whose spike flag this
    // same tick raised -- so a non-zero entry on a tick where nothing emitted is charge the engine
    // invented. See evaluate_ring_arrivals_follow_emissions.
    Vector<f32> largest_pending_row_change;

    // [tick][neuron], 1 where the neuron emitted on that tick.
    Vector<Vector<u8>> spike_flags;

    // Per neuron, straight out of the model's own resolved parameters; NaN where the cell type
    // declares no parameter of that name. These are what the model-derived bounds are built
    // from -- never a measured value.
    Vector<f32> resting_potential;   // EL
    Vector<f32> spike_threshold;     // vth
    Vector<f32> refractory_period;   // t_ref

    s64 settle_tick() const { return (s64)((f64)tick_count * SETTLE_FRACTION); }
    s64 midpoint_tick() const { return settle_tick() + (tick_count - settle_tick()) / 2; }

    // The two windows every across-time comparison in this file uses. See
    // evaluate_synaptic_accumulator_does_not_grow for why they are the second and fourth
    // quarters rather than the two halves of the settled run.
    s64 quarter_tick_count() const { return tick_count / 4; }
    s64 early_window_first_tick() const { return quarter_tick_count(); }
    s64 early_window_last_tick() const { return 2 * quarter_tick_count(); }
    s64 late_window_first_tick() const { return 3 * quarter_tick_count(); }
    s64 late_window_last_tick() const { return tick_count; }

    bool any_neuron_emitted(s64 tick) const {
        for (u8 flag : spike_flags[(usize)tick]) {
            if (flag != 0u) return true;
        }
        return false;
    }
};

// Runs after every tick and BEFORE that tick is sampled, with the engine's buffers in the state
// the tick left them. Used only by the falsification tests, to reconstruct a defect inside a
// real simulation: what the run then records is what a differently-behaving engine would have
// left behind, which is the only ordering under which a reconstructed defect is observable at
// all.
using AfterTickMutation = std::function<void(SpikeEngine &, s64)>;

StabilityRun run_long_simulation(const String &label, const String &model_path, s64 tick_count,
                                 bool use_lazy_synapse_updates,
                                 const AfterTickMutation &after_tick_mutation = nullptr) {
    String engine_input_path = model_path; // SpikeEngine takes a non-const String&
    SpikeEngine engine(engine_input_path, /*enable_hebbian_learning=*/false,
                       use_lazy_synapse_updates);

    StabilityRun run;
    run.label = label;
    run.tick_count = tick_count;
    run.neuron_count = engine.total_neuron_count;
    run.ring_depth = engine.network_input_ring_depth;
    run.step_dt = engine.network_details.step_dt;
    run.per_edge_synapse_variable_count = engine.per_edge_synapse_variable_count;

    const s32 *state_base = engine.cell_state_base.get_contents();
    const s32 *parameter_base = engine.cell_parameter_base.get_contents();
    const f32 *parameters = engine.cell_parameters.get_contents();

    for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
        const nml::Neuron &neuron = engine.network_details.neurons[(usize)neuron_index];
        const nml::CellTypeSpecification &cell_type =
                engine.network_details.cell_types[(usize)neuron.cell_type_index];

        const usize declared_count = cell_type.state_variable_names.size();
        const usize slot_count = nml::cell_state_slot_count(cell_type);
        for (usize slot_index = 0; slot_index < slot_count; slot_index += 1) {
            StateSlotDescription description;
            description.neuron_index = neuron_index;
            description.flat_state_index = state_base[neuron_index] + (s64)slot_index;
            description.is_regime_slot = slot_index >= declared_count;
            description.variable_name = description.is_regime_slot
                                                ? String("<regime>")
                                                : cell_type.state_variable_names[slot_index];
            run.state_slots.push_back(description);
        }

        auto parameter_value = [&](const String &parameter_name) -> f32 {
            for (usize index = 0; index < cell_type.parameter_names.size(); index += 1) {
                if (cell_type.parameter_names[index] == parameter_name) {
                    return parameters[parameter_base[neuron_index] + (s64)index];
                }
            }
            return std::numeric_limits<f32>::quiet_NaN();
        };
        run.resting_potential.push_back(parameter_value("EL"));
        run.spike_threshold.push_back(parameter_value("vth"));
        run.refractory_period.push_back(parameter_value("t_ref"));
    }

    run.state_samples.reserve((usize)tick_count);
    run.spike_flags.reserve((usize)tick_count);
    run.largest_ring_magnitude.reserve((usize)tick_count);
    run.drained_row_residual.reserve((usize)tick_count);
    run.largest_pending_row_change.reserve((usize)tick_count);

    const s64 ring_element_count = (s64)engine.network_input_ring_depth * run.neuron_count;

    // The whole ring as the previous tick left it, so this tick's arrivals can be isolated by
    // difference. Seeded with zeros because SpikeEngine's constructor zero-fills network_inputs,
    // which makes tick 0's difference as meaningful as every later one.
    Vector<f32> previous_ring((usize)ring_element_count, 0.0f);

    for (s64 tick = 0; tick < tick_count; tick += 1) {
        // commit_command_batch waits for the command buffer, so by the time this returns every
        // buffer below holds this tick's finished values and the host may read them directly
        // (Metal's are shared-storage allocations out of the engine arena).
        engine.step_simulation(tick);

        if (after_tick_mutation) after_tick_mutation(engine, tick);

        const f32 *cell_state = engine.cell_state.get_contents();
        Vector<f32> state_frame;
        state_frame.reserve(run.state_slots.size());
        for (const StateSlotDescription &slot_description : run.state_slots) {
            state_frame.push_back(cell_state[slot_description.flat_state_index]);
        }
        run.state_samples.push_back(std::move(state_frame));

        const s32 *spike_flags = engine.spike_flags.get_contents();
        Vector<u8> spike_frame((usize)run.neuron_count, 0);
        for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
            spike_frame[(usize)neuron_index] = spike_flags[neuron_index] != 0 ? 1u : 0u;
        }
        run.spike_flags.push_back(std::move(spike_frame));

        const f32 *ring = engine.network_inputs.get_contents();
        f32 largest_anywhere = 0.0f;
        for (s64 index = 0; index < ring_element_count; index += 1) {
            const f32 magnitude = std::fabs(ring[index]);
            // Written as a negated <= so a NaN in the ring propagates into the recorded
            // maximum instead of being silently discarded by a plain >.
            if (!(magnitude <= largest_anywhere)) largest_anywhere = magnitude;
        }
        run.largest_ring_magnitude.push_back(largest_anywhere);

        const s64 drained_row_base =
                (tick % (s64)engine.network_input_ring_depth) * run.neuron_count;
        f32 largest_in_drained_row = 0.0f;
        for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
            const f32 magnitude = std::fabs(ring[drained_row_base + neuron_index]);
            if (!(magnitude <= largest_in_drained_row)) largest_in_drained_row = magnitude;
        }
        run.drained_row_residual.push_back(largest_in_drained_row);

        // Every element of every OTHER row, differenced against what the previous tick left
        // there. The drained row is excluded because two engine-owned writes land in it during
        // this tick by design -- the host's external stimulus, added before the dispatch, and
        // the end-of-tick clear -- so its change says nothing about scheduling.
        f32 largest_change = 0.0f;
        for (s64 index = 0; index < ring_element_count; index += 1) {
            if (index >= drained_row_base && index < drained_row_base + run.neuron_count) continue;
            const f32 change = std::fabs(ring[index] - previous_ring[(usize)index]);
            if (!(change <= largest_change)) largest_change = change;
        }
        run.largest_pending_row_change.push_back(largest_change);

        for (s64 index = 0; index < ring_element_count; index += 1) {
            previous_ring[(usize)index] = ring[index];
        }
    }

    engine.shutdown();
    return run;
}

// ── the models ───────────────────────────────────────────────────────────────────────────────

String write_driven_single_cell_model(const FixtureDirectory &fixture, s64 duration_milliseconds) {
    String text = read_text_file(fixture_path("nml/glif5_single_cell.nml"));
    text = substitute_once(text, "length=\"350ms\"",
                           "length=\"" + to_string(duration_milliseconds) + "ms\"");
    // The stimulus is extended alongside the run: the whole point is a SUSTAINED drive, and a
    // fixture whose pulse ends part way through would be asserting that a decaying-to-rest cell
    // is stationary, which is true of a broken engine too.
    text = substitute_once(text, "duration=\"300ms\"",
                           "duration=\"" + to_string(duration_milliseconds) + "ms\"");
    text = remove_element_blocks(text, "OutputFile");
    text = remove_element_blocks(text, "EventOutputFile");
    fixture.write("glif5_single_cell.nml", text);
    return fixture.write("top.nml",
                         "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" "
                         "id=\"StabilityDrivenSingleCell\">"
                         "<include href=\"glif5_single_cell.nml\"/></neuroml>");
}

// `stimulus_duration_milliseconds` is separate from the run length only so the silenced-network
// falsification can build the same model with a drive that stops early.
String write_recurrent_network_model(const FixtureDirectory &fixture, s32 neuron_count,
                                     s64 duration_milliseconds,
                                     s64 stimulus_duration_milliseconds) {
    namespace generation = nml::network_generation;

    generation::GeneratedPopulation population;
    population.population_id = "ExcPop";
    population.variant = generation::GlifVariant::Glif1;
    population.bound_instance_id = "excCell";
    population.cell_instance_attributes =
            "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\" t_ref=\"2ms\"";
    population.neuron_count = neuron_count;

    generation::GeneratedProjection projection;
    projection.projection_id = "Recurrent";
    projection.presynaptic_population_id = "ExcPop";
    projection.postsynaptic_population_id = "ExcPop";
    projection.synapse_instance_id = "recurrentSynapse";
    projection.out_degree = 2;

    // One pulseGenerator per cell, each a little stronger than the last. Identical drives would
    // lock the population into one synchronous train, which is a far easier thing to keep
    // bounded than the interleaved recurrent traffic these staggered rates produce.
    Vector<generation::GeneratedStimulus> stimuli;
    for (s32 local_index = 0; local_index < neuron_count; local_index += 1) {
        generation::GeneratedStimulus stimulus;
        stimulus.population_id = "ExcPop";
        stimulus.local_index = local_index;
        stimulus.pulse_generator_id = "drive" + to_string(local_index);
        stimulus.delay = "5ms";
        stimulus.duration = to_string(stimulus_duration_milliseconds) + "ms";
        stimulus.amplitude = to_string(300 + 20 * local_index) + "pA";
        stimuli.push_back(stimulus);
    }

    fixture.write("net.nml",
                  generation::generate_network_nml(
                          "Recurrent", {population},
                          "  <alphaCurrentSynapse id=\"recurrentSynapse\" tau=\"2ms\" "
                          "ibase=\"40pA\"/>",
                          {projection}, stimuli));

    // The generator emits network content only (no <Simulation>), which is what it is for --
    // every caller wraps it in its own LEMS runner. The network's own id is <document_id>Net.
    return fixture.write("model.xml",
                         "<Lems>\n"
                         "  <Target component=\"sim1\"/>\n"
                         "  <Include file=\"net.nml\"/>\n"
                         "  <Simulation id=\"sim1\" length=\"" +
                                 to_string(duration_milliseconds) +
                                 "ms\" step=\"0.1ms\" target=\"RecurrentNet\"/>\n"
                                 "</Lems>\n");
}

String write_delayed_coupling_model(const FixtureDirectory &fixture, s64 duration_milliseconds) {
    String text = read_text_file(fixture_path("nml/delayed_coupling_network.nml"));
    text = substitute_once(text, "length=\"60ms\"",
                           "length=\"" + to_string(duration_milliseconds) + "ms\"");
    text = substitute_once(text, "duration=\"6ms\"",
                           "duration=\"" + to_string(duration_milliseconds) + "ms\"");
    text = remove_element_blocks(text, "OutputFile");
    text = remove_element_blocks(text, "EventOutputFile");
    fixture.write("delayed_coupling_network.nml", text);
    return fixture.write("top.nml",
                         "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" "
                         "id=\"StabilityDelayedCoupling\">"
                         "<include href=\"delayed_coupling_network.nml\"/></neuroml>");
}

// ── verdicts ─────────────────────────────────────────────────────────────────────────────────
//
// Each is a pure function of a recorded run, so the falsification tests can hand the SAME
// function a run carrying a defect and assert it is rejected. A verdict that lived inside a
// TEST body could not be falsified without also duplicating it.

struct Verdict {
    bool passed = true;
    String report;
};

Verdict evaluate_every_sample_is_finite(const StabilityRun &run) {
    for (s64 tick = 0; tick < run.tick_count; tick += 1) {
        for (usize slot_index = 0; slot_index < run.state_slots.size(); slot_index += 1) {
            const f32 value = run.state_samples[(usize)tick][slot_index];
            if (!std::isfinite(value)) {
                return {false, run.label + ": state variable '" +
                                       run.state_slots[slot_index].variable_name + "' of neuron " +
                                       to_string(run.state_slots[slot_index].neuron_index) +
                                       " is not finite at tick " + to_string(tick) + " (value " +
                                       to_string(value) + ")"};
            }
        }
        if (!std::isfinite(run.largest_ring_magnitude[(usize)tick])) {
            return {false, run.label + ": network_inputs holds a non-finite value at tick " +
                                   to_string(tick)};
        }
    }
    return {true, run.label + ": " + to_string(run.tick_count) + " ticks x " +
                          to_string(run.state_slots.size()) + " state slots all finite"};
}

// The bounded-state verdict. See this file's header for why the statistic is the (min, max) pair
// per half rather than the mean.
Verdict evaluate_state_is_stationary(const StabilityRun &run) {
    const s64 settle = run.settle_tick();
    const s64 midpoint = run.midpoint_tick();
    if (midpoint <= settle || run.tick_count <= midpoint) {
        return {false, run.label + ": run is too short to split into two settled halves"};
    }

    String worst_report;
    f64 worst_ratio = 0.0;

    for (usize slot_index = 0; slot_index < run.state_slots.size(); slot_index += 1) {
        f32 first_low = run.state_samples[(usize)settle][slot_index];
        f32 first_high = first_low;
        for (s64 tick = settle; tick < midpoint; tick += 1) {
            const f32 value = run.state_samples[(usize)tick][slot_index];
            first_low = std::min(first_low, value);
            first_high = std::max(first_high, value);
        }
        f32 second_low = run.state_samples[(usize)midpoint][slot_index];
        f32 second_high = second_low;
        for (s64 tick = midpoint; tick < run.tick_count; tick += 1) {
            const f32 value = run.state_samples[(usize)tick][slot_index];
            second_low = std::min(second_low, value);
            second_high = std::max(second_high, value);
        }

        const f64 settled_range =
                (f64)std::max(first_high, second_high) - (f64)std::min(first_low, second_low);
        const f64 largest_magnitude =
                std::max({std::fabs((f64)first_low), std::fabs((f64)first_high),
                          std::fabs((f64)second_low), std::fabs((f64)second_high)});
        const f64 allowed_shift = STATIONARITY_RANGE_FRACTION * settled_range +
                                  STATIONARITY_MAGNITUDE_FRACTION * largest_magnitude;

        const f64 low_shift = std::fabs((f64)second_low - (f64)first_low);
        const f64 high_shift = std::fabs((f64)second_high - (f64)first_high);
        const f64 largest_shift = std::max(low_shift, high_shift);

        const f64 ratio = largest_shift / std::max(allowed_shift, 1e-300);
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            std::ostringstream detail;
            detail.precision(9);
            detail << run.label << ": neuron " << run.state_slots[slot_index].neuron_index
                   << " state variable '" << run.state_slots[slot_index].variable_name
                   << "' first half [" << first_low << ", " << first_high << "] second half ["
                   << second_low << ", " << second_high << "] -- extremes moved by "
                   << largest_shift << ", allowed " << allowed_shift << " (" << settled_range
                   << " settled range, " << largest_magnitude << " largest magnitude)";
            worst_report = detail.str();
        }
    }

    if (worst_ratio > 1.0) return {false, worst_report};
    return {true, "worst of " + to_string(run.state_slots.size()) + " state slots at " +
                          to_string(worst_ratio) + " of its allowance -- " + worst_report};
}

// The one absolute bound in this file, and the only reason it can be absolute is that both ends
// come from the model's own declared parameters. A cell that resets on crossing `vth` cannot
// climb far past it, and one whose only drive is excitatory cannot fall far below `EL`; the
// allowance on each side is the model's own dynamic range |vth - EL|, which is the scale on
// which "far" is defined for THIS model rather than a number read off a run.
//
// Skips any neuron whose type declares no vth (glif5's threshold is a state variable, not a
// parameter) -- callers assert on `checked_neuron_count` so the skip cannot make it vacuous.
struct MembranePotentialBoundsVerdict {
    Verdict verdict;
    s64 checked_neuron_count = 0;
};

MembranePotentialBoundsVerdict evaluate_membrane_potential_within_model_bounds(
        const StabilityRun &run) {
    constexpr f64 OVERSHOOT_ALLOWANCE_FRACTION = 0.1;

    MembranePotentialBoundsVerdict result;
    for (usize slot_index = 0; slot_index < run.state_slots.size(); slot_index += 1) {
        if (run.state_slots[slot_index].variable_name != "v") continue;

        const s64 neuron_index = run.state_slots[slot_index].neuron_index;
        const f32 resting = run.resting_potential[(usize)neuron_index];
        const f32 threshold = run.spike_threshold[(usize)neuron_index];
        if (!std::isfinite(resting) || !std::isfinite(threshold)) continue;
        result.checked_neuron_count += 1;

        const f64 dynamic_range = std::fabs((f64)threshold - (f64)resting);
        const f64 highest_allowed = (f64)threshold + OVERSHOOT_ALLOWANCE_FRACTION * dynamic_range;
        const f64 lowest_allowed = (f64)resting - dynamic_range;

        for (s64 tick = run.settle_tick(); tick < run.tick_count; tick += 1) {
            const f64 value = (f64)run.state_samples[(usize)tick][slot_index];
            if (value > highest_allowed || value < lowest_allowed) {
                std::ostringstream detail;
                detail.precision(9);
                detail << run.label << ": neuron " << neuron_index << " membrane potential "
                       << value << " at tick " << tick << " is outside the model's own ["
                       << lowest_allowed << ", " << highest_allowed << "] (EL=" << resting
                       << ", vth=" << threshold << ")";
                result.verdict = {false, detail.str()};
                return result;
            }
        }
    }
    result.verdict = {true, run.label + ": membrane potential of " +
                                    to_string(result.checked_neuron_count) +
                                    " neuron(s) inside model-derived bounds"};
    return result;
}

// Exact, and exactly one half of the accumulator's contract. The engine clears the
// network_inputs row a tick read at the end of that same tick, behind the tick kernel in the
// same command batch, so once step_simulation returns that row holds literally nothing -- no
// tolerance, no settle period, no statistic. This is the assertion the shipped defect fails on
// its first delivered input, and it fails on EVERY tick after that rather than needing a long
// run to accumulate into view.
//
// What it CANNOT see, and the reason evaluate_ring_arrivals_follow_emissions exists next to it:
// it is a check that the ring EMPTIES, not a check that what the ring holds got there honestly.
// Charge written into a row that is not yet due is drained on schedule by the tick that row
// belongs to, so it never appears as a residual here no matter where it came from. A defect that
// carries a fraction of each tick's input forward into the next row satisfies this verdict on
// every one of 8000 ticks while delivering a hundred times the intended current.
Verdict evaluate_synaptic_accumulator_is_drained(const StabilityRun &run) {
    for (s64 tick = 0; tick < run.tick_count; tick += 1) {
        const f32 residual = run.drained_row_residual[(usize)tick];
        if (!(residual == 0.0f)) {
            std::ostringstream detail;
            detail.precision(9);
            detail << run.label << ": network_inputs row " << (tick % (s64)run.ring_depth)
                   << " still holds " << (f64)residual
                   << " after the tick that read it (tick " << tick
                   << ") -- the accumulator is not being drained";
            return {false, detail.str()};
        }
    }
    return {true, run.label + ": every one of " + to_string(run.tick_count) +
                          " ticks left its network_inputs row at exactly zero"};
}

// The accumulator's other half, and the only whole-ring assertion in this file: charge appears in
// network_inputs ONLY where a spike put it.
//
// The engine has exactly three writers of the ring, and each is pinned to one place:
//   * the host's Stage-1 stimulus add, into the row this tick is about to read;
//   * the end-of-tick clear kernel, over that same row;
//   * the generated propagate stage, into row (tick + edge delay) % ring_depth -- and the master
//     kernel gates the whole call on `if (spike_flags[neuron_index] != 0)`, with every delay at
//     least one tick, so it writes only to rows other than the current one and only on a tick
//     that emitted.
// The first two both act on the row `drained_row_residual` already covers, which is why
// `largest_pending_row_change` excludes it. Everything left is the third writer. So on a tick
// where NO neuron emitted, the whole not-yet-due part of the ring must be bit-identical to what
// the previous tick left there, and the assertion is `== 0` with no tolerance.
//
// This is what rejects an arrival the engine invented rather than delivered: a fraction of each
// tick's input carried forward into the next row, an edge double-counted because a padding lane
// of the k2-tree walk was read as a real neighbour, a synapse delivering an unscaled Q. Every one
// of those writes charge on a tick that scheduled nothing, and this verdict rejects the first
// such tick rather than waiting for a statistic to move.
//
// What it does NOT see: a wrong AMOUNT written on a tick that legitimately emitted. Sizing a
// delivery is a correctness property with no stability signature -- see "WHAT THIS SUITE DOES NOT
// COVER" in this file's header.
struct RingArrivalAccountingVerdict {
    Verdict verdict;
    // Ticks on which nothing emitted, so the ring was required to stand still. Zero of these
    // makes the verdict vacuous, which is why callers assert on it.
    s64 quiet_tick_count = 0;
    // Ticks on which the ring did gain content. Zero of these means the model routes nothing
    // through the not-yet-due rows at all, so the verdict is auditing an always-empty buffer.
    s64 arrival_tick_count = 0;
};

RingArrivalAccountingVerdict evaluate_ring_arrivals_follow_emissions(const StabilityRun &run) {
    RingArrivalAccountingVerdict result;
    for (s64 tick = 0; tick < run.tick_count; tick += 1) {
        const f32 change = run.largest_pending_row_change[(usize)tick];
        const bool emitted = run.any_neuron_emitted(tick);

        if (!(change == 0.0f)) result.arrival_tick_count += 1;
        if (emitted) continue;
        result.quiet_tick_count += 1;

        if (!(change == 0.0f)) {
            std::ostringstream detail;
            detail.precision(9);
            detail << run.label << ": no neuron emitted at tick " << tick
                   << ", yet network_inputs gained " << (f64)change
                   << " outside the row that tick drained -- charge appeared in the accumulator "
                      "that no spike scheduled";
            result.verdict = {false, detail.str()};
            return result;
        }
    }

    result.verdict = {true, run.label + ": over " + to_string(result.quiet_tick_count) +
                                    " ticks with no emission the whole not-yet-due ring stood "
                                    "exactly still, and " +
                                    to_string(result.arrival_tick_count) +
                                    " tick(s) scheduled an arrival"};
    return result;
}

// The rows a tick did NOT read hold arrivals that are not due yet, and those are bounded by the
// model's fan-in and per-spike charge rather than by anything the engine clears. Their bound is
// therefore a comparison against the run's own earlier self.
//
// The two windows are the SECOND and FOURTH quarters of the run, not the two halves of the
// settled part, and the reason is arithmetic: a quantity growing linearly from tick zero has a
// peak proportional to the window's end, so two adjacent halves differ by only 4/3 while the
// second and fourth quarters differ by 2. Against the 1.25 allowed here that is the difference
// between 7% of headroom and 60%. The second quarter is comfortably settled in all three models
// (the slowest drive starts at tick 100 with a 100-tick membrane constant), and unlike the first
// quarter it never contains the pre-stimulus interval where the ring is legitimately empty.
//
// 1.25 is loose against measurement, not against theory: this peak is BIT-IDENTICAL between the
// two windows in all three models, because it is the peak of a settled synaptic response.
//
// Two-sided, and the shrinking direction is not symmetry for its own sake: delivery that STOPS
// half way through the run leaves a settled-looking ring for the whole first half and an empty
// one afterwards, which a one-sided growth bound accepts. The emptiness check below covers the
// same failure at the other end -- traffic that existed only in the first quarter and had already
// stopped by the early window -- which is what an anti-vacuity guard taken over the WHOLE run
// misses.
Verdict evaluate_synaptic_accumulator_does_not_grow(const StabilityRun &run) {
    constexpr f64 ALLOWED_GROWTH_FACTOR = 1.25;

    f32 early_largest = 0.0f;
    for (s64 tick = run.early_window_first_tick(); tick < run.early_window_last_tick(); tick += 1) {
        early_largest = std::max(early_largest, run.largest_ring_magnitude[(usize)tick]);
    }
    f32 late_largest = 0.0f;
    for (s64 tick = run.late_window_first_tick(); tick < run.late_window_last_tick(); tick += 1) {
        late_largest = std::max(late_largest, run.largest_ring_magnitude[(usize)tick]);
    }

    std::ostringstream detail;
    detail.precision(9);
    detail << run.label << ": largest pending network_inputs magnitude was " << (f64)early_largest
           << " over the second quarter of the run and " << (f64)late_largest << " over the fourth";

    if (!(early_largest > 0.0f)) {
        return {false, detail.str() +
                               " -- nothing was pending in the ring over the second quarter, so "
                               "this comparison has no baseline and would accept any fourth "
                               "quarter at all"};
    }
    if ((f64)late_largest > ALLOWED_GROWTH_FACTOR * (f64)early_largest) {
        return {false, detail.str() + " -- growth beyond the allowed " +
                               to_string(ALLOWED_GROWTH_FACTOR) + "x"};
    }
    if ((f64)late_largest * ALLOWED_GROWTH_FACTOR < (f64)early_largest) {
        return {false, detail.str() + " -- a fall beyond the allowed " +
                               to_string(ALLOWED_GROWTH_FACTOR) +
                               "x, so delivery is dying out rather than settling"};
    }
    return {true, detail.str()};
}

// Not runaway and not dying out, over the SECOND and FOURTH quarters of the run.
//
// The windows are the ones evaluate_synaptic_accumulator_does_not_grow uses, and for the same
// arithmetic, which matters more here than it does there because the band is wider. A rate
// growing as r0(1 + a*t) has a window mean set by the window's midpoint, so two adjacent settled
// halves (midpoints at 5N/8 and 7N/8) differ by only a*N/4 while the second and fourth quarters
// (midpoints at 3N/8 and 7N/8) differ by a*N/2. Against the 25% floor below that is the
// difference between rejecting a ramp once it has grown 2x across the run and letting it reach
// 3.7x -- and 3.7x is not a subtle defect, it is the accumulating-input failure this file was
// written for, seen through the rate instead of through the ring.
//
// The band is derived from the statistic rather than picked, because the statistic is a spike
// COUNT and a count is coarse: a settled train whose period does not divide the window length
// lands one spike either side of its own average no matter how perfectly periodic it is. Three
// spikes of slack covers that on both windows at once, so the allowance is 3/N of the early
// window's count. The 25% floor is what stops the large-N case from becoming a periodicity test:
// a recurrent network is entitled to some structure over time, and a quarter is still far below
// the halving or doubling that extinction and self-amplification produce.
//
// It bounds the rate's SHAPE over time and nothing about its LEVEL: a network driven at any
// constant multiple of its intended current is stationary at every multiple. The level is
// covered, from the one side stability can reach, by
// evaluate_firing_rate_stays_below_the_refractory_ceiling; from the other side it is not covered
// here at all (see this file's header).
Verdict evaluate_firing_rate_is_stationary(const StabilityRun &run) {
    constexpr f64 SMALLEST_ALLOWED_DEVIATION = 0.25;
    constexpr f64 COUNT_GRANULARITY_SPIKES = 3.0;
    // Below this the count is too coarse for a ratio to mean anything, so the run is reported as
    // unusable rather than passed.
    constexpr s64 MINIMUM_EARLY_WINDOW_SPIKES = 4;

    auto window_spike_count = [&](s64 first_tick, s64 last_tick) {
        s64 count = 0;
        for (s64 tick = first_tick; tick < last_tick; tick += 1) {
            for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
                count += run.spike_flags[(usize)tick][(usize)neuron_index];
            }
        }
        return count;
    };

    const s64 early_spikes =
            window_spike_count(run.early_window_first_tick(), run.early_window_last_tick());
    const s64 late_spikes =
            window_spike_count(run.late_window_first_tick(), run.late_window_last_tick());

    const String counts = run.label + ": " + to_string(early_spikes) +
                          " spikes over the second quarter of the run, " + to_string(late_spikes) +
                          " over the fourth";

    if (early_spikes < MINIMUM_EARLY_WINDOW_SPIKES) {
        return {false, counts + " -- fewer than " + to_string(MINIMUM_EARLY_WINDOW_SPIKES) +
                               " spikes in the second quarter, so the model is either silent or "
                               "too slow for a rate comparison to say anything"};
    }

    const f64 allowed_deviation =
            std::max(SMALLEST_ALLOWED_DEVIATION, COUNT_GRANULARITY_SPIKES / (f64)early_spikes);
    const f64 ratio = (f64)late_spikes / (f64)early_spikes;
    if (ratio < 1.0 - allowed_deviation) {
        return {false, counts + " -- the rate fell to " + to_string(ratio) +
                               " of its earlier value, past the allowed " +
                               to_string(1.0 - allowed_deviation)};
    }
    if (ratio > 1.0 + allowed_deviation) {
        return {false, counts + " -- the rate rose to " + to_string(ratio) +
                               "x its earlier value, past the allowed " +
                               to_string(1.0 + allowed_deviation)};
    }
    return {true, counts + " (ratio " + to_string(ratio) + ", allowed deviation " +
                          to_string(allowed_deviation) + ")"};
}

// A hard ceiling straight out of the model: a cell holding itself in a refractory regime for
// t_ref after each emission cannot emit more often than once per t_ref. Unlike the ratio above
// this needs no settled baseline and no tolerance -- it is violated the moment the regime
// machinery stops holding, which is the mechanism that makes every rate in this engine finite.
//
// `checked_neuron_count` counts neurons whose type DECLARES a usable t_ref, which is not the same
// thing as neurons this verdict actually said anything about: a neuron that never emitted
// respects any refractory period at all. `emitting_neuron_count` is the one callers must guard
// on for the pass to mean something.
struct RefractoryCeilingVerdict {
    Verdict verdict;
    s64 checked_neuron_count = 0;
    s64 emitting_neuron_count = 0;
};

// A t_ref shorter than one tick floors to zero refractory ticks, which makes the comparison below
// vacuous and is a division by zero for anything that computes a rate ceiling from it. No fixture
// here declares one; a fixture edit that did should skip the neuron rather than crash or quietly
// pass, and should be visible as a shortfall in checked_neuron_count.
s64 refractory_tick_count(const StabilityRun &run, s64 neuron_index) {
    const f32 refractory_period = run.refractory_period[(usize)neuron_index];
    if (!std::isfinite(refractory_period) || refractory_period <= 0.0f) return 0;
    return (s64)std::floor((f64)refractory_period / run.step_dt);
}

RefractoryCeilingVerdict evaluate_firing_rate_respects_refractory_ceiling(const StabilityRun &run) {
    RefractoryCeilingVerdict result;
    for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
        const s64 refractory_ticks = refractory_tick_count(run, neuron_index);
        if (refractory_ticks <= 0) continue;
        result.checked_neuron_count += 1;

        s64 previous_spike_tick = -1;
        for (s64 tick = 0; tick < run.tick_count; tick += 1) {
            if (run.spike_flags[(usize)tick][(usize)neuron_index] == 0) continue;
            if (previous_spike_tick < 0) result.emitting_neuron_count += 1;
            if (previous_spike_tick >= 0 && tick - previous_spike_tick < refractory_ticks) {
                result.verdict = {false, run.label + ": neuron " + to_string(neuron_index) +
                                                 " emitted at ticks " +
                                                 to_string(previous_spike_tick) + " and " +
                                                 to_string(tick) + ", " +
                                                 to_string(tick - previous_spike_tick) +
                                                 " ticks apart, inside its own " +
                                                 to_string(refractory_ticks) +
                                                 "-tick refractory period"};
                return result;
            }
            previous_spike_tick = tick;
        }
    }
    result.verdict = {true, run.label + ": " + to_string(result.checked_neuron_count) +
                                    " neuron(s) respected their declared refractory period, " +
                                    to_string(result.emitting_neuron_count) + " of which emitted"};
    return result;
}

// The ceiling above says a cell cannot fire faster than once per t_ref. This says it is not
// sitting ON that ceiling -- and it exists because saturation is the one runaway the rate
// comparison is structurally blind to.
//
// A network driven so hard that every cell re-fires the instant its refractory clamp releases has
// a rate that is not merely stationary but EXACTLY constant, tick for tick, so
// evaluate_firing_rate_is_stationary reports a ratio of 1.000 and passes -- most confidently at
// the point where things have gone worst. Measured on a reconstructed defect the population sat
// at about 91% of this ceiling and every rate statistic in the file read as a clean steady state.
//
// Both ends are model-derived, in the sense rule 3 of this file's header means: the ceiling is
// the window length over the cell's own declared t_ref, and never a measured rate. The one
// judgement is the fraction of it a healthy run may occupy. A quarter of the ceiling is where the
// fixtures actually sit (see the measured figures on each test), which leaves this bound a factor
// of three above a correct engine and a factor of 1.2 below the saturated one.
//
// Measured over the fourth quarter alone rather than the whole settled run, so that a network
// that saturates only at the end -- the shape a self-amplifying loop produces -- cannot average
// itself back under the bound.
struct RefractoryHeadroomVerdict {
    Verdict verdict;
    s64 checked_neuron_count = 0;
    s64 emitting_neuron_count = 0;
};

RefractoryHeadroomVerdict evaluate_firing_rate_stays_below_the_refractory_ceiling(
        const StabilityRun &run) {
    constexpr f64 LARGEST_ALLOWED_CEILING_FRACTION = 0.75;

    RefractoryHeadroomVerdict result;
    const s64 window_tick_count = run.late_window_last_tick() - run.late_window_first_tick();

    for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
        const s64 refractory_ticks = refractory_tick_count(run, neuron_index);
        if (refractory_ticks <= 0) continue;
        result.checked_neuron_count += 1;

        s64 window_spikes = 0;
        for (s64 tick = run.late_window_first_tick(); tick < run.late_window_last_tick();
             tick += 1) {
            window_spikes += run.spike_flags[(usize)tick][(usize)neuron_index];
        }
        if (window_spikes > 0) result.emitting_neuron_count += 1;

        const f64 ceiling_spikes = (f64)window_tick_count / (f64)refractory_ticks;
        const f64 occupied_fraction = (f64)window_spikes / ceiling_spikes;
        if (occupied_fraction > LARGEST_ALLOWED_CEILING_FRACTION) {
            std::ostringstream detail;
            detail.precision(4);
            detail << run.label << ": neuron " << neuron_index << " emitted " << window_spikes
                   << " times over the fourth quarter of the run against a refractory ceiling of "
                   << ceiling_spikes << " (" << window_tick_count << " ticks / "
                   << refractory_ticks << "-tick t_ref) -- " << occupied_fraction
                   << " of the ceiling, past the allowed "
                   << LARGEST_ALLOWED_CEILING_FRACTION
                   << ", so the population is firing as fast as the refractory clamp permits "
                      "rather than settling";
            result.verdict = {false, detail.str()};
            return result;
        }
    }

    result.verdict = {true, run.label + ": " + to_string(result.checked_neuron_count) +
                                    " neuron(s) stayed clear of their refractory ceiling over the "
                                    "fourth quarter, " +
                                    to_string(result.emitting_neuron_count) + " of which emitted"};
    return result;
}

// Lazy and eager per-edge synapse updates apply the same per-tick step the same number of
// times, in the same order, so they are expected to agree BITWISE and not approximately -- and
// they do, over 8000 ticks. Asserting equality rather than a tolerance is what makes this
// sensitive to the drift it exists for: a per-edge catch-up that lands one step short shows up
// on the first spike, not after it has accumulated past some threshold.
Verdict evaluate_trajectories_are_identical(const StabilityRun &first, const StabilityRun &second) {
    if (first.tick_count != second.tick_count ||
        first.state_slots.size() != second.state_slots.size()) {
        return {false, "trajectory comparison needs two runs of the same model and length"};
    }

    for (s64 tick = 0; tick < first.tick_count; tick += 1) {
        for (usize slot_index = 0; slot_index < first.state_slots.size(); slot_index += 1) {
            const f32 first_value = first.state_samples[(usize)tick][slot_index];
            const f32 second_value = second.state_samples[(usize)tick][slot_index];
            if (first_value != second_value) {
                std::ostringstream detail;
                detail.precision(9);
                detail << first.label << " vs " << second.label << ": neuron "
                       << first.state_slots[slot_index].neuron_index << " state variable '"
                       << first.state_slots[slot_index].variable_name << "' first differs at tick "
                       << tick << ": " << first_value << " vs " << second_value;
                return {false, detail.str()};
            }
        }
        for (s64 neuron_index = 0; neuron_index < first.neuron_count; neuron_index += 1) {
            if (first.spike_flags[(usize)tick][(usize)neuron_index] !=
                second.spike_flags[(usize)tick][(usize)neuron_index]) {
                return {false, first.label + " vs " + second.label + ": neuron " +
                                       to_string(neuron_index) +
                                       " spike raster first differs at tick " + to_string(tick)};
            }
        }
    }
    return {true, first.label + " and " + second.label + " agree bitwise over " +
                          to_string(first.tick_count) + " ticks"};
}

// ── the runs, built once and shared ──────────────────────────────────────────────────────────
//
// Each long run costs a couple of seconds, and several verdicts read the same one, so each is
// built on first use and kept. The fixture directory is scoped to the builder: the model is
// fully parsed by the time SpikeEngine's constructor returns, so nothing needs the files
// afterwards.

const StabilityRun &driven_single_cell_run() {
    static const StabilityRun run = [] {
        FixtureDirectory fixture("driven_single_cell");
        return run_long_simulation("driven single cell (glif5)",
                                   write_driven_single_cell_model(fixture, 800),
                                   LONG_RUN_TICK_COUNT, /*use_lazy_synapse_updates=*/true);
    }();
    return run;
}

const StabilityRun &recurrent_network_run() {
    static const StabilityRun run = [] {
        FixtureDirectory fixture("recurrent_network_lazy");
        return run_long_simulation("recurrent network (lazy)",
                                   write_recurrent_network_model(fixture, 8, 800, 800),
                                   LONG_RUN_TICK_COUNT, /*use_lazy_synapse_updates=*/true);
    }();
    return run;
}

const StabilityRun &recurrent_network_eager_run() {
    static const StabilityRun run = [] {
        FixtureDirectory fixture("recurrent_network_eager");
        return run_long_simulation("recurrent network (eager)",
                                   write_recurrent_network_model(fixture, 8, 800, 800),
                                   LONG_RUN_TICK_COUNT, /*use_lazy_synapse_updates=*/false);
    }();
    return run;
}

const StabilityRun &delayed_coupling_run() {
    static const StabilityRun run = [] {
        FixtureDirectory fixture("delayed_coupling");
        return run_long_simulation("delayed coupling network",
                                   write_delayed_coupling_model(fixture, 800),
                                   LONG_RUN_TICK_COUNT, /*use_lazy_synapse_updates=*/true);
    }();
    return run;
}

// Counts the spikes a run recorded, so a test can say out loud that the model it just checked
// was actually doing something.
s64 total_spike_count(const StabilityRun &run) {
    s64 count = 0;
    for (const Vector<u8> &frame : run.spike_flags) {
        for (u8 flag : frame) count += flag;
    }
    return count;
}

// The narrowest settled span any membrane potential in the run covers. The stationarity verdict
// asks whether a variable's extremes MOVED between two halves, and a variable that never moves
// at all satisfies that trivially -- so every test of it first asserts this is positive, which
// is the cheapest available statement that the simulation is running rather than sitting on its
// initial conditions. `v` specifically, because it is the one variable that has to move in any
// live cell; refractory timers and regime indices are legitimately frozen for a cell that never
// spikes, and asserting on those would fail on the delayed-coupling network's sub-threshold
// target for a correct reason.
f64 narrowest_membrane_potential_range(const StabilityRun &run) {
    f64 narrowest = std::numeric_limits<f64>::infinity();
    for (usize slot_index = 0; slot_index < run.state_slots.size(); slot_index += 1) {
        if (run.state_slots[slot_index].variable_name != "v") continue;
        f32 lowest = run.state_samples[(usize)run.settle_tick()][slot_index];
        f32 highest = lowest;
        for (s64 tick = run.settle_tick(); tick < run.tick_count; tick += 1) {
            const f32 value = run.state_samples[(usize)tick][slot_index];
            lowest = std::min(lowest, value);
            highest = std::max(highest, value);
        }
        narrowest = std::min(narrowest, (f64)highest - (f64)lowest);
    }
    return std::isfinite(narrowest) ? narrowest : 0.0;
}

} // namespace

// ── driven single cell ───────────────────────────────────────────────────────────────────────
//
// glif5: v, an adapting threshold theta, and after-spike currents asc1/asc2, under a sustained
// 0.6nA step for the whole 800ms.

TEST(StabilityDrivenSingleCell, every_state_sample_is_finite) {
    if (!standard_library_available()) GTEST_SKIP();
    const Verdict verdict = evaluate_every_sample_is_finite(driven_single_cell_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

// This is where the state-stationarity verdict earns its place: `theta`, `asc1` and `asc2` are
// continuous variables with no reset pinning either end of them, so their window extremes are
// free to move and a defect that inflates any of them shows up here and nowhere else in the file.
// See "WHAT THIS SUITE DOES NOT COVER" in the header for the models where that is NOT true.
//
// Measured margins under this verdict, worst variable first: asc1 11.6x its allowance, theta
// 26.4x, v 27.6x, and asc2 / refractoryTimeElapsed / the regime index all exactly stationary.
// asc1 is the tightest because it is the slowest to settle, not because it drifts.
//
// `v`'s margin is the one to discount: its window minimum is bit-identically vreset in both
// halves, exactly as on the recurrent network, so only the crossing sample moves. theta, asc1 and
// asc2 are the slots this test is really covering.
TEST(StabilityDrivenSingleCell, state_is_stationary_across_the_settled_run) {
    if (!standard_library_available()) GTEST_SKIP();
    ASSERT_GT(narrowest_membrane_potential_range(driven_single_cell_run()), 0.0)
            << "no membrane potential moved at all over the settled run, so a stationarity check "
               "on it would be checking a constant";
    const Verdict verdict = evaluate_state_is_stationary(driven_single_cell_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

// This model has no synapses at all, so the only thing that ever passes through network_inputs
// is the pulseGenerator's own injected current -- which makes it the cleanest possible check
// that the stimulus delivery path drains itself. The spike-count assertion is what says the
// path was carrying anything: a cell driven through an accumulator that delivered nothing would
// sit at EL and never fire.
TEST(StabilityDrivenSingleCell, synaptic_accumulator_is_drained_every_tick) {
    if (!standard_library_available()) GTEST_SKIP();
    const StabilityRun &run = driven_single_cell_run();
    ASSERT_GT(total_spike_count(run), 0)
            << "the cell never fired, so nothing was delivered through network_inputs and this "
               "test would be checking an accumulator that is empty for a different reason";
    const Verdict verdict = evaluate_synaptic_accumulator_is_drained(run);
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

// The other half of the accumulator contract on the simplest possible ring: this model has no
// out-edges at all, so the propagate stage never runs and NOTHING may ever appear outside the row
// the current tick is reading. Any arrival at all here is charge the engine invented.
TEST(StabilityDrivenSingleCell, no_arrival_appears_in_the_ring_without_an_emission) {
    if (!standard_library_available()) GTEST_SKIP();
    const RingArrivalAccountingVerdict result =
            evaluate_ring_arrivals_follow_emissions(driven_single_cell_run());
    ASSERT_GT(result.quiet_tick_count, 0)
            << "every tick of this run emitted, so there was no tick on which the ring was "
               "required to stand still and this verdict checked nothing";
    EXPECT_EQ(result.arrival_tick_count, 0)
            << "this model has no synapses, so the propagate stage has no edge to write down and "
               "the not-yet-due rows must stay empty for the whole run: " << result.verdict.report;
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

TEST(StabilityDrivenSingleCell, firing_rate_is_stationary) {
    if (!standard_library_available()) GTEST_SKIP();
    const Verdict verdict = evaluate_firing_rate_is_stationary(driven_single_cell_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

TEST(StabilityDrivenSingleCell, firing_rate_respects_the_refractory_ceiling) {
    if (!standard_library_available()) GTEST_SKIP();
    const RefractoryCeilingVerdict result =
            evaluate_firing_rate_respects_refractory_ceiling(driven_single_cell_run());
    EXPECT_EQ(result.checked_neuron_count, driven_single_cell_run().neuron_count)
            << "every neuron in this model declares t_ref, so a skipped one means the parameter "
               "is no longer being resolved";
    EXPECT_EQ(result.emitting_neuron_count, driven_single_cell_run().neuron_count)
            << "a neuron that never emitted respects any refractory period at all, so a shortfall "
               "here means the verdict passed without examining an interval";
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

// Measured: 6 emissions over the fourth quarter against a ceiling of 40.8 (2000 ticks over a
// 50-tick t_ref) -- 0.147 of the ceiling, against an allowance of 0.75.
TEST(StabilityDrivenSingleCell, firing_rate_stays_clear_of_the_refractory_ceiling) {
    if (!standard_library_available()) GTEST_SKIP();
    const RefractoryHeadroomVerdict result =
            evaluate_firing_rate_stays_below_the_refractory_ceiling(driven_single_cell_run());
    ASSERT_GT(result.emitting_neuron_count, 0)
            << "no neuron emitted over the fourth quarter, so every one of them is trivially "
               "clear of its ceiling and this verdict checked nothing";
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

// ── recurrent network ────────────────────────────────────────────────────────────────────────
//
// 8 GLIF1 cells wired back onto themselves at out-degree 2 through a current-based
// alphaCurrentSynapse, each on its own slightly different drive.

TEST(StabilityRecurrentNetwork, every_state_sample_is_finite) {
    if (!standard_library_available()) GTEST_SKIP();
    const Verdict verdict = evaluate_every_sample_is_finite(recurrent_network_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

// Read this one for what it is, not for the margins it reports. Every one of this model's 24
// state slots has extremes PINNED BY CONSTRUCTION, so none of them can register the drift the
// verdict is written to catch:
//
//   * `v` -- every cell here spikes, and a spiking cell's window minimum is bit-identically
//     `vreset` (the reset assignment writes it) and its maximum is whichever sample first landed
//     past `vth`. The minimum cannot move at all. The maximum moves only by where in the tick
//     grid the crossing fell, which is sub-tick jitter of the crossing and not a measure of
//     anything the engine got wrong -- so the margin this test reports is jitter headroom, NOT
//     sensitivity, and it does not become a safety factor by being large.
//   * `refractoryTimeElapsed` -- cycles between 0 and t_ref by the regime machinery's own
//     definition.
//   * the regime index -- cycles over {0, 1} for the same reason.
//
// It is kept because the pinning itself is worth asserting: a `vreset` that moved, or a
// threshold-crossing sample that landed somewhere new, is a real change. It is NOT counted as
// coverage of state drift. The slots that carry that are named in this file's header.
TEST(StabilityRecurrentNetwork, state_extremes_stay_pinned_across_the_settled_run) {
    if (!standard_library_available()) GTEST_SKIP();
    ASSERT_GT(narrowest_membrane_potential_range(recurrent_network_run()), 0.0)
            << "at least one neuron's membrane potential never moved over the settled run";
    const Verdict verdict = evaluate_state_is_stationary(recurrent_network_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

TEST(StabilityRecurrentNetwork, membrane_potential_stays_within_model_derived_bounds) {
    if (!standard_library_available()) GTEST_SKIP();
    const MembranePotentialBoundsVerdict result =
            evaluate_membrane_potential_within_model_bounds(recurrent_network_run());
    EXPECT_EQ(result.checked_neuron_count, recurrent_network_run().neuron_count)
            << "every neuron in this model declares both EL and vth, so a skipped one means the "
               "bound was never applied to it";
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

// The anti-vacuity guard the single cell's counterpart has and this one used to be missing: a
// ring nothing ever passes through is drained on every tick of the most broken engine imaginable.
TEST(StabilityRecurrentNetwork, synaptic_accumulator_is_drained_every_tick) {
    if (!standard_library_available()) GTEST_SKIP();
    const StabilityRun &run = recurrent_network_run();
    ASSERT_GT(total_spike_count(run), 0)
            << "no cell fired, so no edge ever delivered and this test would be checking an "
               "accumulator that is empty for a different reason";
    const Verdict verdict = evaluate_synaptic_accumulator_is_drained(run);
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

// The whole-ring half of the same contract, and the one that discriminates on this model: with
// out-degree 2 and roughly nine ticks in ten carrying no emission at all, an arrival the engine
// invented has thousands of ticks on which to show itself.
//
// Measured: 7422 of the 8000 ticks carried no emission and the ring stood exactly still on every
// one of them; the other 578 scheduled an arrival.
TEST(StabilityRecurrentNetwork, no_arrival_appears_in_the_ring_without_an_emission) {
    if (!standard_library_available()) GTEST_SKIP();
    const RingArrivalAccountingVerdict result =
            evaluate_ring_arrivals_follow_emissions(recurrent_network_run());
    ASSERT_GT(result.quiet_tick_count, 0)
            << "every tick emitted, so the ring was never required to stand still";
    ASSERT_GT(result.arrival_tick_count, 0)
            << "no tick ever scheduled an arrival into a not-yet-due row, so this verdict is "
               "auditing a buffer nothing passes through";
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

// The anti-vacuity guard matters more here than anywhere else in the file: on a model whose
// ring never holds anything, "it did not grow" is true of a completely broken engine too. It is
// taken over the verdict's own EARLY window rather than over the whole run, because traffic that
// existed only in the first quarter and had already stopped satisfies a whole-run guard while
// leaving the comparison itself with no baseline.
TEST(StabilityRecurrentNetwork, synaptic_accumulator_does_not_grow_with_tick_count) {
    if (!standard_library_available()) GTEST_SKIP();
    const StabilityRun &run = recurrent_network_run();

    f32 largest_pending = 0.0f;
    for (s64 tick = run.early_window_first_tick(); tick < run.early_window_last_tick(); tick += 1) {
        largest_pending = std::max(largest_pending, run.largest_ring_magnitude[(usize)tick]);
    }
    ASSERT_GT(largest_pending, 0.0f)
            << "no synaptic arrival was pending in network_inputs anywhere in the second quarter "
               "of the run, so there is nothing here for a growth check to be checking";

    const Verdict verdict = evaluate_synaptic_accumulator_does_not_grow(run);
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

// Measured: 173 spikes over the second quarter of the run, 176 over the fourth -- a ratio of
// 1.017 against a band of [0.75, 1.25]. The driven single cell measures 6 against 6 in a band of
// [0.5, 1.5], its 3-spike granularity allowance being the wider term at that count.
TEST(StabilityRecurrentNetwork, firing_rate_is_stationary) {
    if (!standard_library_available()) GTEST_SKIP();
    const Verdict verdict = evaluate_firing_rate_is_stationary(recurrent_network_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

TEST(StabilityRecurrentNetwork, firing_rate_respects_the_refractory_ceiling) {
    if (!standard_library_available()) GTEST_SKIP();
    const RefractoryCeilingVerdict result =
            evaluate_firing_rate_respects_refractory_ceiling(recurrent_network_run());
    EXPECT_EQ(result.checked_neuron_count, recurrent_network_run().neuron_count);
    EXPECT_EQ(result.emitting_neuron_count, recurrent_network_run().neuron_count)
            << "a neuron that never emitted respects any refractory period at all, so a shortfall "
               "here means the verdict passed without examining an interval";
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

// The saturation check. This is the model it exists for: a recurrent loop is the thing that can
// drive itself into firing as fast as the refractory clamp permits, and that is the state the
// rate comparison above reports as a perfect 1.000.
//
// Measured: the eight cells occupy 0.19 to 0.26 of their ceiling (19 to 26 emissions over the
// fourth quarter against 2000 ticks / a 20-tick t_ref = 100), against an allowance of 0.75.
//
// That margin is a property of THIS fixture's drive, and it closes as the drive rises: at six
// times the fixture's pulseGenerator amplitudes the same measurement reads 0.65. A fixture change
// that raised the amplitudes materially would need this bound re-derived rather than assumed --
// the same caution the header attaches to every margin quoted in this file.
TEST(StabilityRecurrentNetwork, firing_rate_stays_clear_of_the_refractory_ceiling) {
    if (!standard_library_available()) GTEST_SKIP();
    const RefractoryHeadroomVerdict result =
            evaluate_firing_rate_stays_below_the_refractory_ceiling(recurrent_network_run());
    ASSERT_GT(result.emitting_neuron_count, 0)
            << "no neuron emitted over the fourth quarter, so every one of them is trivially "
               "clear of its ceiling and this verdict checked nothing";
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

// The long-run half of the lazy/eager equivalence. A short-run version of this exists in
// tests/kernel_codegen_tests.cpp; what only a long run can see is a per-edge catch-up that
// drifts by an accumulating fraction of a step rather than by a whole one.
TEST(StabilityRecurrentNetwork, lazy_and_eager_synapse_updates_agree_bitwise) {
    if (!standard_library_available()) GTEST_SKIP();
    ASSERT_GT(recurrent_network_run().per_edge_synapse_variable_count, 0)
            << "this model carries no per-edge synapse state, so the lazy and eager paths have "
               "nothing to disagree about and this test would pass on any engine";
    // Per-edge state that was never advanced past its seed is identical under both paths for a
    // reason that has nothing to do with either being right: a lazy catch-up of zero ticks and an
    // eager advance of zero ticks are the same arithmetic. Both paths must actually have carried
    // an edge for the agreement to be evidence.
    ASSERT_GT(total_spike_count(recurrent_network_run()), 0)
            << "no cell fired under the lazy path, so no edge was ever driven and both paths sit "
               "at the seed value";
    ASSERT_GT(total_spike_count(recurrent_network_eager_run()), 0)
            << "no cell fired under the eager path, so no edge was ever driven and both paths sit "
               "at the seed value";
    const Verdict verdict = evaluate_trajectories_are_identical(recurrent_network_run(),
                                                                recurrent_network_eager_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

// ── delayed coupling network ─────────────────────────────────────────────────────────────────
//
// A 100-tick connection delay, so the ring is 101 rows deep and 100 rows hold not-yet-due
// arrivals at every instant -- and a sub-threshold target cell, the one voltage here bounded by
// its leak rather than by a reset.

TEST(StabilityDelayedCouplingNetwork, every_state_sample_is_finite) {
    if (!standard_library_available()) GTEST_SKIP();
    const Verdict verdict = evaluate_every_sample_is_finite(delayed_coupling_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

// The other model whose state stationarity means something. The target cell never crosses its
// threshold, so neither end of its `v` is pinned by a reset -- both are set by the leak against
// the arriving current, and both are free to move if anything about that delivery changes. It is
// the only membrane potential in this file of which that is true.
//
// Measured: every one of the 6 state slots exactly stationary between the two settled halves,
// both extremes bit-identical -- including the target cell's `v` over its [-67.4mV, -64.7mV]
// settled span.
TEST(StabilityDelayedCouplingNetwork, state_is_stationary_across_the_settled_run) {
    if (!standard_library_available()) GTEST_SKIP();
    ASSERT_GT(narrowest_membrane_potential_range(delayed_coupling_run()), 0.0)
            << "the sub-threshold target cell's membrane potential never moved, so the delayed "
               "connection is no longer delivering anything";
    const Verdict verdict = evaluate_state_is_stationary(delayed_coupling_run());
    EXPECT_TRUE(verdict.passed) << verdict.report;
}

TEST(StabilityDelayedCouplingNetwork, membrane_potential_stays_within_model_derived_bounds) {
    if (!standard_library_available()) GTEST_SKIP();
    const MembranePotentialBoundsVerdict result =
            evaluate_membrane_potential_within_model_bounds(delayed_coupling_run());
    EXPECT_EQ(result.checked_neuron_count, delayed_coupling_run().neuron_count);
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

TEST(StabilityDelayedCouplingNetwork, delay_ring_is_drained_and_does_not_grow) {
    if (!standard_library_available()) GTEST_SKIP();
    const StabilityRun &run = delayed_coupling_run();
    ASSERT_GT(run.ring_depth, 2) << "the fixture's connection delay is what makes this the deep-"
                                    "ring case; a depth of 2 means the delay stopped resolving";

    f32 largest_pending = 0.0f;
    for (s64 tick = run.early_window_first_tick(); tick < run.early_window_last_tick(); tick += 1) {
        largest_pending = std::max(largest_pending, run.largest_ring_magnitude[(usize)tick]);
    }
    ASSERT_GT(largest_pending, 0.0f) << "no arrival was pending in the delay ring anywhere in the "
                                        "second quarter of the run, which is the window the "
                                        "growth comparison takes as its baseline";

    const Verdict drained = evaluate_synaptic_accumulator_is_drained(run);
    EXPECT_TRUE(drained.passed) << drained.report;
    const Verdict bounded = evaluate_synaptic_accumulator_does_not_grow(run);
    EXPECT_TRUE(bounded.passed) << bounded.report;
}

// The deep ring is where the drained-row check is at its weakest -- it inspects one row in 101 on
// any given tick -- and so it is where the whole-ring check matters most. Every one of the 100
// not-yet-due rows is differenced here on every tick.
//
// Measured: 7890 of the 8000 ticks carried no emission and all 100 not-yet-due rows stood exactly
// still on every one of them; the other 110 scheduled an arrival.
TEST(StabilityDelayedCouplingNetwork, no_arrival_appears_in_the_ring_without_an_emission) {
    if (!standard_library_available()) GTEST_SKIP();
    const RingArrivalAccountingVerdict result =
            evaluate_ring_arrivals_follow_emissions(delayed_coupling_run());
    ASSERT_GT(result.quiet_tick_count, 0)
            << "every tick emitted, so the ring was never required to stand still";
    ASSERT_GT(result.arrival_tick_count, 0)
            << "no tick ever scheduled an arrival into a not-yet-due row, so the delayed "
               "connection is delivering nothing and this verdict is auditing an empty buffer";
    EXPECT_TRUE(result.verdict.passed) << result.verdict.report;
}

// ── falsification ────────────────────────────────────────────────────────────────────────────
//
// Every verdict above, fed something it must reject. Without these the suite can go vacuous --
// a tolerance widened past the point of meaning, a comparison quietly comparing a run against
// itself -- and still report green, which is the exact failure a stability suite exists to
// prevent.

namespace {

// Reconstructs the shipped network_inputs defect inside a real simulation: after every tick,
// puts back into the row the engine just cleared the running total of everything delivered so
// far, so the next reader of that row sees per_tick_amount, 2x, 3x, ... exactly the
// 5e-10, 1e-9, 1.5e-9, 2e-9 ramp the original defect produced.
AfterTickMutation reinject_accumulated_input(f32 per_tick_amount) {
    auto running_total = std::make_shared<f32>(0.0f);
    return [per_tick_amount, running_total](SpikeEngine &engine, s64 tick) {
        *running_total += per_tick_amount;
        f32 *ring = engine.network_inputs.get_contents();
        const s64 row_base =
                (tick % (s64)engine.network_input_ring_depth) * engine.total_neuron_count;
        for (s64 neuron_index = 0; neuron_index < engine.total_neuron_count; neuron_index += 1) {
            ring[row_base + neuron_index] = *running_total;
        }
    };
}

// The other shape of accumulator defect, and the one the drained-row check cannot see: instead of
// leaving charge in the row it just read, the engine carries a fraction of what that row held
// forward into the NEXT row, where it is delivered on schedule one tick later along with that
// tick's own input. Every row still drains to exactly zero on the tick that reads it; the
// accumulator's residual is exactly zero on all 8000 ticks; and the network receives roughly
// 1/(1-retained_fraction) times the current the model asked for.
//
// This is the bounded form of the shipped defect. The unbounded one grew without limit and was
// visible in any statistic taken across time; this one settles to a fixed multiple of the
// intended drive within a few hundred ticks and is invisible in every such statistic, because a
// network at a hundred times its intended drive is perfectly stable -- just wrong.
//
// `retained_fraction` is what survives each hand-off, so the carried amount converges on
// per_tick_amount * retained_fraction / (1 - retained_fraction) rather than growing without
// bound: at 0.99 that is ninety-nine extra copies of every tick's input, arriving forever.
AfterTickMutation carry_a_fraction_of_each_tick_forward(f32 per_tick_amount,
                                                        f32 retained_fraction) {
    auto carried = std::make_shared<f32>(0.0f);
    return [per_tick_amount, retained_fraction, carried](SpikeEngine &engine, s64 tick) {
        *carried = retained_fraction * (*carried + per_tick_amount);
        f32 *ring = engine.network_inputs.get_contents();
        const s64 next_row_base =
                ((tick + 1) % (s64)engine.network_input_ring_depth) * engine.total_neuron_count;
        for (s64 neuron_index = 0; neuron_index < engine.total_neuron_count; neuron_index += 1) {
            ring[next_row_base + neuron_index] += *carried;
        }
    };
}

// An arrival scheduled `rows_ahead` rows into the future that no spike asked for. On a deep ring
// this is the defect the drained-row check is furthest from being able to see: the row it lands
// in is not the row that check inspects on this tick, and by the time it IS that row the arrival
// has been delivered and cleared on schedule, so the residual is zero at every instant the check
// looks.
//
// It is the shape a mis-computed delay produces -- an edge scheduling into the wrong row, or a
// ring index wrapping one period out.
AfterTickMutation schedule_an_unearned_arrival(s64 rows_ahead, f32 amount) {
    return [rows_ahead, amount](SpikeEngine &engine, s64 tick) {
        f32 *ring = engine.network_inputs.get_contents();
        const s64 row_base = ((tick + rows_ahead) % (s64)engine.network_input_ring_depth) *
                             engine.total_neuron_count;
        for (s64 neuron_index = 0; neuron_index < engine.total_neuron_count; neuron_index += 1) {
            ring[row_base + neuron_index] += amount;
        }
    };
}

// Injects a current large enough that one forward-Euler step overflows f32, so the non-finite
// samples the finiteness verdict sees are ones the engine's own integration produced rather than
// values written in by hand.
//
// The sign is what makes the overflow observable, and it is worth stating because getting it
// wrong makes this test silently vacuous: a POSITIVE overflow drives v to +inf, which satisfies
// the cell's own threshold condition, so the same kernel invocation resets it and the sample
// that reaches the host is a perfectly finite vreset. Driven negative, nothing resets it -- v
// goes to -inf, and the next step's (EL - v) turns that into a NaN that persists.
AfterTickMutation inject_overflowing_current(s64 from_tick) {
    return [from_tick](SpikeEngine &engine, s64 tick) {
        if (tick < from_tick) return;
        f32 *ring = engine.network_inputs.get_contents();
        const s64 row_base =
                (tick % (s64)engine.network_input_ring_depth) * engine.total_neuron_count;
        for (s64 neuron_index = 0; neuron_index < engine.total_neuron_count; neuron_index += 1) {
            ring[row_base + neuron_index] = -1e36f;
        }
    };
}

// A copy of `run` whose every neuron fires as often as its own refractory period allows over
// the second settled half -- the physical ceiling on runaway, applied to a real raster. Built
// this way rather than by adding arbitrary spikes so that the result still satisfies the
// refractory-ceiling verdict, which isolates what is being falsified to the rate comparison
// alone.
StabilityRun with_saturated_firing_in_second_half(const StabilityRun &run) {
    StabilityRun saturated = run;
    saturated.label = run.label + " [firing saturated in the second half]";

    for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
        // Zero refractory ticks is a modulo by zero below, so it is refused rather than skipped:
        // a fixture whose t_ref fell under one step would otherwise turn this falsification into
        // a crash, and the falsification tests are the ones that must not fail obscurely.
        const s64 refractory_ticks = refractory_tick_count(run, neuron_index);
        if (refractory_ticks <= 0) continue;

        // The synthetic train starts one whole refractory period after the midpoint, not at it:
        // the last REAL emission before the midpoint can be as late as midpoint - 1, and a
        // synthetic spike placed at the midpoint itself would then sit inside that emission's
        // refractory period -- making this run violate the refractory ceiling as well, which is
        // the one verdict it must not violate if it is to isolate the rate comparison.
        const s64 first_synthetic_tick = run.midpoint_tick() + refractory_ticks;
        for (s64 tick = run.midpoint_tick(); tick < run.tick_count; tick += 1) {
            const bool on_the_train = tick >= first_synthetic_tick &&
                                      (tick - first_synthetic_tick) % refractory_ticks == 0;
            saturated.spike_flags[(usize)tick][(usize)neuron_index] = on_the_train ? 1u : 0u;
        }
    }
    return saturated;
}

// A copy of `run` firing at its refractory ceiling for the WHOLE run rather than only the second
// half. The distinction is the entire point of the headroom verdict: a network that saturates
// abruptly half way through moves its rate and the rate comparison rejects it, but one that is
// already saturated when the first measurement window opens has a rate that is exactly constant,
// and every comparison of the run against its own earlier self reports a clean steady state.
StabilityRun with_firing_saturated_throughout(const StabilityRun &run) {
    StabilityRun saturated = run;
    saturated.label = run.label + " [firing saturated for the whole run]";

    for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
        const s64 refractory_ticks = refractory_tick_count(run, neuron_index);
        if (refractory_ticks <= 0) continue;
        for (s64 tick = 0; tick < run.tick_count; tick += 1) {
            saturated.spike_flags[(usize)tick][(usize)neuron_index] =
                    tick % refractory_ticks == 0 ? 1u : 0u;
        }
    }
    return saturated;
}

// A copy of `run` with `slot_index`'s samples replaced by a linear ramp spanning the same range the
// real variable spans. Built out of a REAL run rather than from nothing, so what the verdict is
// rejecting is a real trajectory with one variable made to inflate, at the amplitude that
// variable really has.
StabilityRun with_linear_drift_on_slot(const StabilityRun &run, usize slot_index) {
    StabilityRun drifting = run;
    drifting.label = run.label + " [linear drift injected]";

    f32 lowest = run.state_samples[0][slot_index];
    f32 highest = lowest;
    for (const Vector<f32> &frame : run.state_samples) {
        lowest = std::min(lowest, frame[slot_index]);
        highest = std::max(highest, frame[slot_index]);
    }

    for (s64 tick = 0; tick < drifting.tick_count; tick += 1) {
        const f32 fraction = (f32)tick / (f32)drifting.tick_count;
        drifting.state_samples[(usize)tick][slot_index] = lowest + fraction * (highest - lowest);
    }
    return drifting;
}

// A copy of `run` whose firing rate grows LINEARLY across the run by `growth_factor`, built by
// dropping real emissions rather than by inventing any. Dropping is what makes it usable: the
// result is still a real raster, it cannot violate the refractory ceiling, and the rate
// comparison is therefore the only verdict it can be rejected by.
//
// A linear rate ramp is the raster this file's founding defect produces -- an input accumulating
// with tick count drives an ever-shorter interval between crossings -- and it is the shape the
// two-window choice is sized against.
//
// The thinning is deterministic, not sampled: each neuron carries a credit that gains
// keep_fraction per real emission and spends one whole unit per emission it keeps, so the kept
// density tracks keep_fraction with no run-to-run variation for a bound to have to absorb.
StabilityRun with_rate_ramped_by_dropping_emissions(const StabilityRun &run, f64 growth_factor) {
    StabilityRun ramping = run;
    ramping.label = run.label + " [rate ramped " + to_string(growth_factor) + "x across the run]";

    const f64 earliest_keep_fraction = 1.0 / growth_factor;
    for (s64 neuron_index = 0; neuron_index < run.neuron_count; neuron_index += 1) {
        f64 credit = 0.0;
        for (s64 tick = 0; tick < run.tick_count; tick += 1) {
            if (ramping.spike_flags[(usize)tick][(usize)neuron_index] == 0) continue;

            const f64 position = (f64)tick / (f64)run.tick_count;
            credit += earliest_keep_fraction + (1.0 - earliest_keep_fraction) * position;
            if (credit >= 1.0) {
                credit -= 1.0;
            } else {
                ramping.spike_flags[(usize)tick][(usize)neuron_index] = 0u;
            }
        }
    }
    return ramping;
}

} // namespace

// The shipped defect, reconstructed and driving a real recurrent network, against the two
// verdicts that exist for it.
//
// The state-stationarity verdict is deliberately NOT among them, and that absence is the finding
// this suite is built around: every cell here resets on crossing its threshold, so a ramping
// input raises the RATE and leaves v pinned between vreset and vth exactly as before. Cell state
// stayed bounded and plausible while every voltage and spike time in the model was wrong -- which
// is how the original defect passed 196 tests, and why the accumulator gets an exact assertion
// of its own rather than being covered by the bounded-state one.
//
// The ramp injected here is 20pA per tick, a twenty-fifth of the 0.5nA per tick the real defect
// produced.
TEST(StabilityFalsification, undrained_synaptic_accumulator_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    FixtureDirectory fixture("falsify_undrained");
    const StabilityRun run = run_long_simulation(
            "undrained accumulator", write_recurrent_network_model(fixture, 8, 300, 300), 3000,
            /*use_lazy_synapse_updates=*/true, reinject_accumulated_input(2e-11f));

    const Verdict drained = evaluate_synaptic_accumulator_is_drained(run);
    EXPECT_FALSE(drained.passed)
            << "an accumulator that is never drained was accepted: " << drained.report;

    const Verdict bounded = evaluate_synaptic_accumulator_does_not_grow(run);
    EXPECT_FALSE(bounded.passed)
            << "an accumulator growing linearly with tick count was accepted: " << bounded.report;

    // The one thing that must still hold: this defect must not also be breaking the refractory
    // machinery, or the rejections above would not be attributable to the accumulator.
    const RefractoryCeilingVerdict ceiling =
            evaluate_firing_rate_respects_refractory_ceiling(run);
    EXPECT_TRUE(ceiling.verdict.passed) << ceiling.verdict.report;
}

// The accumulator defect the drained-row check is structurally blind to, reconstructed inside a
// real recurrent network: a fraction of every tick's input carried forward into the next ring row
// instead of dropped. It settles at about a hundred extra copies of the drive, and the network
// runs at roughly that multiple of its intended current for the whole run.
//
// Measured on this exact run, with every other verdict in the file applied to it:
//     drained row        PASS -- all 3000 ticks left their row at exactly zero
//     ring not growing   PASS -- 4.39491146e-08 in the second quarter, bit-identical in the
//                                fourth, because a geometric carry converges long before either
//     rate stationary    PASS -- 272 spikes against 272, ratio 1.000000
//     state stationary   PASS -- worst of 24 slots at 0.000000 of its allowance
// against a raster carrying 1088 emissions where the same model unmutated carries about 260.
// Every verdict this file had before agrees the engine is healthy.
//
// Those four are not asserted on here. A later strengthening of any of them would be an
// improvement, not a regression, and an assertion that a defect slips past a verdict turns that
// improvement into a failure. They are recorded above instead, where a reader can check them.
//
// The whole-ring check is the one that rejects it, and it does so on the first tick where nothing
// emitted, not after a statistic has moved.
TEST(StabilityFalsification, input_carried_forward_into_the_next_ring_row_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    FixtureDirectory fixture("falsify_carried_forward");
    // 0.4nA per tick is this model's own pulseGenerator amplitude, so the carried total settles
    // at about a hundred times the current the fixture asks for.
    const StabilityRun run = run_long_simulation(
            "input carried forward", write_recurrent_network_model(fixture, 8, 300, 300), 3000,
            /*use_lazy_synapse_updates=*/true,
            carry_a_fraction_of_each_tick_forward(4e-10f, 0.99f));

    const RingArrivalAccountingVerdict arrivals = evaluate_ring_arrivals_follow_emissions(run);
    EXPECT_FALSE(arrivals.verdict.passed)
            << "charge appearing in the ring on ticks where nothing emitted was accepted: "
            << arrivals.verdict.report;

    // The same isolation the other accumulator falsification makes: this defect must not also be
    // breaking the refractory machinery, or the rejection would not be attributable to the ring.
    const RefractoryCeilingVerdict ceiling = evaluate_firing_rate_respects_refractory_ceiling(run);
    EXPECT_TRUE(ceiling.verdict.passed) << ceiling.verdict.report;
}

// The same accounting on the DEEP ring, which is where the coverage argument is sharpest. The
// delayed-coupling fixture's 10ms connection delay makes the ring 101 rows deep, so a per-tick
// inspection of the one row the current tick drained sees 1/101 of the buffer; the arrival here
// is placed 50 rows ahead, in a row that is drained -- to exactly zero, on schedule -- fifty ticks
// after it appears. Every residual this file used to measure is zero for the whole run.
//
// Measured on this exact run: the drained-row check passes on all 3000 ticks, and the growth
// check reads 2.76828183e-09 in the second quarter and the same to the bit in the fourth.
//
// The whole-ring difference sees it on the first tick that emitted nothing.
TEST(StabilityFalsification, an_arrival_scheduled_deep_in_the_delay_ring_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    FixtureDirectory fixture("falsify_deep_ring_arrival");
    const StabilityRun run = run_long_simulation(
            "unearned deep-ring arrival", write_delayed_coupling_model(fixture, 300), 3000,
            /*use_lazy_synapse_updates=*/true,
            schedule_an_unearned_arrival(/*rows_ahead=*/50, 5e-11f));
    ASSERT_GT(run.ring_depth, 50) << "the ring is no longer deep enough for an arrival 50 rows "
                                     "ahead to be a different row from the one being drained";

    const RingArrivalAccountingVerdict arrivals = evaluate_ring_arrivals_follow_emissions(run);
    EXPECT_FALSE(arrivals.verdict.passed)
            << "an arrival scheduled into a not-yet-due row by no spike at all was accepted: "
            << arrivals.verdict.report;
}

TEST(StabilityFalsification, arithmetic_overflow_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    FixtureDirectory fixture("falsify_overflow");
    const StabilityRun run = run_long_simulation(
            "f32 overflow", write_driven_single_cell_model(fixture, 50), 500,
            /*use_lazy_synapse_updates=*/true, inject_overflowing_current(200));

    const Verdict verdict = evaluate_every_sample_is_finite(run);
    EXPECT_FALSE(verdict.passed)
            << "a run containing non-finite state was accepted: " << verdict.report;
}

// The rate verdict's runaway direction, at its physical ceiling: every neuron of a real recurrent
// raster firing as often as its own t_ref allows over the second settled half.
//
// A raster mutation rather than a live run, because a linearly accumulating INPUT cannot produce
// this: the accumulated current at the end of a settled run is only 4/3 of what it was half way
// through, which moves the rate by tens of percent and not by the factor this verdict is for.
// The rate assertion guards a self-amplifying loop, which is a different defect from the
// accumulator one above, and the two have separate falsifications because they are separate
// failures.
TEST(StabilityFalsification, runaway_firing_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    const StabilityRun saturated = with_saturated_firing_in_second_half(recurrent_network_run());

    const Verdict rate = evaluate_firing_rate_is_stationary(saturated);
    EXPECT_FALSE(rate.passed) << "a network firing at its refractory ceiling over the second half "
                                 "of the run was accepted as stationary: "
                              << rate.report;

    // Saturation is the ceiling, not a violation of it -- so the refractory verdict must still
    // pass, which is what says the rejection above came from the rate comparison alone.
    const RefractoryCeilingVerdict ceiling =
            evaluate_firing_rate_respects_refractory_ceiling(saturated);
    EXPECT_TRUE(ceiling.verdict.passed) << ceiling.verdict.report;
}

// The blind spot the headroom verdict was added for, and the whole reason it is not redundant
// with the rate comparison above. The raster here is at the refractory ceiling for the ENTIRE
// run, not just the second half, so there is no moment at which the rate changes.
//
// The two EXPECT_TRUEs are the demonstration, not incidental checks:
//   * the rate comparison passes, and MUST pass -- an exactly constant rate is stationary, which
//     is precisely what that verdict means. No tightening of it could reject this run without
//     making it wrong. This is the sense in which the rate statistic is blindest exactly where
//     the network has gone worst.
//   * the refractory ceiling passes, because one emission per t_ref is the ceiling and not a
//     breach of it.
// Only the headroom verdict rejects, which is what says the two refractory verdicts are asking
// different questions rather than the same one twice.
TEST(StabilityFalsification, firing_at_the_refractory_ceiling_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    const StabilityRun saturated = with_firing_saturated_throughout(recurrent_network_run());

    const RefractoryHeadroomVerdict headroom =
            evaluate_firing_rate_stays_below_the_refractory_ceiling(saturated);
    EXPECT_FALSE(headroom.verdict.passed)
            << "a network emitting as often as its own t_ref permits was accepted as leaving "
               "headroom below its ceiling: "
            << headroom.verdict.report;

    const Verdict rate = evaluate_firing_rate_is_stationary(saturated);
    EXPECT_TRUE(rate.passed)
            << "a rate that is exactly constant was rejected as non-stationary, which means the "
               "rate verdict no longer means what this file says it means: "
            << rate.report;

    const RefractoryCeilingVerdict ceiling =
            evaluate_firing_rate_respects_refractory_ceiling(saturated);
    EXPECT_TRUE(ceiling.verdict.passed) << ceiling.verdict.report;
}

// The rate comparison's own sensitivity, at the shape it is sized for. A real raster thinned so
// its rate grows linearly by 2.5x across the run is rejected by the second-quarter-against-fourth
// windows this verdict uses; the two settled HALVES it used to compare put the same raster at a
// ratio of about 1.19 and accepted it, and would have gone on accepting it up to a growth of
// roughly 3.7x across the run.
//
// Only dropping emissions, never adding any, so the refractory verdicts must both still pass --
// which is what says the rejection came from the rate comparison and not from the raster having
// been made physically impossible.
TEST(StabilityFalsification, a_linearly_ramping_firing_rate_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    const StabilityRun ramping =
            with_rate_ramped_by_dropping_emissions(recurrent_network_run(), 2.5);

    const Verdict rate = evaluate_firing_rate_is_stationary(ramping);
    EXPECT_FALSE(rate.passed) << "a firing rate growing by 2.5x across the run was accepted as "
                                 "stationary: "
                              << rate.report;

    const RefractoryCeilingVerdict ceiling =
            evaluate_firing_rate_respects_refractory_ceiling(ramping);
    EXPECT_TRUE(ceiling.verdict.passed) << ceiling.verdict.report;
    const RefractoryHeadroomVerdict headroom =
            evaluate_firing_rate_stays_below_the_refractory_ceiling(ramping);
    EXPECT_TRUE(headroom.verdict.passed) << headroom.verdict.report;
}

// The rate verdict's other direction: a network whose drive stops part way through goes silent,
// and silence is not stationarity.
//
// The drive outlasts the verdict's early window and stops before its late one -- 120ms of a 200ms
// run, against windows at 50-100ms and 150-200ms -- so the rejection comes from the ratio itself
// having collapsed, with a real spike count on both sides of it. A drive that stopped before the
// early window too would leave that window empty, and the verdict would refuse the run as
// unusable rather than reject it as non-stationary: a true answer to a different question, and
// one that would leave the falling branch of the comparison untested.
TEST(StabilityFalsification, silenced_network_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    FixtureDirectory fixture("falsify_silenced");
    const StabilityRun run = run_long_simulation(
            "silenced network",
            write_recurrent_network_model(fixture, 8, /*duration=*/200, /*stimulus_duration=*/120),
            2000, /*use_lazy_synapse_updates=*/true);

    const Verdict verdict = evaluate_firing_rate_is_stationary(run);
    EXPECT_FALSE(verdict.passed)
            << "a network that fell silent was accepted as stationary: " << verdict.report;
    EXPECT_NE(verdict.report.find("the rate fell to"), String::npos)
            << "the rejection did not come from the rate comparison itself, so the falling branch "
               "of the band is still untested: "
            << verdict.report;

    // Delivery that stops half way is also what the growth check's shrinking direction is for,
    // and it used to be one-sided.
    const Verdict bounded = evaluate_synaptic_accumulator_does_not_grow(run);
    EXPECT_FALSE(bounded.passed)
            << "synaptic delivery that stopped half way through the run was accepted: "
            << bounded.report;
}

// Not a verdict but a fixture-preparation step, and falsified for the same reason the verdicts
// are: it used to fail SILENTLY. An <OutputFile> whose close tag it cannot find is left in the
// model, and both checked-in fixtures name their recordings relatively, so the engine then writes
// a .dat file into the working directory this binary shares with every other suite in it.
TEST(StabilityFalsification, an_unclosed_output_block_is_refused_rather_than_left_in_place) {
    const String self_closing =
            "<Simulation id=\"sim1\"><OutputFile id=\"trace\" fileName=\"trace.dat\"/>"
            "</Simulation>";
    EXPECT_THROW(remove_element_blocks(self_closing, "OutputFile"), std::runtime_error);

    // The shapes it must still handle: a real block is removed, and a fixture declaring no
    // outputs at all is passed through untouched.
    EXPECT_EQ(remove_element_blocks("<a/><OutputFile id=\"t\"><OutputColumn/></OutputFile><b/>",
                                    "OutputFile"),
              "<a/><b/>");
    EXPECT_EQ(remove_element_blocks("<a/><b/>", "OutputFile"), "<a/><b/>");
}

// The state-stationarity verdict, fed each state variable of a real run in turn with that one
// variable made to ramp. Every variable, not one: a verdict that rejects a ramp on `v` and
// silently ignores one on `asc2` is only a third of a test.
TEST(StabilityFalsification, linear_state_drift_is_rejected_on_every_state_variable) {
    if (!standard_library_available()) GTEST_SKIP();
    const StabilityRun &run = driven_single_cell_run();

    s64 rejected_count = 0;
    for (usize slot_index = 0; slot_index < run.state_slots.size(); slot_index += 1) {
        const Verdict verdict = evaluate_state_is_stationary(with_linear_drift_on_slot(run, slot_index));
        EXPECT_FALSE(verdict.passed)
                << "a linear ramp on state variable '" << run.state_slots[slot_index].variable_name
                << "' of neuron " << run.state_slots[slot_index].neuron_index
                << " was accepted as stationary: " << verdict.report;
        if (!verdict.passed) rejected_count += 1;
    }
    EXPECT_EQ(rejected_count, (s64)run.state_slots.size());
}

// The model-derived bound, fed a real run whose voltage has been ramped past the ceiling those
// parameters imply.
TEST(StabilityFalsification, membrane_potential_above_the_model_bound_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    StabilityRun run = recurrent_network_run();
    run.label += " [voltage inflated]";

    const f32 threshold = run.spike_threshold[0];
    const f32 resting = run.resting_potential[0];
    ASSERT_TRUE(std::isfinite(threshold) && std::isfinite(resting));

    // Just past the allowance the verdict derives from |vth - EL|, so this rejects the bound's
    // own edge rather than some value far outside it.
    const f32 inflated = threshold + 0.2f * std::fabs(threshold - resting);
    for (usize slot_index = 0; slot_index < run.state_slots.size(); slot_index += 1) {
        if (run.state_slots[slot_index].variable_name != "v" ||
            run.state_slots[slot_index].neuron_index != 0) {
            continue;
        }
        run.state_samples[(usize)(run.tick_count - 1)][slot_index] = inflated;
    }

    const MembranePotentialBoundsVerdict result =
            evaluate_membrane_potential_within_model_bounds(run);
    EXPECT_FALSE(result.verdict.passed)
            << "a membrane potential past the model's own ceiling was accepted: "
            << result.verdict.report;
}

// The lazy/eager verdict, fed the two real trajectories with one shifted by a single tick --
// which is precisely what a per-edge catch-up that takes one step too few or too many produces.
TEST(StabilityFalsification, one_tick_trajectory_shift_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    StabilityRun shifted = recurrent_network_eager_run();
    shifted.label += " [shifted one tick]";
    shifted.state_samples.erase(shifted.state_samples.begin());
    shifted.state_samples.push_back(shifted.state_samples.back());
    shifted.spike_flags.erase(shifted.spike_flags.begin());
    shifted.spike_flags.push_back(shifted.spike_flags.back());

    const Verdict verdict = evaluate_trajectories_are_identical(recurrent_network_run(), shifted);
    EXPECT_FALSE(verdict.passed)
            << "a trajectory shifted by one tick was accepted as identical: " << verdict.report;
}

// The refractory ceiling, fed a real raster with one extra emission inserted immediately after
// a real one.
TEST(StabilityFalsification, firing_inside_the_refractory_period_is_rejected) {
    if (!standard_library_available()) GTEST_SKIP();
    StabilityRun run = recurrent_network_run();
    run.label += " [extra emission inserted]";

    s64 inserted_at = -1;
    for (s64 tick = 0; tick + 1 < run.tick_count && inserted_at < 0; tick += 1) {
        if (run.spike_flags[(usize)tick][0] == 0) continue;
        run.spike_flags[(usize)(tick + 1)][0] = 1;
        inserted_at = tick + 1;
    }
    ASSERT_GE(inserted_at, 0) << "neuron 0 never fired, so there was nowhere to insert";

    const RefractoryCeilingVerdict result = evaluate_firing_rate_respects_refractory_ceiling(run);
    EXPECT_FALSE(result.verdict.passed)
            << "two emissions one tick apart were accepted against a 20-tick refractory period: "
            << result.verdict.report;
}
