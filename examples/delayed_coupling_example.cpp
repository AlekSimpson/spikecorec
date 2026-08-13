// ── Axonal delays: a spike arrives on its own tick, and not before ──────────────────────
//
// A connection that declares no delay still costs one tick: a source scatters its weight into
// the target's accumulator and the target drains it on its next step. A real
// <connectionWD delay="10ms"/> has to cost exactly ten milliseconds instead, which needs the
// DELAY RING -- `network_inputs` is not one flat [neuron_count] array but a ring of
// `max_delay_ticks + 1` rows, each a full [neuron_count] wide, and an arrival is scattered
// into the row belonging to its own arrival tick. A cell's read of its row is a plain load,
// not a drain: the row is zeroed afterwards, by a separate clear kernel dispatched behind the
// tick kernel in the same command batch, so it is clean again by the time the ring wraps onto
// it. That split is exactly the redesign this example demonstrates -- every other row holds
// arrivals that are not due yet and must survive untouched.
//
// This measures the delivery offset directly rather than describing it. One source drives
// three otherwise-untouched targets over three connections declaring 1ms, 5ms and 10ms. Each
// target's membrane potential is watched tick by tick; the tick it first moves is the tick
// the spike landed. The synapse is deliberately subthreshold, so a target's v rises and then
// decays back without firing -- an arrival is a clean, isolated event rather than something
// that has to be inferred from a spike train.
//
// A target's v is bit-exactly its resting EL until its own arrival: nothing else touches it,
// and a resting cell's membrane derivative is exactly zero. So "the first tick v moved" is an
// exact statement, not a threshold crossing, and the ARRIVAL_MOVEMENT_THRESHOLD below only
// has to be small enough to catch the first tick of an alpha current's rise (see its note).

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::examples;

namespace {

// Neuron 0 is the source; 1, 2 and 3 are the targets, one per declared delay.
constexpr s64 NEURON_COUNT = 4;
const char *const DECLARED_DELAYS[] = {"1ms", "5ms", "10ms"};

// The delay synapse's `ibase`, in amperes: the peak of the alpha current profile one arriving
// spike injects. 0.2nA delivers e * ibase * tau = 1.1pC into a 100pF membrane leaking with a
// 10ms time constant, which peaks at about 6.5mV -- unmistakable against a resting cell, and
// a third of the 20mV it would take to actually fire one.
constexpr f64 SUBTHRESHOLD_PEAK_CURRENT = 0.2 * NANOAMPERE;

// Rise and decay time of that alpha profile.
const char *const SYNAPSE_TIME_CONSTANT = "2ms";

// How far v has to move in one tick to count as "the arrival landed", in volts.
//
// An alpha current starts at zero and ramps, so the first tick after an arrival is the
// smallest step there is: the synapse integrates one dt to i = dt * e * ibase / tau = 27pA,
// which moves a 100pF membrane by dt * 27pA / 100pF = 27uV. This threshold sits 27x below
// that, and infinitely above what precedes it -- v does not move at all until the arrival.
constexpr f32 ARRIVAL_MOVEMENT_THRESHOLD = 1e-6f;

std::string delayed_coupling_network_nml() {
    std::ostringstream document;
    document << "<neuroml id=\"delayedCouplingNetwork\">\n    "
             << glif_cell_instance(GlifVariant::Glif1, "delayCell") << "\n"
             << "    <alphaCurrentSynapse id=\"delaySynapse\" tau=\"" << SYNAPSE_TIME_CONSTANT
             << "\" ibase=\"" << nanoampere_attribute(SUBTHRESHOLD_PEAK_CURRENT) << "\"/>\n"
                "    <pulseGenerator id=\"sourceDrive\" delay=\"20ms\" duration=\"1000ms\" "
                "amplitude=\"0.6nA\"/>\n\n"
                "    <network id=\"delayedCouplingNet\">\n"
                "        <population id=\"pop\" component=\"delayCell\" size=\""
             << NEURON_COUNT << "\"/>\n\n";

    for (usize delay_index = 0; delay_index < std::size(DECLARED_DELAYS); ++delay_index) {
        document << "        <projection id=\"delayed" << delay_index
                 << "\" presynapticPopulation=\"pop\"\n"
                    "                    postsynapticPopulation=\"pop\" synapse=\"delaySynapse\">\n"
                    "            <connectionWD id=\"0\" preCellId=\"../pop[0]\" postCellId=\"../pop["
                 << delay_index + 1 << "]\"\n                          weight=\"1\" delay=\""
                 << DECLARED_DELAYS[delay_index] << "\"/>\n        </projection>\n\n";
    }

    document << "        <explicitInput target=\"pop[0]\" input=\"sourceDrive\"/>\n"
                "    </network>\n</neuroml>\n";
    return document.str();
}

std::string delayed_coupling_lems_document(const std::string &network_file_name) {
    std::ostringstream document;
    document << "<Lems>\n    <Target component=\"delayedCouplingSimulation\"/>\n"
             << glif1_component_type() << "\n    <Include file=\"" << network_file_name
             << "\"/>\n\n"
                "    <Simulation id=\"delayedCouplingSimulation\" length=\"60ms\" step=\"0.1ms\"\n"
                "                target=\"delayedCouplingNet\"/>\n</Lems>\n";
    return document.str();
}

} // namespace

int main(int argument_count, char **argument_values) {
    try {
        const ExampleOptions options =
                parse_example_options(argument_count, argument_values,
                                      /*default_synapse_peak_current=*/SUBTHRESHOLD_PEAK_CURRENT);
        configure_logging(options);

        // The first local owning anything GPU-backed, so it destructs last.
        GpuContextScope gpu_context;

        const GeneratedModelDirectory model_directory("delayed_coupling");
        model_directory.write("delayed_coupling_network.nml", delayed_coupling_network_nml());
        String model_path = model_directory.write(
                "delayed_coupling_model.xml",
                delayed_coupling_lems_document("delayed_coupling_network.nml"));

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

        const f64 seconds_per_tick = engine.network_details.step_dt;
        const s64 ticks_per_millisecond = (s64)std::llround(0.001 / seconds_per_tick);

        print_heading("Three connections, three declared delays, one source");
        std::cout << "  " << engine.total_neuron_count << " neurons; neuron 0 is driven, 1..3 "
                     "receive only its spikes\n"
                  << "  ring depth " << engine.network_input_ring_depth << " rows, which is one "
                     "more than the largest\n  declared delay ("
                  << (engine.network_input_ring_depth - 1) << " ticks = "
                  << (f64)(engine.network_input_ring_depth - 1) / (f64)ticks_per_millisecond
                  << "ms): an arrival scheduled this tick can never land in\n  the row being "
                     "drained this tick.\n";

        // The first tick each target's membrane potential moves at all. Everything but neuron
        // 0 starts at rest and is otherwise untouched, so the first movement IS the arrival.
        const s32 *spike_flags = engine.spike_flags.get_contents();
        spikecorec::Vector<f32> previous_potential((usize)NEURON_COUNT, 0.0f);
        spikecorec::Vector<s64> arrival_tick((usize)NEURON_COUNT, -1);
        s64 source_first_spike_tick = -1;

        for (s64 neuron_index = 0; neuron_index < NEURON_COUNT; ++neuron_index) {
            previous_potential[(usize)neuron_index] = read_cell_state(engine, neuron_index, "v");
        }

        run_simulation(engine, options.tick_count, [&](s64 tick) {
            if (source_first_spike_tick < 0 && spike_flags[0] != 0) source_first_spike_tick = tick;

            for (s64 neuron_index = 1; neuron_index < NEURON_COUNT; ++neuron_index) {
                const f32 potential = read_cell_state(engine, neuron_index, "v");
                if (arrival_tick[(usize)neuron_index] < 0 &&
                    potential - previous_potential[(usize)neuron_index] >
                            ARRIVAL_MOVEMENT_THRESHOLD) {
                    arrival_tick[(usize)neuron_index] = tick;
                }
                previous_potential[(usize)neuron_index] = potential;
            }
        });

        if (source_first_spike_tick < 0) {
            std::cout << "\n  The source never fired -- nothing to measure.\n";
            engine.shutdown();
            return 1;
        }

        std::cout << "\n  Source fired at tick " << source_first_spike_tick << ".\n\n"
                  << "    target   declared    measured offset    as declared?\n";
        bool every_delay_matched = true;
        for (usize delay_index = 0; delay_index < std::size(DECLARED_DELAYS); ++delay_index) {
            const s64 target_index = (s64)delay_index + 1;
            const s64 measured_offset =
                    arrival_tick[(usize)target_index] < 0
                            ? -1
                            : arrival_tick[(usize)target_index] - source_first_spike_tick;
            const s64 expected_offset =
                    (s64)std::llround(std::atof(DECLARED_DELAYS[delay_index])) *
                    ticks_per_millisecond;
            const bool matched = measured_offset == expected_offset;
            every_delay_matched = every_delay_matched && matched;

            std::cout << "    " << std::setw(6) << target_index << "   " << std::setw(8)
                      << DECLARED_DELAYS[delay_index] << "    " << std::setw(11) << measured_offset
                      << " ticks    " << (matched ? "yes" : "NO") << "\n";
        }

        std::cout << "\n  " << (every_delay_matched ? "Every" : "Not every")
                  << " spike landed on exactly the tick its connection asked for -- not\n"
                     "  earlier, and not smeared across the ticks in between.\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "delayed_coupling_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
