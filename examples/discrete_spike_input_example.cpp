// ── Driving a model from a literal 0/1 array ────────────────────────────────────────────
//
// The third way to drive a model, after a NeuroML-declared generator (<pulseGenerator>, as
// every other example here uses) and a NeuroML-declared spike train (<spikeArray>): a
// host-provided bit sequence, one value per tick per input neuron, with no generator
// ComponentType behind it at all.
//
//     [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, ...]
//
// Use it when the input comes from outside the model entirely -- recorded data, an upstream
// encoder, a dataset, another simulation. The mechanism is deliberately thin: `input_event_
// streams` is an ordinary public vector of per-tick series, and step_simulation adds each
// one's value at index `tick` into its neuron's accumulator. A caller appending its own
// NeuronInputStream is indistinguishable, to the engine, from one the parser built. This
// model declares no <explicitInput> at all, so the two streams below are the ONLY drive.
//
// The network is two GLIF1 cells with no projections between them, so the bit-to-spike
// mapping is exact and unambiguous: every spike is attributable to one bit.
//
// The run feeds the same pattern to both cells at two different rates, and the fast one loses
// bits. It says why: GLIF1's refractory period is 5ms, so a bit landing inside it finds the
// membrane pinned at vreset and does nothing at all. The fix is to slow the input down or
// shorten tRef -- not to raise the amplitude, which changes nothing during a refractory
// period.

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::examples;

namespace {

constexpr s64 SLOW_TICKS_PER_BIT = 100; // 10ms per bit -- comfortably clear of the 5ms tRef
constexpr s64 FAST_TICKS_PER_BIT = 20;  // 2ms per bit -- well inside it

// Amperes: an input stream's value lands in the same accumulator a synapse delivers its
// current into. Unlike a synapse it is applied for exactly the one tick it is listed against,
// so 25nA over one 0.1ms tick moves a resting 100pF membrane 25mV, past the 20mV it needs to
// reach threshold, and one bit is one spike whenever the cell is free to fire.
constexpr f64 BIT_AMPLITUDE = 25.0 * NANOAMPERE;

const char *const DEFAULT_PATTERN = "1001010001101";

std::string discrete_input_network_nml() {
    std::ostringstream document;
    document << "<neuroml id=\"discreteSpikeInputNetwork\">\n    "
             << glif_cell_instance(GlifVariant::Glif1, "inputCell") << "\n\n"
             << "    <network id=\"discreteSpikeInputNet\">\n"
                "        <population id=\"pop\" component=\"inputCell\" size=\"2\"/>\n"
                "    </network>\n</neuroml>\n";
    return document.str();
}

std::string discrete_input_lems_document(const std::string &network_file_name) {
    std::ostringstream document;
    document << "<Lems>\n    <Target component=\"discreteSpikeInputSimulation\"/>\n"
             << glif1_component_type() << "\n    <Include file=\"" << network_file_name
             << "\"/>\n\n"
                "    <Simulation id=\"discreteSpikeInputSimulation\" length=\"140ms\" "
                "step=\"0.1ms\"\n                target=\"discreteSpikeInputNet\"/>\n</Lems>\n";
    return document.str();
}

// Brackets, commas and whitespace are ignored, so `0001001010` and `[0,0,0,1,0,0,1,0,1,0]`
// read the same.
spikecorec::Vector<s32> parse_bit_pattern(const std::string &text) {
    spikecorec::Vector<s32> bits;
    for (char character : text) {
        if (character == '0' || character == '1') bits.push_back(character - '0');
    }
    if (bits.empty()) throw std::runtime_error("--pattern contained no 0 or 1 digits");
    return bits;
}

// One bit per `ticks_per_bit` ticks: the amplitude on the tick the bit starts, zero on every
// tick until the next one.
spikecorec::Vector<f64> bit_stream_values(const spikecorec::Vector<s32> &bits, s64 ticks_per_bit) {
    spikecorec::Vector<f64> values((usize)(bits.size() * (usize)ticks_per_bit), 0.0);
    for (usize bit_index = 0; bit_index < bits.size(); ++bit_index) {
        if (bits[bit_index] != 0) values[bit_index * (usize)ticks_per_bit] = BIT_AMPLITUDE;
    }
    return values;
}

// `input bits` over `output spikes`, drawn on the same tick axis so a lost bit is visible as a
// gap rather than a number.
void print_aligned_streams(const spikecorec::Vector<s32> &bits, s64 ticks_per_bit,
                           const spikecorec::Vector<s64> &spike_ticks, s64 column_ticks) {
    const s64 column_count = ((s64)bits.size() * ticks_per_bit + column_ticks - 1) / column_ticks;

    std::string bit_row(( usize)column_count, ' ');
    for (usize bit_index = 0; bit_index < bits.size(); ++bit_index) {
        if (bits[bit_index] == 0) continue;
        const s64 column = (s64)bit_index * ticks_per_bit / column_ticks;
        if (column < column_count) bit_row[(usize)column] = '|';
    }

    std::string spike_row((usize)column_count, ' ');
    for (s64 spike_tick : spike_ticks) {
        const s64 column = spike_tick / column_ticks;
        if (column < column_count) spike_row[(usize)column] = '|';
    }

    std::cout << "    input bits    " << bit_row << "\n"
              << "    output spikes " << spike_row << "\n";
}

} // namespace

int main(int argument_count, char **argument_values) {
    try {
        const ExampleOptions options =
                parse_example_options(argument_count, argument_values,
                                      /*default_synapse_peak_current=*/0.0);
        configure_logging(options);

        // The first local owning anything GPU-backed, so it destructs last.
        GpuContextScope gpu_context;

        const spikecorec::Vector<s32> bits = parse_bit_pattern(
                options.input_pattern.empty() ? DEFAULT_PATTERN : options.input_pattern);

        const GeneratedModelDirectory model_directory("discrete_spike_input");
        model_directory.write("discrete_spike_input_network.nml", discrete_input_network_nml());
        String model_path = model_directory.write(
                "discrete_spike_input_model.xml",
                discrete_input_lems_document("discrete_spike_input_network.nml"));

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

        // The model declares no stimulus at all, so this is the whole drive. Appending here
        // is all it takes: step_simulation reads input_event_streams every tick and does not
        // care where an entry came from.
        engine.input_event_streams.push_back(
                NeuronInputStream{/*neuron_index=*/0, bit_stream_values(bits, SLOW_TICKS_PER_BIT)});
        engine.input_event_streams.push_back(
                NeuronInputStream{/*neuron_index=*/1, bit_stream_values(bits, FAST_TICKS_PER_BIT)});

        s64 set_bit_count = 0;
        for (s32 bit : bits) set_bit_count += bit;

        print_heading("A host-provided bit sequence, fed straight into the engine");
        std::cout << "  pattern  ";
        for (s32 bit : bits) std::cout << bit;
        std::cout << "   (" << bits.size() << " bits, " << set_bit_count << " set)\n"
                  << "  " << engine.total_neuron_count
                  << " neurons, no projections between them, no <explicitInput> in the model\n";

        const SpikeObservation observation = run_simulation(engine, options.tick_count);

        std::cout << "\n  slow -- neuron 0, one pattern element every " << SLOW_TICKS_PER_BIT
                  << " ticks (" << (f64)SLOW_TICKS_PER_BIT * engine.network_details.step_dt * 1000.0
                  << "ms)\n";
        print_aligned_streams(bits, SLOW_TICKS_PER_BIT, observation.spike_ticks_per_neuron[0],
                              /*column_ticks=*/SLOW_TICKS_PER_BIT / 2);
        std::cout << "                   " << set_bit_count << " bits in -> "
                  << observation.spike_count_per_neuron[0] << " spikes out\n";

        std::cout << "\n  fast -- neuron 1, one pattern element every " << FAST_TICKS_PER_BIT
                  << " ticks (" << (f64)FAST_TICKS_PER_BIT * engine.network_details.step_dt * 1000.0
                  << "ms)\n";
        print_aligned_streams(bits, FAST_TICKS_PER_BIT, observation.spike_ticks_per_neuron[1],
                              /*column_ticks=*/FAST_TICKS_PER_BIT / 2);
        std::cout << "                   " << set_bit_count << " bits in -> "
                  << observation.spike_count_per_neuron[1] << " spikes out\n";

        // Every set bit that arrives while the cell is still inside its 5ms refractory window
        // is a bit that cannot possibly produce a spike, whatever its amplitude.
        const s64 refractory_ticks = 50; // tRef = 5ms at 0.1ms per tick
        s64 bits_inside_refractory_period = 0;
        s64 previous_set_bit_tick = -refractory_ticks - 1;
        for (usize bit_index = 0; bit_index < bits.size(); ++bit_index) {
            if (bits[bit_index] == 0) continue;
            const s64 bit_tick = (s64)bit_index * FAST_TICKS_PER_BIT;
            if (bit_tick - previous_set_bit_tick <= refractory_ticks) {
                bits_inside_refractory_period += 1;
            } else {
                previous_set_bit_tick = bit_tick;
            }
        }

        std::cout << "\n  " << bits_inside_refractory_period << " of the fast stream's "
                  << set_bit_count << " set bits arrive within tRef (5ms) of the spike\n"
                  << "  before them, so they land on a membrane pinned at vreset and do nothing.\n"
                  << "  Raising the amplitude cannot recover those; slowing the input down can.\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "discrete_spike_input_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
