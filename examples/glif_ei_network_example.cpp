// ── An excitatory / inhibitory network: many populations, one flat index space ───────────
//
// E/I means excitatory/inhibitory -- the standard cortical motif of two neuron populations,
// one whose spikes push their targets towards firing and one that pushes them away. Here
// ExcPop is eight GLIF1 cells and InhPop is two GLIF3 cells, so the model also carries TWO
// cell types, which is what makes the engine's type-sectioned `cell_state` layout observable
// rather than degenerate.
//
// Three things this shows that the single-population torus examples cannot:
//
//   * the flat neuron index space populations share. ExcPop is neurons 0..7 and InhPop is
//     8..9; nothing is renumbered, and `cell_type_index` is what resolves a neuron to the
//     dynamics it runs.
//   * cell_state sectioned BY TYPE rather than by population -- every GLIF1 cell's state
//     first, then every GLIF3 cell's, each neuron occupying one slot per state variable ITS
//     type declares (2 for GLIF1, 4 for GLIF3).
//   * inhibition actually inhibiting. The run builds the same network twice, once with the
//     inhibitory weight zeroed and once with it live, and reports the excitatory
//     population's spike count both times. The difference is the whole point of the motif.
//
// Sign is the only thing that makes a connection inhibitory here: an arriving weight is
// summed into the target's synaptic accumulator, so a negative one drives v away from
// threshold. There is no separate inhibitory synapse mechanism, because synapse
// ComponentTypes are not lowered yet -- see examples/README.md.

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::examples;

namespace {

constexpr s64 EXCITATORY_NEURON_COUNT = 8;
constexpr s64 INHIBITORY_NEURON_COUNT = 2;

// Weights are in nanoamps: the cells read their synaptic accumulator through
// synapticCurrentScale="1nA". 25 is one full recruiting event (see glif_torus_network.h).
constexpr f64 EXCITATORY_WEIGHT = 25.0;

std::string excitatory_inhibitory_network_nml(f64 inhibitory_weight) {
    std::ostringstream document;
    document << R"XML(<neuroml id="excitatoryInhibitoryNetwork">
    )XML" << glif_cell_instance(GlifVariant::Glif1, "excitatoryCell") << R"XML(
    )XML" << glif_cell_instance(GlifVariant::Glif3, "inhibitoryCell") << R"XML(
    <expOneSynapse id="excitatorySynapse" gbase="10nS" erev="0mV" tauDecay="3ms"/>
    <expOneSynapse id="inhibitorySynapse" gbase="10nS" erev="-80mV" tauDecay="8ms"/>
    <pulseGenerator id="excitatoryDrive" delay="20ms" duration="1000ms" amplitude="0.6"/>

    <network id="excitatoryInhibitoryNet">
        <population id="ExcPop" component="excitatoryCell" size=")XML"
             << EXCITATORY_NEURON_COUNT << R"XML("/>
        <population id="InhPop" component="inhibitoryCell" size=")XML"
             << INHIBITORY_NEURON_COUNT << R"XML("/>

)XML";

    // Recurrent excitation: a ring through ExcPop, so activity started at ExcPop[0] travels.
    document << "        <projection id=\"excToExc\" presynapticPopulation=\"ExcPop\"\n"
                "                    postsynapticPopulation=\"ExcPop\" "
                "synapse=\"excitatorySynapse\">\n";
    for (s64 source_index = 0; source_index < EXCITATORY_NEURON_COUNT; ++source_index) {
        const s64 target_index = (source_index + 1) % EXCITATORY_NEURON_COUNT;
        document << "            <connectionWD id=\"" << source_index << "\" preCellId=\"../ExcPop["
                 << source_index << "]\" postCellId=\"../ExcPop[" << target_index
                 << "]\" weight=\"" << EXCITATORY_WEIGHT << "\" delay=\"1ms\"/>\n";
    }
    document << "        </projection>\n\n";

    // Feedforward drive onto the inhibitory population.
    document << "        <projection id=\"excToInh\" presynapticPopulation=\"ExcPop\"\n"
                "                    postsynapticPopulation=\"InhPop\" "
                "synapse=\"excitatorySynapse\">\n";
    for (s64 source_index = 0; source_index < EXCITATORY_NEURON_COUNT; ++source_index) {
        document << "            <connectionWD id=\"" << source_index << "\" preCellId=\"../ExcPop["
                 << source_index << "]\" postCellId=\"../InhPop["
                 << source_index % INHIBITORY_NEURON_COUNT << "]\" weight=\"" << EXCITATORY_WEIGHT
                 << "\" delay=\"1ms\"/>\n";
    }
    document << "        </projection>\n\n";

    // Feedback inhibition onto every excitatory cell. A NEGATIVE weight is the whole
    // mechanism; with inhibitory_weight == 0 these connections still exist and still deliver,
    // they just deliver nothing.
    document << "        <projection id=\"inhToExc\" presynapticPopulation=\"InhPop\"\n"
                "                    postsynapticPopulation=\"ExcPop\" "
                "synapse=\"inhibitorySynapse\">\n";
    s64 connection_id = 0;
    for (s64 source_index = 0; source_index < INHIBITORY_NEURON_COUNT; ++source_index) {
        for (s64 target_index = 0; target_index < EXCITATORY_NEURON_COUNT; ++target_index) {
            document << "            <connectionWD id=\"" << connection_id++
                     << "\" preCellId=\"../InhPop[" << source_index << "]\" postCellId=\"../ExcPop["
                     << target_index << "]\" weight=\"" << -inhibitory_weight
                     << "\" delay=\"1ms\"/>\n";
        }
    }
    document << "        </projection>\n\n";

    document << "        <explicitInput target=\"ExcPop[0]\" input=\"excitatoryDrive\"/>\n"
                "    </network>\n</neuroml>\n";
    return document.str();
}

std::string excitatory_inhibitory_lems_document(const std::string &network_file_name,
                                                const std::string &recording_path) {
    std::ostringstream document;
    document << "<Lems>\n    <Target component=\"excitatoryInhibitorySimulation\"/>\n"
             << glif1_component_type() << glif3_component_type() << "\n    <Include file=\""
             << network_file_name << "\"/>\n\n"
             << "    <Simulation id=\"excitatoryInhibitorySimulation\" length=\"200ms\" "
                "step=\"0.1ms\" target=\"excitatoryInhibitoryNet\">\n"
                "        <OutputFile id=\"membrane\" fileName=\""
             << recording_path << "\"/>\n    </Simulation>\n</Lems>\n";
    return document.str();
}

struct RunOutcome {
    s64 excitatory_spike_count = 0;
    s64 inhibitory_spike_count = 0;
};

// One complete build-run-report cycle. The engine is scoped so it is destroyed -- and its
// recorder closed -- before the next run constructs another one.
RunOutcome run_one_configuration(const GeneratedModelDirectory &model_directory,
                                 const std::string &configuration_name, f64 inhibitory_weight,
                                 const std::string &recording_path, s64 tick_count,
                                 bool print_layout) {
    const std::string network_file_name = configuration_name + "_network.nml";
    model_directory.write(network_file_name, excitatory_inhibitory_network_nml(inhibitory_weight));
    String model_path = model_directory.write(
            configuration_name + "_model.xml",
            excitatory_inhibitory_lems_document(network_file_name, recording_path));

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    if (print_layout) {
        print_heading("Layout: two populations, two cell types, one flat index space");

        const s32 *cell_type_index = engine.cell_type_index.get_contents();
        const s32 *cell_state_base = engine.cell_state_base.get_contents();

        std::cout << "  neuron  population  cell type            state slot\n";
        for (s64 neuron_index = 0; neuron_index < engine.total_neuron_count; ++neuron_index) {
            const nml::CellTypeSpecification &cell_type =
                    engine.network_details.cell_types[(usize)cell_type_index[neuron_index]];
            std::cout << "  " << std::setw(6) << neuron_index << "  " << std::setw(10)
                      << (neuron_index < EXCITATORY_NEURON_COUNT ? "ExcPop" : "InhPop") << "  "
                      << std::setw(19) << std::left << cell_type.name << std::right
                      << "  " << std::setw(9) << cell_state_base[neuron_index] << " ("
                      << cell_type.state_variable_names.size() << " wide)\n";
        }
        std::cout << "\n  The state slots run 0,2,4,... across the GLIF1 section and then jump to\n"
                     "  4-wide slots for the GLIF3 section: cell_state is sectioned by TYPE, not\n"
                     "  by population, and neuron indices are never renumbered.\n";
    }

    const SpikeObservation observation = run_simulation(engine, tick_count);

    RunOutcome outcome;
    for (s64 neuron_index = 0; neuron_index < engine.total_neuron_count; ++neuron_index) {
        if (neuron_index < EXCITATORY_NEURON_COUNT) {
            outcome.excitatory_spike_count += observation.spike_count_per_neuron[(usize)neuron_index];
        } else {
            outcome.inhibitory_spike_count += observation.spike_count_per_neuron[(usize)neuron_index];
        }
    }

    engine.shutdown();
    return outcome;
}

} // namespace

int main(int argument_count, char **argument_values) {
    try {
        const ExampleOptions options =
                parse_example_options(argument_count, argument_values,
                                      /*default_connection_weight=*/60.0);
        configure_logging(options);

        // The first local owning anything GPU-backed, so it destructs last.
        GpuContextScope gpu_context;

        std::filesystem::create_directories(options.recording_directory);
        const GeneratedModelDirectory model_directory("glif_ei_network");

        const RunOutcome without_inhibition = run_one_configuration(
                model_directory, "without_inhibition", /*inhibitory_weight=*/0.0,
                options.recording_directory + "/glif_ei_without_inhibition.spire",
                options.tick_count, /*print_layout=*/true);

        const RunOutcome with_inhibition = run_one_configuration(
                model_directory, "with_inhibition", options.connection_weight,
                options.recording_directory + "/glif_ei_with_inhibition.spire",
                options.tick_count, /*print_layout=*/false);

        print_heading("Does the inhibitory population actually inhibit?");
        std::cout << "  Same network, same drive, same 200ms. The only difference is the weight\n"
                     "  on the InhPop -> ExcPop projection.\n\n"
                  << "                          ExcPop spikes   InhPop spikes\n"
                  << "    inhibitory weight " << std::setw(6) << 0.0 << std::setw(14)
                  << without_inhibition.excitatory_spike_count << std::setw(16)
                  << without_inhibition.inhibitory_spike_count << "\n"
                  << "    inhibitory weight " << std::setw(6) << -options.connection_weight
                  << std::setw(14) << with_inhibition.excitatory_spike_count << std::setw(16)
                  << with_inhibition.inhibitory_spike_count << "\n\n";

        if (with_inhibition.excitatory_spike_count < without_inhibition.excitatory_spike_count) {
            std::cout << "  The excitatory population fired "
                      << without_inhibition.excitatory_spike_count -
                                 with_inhibition.excitatory_spike_count
                      << " fewer times with feedback inhibition\n"
                         "  live. Every one of those suppressed spikes travelled ExcPop -> InhPop\n"
                         "  -> ExcPop, so it is genuine network-scale propagation and not the\n"
                         "  directly-driven cell talking to itself.\n";
        } else {
            std::cout << "  Inhibition did not reduce excitatory firing at this weight. Raise it\n"
                         "  with --weight.\n";
        }
    } catch (const std::exception &error) {
        std::cerr << "glif_ei_network_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
