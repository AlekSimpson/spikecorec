#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/nml/nml.h"

using namespace std;
using namespace spikecorec;

// ── breadth sweep of the vendored NeuroML standard library ───────────────────────────────────
//
// Every other simulation test in this tree is DEPTH: it takes one model we already believe in
// and checks it hard (exit_model_validation_tests.cpp checks four GLIF cells tick-for-tick
// against jNeuroML). This file is the BREADTH counterpart, and it answers a question nothing
// else asks: of the point cells and current-based synapses the vendored standard library
// actually defines, how many can this engine construct, compile, step and read back?
//
// It exists because the alternative is anecdote. Before this file the answer was discovered one
// model at a time, whenever somebody happened to write a fixture and it happened to break.
//
// ── HOW THE CANDIDATES WERE ENUMERATED ──────────────────────────────────────────────────────
// Not from a remembered list. STANDARD_LIBRARY_CLASSIFICATION below names EVERY ComponentType
// declared by the three standard-library files that can define a point cell or a synapse --
// third_party/neuroml2/std_lib/{Cells,Synapses,PyNN}.xml, 68 + 24 + 16 = 108 of them -- and
// gives each one a role. `every_declared_component_type_is_classified` re-scans those three
// files at run time and fails if the two sets differ in either direction, so a ComponentType
// added to (or renamed in) the vendored library cannot slip past this sweep unclassified.
//
// The three files are the whole surface. The rest of the vendored library declares no point
// cell and no synapse placeable on a projection: Channels.xml is gates, rates and ion channels
// (Phase 3), Networks.xml is structure (population, projection, connection, inputList),
// Simulation.xml is the runner, NeuroMLCoreCompTypes.xml is metadata, and Inputs.xml is
// stimulus. Inputs.xml is the one worth being explicit about, because three of its types are
// NAMED like synapses -- `poissonFiringSynapse`, `transientPoissonFiringSynapse` and
// `timedSynapticInput`. All three extend `baseVoltageDepPointCurrentSpiking`: they are spike
// SOURCES with a synapse bundled inside, driven by their own rate or spike list, not synapses a
// <projection> can name. Its generators drive the models below but are not themselves swept.
//
// ── WHAT IS EXCLUDED, AND WHY ───────────────────────────────────────────────────────────────
//   * abstract bases (baseCell, baseSynapse, basePyNNCell, ...) -- no Dynamics of their own,
//     nothing to instantiate.
//   * CONDUCTANCE-BASED SYNAPSES (expOneSynapse, alphaSynapse, expTwoSynapse, expThreeSynapse,
//     blockingPlasticSynapse, doubleSynapse, stdpSynapse, expCondSynapse, alphaCondSynapse) --
//     excluded by owner ruling. The engine refuses them by name today; that refusal is spot
//     checked once, in `a_conductance_based_synapse_is_refused_by_name`, so "excluded" cannot
//     quietly become "silently mis-simulated".
//   * CONTINUOUSLY TRANSMITTING synapses (gapJunction, silentSynapse, linearGradedSynapse,
//     gradedSynapse) -- these read the PEER's voltage every tick rather than responding to an
//     event, which is a different coupling mechanism from the event-driven scatter the engine
//     implements at all.
//   * synapse SUB-MECHANISMS (tsodyksMarkram*, voltageConcDepBlockMechanism, ...) -- children of
//     a synapse, not synapses.
//   * BIOPHYSICAL / MULTICOMPARTMENT (Phase 3): channel densities and populations, concentration
//     models, morphology and segment structure, and the composed cells built out of them (`cell`,
//     `cell2CaPools`, `pointCellCondBased`, `pointCellCondBasedCa`).
//
// Two judgement calls are recorded here rather than buried:
//   * `IF_cond_alpha` / `IF_cond_exp` / `EIF_cond_*` are INCLUDED as point cells even though
//     "cond" is in the name. The exclusion above is of conductance-based SYNAPSES. Those cells'
//     own LEMS dynamics sum `synapses[*]/i` exactly like the `IF_curr_*` pair -- their `e_rev_E`
//     / `e_rev_I` parameters carry the standard library's own comment "This parameter is never
//     used in the NeuroML2 description of this cell". The name describes the synapse you are
//     expected to attach, not the cell.
//   * `HH_cond_exp` and `pinskyRinzelCA3Cell` are INCLUDED. Both write Hodgkin-Huxley style
//     kinetics, but both are single self-contained point ComponentTypes with no channel,
//     morphology or concentration children -- nothing structurally Phase 3. The Phase-3
//     exclusion above is of COMPOSED cells, not of stiff equations.
//
// ── THE VERDICT VOCABULARY ──────────────────────────────────────────────────────────────────
//   runs     -- constructs, compiles, steps, and every state variable stays finite while the
//               membrane variable moves by at least the stated amount (and emits at least the
//               stated number of spikes, for a type that has a spike port at all).
//   inert    -- constructs, compiles and steps, finite throughout, but the state provably cannot
//               move: the ComponentType declares no path by which any current can reach it.
//               This is a property of the MODEL, not a failure of the engine, and it is a
//               separate verdict precisely so it cannot be confused with one.
//   refused  -- construction throws. The expected message substring is asserted, so a refusal
//               silently changing into a different refusal fails here.
//
// There is deliberately no "broken" row: the sweep found no candidate that constructs and then
// produces a wrong-looking number, and encoding one as an expectation would freeze a defect into
// the suite. A candidate that starts producing non-finite or frozen state fails its own case.
//
// ── WHAT THE SWEEP FOUND (and what is therefore asserted below) ──────────────────────────────
// 1. `iafRefCell` and `iafTauRefCell` -- two of the four core NeuroML integrate-and-fire point
//    cells -- CANNOT BE CONSTRUCTED. Both are refused with "'v' carries both a regime-scoped
//    TimeDerivative and one outside any Regime". Neither declares a regime-free TimeDerivative:
//    it arrives from the parent they extend (`iafCell` / `iafTauCell`) through extends
//    flattening, and the child re-expresses the same derivative inside its `integrating` Regime.
//    `an_iaf_refractory_cell_rebased_off_its_parent_compiles` is the control: iafRefCell's
//    Dynamics body VERBATIM, rebased onto `baseIafCapCell` so nothing regime-free is inherited,
//    compiles and fires. So the refusal is the flattening, not the dynamics. This is a GAP, and
//    the expectation below records the refusal as today's honest state, not as correct.
// 2. `HH_cond_exp` diverges to NaN at the 0.1ms step every other model here uses, and integrates
//    clean action potentials at 0.001ms. Forward Euler on Traub gate kinetics is unstable at
//    0.1ms (the gate time constants fall below 0.1ms during the upstroke), so this is a
//    step-size requirement rather than an engine defect -- but it is a real limit on what the
//    engine can be handed, and Phase 3 will need an exponential-Euler gate update to lift it.
//    The candidate below therefore runs at 0.001ms. The 0.1ms divergence is NOT asserted: it
//    would fail the day somebody fixes it, which is the wrong way round.
// 3. `pinskyRinzelCA3Cell`, `EIF_cond_exp_isfa_ista` and `EIF_cond_alpha_isfa_ista` are all
//    refused for the same reason -- `<ConditionalDerivedVariable>` never reaches codegen with
//    its guards. For pinskyRinzel that is a correct Phase-3 refusal. For the two EIF cells it
//    costs a PHASE-2 point cell: their dynamics are AdEx, and `adExIaFCell` -- the same model
//    without the PyNN wrapper's `delta_T .gt. 0` guard -- runs here.
// 4. All three current-based synapses deliver EXACTLY the analytic charge of their own impulse
//    response, scaled by the per-edge weight. That is asserted numerically per synapse
//    (`expected_delivered_current_per_event`), because it is the one check that discriminates a
//    unit scale, a sign and a weight all at once. The agreement measured is nine significant
//    figures, so the check has room to discriminate rather than being met by luck.
// 5. A `pulseGenerator` declaring `amplitude="0nA"` injects ONE AMPERE. That is a defect in
//    src/core/engine.cpp, not in any ComponentType, and this file does not own that file -- it
//    is recorded here, where the sweep found it, as the DISABLED reproduction
//    `DISABLED_a_zero_amplitude_current_injector_delivers_no_current`, which carries the full
//    diagnosis. It is disabled rather than deleted, and asserts what is CORRECT rather than
//    what happens, so it starts passing when the defect is fixed instead of freezing it in.
//
// ── COST ─────────────────────────────────────────────────────────────────────────────────────
// Single cells, a few thousand ticks each; the whole file is roughly 20 seconds, of which
// `HH_cond_exp` (8000 ticks at 0.001ms) and `hindmarshRose1984Cell` (10000 ticks) are half.
//
// ── PARAMETER VALUES ─────────────────────────────────────────────────────────────────────────
// None of these ComponentTypes declares a `defaultValue` for any Parameter (only the synapses'
// dimensionless `weight` Property has one, and it is set explicitly wherever it matters), so
// every value below is chosen. Each candidate's `note` says where its numbers come from: a
// published parameterisation where one exists, otherwise a physiologically sane set, with the
// drive picked to put the cell somewhere informative rather than to flatter it.

namespace {

// ── enumeration of the vendored library ─────────────────────────────────────────────────────

String standard_library_file_path(const String &file_name) {
    return String(SPIKECOREC_NML_STD_LIB_DIR) + "/" + file_name;
}

// Every `<ComponentType ... name="X" ...>` in one standard-library file, in declaration order.
// A deliberately literal scan rather than an XML parse: this has to be able to disagree with
// the front-end about what the library contains, which it cannot do if it asks the front-end.
spikecorec::Vector<String> enumerate_component_type_names(const String &file_name) {
    std::ifstream file(standard_library_file_path(file_name));
    if (!file.is_open()) {
        throw std::runtime_error("standard_library_coverage_tests: could not open vendored "
                                 "standard-library file '" + file_name + "'");
    }
    const String file_text((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    spikecorec::Vector<String> component_type_names;
    usize search_position = 0;
    while (true) {
        const usize element_position = file_text.find("<ComponentType", search_position);
        if (element_position == String::npos) break;

        const usize element_end = file_text.find('>', element_position);
        if (element_end == String::npos) break;

        const usize name_position = file_text.find("name=\"", element_position);
        if (name_position != String::npos && name_position < element_end) {
            const usize value_start = name_position + 6;
            const usize value_end = file_text.find('"', value_start);
            component_type_names.push_back(file_text.substr(value_start, value_end - value_start));
        }
        search_position = element_end + 1;
    }
    return component_type_names;
}

enum class LibraryRole {
    POINT_CELL_CANDIDATE,
    CURRENT_BASED_SYNAPSE_CANDIDATE,
    EXCLUDED_ABSTRACT_BASE,
    EXCLUDED_CONDUCTANCE_BASED_SYNAPSE,
    EXCLUDED_CONTINUOUSLY_TRANSMITTING_SYNAPSE,
    EXCLUDED_SYNAPSE_SUBMECHANISM,
    EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT,
    EXCLUDED_NOT_A_CELL_OR_SYNAPSE,
};

struct LibraryClassification {
    const char *component_type_name;
    LibraryRole role;
};

// The whole declared surface of Cells.xml, Synapses.xml and PyNN.xml, in declaration order.
const LibraryClassification STANDARD_LIBRARY_CLASSIFICATION[] = {
        // ── Cells.xml ───────────────────────────────────────────────────────────────────────
        {"baseCell", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseSpikingCell", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseCellMembPot", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseCellMembPotDL", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseChannelPopulation", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelPopulation", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelPopulationNernst", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"baseChannelDensity", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"baseChannelDensityCond", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"variableParameter", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"inhomogeneousValue", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensityNonUniform", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensityNonUniformNernst", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensityNonUniformGHK", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensity", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensityVShift", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensityNernst", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensityNernstCa2", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensityGHK", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"channelDensityGHK2", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"pointCellCondBased", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"pointCellCondBasedCa", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"distal", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"proximal", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"parent", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"segment", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"segmentGroup", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"member", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"from", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"to", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"include", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"path", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"subTree", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"inhomogeneousParameter", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"proximalDetails", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"distalDetails", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"morphology", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"specificCapacitance", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"initMembPotential", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"spikeThresh", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"membraneProperties", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"membraneProperties2CaPools", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"biophysicalProperties", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"biophysicalProperties2CaPools", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"intracellularProperties", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"intracellularProperties2CaPools", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"resistivity", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"concentrationModel", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"decayingPoolConcentrationModel", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"fixedFactorConcentrationModel", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"fixedFactorConcentrationModelTraub", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"species", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"cell", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"cell2CaPools", LibraryRole::EXCLUDED_BIOPHYSICAL_OR_MULTICOMPARTMENT},
        {"baseCellMembPotCap", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseIaf", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"iafTauCell", LibraryRole::POINT_CELL_CANDIDATE},
        {"iafTauRefCell", LibraryRole::POINT_CELL_CANDIDATE},
        {"baseIafCapCell", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"iafCell", LibraryRole::POINT_CELL_CANDIDATE},
        {"iafRefCell", LibraryRole::POINT_CELL_CANDIDATE},
        {"izhikevichCell", LibraryRole::POINT_CELL_CANDIDATE},
        {"izhikevich2007Cell", LibraryRole::POINT_CELL_CANDIDATE},
        {"adExIaFCell", LibraryRole::POINT_CELL_CANDIDATE},
        {"fitzHughNagumo1969Cell", LibraryRole::POINT_CELL_CANDIDATE},
        {"fitzHughNagumoCell", LibraryRole::POINT_CELL_CANDIDATE},
        {"pinskyRinzelCA3Cell", LibraryRole::POINT_CELL_CANDIDATE},
        {"hindmarshRose1984Cell", LibraryRole::POINT_CELL_CANDIDATE},

        // ── Synapses.xml ────────────────────────────────────────────────────────────────────
        {"baseSynapse", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseVoltageDepSynapse", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseSynapseDL", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseCurrentBasedSynapse", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"alphaCurrentSynapse", LibraryRole::CURRENT_BASED_SYNAPSE_CANDIDATE},
        {"baseConductanceBasedSynapse", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"baseConductanceBasedSynapseTwo", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"expOneSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"alphaSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"expTwoSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"expThreeSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"baseBlockMechanism", LibraryRole::EXCLUDED_SYNAPSE_SUBMECHANISM},
        {"voltageConcDepBlockMechanism", LibraryRole::EXCLUDED_SYNAPSE_SUBMECHANISM},
        {"basePlasticityMechanism", LibraryRole::EXCLUDED_SYNAPSE_SUBMECHANISM},
        {"tsodyksMarkramDepMechanism", LibraryRole::EXCLUDED_SYNAPSE_SUBMECHANISM},
        {"tsodyksMarkramDepFacMechanism", LibraryRole::EXCLUDED_SYNAPSE_SUBMECHANISM},
        {"blockingPlasticSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"doubleSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"stdpSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"gapJunction", LibraryRole::EXCLUDED_CONTINUOUSLY_TRANSMITTING_SYNAPSE},
        {"baseGradedSynapse", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"silentSynapse", LibraryRole::EXCLUDED_CONTINUOUSLY_TRANSMITTING_SYNAPSE},
        {"linearGradedSynapse", LibraryRole::EXCLUDED_CONTINUOUSLY_TRANSMITTING_SYNAPSE},
        {"gradedSynapse", LibraryRole::EXCLUDED_CONTINUOUSLY_TRANSMITTING_SYNAPSE},

        // ── PyNN.xml ────────────────────────────────────────────────────────────────────────
        {"basePyNNCell", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"basePyNNIaFCell", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"basePyNNIaFCondCell", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"IF_curr_alpha", LibraryRole::POINT_CELL_CANDIDATE},
        {"IF_curr_exp", LibraryRole::POINT_CELL_CANDIDATE},
        {"IF_cond_alpha", LibraryRole::POINT_CELL_CANDIDATE},
        {"IF_cond_exp", LibraryRole::POINT_CELL_CANDIDATE},
        {"EIF_cond_exp_isfa_ista", LibraryRole::POINT_CELL_CANDIDATE},
        {"EIF_cond_alpha_isfa_ista", LibraryRole::POINT_CELL_CANDIDATE},
        {"HH_cond_exp", LibraryRole::POINT_CELL_CANDIDATE},
        {"basePynnSynapse", LibraryRole::EXCLUDED_ABSTRACT_BASE},
        {"expCondSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"expCurrSynapse", LibraryRole::CURRENT_BASED_SYNAPSE_CANDIDATE},
        {"alphaCondSynapse", LibraryRole::EXCLUDED_CONDUCTANCE_BASED_SYNAPSE},
        {"alphaCurrSynapse", LibraryRole::CURRENT_BASED_SYNAPSE_CANDIDATE},
        {"SpikeSourcePoisson", LibraryRole::EXCLUDED_NOT_A_CELL_OR_SYNAPSE},
};

// ── driving one candidate ───────────────────────────────────────────────────────────────────

enum class ExpectedVerdict {
    RUNS,
    INERT,
    REFUSED,
};

struct CandidateModel {
    // The ComponentType under test. Doubles as the gtest instance name, so a failure names the
    // type directly.
    const char *component_type_name;

    // The children of the generated <neuroml> document: the cell (and, for a synapse candidate,
    // the two cells and the projection), the stimulus, the <network> and the <Simulation>.
    const char *model_body;

    // Which neuron the verdict is read from. 0 for a single cell; 1 for a synapse candidate,
    // where neuron 0 is the driven presynaptic cell and neuron 1 the cell under observation.
    s64 observed_neuron_index;

    ExpectedVerdict expected_verdict;

    // Least number of ticks the observed neuron must raise its spike flag on. Zero and below
    // are NOT "no requirement" -- both are checked against EXACTLY zero, because "at least
    // zero" asserts nothing. The two negative/zero cases differ only in why:
    //   -1  the ComponentType declares no EventOut at all, so any spike is invented.
    //    0  the candidate is deliberately driven subthreshold (the synapse candidates), so a
    //       spike would mean the drive is not subthreshold and the membrane excursion measured
    //       alongside it is an artifact of the post-reset sample rather than a real response.
    s64 minimum_expected_spike_count;

    // Least excursion, over the whole run, of the observed neuron's FIRST state variable -- the
    // membrane potential for every candidate here (`v`, `V` or `Vs`), in that variable's own SI
    // units. Zero for an INERT candidate, which must not move at all.
    f64 minimum_expected_membrane_excursion;

    // For a synapse candidate: the current one presynaptic event must scatter into the target's
    // ring slot, in amps. That is the synapse's whole impulse-response charge divided by dt (see
    // engine.h on network_inputs), so it is fixed by tau, the baseline amplitude and the
    // per-edge weight, and checking it pins the unit scale, the sign and the weight at once.
    // Zero when the candidate is not a synapse.
    f64 expected_delivered_current_per_event;

    // Substring the refusal message must contain. Empty unless the verdict is REFUSED.
    const char *expected_refusal_substring;

    // Where the parameter values came from, and anything the numbers above depend on.
    const char *note;
};

// What one candidate's run produced, for the assertions and for the printed inventory.
struct CandidateOutcome {
    bool constructed = false;
    String failure_message;
    s64 tick_count = 0;
    s64 observed_spike_count = 0;
    s64 driving_spike_count = 0;
    bool every_sample_finite = true;
    f64 membrane_minimum = 0.0;
    f64 membrane_maximum = 0.0;
    f64 total_delivered_current = 0.0;
    f64 wall_clock_seconds = 0.0;
};

// The generated .nml pair for one candidate. Written under a per-process directory rather than
// into the working tree, and never read back: no candidate declares an <OutputFile>, so the
// engine writes no recording and nothing here depends on the process's working directory.
filesystem::path candidate_model_directory(const String &component_type_name) {
    static const filesystem::path root = [] {
        const filesystem::path path =
                filesystem::temp_directory_path() / "spikecorec_standard_library_coverage";
        std::error_code ignored;
        filesystem::remove_all(path, ignored);
        filesystem::create_directories(path);
        return path;
    }();

    const filesystem::path directory = root / component_type_name;
    filesystem::create_directories(directory);
    return directory;
}

// The wrapper is <include>-only for the same reason every other fixture in this tree uses one:
// NML_Parser XSD-validates only the top-level file it is handed, and a bare <ComponentType>
// declaration (which one candidate below needs) is raw LEMS rather than schema-valid NeuroML.
String write_candidate_model(const CandidateModel &candidate) {
    const filesystem::path directory = candidate_model_directory(candidate.component_type_name);

    std::ofstream model_file(directory / "model.nml");
    model_file << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" "
               << "id=\"StandardLibraryCoverageModel\">\n"
               << candidate.model_body << "</neuroml>\n";
    model_file.close();

    std::ofstream wrapper_file(directory / "top.nml");
    wrapper_file << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" "
                 << "id=\"StandardLibraryCoverageTop\">\n"
                 << "  <include href=\"model.nml\"/>\n"
                 << "</neuroml>\n";
    wrapper_file.close();

    return (directory / "top.nml").string();
}

CandidateOutcome drive_candidate(const CandidateModel &candidate) {
    CandidateOutcome outcome;
    String engine_input_path = write_candidate_model(candidate); // SpikeEngine takes String&

    const auto started_at = std::chrono::steady_clock::now();
    try {
        SpikeEngine engine(engine_input_path, /*enable_hebbian_learning=*/false);
        outcome.constructed = true;
        outcome.tick_count = engine.lifetime;

        const s32 *state_base_values = engine.cell_state_base.get_contents();
        const s64 membrane_state_index = state_base_values[candidate.observed_neuron_index];

        // How many cell_state slots the whole model occupies, so the finiteness sweep can cover
        // every state variable of every neuron rather than only the one being judged.
        const s64 state_element_count = engine.cell_state_element_count;

        outcome.membrane_minimum = engine.cell_state.get_contents()[membrane_state_index];
        outcome.membrane_maximum = outcome.membrane_minimum;

        // Where a delivery scheduled during tick T lands: every edge in these models carries the
        // minimum one-tick delay, so the propagation run inside tick T writes row (T + 1). The
        // row is empty before that write (nothing else targets this neuron) and is cleared again
        // once tick T+1 has read it, so summing exactly this slot counts each delivery once,
        // whatever depth the ring happens to have.
        const s64 ring_depth = engine.network_input_ring_depth;
        const s64 neuron_count = engine.total_neuron_count;

        for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
            engine.step_simulation(tick);

            const s32 *spike_flag_values = engine.spike_flags.get_contents();
            if (spike_flag_values[candidate.observed_neuron_index] != 0) {
                outcome.observed_spike_count += 1;
            }
            if (candidate.observed_neuron_index != 0 && spike_flag_values[0] != 0) {
                outcome.driving_spike_count += 1;
            }

            const f32 *state_values = engine.cell_state.get_contents();
            for (s64 state_index = 0; state_index < state_element_count; state_index += 1) {
                if (!std::isfinite(state_values[state_index])) outcome.every_sample_finite = false;
            }

            const f64 membrane_value = state_values[membrane_state_index];
            if (membrane_value < outcome.membrane_minimum) outcome.membrane_minimum = membrane_value;
            if (membrane_value > outcome.membrane_maximum) outcome.membrane_maximum = membrane_value;

            if (candidate.expected_delivered_current_per_event != 0.0) {
                const f32 *ring_values = engine.network_inputs.get_contents();
                const s64 arrival_row = (tick + 1) % ring_depth;
                outcome.total_delivered_current +=
                        ring_values[arrival_row * neuron_count + candidate.observed_neuron_index];
            }
        }

        engine.shutdown();
    } catch (const std::exception &error) {
        outcome.failure_message = error.what();
    }
    outcome.wall_clock_seconds =
            std::chrono::duration<f64>(std::chrono::steady_clock::now() - started_at).count();

    return outcome;
}

// ── the inventory, printed once at the end of the process ───────────────────────────────────

struct InventoryRow {
    String component_type_name;
    String verdict;
    String evidence;
};

spikecorec::Vector<InventoryRow> &recorded_inventory() {
    static spikecorec::Vector<InventoryRow> rows;
    return rows;
}

void record_inventory_row(const CandidateModel &candidate, const CandidateOutcome &outcome) {
    InventoryRow row;
    row.component_type_name = candidate.component_type_name;

    char evidence[512];
    if (!outcome.constructed) {
        row.verdict = "refused";
        const String &message = outcome.failure_message;
        row.evidence = message.size() > 96 ? message.substr(0, 96) + "..." : message;
    } else {
        row.verdict = candidate.expected_verdict == ExpectedVerdict::INERT ? "inert" : "runs";
        std::snprintf(evidence, sizeof(evidence),
                      "%lld ticks, %lld spikes, membrane [%+.5g, %+.5g], %s, %.2fs",
                      (long long)outcome.tick_count, (long long)outcome.observed_spike_count,
                      outcome.membrane_minimum, outcome.membrane_maximum,
                      outcome.every_sample_finite ? "finite" : "NON-FINITE",
                      outcome.wall_clock_seconds);
        row.evidence = evidence;
    }
    recorded_inventory().push_back(row);
}

class InventoryPrinter : public ::testing::Environment {
public:
    void TearDown() override {
        if (recorded_inventory().empty()) return;
        std::printf("\n── NeuroML standard-library coverage inventory ──────────────────────────"
                    "──────\n");
        for (const InventoryRow &row : recorded_inventory()) {
            std::printf("  %-26s %-8s %s\n", row.component_type_name.c_str(), row.verdict.c_str(),
                        row.evidence.c_str());
        }
        std::printf("────────────────────────────────────────────────────────────────────────"
                    "──────\n\n");
    }
};

[[maybe_unused]] const ::testing::Environment *const INVENTORY_PRINTER =
        ::testing::AddGlobalTestEnvironment(new InventoryPrinter);

// ── the candidates ──────────────────────────────────────────────────────────────────────────
//
// Every single-cell candidate uses the same skeleton: one instance, a one-neuron population, a
// pulseGenerator wired through <explicitInput> where the ComponentType can receive current at
// all, and a <Simulation> with no <OutputFile> (the verdict is read straight out of the engine's
// buffers, so nothing here writes or reads a recording).

const CandidateModel CANDIDATE_MODELS[] = {
        {
                "iafCell",
                "  <iafCell id=\"coverageCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
                "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
                "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
                "                  amplitude=\"0.5nA\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "    <explicitInput target=\"CoveragePop[0]\" input=\"coverageDrive\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 20, 0.015, 0.0, "",
                "Textbook cortical LIF (C=100pF, gL=10nS, tau=10ms). 0.5nA gives a 50mV "
                "steady-state deflection, comfortably over the 20mV to threshold, so the cell "
                "fires throughout the pulse.",
        },
        {
                "iafRefCell",
                "  <iafRefCell id=\"coverageCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
                "              leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"\n"
                "              refract=\"5ms\"/>\n"
                "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
                "                  amplitude=\"0.5nA\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "    <explicitInput target=\"CoveragePop[0]\" input=\"coverageDrive\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::REFUSED, 0, 0.0, 0.0,
                "carries both a regime-scoped TimeDerivative and one outside any Regime",
                "GAP, not a correct refusal. iafRefCell declares no regime-free TimeDerivative "
                "of its own -- it inherits one from iafCell and re-expresses it inside its "
                "integrating Regime. See the control test in this file.",
        },
        {
                "iafTauCell",
                "  <iafTauCell id=\"coverageCell\" leakReversal=\"-70mV\" tau=\"10ms\"\n"
                "              thresh=\"-50mV\" reset=\"-70mV\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::INERT, 0, 0.0, 0.0, "",
                "INERT BY CONSTRUCTION. iafTauCell's whole Dynamics is dv/dt = (leakReversal - "
                "v)/tau: no <Attachments>, no iSyn, no term any external current could enter "
                "through. OnStart puts v at leakReversal, which is the fixed point, so v is "
                "provably constant and the cell can never fire. Driving it with a pulseGenerator "
                "changes nothing (measured). No stimulus is declared here for that reason.",
        },
        {
                "iafTauRefCell",
                "  <iafTauRefCell id=\"coverageCell\" leakReversal=\"-70mV\" tau=\"10ms\"\n"
                "                 thresh=\"-50mV\" reset=\"-70mV\" refract=\"5ms\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::REFUSED, 0, 0.0, 0.0,
                "carries both a regime-scoped TimeDerivative and one outside any Regime",
                "Same GAP as iafRefCell, inherited from iafTauCell instead. Note this type is "
                "unreachable for two independent reasons: even once the flattening is fixed it "
                "is inert, exactly like its parent.",
        },
        {
                "izhikevichCell",
                "  <izhikevichCell id=\"coverageCell\" v0=\"-70mV\" thresh=\"30mV\" a=\"0.02\"\n"
                "                  b=\"0.2\" c=\"-65\" d=\"6\"/>\n"
                "  <pulseGeneratorDL id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
                "                    amplitude=\"10\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "    <explicitInput target=\"CoveragePop[0]\" input=\"coverageDrive\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 5, 0.08, 0.0, "",
                "Izhikevich 2003 Regular Spiking (a=0.02, b=0.2, c=-65, d=6). The drive is a "
                "pulseGeneratorDL, not a pulseGenerator: this type's <Attachments> is "
                "basePointCurrentDL and its ISyn selects synapses[*]/I, both dimensionless. "
                "I=10 is the usual RS demonstration current.",
        },
        {
                "izhikevich2007Cell",
                "  <izhikevich2007Cell id=\"coverageCell\" v0=\"-60mV\" C=\"100pF\"\n"
                "                      k=\"0.7nS_per_mV\" vr=\"-60mV\" vt=\"-40mV\"\n"
                "                      vpeak=\"35mV\" a=\"0.03per_ms\" b=\"-2nS\" c=\"-50mV\"\n"
                "                      d=\"100pA\"/>\n"
                "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
                "                  amplitude=\"150pA\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "    <explicitInput target=\"CoveragePop[0]\" input=\"coverageDrive\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 5, 0.08, 0.0, "",
                "Izhikevich 2007 Regular Spiking, the same parameter set and drive as "
                "tests/fixtures/nml/izhikevich_network.nml, which is validated against jNeuroML.",
        },
        {
                "adExIaFCell",
                "  <adExIaFCell id=\"coverageCell\" C=\"281pF\" gL=\"30nS\" EL=\"-70.6mV\"\n"
                "               reset=\"-70.6mV\" VT=\"-50.4mV\" thresh=\"-40.4mV\" delT=\"2mV\"\n"
                "               tauw=\"144ms\" refract=\"2ms\" a=\"4nS\" b=\"0.0805nA\"/>\n"
                "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
                "                  amplitude=\"0.8nA\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "    <explicitInput target=\"CoveragePop[0]\" input=\"coverageDrive\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 3, 0.025, 0.0, "",
                "Brette & Gerstner (2005) regular-spiking-with-adaptation parameter set. "
                "thresh (spike detection) is held 10mV above VT (the exponential's foot) so the "
                "exponential term stays bounded at the detection point.",
        },
        {
                "fitzHughNagumo1969Cell",
                "  <fitzHughNagumo1969Cell id=\"coverageCell\" a=\"0.7\" b=\"0.8\" I=\"0.8\"\n"
                "                          phi=\"0.08\" V0=\"-1.0\" W0=\"0\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, -1, 3.0, 0.0, "",
                "FitzHugh's own 1969 parameters (a=0.7, b=0.8, phi=0.08), I=0.8 which is inside "
                "the oscillatory band, so the cell limit-cycles with no stimulus at all -- which "
                "is just as well, since it declares no synapse attachment. Its Constant TS is "
                "1ms, so one cycle fits inside 350ms. It declares no EventOut, so the correct "
                "spike count is exactly zero; V swinging roughly +-1.9 is the evidence.",
        },
        {
                "fitzHughNagumoCell",
                "  <fitzHughNagumoCell id=\"coverageCell\" I=\"0.8\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"60s\" step=\"10ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, -1, 3.0, 0.0, "",
                "The superseded FitzHugh-Nagumo, whose only Parameter is I. Note the window: "
                "this type's Constant is SEC=1s where fitzHughNagumo1969Cell's is TS=1ms, so its "
                "limit cycle takes about 40 SECONDS. At the 350ms every other cell here uses it "
                "moves by 0.3 and looks stalled; 60s at a 10ms step covers a full cycle in 6000 "
                "ticks. It declares no OnStart (V and W start at zero, which is not a fixed "
                "point) and no EventOut.",
        },
        {
                "hindmarshRose1984Cell",
                "  <hindmarshRose1984Cell id=\"coverageCell\" a=\"1\" b=\"3\" c=\"1\" d=\"5\"\n"
                "                         s=\"4\" x1=\"-1.6\" r=\"0.002\" x0=\"-0.5\" y0=\"0\"\n"
                "                         z0=\"0\" v_scaling=\"35mV\" C=\"28pF\"/>\n"
                "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"1000ms\"\n"
                "                  amplitude=\"3.2nA\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "    <explicitInput target=\"CoveragePop[0]\" input=\"coverageDrive\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"1000ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 20, 0.1, 0.0, "",
                "Hindmarsh & Rose's own bursting parameters (a=1, b=3, c=1, d=5, s=4, x1=-1.6, "
                "r=0.002). The drive matters and is not arbitrary: iSyn enters as a "
                "dimensionless I = iSyn*MSEC/(C*v_scaling), so 3.2nA against C=28pF and "
                "v_scaling=35mV is I~3.2, which is the bursting regime. At 0.1nA (I~0.1) the "
                "system sits at a stable fixed point and fires once -- which reads exactly like "
                "a dead cell and is not one.",
        },
        {
                "pinskyRinzelCA3Cell",
                "  <pinskyRinzelCA3Cell id=\"coverageCell\" iSoma=\"0.75uA_per_cm2\"\n"
                "                       iDend=\"0uA_per_cm2\" gLs=\"0.1mS_per_cm2\"\n"
                "                       gLd=\"0.1mS_per_cm2\" gNa=\"30mS_per_cm2\"\n"
                "                       gKdr=\"15mS_per_cm2\" gCa=\"10mS_per_cm2\"\n"
                "                       gKahp=\"0.8mS_per_cm2\" gKC=\"15mS_per_cm2\"\n"
                "                       gc=\"2.1mS_per_cm2\" eNa=\"60mV\" eCa=\"80mV\"\n"
                "                       eK=\"-75mV\" eL=\"-60mV\" pp=\"0.5\" cm=\"3uF_per_cm2\"\n"
                "                       alphac=\"2\" betac=\"0.1\" gNmda=\"0mS_per_cm2\"\n"
                "                       gAmpa=\"0mS_per_cm2\" qd0=\"0\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::REFUSED, 0, 0.0, 0.0,
                "ConditionalDerivedVariable 'alphaqd' is not supported",
                "CORRECT refusal. Pinsky & Rinzel (1994) parameters; the type is driven by its "
                "own iSoma parameter rather than by any stimulus component. It leans on "
                "<ConditionalDerivedVariable> five times over and is squarely Phase 3, so "
                "refusing it by name is the right answer today.",
        },
        {
                "IF_curr_alpha",
                "  <IF_curr_alpha id=\"coverageCell\" cm=\"1\" i_offset=\"0.9\" tau_syn_E=\"5\"\n"
                "                 tau_syn_I=\"5\" v_init=\"-65\" tau_m=\"20\" tau_refrac=\"2\"\n"
                "                 v_reset=\"-65\" v_rest=\"-65\" v_thresh=\"-50\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 5, 0.012, 0.0, "",
                "PyNN's own IF_curr_alpha defaults, with i_offset raised from 0 to 0.9nA. The "
                "cell is driven by that parameter rather than by a stimulus component: "
                "i_offset*tau_m/cm = 18mV of steady-state deflection against the 15mV to "
                "threshold.",
        },
        {
                "IF_curr_exp",
                "  <IF_curr_exp id=\"coverageCell\" cm=\"1\" i_offset=\"0.9\" tau_syn_E=\"5\"\n"
                "               tau_syn_I=\"5\" v_init=\"-65\" tau_m=\"20\" tau_refrac=\"2\"\n"
                "               v_reset=\"-65\" v_rest=\"-65\" v_thresh=\"-50\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 5, 0.012, 0.0, "",
                "As IF_curr_alpha. The two differ only in the synapse they expect to be given, "
                "so with no synapse attached their cell dynamics are identical and so is the "
                "result -- which is itself worth knowing.",
        },
        {
                "IF_cond_alpha",
                "  <IF_cond_alpha id=\"coverageCell\" cm=\"1\" i_offset=\"0.9\" tau_syn_E=\"5\"\n"
                "                 tau_syn_I=\"5\" v_init=\"-65\" tau_m=\"20\" tau_refrac=\"2\"\n"
                "                 v_reset=\"-65\" v_rest=\"-65\" v_thresh=\"-50\"\n"
                "                 e_rev_E=\"0\" e_rev_I=\"-70\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 5, 0.012, 0.0, "",
                "Included despite the name: this cell's own dynamics are current-based and "
                "identical to IF_curr_alpha's (see the file header). e_rev_E/e_rev_I are set "
                "because they are declared Parameters, and are unused by the dynamics -- the "
                "standard library says so itself.",
        },
        {
                "IF_cond_exp",
                "  <IF_cond_exp id=\"coverageCell\" cm=\"1\" i_offset=\"0.9\" tau_syn_E=\"5\"\n"
                "               tau_syn_I=\"5\" v_init=\"-65\" tau_m=\"20\" tau_refrac=\"2\"\n"
                "               v_reset=\"-65\" v_rest=\"-65\" v_thresh=\"-50\"\n"
                "               e_rev_E=\"0\" e_rev_I=\"-70\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, 5, 0.012, 0.0, "",
                "As IF_cond_alpha.",
        },
        {
                "EIF_cond_exp_isfa_ista",
                "  <EIF_cond_exp_isfa_ista id=\"coverageCell\" cm=\"0.281\" i_offset=\"0.8\"\n"
                "                          tau_syn_E=\"5\" tau_syn_I=\"5\" v_init=\"-70.6\"\n"
                "                          tau_m=\"9.3667\" tau_refrac=\"0.1\" v_reset=\"-70.6\"\n"
                "                          v_rest=\"-70.6\" v_thresh=\"-50.4\" e_rev_E=\"0\"\n"
                "                          e_rev_I=\"-80\" v_spike=\"-40.4\" delta_T=\"2\"\n"
                "                          tau_w=\"144\" a=\"4\" b=\"0.0805\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::REFUSED, 0, 0.0, 0.0,
                "ConditionalDerivedVariable 'delta_I' is not supported",
                "GAP, low severity. PyNN's own EIF defaults (the same Brette & Gerstner model "
                "adExIaFCell expresses, which RUNS here). What is refused is the PyNN wrapper's "
                "`delta_T .gt. 0` guard, expressed as a ConditionalDerivedVariable; the type "
                "also carries a <DerivedParameter> using the Heaviside H(), which this sweep "
                "never reaches.",
        },
        {
                "EIF_cond_alpha_isfa_ista",
                "  <EIF_cond_alpha_isfa_ista id=\"coverageCell\" cm=\"0.281\" i_offset=\"0.8\"\n"
                "                            tau_syn_E=\"5\" tau_syn_I=\"5\" v_init=\"-70.6\"\n"
                "                            tau_m=\"9.3667\" tau_refrac=\"0.1\"\n"
                "                            v_reset=\"-70.6\" v_rest=\"-70.6\"\n"
                "                            v_thresh=\"-50.4\" e_rev_E=\"0\" e_rev_I=\"-80\"\n"
                "                            v_spike=\"-40.4\" delta_T=\"2\" tau_w=\"144\"\n"
                "                            a=\"4\" b=\"0.0805\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::REFUSED, 0, 0.0, 0.0,
                "ConditionalDerivedVariable 'delta_I' is not supported",
                "As EIF_cond_exp_isfa_ista -- the two differ only in the synapse they expect.",
        },
        {
                "HH_cond_exp",
                "  <HH_cond_exp id=\"coverageCell\" cm=\"0.2\" i_offset=\"1.0\"\n"
                "               tau_syn_E=\"0.2\" tau_syn_I=\"2\" v_init=\"-65\"\n"
                "               v_offset=\"-63\" e_rev_E=\"0\" e_rev_I=\"-80\" e_rev_K=\"-90\"\n"
                "               e_rev_Na=\"50\" e_rev_leak=\"-65\" g_leak=\"0.01\"\n"
                "               gbar_K=\"6\" gbar_Na=\"20\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"8ms\" step=\"0.001ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                0, ExpectedVerdict::RUNS, -1, 0.1, 0.0, "",
                "PyNN's own HH_cond_exp defaults with i_offset raised to 1.0nA. NOTE THE STEP: "
                "0.001ms, not the 0.1ms every other candidate uses. At 0.1ms this model diverges "
                "to NaN within the first action potential -- forward Euler is unstable once the "
                "Traub gate time constants fall below the step, which they do on the upstroke. "
                "At 0.001ms it integrates a clean AP (v from -82mV to +45mV, m 0 -> 0.9999). "
                "That is a step-size requirement, not an engine defect, but it is a real limit "
                "and lifting it needs an exponential-Euler gate update. The divergence at 0.1ms "
                "is deliberately NOT asserted: it would fail the day it is fixed. This type "
                "declares no EventOut, so the correct spike count is exactly zero.",
        },
        {
                // ── current-based synapses ─────────────────────────────────────────────────
                // Two iafCells: neuron 0 is driven to fire by a pulseGenerator, neuron 1
                // receives nothing but the projection. The per-edge weight is 0.05, deliberately
                // far below the ~20mV neuron 1 needs to reach threshold, so its membrane
                // RESPONSE is visible. At weight 1 each event delivers about 54mV in a single
                // tick, neuron 1 fires and resets within that same tick, and every post-tick
                // sample reads exactly the resting potential -- a synapse that is working
                // perfectly looks completely dead.
                "alphaCurrentSynapse",
                "  <iafCell id=\"drivingCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
                "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
                "  <iafCell id=\"receivingCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
                "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
                "  <alphaCurrentSynapse id=\"coverageSynapse\" tau=\"2ms\" ibase=\"1nA\"\n"
                "                       weight=\"1\"/>\n"
                "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
                "                  amplitude=\"0.5nA\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"DrivingPop\" component=\"drivingCell\"\n"
                "                type=\"populationList\" size=\"1\">\n"
                "      <instance id=\"0\"><location x=\"0\" y=\"0\" z=\"0\"/></instance>\n"
                "    </population>\n"
                "    <population id=\"ReceivingPop\" component=\"receivingCell\"\n"
                "                type=\"populationList\" size=\"1\">\n"
                "      <instance id=\"0\"><location x=\"0\" y=\"0\" z=\"0\"/></instance>\n"
                "    </population>\n"
                "    <projection id=\"coverageProjection\" presynapticPopulation=\"DrivingPop\"\n"
                "                postsynapticPopulation=\"ReceivingPop\"\n"
                "                synapse=\"coverageSynapse\">\n"
                "      <connectionWD id=\"0\" preCellId=\"../DrivingPop/0/drivingCell\"\n"
                "                    postCellId=\"../ReceivingPop/0/receivingCell\"\n"
                "                    weight=\"0.05\" delay=\"0ms\"/>\n"
                "    </projection>\n"
                "    <inputList id=\"coverageInputs\" component=\"coverageDrive\"\n"
                "               population=\"DrivingPop\">\n"
                "      <input id=\"0\" target=\"../DrivingPop/0/drivingCell\"\n"
                "             destination=\"synapses\"/>\n"
                "    </inputList>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                1, ExpectedVerdict::RUNS, 0, 0.004, 2.7182818e-9, "",
                "One event's whole alpha response carries e*ibase*tau*weight = 2.71828 * 1nA * "
                "2ms * 0.05 = 2.718pC of charge, which the engine delivers in one tick as "
                "charge/dt = 2.718nA. tau/ibase are the values the jNeuroML-checked "
                "izhikevich_network fixture uses.",
        },
        {
                "expCurrSynapse",
                "  <iafCell id=\"drivingCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
                "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
                "  <iafCell id=\"receivingCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
                "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
                "  <expCurrSynapse id=\"coverageSynapse\" tau_syn=\"5\" weight=\"1\"/>\n"
                "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
                "                  amplitude=\"0.5nA\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"DrivingPop\" component=\"drivingCell\"\n"
                "                type=\"populationList\" size=\"1\">\n"
                "      <instance id=\"0\"><location x=\"0\" y=\"0\" z=\"0\"/></instance>\n"
                "    </population>\n"
                "    <population id=\"ReceivingPop\" component=\"receivingCell\"\n"
                "                type=\"populationList\" size=\"1\">\n"
                "      <instance id=\"0\"><location x=\"0\" y=\"0\" z=\"0\"/></instance>\n"
                "    </population>\n"
                "    <projection id=\"coverageProjection\" presynapticPopulation=\"DrivingPop\"\n"
                "                postsynapticPopulation=\"ReceivingPop\"\n"
                "                synapse=\"coverageSynapse\">\n"
                "      <connectionWD id=\"0\" preCellId=\"../DrivingPop/0/drivingCell\"\n"
                "                    postCellId=\"../ReceivingPop/0/receivingCell\"\n"
                "                    weight=\"0.05\" delay=\"0ms\"/>\n"
                "    </projection>\n"
                "    <inputList id=\"coverageInputs\" component=\"coverageDrive\"\n"
                "               population=\"DrivingPop\">\n"
                "      <input id=\"0\" target=\"../DrivingPop/0/drivingCell\"\n"
                "             destination=\"synapses\"/>\n"
                "    </inputList>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                1, ExpectedVerdict::RUNS, 0, 0.004, 2.5e-9, "",
                "PyNN's current-based exponential synapse. Its own current is dimensionless "
                "times a 1nA constant and its tau_syn is dimensionless milliseconds, so one "
                "event carries weight * 1nA * 5ms * 0.05 = 2.5pC, delivered in one tick as "
                "2.5nA. tau_syn=5 is PyNN's default.",
        },
        {
                "alphaCurrSynapse",
                "  <iafCell id=\"drivingCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
                "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
                "  <iafCell id=\"receivingCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
                "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
                "  <alphaCurrSynapse id=\"coverageSynapse\" tau_syn=\"5\" weight=\"1\"/>\n"
                "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
                "                  amplitude=\"0.5nA\"/>\n"
                "  <network id=\"CoverageNet\">\n"
                "    <population id=\"DrivingPop\" component=\"drivingCell\"\n"
                "                type=\"populationList\" size=\"1\">\n"
                "      <instance id=\"0\"><location x=\"0\" y=\"0\" z=\"0\"/></instance>\n"
                "    </population>\n"
                "    <population id=\"ReceivingPop\" component=\"receivingCell\"\n"
                "                type=\"populationList\" size=\"1\">\n"
                "      <instance id=\"0\"><location x=\"0\" y=\"0\" z=\"0\"/></instance>\n"
                "    </population>\n"
                "    <projection id=\"coverageProjection\" presynapticPopulation=\"DrivingPop\"\n"
                "                postsynapticPopulation=\"ReceivingPop\"\n"
                "                synapse=\"coverageSynapse\">\n"
                "      <connectionWD id=\"0\" preCellId=\"../DrivingPop/0/drivingCell\"\n"
                "                    postCellId=\"../ReceivingPop/0/receivingCell\"\n"
                "                    weight=\"0.05\" delay=\"0ms\"/>\n"
                "    </projection>\n"
                "    <inputList id=\"coverageInputs\" component=\"coverageDrive\"\n"
                "               population=\"DrivingPop\">\n"
                "      <input id=\"0\" target=\"../DrivingPop/0/drivingCell\"\n"
                "             destination=\"synapses\"/>\n"
                "    </inputList>\n"
                "  </network>\n"
                "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
                "              target=\"CoverageNet\"/>\n",
                1, ExpectedVerdict::RUNS, 0, 0.004, 6.7955e-9, "",
                "PyNN's current-based alpha synapse: the same 5ms as expCurrSynapse but shaped, "
                "so one event carries e times as much charge -- 2.71828 * 1nA * 5ms * 0.05 = "
                "6.7957pC, delivered as 6.7957nA in one tick.",
        },
};

// ── the sweep ───────────────────────────────────────────────────────────────────────────────

class StandardLibraryCoverage : public ::testing::TestWithParam<CandidateModel> {};

TEST_P(StandardLibraryCoverage, matches_its_recorded_verdict) {
    const CandidateModel &candidate = GetParam();
    const CandidateOutcome outcome = drive_candidate(candidate);
    record_inventory_row(candidate, outcome);

    if (candidate.expected_verdict == ExpectedVerdict::REFUSED) {
        ASSERT_FALSE(outcome.constructed)
                << candidate.component_type_name
                << " is recorded as refused but constructed. If it is genuinely supported now, "
                   "move it to RUNS and give it the evidence the other running candidates carry.";
        EXPECT_NE(outcome.failure_message.find(candidate.expected_refusal_substring), String::npos)
                << candidate.component_type_name << " is still refused, but with a different "
                << "message than recorded.\n  expected to contain: "
                << candidate.expected_refusal_substring << "\n  actual: "
                << outcome.failure_message;
        return;
    }

    ASSERT_TRUE(outcome.constructed)
            << candidate.component_type_name << " is recorded as running but was refused: "
            << outcome.failure_message;
    ASSERT_GT(outcome.tick_count, 0) << candidate.component_type_name << " ran zero ticks";

    EXPECT_TRUE(outcome.every_sample_finite)
            << candidate.component_type_name
            << " produced a non-finite value in some cell_state slot";

    const f64 membrane_excursion = outcome.membrane_maximum - outcome.membrane_minimum;

    if (candidate.expected_verdict == ExpectedVerdict::INERT) {
        // Not "small": exactly constant. This candidate's state has no term any input could
        // reach, so anything other than a flat trace is the engine writing into it.
        EXPECT_EQ(outcome.membrane_minimum, outcome.membrane_maximum)
                << candidate.component_type_name
                << " is recorded as inert by construction but its membrane variable moved by "
                << membrane_excursion;
        EXPECT_EQ(outcome.observed_spike_count, 0)
                << candidate.component_type_name << " is recorded as inert but emitted "
                << outcome.observed_spike_count << " spikes";
        return;
    }

    EXPECT_GE(membrane_excursion, candidate.minimum_expected_membrane_excursion)
            << candidate.component_type_name << " constructed and stepped but its membrane "
            << "variable barely moved: range [" << outcome.membrane_minimum << ", "
            << outcome.membrane_maximum << "]. A model that runs and silently does nothing is "
            << "the failure this sweep exists to catch.";

    if (candidate.minimum_expected_spike_count <= 0) {
        // Either the ComponentType declares no EventOut at all, or the candidate is driven
        // deliberately subthreshold. Both mean exactly zero, and exactly zero is what is
        // checked -- "at least zero" would assert nothing.
        //
        // The subthreshold case is load-bearing, not cosmetic. State is read AFTER each tick
        // completes, and a cell that crosses threshold and resets inside one tick is back at
        // its resting potential by the time it is sampled. Raise the synapse candidates' weight
        // to 1 and the postsynaptic cell fires on every presynaptic event while every sample
        // reads exactly -70mV -- a synapse working perfectly, indistinguishable from a dead one.
        // Requiring no spike is what keeps the membrane excursion beside it meaningful.
        EXPECT_EQ(outcome.observed_spike_count, 0)
                << candidate.component_type_name << " emitted " << outcome.observed_spike_count
                << " spikes where exactly zero was expected";
    } else {
        EXPECT_GE(outcome.observed_spike_count, candidate.minimum_expected_spike_count)
                << candidate.component_type_name << " emitted " << outcome.observed_spike_count
                << " spikes under a drive chosen to make it fire at least "
                << candidate.minimum_expected_spike_count << " times";
    }

    if (candidate.expected_delivered_current_per_event != 0.0) {
        ASSERT_GT(outcome.driving_spike_count, 0)
                << candidate.component_type_name
                << ": the presynaptic cell never fired, so the synapse was never exercised";

        const f64 delivered_per_event =
                outcome.total_delivered_current / (f64)outcome.driving_spike_count;
        EXPECT_NEAR(delivered_per_event, candidate.expected_delivered_current_per_event,
                    candidate.expected_delivered_current_per_event * 0.01)
                << candidate.component_type_name << " delivered " << delivered_per_event
                << "A per presynaptic event where its own impulse response carries "
                << candidate.expected_delivered_current_per_event
                << "A. A disagreement here is a unit scale, a sign or the per-edge weight.";
    }
}

INSTANTIATE_TEST_SUITE_P(
        NeuroMLStandardLibrary, StandardLibraryCoverage,
        ::testing::ValuesIn(CANDIDATE_MODELS),
        [](const ::testing::TestParamInfo<CandidateModel> &parameter_info) {
            return String(parameter_info.param.component_type_name);
        });

// ── the inventory is complete ───────────────────────────────────────────────────────────────

TEST(StandardLibraryCoverageInventory, every_declared_component_type_is_classified) {
    spikecorec::Vector<String> declared_names;
    for (const String &file_name : {String("Cells.xml"), String("Synapses.xml"),
                                    String("PyNN.xml")}) {
        for (const String &name : enumerate_component_type_names(file_name)) {
            declared_names.push_back(name);
        }
    }
    ASSERT_FALSE(declared_names.empty())
            << "no ComponentType was found in the vendored standard library at "
            << SPIKECOREC_NML_STD_LIB_DIR << " -- the checkout is broken, and skipping over that "
            << "would retract this whole sweep while still reporting green";

    for (const String &declared_name : declared_names) {
        bool is_classified = false;
        for (const LibraryClassification &entry : STANDARD_LIBRARY_CLASSIFICATION) {
            if (declared_name == entry.component_type_name) {
                is_classified = true;
                break;
            }
        }
        EXPECT_TRUE(is_classified)
                << "the vendored standard library declares ComponentType '" << declared_name
                << "', which this sweep does not classify. Add it to "
                   "STANDARD_LIBRARY_CLASSIFICATION as a candidate or as an exclusion with a "
                   "reason -- an unclassified type is one nobody has decided about.";
    }

    for (const LibraryClassification &entry : STANDARD_LIBRARY_CLASSIFICATION) {
        bool is_declared = false;
        for (const String &declared_name : declared_names) {
            if (declared_name == entry.component_type_name) {
                is_declared = true;
                break;
            }
        }
        EXPECT_TRUE(is_declared)
                << "this sweep classifies '" << entry.component_type_name
                << "', which the vendored standard library no longer declares";
    }
}

TEST(StandardLibraryCoverageInventory, every_candidate_role_has_a_driven_model) {
    for (const LibraryClassification &entry : STANDARD_LIBRARY_CLASSIFICATION) {
        if (entry.role != LibraryRole::POINT_CELL_CANDIDATE &&
            entry.role != LibraryRole::CURRENT_BASED_SYNAPSE_CANDIDATE) {
            continue;
        }
        bool has_model = false;
        for (const CandidateModel &candidate : CANDIDATE_MODELS) {
            if (String(candidate.component_type_name) == entry.component_type_name) {
                has_model = true;
                break;
            }
        }
        EXPECT_TRUE(has_model) << entry.component_type_name
                               << " is classified as a candidate but no model drives it, so the "
                                  "sweep silently does not cover it";
    }

    for (const CandidateModel &candidate : CANDIDATE_MODELS) {
        bool is_a_candidate_role = false;
        for (const LibraryClassification &entry : STANDARD_LIBRARY_CLASSIFICATION) {
            if (String(candidate.component_type_name) != entry.component_type_name) continue;
            is_a_candidate_role = entry.role == LibraryRole::POINT_CELL_CANDIDATE ||
                                  entry.role == LibraryRole::CURRENT_BASED_SYNAPSE_CANDIDATE;
            break;
        }
        EXPECT_TRUE(is_a_candidate_role)
                << candidate.component_type_name
                << " is driven by this sweep but is not classified as a candidate";
    }
}

// ── the two isolating controls ──────────────────────────────────────────────────────────────

// The control for the iafRefCell / iafTauRefCell gap. This is iafRefCell's Dynamics body
// VERBATIM -- same StateVariables, same DerivedVariables, same OnStart, same two Regimes -- with
// exactly one thing changed: it extends baseIafCapCell rather than iafCell, so no regime-free
// TimeDerivative for 'v' is inherited. It compiles and fires. That isolates the refusal to
// extends flattening merging a parent's unregimed TimeDerivative into a child that re-expresses
// the same derivative inside a Regime, and rules out the regime machinery, the refractory
// transition and the dynamics themselves.
TEST(StandardLibraryCoverageControl, an_iaf_refractory_cell_rebased_off_its_parent_compiles) {
    CandidateModel rebased_candidate{};
    rebased_candidate.component_type_name = "iafRefCell_rebased_control";
    rebased_candidate.observed_neuron_index = 0;
    rebased_candidate.model_body =
            "  <ComponentType name=\"RebasedIafRefCell\" extends=\"baseIafCapCell\">\n"
            "    <Parameter name=\"leakConductance\" dimension=\"conductance\"/>\n"
            "    <Parameter name=\"leakReversal\" dimension=\"voltage\"/>\n"
            "    <Parameter name=\"refract\" dimension=\"time\"/>\n"
            "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>\n"
            "    <EventPort name=\"spike\" direction=\"out\"/>\n"
            "    <Dynamics>\n"
            "      <StateVariable name=\"v\" exposure=\"v\" dimension=\"voltage\"/>\n"
            "      <StateVariable name=\"lastSpikeTime\" dimension=\"time\"/>\n"
            "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\"\n"
            "                       select=\"synapses[*]/i\" reduce=\"add\"/>\n"
            "      <DerivedVariable name=\"iMemb\" dimension=\"current\" exposure=\"iMemb\"\n"
            "                       value=\"leakConductance * (leakReversal - v) + iSyn\"/>\n"
            "      <OnStart>\n"
            "        <StateAssignment variable=\"v\" value=\"leakReversal\"/>\n"
            "      </OnStart>\n"
            "      <Regime name=\"refractory\">\n"
            "        <OnEntry>\n"
            "          <StateAssignment variable=\"lastSpikeTime\" value=\"t\"/>\n"
            "          <StateAssignment variable=\"v\" value=\"reset\"/>\n"
            "        </OnEntry>\n"
            "        <OnCondition test=\"t .gt. lastSpikeTime + refract\">\n"
            "          <Transition regime=\"integrating\"/>\n"
            "        </OnCondition>\n"
            "      </Regime>\n"
            "      <Regime name=\"integrating\" initial=\"true\">\n"
            "        <TimeDerivative variable=\"v\" value=\"iMemb / C\"/>\n"
            "        <OnCondition test=\"v .gt. thresh\">\n"
            "          <EventOut port=\"spike\"/>\n"
            "          <Transition regime=\"refractory\"/>\n"
            "        </OnCondition>\n"
            "      </Regime>\n"
            "    </Dynamics>\n"
            "  </ComponentType>\n"
            "  <RebasedIafRefCell id=\"coverageCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
            "                     leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"\n"
            "                     refract=\"5ms\"/>\n"
            "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
            "                  amplitude=\"0.5nA\"/>\n"
            "  <network id=\"CoverageNet\">\n"
            "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
            "    <explicitInput target=\"CoveragePop[0]\" input=\"coverageDrive\"/>\n"
            "  </network>\n"
            "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
            "              target=\"CoverageNet\"/>\n";

    const CandidateOutcome outcome = drive_candidate(rebased_candidate);

    ASSERT_TRUE(outcome.constructed)
            << "iafRefCell's own dynamics, rebased so nothing regime-free is inherited, were "
               "still refused: "
            << outcome.failure_message
            << "\nThat would mean the iafRefCell refusal is NOT extends flattening, and the "
               "note recorded against iafRefCell in this file is wrong.";
    EXPECT_TRUE(outcome.every_sample_finite);
    EXPECT_GT(outcome.observed_spike_count, 10)
            << "the rebased cell compiled but hardly fired, so it does not establish that the "
               "dynamics themselves are fine";

    // The 5ms refractory period has to be doing something, or the control would pass just as
    // well against a cell that ignored it. Without it the same drive fires ~58 times over the
    // 300ms pulse; with it, roughly half that.
    EXPECT_LT(outcome.observed_spike_count, 45)
            << "the rebased cell fired " << outcome.observed_spike_count
            << " times, which is the rate of a cell with NO refractory period -- the refractory "
               "Regime is not being honoured";
}

// ── a defect this sweep found, recorded as a DISABLED reproduction ──────────────────────────
//
// DISABLED because it FAILS today, and it asserts what is CORRECT rather than what happens.
// Nothing about it is speculative -- the numbers below were measured.
//
// A `pulseGenerator` declaring `amplitude="0nA"` injects ONE AMPERE into its target for every
// tick of its window. Twelve orders of magnitude, silently: the run is finite throughout, the
// suite stays green, and the target fires on exactly every tick of the injection window while
// every post-tick sample of its membrane potential reads exactly the resting potential (it
// crosses threshold and resets inside the tick). Measured on the iafCell above: amplitude
// "0nA" -> 3000 spikes over the 3000-tick window, v pinned at -70mV; amplitude "1pA" -> 0
// spikes and a correct 0.1mV deflection; amplitude "0.5nA" -> 58 spikes. The injected stream's
// own value reads 1.0 where the declared amplitude is 0 and 1e-12 where it is 1pA, so the
// parse is fine and the magnitude is chosen wrongly downstream of it.
//
// Cause, at src/core/engine.cpp create_event_stream():
//
//     f64 event_magnitude = weight;
//     if (amplitude != 0.0)  event_magnitude = amplitude * weight;
//     else if (rate != 0.0)  event_magnitude = rate * weight;
//
// The fallthrough to a bare `weight` is there for `spikeArray`, which carries neither an
// amplitude nor a rate and whose events legitimately deliver the target's own dimensionless
// <inputW weight>. But the chain decides which KIND of component it has by looking at the
// VALUES, so a current injector whose amplitude is genuinely zero is indistinguishable from a
// component that declares no amplitude at all, and gets weight (1.0) in amps.
// SimulationInputConfig already carries `input_component_type_name`, so the kind is available
// without inference.
//
// Disabling a stimulus by zeroing its amplitude is an ordinary thing to do to a model, which is
// what makes this worth a standing record. This file does not own src/core/engine.cpp; when
// that is fixed, drop the DISABLED_ prefix.
TEST(StandardLibraryCoverageControl,
     DISABLED_a_zero_amplitude_current_injector_delivers_no_current) {
    CandidateModel zero_amplitude_candidate{};
    zero_amplitude_candidate.component_type_name = "pulseGenerator_zero_amplitude";
    zero_amplitude_candidate.observed_neuron_index = 0;
    zero_amplitude_candidate.model_body =
            "  <iafCell id=\"coverageCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
            "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
            "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
            "                  amplitude=\"0nA\"/>\n"
            "  <network id=\"CoverageNet\">\n"
            "    <population id=\"CoveragePop\" component=\"coverageCell\" size=\"1\"/>\n"
            "    <explicitInput target=\"CoveragePop[0]\" input=\"coverageDrive\"/>\n"
            "  </network>\n"
            "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
            "              target=\"CoverageNet\"/>\n";

    const CandidateOutcome outcome = drive_candidate(zero_amplitude_candidate);

    ASSERT_TRUE(outcome.constructed) << outcome.failure_message;
    EXPECT_EQ(outcome.observed_spike_count, 0)
            << "an iafCell resting at its own leak reversal, driven by a pulseGenerator of "
               "amplitude 0, fired " << outcome.observed_spike_count
            << " times. Declaring no stimulus at all gives 0 (measured), so this is the zero "
               "amplitude being read as a magnitude of 1.";
    EXPECT_EQ(outcome.membrane_minimum, outcome.membrane_maximum)
            << "a zero-amplitude injector moved the membrane potential";
}

// The owner ruling excludes conductance-based synapses from this sweep. That exclusion is only
// safe while the engine keeps REFUSING them: silently running one as if it were current-based
// would be a different model, not an approximation. Spot checked on one representative so
// "excluded" can never quietly become "silently mis-simulated".
TEST(StandardLibraryCoverageControl, a_conductance_based_synapse_is_refused_by_name) {
    CandidateModel conductance_candidate{};
    conductance_candidate.component_type_name = "expOneSynapse_exclusion_control";
    conductance_candidate.observed_neuron_index = 1;
    conductance_candidate.model_body =
            "  <iafCell id=\"drivingCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
            "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
            "  <iafCell id=\"receivingCell\" C=\"100pF\" leakConductance=\"10nS\"\n"
            "           leakReversal=\"-70mV\" thresh=\"-50mV\" reset=\"-70mV\"/>\n"
            "  <expOneSynapse id=\"coverageSynapse\" gbase=\"5nS\" erev=\"0mV\"\n"
            "                 tauDecay=\"3ms\"/>\n"
            "  <pulseGenerator id=\"coverageDrive\" delay=\"20ms\" duration=\"300ms\"\n"
            "                  amplitude=\"0.5nA\"/>\n"
            "  <network id=\"CoverageNet\">\n"
            "    <population id=\"DrivingPop\" component=\"drivingCell\"\n"
            "                type=\"populationList\" size=\"1\">\n"
            "      <instance id=\"0\"><location x=\"0\" y=\"0\" z=\"0\"/></instance>\n"
            "    </population>\n"
            "    <population id=\"ReceivingPop\" component=\"receivingCell\"\n"
            "                type=\"populationList\" size=\"1\">\n"
            "      <instance id=\"0\"><location x=\"0\" y=\"0\" z=\"0\"/></instance>\n"
            "    </population>\n"
            "    <projection id=\"coverageProjection\" presynapticPopulation=\"DrivingPop\"\n"
            "                postsynapticPopulation=\"ReceivingPop\"\n"
            "                synapse=\"coverageSynapse\">\n"
            "      <connectionWD id=\"0\" preCellId=\"../DrivingPop/0/drivingCell\"\n"
            "                    postCellId=\"../ReceivingPop/0/receivingCell\"\n"
            "                    weight=\"0.05\" delay=\"0ms\"/>\n"
            "    </projection>\n"
            "    <inputList id=\"coverageInputs\" component=\"coverageDrive\"\n"
            "               population=\"DrivingPop\">\n"
            "      <input id=\"0\" target=\"../DrivingPop/0/drivingCell\"\n"
            "             destination=\"synapses\"/>\n"
            "    </inputList>\n"
            "  </network>\n"
            "  <Simulation id=\"coverageSim\" length=\"350ms\" step=\"0.1ms\"\n"
            "              target=\"CoverageNet\"/>\n";

    const CandidateOutcome outcome = drive_candidate(conductance_candidate);

    ASSERT_FALSE(outcome.constructed)
            << "a conductance-based synapse was accepted. It is excluded from this sweep on the "
               "understanding that the engine refuses it; running one through the current-based "
               "path computes a different model.";
    EXPECT_NE(outcome.failure_message.find("conductance-based synapses are not supported"),
              String::npos)
            << "expOneSynapse is still refused, but no longer as a conductance-based synapse: "
            << outcome.failure_message;
}

} // namespace
