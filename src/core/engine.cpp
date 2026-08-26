//
// Created by Alek Simpson on 5/30/26.
//

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/nml/dynamics_codegen.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

// The starting value of `variable_name` for a type, folded from its OnStart.
f64 starting_value_for(const Vector<DynamicsInstruction> &dynamics,
                       const String &variable_name,
                       const Vector<String> &parameter_names,
                       const Vector<Real> &parameter_values,
                       const String &owner_name) {
    for (const DynamicsInstruction &instruction : dynamics) {
        if (instruction.stage != DynamicsStage::Initialize) continue;
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
        if (instruction.target != variable_name) continue;

        return evaluate_initial_value(instruction.expression, parameter_names,
                                      parameter_values, owner_name);
    }

    // LEMS leaves a StateVariable with no OnStart at zero.
    return 0.0;
}

} // namespace

// ── construction ──────────────────────────────────────────────────────────────────

SpikeEngine::SpikeEngine(const String &lems_input_file)
    : SpikeEngine(lems_input_file, {}, "", 1.0, 0.0) {}

SpikeEngine::SpikeEngine(const String &lems_input_file,
                         const vector<vector<s32>> &adjacency,
                         const String &synapse_component_id,
                         f64 connection_weight,
                         f64 connection_delay_seconds)
    : logger(log::make_logger()) {

    // The backend is a member, constructed with this engine: there is no process-global
    // context to initialize and nothing to check before allocating.
    NML_Parser parser;
    if (!parser.validate_against_schema(lems_input_file)) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: " + lems_input_file + " failed lems schema validation:\n" +
                parser.last_schema_validation_errors);
    }

    network_details = parser.parse_lems(lems_input_file);

    if (!adjacency.empty()) {
        apply_topology(adjacency, synapse_component_id, connection_weight,
                       connection_delay_seconds);
    }

    layout = compute_model_layout(network_details);

    total_neuron_count = layout.total_neuron_count;
    lifetime = network_details.total_tick_count;
    step_dt = (f32)network_details.step_dt;
    if (network_details.random_seed.has_value()) simulation_seed = *network_details.random_seed;

    if (total_neuron_count == 0) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: " + lems_input_file + " defines no neurons — check that the "
                "Simulation's target names a network with at least one population");
    }
    if (step_dt <= 0.0f) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: " + lems_input_file + " gives the Simulation no usable step "
                "(parsed " + to_string(network_details.step_dt) + " s)");
    }

    allocate_model_buffers();
    initialize_model_buffers();
    initialize_cell_state();
    build_weight_matrix();
    collect_stimulus();

    const String master_kernel_source = generate_master_kernel(network_details, layout);
    logger->debug("SpikeEngine: generated master kernel, {} bytes", master_kernel_source.size());

    Optional<EngineFunction> compiled = gpu.create_function("master_step", master_kernel_source);
    if (!compiled.has_value()) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: the generated master kernel failed to compile — see the log above "
                "for the compiler's own diagnostic");
    }
    kernel_function = *compiled;

    spike_counts_per_neuron.assign((usize)total_neuron_count, 0);

    for (const RecordingConfig &profile : network_details.recording_profiles) {
        for (const RecordingSelection &selection : profile.selections) {
            if (!selection.event_port.empty()) continue;
            if (selection.neuron_index < 0) continue;
            traced_selections.push_back(selection);
        }
    }

    alive = true;
    logger->info("SpikeEngine constructed from {}: {} neurons, {} edges, {} cell types, "
                 "{} synapse prototypes, dt={} s, {} ticks, spike history {} rows",
                 lems_input_file, total_neuron_count, layout.total_edge_count,
                 network_details.cell_types.size(),
                 network_details.synapse_prototypes.size(),
                 network_details.step_dt, lifetime, layout.spike_history_length);
}

void SpikeEngine::allocate_model_buffers() {
    const s64 prototype_count = (s64)network_details.synapse_prototypes.size();
    const s64 synapse_state_count = layout.widest_synapse_state_count;

    Vector<EnginePointer> data_partitions;
    gpu.partition(sizeof(f32) * layout.cell_state_length, EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(f32) * layout.cell_parameter_length, EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(f32) * layout.synapse_parameter_length, EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(f32) * 2 * total_neuron_count, EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(u8) * layout.spike_history_length * total_neuron_count,
                  EngineDatatype::UNSIGNED8, data_partitions)
       .partition(sizeof(s64) * total_neuron_count, EngineDatatype::SIGNED64, data_partitions)
       // One accumulator per (tick parity, prototype, neuron) and one state per
       // (prototype, state variable, neuron) -- both independent of edge count, which is
       // the whole point of aggregating.
       .partition(sizeof(f32) * 2 * prototype_count * total_neuron_count,
                  EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(f32) * prototype_count * synapse_state_count * total_neuron_count,
                  EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(f32), EngineDatatype::FLOAT32, data_partitions);

    model_slab = gpu.allocate(data_partitions);

    // EnginePointer is a non-owning value: copying one produces a second name for the same
    // range, and the slab is what owns the storage. There is nothing to move.
    cell_state         = data_partitions[0];
    cell_parameters    = data_partitions[1];
    synapse_parameters = data_partitions[2];
    network_inputs     = data_partitions[3];
    spike_history      = data_partitions[4];
    last_spiked        = data_partitions[5];
    synapse_arrivals   = data_partitions[6];
    synapse_state      = data_partitions[7];
    empty_edge_plane   = data_partitions[8];

    logger->debug("SpikeEngine: model slab {} bytes — cell_state {}, cell_parameters {}, "
                  "synapse_parameters {}, network_inputs {}, spike_history {}, "
                  "synapse_arrivals {}, synapse_state {}",
                  model_slab.total_bytes, layout.cell_state_length, layout.cell_parameter_length,
                  layout.synapse_parameter_length, 2 * total_neuron_count,
                  layout.spike_history_length * total_neuron_count,
                  2 * prototype_count * total_neuron_count,
                  prototype_count * synapse_state_count * total_neuron_count);
}

void SpikeEngine::initialize_model_buffers() {
    const s64 prototype_count = (s64)network_details.synapse_prototypes.size();
    const s64 synapse_state_count = layout.widest_synapse_state_count;

    // The slab is uninitialized memory and every one of these is read before it is written
    // on the first tick.
    memset(network_inputs.get_contents(), 0, (usize)(2 * total_neuron_count) * sizeof(f32));
    memset(spike_history.get_contents(), 0,
           (usize)(layout.spike_history_length * total_neuron_count) * sizeof(u8));
    if (!synapse_arrivals.is_empty()) {
        memset(synapse_arrivals.get_contents(), 0,
               (usize)(2 * prototype_count * total_neuron_count) * sizeof(f32));
    }
    if (!empty_edge_plane.is_empty()) {
        memset(empty_edge_plane.get_contents(), 0, sizeof(f32));
    }

    // Each synapse type's state variables start at the value its OnStart gives them, which
    // for the aggregate is that starting value times zero incoming spikes -- i.e. the
    // starting value itself only when the type declares a non-zero one.
    if (!synapse_state.is_empty()) {
        f32 *synapse_state_data = synapse_state.get_contents_as<f32>();
        for (s64 prototype_index = 0; prototype_index < prototype_count; prototype_index += 1) {
            const ComponentPrototype &prototype =
                    network_details.synapse_prototypes[(usize)prototype_index];
            const SynapseTypeSpecification &synapse_type =
                    network_details.synapse_types[(usize)prototype.type_index];

            for (s64 slot = 0; slot < synapse_state_count; slot += 1) {
                const f64 initial = (slot < (s64)synapse_type.state_variable_names.size())
                        ? starting_value_for(synapse_type.dynamics,
                                             synapse_type.state_variable_names[(usize)slot],
                                             synapse_type.parameter_names,
                                             prototype.starting_parameters, synapse_type.name)
                        : 0.0;
                f32 *row = synapse_state_data +
                           (prototype_index * synapse_state_count + slot) * total_neuron_count;
                std::fill(row, row + total_neuron_count, (f32)initial);
            }
        }
    }

    // Negative means "has never fired" -- see the refractory gate in dynamics_codegen. A
    // zero would read as "fired on tick 0", holding every cell in the model refractory from
    // the start and the undriven ones forever.
    s64 *last_spiked_data = last_spiked.get_contents_as<s64>();
    std::fill(last_spiked_data, last_spiked_data + total_neuron_count, NEVER_SPIKED_TICK);

    // Parameter rows, one per prototype, in the column order the type declared.
    f32 *cell_parameter_data = static_cast<f32 *>(cell_parameters.get_contents());
    for (usize index = 0; index < network_details.cell_prototypes.size(); index += 1) {
        const ComponentPrototype &prototype = network_details.cell_prototypes[index];
        const s64 base = layout.cell_prototype_parameter_base[index];
        for (usize column = 0; column < prototype.starting_parameters.size(); column += 1) {
            cell_parameter_data[base + (s64)column] =
                    (f32)prototype.starting_parameters[column].float64;
        }
    }

    f32 *synapse_parameter_data = static_cast<f32 *>(synapse_parameters.get_contents());
    for (usize index = 0; index < network_details.synapse_prototypes.size(); index += 1) {
        const ComponentPrototype &prototype = network_details.synapse_prototypes[index];
        const s64 base = layout.synapse_prototype_parameter_base[index];
        for (usize column = 0; column < prototype.starting_parameters.size(); column += 1) {
            synapse_parameter_data[base + (s64)column] =
                    (f32)prototype.starting_parameters[column].float64;
        }
    }

}

// Runs each cell type's OnStart once per neuron. This is not a formality: iafCell's
// OnStart is `v = leakReversal`, and a cell left at 0 V when its threshold is -50 mV is
// already over threshold on tick 0 — the whole network fires once and then sits at reset
// forever, which looks exactly like a working simulation of a dead network.
void SpikeEngine::initialize_cell_state() {
    f32 *cell_state_data = static_cast<f32 *>(cell_state.get_contents());

    for (usize index = 0; index < network_details.populations.size(); index += 1) {
        const PopulationLayout &population = network_details.populations[index];
        const CellTypeSpecification &cell_type =
                network_details.cell_types[(usize)population.cell_type_index];
        const ComponentPrototype &prototype =
                network_details.cell_prototypes[(usize)population.prototype_index];

        const s64 base = layout.population_state_base[index];

        for (usize slot = 0; slot < cell_type.state_variable_names.size(); slot += 1) {
            const f64 initial = starting_value_for(
                    cell_type.dynamics, cell_type.state_variable_names[slot],
                    cell_type.parameter_names, prototype.starting_parameters, cell_type.name);

            f32 *run = cell_state_data + base + (s64)slot * population.neuron_count;
            std::fill(run, run + population.neuron_count, (f32)initial);

            logger->debug("SpikeEngine: population '{}' starts {} = {}",
                          population.population_name,
                          cell_type.state_variable_names[slot], initial);
        }
    }
}

// Every per-edge quantity lives in the WeightMatrix, and none of it as a per-edge value:
// the k^2-tree says which (source, target) pairs exist, and a shared low-rank basis says
// what each edge's weight and delay are. The synapse prototype is a run table, and per-edge
// synapse STATE does not exist at all -- it aggregates into synapse_state, one accumulator
// per (target, prototype).
void SpikeEngine::build_weight_matrix() {
    const Vector<Vector<s32>> network = build_adjacency_list(network_details);

    // rank -1 means "derive it from what the projections below actually contain" rather
    // than a constant someone guessed; capacity 0 leaves plasticity off and allocates
    // nothing for it.
    weights = WeightMatrix(gpu, network, /*rank=*/-1, /*check_indexing=*/true,
                           /*max_neighbor_count=*/-1, /*weight_seed=*/(s64)simulation_seed,
                           /*plasticity_delta_capacity=*/0);

    // The same run coalescing the codegen bakes into the kernel, from the same function --
    // the two must agree on the ordering exactly, or edges get the wrong synapse.
    Vector<s64> first_edge_ordinal;
    Vector<s64> edge_count;
    Vector<s64> synapse_prototype;
    Vector<f32> weight;
    Vector<s32> delay_ticks;
    collect_projection_runs(network_details, first_edge_ordinal, edge_count,
                            synapse_prototype, weight, delay_ticks);

    Vector<s32> synapse_prototype_narrow;
    synapse_prototype_narrow.reserve(synapse_prototype.size());
    for (s64 prototype_index : synapse_prototype) {
        synapse_prototype_narrow.push_back((s32)prototype_index);
    }

    weights.declare_projections(first_edge_ordinal, edge_count, synapse_prototype_narrow,
                                weight, delay_ticks);

    logger->debug("SpikeEngine: weight matrix built — {} nodes, {} edges, {} projection runs, "
                  "rank {}, worst weight error {:.3e}",
                  weights.node_count, weights.total_edge_count, first_edge_ordinal.size(),
                  weights.rank, weights.measured_weight_fit_error);
}

// A spike is a binary event; what it is worth in current depends entirely on the cell it
// lands on. When the model declines to say, the sensible reading of "a spike arrived" is
// "enough to make this cell fire", which is C * (threshold - resting) / dt: the charge that
// moves the membrane the whole way in one tick.
//
// Every term comes from the target's own declarations rather than from guessed names. The
// capacitance is the parameter whose DIMENSION is capacitance, whatever the model calls it.
// The threshold and the resting value come from the cell's own spike condition and its
// OnStart -- so GLIF1, which tests the parameter `vth`, and GLIF4, which tests the state
// variable `theta`, both resolve without either being special-cased.
f64 SpikeEngine::default_spike_amplitude_for(s64 neuron_index) const {
    // Strictly above threshold, because the comparison that fires a cell is a strict one:
    // landing exactly on it does not spike.
    constexpr f64 THRESHOLD_OVERSHOOT = 1.05;

    for (usize index = 0; index < network_details.populations.size(); index += 1) {
        const PopulationLayout &population = network_details.populations[index];
        if (neuron_index < population.first_neuron_index) continue;
        if (neuron_index >= population.first_neuron_index + population.neuron_count) continue;

        const CellTypeSpecification &cell_type =
                network_details.cell_types[(usize)population.cell_type_index];
        const ComponentPrototype &prototype =
                network_details.cell_prototypes[(usize)population.prototype_index];

        f64 capacitance = 0.0;
        for (usize slot = 0; slot < cell_type.parameter_dimensions.size(); slot += 1) {
            if (cell_type.parameter_dimensions[slot] != "capacitance") continue;
            if (slot >= prototype.starting_parameters.size()) continue;
            capacitance = prototype.starting_parameters[slot].float64;
        }
        if (capacitance <= 0.0) {
            log::throw_runtime_error(*logger,
                    "SpikeEngine: a spike train targets a '" + cell_type.name +
                    "', which declares no parameter of dimension capacitance, so there is "
                    "no way to work out what one event should inject. Give the input an "
                    "amplitude");
        }

        String membrane_name;
        String threshold_symbol;
        if (!find_spike_threshold_condition(cell_type, membrane_name, threshold_symbol)) {
            log::throw_runtime_error(*logger,
                    "SpikeEngine: a spike train targets a '" + cell_type.name +
                    "', whose spike condition is not a comparison this can read, so there "
                    "is no threshold to aim at. Give the input an amplitude");
        }

        // The swing to cover is measured from the lowest the membrane goes, not from where
        // it starts. A cell that has just fired sits at its reset value, which for every
        // iaf and GLIF cell is below the leak reversal -- sizing the event from the
        // starting potential makes the first spike fire and the ones after it fall short.
        f64 membrane_floor = starting_value_for(cell_type.dynamics, membrane_name,
                                                cell_type.parameter_names,
                                                prototype.starting_parameters,
                                                cell_type.name);

        for (const DynamicsInstruction &instruction : cell_type.dynamics) {
            if (instruction.stage != DynamicsStage::Reset) continue;
            if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
            if (instruction.target != membrane_name) continue;

            // Only a reset that folds to a value counts. GLIF2 resets to
            // `vreset + resetScale*(v - vth)`, which depends on the overshoot and cannot be
            // known here; its plain vreset floor is covered by the OnStart value instead.
            try {
                membrane_floor = std::min(
                        membrane_floor,
                        evaluate_initial_value(instruction.expression,
                                               cell_type.parameter_names,
                                               prototype.starting_parameters,
                                               cell_type.name));
            } catch (const std::runtime_error &) {
                // not foldable; the OnStart value stands
            }
        }

        const f64 resting = membrane_floor;

        // The threshold is a parameter in GLIF1/2/3 and a state variable in GLIF4/5.
        bool threshold_found = false;
        f64 threshold = 0.0;
        for (usize slot = 0; slot < cell_type.parameter_names.size(); slot += 1) {
            if (cell_type.parameter_names[slot] != threshold_symbol) continue;
            if (slot >= prototype.starting_parameters.size()) continue;
            threshold = prototype.starting_parameters[slot].float64;
            threshold_found = true;
        }
        for (const String &state_name : cell_type.state_variable_names) {
            if (threshold_found || state_name != threshold_symbol) continue;
            threshold = starting_value_for(cell_type.dynamics, state_name,
                                           cell_type.parameter_names,
                                           prototype.starting_parameters, cell_type.name);
            threshold_found = true;
        }

        if (!threshold_found || threshold <= resting) {
            log::throw_runtime_error(*logger,
                    "SpikeEngine: a spike train targets a '" + cell_type.name +
                    "', whose threshold '" + threshold_symbol + "' resolves to no value "
                    "above its starting membrane potential, so no finite current would make "
                    "it fire. Give the input an amplitude");
        }

        return THRESHOLD_OVERSHOOT * capacitance * (threshold - resting) /
               network_details.step_dt;
    }

    log::throw_runtime_error(*logger,
            "SpikeEngine: spike train targets neuron " + to_string(neuron_index) +
            ", which is outside every population");
}

void SpikeEngine::apply_topology(const vector<vector<s32>> &adjacency,
                                 const String &synapse_component_id,
                                 f64 connection_weight,
                                 f64 connection_delay_seconds) {
    const s64 neuron_count = (s64)network_details.neurons.size();
    if ((s64)adjacency.size() != neuron_count) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: the topology has " + to_string(adjacency.size()) +
                " rows but the model's populations declare " + to_string(neuron_count) +
                " neurons; one row per neuron is required");
    }

    // An instance id may be document-scoped ("net.syn"), so match the trailing name too.
    s64 prototype_index = -1;
    String available;
    for (usize index = 0; index < network_details.synapse_prototypes.size(); index += 1) {
        const String &instance_id = network_details.synapse_prototypes[index].instance_id;
        available += (available.empty() ? "" : ", ") + instance_id;

        const usize separator = instance_id.rfind('.');
        const String leaf = separator == String::npos ? instance_id
                                                      : instance_id.substr(separator + 1);
        if (instance_id == synapse_component_id || leaf == synapse_component_id) {
            prototype_index = (s64)index;
        }
    }
    if (prototype_index < 0) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: the topology names synapse '" + synapse_component_id +
                "', which the model does not declare. It declares: " +
                (available.empty() ? "no synapses at all" : available));
    }

    const s64 type_index =
            network_details.synapse_prototypes[(usize)prototype_index].type_index;

    // At least one tick: the engine's synaptic latency has no zero-delay path.
    const s64 delay_ticks = std::max<s64>(
            1, (s64)std::llround(connection_delay_seconds / network_details.step_dt));

    s64 edge_count = 0;
    for (s64 source = 0; source < neuron_count; source += 1) {
        Vector<NetworkEdge> &edges = network_details.neurons[(usize)source].outgoing_edges;
        edges.clear();
        edges.reserve(adjacency[(usize)source].size());

        for (s32 target : adjacency[(usize)source]) {
            if (target < 0 || (s64)target >= neuron_count) {
                log::throw_runtime_error(*logger,
                        "SpikeEngine: the topology sends neuron " + to_string(source) +
                        " to " + to_string(target) + ", which is outside the " +
                        to_string(neuron_count) + " neurons the model declares");
            }

            NetworkEdge edge;
            edge.target_neuron_index = target;
            edge.synapse_type_index = type_index;
            edge.synapse_prototype_index = prototype_index;
            edge.weight = connection_weight;
            edge.delay_tick_count = delay_ticks;
            edges.push_back(edge);
            edge_count += 1;
        }
    }

    logger->info("SpikeEngine: topology applied — {} neurons, {} edges, synapse '{}', "
                 "weight {}, delay {} ticks",
                 neuron_count, edge_count, synapse_component_id, connection_weight,
                 delay_ticks);
}

void SpikeEngine::collect_stimulus() {
    for (const SimulationInputConfig &profile : network_details.input_profiles) {
        for (const InputTarget &target : profile.targets) {
            if (target.neuron_index < 0) continue;

            if (profile.continuous_current_injection) {
                continuous_injection_targets.push_back(target.neuron_index);
                continuous_injection_amplitudes.push_back(
                        (f32)(profile.amplitude * target.weight));
                continuous_injection_start_ticks.push_back(profile.start_tick);
                continuous_injection_end_ticks.push_back(
                        profile.end_tick > profile.start_tick ? profile.end_tick : lifetime);
                continue;
            }

            if (target.event_ticks.empty()) continue;

            // A spike train injects a current for one tick at each event time. A model
            // that names an amplitude gets that; one that does not -- which is every
            // standard spike source, since LEMS binds only attributes matching a declared
            // Parameter -- gets whatever makes its target fire. A spike is a binary event
            // and its worth in current is a property of the cell receiving it.
            const f64 amplitude = profile.amplitude != 0.0
                    ? profile.amplitude
                    : default_spike_amplitude_for(target.neuron_index);

            ScheduledSpikeTrain train;
            train.neuron_index = target.neuron_index;
            train.magnitude = (f32)(amplitude * target.weight);
            train.event_ticks = target.event_ticks;
            scheduled_spike_trains.push_back(std::move(train));
        }
    }

    logger->info("SpikeEngine: stimulus — {} continuous injections, {} spike trains",
                 continuous_injection_targets.size(), scheduled_spike_trains.size());
}

SpikeEngine::~SpikeEngine() {
    if (alive) shutdown();
}

// Host-side stimulus lands in the row this tick's kernel is about to drain, so injected
// current reaches the cell on the tick the model asked for rather than the one after.
void SpikeEngine::apply_stimulus(s64 tick) {
    f32 *input_data = static_cast<f32 *>(network_inputs.get_contents());
    const s64 row_base = (tick % 2) * total_neuron_count;

    for (usize index = 0; index < continuous_injection_targets.size(); index += 1) {
        if (tick < continuous_injection_start_ticks[index]) continue;
        if (tick >= continuous_injection_end_ticks[index]) continue;

        input_data[row_base + continuous_injection_targets[index]] +=
                continuous_injection_amplitudes[index];
    }

    // Event times are sorted, so each train only ever walks forward. The first loop skips
    // any event the run has already passed, which matters when a model schedules several
    // events inside one tick or when step() is called out of order by a test.
    for (ScheduledSpikeTrain &train : scheduled_spike_trains) {
        while (train.cursor < train.event_ticks.size() &&
               (s64)train.event_ticks[train.cursor] < tick) {
            train.cursor += 1;
        }
        while (train.cursor < train.event_ticks.size() &&
               (s64)train.event_ticks[train.cursor] == tick) {
            input_data[row_base + train.neuron_index] += train.magnitude;
            train.cursor += 1;
        }
    }
}

// One buffer the kernel declares, or the placeholder when the model has no such plane. A
// network with no connections allocates none of them, and there is no way to bind
// "nothing" to a buffer slot a kernel names.
EnginePointer SpikeEngine::resolve_edge_plane(const EnginePointer &plane) const {
    return plane.is_empty() ? empty_edge_plane : plane;
}

void SpikeEngine::step_simulation(s64 tick) {
    apply_stimulus(tick);

    // Scalars live on the stack for the duration of the dispatch and reach the kernel as
    // inline constant data: an EnginePointer with inline_scalar set means "these bytes",
    // not "this range of a slab", and run_function is what tells the two apart.
    const s32 neuron_count_argument = (s32)total_neuron_count;
    const s32 spike_history_length_argument = (s32)layout.spike_history_length;
    const s32 rank_float4_stride_argument = (s32)weights.rank_float4_stride;

    const K2Tree &tree = weights.k2tree;

    Vector<EnginePointer> parameters = {
        inline_scalar_argument(tick),                            // 0
        inline_scalar_argument(step_dt),                         // 1
        inline_scalar_argument(neuron_count_argument),           // 2
        inline_scalar_argument(spike_history_length_argument),   // 3
        inline_scalar_argument(rank_float4_stride_argument),     // 4
        cell_state,                                              // 5
        resolve_edge_plane(cell_parameters),                     // 6
        resolve_edge_plane(synapse_parameters),                  // 7
        network_inputs,                                          // 8
        spike_history,                                           // 9
        last_spiked,                                             // 10
        resolve_edge_plane(tree.internal_node_words),            // 11
        resolve_edge_plane(tree.leaf_node_words),                // 12
        resolve_edge_plane(tree.rank_superblock_table),          // 13
        resolve_edge_plane(tree.rank_subblock_table),            // 14
        inline_scalar_argument(tree.branching_factor),           // 15
        inline_scalar_argument(tree.superblock_size_words),      // 16
        inline_scalar_argument(tree.padded_node_count),          // 17
        inline_scalar_argument(tree.tree_height),                // 18
        inline_scalar_argument(tree.internal_bit_count),         // 19
        resolve_edge_plane(weights.U_matrix),                    // 20
        resolve_edge_plane(weights.V_matrix),                    // 21
        resolve_edge_plane(weights.coefficient_range(WeightMatrix::DEFAULT_MATRIX_INDEX)), // 22
        resolve_edge_plane(weights.coefficient_range(WeightMatrix::DELAY_MATRIX_INDEX)),   // 23
        resolve_edge_plane(weights.edge_row_offset),             // 24
        resolve_edge_plane(synapse_arrivals),                    // 25
        resolve_edge_plane(synapse_state),                       // 26
    };

    // job_count is the neuron count, not a block count: run_function dispatches TOTAL
    // threads and works out the groups itself.
    if (!gpu.run_function(kernel_function, parameters, total_neuron_count)) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: tick " + to_string(tick) + " failed on the GPU");
    }

    record_tick(tick);
}

void SpikeEngine::run() {
    logger->info("SpikeEngine: running {} ticks at dt={} s", lifetime, step_dt);

    for (s64 tick = 0; tick < lifetime; tick += 1) step_simulation(tick);

    logger->info("SpikeEngine: run finished — {} spikes, mean rate {:.2f} Hz, {:.1f}% of "
                 "neurons spiked at least once",
                 recorded_spikes.size(), mean_firing_rate_hertz(),
                 100.0 * fraction_of_neurons_that_spiked());
}

void SpikeEngine::record_tick(s64 tick) {
    const u8 *spike_row = static_cast<const u8 *>(spike_history.get_contents()) +
                          (tick % layout.spike_history_length) * total_neuron_count;

    for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
        if (spike_row[neuron_index] == 0) continue;

        spike_counts_per_neuron[(usize)neuron_index] += 1;
        recorded_spikes.push_back(RecordedSpike{(f64)tick * network_details.step_dt,
                                                neuron_index});
    }

    if (membrane_video_recorder && tick % membrane_video_frame_stride == 0) {
        const f32 *cell_state_data = static_cast<const f32 *>(cell_state.get_contents());
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
            membrane_frame_scratch[(usize)neuron_index] =
                    cell_state_data[membrane_offset_per_neuron[(usize)neuron_index]];
        }
        membrane_video_recorder->record_frame(membrane_frame_scratch.data(),
                                              total_neuron_count);
    }

    if (traced_selections.empty()) return;

    recorded_trace_times.push_back((f64)tick * network_details.step_dt);
    for (const RecordingSelection &selection : traced_selections) {
        recorded_traces.push_back(
                read_state_variable(selection.neuron_index, selection.variable_name));
    }
}

// ── readback ──────────────────────────────────────────────────────────────────────

f32 SpikeEngine::read_state_variable(s64 neuron_index, const String &variable_name) const {
    for (usize index = 0; index < network_details.populations.size(); index += 1) {
        const PopulationLayout &population = network_details.populations[index];
        if (neuron_index < population.first_neuron_index) continue;
        if (neuron_index >= population.first_neuron_index + population.neuron_count) continue;

        const CellTypeSpecification &cell_type =
                network_details.cell_types[(usize)population.cell_type_index];

        for (usize slot = 0; slot < cell_type.state_variable_names.size(); slot += 1) {
            if (cell_type.state_variable_names[slot] != variable_name) continue;

            const s64 local_index = neuron_index - population.first_neuron_index;
            const s64 offset = layout.population_state_base[index] +
                               (s64)slot * population.neuron_count + local_index;
            return static_cast<const f32 *>(cell_state.get_contents())[offset];
        }

        log::throw_runtime_error(*logger,
                "SpikeEngine: neuron " + to_string(neuron_index) + " is a '" +
                cell_type.name + "', which declares no state variable '" +
                variable_name + "'");
    }

    log::throw_runtime_error(*logger,
            "SpikeEngine: neuron index " + to_string(neuron_index) +
            " is outside every population");
}

f64 SpikeEngine::mean_firing_rate_hertz() const {
    if (total_neuron_count == 0 || lifetime == 0 || network_details.step_dt <= 0.0) return 0.0;

    s64 total_spikes = 0;
    for (s64 count : spike_counts_per_neuron) total_spikes += count;

    const f64 elapsed_seconds = (f64)lifetime * network_details.step_dt;
    return (f64)total_spikes / ((f64)total_neuron_count * elapsed_seconds);
}

f64 SpikeEngine::fraction_of_neurons_that_spiked() const {
    if (total_neuron_count == 0) return 0.0;

    s64 spiking_neurons = 0;
    for (s64 count : spike_counts_per_neuron) spiking_neurons += count > 0 ? 1 : 0;

    return (f64)spiking_neurons / (f64)total_neuron_count;
}

// ── membrane video ────────────────────────────────────────────────────────────────

void SpikeEngine::record_membrane_video(const String &path, s64 frame_stride) {
    if (frame_stride < 1) {
        log::throw_runtime_error(*logger,
                "record_membrane_video: frame_stride must be at least 1 (got " +
                to_string(frame_stride) + ")");
    }

    // Where each neuron's `v` lives, resolved once. Every GLIF and iaf cell exposes it;
    // a cell type that does not is an error rather than a silently flat row in the video.
    membrane_offset_per_neuron.assign((usize)total_neuron_count, -1);

    for (usize index = 0; index < network_details.populations.size(); index += 1) {
        const PopulationLayout &population = network_details.populations[index];
        const CellTypeSpecification &cell_type =
                network_details.cell_types[(usize)population.cell_type_index];

        s64 slot = -1;
        for (usize candidate = 0; candidate < cell_type.state_variable_names.size();
             candidate += 1) {
            if (cell_type.state_variable_names[candidate] == "v") slot = (s64)candidate;
        }
        if (slot < 0) {
            log::throw_runtime_error(*logger,
                    "record_membrane_video: cell type '" + cell_type.name +
                    "' declares no membrane potential `v`, so there is nothing to render");
        }

        for (s64 member = 0; member < population.neuron_count; member += 1) {
            membrane_offset_per_neuron[(usize)(population.first_neuron_index + member)] =
                    layout.population_state_base[index] + slot * population.neuron_count +
                    member;
        }
    }

    membrane_video_frame_stride = frame_stride;
    membrane_frame_scratch.assign((usize)total_neuron_count, 0.0f);
    membrane_video_recorder = std::make_unique<SimulationRecorder>(path, total_neuron_count);

    logger->info("record_membrane_video: {} neurons every {} ticks -> {}",
                 total_neuron_count, frame_stride, path);
}

void SpikeEngine::write_spike_file(const String &path) const {
    ofstream file(path);
    if (!file) {
        logger->error("write_spike_file: cannot open '{}' for writing", path);
        return;
    }

    for (const RecordedSpike &spike : recorded_spikes) {
        file << setprecision(9) << spike.time_seconds << "\t" << spike.neuron_index << "\n";
    }
    logger->info("write_spike_file: wrote {} spikes to {}", recorded_spikes.size(), path);
}

// ── recording output ──────────────────────────────────────────────────────────────

void SpikeEngine::write_recordings() {
    if (membrane_video_recorder) {
        membrane_video_recorder->finish();
        logger->info("SpikeEngine: membrane video recording closed");
    }

    for (const RecordingConfig &profile : network_details.recording_profiles) {
        if (profile.output_filenames.empty()) continue;

        const String &filename = profile.output_filenames.front();
        const OutputFileFormat format = profile.file_output_format.front();

        ofstream file(filename);
        if (!file) {
            logger->error("SpikeEngine: cannot open '{}' for writing", filename);
            continue;
        }

        if (format == OutputFileFormat::SPIKE_EVENTS) {
            // The NeuroML event-file convention: one "time id" line per spike, ordered by
            // time, restricted to the neurons the EventSelections named.
            Set<s64> selected;
            for (const RecordingSelection &selection : profile.selections) {
                if (selection.neuron_index >= 0) selected.insert(selection.neuron_index);
            }

            s64 written = 0;
            for (const RecordedSpike &spike : recorded_spikes) {
                if (!selected.empty() && selected.count(spike.neuron_index) == 0) continue;

                file << setprecision(9) << spike.time_seconds << "\t"
                     << spike.neuron_index << "\n";
                written += 1;
            }
            logger->info("SpikeEngine: wrote {} spikes to {}", written, filename);
            continue;
        }

        // The NML column-matrix convention: first column time, one further column per
        // OutputColumn, in the order the file declared them.
        Vector<usize> columns;
        for (const RecordingSelection &selection : profile.selections) {
            for (usize index = 0; index < traced_selections.size(); index += 1) {
                if (traced_selections[index].quantity_path != selection.quantity_path) continue;
                columns.push_back(index);
                break;
            }
        }

        for (usize row = 0; row < recorded_trace_times.size(); row += 1) {
            file << setprecision(9) << recorded_trace_times[row];
            for (usize column : columns) {
                file << "\t" << recorded_traces[row * traced_selections.size() + column];
            }
            file << "\n";
        }
        logger->info("SpikeEngine: wrote {} rows x {} columns to {}",
                     recorded_trace_times.size(), columns.size(), filename);
    }
}

void SpikeEngine::shutdown() {
    if (!alive) return;

    logger->info("SpikeEngine: shutting down");

    gpu.release_function(kernel_function);

    // Move-assigning an empty one releases this matrix's slab, which has to happen while
    // the backend that owns it is still alive.
    weights = WeightMatrix();

    // The model slab goes the same way. Everything else the backend holds is released by
    // its own destructor, which runs after this.
    gpu.deallocate_slab(model_slab);

    alive = false;
}
