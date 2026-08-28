//
// Created by Alek Simpson on 5/30/26.
//

#include <random>
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

    return 0.0;
}

} // namespace

SpikeEngine::SpikeEngine(const String &lems_input_file, bool enable_hebbian_plasticity)
    : SpikeEngine(lems_input_file, {}, "", 1.0, 0.0, enable_hebbian_plasticity) {}

SpikeEngine::SpikeEngine(const String &lems_input_file,
                         const vector<vector<s32>> &adjacency,
                         const String &synapse_component_id,
                         f64 connection_weight,
                         f64 connection_delay_seconds,
                         bool enable_hebbian_plasticity)
    : SpikeEngine(lems_input_file, adjacency,
                  synapse_component_id.empty() ? vector<String>{}
                                               : vector<String>{synapse_component_id},
                  {}, connection_weight, connection_delay_seconds,
                  enable_hebbian_plasticity) {}

SpikeEngine::SpikeEngine(const String &lems_input_file,
                         const vector<vector<s32>> &adjacency,
                         const vector<String> &synapse_component_ids,
                         const vector<f64> &synapse_proportions,
                         f64 connection_weight,
                         f64 connection_delay_seconds,
                         bool enable_hebbian_plasticity)
    : logger(log::make_logger())
    , hebbian_plasticity_enabled(enable_hebbian_plasticity) {

    NML_Parser parser;
    if (!parser.validate_against_schema(lems_input_file)) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: " + lems_input_file + " failed lems schema validation:\n" +
                parser.last_schema_validation_errors);
    }

    network_details = parser.parse_lems(lems_input_file);

    if (!adjacency.empty()) {
        apply_topology(adjacency, synapse_component_ids, synapse_proportions,
                       connection_weight, connection_delay_seconds);
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

    const String master_kernel_source =
            generate_master_kernel(network_details, layout, hebbian_plasticity_enabled);
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
       .partition(sizeof(f32) * 2 * prototype_count * total_neuron_count,
                  EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(f32) * prototype_count * synapse_state_count * total_neuron_count,
                  EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(f32), EngineDatatype::FLOAT32, data_partitions);

    model_slab = gpu.allocate(data_partitions);

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

    // Each synapse type's state variables start at the value its OnStart gives them
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

    // negative means has never fired
    s64 *last_spiked_data = last_spiked.get_contents_as<s64>();
    std::fill(last_spiked_data, last_spiked_data + total_neuron_count, NEVER_SPIKED_TICK);

    // parameter rows, one per prototype, in the column order the type declared.
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

void SpikeEngine::build_weight_matrix() {
    const Vector<Vector<s32>> network = build_adjacency_list(network_details);

    // rank -1 means "derive it from what the projections below actually contain"
    weights = WeightMatrix(gpu, network, /*rank=*/-1, /*check_indexing=*/true,
                           /*max_neighbor_count=*/-1, /*weight_seed=*/(s64)simulation_seed,
                           correction_ceiling_fraction, weight_fit_rank_budget);

    // room for updates to queue into, on top of whatever the fit turns out to need. Zero
    // when nothing writes updates, so a model with no plasticity and an exact fit
    // allocates no correction layer at all.
    weights.plasticity_reserve_entries = hebbian_plasticity_enabled 
        ? DEFAULT_PLASTICITY_DELTA_CAPACITY 
        : 0;

    // The same run coalescing the codegen bakes into the kernel
    Vector<s64> edge_ordinal;
    Vector<s64> edge_count;
    Vector<s64> synapse_prototype;
    Vector<f32> weight;
    Vector<s32> delay_ticks;
    collect_projection_runs(network_details, edge_ordinal, edge_count,
                            synapse_prototype, weight, delay_ticks);

    Vector<s32> synapse_prototype_narrow;
    synapse_prototype_narrow.reserve(synapse_prototype.size());
    for (s64 prototype_index : synapse_prototype) {
        synapse_prototype_narrow.push_back((s32)prototype_index);
    }

    weights.declare_projections(edge_ordinal, edge_count, synapse_prototype_narrow,
                                weight, delay_ticks);

    if (hebbian_plasticity_enabled && weights.total_edge_count > 0) {
        plasticity_target_root_mean_square = weights.neighbor_weight_stats().root_mean_square;
    }

    logger->debug("SpikeEngine: weight matrix built — {} nodes, {} edges, {} projection runs, "
                  "rank {}, worst weight error {:.3e}",
                  weights.node_count, weights.total_edge_count, edge_ordinal.size(),
                  weights.rank, weights.measured_weight_fit_error);
}

f64 SpikeEngine::default_spike_amplitude_for(s64 neuron_index) const {
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

        f64 membrane_floor = starting_value_for(cell_type.dynamics, membrane_name,
                                                cell_type.parameter_names,
                                                prototype.starting_parameters,
                                                cell_type.name);

        for (const DynamicsInstruction &instruction : cell_type.dynamics) {
            if (instruction.stage != DynamicsStage::Reset) continue;
            if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
            if (instruction.target != membrane_name) continue;

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
                                 const vector<String> &synapse_component_ids,
                                 const vector<f64> &synapse_proportions,
                                 f64 connection_weight,
                                 f64 connection_delay_seconds) {
    const s64 neuron_count = (s64)network_details.neurons.size();
    if ((s64)adjacency.size() != neuron_count) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: the topology has " + to_string(adjacency.size()) +
                " rows but the model's populations declare " + to_string(neuron_count) +
                " neurons; one row per neuron is required");
    }
    if (synapse_component_ids.empty()) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: the topology names no synapse, so its edges would carry "
                "nothing");
    }

    // An instance id may be document-scoped ("net.syn"), so match the trailing name too.
    Vector<s64> prototype_index_of_choice;
    String available;
    for (const ComponentPrototype &prototype : network_details.synapse_prototypes) {
        available += (available.empty() ? "" : ", ") + prototype.instance_id;
    }

    for (const String &wanted : synapse_component_ids) {
        s64 prototype_index = -1;
        for (usize index = 0; index < network_details.synapse_prototypes.size(); index += 1) {
            const String &instance_id = network_details.synapse_prototypes[index].instance_id;
            const usize separator = instance_id.rfind('.');
            const String leaf = separator == String::npos ? instance_id
                                                          : instance_id.substr(separator + 1);
            if (instance_id == wanted || leaf == wanted) prototype_index = (s64)index;
        }
        if (prototype_index < 0) {
            log::throw_runtime_error(*logger,
                    "SpikeEngine: the topology names synapse '" + wanted +
                    "', which the model does not declare. It declares: " +
                    (available.empty() ? "no synapses at all" : available));
        }
        prototype_index_of_choice.push_back(prototype_index);
    }

    const usize choice_count = prototype_index_of_choice.size();

    Vector<f64> cumulative_share(choice_count, 0.0);
    if (synapse_proportions.empty()) {
        for (usize index = 0; index < choice_count; index += 1) {
            cumulative_share[index] = (f64)(index + 1) / (f64)choice_count;
        }
    } else {
        if (synapse_proportions.size() != choice_count) {
            log::throw_runtime_error(*logger,
                    "SpikeEngine: the topology gives " + to_string(synapse_proportions.size()) +
                    " proportions for " + to_string(choice_count) + " synapses; there must "
                    "be one proportion per synapse, or none at all for an equal share each");
        }

        f64 total_share = 0.0;
        for (f64 share : synapse_proportions) {
            if (share < 0.0) {
                log::throw_runtime_error(*logger,
                        "SpikeEngine: the topology gives a negative synapse proportion (" +
                        to_string(share) + ")");
            }
            total_share += share;
        }
        if (total_share <= 0.0) {
            log::throw_runtime_error(*logger,
                    "SpikeEngine: the topology's synapse proportions sum to zero, so no "
                    "synapse could ever be chosen");
        }

        f64 running_share = 0.0;
        for (usize index = 0; index < choice_count; index += 1) {
            running_share += synapse_proportions[index] / total_share;
            cumulative_share[index] = running_share;
        }
    }

    const u64 assignment_seed = network_details.random_seed.has_value()
                                        ? *network_details.random_seed
                                        : 0x5CC0DEu;
    std::mt19937_64 generator(assignment_seed);
    std::uniform_real_distribution<f64> uniform(0.0, 1.0);

    const s64 delay_ticks = std::max<s64>(
            1, (s64)std::llround(connection_delay_seconds / network_details.step_dt));

    synapse_choice_per_neuron.assign((usize)neuron_count, 0);
    Vector<s64> edges_per_choice(choice_count, 0);
    Vector<s64> neurons_per_choice(choice_count, 0);

    s64 edge_count = 0;
    for (s64 source = 0; source < neuron_count; source += 1) {
        // One draw per neuron, not per edge. Everything leaving this cell carries the
        // same synapse, which is what makes it an excitatory or an inhibitory cell.
        usize choice = choice_count - 1;
        if (choice_count > 1) {
            const f64 draw = uniform(generator);
            for (usize candidate = 0; candidate < choice_count; candidate += 1) {
                if (draw >= cumulative_share[candidate]) continue;
                choice = candidate;
                break;
            }
        }
        synapse_choice_per_neuron[(usize)source] = (s32)choice;
        neurons_per_choice[choice] += 1;

        const s64 prototype_index = prototype_index_of_choice[choice];
        const s64 type_index =
                network_details.synapse_prototypes[(usize)prototype_index].type_index;

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
            edges_per_choice[choice] += 1;
        }
    }

    String assignment;
    for (usize choice = 0; choice < choice_count; choice += 1) {
        assignment += (assignment.empty() ? "" : ", ") + synapse_component_ids[choice] +
                      " on " + to_string(neurons_per_choice[choice]) + " cells (" +
                      to_string(edges_per_choice[choice]) + " edges)";
    }

    logger->info("SpikeEngine: topology applied — {} neurons, {} edges, weight {}, "
                 "delay {} ticks; {}",
                 neuron_count, edge_count, connection_weight, delay_ticks, assignment);
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

void SpikeEngine::apply_stimulus(s64 tick) {
    f32 *input_data = static_cast<f32 *>(network_inputs.get_contents());
    const s64 row_base = (tick % 2) * total_neuron_count;

    for (usize index = 0; index < continuous_injection_targets.size(); index += 1) {
        if (tick < continuous_injection_start_ticks[index]) continue;
        if (tick >= continuous_injection_end_ticks[index]) continue;

        input_data[row_base + continuous_injection_targets[index]] +=
                continuous_injection_amplitudes[index];
    }

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

EnginePointer SpikeEngine::resolve_edge_plane(const EnginePointer &plane) const {
    return plane.is_empty() ? empty_edge_plane : plane;
}

void SpikeEngine::step_simulation(s64 tick) {
    apply_stimulus(tick);

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
        resolve_edge_plane(weights.sparse_delta_row_start),      // 27
        resolve_edge_plane(weights.sparse_delta_edge_ordinal),   // 28
        resolve_edge_plane(weights.sparse_delta_value),          // 29
    };

    const s32 plasticity_capacity_argument = (s32)weights.sparse_delta_capacity;
    if (hebbian_plasticity_enabled) {
        parameters.push_back(resolve_edge_plane(weights.pending_delta_edge_ordinal)); // 30
        parameters.push_back(resolve_edge_plane(weights.pending_delta_value));        // 31
        parameters.push_back(resolve_edge_plane(weights.pending_delta_count));        // 32
        parameters.push_back(inline_scalar_argument(plasticity_capacity_argument));   // 33
    }

    if (!gpu.run_function(kernel_function, parameters, total_neuron_count)) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: tick " + to_string(tick) + " failed on the GPU");
    }

    if (hebbian_plasticity_enabled &&
        plasticity_fold_every_n_ticks > 0 &&
        (tick + 1) % plasticity_fold_every_n_ticks == 0) {

        weights.compact_pending_deltas();

        // re optimise the basis
        if (weights.is_refit_due()) {
            weights.refit();

            if (plasticity_target_root_mean_square >= 0.0f) {
                weights.scale_neighbor_weights_to_root_mean_square(
                        plasticity_target_root_mean_square);
            }
        }
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

void SpikeEngine::record_membrane_video(const String &path, s64 frame_stride) {
    if (frame_stride < 1) {
        log::throw_runtime_error(*logger,
                "record_membrane_video: frame_stride must be at least 1 (got " +
                to_string(frame_stride) + ")");
    }

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

    gpu.deallocate_slab(model_slab);

    alive = false;
}
