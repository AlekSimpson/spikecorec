//
// Created by Alek Simpson on 5/30/26.
//

#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/units.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

SpikeEngine::SpikeEngine(const String &lems_input_file, bool enable_hebbian_plasticity, f32 fit_tolerance)
    : SpikeEngine(lems_input_file, {}, "", 1.0, 0.0, enable_hebbian_plasticity, fit_tolerance) {}

SpikeEngine::SpikeEngine(const String &lems_input_file,
                         const vector<vector<s32>> &adjacency,
                         const String &synapse_component_id,
                         f64 connection_weight,
                         f64 connection_delay_seconds,
                         bool enable_hebbian_plasticity,
                         f32 fit_tolerance)
    : SpikeEngine(lems_input_file, adjacency,
                  synapse_component_id.empty() ? vector<String>{}
           : vector<String>{synapse_component_id},
                  {}, connection_weight, connection_delay_seconds,
                  enable_hebbian_plasticity, fit_tolerance) {}

SpikeEngine::SpikeEngine(const String &lems_input_file,
                         const vector<vector<s32>> &adjacency,
                         const vector<String> &synapse_component_ids,
                         const vector<f64> &synapse_proportions,
                         f64 connection_weight,
                         f64 connection_delay_seconds,
                         bool enable_hebbian_plasticity,
                         f32 fit_tolerance)
    : logger(log::make_logger())
    , hebbian_plasticity_enabled(enable_hebbian_plasticity)
    , fit_tolerance(fit_tolerance) {

    if (!context.validate_lems_schema(lems_input_file)) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: " + lems_input_file + " failed lems schema validation:\n" +
                context.last_schema_validation_errors);
    }

    context.parse(lems_input_file);

    if (!adjacency.empty()) {
        apply_topology(adjacency, synapse_component_ids, synapse_proportions,
                       connection_weight, connection_delay_seconds);
    }

    total_neuron_count = context.simulation.total_neuron_count;
    spike_history_row_count = spike_history_length(context);
    lifetime = context.simulation.total_tick_count;
    step_dt = context.simulation.step_dt;
    if (context.simulation.random_seed.has_value()) {
        simulation_seed = *context.simulation.random_seed;
    }
    random_generator = RandomGenerator(simulation_seed);

    if (total_neuron_count == 0) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: " + lems_input_file + " defines no neurons — check that the "
                "Simulation's target names a network with at least one population");
    }
    if (step_dt <= 0.0f) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: " + lems_input_file + " gives the Simulation no usable step "
                "(parsed " + to_string(context.simulation.step_dt) + " s)");
    }

    Codegen compiler(context, &gpu);
    compiler.check_cell_inputs();

    compiler.allocate_cell_model_memory();
    cell_state = compiler.data_partitions[0];
    network_inputs = compiler.data_partitions[1];
    spike_history = compiler.data_partitions[2];
    last_spiked = compiler.data_partitions[3];
    empty_edge_plane = compiler.data_partitions[4];
    model_pointer = compiler.data_partitions[5];

    initialize_model_buffers();
    compiler.initialize_cell_state(random_generator);

    build_weight_matrix(compiler.edge_plane_count(), compiler.updated_edge_plane_count());

    weights.plasticity_reserve_entries = 0;
    if (hebbian_plasticity_enabled) {
        weights.plasticity_reserve_entries = DEFAULT_PLASTICITY_DELTA_CAPACITY;

        if (weights.total_edge_count > 0) {
            plasticity_target_root_mean_square = weights.neighbor_weight_stats().root_mean_square;
        }
        logger->warn("SpikeEngine: plasticity is enabled, but the generated kernel stages no "
                     "plasticity deltas yet, so weights will not change");
    }
    weights.resize_delta_capacity(weights.delta_capacity_for_threshold(refit_occupancy_threshold_fraction));

    collect_stimulus();

    compiler.translate_kernel_code();

    // Refilled before every tick. At least one slot, so there is always a buffer to bind.
    random_values_count = compiler.random_values_count;
    Vector<EnginePointer> random_partitions;
    gpu.partition((u64)std::max<s64>(1, random_values_count) * sizeof(f32), EngineDatatype::FLOAT32, random_partitions);
    random_values_pointer = gpu.allocate(random_partitions);
    random_values = random_partitions[0];

    // Fixed once the weight matrix is built, so baked in rather than bound. Metal also has
    // only 31 buffer slots for everything master_step takes.
    compiler.bake_kernel_constant("branching_factor", weights.k2tree.branching_factor);
    compiler.bake_kernel_constant("superblock_size_words", weights.k2tree.superblock_size_words);
    compiler.bake_kernel_constant("padded_node_count", weights.k2tree.padded_node_count);
    compiler.bake_kernel_constant("tree_height", weights.k2tree.tree_height);
    compiler.bake_kernel_constant("internal_bit_count", weights.k2tree.internal_bit_count);
    compiler.bake_kernel_constant("pending_delta_capacity", weights.pending_delta_capacity);
    compiler.bake_kernel_constant("projection_run_count", projection_run_count);
    synapse_active_ticks = compiler.synapse_active_ticks();
    compiler.bake_kernel_constant("synapse_active_ticks", synapse_active_ticks);
    master_kernel_source = compiler.compile();
    logger->debug("SpikeEngine: generated master kernel, {} bytes", master_kernel_source.size());

    Optional<EngineFunction> result = gpu.create_function("master_step", master_kernel_source);
    if (!result.has_value()) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: the generated master kernel failed to compile — see the log above "
                "for the compiler's own diagnostic");
    }
    kernel_function = *result;

    kernel_parameter_names = compiler.kernel_parameter_names();
    register_kernel_arguments();

    spike_counts_per_neuron.assign((usize)total_neuron_count, 0);

    // A column's neuron_index is resolve_path's cell_state index of the variable itself.
    for (const RecordingConfig &profile : context.simulation.recording_profiles) {
        for (const RecordingSelection &selection : profile.selections) {
            if (!selection.event_port.empty()) continue;
            if (selection.neuron_index < 0) continue;
            traced_selections.push_back(selection);
        }
    }

    alive = true;
    logger->info("SpikeEngine constructed from {}: {} neurons, {} edges, {} cell types, "
                 "{} synapse components, dt={} s, {} ticks, spike history {} rows",
                 lems_input_file, total_neuron_count, weights.total_edge_count,
                 compiler.component_type_templates.size(),
                 context.simulation.synapse_instances.size(),
                 step_dt, lifetime, spike_history_row_count);
}

void SpikeEngine::initialize_model_buffers() {
    std::memset(network_inputs.get_contents(), 0, network_inputs.total_bytes);
    std::memset(spike_history.get_contents(), 0, spike_history.total_bytes);
    std::memset(empty_edge_plane.get_contents(), 0, empty_edge_plane.total_bytes);
    std::fill_n(last_spiked.get_contents_as<s64>(), total_neuron_count, NEVER_SPIKED_TICK);
}

void SpikeEngine::build_weight_matrix(s64 matrix_count, s64 updated_plane_count) {
    Vector<Vector<NML_NetworkEdge>> edges_by_source((usize)total_neuron_count);
    for (const Vector<NML_NetworkEdge> &row : context.simulation.network_data.list) {
        for (const NML_NetworkEdge &edge : row) edges_by_source[(usize)edge.parent].push_back(edge);
    }

    // Synapse components are numbered in the order the network first uses them.
    UnorderedMap<String, s32> synapse_prototype_indices;
    for (const NML_ComponentInstance &synapse : context.simulation.synapse_instances) {
        synapse_prototype_indices.emplace(synapse.id, (s32)synapse_prototype_indices.size());
    }

    // The canonical edge order is each source's targets ascending, which is the order the
    // k^2-tree walks a row in. Equal neighbouring edges share one projection run.
    vector<vector<s32>> network((usize)total_neuron_count);
    Vector<s64> first_edge_ordinal;
    Vector<s64> edge_count;
    Vector<s32> synapse_prototype;
    Vector<f32> weight;
    Vector<s32> delay_ticks;

    s64 ordinal = 0;
    for (usize source = 0; source < edges_by_source.size(); source += 1) {
        Vector<NML_NetworkEdge> &edges = edges_by_source[source];
        std::stable_sort(edges.begin(), edges.end(),
                         [](const NML_NetworkEdge &left, const NML_NetworkEdge &right) {
                             return left.child < right.child;
                         });

        for (usize index = 0; index < edges.size(); index += 1) {
            const NML_NetworkEdge &edge = edges[index];
            if (index > 0 && edges[index - 1].child == edge.child) {
                logger->warn("SpikeEngine: neuron {} connects to neuron {} more than once; "
                             "keeping the first connection", source, edge.child);
                continue;
            }

            auto prototype = synapse_prototype_indices.find(edge.component_id);
            if (prototype == synapse_prototype_indices.end()) {
                log::throw_runtime_error(*logger,
                        "SpikeEngine: the connection " + to_string(source) + " -> " +
                        to_string(edge.child) + " names no synapse, so there is nothing to "
                        "carry its current");
            }

            network[source].push_back((s32)edge.child);

            // The engine's synaptic latency is one tick, so a connection that names no
            // delay still arrives a tick later rather than instantaneously.
            const s32 edge_delay = (s32)std::max<s64>(1, edge.delay_ticks);
            const bool extends_current_run =
                    !first_edge_ordinal.empty() &&
                    synapse_prototype.back() == prototype->second &&
                    weight.back() == edge.weight &&
                    delay_ticks.back() == edge_delay &&
                    first_edge_ordinal.back() + edge_count.back() == ordinal;

            if (extends_current_run) {
                edge_count.back() += 1;
            } else {
                first_edge_ordinal.push_back(ordinal);
                edge_count.push_back(1);
                synapse_prototype.push_back(prototype->second);
                weight.push_back(edge.weight);
                delay_ticks.push_back(edge_delay);
            }
            ordinal += 1;
        }
    }

    // Every per-edge variable's starting value, per run: the connection's weight and delay,
    // then each of its synapse's StateVariables at its OnStart value.
    Vector<Vector<f32>> starting_state_per_prototype;
    for (const NML_ComponentInstance &synapse : context.simulation.synapse_instances) {
        const UnorderedMap<String, f64> values = starting_values(context, synapse);
        Vector<f32> &starting_state = starting_state_per_prototype.emplace_back();
        for (const String &name : synapse.component_type->state_variable_names) {
            starting_state.push_back((f32)values.at(name));
        }
    }
    Vector<Vector<f32>> initial_values((usize)matrix_count);
    for (usize run_index = 0; run_index < synapse_prototype.size(); run_index += 1) {
        const Vector<f32> &starting_state = starting_state_per_prototype[(usize)synapse_prototype[run_index]];
        for (s64 plane = 0; plane < matrix_count; plane += 1) {
            f32 value = 0.0f;
            if (plane == WeightMatrix::WEIGHT_PLANE) {
                value = weight[run_index];
            } else if (plane == WeightMatrix::DELAY_PLANE) {
                value = (f32)delay_ticks[run_index];
            } else if ((usize)(plane - WeightMatrix::FIRST_STATE_VARIABLE_PLANE) < starting_state.size()) {
                value = starting_state[(usize)(plane - WeightMatrix::FIRST_STATE_VARIABLE_PLANE)];
            }
            initial_values[(usize)plane].push_back(value);
        }
    }

    weights = WeightMatrix(gpu, network, /*rank=*/-1, /*check_indexing=*/true,
                           /*max_neighbor_count=*/-1, /*weight_seed=*/(s64)simulation_seed,
                           matrix_count, updated_plane_count);
    weights.fit_tolerance = fit_tolerance;
    if (ordinal > 0) weights.declare_projections(first_edge_ordinal, edge_count, synapse_prototype, initial_values);

    // The runs on the device, for the kernel's per-edge prototype lookup.
    Vector<EnginePointer> run_partitions;
    gpu.partition(first_edge_ordinal.size() * sizeof(s64), EngineDatatype::SIGNED64, run_partitions)
       .partition(synapse_prototype.size() * sizeof(s32), EngineDatatype::SIGNED32, run_partitions);
    projection_run_pointer = gpu.allocate(run_partitions);
    projection_first_edge_ordinal = run_partitions[0];
    projection_synapse_prototype = run_partitions[1];
    if (!first_edge_ordinal.empty()) {
        std::copy(first_edge_ordinal.begin(), first_edge_ordinal.end(), projection_first_edge_ordinal.get_contents_as<s64>());
        std::copy(synapse_prototype.begin(), synapse_prototype.end(), projection_synapse_prototype.get_contents_as<s32>());
    }
    projection_run_count = (s32)first_edge_ordinal.size();

    logger->debug("SpikeEngine: weight matrix built — {} nodes, {} edges, {} projection runs, rank {}",
                  weights.node_count, weights.total_edge_count, first_edge_ordinal.size(), weights.rank);
}

void SpikeEngine::register_kernel_arguments() {
    kernel_argument_sources = {
        {"rank_float4_stride",        [this] { return inline_scalar_argument(rank_float4_stride_argument); }},
        {"sparse_delta_capacity",     [this] { return inline_scalar_argument(sparse_delta_capacity_argument); }},
        {"cell_state",                [this] { return cell_state; }},
        {"network_inputs",            [this] { return network_inputs; }},
        {"spike_history",             [this] { return spike_history; }},
        {"last_spiked",               [this] { return last_spiked; }},
        {"internal_node_words",       [this] { return resolve_edge_plane(weights.k2tree.internal_node_words); }},
        {"leaf_node_words",           [this] { return resolve_edge_plane(weights.k2tree.leaf_node_words); }},
        {"rank_superblock_table",     [this] { return resolve_edge_plane(weights.k2tree.rank_superblock_table); }},
        {"rank_subblock_table",       [this] { return resolve_edge_plane(weights.k2tree.rank_subblock_table); }},
        {"basis_u",                   [this] { return resolve_edge_plane(weights.U_matrix); }},
        {"basis_v",                   [this] { return resolve_edge_plane(weights.V_matrix); }},
        {"edge_coefficients",         [this] { return resolve_edge_plane(weights.coefficients); }},
        {"edge_row_offset",           [this] { return resolve_edge_plane(weights.edge_row_offset); }},
        {"sparse_delta_row_start",    [this] { return resolve_edge_plane(weights.sparse_delta_row_start); }},
        {"sparse_delta_edge_ordinal", [this] { return resolve_edge_plane(weights.sparse_delta_edge_ordinal); }},
        {"sparse_delta_value",        [this] { return resolve_edge_plane(weights.sparse_delta_value); }},
        {"pending_delta_edge_ordinal", [this] { return resolve_edge_plane(weights.pending_delta_edge_ordinal); }},
        {"pending_delta_value",       [this] { return resolve_edge_plane(weights.pending_delta_value); }},
        {"pending_delta_matrix_index", [this] { return resolve_edge_plane(weights.pending_delta_matrix_index); }},
        {"pending_delta_count",       [this] { return resolve_edge_plane(weights.pending_delta_count); }},
        {"projection_first_edge_ordinal", [this] { return resolve_edge_plane(projection_first_edge_ordinal); }},
        {"projection_synapse_prototype",  [this] { return resolve_edge_plane(projection_synapse_prototype); }},
        {"random_values",             [this] { return random_values; }},
    };

    for (const String &name : kernel_parameter_names) {
        if (name == "tick" || kernel_argument_sources.count(name) != 0) continue;
        log::throw_runtime_error(*logger,
                "SpikeEngine: master_step takes '" + name + "', which the engine has nothing to bind to");
    }
}

const NML_ComponentInstance *SpikeEngine::population_of_neuron(s64 neuron_index, s64 &first_neuron) const {
    first_neuron = 0;
    for (const NML_ComponentInstance *population : network_populations(context)) {
        const s64 population_size = context.get_population_size(population);
        if (neuron_index >= first_neuron && neuron_index < first_neuron + population_size) return population;
        first_neuron += population_size;
    }
    return nullptr;
}

s64 SpikeEngine::cell_memory_index_of(s64 neuron_index, const String &variable_name) const {
    s64 first_neuron = 0;
    const NML_ComponentInstance *population = population_of_neuron(neuron_index, first_neuron);
    if (!population) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: neuron index " + to_string(neuron_index) + " is outside every population");
    }

    const NML_ComponentType &cell_type = *population_cell(context, *population).component_type;
    const Vector<String> &variable_names = cell_type.state_variable_names;
    auto variable = std::find(variable_names.begin(), variable_names.end(), variable_name);
    if (variable == variable_names.end()) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: neuron " + to_string(neuron_index) + " is a '" + cell_type.name +
                "', which declares no state variable '" + variable_name + "'");
    }

    return context.simulation.population_base_indices.at(population->id) +
           (neuron_index - first_neuron) * (s64)variable_names.size() +
           (s64)(variable - variable_names.begin());
}

f64 SpikeEngine::default_spike_amplitude_for(s64 neuron_index) const {
    constexpr f64 THRESHOLD_OVERSHOOT = 1.05;

    s64 first_neuron = 0;
    const NML_ComponentInstance *population = population_of_neuron(neuron_index, first_neuron);
    if (!population) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: spike train targets neuron " + to_string(neuron_index) +
                ", which is outside every population");
    }
    const NML_ComponentInstance &cell = population_cell(context, *population);
    const NML_ComponentType &cell_type = *cell.component_type;
    const UnorderedMap<String, f64> values = starting_values(context, cell);

    f64 capacitance = 0.0;
    for (const NML_ComponentType *type = &cell_type; type; type = type->extends) {
        for (const NML_Node *element : children_of(type->source_node)) {
            const NML_Tag &tag = element->body;
            if (tag.tag_type != NML_DeclarationType::Parameter) continue;
            if (tag.get_attribute("dimension") != "capacitance") continue;
            if (cell_type.find_declaration(tag.namespace_key()) != element) continue;

            auto value = values.find(tag.get_attribute("name"));
            if (value != values.end()) capacitance = value->second;
        }
    }
    if (capacitance <= 0.0) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: a spike train targets a '" + cell_type.name +
                "', which declares no parameter of dimension capacitance, so there is "
                "no way to work out what one event should inject. Give the input an "
                "amplitude");
    }

    // The spiking OnCondition is the one with an EventOut; its test should read
    // "membrane > threshold".
    String spike_test;
    for (const NML_DynamicsExpression &entry : cell_type.dynamics) {
        if (entry.source_tag != NML_DeclarationType::EventOut || entry.condition.empty()) continue;
        spike_test = entry.condition;
        break;
    }

    String membrane_name;
    f64 threshold = 0.0;
    bool threshold_found = false;
    if (!spike_test.empty()) {
        std::unique_ptr<LemsParseNode> test(parse_lems_expression(spike_test, cell_type.name));
        const auto *comparison = dynamic_cast<const BinaryNode<LemsParseBody> *>(test.get());
        const String comparison_operator = test->body.token.lexeme;
        if (comparison && test->body.syntax_type == LemsNodeSubtype::OPERATOR &&
            (comparison_operator == ">" || comparison_operator == ">=") &&
            comparison->left->body.syntax_type == LemsNodeSubtype::IDENTIFIER) {
            membrane_name = comparison->left->body.token.lexeme;
            try {
                threshold = evaluate_lems(comparison->right, values, cell_type.name);
                threshold_found = true;
            } catch (const std::runtime_error &) {
                // not foldable to a starting value
            }
        }
    }
    if (membrane_name.empty()) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: a spike train targets a '" + cell_type.name +
                "', whose spike condition is not a comparison this can read, so there "
                "is no threshold to aim at. Give the input an amplitude");
    }

    // The lowest the membrane starts or resets to.
    f64 resting = values.count(membrane_name) ? values.at(membrane_name) : 0.0;
    for (const NML_DynamicsExpression &entry : cell_type.dynamics) {
        if (entry.source_tag != NML_DeclarationType::StateAssignment) continue;
        if (entry.condition != spike_test || entry.target != membrane_name) continue;

        try {
            resting = std::min(resting, evaluate_lems(entry.expression, values, cell_type.name));
        } catch (const std::runtime_error &) {
            // not foldable; the OnStart value stands
        }
    }

    if (!threshold_found || threshold <= resting) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: a spike train targets a '" + cell_type.name +
                "', whose threshold resolves to no value above its starting membrane "
                "potential, so no finite current would make it fire. Give the input an amplitude");
    }

    return THRESHOLD_OVERSHOOT * capacitance * (threshold - resting) / step_dt;
}

// Replaces the document's connections with adjacency, where adjacency[n] lists neuron n's
// targets. Every edge carries connection_weight and connection_delay_seconds. The synapses
// are handed out in order, each to its proportion of the edges (equal shares when no
// proportions are given).
void SpikeEngine::apply_topology(const vector<vector<s32>> &adjacency,
                                 const vector<String> &synapse_component_ids,
                                 const vector<f64> &synapse_proportions,
                                 f64 connection_weight,
                                 f64 connection_delay_seconds) {
    NML_Context::NML_SimulationContext &simulation = context.simulation;
    const s64 neuron_count = simulation.total_neuron_count;

    if ((s64)adjacency.size() > neuron_count) {
        log::throw_runtime_error(*logger, "apply_topology: the adjacency list has " + to_string(adjacency.size()) +
                                 " nodes but the network has " + to_string(neuron_count) + " neurons");
    }
    for (usize source = 0; source < adjacency.size(); source += 1) {
        for (s32 target : adjacency[source]) {
            if (target >= 0 && target < neuron_count) continue;
            log::throw_runtime_error(*logger, "apply_topology: neuron " + to_string(source) + " connects to " +
                                     to_string(target) + ", outside the network's " + to_string(neuron_count) + " neurons");
        }
    }
    if (synapse_component_ids.empty()) {
        log::throw_runtime_error(*logger, "apply_topology: no synapse to carry the edges' current");
    }
    if (!synapse_proportions.empty() && synapse_proportions.size() != synapse_component_ids.size()) {
        log::throw_runtime_error(*logger, "apply_topology: one proportion per synapse is required");
    }

    // Prototype numbering follows this order, in build_weight_matrix and in codegen.
    simulation.synapse_instances.clear();
    for (const String &synapse_id : synapse_component_ids) {
        const NML_ComponentInstance *synapse = context.find_instance(synapse_id);
        if (!synapse) log::throw_runtime_error(*logger, "apply_topology: no synapse '" + synapse_id + "'");
        simulation.synapse_instances.push_back(*synapse);
    }

    s64 edge_count = 0;
    for (const vector<s32> &targets : adjacency) edge_count += (s64)targets.size();

    // Edge ordinals [first_edge_of_synapse[k], first_edge_of_synapse[k + 1]) use synapse k.
    f64 proportion_total = 0.0;
    for (usize index = 0; index < synapse_component_ids.size(); index += 1) {
        proportion_total += synapse_proportions.empty() ? 1.0 : synapse_proportions[index];
    }
    Vector<s64> first_edge_of_synapse = {0};
    f64 running_proportion = 0.0;
    for (usize index = 0; index < synapse_component_ids.size(); index += 1) {
        running_proportion += synapse_proportions.empty() ? 1.0 : synapse_proportions[index];
        first_edge_of_synapse.push_back(std::llround(running_proportion / proportion_total * (f64)edge_count));
    }

    const s64 delay_ticks = units::seconds_to_ticks(connection_delay_seconds, simulation.step_dt);
    simulation.network_data = AdjacencyList((int)neuron_count, 0);
    simulation.total_edge_count = 0;
    simulation.maximum_edge_delay = delay_ticks;

    s64 edge_ordinal = 0;
    usize synapse_index = 0;
    for (usize source = 0; source < adjacency.size(); source += 1) {
        for (s32 target : adjacency[source]) {
            if (edge_ordinal >= first_edge_of_synapse[synapse_index + 1]) synapse_index += 1;

            NML_NetworkEdge edge;
            edge.component_id = synapse_component_ids[synapse_index];
            edge.weight = (f32)connection_weight;
            edge.delay_ticks = delay_ticks;
            edge.parent = (s64)source;
            edge.child = target;
            simulation.network_data.add(edge);

            simulation.total_edge_count += 1;
            edge_ordinal += 1;
        }
    }
}

void SpikeEngine::collect_stimulus() {
    for (const SimulationInputConfig &profile : context.simulation.input_profiles) {
        for (const InputTarget &target : profile.targets) {
            // Targets are resolve_path cell-memory indices; network_inputs is per neuron.
            const s64 target_neuron = context.neuron_index_of(target.neuron_index);
            if (target_neuron < 0) continue;

            if (profile.continuous_current_injection) {
                continuous_injection_targets.push_back(target_neuron);
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
                    : default_spike_amplitude_for(target_neuron);

            ScheduledSpikeTrain train;
            train.neuron_index = target_neuron;
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

    // This tick's draw for every random() slot.
    f32 *random_value_data = random_values.get_contents_as<f32>();
    for (s64 index = 0; index < random_values_count; index += 1) {
        random_value_data[index] = (f32)random_generator.uniform();
    }

    rank_float4_stride_argument = (s32)weights.rank_float4_stride;
    sparse_delta_capacity_argument = weights.sparse_delta_capacity;

    Vector<EnginePointer> parameters;
    parameters.reserve(kernel_parameter_names.size());
    for (const String &name : kernel_parameter_names) {
        parameters.push_back(name == "tick" ? inline_scalar_argument(tick) : kernel_argument_sources.at(name)());
    }

    if (!gpu.run_function(kernel_function, parameters, total_neuron_count)) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: tick " + to_string(tick) + " failed on the GPU");
    }

    // Updates queued this tick by edges with no entry in S yet; merge them so the next tick
    // reads them.
    const bool plasticity_fold_due = hebbian_plasticity_enabled && plasticity_fold_every_n_ticks > 0 &&
                                     (tick + 1) % plasticity_fold_every_n_ticks == 0;
    if (weights.updated_plane_count > 0 || plasticity_fold_due) weights.compact_pending_deltas();

    // Fold S into the basis and empty it. This runs between dispatches, so no kernel reads the
    // matrix while it changes.
    const bool refit_scheduled = refit_every_n_ticks > 0 && (tick + 1) % refit_every_n_ticks == 0;
    const bool spaced_out = tick - last_refit_tick >= minimum_ticks_between_refits;
    if (spaced_out && (refit_scheduled || weights.is_refit_due(refit_occupancy_threshold_fraction))) {
        weights.fit_tolerance = fit_tolerance;
        weights.refit();
        last_refit_tick = tick;

        if (hebbian_plasticity_enabled && plasticity_target_root_mean_square >= 0.0f) {
            weights.scale_neighbor_weights_to_root_mean_square(plasticity_target_root_mean_square);
        }
        // S is empty now, so it goes back to the size the threshold asks for.
        weights.resize_delta_capacity(weights.delta_capacity_for_threshold(refit_occupancy_threshold_fraction));
    }

    record_tick(tick);
}

void SpikeEngine::set_refit_occupancy_threshold_fraction(f32 fraction) {
    refit_occupancy_threshold_fraction = fraction;
    s64 largest_plane = 0;
    for (s64 entry_count : weights.sparse_delta_entry_count) largest_plane = std::max(largest_plane, entry_count);
    weights.resize_delta_capacity(std::max(weights.delta_capacity_for_threshold(fraction), largest_plane));
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
                          (tick % spike_history_row_count) * total_neuron_count;

    for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
        if (spike_row[neuron_index] == 0) continue;

        spike_counts_per_neuron[(usize)neuron_index] += 1;
        recorded_spikes.push_back(RecordedSpike{(f64)tick * step_dt, neuron_index});
    }

    const f32 *cell_state_data = static_cast<const f32 *>(cell_state.get_contents());

    if (membrane_video_recorder && tick % membrane_video_frame_stride == 0) {
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
            membrane_frame_scratch[(usize)neuron_index] =
                    cell_state_data[membrane_offset_per_neuron[(usize)neuron_index]];
        }
        membrane_video_recorder->record_frame(membrane_frame_scratch.data(),
                                              total_neuron_count);
    }

    if (traced_selections.empty()) return;

    recorded_trace_times.push_back((f64)tick * step_dt);
    for (const RecordingSelection &selection : traced_selections) {
        recorded_traces.push_back(cell_state_data[selection.neuron_index]);
    }
}

f32 SpikeEngine::read_state_variable(s64 neuron_index, const String &variable_name) const {
    return static_cast<const f32 *>(cell_state.get_contents())[cell_memory_index_of(neuron_index, variable_name)];
}

s64 SpikeEngine::synapse_state_variable_plane(const String &synapse_component_id,
                                              const String &variable_name) const {
    for (const NML_ComponentInstance &synapse : context.simulation.synapse_instances) {
        if (synapse.id != synapse_component_id) continue;

        const Vector<String> &names = synapse.component_type->state_variable_names;
        for (usize index = 0; index < names.size(); index += 1) {
            if (names[index] == variable_name) return WeightMatrix::FIRST_STATE_VARIABLE_PLANE + (s64)index;
        }
        log::throw_runtime_error(*logger, "synapse_state_variable_plane: synapse '" + synapse_component_id +
                                 "' has no state variable '" + variable_name + "'");
    }
    log::throw_runtime_error(*logger, "synapse_state_variable_plane: the network uses no synapse '" +
                             synapse_component_id + "'");
}

f64 SpikeEngine::mean_firing_rate_hertz() const {
    if (total_neuron_count == 0 || lifetime == 0 || step_dt <= 0.0) return 0.0;

    s64 total_spikes = 0;
    for (s64 count : spike_counts_per_neuron) total_spikes += count;

    const f64 elapsed_seconds = (f64)lifetime * step_dt;
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
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
        membrane_offset_per_neuron[(usize)neuron_index] = cell_memory_index_of(neuron_index, "v");
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

    for (const RecordingConfig &profile : context.simulation.recording_profiles) {
        if (profile.output_filenames.empty()) continue;

        const String &filename = profile.output_filenames.front();
        const OutputFileFormat format = profile.file_output_format.front();

        ofstream file(filename);
        if (!file) {
            logger->error("SpikeEngine: cannot open '{}' for writing", filename);
            continue;
        }

        if (format == OutputFileFormat::SPIKE_EVENTS) {
            // An event selection's neuron_index is the cell's cell-memory index.
            Set<s64> selected;
            for (const RecordingSelection &selection : profile.selections) {
                const s64 selected_neuron = context.neuron_index_of(selection.neuron_index);
                if (selected_neuron >= 0) selected.insert(selected_neuron);
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

    weights = WeightMatrix();

    gpu.deallocate_slab(model_pointer);
    gpu.deallocate_slab(projection_run_pointer);
    gpu.deallocate_slab(random_values_pointer);

    alive = false;
}
