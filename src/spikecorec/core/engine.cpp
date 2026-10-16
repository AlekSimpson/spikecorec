//
// Created by Alek Simpson on 5/30/26.
//

#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "engine.h"
#include "backend.h"
#include "units.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

Spike_Engine::Spike_Engine(const String &lems_input_file, bool enable_hebbian_plasticity, f32 fit_tolerance,
                         Weight_Matrix::Fit_Method fit_method, f32 expected_active_edge_fraction)
    : Spike_Engine(lems_input_file, {}, "", 1.0, 0.0, enable_hebbian_plasticity, fit_tolerance, fit_method,
                  expected_active_edge_fraction) {}

Spike_Engine::Spike_Engine(const String &lems_input_file,
                         const Vector<Vector<s32>> &adjacency,
                         const String &synapse_component_id,
                         f64 connection_weight,
                         f64 connection_delay_seconds,
                         bool enable_hebbian_plasticity,
                         f32 fit_tolerance,
                         Weight_Matrix::Fit_Method fit_method,
                         f32 expected_active_edge_fraction)
    : Spike_Engine(lems_input_file, adjacency,
                  synapse_component_id.empty() ? Vector<String>{}
           : Vector<String>{synapse_component_id},
                  {}, connection_weight, connection_delay_seconds,
                  enable_hebbian_plasticity, fit_tolerance, fit_method, expected_active_edge_fraction) {}

Spike_Engine::Spike_Engine(const String &lems_input_file,
                         const Vector<Vector<s32>> &adjacency,
                         const Vector<String> &synapse_component_ids,
                         const Vector<f64> &synapse_proportions,
                         f64 connection_weight,
                         f64 connection_delay_seconds,
                         bool enable_hebbian_plasticity,
                         f32 fit_tolerance,
                         Weight_Matrix::Fit_Method fit_method,
                         f32 expected_active_edge_fraction)
    : Spike_Engine(lems_input_file, adjacency, synapse_component_ids, synapse_proportions, connection_weight,
                  connection_delay_seconds, Connection_Values_Function(), nullptr, enable_hebbian_plasticity,
                  fit_tolerance, fit_method, expected_active_edge_fraction) {}

Spike_Engine::Spike_Engine(const String &lems_input_file,
                         const Vector<Vector<s32>> &adjacency,
                         const Vector<String> &synapse_component_ids,
                         const Connection_Values_Function &connection_values,
                         bool enable_hebbian_plasticity,
                         f32 fit_tolerance,
                         Weight_Matrix::Fit_Method fit_method,
                         f32 expected_active_edge_fraction)
    : Spike_Engine(lems_input_file, adjacency, synapse_component_ids, Vector<f64>{}, 1.0, 0.0, connection_values,
                  nullptr, enable_hebbian_plasticity, fit_tolerance, fit_method, expected_active_edge_fraction) {
    if (!connection_values && !adjacency.empty()) {
        log::throw_invalid_argument(*logger, "Spike_Engine: connection_values is empty, so no connection has values");
    }
}

Spike_Engine::Spike_Engine(const String &lems_input_file,
                         const Vector<String> &synapse_component_ids,
                         const Connection_Arrays &connections,
                         bool enable_hebbian_plasticity,
                         f32 fit_tolerance,
                         Weight_Matrix::Fit_Method fit_method,
                         f32 expected_active_edge_fraction)
    : Spike_Engine(lems_input_file, {}, synapse_component_ids, Vector<f64>{}, 1.0, 0.0, Connection_Values_Function(),
                  &connections, enable_hebbian_plasticity, fit_tolerance, fit_method, expected_active_edge_fraction) {}

Spike_Engine::Spike_Engine(const String &lems_input_file,
                         const Vector<Vector<s32>> &adjacency,
                         const Vector<String> &synapse_component_ids,
                         const Vector<f64> &synapse_proportions,
                         f64 connection_weight,
                         f64 connection_delay_seconds,
                         const Connection_Values_Function &connection_values,
                         const Connection_Arrays *connection_arrays,
                         bool enable_hebbian_plasticity,
                         f32 fit_tolerance,
                         Weight_Matrix::Fit_Method fit_method,
                         f32 expected_active_edge_fraction)
    : logger(log::make_logger())
    , hebbian_plasticity_enabled(enable_hebbian_plasticity)
    , fit_tolerance(fit_tolerance)
    , fit_method(fit_method)
    , expected_active_edge_fraction(expected_active_edge_fraction) {
    if (!(expected_active_edge_fraction >= 0.0f && expected_active_edge_fraction <= 1.0f)) {
        log::throw_invalid_argument(*logger, "Spike_Engine: expected_active_edge_fraction " +
                                             to_string(expected_active_edge_fraction) + " is not between 0 and 1");
    }

    if (!context.validate_lems_schema(lems_input_file)) {
        log::throw_runtime_error(*logger,
                "Spike_Engine: " + lems_input_file + " failed lems schema validation:\n" +
                context.last_schema_validation_errors);
    }

    context.parse(lems_input_file);

    if (connection_arrays != nullptr) {
        sort_connection_arrays(*connection_arrays, context.simulation.total_neuron_count);
        const Connection_Arrays connections = *connection_arrays;
        const Connection_Values_Function values_of = [this, connections](s32 source, s32 target) {
            const Vector<s32> &targets = connection_rows.adjacency[static_cast<usize>(source)];
            const s64 position = std::lower_bound(targets.begin(), targets.end(), target) - targets.begin();
            const u32 index = connection_rows.array_index[static_cast<usize>(connection_rows.row_start[static_cast<usize>(source)] + position)];
            return Connection_Values{connections.synapse_indices[index], static_cast<f64>(connections.weights[index]),
                                     static_cast<f64>(connections.delays_seconds[index])};
        };
        apply_topology(connection_rows.adjacency, synapse_component_ids, {}, 1.0, 0.0, values_of);
    } else if (!adjacency.empty()) {
        apply_topology(adjacency, synapse_component_ids, synapse_proportions,
                       connection_weight, connection_delay_seconds, connection_values);
    }

    total_neuron_count = context.simulation.total_neuron_count;
    spike_history_row_count = spike_history_length(context);
    lifetime = context.simulation.total_tick_count;
    step_dt = context.simulation.step_dt;
    if (context.simulation.random_seed.has_value()) {
        simulation_seed = *context.simulation.random_seed;
    }
    random_generator = Random_Generator(simulation_seed);

    if (total_neuron_count == 0) {
        log::throw_runtime_error(*logger,
                "Spike_Engine: " + lems_input_file + " defines no neurons — check that the "
                "Simulation's target names a network with at least one population");
    }
    if (step_dt <= 0.0f) {
        log::throw_runtime_error(*logger,
                "Spike_Engine: " + lems_input_file + " gives the Simulation no usable step "
                "(parsed " + to_string(context.simulation.step_dt) + " s)");
    }

    // The engine records only cells' state variables and spikes. Leaving out anything else would
    // shift the columns after it, or turn an event file's selection into no filter at all.
    for (const Recording_Config &profile : context.simulation.recording_profiles) {
        for (const Recording_Selection &selection : profile.selections) {
            if (selection.neuron_index >= 0) continue;
            const bool is_column = selection.event_port.empty();
            log::throw_runtime_error(*logger,
                    "Spike_Engine: " + String(is_column ? "OutputColumn '" : "EventSelection '") +
                    selection.selection_id + "' in '" + profile.output_filenames.front() + "' records '" +
                    selection.quantity_path + "', which is " + (is_column ? "no state variable of a cell" : "no cell") +
                    "; the engine records only cells' state variables and spikes");
        }
    }

    Codegen compiler(context, &gpu);
    compiler.check_population_cells();
    compiler.check_cell_inputs();
    compiler.collect_synapse_deliveries();
    for (const auto &[synapse_id, delivery] : compiler.synapse_deliveries) {
        synapse_edge_state_variable_names[synapse_id] = delivery.edge_state_variable_names;
    }

    compiler.allocate_cell_model_memory();
    cell_state = compiler.data_partitions[0];
    network_inputs = compiler.data_partitions[1];
    spike_history = compiler.data_partitions[2];
    last_spiked = compiler.data_partitions[3];
    empty_edge_plane = compiler.data_partitions[4];
    event_arrival_count = compiler.data_partitions[5];
    input_row_start = compiler.data_partitions[6];
    input_entry_prototype = compiler.data_partitions[7];
    input_values = compiler.data_partitions[8];
    input_spike_counts = compiler.data_partitions[9];
    model_pointer = compiler.data_partitions[10];
    input_entry_count = static_cast<s64>(compiler.input_entries.size());
    input_value_stride = compiler.input_value_stride;
    input_spike_count_size = compiler.input_spike_count_size;

    initialize_model_buffers();
    compiler.initialize_cell_state(random_generator);
    compiler.initialize_input_state(random_generator);

    build_weight_matrix(compiler, compiler.edge_plane_count(), compiler.updated_edge_plane_count());

    weights.plasticity_reserve_entries = 0;
    if (hebbian_plasticity_enabled) {
        weights.plasticity_reserve_entries = DEFAULT_PLASTICITY_DELTA_CAPACITY;

        if (weights.total_edge_count > 0) {
            plasticity_target_root_mean_square = weights.neighbor_weight_stats().root_mean_square;
        }
        logger->warn("Spike_Engine: plasticity is enabled, but the generated kernel stages no "
                     "plasticity deltas yet, so weights will not change");
    }
    weights.resize_delta_capacity(weights.delta_capacity_for_threshold(refit_occupancy_threshold_fraction));
    // The kernel queues at most one update per plane per edge each tick, and only for edges with no
    // S entry, so the queue is sized like S at the refit threshold; a walk it cannot take waits for
    // another pass of the tick. The kernel bakes its size in.
    weights.resize_pending_delta_capacity(weights.pending_capacity_for_threshold(refit_occupancy_threshold_fraction));

    compiler.translate_kernel_code();

    // Refilled before every tick. At least one slot, so there is always a buffer to bind.
    random_values_count = compiler.random_values_count;
    Vector<Engine_Pointer> random_partitions;
    gpu.partition(static_cast<u64>(std::max<s64>(1, random_values_count)) * sizeof(f32), Engine_Datatype::FLOAT32, random_partitions);
    random_values_pointer = gpu.allocate(random_partitions);
    random_values = random_partitions[0];

    // Fixed once the weight matrix is built, so baked in. This also keeps master_step within
    // Metal's 31 buffer slots.
    compiler.bake_kernel_constant("branching_factor", weights.k2tree.branching_factor);
    compiler.bake_kernel_constant("superblock_size_words", weights.k2tree.superblock_size_words);
    compiler.bake_kernel_constant("padded_node_count", weights.k2tree.padded_node_count);
    compiler.bake_kernel_constant("tree_height", weights.k2tree.tree_height);
    compiler.bake_kernel_constant("internal_bit_count", weights.k2tree.internal_bit_count);
    compiler.bake_kernel_constant("pending_delta_capacity", weights.pending_delta_capacity);
    // With no edges, edge_row_offset and S are placeholders the walk's reservation must not read.
    compiler.bake_kernel_constant("updated_plane_count", weights.total_edge_count > 0 ? weights.updated_plane_count : 0);
    compiler.bake_kernel_constant("run_count", run_count);
    synapse_active_ticks = compiler.synapse_active_ticks();
    compiler.bake_kernel_constant("synapse_active_ticks", synapse_active_ticks);
    master_kernel_source = compiler.compile();
    logger->debug("Spike_Engine: generated master kernel, {} bytes", master_kernel_source.size());

    Optional<Engine_Function> result = gpu.create_function("master_step", master_kernel_source);
    if (!result.has_value()) {
        log::throw_runtime_error(*logger,
                "Spike_Engine: the generated master kernel failed to compile — see the log above "
                "for the compiler's own diagnostic");
    }
    kernel_function = *result;

    kernel_parameter_names = compiler.kernel_parameter_names();
    register_kernel_arguments();

    spike_counts_per_neuron.assign(static_cast<usize>(total_neuron_count), 0);

    // A traced selection's neuron_index is the variable's own cell_state index.
    for (const Recording_Config &profile : context.simulation.recording_profiles) {
        for (const Recording_Selection &selection : profile.selections) {
            if (!selection.event_port.empty()) continue;
            traced_selections.push_back(selection);
        }
    }

    calibrate_zero_start_planes();

    alive = true;
    logger->info("Spike_Engine constructed from {}: {} neurons, {} edges, {} cell types, "
                 "{} synapse components, dt={} s, {} ticks, spike history {} rows",
                 lems_input_file, total_neuron_count, weights.total_edge_count,
                 compiler.component_type_templates.size(),
                 context.simulation.synapse_instances.size(),
                 step_dt, lifetime, spike_history_row_count);
}

void Spike_Engine::initialize_model_buffers() {
    std::memset(network_inputs.get_contents(), 0, network_inputs.total_bytes);
    std::memset(spike_history.get_contents(), 0, spike_history.total_bytes);
    std::memset(empty_edge_plane.get_contents(), 0, empty_edge_plane.total_bytes);
    std::memset(event_arrival_count.get_contents(), 0, event_arrival_count.total_bytes);
    std::fill_n(last_spiked.get_contents_as<s64>(), total_neuron_count, NEVER_SPIKED_TICK);
}

void Spike_Engine::build_weight_matrix(const Codegen &compiler, s64 matrix_count, s64 updated_plane_count) {
    // A synapse's prototype index is its position in synapse_instances, as in codegen.
    Unordered_Map<String, s32> synapse_prototype_indices;
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        synapse_prototype_indices.emplace(synapse.id, static_cast<s32>(synapse_prototype_indices.size()));
    }

    // Each connection of a weighted arrival adds the synapse's starting value to its target's sum.
    Vector<f32> weighted_arrival_starting_value;
    bool any_weighted_arrival_starting_value = false;
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        const Synapse_Delivery &delivery = compiler.synapse_delivery(synapse);
        weighted_arrival_starting_value.push_back(delivery.weighted_arrival ? static_cast<f32>(delivery.starting_value) : 0.0f);
        any_weighted_arrival_starting_value = any_weighted_arrival_starting_value || weighted_arrival_starting_value.back() != 0.0f;
    }
    f32 *weighted_arrival_sums = any_weighted_arrival_starting_value
            ? network_inputs.get_contents_as<f32>() + Codegen::RUNNING_SUM_ROW * total_neuron_count
            : nullptr;

    // Edge ordinals: sources ascending, each source's targets ascending (the k^2-tree's row order).
    // Consecutive edges with the same synapse, weight and delay share a run.
    Vector<Vector<s32>> network(static_cast<usize>(total_neuron_count));
    Vector<s64> first_edge_ordinal;
    Vector<s64> edge_count;
    Vector<s32> synapse_prototype;
    Vector<f32> weight;
    Vector<s32> delay_ticks;

    // One source's connections at a time, from the adjacency topology or from network_data. Each
    // network_data row is released once read, so the parsed connections (about 80 bytes each) are
    // gone before the k^2-tree and the fits need memory; codegen keeps each synapse's largest weight.
    struct Row_Edge {
        s32 child;
        s32 prototype;
        f32 weight;
        s32 delay_ticks;
    };
    Vector<Row_Edge> row;
    Vector<Vector<NML_Network_Edge>> &edges_by_source = context.simulation.network_data.list;
    const Vector<Vector<s32>> *adjacency = topology_connections.adjacency;
    const usize source_count = adjacency != nullptr ? adjacency->size() : edges_by_source.size();
    s64 topology_edge = 0;
    usize topology_synapse = 0;

    Vector<f64> largest_weight_per_prototype(synapse_prototype_indices.size(), -1.0);
    s64 ordinal = 0;
    for (usize source = 0; source < source_count; source += 1) {
        row.clear();
        if (adjacency != nullptr && topology_connections.connection_values) {
            for (s32 target : (*adjacency)[source]) {
                const Connection_Values values = topology_connections.connection_values(static_cast<s32>(source), target);
                const s64 edge_delay_ticks = units::seconds_to_ticks(values.delay_seconds, context.simulation.step_dt);
                row.push_back({target, values.synapse_index, static_cast<f32>(values.weight),
                               static_cast<s32>(std::min<s64>(edge_delay_ticks, Numeric_Limits<s32>::max()))});
            }
        } else if (adjacency != nullptr) {
            const Vector<s64> &first_edge_of_synapse = topology_connections.first_edge_of_synapse;
            for (s32 target : (*adjacency)[source]) {
                while (topology_edge >= first_edge_of_synapse[topology_synapse + 1]) topology_synapse += 1;
                row.push_back({target, static_cast<s32>(topology_synapse), topology_connections.weight,
                               static_cast<s32>(topology_connections.delay_ticks)});
                topology_edge += 1;
            }
        } else {
            for (const NML_Network_Edge &edge : edges_by_source[source]) {
                auto prototype = synapse_prototype_indices.find(edge.component_id);
                if (prototype == synapse_prototype_indices.end()) {
                    log::throw_runtime_error(*logger,
                            "Spike_Engine: the connection " + to_string(source) + " -> " +
                            to_string(edge.child) + " names no synapse, so there is nothing to "
                            "carry its current");
                }
                row.push_back({static_cast<s32>(edge.child), prototype->second, edge.weight,
                               static_cast<s32>(std::min<s64>(edge.delay_ticks, Numeric_Limits<s32>::max()))});
            }
            Vector<NML_Network_Edge>().swap(edges_by_source[source]);
        }
        std::stable_sort(row.begin(), row.end(),
                         [](const Row_Edge &left, const Row_Edge &right) { return left.child < right.child; });
        network[source].reserve(row.size());

        for (usize index = 0; index < row.size(); index += 1) {
            const Row_Edge &edge = row[index];
            if (index > 0 && row[index - 1].child == edge.child) {
                logger->warn("Spike_Engine: neuron {} connects to neuron {} more than once; "
                             "keeping the first connection", source, edge.child);
                continue;
            }

            if (!std::isfinite(edge.weight)) {
                log::throw_runtime_error(*logger,
                        "Spike_Engine: the connection " + to_string(source) + " -> " + to_string(edge.child) +
                        " has the weight " + to_string(edge.weight) + ", which the weight matrix cannot hold");
            }

            network[source].push_back(edge.child);
            f64 &largest = largest_weight_per_prototype[static_cast<usize>(edge.prototype)];
            largest = std::max(largest, std::fabs(static_cast<f64>(edge.weight)));
            if (any_weighted_arrival_starting_value) {
                weighted_arrival_sums[edge.child] += weighted_arrival_starting_value[static_cast<usize>(edge.prototype)];
            }

            // The engine's synaptic latency is one tick, so every delay is at least one tick.
            const s32 edge_delay = std::max<s32>(1, edge.delay_ticks);
            const bool extends_current_run =
                    !first_edge_ordinal.empty() &&
                    synapse_prototype.back() == edge.prototype &&
                    weight.back() == edge.weight &&
                    delay_ticks.back() == edge_delay &&
                    first_edge_ordinal.back() + edge_count.back() == ordinal;

            if (extends_current_run) {
                edge_count.back() += 1;
            } else {
                first_edge_ordinal.push_back(ordinal);
                edge_count.push_back(1);
                synapse_prototype.push_back(edge.prototype);
                weight.push_back(edge.weight);
                delay_ticks.push_back(edge_delay);
            }
            ordinal += 1;
        }
    }
    Vector<Row_Edge>().swap(row);
    context.simulation.network_data = Adjacency_List(0, 0);
    topology_connections = Topology_Connections();
    connection_rows = Connection_Rows();
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        const f64 largest = largest_weight_per_prototype[static_cast<usize>(synapse_prototype_indices.at(synapse.id))];
        if (largest >= 0.0) context.simulation.largest_connection_weight_per_synapse[synapse.id] = largest;
    }

    // Each run's starting value per plane: weight, delay, then the StateVariables its synapse keeps
    // per connection at their OnStart values (0 past the synapse's own count).
    Vector<Vector<f32>> starting_state_per_prototype;
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        const Unordered_Map<String, f64> values = starting_values(context, synapse);
        Vector<f32> &starting_state = starting_state_per_prototype.emplace_back();
        for (const String &name : compiler.synapse_delivery(synapse).edge_state_variable_names) {
            starting_state.push_back(static_cast<f32>(values.at(name)));
            if (std::isfinite(starting_state.back())) continue;
            log::throw_runtime_error(*logger,
                    "Spike_Engine: synapse '" + synapse.id + "' starts '" + name + "' at " +
                    to_string(values.at(name)) + ", which the weight matrix cannot hold");
        }
    }
    Vector<Vector<f32>> initial_values(static_cast<usize>(matrix_count));
    for (usize run_index = 0; run_index < synapse_prototype.size(); run_index += 1) {
        const Vector<f32> &starting_state = starting_state_per_prototype[static_cast<usize>(synapse_prototype[run_index])];
        for (s64 plane = 0; plane < matrix_count; plane += 1) {
            f32 value = 0.0f;
            if (plane == Weight_Matrix::WEIGHT_PLANE) {
                value = weight[run_index];
            } else if (plane == Weight_Matrix::DELAY_PLANE) {
                value = static_cast<f32>(delay_ticks[run_index]);
            } else if (static_cast<usize>(plane - Weight_Matrix::FIRST_STATE_VARIABLE_PLANE) < starting_state.size()) {
                value = starting_state[static_cast<usize>(plane - Weight_Matrix::FIRST_STATE_VARIABLE_PLANE)];
            }
            initial_values[static_cast<usize>(plane)].push_back(value);
        }
    }

    weights = Weight_Matrix(gpu, network, /*rank=*/-1, /*check_indexing=*/true,
                           /*max_neighbor_count=*/-1, /*weight_seed=*/static_cast<s64>(simulation_seed),
                           matrix_count, updated_plane_count);
    // The matrix holds the topology now; released before the fits need memory.
    Vector<Vector<s32>>().swap(network);
    weights.fit_tolerance = fit_tolerance;
    weights.fit_method = fit_method;
    weights.expected_active_edge_fraction = expected_active_edge_fraction;
    if (ordinal > 0) weights.declare_runs(first_edge_ordinal, edge_count, synapse_prototype, initial_values);

    // The runs on the device, for the kernel's per-edge prototype lookup. declare_runs has merged
    // them by prototype, since the values are in the basis.
    const Vector<s64> &prototype_run_first_edge_ordinal = weights.run_first_edge_ordinal;
    const Vector<s32> &prototype_run_synapse_prototype = weights.run_synapse_prototype;
    Vector<Engine_Pointer> run_partitions;
    gpu.partition(prototype_run_first_edge_ordinal.size() * sizeof(s64), Engine_Datatype::SIGNED64, run_partitions)
       .partition(prototype_run_synapse_prototype.size() * sizeof(s32), Engine_Datatype::SIGNED32, run_partitions);
    run_pointer = gpu.allocate(run_partitions);
    run_first_edge_ordinal = run_partitions[0];
    run_synapse_prototype = run_partitions[1];
    if (!prototype_run_first_edge_ordinal.empty()) {
        std::copy(prototype_run_first_edge_ordinal.begin(), prototype_run_first_edge_ordinal.end(),
                  run_first_edge_ordinal.get_contents_as<s64>());
        std::copy(prototype_run_synapse_prototype.begin(), prototype_run_synapse_prototype.end(),
                  run_synapse_prototype.get_contents_as<s32>());
    }
    run_count = static_cast<s32>(prototype_run_first_edge_ordinal.size());

    logger->debug("Spike_Engine: weight matrix built — {} nodes, {} edges, {} runs ({} by prototype), rank {}",
                  weights.node_count, weights.total_edge_count, first_edge_ordinal.size(), run_count, weights.rank);
}

void Spike_Engine::register_kernel_arguments() {
    kernel_argument_sources = {
        {"rank_float4_stride",        [this] { return inline_scalar_argument(rank_float4_stride_argument); }},
        {"simulation_time",           [this] { return inline_scalar_argument(simulation_time_argument); }},
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
        {"pending_delta_control",     [this] { return resolve_edge_plane(weights.pending_delta_control); }},
        {"run_first_edge_ordinal", [this] { return resolve_edge_plane(run_first_edge_ordinal); }},
        {"run_synapse_prototype",  [this] { return resolve_edge_plane(run_synapse_prototype); }},
        {"random_values",             [this] { return random_values; }},
        {"event_arrival_count",       [this] { return event_arrival_count; }},
        {"input_row_start",           [this] { return input_row_start; }},
        {"input_entry_prototype",     [this] { return input_entry_prototype; }},
        {"input_values",              [this] { return input_values; }},
        {"input_spike_counts",        [this] { return input_spike_counts; }},
    };

    for (const String &name : kernel_parameter_names) {
        if (name == "tick" || kernel_argument_sources.count(name) != 0) continue;
        log::throw_runtime_error(*logger,
                "Spike_Engine: master_step takes '" + name + "', which the engine has nothing to bind to");
    }
}

const NML_Component_Instance *Spike_Engine::population_of_neuron(s64 neuron_index, s64 &first_neuron) const {
    first_neuron = 0;
    for (const NML_Component_Instance *population : network_populations(context)) {
        const s64 population_size = context.get_population_size(population);
        if (neuron_index >= first_neuron && neuron_index < first_neuron + population_size) return population;
        first_neuron += population_size;
    }
    return nullptr;
}

s64 Spike_Engine::cell_memory_index_of(s64 neuron_index, const String &variable_name) const {
    s64 first_neuron = 0;
    const NML_Component_Instance *population = population_of_neuron(neuron_index, first_neuron);
    if (!population) {
        log::throw_runtime_error(*logger,
                "Spike_Engine: neuron index " + to_string(neuron_index) + " is outside every population");
    }

    const NML_Component_Type &cell_type = *population_cell(context, *population).component_type;
    const Vector<String> &variable_names = cell_type.state_variable_names;
    auto variable = std::find(variable_names.begin(), variable_names.end(), variable_name);
    if (variable == variable_names.end()) {
        log::throw_runtime_error(*logger,
                "Spike_Engine: neuron " + to_string(neuron_index) + " is a '" + cell_type.name +
                "', which declares no state variable '" + variable_name + "'");
    }

    return context.simulation.population_base_indices.at(population->id) +
           (neuron_index - first_neuron) * NML_Context::cell_slot_count(cell_type) +
           static_cast<s64>(variable - variable_names.begin());
}

// adjacency[n] lists neuron n's targets. Every edge gets connection_weight and the delay; the
// synapses take consecutive edges in order, by proportion (equal shares when none are given), or
// connection_values gives each edge's synapse, weight and delay.
void Spike_Engine::apply_topology(const Vector<Vector<s32>> &adjacency,
                                 const Vector<String> &synapse_component_ids,
                                 const Vector<f64> &synapse_proportions,
                                 f64 connection_weight,
                                 f64 connection_delay_seconds,
                                 const Connection_Values_Function &connection_values) {
    NML_Context::NML_Simulation_Context &simulation = context.simulation;
    const s64 neuron_count = simulation.total_neuron_count;

    if (static_cast<s64>(adjacency.size()) > neuron_count) {
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
        const NML_Component_Instance *synapse = context.find_instance(synapse_id);
        if (!synapse) log::throw_runtime_error(*logger, "apply_topology: no synapse '" + synapse_id + "'");
        simulation.synapse_instances.push_back(*synapse);
    }

    s64 edge_count = 0;
    for (const Vector<s32> &targets : adjacency) edge_count += static_cast<s64>(targets.size());

    // In adjacency order, synapse k takes the edges
    // [first_edge_of_synapse[k], first_edge_of_synapse[k + 1]). A share may round to no edges.
    // With connection_values every edge names its own synapse instead.
    Vector<s64> first_edge_of_synapse = {0};
    if (!connection_values) {
        f64 proportion_total = 0.0;
        for (usize index = 0; index < synapse_component_ids.size(); index += 1) {
            const f64 proportion = synapse_proportions.empty() ? 1.0 : synapse_proportions[index];
            if (!std::isfinite(proportion) || proportion < 0.0) {
                log::throw_runtime_error(*logger, "apply_topology: synapse '" + synapse_component_ids[index] +
                                         "' has the proportion " + to_string(proportion) + "; each must be finite and not negative");
            }
            proportion_total += proportion;
        }
        if (proportion_total <= 0.0) {
            log::throw_runtime_error(*logger, "apply_topology: the synapse proportions sum to zero, so no synapse takes the edges");
        }
        f64 running_proportion = 0.0;
        for (usize index = 0; index < synapse_component_ids.size(); index += 1) {
            running_proportion += synapse_proportions.empty() ? 1.0 : synapse_proportions[index];
            first_edge_of_synapse.push_back(std::llround(running_proportion / proportion_total * static_cast<f64>(edge_count)));
        }
        first_edge_of_synapse.back() = edge_count;
    }

    const s64 delay_ticks = units::seconds_to_ticks(connection_delay_seconds, simulation.step_dt);
    simulation.network_data = Adjacency_List(0, 0);
    simulation.total_edge_count = edge_count;
    topology_connections.adjacency = &adjacency;
    topology_connections.connection_values = connection_values;
    topology_connections.first_edge_of_synapse = first_edge_of_synapse;
    topology_connections.weight = static_cast<f32>(connection_weight);
    topology_connections.delay_ticks = delay_ticks;

    // One pass over the connections for the longest delay, which sizes the spike history, and for
    // one connection per synapse and target population, for codegen's checks.
    Vector<s64> population_ends;
    for (const NML_Component_Instance *population : network_populations(context)) {
        population_ends.push_back((population_ends.empty() ? 0 : population_ends.back()) + context.get_population_size(population));
    }
    Vector<u8> sampled(synapse_component_ids.size() * population_ends.size(), 0);
    simulation.sampled_connections.clear();
    simulation.maximum_edge_delay = connection_values ? 0 : delay_ticks;
    simulation.minimum_edge_delay = connection_values ? 0 : delay_ticks;
    s64 edge_ordinal = 0;
    usize proportion_synapse = 0;
    for (usize source = 0; source < adjacency.size(); source += 1) {
        for (s32 target : adjacency[source]) {
            usize synapse_index = 0;
            f64 weight = connection_weight;
            s64 edge_delay_ticks = delay_ticks;
            if (connection_values) {
                const Connection_Values values = connection_values(static_cast<s32>(source), target);
                if (values.synapse_index < 0 || static_cast<usize>(values.synapse_index) >= synapse_component_ids.size()) {
                    log::throw_runtime_error(*logger, "apply_topology: the connection " + to_string(source) + " -> " +
                                             to_string(target) + " names synapse " + to_string(values.synapse_index) +
                                             ", outside the " + to_string(synapse_component_ids.size()) + " given");
                }
                if (!std::isfinite(values.delay_seconds) || values.delay_seconds < 0.0) {
                    log::throw_runtime_error(*logger, "apply_topology: the connection " + to_string(source) + " -> " +
                                             to_string(target) + " has the delay " + to_string(values.delay_seconds) +
                                             " s; each must be finite and not negative");
                }
                synapse_index = static_cast<usize>(values.synapse_index);
                weight = values.weight;
                edge_delay_ticks = units::seconds_to_ticks(values.delay_seconds, simulation.step_dt);
                simulation.maximum_edge_delay = std::max(simulation.maximum_edge_delay, edge_delay_ticks);
                simulation.minimum_edge_delay = edge_ordinal == 0 ? edge_delay_ticks
                                                                  : std::min(simulation.minimum_edge_delay, edge_delay_ticks);
            } else {
                while (edge_ordinal >= first_edge_of_synapse[proportion_synapse + 1]) proportion_synapse += 1;
                synapse_index = proportion_synapse;
            }
            edge_ordinal += 1;

            const usize population = static_cast<usize>(
                    std::upper_bound(population_ends.begin(), population_ends.end(), static_cast<s64>(target)) - population_ends.begin());
            if (population >= population_ends.size()) continue;
            u8 &seen = sampled[synapse_index * population_ends.size() + population];
            if (seen != 0) continue;
            seen = 1;

            NML_Network_Edge edge;
            edge.component_id = synapse_component_ids[synapse_index];
            edge.weight = static_cast<f32>(weight);
            edge.delay_ticks = edge_delay_ticks;
            edge.parent = static_cast<s64>(source);
            edge.child = target;
            simulation.sampled_connections.push_back(edge);
        }
    }
}

// A counting sort by source puts each row's connections in array order; each row is then sorted by
// target, which also brings a repeated pair together.
void Spike_Engine::sort_connection_arrays(const Connection_Arrays &connections, s64 neuron_count) {
    const s64 connection_count = connections.connection_count;
    if (connection_count < 0 || connection_count > static_cast<s64>(Numeric_Limits<u32>::max())) {
        log::throw_invalid_argument(*logger, "Spike_Engine: connection_count " + to_string(connection_count) +
                                             " is outside 0 to " + to_string(Numeric_Limits<u32>::max()));
    }
    if (connection_count > 0 && (connections.source_neurons == nullptr || connections.target_neurons == nullptr ||
                                 connections.synapse_indices == nullptr || connections.weights == nullptr ||
                                 connections.delays_seconds == nullptr)) {
        log::throw_invalid_argument(*logger, "Spike_Engine: a Connection_Arrays array is missing");
    }
    for (s64 index = 0; index < connection_count; index += 1) {
        const u32 source = connections.source_neurons[index];
        const u32 target = connections.target_neurons[index];
        if (static_cast<s64>(source) < neuron_count && static_cast<s64>(target) < neuron_count) continue;
        log::throw_runtime_error(*logger, "Spike_Engine: connection " + to_string(index) + " is " + to_string(source) +
                                          " -> " + to_string(target) + ", outside the network's " +
                                          to_string(neuron_count) + " neurons");
    }

    Connection_Rows &rows = connection_rows;
    rows.row_start.assign(static_cast<usize>(neuron_count + 1), 0);
    for (s64 index = 0; index < connection_count; index += 1) rows.row_start[connections.source_neurons[index] + 1] += 1;
    for (s64 source = 0; source < neuron_count; source += 1) {
        rows.row_start[static_cast<usize>(source + 1)] += rows.row_start[static_cast<usize>(source)];
    }
    rows.array_index.resize(static_cast<usize>(connection_count));
    Vector<s64> next_slot(rows.row_start.begin(), rows.row_start.end() - 1);
    for (s64 index = 0; index < connection_count; index += 1) {
        s64 &slot = next_slot[connections.source_neurons[index]];
        rows.array_index[static_cast<usize>(slot)] = static_cast<u32>(index);
        slot += 1;
    }
    Vector<s64>().swap(next_slot);

    rows.adjacency.assign(static_cast<usize>(neuron_count), {});
    for (s64 source = 0; source < neuron_count; source += 1) {
        const auto first = rows.array_index.begin() + rows.row_start[static_cast<usize>(source)];
        const auto last = rows.array_index.begin() + rows.row_start[static_cast<usize>(source + 1)];
        std::sort(first, last, [&](u32 left, u32 right) {
            return connections.target_neurons[left] < connections.target_neurons[right];
        });
        Vector<s32> &targets = rows.adjacency[static_cast<usize>(source)];
        targets.reserve(static_cast<usize>(last - first));
        for (auto entry = first; entry != last; ++entry) {
            const s32 target = static_cast<s32>(connections.target_neurons[*entry]);
            if (!targets.empty() && targets.back() == target) {
                log::throw_runtime_error(*logger, "Spike_Engine: the connection " + to_string(source) + " -> " +
                                                  to_string(target) + " is given twice (entries " +
                                                  to_string(*(entry - 1)) + " and " + to_string(*entry) +
                                                  "); each ordered pair holds one connection");
            }
            targets.push_back(target);
        }
    }
}

Spike_Engine::~Spike_Engine() {
    if (alive) shutdown();
}

Engine_Pointer Spike_Engine::resolve_edge_plane(const Engine_Pointer &plane) const {
    return plane.is_empty() ? empty_edge_plane : plane;
}

namespace {

// A tick's time in seconds. When 1 / dt is a whole number, divides by it so the time rounds once.
f64 time_of_tick(s64 tick, f64 step_dt) {
    const f64 steps_per_second = std::round(1.0 / step_dt);
    if (steps_per_second >= 1.0 && std::fabs(steps_per_second * step_dt - 1.0) < 1e-12) {
        return static_cast<f64>(tick) / steps_per_second;
    }
    return static_cast<f64>(tick) * step_dt;
}

} // namespace

void Spike_Engine::step_simulation(s64 tick) {
    if (!alive) {
        log::throw_runtime_error(*logger, "step_simulation: the engine has been shut down");
    }
    // The kernel indexes two-row buffers by tick parity, rings by tick modulo, and each spike
    // train at its offset plus the tick, so ticks must come one at a time from 0 to lifetime - 1.
    if (tick != ticks_stepped || tick >= lifetime) {
        log::throw_runtime_error(*logger,
                "step_simulation: got tick " + to_string(tick) + ", but ticks run in order from 0 and " +
                (ticks_stepped >= lifetime ? "all " + to_string(lifetime) + " of this run's ticks have run"
                                           : "the next is " + to_string(ticks_stepped)));
    }

    const bool plasticity_fold_due = hebbian_plasticity_enabled && plasticity_fold_every_n_ticks > 0 &&
                                     (tick + 1) % plasticity_fold_every_n_ticks == 0;
    dispatch_tick(tick, plasticity_fold_due);

    // Refit between dispatches, so no kernel reads the matrix while it changes.
    const bool refit_scheduled = refit_every_n_ticks > 0 && (tick + 1) % refit_every_n_ticks == 0;
    const bool spaced_out = tick - last_refit_tick >= minimum_ticks_between_refits;
    if (spaced_out && (refit_scheduled || weights.is_refit_due(refit_occupancy_threshold_fraction))) {
        weights.fit_tolerance = fit_tolerance;
        weights.fit_method = fit_method;
        weights.refit();
        last_refit_tick = tick;

        if (hebbian_plasticity_enabled && plasticity_target_root_mean_square >= 0.0f) {
            weights.scale_neighbor_weights_to_root_mean_square(plasticity_target_root_mean_square);
        }
        // S is empty, so resize it back to the threshold's size.
        weights.resize_delta_capacity(weights.delta_capacity_for_threshold(refit_occupancy_threshold_fraction));
    }

    record_tick(tick);
}

void Spike_Engine::dispatch_tick(s64 tick, bool plasticity_fold_due) {
    // This tick's draw for every random() slot.
    f32 *random_value_data = random_values.get_contents_as<f32>();
    for (s64 index = 0; index < random_values_count; index += 1) {
        random_value_data[index] = static_cast<f32>(random_generator.uniform());
    }

    rank_float4_stride_argument = static_cast<s32>(weights.rank_float4_stride);
    simulation_time_argument = static_cast<decltype(simulation_time_argument)>(time_of_tick(tick, step_dt));

    // Resolved again before every walk pass: a merge can move S into a larger slab.
    Vector<Engine_Pointer> parameters;
    const auto resolve_parameters = [&]() {
        parameters.clear();
        parameters.reserve(kernel_parameter_names.size());
        for (const String &name : kernel_parameter_names) {
            parameters.push_back(name == "tick" ? inline_scalar_argument(tick) : kernel_argument_sources.at(name)());
        }
    };
    resolve_parameters();

    weights.begin_first_walk_pass();
    if (!gpu.run_function(kernel_function, parameters, total_neuron_count)) {
        log::throw_runtime_error(*logger,
                "Spike_Engine: tick " + to_string(tick) + " failed on the GPU");
    }

    // Merge the updates queued this tick into S so the next tick reads them. A neuron whose walk
    // the queue could not take walks in another pass of the tick, after a merge made room.
    if (weights.updated_plane_count > 0 || plasticity_fold_due) weights.compact_pending_deltas();
    for (s64 deferred = weights.begin_next_walk_pass(); deferred > 0; deferred = weights.begin_next_walk_pass()) {
        resolve_parameters();
        if (!gpu.run_function(kernel_function, parameters, deferred)) {
            log::throw_runtime_error(*logger,
                    "Spike_Engine: a deferred walk of tick " + to_string(tick) + " failed on the GPU");
        }
        weights.compact_pending_deltas();
        deferred_walk_pass_count += 1;
    }
}

void Spike_Engine::calibrate_zero_start_planes() {
    if (weights.updated_plane_count == 0) return;
    const Vector<s64> planes = weights.planes_awaiting_lane_budget();
    if (planes.empty()) return;

    // Every state a tick changes lives in the model slab, S, the queue and the random generator.
    const u8 *model_bytes = model_pointer.get_contents_as<u8>();
    const Vector<u8> saved_model(model_bytes, model_bytes + model_pointer.total_bytes);
    const Random_Generator saved_random_generator = random_generator;

    // Every neuron spikes once, neuron n at tick (n * CALIBRATION_SPIKE_ORDER_MULTIPLIER) mod the
    // spread: a fixed shuffle, so a spike's tick does not follow the neuron's place in the network.
    // Each synapse then runs through its whole response: the spike reaches it after its delay, and
    // it returns to rest once synapse_active_ticks pass. A spike is written after its tick's
    // dispatch, which would otherwise overwrite that tick's spike row.
    constexpr s64 CALIBRATION_SPIKE_ORDER_MULTIPLIER = 7919;
    const s64 settle_ticks = synapse_active_ticks >= 0 ? std::min<s64>(synapse_active_ticks, CALIBRATION_TICK_LIMIT)
                                                       : CALIBRATION_TICK_LIMIT;
    const s64 spike_spread = std::max<s64>(1, settle_ticks / CALIBRATION_SPIKE_SPREAD_DIVISOR);
    s64 *last_spiked_ticks = last_spiked.get_contents_as<s64>();
    u8 *spike_rows = spike_history.get_contents_as<u8>();
    Vector<Vector<s64>> neurons_spiking_at(static_cast<usize>(spike_spread));
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
        neurons_spiking_at[static_cast<usize>((neuron_index * CALIBRATION_SPIKE_ORDER_MULTIPLIER) % spike_spread)]
            .push_back(neuron_index);
    }
    const auto force_spikes = [&](s64 tick) {
        if (tick >= spike_spread) return;
        const s64 row = (tick % spike_history_row_count) * total_neuron_count;
        for (s64 neuron_index : neurons_spiking_at[static_cast<usize>(tick)]) {
            last_spiked_ticks[neuron_index] = tick;
            spike_rows[row + neuron_index] = 1;
        }
    };
    force_spikes(0);

    // Every neuron spikes, so S comes to hold every edge of each plane the kernel updates. Sized for
    // that once, S never grows by copying into a larger slab.
    weights.resize_delta_capacity(weights.updated_plane_count * weights.total_edge_count + weights.plasticity_reserve_entries);

    // The values taken are those at the tick where the planes' updates are largest (by their sum of
    // squares); a tick is kept when it beats the kept one by 5%, which saves copying S every tick.
    constexpr f64 KEPT_TICK_IMPROVEMENT = 1.05;
    const s64 tick_count = spike_spread + settle_ticks;
    Vector<Vector<f32>> kept_values(planes.size());
    f64 kept_energy = 0.0;
    s64 kept_tick = 0;
    for (s64 tick = 1; tick <= tick_count; tick += 1) {
        dispatch_tick(tick, false);
        force_spikes(tick);
        f64 energy = 0.0;
        for (s64 plane : planes) energy += weights.plane_update_sum_of_squares(plane);
        if (energy <= kept_energy * KEPT_TICK_IMPROVEMENT) continue;
        kept_energy = energy;
        kept_tick = tick;
        for (usize index = 0; index < planes.size(); index += 1) {
            kept_values[index] = Vector<f32>();
            kept_values[index] = weights.plane_update_values(planes[index]);
        }
    }

    std::memcpy(model_pointer.get_contents(), saved_model.data(), saved_model.size());
    random_generator = saved_random_generator;
    weights.discard_updates(refit_occupancy_threshold_fraction);
    deferred_walk_pass_count = 0;

    // The planes hold no lanes, so their values are their updates. Fitted once S is back to its
    // usual size, since the fits need the memory.
    for (usize index = 0; index < planes.size(); index += 1) {
        weights.calibrate_lane_budget(planes[index], kept_values[index]);
        Vector<f32>().swap(kept_values[index]);
    }
    calibration_tick_count = tick_count;
    logger->debug("Spike_Engine: calibrated the lane budgets of {} planes over {} ticks after a spike on every "
                  "neuron, spread over {} ticks, from the values at tick {}", planes.size(), tick_count, spike_spread,
                  kept_tick);
}

void Spike_Engine::set_refit_occupancy_threshold_fraction(f32 fraction) {
    refit_occupancy_threshold_fraction = fraction;
    s64 largest_plane = 0;
    for (s64 entry_count : weights.sparse_delta_entry_count) largest_plane = std::max(largest_plane, entry_count);
    weights.resize_delta_capacity(std::max(weights.delta_capacity_for_threshold(fraction), largest_plane));
}

void Spike_Engine::run(const Function<void(s64)> &after_each_tick) {
    if (!alive) {
        log::throw_runtime_error(*logger, "run: the engine has been shut down");
    }
    // A second run would start from the first one's state, so the engine runs only once.
    if (ticks_stepped >= lifetime) {
        log::throw_runtime_error(*logger,
                "run: all " + to_string(lifetime) + " ticks of this run have run; build a new Spike_Engine "
                "to run the model again");
    }
    logger->info("Spike_Engine: running ticks {} to {} at dt={} s", ticks_stepped, lifetime - 1, step_dt);

    for (s64 tick = ticks_stepped; tick < lifetime; tick += 1) {
        step_simulation(tick);
        if (after_each_tick) after_each_tick(tick);
    }

    logger->info("Spike_Engine: run finished — {} spikes, mean rate {:.2f} Hz, {:.1f}% of "
                 "neurons spiked at least once",
                 recorded_spikes.size(), mean_firing_rate_hertz(),
                 100.0 * fraction_of_neurons_that_spiked());
}

void Spike_Engine::record_tick(s64 tick) {
    const u8 *spike_row = static_cast<const u8 *>(spike_history.get_contents()) +
                          (tick % spike_history_row_count) * total_neuron_count;
    const f32 *cell_state_data = static_cast<const f32 *>(cell_state.get_contents());

    {
        const Lock_Guard<Mutex> results_lock(results_mutex);
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
            if (spike_row[neuron_index] == 0) continue;

            spike_counts_per_neuron[static_cast<usize>(neuron_index)] += 1;
            recorded_spikes.push_back(Recorded_Spike{static_cast<f64>(tick) * step_dt, neuron_index});
        }

        if (!traced_selections.empty()) {
            recorded_trace_times.push_back(static_cast<f64>(tick) * step_dt);
            for (const Recording_Selection &selection : traced_selections) {
                recorded_traces.push_back(cell_state_data[selection.neuron_index]);
            }
        }
        // Advanced with the tick's results, so a reader sees the two agree.
        ticks_stepped = tick + 1;
    }

    if (membrane_video_recorder && tick % membrane_video_frame_stride == 0) {
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
            membrane_frame_scratch[static_cast<usize>(neuron_index)] =
                    cell_state_data[membrane_offset_per_neuron[static_cast<usize>(neuron_index)]];
        }
        membrane_video_recorder->record_frame(membrane_frame_scratch.data(),
                                              total_neuron_count);
    }
}

f32 Spike_Engine::read_state_variable(s64 neuron_index, const String &variable_name) const {
    return static_cast<const f32 *>(cell_state.get_contents())[cell_memory_index_of(neuron_index, variable_name)];
}

s64 Spike_Engine::synapse_state_variable_plane(const String &synapse_component_id,
                                              const String &variable_name) const {
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        if (synapse.id != synapse_component_id) continue;

        const Vector<String> &names = synapse_edge_state_variable_names.at(synapse.id);
        for (usize index = 0; index < names.size(); index += 1) {
            if (names[index] == variable_name) return Weight_Matrix::FIRST_STATE_VARIABLE_PLANE + static_cast<s64>(index);
        }
        const Vector<String> &declared = synapse.component_type->state_variable_names;
        if (std::find(declared.begin(), declared.end(), variable_name) != declared.end()) {
            log::throw_runtime_error(*logger, "synapse_state_variable_plane: synapse '" + synapse_component_id +
                                     "' keeps no '" + variable_name + "' per connection: it delivers weighted "
                                     "arrivals, whose increments only their target's sum keeps");
        }
        log::throw_runtime_error(*logger, "synapse_state_variable_plane: synapse '" + synapse_component_id +
                                 "' has no state variable '" + variable_name + "'");
    }
    log::throw_runtime_error(*logger, "synapse_state_variable_plane: the network uses no synapse '" +
                             synapse_component_id + "'");
}

f64 Spike_Engine::mean_firing_rate_hertz() const {
    const Lock_Guard<Mutex> results_lock(results_mutex);
    if (total_neuron_count == 0 || ticks_stepped == 0 || step_dt <= 0.0) return 0.0;

    s64 total_spikes = 0;
    for (s64 count : spike_counts_per_neuron) total_spikes += count;

    const f64 elapsed_seconds = static_cast<f64>(ticks_stepped) * step_dt;
    return static_cast<f64>(total_spikes) / (static_cast<f64>(total_neuron_count) * elapsed_seconds);
}

f64 Spike_Engine::fraction_of_neurons_that_spiked() const {
    if (total_neuron_count == 0) return 0.0;

    const Lock_Guard<Mutex> results_lock(results_mutex);
    s64 spiking_neurons = 0;
    for (s64 count : spike_counts_per_neuron) spiking_neurons += count > 0 ? 1 : 0;

    return static_cast<f64>(spiking_neurons) / static_cast<f64>(total_neuron_count);
}

void Spike_Engine::record_membrane_video(const String &path, s64 frame_stride) {
    if (frame_stride < 1) {
        log::throw_runtime_error(*logger,
                "record_membrane_video: frame_stride must be at least 1 (got " +
                to_string(frame_stride) + ")");
    }

    membrane_offset_per_neuron.assign(static_cast<usize>(total_neuron_count), -1);
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; neuron_index += 1) {
        membrane_offset_per_neuron[static_cast<usize>(neuron_index)] = cell_memory_index_of(neuron_index, "v");
    }

    membrane_video_frame_stride = frame_stride;
    membrane_frame_scratch.assign(static_cast<usize>(total_neuron_count), 0.0f);
    membrane_video_recorder = std::make_unique<Simulation_Recorder>(path, total_neuron_count);

    logger->info("record_membrane_video: {} neurons every {} ticks -> {}",
                 total_neuron_count, frame_stride, path);
}

namespace {

// Throws unless the stream wrote everything: a failed open, or a write that ran out of disk.
void require_written(const Output_File_Stream &file, const String &path, const char *writer_name) {
    if (file) return;
    log::throw_runtime_error(log::logger(), String(writer_name) + ": cannot write '" + path + "'");
}

} // namespace

void Spike_Engine::write_spike_file(const String &path) const {
    Output_File_Stream file(path);
    require_written(file, path, "write_spike_file");

    const Lock_Guard<Mutex> results_lock(results_mutex);
    for (const Recorded_Spike &spike : recorded_spikes) {
        file << setprecision(9) << spike.time_seconds << "\t" << spike.neuron_index << "\n";
    }
    file.close();
    require_written(file, path, "write_spike_file");
    logger->info("write_spike_file: wrote {} spikes to {}", recorded_spikes.size(), path);
}

void Spike_Engine::write_recordings() {
    if (membrane_video_recorder) {
        membrane_video_recorder->finish();
        logger->info("Spike_Engine: membrane video recording closed");
    }

    const Lock_Guard<Mutex> results_lock(results_mutex);
    for (const Recording_Config &profile : context.simulation.recording_profiles) {
        if (profile.output_filenames.empty()) continue;

        const Path declared_name = profile.output_filenames.front();
        const String filename = (declared_name.is_absolute() || profile.output_directory.empty()
                                 ? declared_name : Path(profile.output_directory) / declared_name).string();
        const Output_File_Format format = profile.file_output_format.front();

        if (format == Output_File_Format::SPIKE_EVENTS) {
            Output_File_Stream file(filename);
            require_written(file, filename, "write_recordings");

            // Each spike of a selected neuron, once per selection naming it, beside the selection's
            // id. An event selection's neuron_index is a cell_state index, not a neuron index.
            Unordered_Map<s64, Vector<String>> selection_ids_per_neuron;
            for (const Recording_Selection &selection : profile.selections) {
                selection_ids_per_neuron[context.neuron_index_of(selection.neuron_index)].push_back(selection.selection_id);
            }
            const bool id_first = profile.event_format == "ID_TIME";

            s64 written = 0;
            for (const Recorded_Spike &spike : recorded_spikes) {
                auto selection_ids = selection_ids_per_neuron.find(spike.neuron_index);
                if (selection_ids == selection_ids_per_neuron.end()) continue;

                for (const String &selection_id : selection_ids->second) {
                    if (id_first) {
                        file << selection_id << "\t" << setprecision(9) << spike.time_seconds << "\n";
                    } else {
                        file << setprecision(9) << spike.time_seconds << "\t" << selection_id << "\n";
                    }
                    written += 1;
                }
            }
            file.close();
            require_written(file, filename, "write_recordings");
            logger->info("Spike_Engine: wrote {} spikes to {}", written, filename);
            continue;
        }

        Vector<usize> columns;
        for (const Recording_Selection &selection : profile.selections) {
            for (usize index = 0; index < traced_selections.size(); index += 1) {
                if (traced_selections[index].quantity_path != selection.quantity_path) continue;
                columns.push_back(index);
                break;
            }
        }

        // A .spire file holds one frame per recorded tick, one value per column, and no time.
        if (format != Output_File_Format::NML_STANDARD) {
            Simulation_Recorder recorder(filename, static_cast<s64>(columns.size()));
            Vector<f32> frame(columns.size());
            for (usize row = 0; row < recorded_trace_times.size(); row += 1) {
                for (usize column = 0; column < columns.size(); column += 1) {
                    frame[column] = recorded_traces[row * traced_selections.size() + columns[column]];
                }
                recorder.record_frame(frame.data(), static_cast<s64>(frame.size()));
            }
            recorder.finish();
            logger->info("Spike_Engine: wrote {} frames x {} columns to {}",
                         recorded_trace_times.size(), columns.size(), filename);
            continue;
        }

        Output_File_Stream file(filename);
        require_written(file, filename, "write_recordings");
        for (usize row = 0; row < recorded_trace_times.size(); row += 1) {
            file << setprecision(9) << recorded_trace_times[row];
            for (usize column : columns) {
                file << "\t" << recorded_traces[row * traced_selections.size() + column];
            }
            file << "\n";
        }
        file.close();
        require_written(file, filename, "write_recordings");
        logger->info("Spike_Engine: wrote {} rows x {} columns to {}",
                     recorded_trace_times.size(), columns.size(), filename);
    }
}

void Spike_Engine::shutdown() {
    if (!alive) return;

    logger->info("Spike_Engine: shutting down");

    gpu.release_function(kernel_function);

    weights = Weight_Matrix();

    gpu.deallocate_slab(model_pointer);
    gpu.deallocate_slab(run_pointer);
    gpu.deallocate_slab(random_values_pointer);

    alive = false;
}
