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

// Every arena carve is rounded up to EngineAllocator::ALLOCATION_ALIGNMENT, so the slab
// has to be sized with that rounding included or the last few buffers do not fit.
u64 aligned_byte_count(u64 element_bytes, s64 length) {
    if (length <= 0) return 0;

    const u64 alignment = EngineAllocator::ALLOCATION_ALIGNMENT;
    const u64 requested = element_bytes * (u64)length;
    return (requested + alignment - 1) / alignment * alignment;
}

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
    : logger(log::make_logger()) {

    if (!gpu_context_is_initialized()) initialize_gpu_context();

    NML_Parser parser;
    if (!parser.validate_against_schema(lems_input_file)) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: " + lems_input_file + " failed NeuroML schema validation:\n" +
                parser.last_schema_validation_errors);
    }

    network_details = parser.parse_lems(lems_input_file);
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

    block_count = (s32)((total_neuron_count + thread_count_per_block - 1) /
                        thread_count_per_block);

    allocate_model_buffers();
    initialize_cell_state();
    build_weight_matrix();
    collect_stimulus();

    master_kernel_source = generate_master_kernel(network_details, layout);
    logger->debug("SpikeEngine: generated master kernel, {} bytes:\n{}",
                  master_kernel_source.size(), master_kernel_source);
    master_kernel = compile_kernel(master_kernel_source.c_str(), "master_step");

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
    const u64 gpu_bytes =
            aligned_byte_count(sizeof(f32), layout.cell_state_length) +
            aligned_byte_count(sizeof(f32), layout.cell_parameter_length) +
            aligned_byte_count(sizeof(f32), layout.synapse_parameter_length) +
            aligned_byte_count(sizeof(f32), 2 * total_neuron_count) +
            aligned_byte_count(sizeof(u8), layout.spike_history_length * total_neuron_count) +
            aligned_byte_count(sizeof(s64), total_neuron_count) +
            aligned_byte_count(sizeof(f32), 1);

    allocator = EngineAllocator(0, gpu_bytes);

    cell_state         = allocator.allocate_gpu(sizeof(f32), layout.cell_state_length);
    cell_parameters    = allocator.allocate_gpu(sizeof(f32), layout.cell_parameter_length);
    synapse_parameters = allocator.allocate_gpu(sizeof(f32), layout.synapse_parameter_length);
    network_inputs     = allocator.allocate_gpu(sizeof(f32), 2 * total_neuron_count);
    spike_history      = allocator.allocate_gpu(
            sizeof(u8), layout.spike_history_length * total_neuron_count);
    last_spiked        = allocator.allocate_gpu(sizeof(s64), total_neuron_count);
    edge_placeholder   = allocator.allocate_gpu(sizeof(f32), 1);

    // The arena does not clear its slab, and every one of these is read before it is
    // written on the first tick.
    memset(network_inputs.get_contents(), 0, (usize)(2 * total_neuron_count) * sizeof(f32));
    memset(spike_history.get_contents(), 0,
           (usize)(layout.spike_history_length * total_neuron_count) * sizeof(u8));
    memset(last_spiked.get_contents(), 0, (usize)total_neuron_count * sizeof(s64));

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

    logger->debug("SpikeEngine: arena {} bytes — cell_state {}, cell_parameters {}, "
                  "synapse_parameters {}, network_inputs {}, spike_history {}",
                  gpu_bytes, layout.cell_state_length, layout.cell_parameter_length,
                  layout.synapse_parameter_length, 2 * total_neuron_count,
                  layout.spike_history_length * total_neuron_count);
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

// Everything per-edge goes into the WeightMatrix: the connection weight, the delay, the
// edge's synapse prototype, and that synapse's state variables. The k^2-tree is what says
// which (source, target) pairs exist, so only real edges are ever addressed, and each
// matrix's Ck is pinned to zero so the stored value is the value — a weight of 5e-10 reads
// back as 5e-10 rather than as whatever the shared basis reconstructs near it.
void SpikeEngine::build_weight_matrix() {
    const Vector<Vector<s64>> adjacency = build_adjacency_list(network_details);

    Vector<vector<s32>> network((usize)total_neuron_count);
    for (usize source = 0; source < adjacency.size(); source += 1) {
        network[source].reserve(adjacency[source].size());
        for (s64 target : adjacency[source]) network[source].push_back((s32)target);
    }

    weights.emplace(network, /*rank=*/1, /*check_indexing=*/true, /*max_neighbor_count=*/-1,
                    /*weight_seed=*/(s64)simulation_seed);
    weights->configure_per_edge_variable_count(layout.per_edge_variable_count);

    for (usize source = 0; source < network_details.neurons.size(); source += 1) {
        for (const NetworkEdge &edge : network_details.neurons[source].outgoing_edges) {
            const s32 source_node = (s32)source;
            const s32 target_node = (s32)edge.target_neuron_index;

            weights->set_edge_weight(source_node, target_node, (f32)edge.weight);

            // The engine's synaptic latency is one tick, so a connection that names no
            // delay still arrives a tick later rather than instantaneously.
            weights->set_edge_delay_ticks(source_node, target_node,
                                          (s32)std::max<s64>(1, edge.delay_tick_count));

            if (edge.synapse_prototype_index < 0) {
                log::throw_runtime_error(*logger,
                        "SpikeEngine: the connection " + to_string(source) + " -> " +
                        to_string(edge.target_neuron_index) +
                        " names no synapse, so there is nothing to carry its current");
            }
            weights->set_edge_variable(0, source_node, target_node,
                                       (f32)edge.synapse_prototype_index);

            const ComponentPrototype &prototype =
                    network_details.synapse_prototypes[(usize)edge.synapse_prototype_index];
            const SynapseTypeSpecification &synapse_type =
                    network_details.synapse_types[(usize)prototype.type_index];

            for (usize slot = 0; slot < synapse_type.state_variable_names.size(); slot += 1) {
                const f64 initial = starting_value_for(
                        synapse_type.dynamics, synapse_type.state_variable_names[slot],
                        synapse_type.parameter_names, prototype.starting_parameters,
                        synapse_type.name);

                weights->set_edge_variable((s64)slot + 1, source_node, target_node,
                                           (f32)initial);
            }
        }
    }

    logger->debug("SpikeEngine: weight matrix built — {} nodes, {} edges, "
                  "max_neighbor_count {}, {} per-edge variables",
                  weights->node_count, weights->total_edge_count,
                  weights->max_neighbor_count, layout.per_edge_variable_count);
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

            // A spike train is not a current. spikeArray is a spike SOURCE -- it emits
            // events and declares no amplitude at all -- and timedSynapticInput turns its
            // train into current by routing it through a named synapse. Neither mechanism
            // exists here yet, and injecting the component's amplitude (zero, for every
            // standard spike source) would load the model, run it, and write a completely
            // silent recording with nothing but a log line to say why.
            log::throw_runtime_error(*logger,
                    "SpikeEngine: input '" + profile.input_component_id + "' of type '" +
                    profile.input_component_type_name + "' delivers a spike train. A train "
                    "produces current only through a synapse (timedSynapticInput) or as "
                    "presynaptic events (spikeArray), and neither is wired up yet — use a "
                    "pulseGenerator for Phase 1 stimulus rather than have this run silent");
        }
    }

    logger->info("SpikeEngine: stimulus — {} continuous injections",
                 continuous_injection_targets.size());
}

SpikeEngine::~SpikeEngine() {
    if (alive) shutdown();
}

// ── running ───────────────────────────────────────────────────────────────────────

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
}

// The Sk plane of one matrix in the weight family, or the placeholder when that matrix was
// never registered. `matrix_index` is checked rather than trusted: delay_matrix_index is
// -1 until the first connection sets a delay, and indexing sparse_delta_buffers with it is
// an out-of-bounds read that returns a plausible-looking pointer instead of crashing where
// the mistake is.
void *SpikeEngine::resolve_edge_plane(s64 matrix_index) {
    if (matrix_index < 0) return edge_placeholder.get_contents();
    if (matrix_index >= (s64)weights->sparse_delta_buffers.size()) {
        return edge_placeholder.get_contents();
    }

    GpuPointer<f32> &plane = weights->sparse_delta_buffers[(usize)matrix_index];
    if (plane.pointer == nullptr) return edge_placeholder.get_contents();

    return (void *)plane.get_contents();
}

void SpikeEngine::step_simulation(s64 tick) {
    apply_stimulus(tick);

    const s32 neuron_count_argument = (s32)total_neuron_count;
    const s32 spike_history_length_argument = (s32)layout.spike_history_length;
    const s32 max_neighbor_count_argument = (s32)weights->max_neighbor_count;

    void *cell_state_pointer         = cell_state.get_contents();
    void *cell_parameters_pointer    = cell_parameters.get_contents();
    void *synapse_parameters_pointer = synapse_parameters.get_contents();
    void *network_inputs_pointer     = network_inputs.get_contents();
    void *spike_history_pointer      = spike_history.get_contents();
    void *last_spiked_pointer        = last_spiked.get_contents();

    const K2Tree &tree = weights->k2tree;
    void *internal_words_pointer   = (void *)tree.internal_node_words.get_contents();
    void *leaf_words_pointer       = (void *)tree.leaf_node_words.get_contents();
    void *superblock_table_pointer = (void *)tree.rank_superblock_table.get_contents();
    void *subblock_table_pointer   = (void *)tree.rank_subblock_table.get_contents();

    // A model with no connections registers no delay matrix and allocates no per-edge
    // planes, so these are absent rather than empty. The kernel never reads them -- there
    // is no adjacency row to walk -- but every argument still has to be a real registered
    // address, because metal_dispatch binds a pointer it cannot resolve as raw bytes.
    void *edge_weights_pointer = resolve_edge_plane(WeightMatrix::DEFAULT_MATRIX_INDEX);
    void *edge_delays_pointer = resolve_edge_plane(weights->delay_matrix_index);
    void *edge_variables_pointer = weights->per_edge_variable_values.pointer != nullptr
            ? (void *)weights->per_edge_variable_values.get_contents()
            : edge_placeholder.get_contents();

    const void *arguments[] = {
        &tick, &step_dt, &neuron_count_argument, &spike_history_length_argument,
        &max_neighbor_count_argument,
        &cell_state_pointer, &cell_parameters_pointer, &synapse_parameters_pointer,
        &network_inputs_pointer, &spike_history_pointer, &last_spiked_pointer,
        &internal_words_pointer, &leaf_words_pointer,
        &superblock_table_pointer, &subblock_table_pointer,
        &tree.branching_factor, &tree.superblock_size_words, &tree.padded_node_count,
        &tree.tree_height, &tree.internal_bit_count,
        &edge_weights_pointer, &edge_delays_pointer, &edge_variables_pointer,
    };
    const usize argument_sizes[] = {
        sizeof(s64), sizeof(f32), sizeof(s32), sizeof(s32), sizeof(s32),
        sizeof(void *), sizeof(void *), sizeof(void *),
        sizeof(void *), sizeof(void *), sizeof(void *),
        sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *),
        sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s32),
        sizeof(void *), sizeof(void *), sizeof(void *),
    };

    metal_dispatch(master_kernel, LaunchConfig{(u32)block_count, (u32)thread_count_per_block},
                   arguments, argument_sizes,
                   (u32)(sizeof(arguments) / sizeof(arguments[0])));
    synchronize_gpu_work();

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

// ── recording output ──────────────────────────────────────────────────────────────

void SpikeEngine::write_recordings() const {
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

    release_kernel(master_kernel);
    weights.reset();
    allocator = EngineAllocator();

    alive = false;
}
