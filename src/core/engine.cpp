//
// Created by Alek Simpson on 5/30/26.
//

#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/kernel_codegen.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/recording.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::log;

namespace {

// One generated-kernel argument, as metal_dispatch wants it: a pointer to the argument's
// own storage plus that storage's size. For a buffer argument the storage holds the data
// pointer; for a scalar it holds the value.
struct KernelArgumentBinding {
    const void *storage = nullptr;
    usize byte_size = 0;
};

// Releases a compiled kernel however its scope is left. The one-shot initialize kernel is
// compiled, dispatched and released inside the constructor, and dispatch_master_kernel
// throws when the generated kernel names an argument the engine does not supply -- a bare
// release_kernel call after the dispatch is skipped on exactly that path and leaks the
// pipeline state.
struct KernelHandleGuard {
    KernelHandle handle;

    explicit KernelHandleGuard(KernelHandle compiled_kernel) : handle(compiled_kernel) {}
    ~KernelHandleGuard() { release_kernel(handle); }

    KernelHandleGuard(const KernelHandleGuard &) = delete;
    KernelHandleGuard &operator=(const KernelHandleGuard &) = delete;
};

// What one sub-range of `byte_count` bytes actually costs in the arena. EngineAllocator
// starts every sub-range on a 16-byte boundary, so rounding each request up to that
// boundary and summing gives the exact pool size: the cursor then lands aligned after
// every allocation and no request ever pays for padding it was not sized for.
u64 arena_cost_of(u64 byte_count) {
    const u64 alignment = EngineAllocator::ALLOCATION_ALIGNMENT;
    return (byte_count + alignment - 1) / alignment * alignment;
}

// An arena sub-range, retyped. The arena hands out GpuPointer<void>; the two backends
// spell a sub-range differently (see EngineAllocator::allocate_gpu), so both spellings are
// carried across rather than cast through one of them.
template <typename ElementType>
GpuPointer<ElementType> allocate_engine_buffer(EngineAllocator &arena, s64 element_count) {
    GpuPointer<void> arena_range = arena.allocate_gpu(sizeof(ElementType), element_count);

    GpuPointer<ElementType> typed_range;
#ifdef SPIKECOREC_CUDA
    typed_range.pointer = static_cast<ElementType *>(arena_range.pointer);
#elif defined(SPIKECOREC_METAL)
    typed_range.pointer = arena_range.pointer;
    typed_range.offset_bytes = arena_range.offset_bytes;
#endif
    return typed_range;
}

// get_contents() dereferences the handle, so a zero-length buffer (which the arena
// answers with a null handle) has to be asked about rather than read.
template <typename ElementType>
ElementType *buffer_contents_or_null(GpuPointer<ElementType> &buffer) {
    return buffer.pointer == nullptr ? nullptr : buffer.get_contents();
}

template <typename ElementType>
void zero_fill_engine_buffer(GpuPointer<ElementType> &buffer, s64 element_count) {
    ElementType *contents = buffer_contents_or_null(buffer);
    if (contents == nullptr || element_count <= 0) return;

    memset(contents, 0, static_cast<usize>(element_count) * sizeof(ElementType));
}

// The NeuroML XSD describes <neuroml> documents. The document parse_lems consumes is a
// LEMS one -- it is the only kind that carries a step size, a duration and a target
// network -- and its <Lems> root has no declaration in that schema, so validating one
// against it always fails. Validation is therefore applied to the documents the schema
// actually describes.
bool document_root_is_neuroml(const String &file_path) {
    xmlDocPtr document = xmlReadFile(file_path.c_str(), nullptr, XML_PARSE_NOBLANKS);
    if (document == nullptr) return false;

    const xmlNodePtr root = xmlDocGetRootElement(document);
    const bool is_neuroml = root != nullptr && root->name != nullptr &&
                            String(reinterpret_cast<const char *>(root->name)) == "neuroml";

    xmlFreeDoc(document);
    return is_neuroml;
}

NML_ParseResult parse_and_validate_model(const String &input_file, EngineLogger &engine_logger) {
    NML_Parser parser;

    if (document_root_is_neuroml(input_file) && !parser.validate_against_schema(input_file)) {
        log::throw_runtime_error(
                engine_logger,
                fmt::format("SpikeEngine: '{}' is not valid NeuroML: {}", input_file,
                            parser.last_schema_validation_errors));
    }

    try {
        return parser.parse_lems(input_file);
    } catch (const std::exception &parse_error) {
        log::throw_runtime_error(
                engine_logger,
                fmt::format("SpikeEngine: could not parse '{}': {}", input_file,
                            parse_error.what()));
    }
}

// WeightMatrix indexes nodes with s32; build_adjacency_list reports the parser's s64 global
// neuron indices, which is the only reason this conversion exists.
vector<vector<s32>> weight_matrix_network_for(const NML_ParseResult &parse_result) {
    // `auto`, because `Vector` is ambiguous at file scope here: spikecorec and
    // spikecorec::log each declare an alias template of that name and both are in scope.
    const auto adjacency = build_adjacency_list(parse_result);

    vector<vector<s32>> network(adjacency.size());
    for (usize source_index = 0; source_index < adjacency.size(); ++source_index) {
        network[source_index].reserve(adjacency[source_index].size());
        for (s64 target_index : adjacency[source_index]) {
            network[source_index].push_back(static_cast<s32>(target_index));
        }
    }

    return network;
}

} // namespace

// ── input event streams ───────────────────────────────────────────────────────

spikecorec::Vector<f64> spikecorec::create_event_stream(
    f64 rate,
    f64 amplitude,
    f64 weight,
    const spikecorec::Vector<s32> &event_ticks,
    bool continuous_current_injection
) {
    if (event_ticks.empty()) return {};

    const s64 last_event_tick = static_cast<s64>(event_ticks.back());
    if (last_event_tick < 0) return {};

    // A NeuroML input component carries an amplitude (a current injector) or a rate (a
    // rate-driven generator), never both; a spikeArray carries neither and its events
    // deliver the target's own <inputW weight>, which defaults to 1.0.
    f64 event_magnitude = weight;
    if (amplitude != 0.0) {
        event_magnitude = amplitude * weight;
    } else if (rate != 0.0) {
        event_magnitude = rate * weight;
    }

    Vector<f64> stream(static_cast<usize>(last_event_tick) + 1, 0.0);

    if (continuous_current_injection) {
        // {start_tick, end_tick}: one window with current flowing across all of it, not
        // two isolated impulses.
        const s64 first_event_tick = event_ticks.front() < 0 ? 0 : static_cast<s64>(event_ticks.front());
        for (s64 tick = first_event_tick; tick <= last_event_tick; ++tick) {
            stream[static_cast<usize>(tick)] = event_magnitude;
        }
        return stream;
    }

    for (s32 event_tick : event_ticks) {
        if (event_tick < 0) continue;
        stream[static_cast<usize>(event_tick)] = event_magnitude;
    }

    return stream;
}

// ── edge aggregation ──────────────────────────────────────────────────────────

spikecorec::Vector<spikecorec::AggregatedNetworkEdge> spikecorec::aggregate_network_edges(
    const NML_ParseResult &parse_result,
    s32 default_delay_tick_count,
    EngineLogger &engine_logger
) {
    Vector<AggregatedNetworkEdge> aggregated_edges;
    UnorderedMap<s64, usize> position_of_pair;

    const s64 neuron_count = (s64)parse_result.neurons.size();

    for (usize source_index = 0; source_index < parse_result.neurons.size(); ++source_index) {
        for (const NetworkEdge &edge : parse_result.neurons[source_index].outgoing_edges) {
            const s32 source_node = (s32)source_index;
            const s32 target_node = (s32)edge.target_neuron_index;

            // An edge that declares no delay resolves to the matrix's own default, so an
            // undeclared delay and an explicitly-declared default one do not read as a
            // conflict with each other.
            const s32 delay_tick_count = edge.delay_tick_count > 0
                                                 ? (s32)edge.delay_tick_count
                                                 : default_delay_tick_count;

            const s64 pair_key = (s64)source_node * neuron_count + (s64)target_node;
            const auto existing = position_of_pair.find(pair_key);
            if (existing == position_of_pair.end()) {
                position_of_pair.emplace(pair_key, aggregated_edges.size());
                aggregated_edges.push_back(AggregatedNetworkEdge{
                        source_node, target_node, (f32)edge.weight, delay_tick_count});
                continue;
            }

            AggregatedNetworkEdge &collapsed = aggregated_edges[existing->second];
            if (collapsed.delay_tick_count != delay_tick_count) {
                log::throw_runtime_error(
                        engine_logger,
                        fmt::format("SpikeEngine: neurons {} -> {} are connected by two edges "
                                    "with different delays ({} and {} ticks). The adjacency "
                                    "holds one slot per ordered pair, so both cannot be "
                                    "represented; give the parallel connections the same "
                                    "delay",
                                    source_node, target_node, collapsed.delay_tick_count,
                                    delay_tick_count));
            }
            collapsed.summed_weight += (f32)edge.weight;
        }
    }

    return aggregated_edges;
}

// ── constructor / destructor ──────────────────────────────────────────────────

SpikeEngine::SpikeEngine(String &neuroml_input_file, bool enable_hebbian_learning)
    : logger(make_logger())
    , network_details(parse_and_validate_model(neuroml_input_file, *logger))
    , weights(weight_matrix_network_for(network_details))
    , total_neuron_count((s64)network_details.neurons.size())
    , input_neuron_count(0)
    , thread_count_per_block(256)
    , block_count(0)
    , hebbian_learning_enabled(enable_hebbian_learning)
    , alive(true)
    // Off on the NeuroML path: every neuron steps every tick, so none of the active-set
    // buffers exist and the arena is not sized for them.
    , active_set_optimization_enabled(false)
{
    // TODO: in the future we can create custom spikecorec component types which enable
    // custom engine features like our builtin hebbian learning mechanism

    const Vector<CellTypeSpecification> &cell_types = network_details.cell_types;

    // ── 1. type-sectioned layout ─────────────────────────────────────────────
    // cell_state holds every neuron of type 0, then every neuron of type 1, and so on;
    // cell_parameters follows the identical scheme. Global neuron indices are NOT
    // renumbered -- Neuron, the adjacency and InputTarget::neuron_index all use the
    // parser's numbering, and renumbering would invalidate every one of them.
    Vector<s64> neuron_count_by_type(cell_types.size(), 0);
    for (const Neuron &neuron : network_details.neurons) {
        if (neuron.cell_type_index < 0 ||
            neuron.cell_type_index >= (s64)cell_types.size()) {
            log::throw_runtime_error(
                    *logger,
                    fmt::format("SpikeEngine: neuron has cell type index {}, which names no "
                                "cell type ({} declared)",
                                neuron.cell_type_index, cell_types.size()));
        }
        neuron_count_by_type[(usize)neuron.cell_type_index] += 1;
    }

    Vector<s64> state_section_start(cell_types.size(), 0);
    Vector<s64> parameter_section_start(cell_types.size(), 0);
    for (usize type_index = 0; type_index < cell_types.size(); ++type_index) {
        state_section_start[type_index] = cell_state_element_count;
        parameter_section_start[type_index] = cell_parameter_element_count;

        cell_state_element_count += neuron_count_by_type[type_index] *
                                    (s64)cell_types[type_index].state_variable_names.size();
        cell_parameter_element_count += neuron_count_by_type[type_index] *
                                        (s64)cell_types[type_index].parameter_names.size();
    }

    // ── 2. edge weights and delays ───────────────────────────────────────────
    // Ahead of the arena because the ring depth the network_inputs allocation is sized by
    // comes out of the delays wired here.
    //
    // The adjacency holds ONE slot per ordered pair, so a model that declares two
    // projections between the same pair -- an AMPA and an NMDA between one cell pair, or two
    // <connectionWD> entries that resolve to it -- has to be collapsed into that one slot
    // before anything is written. Aggregating first is what makes the collapse a decision
    // rather than an accident: writing each edge as it comes re-reads the slot the previous
    // one just wrote and overwrites it, so the last projection declared would silently be
    // the only one that survived.
    const Vector<AggregatedNetworkEdge> aggregated_edges =
            aggregate_network_edges(network_details, weights.constant_delay_ticks, *logger);

    for (const AggregatedNetworkEdge &edge : aggregated_edges) {
        // WeightMatrix has no per-edge weight setter: U/V is one shared low-rank plane, so
        // an exact per-edge value is expressed as the sparse delta that carries the
        // reconstruction onto it (arch section 4.3's Sk), which is what get() reads back.
        const f32 reconstructed_weight = weights.get(edge.source_node, edge.target_node);
        weights.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, edge.source_node,
                                      edge.target_node,
                                      edge.summed_weight - reconstructed_weight);

        weights.set_edge_delay_ticks(edge.source_node, edge.target_node, edge.delay_tick_count);
    }

    // Flattened per-edge delay, staged on the host until the arena exists. Every edge is
    // read back through get_edge_delay_ticks rather than from the model, so the rounding,
    // the >= 1 floor and the constant-delay fallback are decided in exactly one place. The
    // slot convention is WeightMatrix's own: row-major by source neuron, and within a row
    // the neighbour's position in the order get_neighbors enumerates it -- which is the
    // order the generated kernel's k^2-tree walk reproduces.
    const s64 edge_slot_count = weights.node_count * weights.max_neighbor_count;
    Vector<s32> staged_edge_delay_ticks((usize)(edge_slot_count > 0 ? edge_slot_count : 0), 0);

    s32 maximum_edge_delay_ticks = 0;
    if (edge_slot_count > 0) {
        Vector<s32> neighbor_indices((usize)weights.max_neighbor_count, 0);
        for (s64 source_node = 0; source_node < weights.node_count; ++source_node) {
            const s64 degree = weights.get_neighbors(source_node, neighbor_indices.data());
            for (s64 slot = 0; slot < degree; ++slot) {
                const s32 delay_tick_count = weights.get_edge_delay_ticks(
                        (s32)source_node, neighbor_indices[(usize)slot]);

                // The whole delay-ring scheme rests on this: with every delay at least one
                // tick, nothing writes row `tick % ring_depth` while that tick's kernel is
                // running, which is what lets a cell read its slot with a plain load and lets
                // the row be zeroed as a whole row once the dispatch has finished. A zero
                // delay would have a spike land in the row being read and cleared around it.
                // get_edge_delay_ticks already floors at 1 and constant_delay_ticks defaults
                // to 1, so this is a guard against a future change, not a live case.
                if (delay_tick_count < 1) {
                    log::throw_runtime_error(
                            *logger,
                            fmt::format("SpikeEngine: edge {} -> {} has a delay of {} ticks; the "
                                        "network_inputs ring requires every delay to be at least "
                                        "one tick",
                                        source_node, neighbor_indices[(usize)slot],
                                        delay_tick_count));
                }

                staged_edge_delay_ticks[(usize)(source_node * weights.max_neighbor_count + slot)] =
                        delay_tick_count;
                maximum_edge_delay_ticks = std::max(maximum_edge_delay_ticks, delay_tick_count);
            }
        }
    }

    // Strictly greater than the largest delay, so an arrival scheduled this tick never lands
    // in the row being read this tick. Two rows minimum: delay is always at least one
    // tick, so a single-row ring could not represent even the shortest edge.
    network_input_ring_depth = std::max(maximum_edge_delay_ticks + 1, 2);
    const s64 network_input_element_count =
            (s64)network_input_ring_depth * total_neuron_count;

    // ── 3. arena ─────────────────────────────────────────────────────────────
    // Sized to exactly what is allocated below, rounding each sub-range up to the arena's
    // own alignment so the last allocation cannot fall off the end of the slab. Nothing is
    // taken from the CPU pool: both backends hand back host-visible memory, so every
    // buffer is filled in place rather than staged and copied.
    u64 gpu_pool_byte_count =
            arena_cost_of(sizeof(f32) * (u64)cell_state_element_count) +
            arena_cost_of(sizeof(f32) * (u64)cell_parameter_element_count) +
            arena_cost_of(sizeof(f32) * (u64)network_input_element_count) +
            arena_cost_of(sizeof(s32) * (u64)edge_slot_count) +          // edge_delay_ticks
            arena_cost_of(sizeof(s32) * (u64)total_neuron_count) * 4 +  // bases, type, flags
            arena_cost_of(sizeof(s64) * (u64)total_neuron_count);       // last_spiked
    if (hebbian_learning_enabled) {
        gpu_pool_byte_count += arena_cost_of(sizeof(s64) * (u64)total_neuron_count);
    }

    allocator = EngineAllocator(/*cpu_total_bytes=*/0, gpu_pool_byte_count);

    cell_state = allocate_engine_buffer<f32>(allocator, cell_state_element_count);
    cell_parameters = allocate_engine_buffer<f32>(allocator, cell_parameter_element_count);
    network_inputs = allocate_engine_buffer<f32>(allocator, network_input_element_count);
    edge_delay_ticks = allocate_engine_buffer<s32>(allocator, edge_slot_count);
    cell_state_base = allocate_engine_buffer<s32>(allocator, total_neuron_count);
    cell_parameter_base = allocate_engine_buffer<s32>(allocator, total_neuron_count);
    cell_type_index = allocate_engine_buffer<s32>(allocator, total_neuron_count);
    spike_flags = allocate_engine_buffer<s32>(allocator, total_neuron_count);
    last_spiked = allocate_engine_buffer<s64>(allocator, total_neuron_count);
    if (hebbian_learning_enabled) {
        last_tick_updated = allocate_engine_buffer<s64>(allocator, total_neuron_count);
    }

    zero_fill_engine_buffer(cell_state, cell_state_element_count);
    zero_fill_engine_buffer(cell_parameters, cell_parameter_element_count);
    zero_fill_engine_buffer(network_inputs, network_input_element_count);
    zero_fill_engine_buffer(spike_flags, total_neuron_count);
    zero_fill_engine_buffer(last_spiked, total_neuron_count);
    zero_fill_engine_buffer(last_tick_updated, total_neuron_count);

    if (edge_slot_count > 0) {
        memcpy(edge_delay_ticks.get_contents(), staged_edge_delay_ticks.data(),
               (usize)edge_slot_count * sizeof(s32));
    }

    // ── 4. per-neuron scaffolding ────────────────────────────────────────────
    // One load each, so a kernel thread resolves its slot without searching type
    // boundaries.
    s32 *cell_state_base_values = buffer_contents_or_null(cell_state_base);
    s32 *cell_parameter_base_values = buffer_contents_or_null(cell_parameter_base);
    s32 *cell_type_index_values = buffer_contents_or_null(cell_type_index);
    f32 *cell_parameter_values = buffer_contents_or_null(cell_parameters);

    Vector<s64> next_position_in_type(cell_types.size(), 0);
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
        const Neuron &neuron = network_details.neurons[(usize)neuron_index];
        const usize type_index = (usize)neuron.cell_type_index;
        const CellTypeSpecification &cell_type = cell_types[type_index];

        const s64 position_in_type = next_position_in_type[type_index];
        next_position_in_type[type_index] += 1;

        const s64 state_base = state_section_start[type_index] +
                               position_in_type * (s64)cell_type.state_variable_names.size();
        const s64 parameter_base = parameter_section_start[type_index] +
                                   position_in_type * (s64)cell_type.parameter_names.size();

        cell_state_base_values[neuron_index] = (s32)state_base;
        cell_parameter_base_values[neuron_index] = (s32)parameter_base;
        cell_type_index_values[neuron_index] = (s32)type_index;

        // ── 5. starting parameters ───────────────────────────────────────────
        // A population names a prototype, not a type: two prototypes of one type may
        // differ, so the values come from the prototype, in the type's parameter order.
        if (neuron.prototype_index < 0 ||
            neuron.prototype_index >= (s64)network_details.cell_prototypes.size()) {
            log::throw_runtime_error(
                    *logger,
                    fmt::format("SpikeEngine: neuron {} has prototype index {}, which names "
                                "no cell prototype",
                                neuron_index, neuron.prototype_index));
        }

        const ComponentPrototype &prototype =
                network_details.cell_prototypes[(usize)neuron.prototype_index];
        if (prototype.starting_parameters.size() != cell_type.parameter_names.size()) {
            log::throw_runtime_error(
                    *logger,
                    fmt::format("SpikeEngine: prototype '{}' carries {} starting parameters "
                                "but cell type '{}' declares {}",
                                prototype.instance_id, prototype.starting_parameters.size(),
                                cell_type.name, cell_type.parameter_names.size()));
        }

        for (usize slot = 0; slot < prototype.starting_parameters.size(); ++slot) {
            // Real is a union; a parameter value is always floating point.
            cell_parameter_values[parameter_base + (s64)slot] =
                    (f32)prototype.starting_parameters[slot].float64;
        }
    }

    block_count = (s32)((total_neuron_count + thread_count_per_block - 1) /
                        thread_count_per_block);

    // ── 6. initial cell state ────────────────────────────────────────────────
    // The OnStart bodies, run once. Compiled, dispatched and released here: nothing after
    // initialisation ever runs them again.
    const GeneratedKernel initialize_kernel_source = generate_initialize_kernel(network_details);
    {
        // Guarded rather than released on the line after the dispatch: dispatch_master_kernel
        // throws when the generated kernel asks for an argument the engine does not provide,
        // and the pipeline state has to be released on that path too.
        KernelHandleGuard initialize_kernel(
                compile_kernel(initialize_kernel_source.source.c_str(),
                               initialize_kernel_source.function_name.c_str()));
        dispatch_master_kernel(initialize_kernel.handle,
                               initialize_kernel_source.argument_names, /*tick=*/0);
    }

    // ── 7. input event streams ───────────────────────────────────────────────
    for (const SimulationInputConfig &input_profile : network_details.input_profiles) {
        for (const InputTarget &input_target : input_profile.targets) {
            if (input_target.neuron_index < 0 ||
                input_target.neuron_index >= total_neuron_count) {
                continue;
            }

            Vector<f64> stream_values = create_event_stream(
                    input_profile.rate, input_profile.amplitude, input_target.weight,
                    input_target.event_ticks, input_profile.continuous_current_injection);
            if (stream_values.empty()) continue;

            input_neuron_count += 1;
            input_event_streams.push_back(
                    NeuronInputStream{input_target.neuron_index, std::move(stream_values)});
        }
    }

    // ── 8. engine utility values ─────────────────────────────────────────────
    lifetime = network_details.total_tick_count;
    if (network_details.random_seed.has_value()) {
        simulation_seed = network_details.random_seed.value();
    }

    // ── 9. the per-tick master kernel ────────────────────────────────────────
    auto compile_and_own = [](const GeneratedKernel &generated) {
        return SharedPointer<KernelHandle>(
                new KernelHandle(compile_kernel(generated.source.c_str(),
                                                generated.function_name.c_str())),
                [](KernelHandle *compiled_kernel) {
                    release_kernel(*compiled_kernel);
                    delete compiled_kernel;
                });
    };

    const GeneratedKernel tick_kernel_source = generate_tick_kernel(network_details);
    tick_kernel_argument_names = tick_kernel_source.argument_names;
    tick_kernel = compile_and_own(tick_kernel_source);

    // The end-of-tick ring clear, dispatched behind the master kernel every tick. Model
    // independent -- it zeroes one row of network_inputs and reads none of the dynamics --
    // so it is generated from nothing but the backend.
    const GeneratedKernel ring_row_clear_kernel_source = generate_ring_row_clear_kernel();
    ring_row_clear_kernel_argument_names = ring_row_clear_kernel_source.argument_names;
    ring_row_clear_kernel = compile_and_own(ring_row_clear_kernel_source);

    // ── 10. recorders ────────────────────────────────────────────────────────
    recording_profiles = network_details.recording_profiles;

    // A selection names a variable of one neuron; its frame slot is that variable's slot
    // inside the neuron's own cell_state chunk.
    auto flat_cell_state_index = [&](s64 neuron_index, const String &variable_name) -> s64 {
        const Neuron &neuron = network_details.neurons[(usize)neuron_index];
        const CellTypeSpecification &cell_type = cell_types[(usize)neuron.cell_type_index];

        for (usize slot = 0; slot < cell_type.state_variable_names.size(); ++slot) {
            if (cell_type.state_variable_names[slot] == variable_name) {
                return cell_state_base_values[neuron_index] + (s64)slot;
            }
        }

        logger->warn("Recording selects '{}' on neuron {}, which cell type '{}' does not "
                     "declare; recording its first state variable instead",
                     variable_name, neuron_index, cell_type.name);
        return cell_state_base_values[neuron_index];
    };

    for (const RecordingConfig &recording_profile : recording_profiles) {
        for (usize output_index = 0;
             output_index < recording_profile.output_filenames.size(); ++output_index) {
            const bool gathers_spike_flags =
                    output_index < recording_profile.file_output_format.size() &&
                    recording_profile.file_output_format[output_index] ==
                            OutputFileFormat::SPIKE_EVENTS;

            Vector<s64> gathered_indices;
            for (const RecordingSelection &selection : recording_profile.selections) {
                if (selection.neuron_index < 0 ||
                    selection.neuron_index >= total_neuron_count) {
                    continue;
                }
                gathered_indices.push_back(
                        gathers_spike_flags
                                ? selection.neuron_index
                                : flat_cell_state_index(selection.neuron_index,
                                                        selection.variable_name));
            }

            // The parser does not populate `selections` yet. An empty one records every
            // neuron -- its first state variable, the membrane-potential analogue the
            // legacy static recorder wrote, or its spike flag for an event file.
            if (gathered_indices.empty()) {
                for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
                    gathered_indices.push_back(gathers_spike_flags
                                                       ? neuron_index
                                                       : cell_state_base_values[neuron_index]);
                }
            }

            // A ComponentType with no <StateVariable> makes cell_state a zero-element
            // buffer, which the arena answers with a null handle. A stream gathering from it
            // would dereference that null once per tick, in the recorder rather than
            // anywhere near the model that caused it, so it is refused here with the reason
            // named. A spike-event stream reads spike_flags instead and is unaffected.
            if (!gathers_spike_flags && !gathered_indices.empty() &&
                cell_state_element_count == 0) {
                log::throw_runtime_error(
                        *logger,
                        fmt::format("SpikeEngine: recording profile writes '{}' from cell "
                                    "state, but no cell type in this model declares a "
                                    "StateVariable, so there is no cell state to record",
                                    recording_profile.output_filenames[output_index]));
            }

            RecordingStream stream;
            stream.gathers_spike_flags = gathers_spike_flags;
            stream.frame_values.assign(gathered_indices.size(), 0.0f);
            stream.gathered_indices = std::move(gathered_indices);
            stream.recorder = make_unique<SimulationRecorder>(
                    recording_profile.output_filenames[output_index],
                    (s64)stream.frame_values.size());

            recording_streams.push_back(std::move(stream));
        }
    }

    logger->info("SpikeEngine constructed from '{}': neuron_count={} cell_types={} "
                 "cell_state_elements={} cell_parameter_elements={} input_streams={} "
                 "recorders={} lifetime={} hebbian_learning={}",
                 neuroml_input_file, total_neuron_count, cell_types.size(),
                 cell_state_element_count, cell_parameter_element_count,
                 input_event_streams.size(), recording_streams.size(), lifetime,
                 hebbian_learning_enabled);
}

SpikeEngine::~SpikeEngine() {
    logger->info("Spike Engine shutting down.");
    if (alive) shutdown();
}

// ── dispatch ──────────────────────────────────────────────────────────────────

void SpikeEngine::dispatch_master_kernel(
    const KernelHandle &kernel,
    const Vector<String> &argument_names,
    s64 tick,
    MetalCommandBatch *batch
) {
    // Every binding below points at one of these locals, so they all have to outlive the
    // metal_dispatch call at the bottom of this function.
    f32 *cell_state_contents = buffer_contents_or_null(cell_state);
    f32 *cell_parameters_contents = buffer_contents_or_null(cell_parameters);
    f32 *network_inputs_contents = buffer_contents_or_null(network_inputs);
    s64 *last_spiked_contents = buffer_contents_or_null(last_spiked);
    s32 *spike_flags_contents = buffer_contents_or_null(spike_flags);
    s32 *cell_state_base_contents = buffer_contents_or_null(cell_state_base);
    s32 *cell_parameter_base_contents = buffer_contents_or_null(cell_parameter_base);
    s32 *cell_type_index_contents = buffer_contents_or_null(cell_type_index);

    const s32 neuron_count_value = (s32)total_neuron_count;
    const f32 step_dt_value = (f32)network_details.step_dt;
    const s64 tick_value = tick;

    // Stage 6, Propagate, runs inside the generated cell device functions, which walk the
    // adjacency and reconstruct each edge's weight themselves -- so the k^2-tree, the shared
    // U/V basis and the per-edge arrays are all kernel arguments.
    u32 *internal_node_words_contents = buffer_contents_or_null(weights.k2tree.internal_node_words);
    u32 *leaf_node_words_contents = buffer_contents_or_null(weights.k2tree.leaf_node_words);
    u32 *rank_superblock_table_contents =
            buffer_contents_or_null(weights.k2tree.rank_superblock_table);
    u16 *rank_subblock_table_contents = buffer_contents_or_null(weights.k2tree.rank_subblock_table);
    float4 *u_matrix_contents = buffer_contents_or_null(weights.U_matrix);
    float4 *v_matrix_contents = buffer_contents_or_null(weights.V_matrix);
    s32 *edge_delay_ticks_contents = buffer_contents_or_null(edge_delay_ticks);

    // The default matrix's Ck and Sk: the kernel reconstructs an edge as
    // Σ U·Ck·V + Sk, which is exactly what WeightMatrix::get() reports on the host, and
    // where the exact per-edge weights this engine wires in actually live.
    f32 *edge_weight_coefficients_contents = buffer_contents_or_null(
            weights.coefficient_vectors[(usize)WeightMatrix::DEFAULT_MATRIX_INDEX]);
    f32 *edge_weight_deltas_contents = buffer_contents_or_null(
            weights.sparse_delta_buffers[(usize)WeightMatrix::DEFAULT_MATRIX_INDEX]);

    const s32 branching_factor_value = weights.k2tree.branching_factor;
    const s32 superblock_size_words_value = weights.k2tree.superblock_size_words;
    const s32 padded_node_count_value = weights.k2tree.padded_node_count;
    const s32 tree_height_value = weights.k2tree.tree_height;
    const s32 internal_bit_count_value = weights.k2tree.internal_bit_count;
    const s64 rank_float4_stride_value = weights.rank_float4_stride;
    const f32 constant_weight_value = use_constant_weight ? weights.constant_weight : 0.0f;
    const s32 max_neighbor_count_value = (s32)weights.max_neighbor_count;
    const s32 ring_depth_value = network_input_ring_depth;

    // metal_dispatch tells a buffer from a scalar by whether the slot is pointer-sized and
    // its contents resolve to a registered allocation, so a pointer-sized scalar (tick,
    // rank_float4_stride) is only ever misread if its VALUE equals a live buffer address --
    // which a tick index or a stride never is.
    const UnorderedMap<String, KernelArgumentBinding> available_arguments = {
        {"cell_state", {&cell_state_contents, sizeof(cell_state_contents)}},
        {"cell_parameters", {&cell_parameters_contents, sizeof(cell_parameters_contents)}},
        {"network_inputs", {&network_inputs_contents, sizeof(network_inputs_contents)}},
        {"last_spiked", {&last_spiked_contents, sizeof(last_spiked_contents)}},
        {"spike_flags", {&spike_flags_contents, sizeof(spike_flags_contents)}},
        {"cell_state_base", {&cell_state_base_contents, sizeof(cell_state_base_contents)}},
        {"cell_parameter_base",
         {&cell_parameter_base_contents, sizeof(cell_parameter_base_contents)}},
        {"cell_type_index", {&cell_type_index_contents, sizeof(cell_type_index_contents)}},
        {"neuron_count", {&neuron_count_value, sizeof(neuron_count_value)}},
        {"dt", {&step_dt_value, sizeof(step_dt_value)}},
        {"tick", {&tick_value, sizeof(tick_value)}},

        {"internal_node_words", {&internal_node_words_contents, sizeof(internal_node_words_contents)}},
        {"leaf_node_words", {&leaf_node_words_contents, sizeof(leaf_node_words_contents)}},
        {"rank_superblock_table",
         {&rank_superblock_table_contents, sizeof(rank_superblock_table_contents)}},
        {"rank_subblock_table",
         {&rank_subblock_table_contents, sizeof(rank_subblock_table_contents)}},
        {"U_matrix", {&u_matrix_contents, sizeof(u_matrix_contents)}},
        {"V_matrix", {&v_matrix_contents, sizeof(v_matrix_contents)}},
        {"edge_weight_coefficients",
         {&edge_weight_coefficients_contents, sizeof(edge_weight_coefficients_contents)}},
        {"edge_weight_deltas",
         {&edge_weight_deltas_contents, sizeof(edge_weight_deltas_contents)}},
        {"edge_delay_ticks", {&edge_delay_ticks_contents, sizeof(edge_delay_ticks_contents)}},
        {"branching_factor", {&branching_factor_value, sizeof(branching_factor_value)}},
        {"superblock_size_words",
         {&superblock_size_words_value, sizeof(superblock_size_words_value)}},
        {"padded_node_count", {&padded_node_count_value, sizeof(padded_node_count_value)}},
        {"tree_height", {&tree_height_value, sizeof(tree_height_value)}},
        {"internal_bit_count", {&internal_bit_count_value, sizeof(internal_bit_count_value)}},
        {"rank_float4_stride", {&rank_float4_stride_value, sizeof(rank_float4_stride_value)}},
        {"constant_weight", {&constant_weight_value, sizeof(constant_weight_value)}},
        {"max_neighbor_count", {&max_neighbor_count_value, sizeof(max_neighbor_count_value)}},
        {"ring_depth", {&ring_depth_value, sizeof(ring_depth_value)}},
    };

    Vector<const void *> argument_storage;
    Vector<usize> argument_byte_sizes;
    argument_storage.reserve(argument_names.size());
    argument_byte_sizes.reserve(argument_names.size());

    for (const String &argument_name : argument_names) {
        const auto binding = available_arguments.find(argument_name);
        if (binding == available_arguments.end()) {
            log::throw_runtime_error(
                    *logger,
                    fmt::format("dispatch_master_kernel: the generated kernel asks for "
                                "argument '{}', which this engine does not provide",
                                argument_name));
        }

        argument_storage.push_back(binding->second.storage);
        argument_byte_sizes.push_back(binding->second.byte_size);
    }

    metal_dispatch(kernel, LaunchConfig{(u32)block_count, (u32)thread_count_per_block},
                   argument_storage.data(), argument_byte_sizes.data(),
                   (u32)argument_storage.size(), batch);
}

// ── per-tick loop ─────────────────────────────────────────────────────────────

void SpikeEngine::step_simulation(s64 tick) {
    if (!alive || !tick_kernel || !ring_row_clear_kernel) {
        log::throw_runtime_error(
                *logger, fmt::format("step_simulation: engine is not running (tick={})", tick));
    }

    logger->trace("step_simulation: tick={}", tick);

    // Stage 1, Deliver. Each input's stream carries its magnitude at the ticks it fires
    // and zero everywhere else, so this tick's contribution is just the value at index
    // `tick`. A stream shorter than the simulation has already finished.
    //
    // External stimulus carries no synaptic delay, so it goes straight into the row the
    // kernel is about to read -- this tick's.
    f32 *network_input_values = buffer_contents_or_null(network_inputs);
    const s64 current_ring_row_base =
            (tick % (s64)network_input_ring_depth) * total_neuron_count;
    for (const NeuronInputStream &input_stream : input_event_streams) {
        if (tick < 0 || tick >= (s64)input_stream.values.size()) continue;

        network_input_values[current_ring_row_base + input_stream.neuron_index] +=
                (f32)input_stream.values[(usize)tick];
    }

    // Stages 2 through 5 -- Integrate, Detect, Emit, Reset -- are the generated dynamics, and
    // stage 6, Propagate, is emitted into every generated cell device function as identical
    // boilerplate, so it runs inside the same kernel and must NOT be dispatched separately.
    // The generated kernel also lowers each neuron's spike flag itself, at the top of the
    // master kernel, so no host write lands between two launches.
    //
    // Behind it, in the same command batch, the ring clear zeroes row tick % ring_depth of
    // network_inputs: the row this tick's cells just read, and only that row. Batched rather
    // than dispatched separately because two encodes in one Metal command buffer execute in
    // order, and the whole point is that the clear happens after every cell has read the row
    // and after propagation has finished scheduling into the OTHER rows, which it must not
    // touch -- those hold arrivals not yet due.
    MetalCommandBatch *tick_batch = begin_command_batch();
    dispatch_master_kernel(*tick_kernel, tick_kernel_argument_names, tick, tick_batch);
    dispatch_master_kernel(*ring_row_clear_kernel, ring_row_clear_kernel_argument_names, tick,
                           tick_batch);
    commit_command_batch(tick_batch);

    // Stage 8, Record.
    record_tick_frames();
}

void SpikeEngine::record_tick_frames() {
    if (recording_streams.empty()) return;

    // Asked for rather than read: a zero-element buffer is a null handle, which get_contents()
    // would dereference.
    const f32 *cell_state_values = buffer_contents_or_null(cell_state);
    const s32 *spike_flag_values = buffer_contents_or_null(spike_flags);

    for (RecordingStream &stream : recording_streams) {
        if (stream.gathered_indices.empty()) continue;

        // The constructor refuses to build a stream over a buffer this model does not have,
        // so this reports the two having drifted apart rather than gathering from a null.
        const bool source_buffer_exists = stream.gathers_spike_flags
                                                  ? spike_flag_values != nullptr
                                                  : cell_state_values != nullptr;
        if (!source_buffer_exists) {
            log::throw_runtime_error(
                    *logger,
                    fmt::format("record_tick_frames: a stream gathers {} values, which this "
                                "model allocated no buffer for",
                                stream.gathers_spike_flags ? "spike flag" : "cell state"));
        }

        for (usize slot = 0; slot < stream.gathered_indices.size(); ++slot) {
            const usize gathered_index = (usize)stream.gathered_indices[slot];
            stream.frame_values[slot] = stream.gathers_spike_flags
                                                ? (f32)spike_flag_values[gathered_index]
                                                : cell_state_values[gathered_index];
        }

        stream.recorder->record_frame(stream.frame_values.data(),
                                      (s64)stream.frame_values.size());
    }
}

void SpikeEngine::shutdown() {
    if (!alive) return;

    logger->info("shutdown: closing recorders and releasing the compiled kernels");

    for (RecordingStream &stream : recording_streams) {
        if (stream.recorder) stream.recorder->finish();
    }

    // Runs the deleter installed at compile time, which releases the pipeline state.
    tick_kernel.reset();
    ring_row_clear_kernel.reset();

    // Nothing else is released here on purpose. Every engine buffer is an EngineAllocator
    // sub-range that aliases the arena's slab and owns nothing; passing one to deallocate()
    // would take the whole slab down with it. The arena releases both slabs itself, when
    // the engine is destroyed.

    alive = false;
}

// ── legacy, pending rework ────────────────────────────────────────────────────
// Everything below predates the NeuroML path and depends on the hardcoded-LIF members and
// buffers now commented out in engine.h (resting_membrane_potential, decay_rate,
// learning_rate, spike_threshold, spike_period, membrane_potentials, the active set, the
// input-neuron indices, cell_state_logs). Commented out rather than deleted or
// half-adapted, so the owner can rework each one deliberately.

// ── default constructor (hardcoded 15x15 random-outdegree reservoir) ──────────
// SpikeEngine::SpikeEngine()
//     : logger(make_logger())
//     , weights(random_fixed_outdegree(15), 1, true)
//     , neuron_count(15 * 15)
//     , input_neuron_count(0)
//     , thread_count_per_block(256)
//     , block_count(0)
//     , resting_membrane_potential(0.1f)
//     , decay_rate(0.01f)
//     , learning_rate(0.0f) // plasticity off by default
//     , spike_period(1)
//     , spike_threshold(1.0f)
//     , alive(true)
//     , active_set_optimization_enabled(true)
// {
//
//     block_count = (s32) ((neuron_count + thread_count_per_block - 1) / thread_count_per_block);
//
//     usize neuron_f32_byte_size = (usize) neuron_count * sizeof(f32);
//     usize neuron_s32_byte_size = (usize) neuron_count * sizeof(s32);
//     usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);
//
//     network_inputs = allocate<f32>(neuron_f32_byte_size);
//     memset(network_inputs.get_contents(), 0, neuron_f32_byte_size);
//
//     membrane_potentials = allocate<f32>(neuron_f32_byte_size);
//     std::fill(membrane_potentials.get_contents(),
//             membrane_potentials.get_contents() + neuron_count,
//             resting_membrane_potential);
//
//     last_spiked = allocate<s64>(neuron_s64_byte_size);
//     memset(last_spiked.get_contents(), 0, neuron_s64_byte_size);
//
//     last_tick_updated = allocate<s64>(neuron_s64_byte_size);
//     memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);
//
//     active_neuron_indices = allocate<s32>(neuron_s32_byte_size);
//     next_active_neuron_indices = allocate<s32>(neuron_s32_byte_size);
//
//     active_neuron_count = allocate<s32>(sizeof(s32));
//     next_active_neuron_count = allocate<s32>(sizeof(s32));
//     active_neuron_count.get_contents()[0] = 0;
//     next_active_neuron_count.get_contents()[0] = 0;
//
//     active_generation = allocate<s32>(neuron_s32_byte_size);
//     s32 *active_generation_data = active_generation.get_contents();
//     for (s64 neuron_index = 0; neuron_index < neuron_count; ++neuron_index)
//         active_generation_data[neuron_index] = -1;
//
//     input_staging = allocate<f32>(neuron_f32_byte_size);
//     override_staging = allocate<s64>(neuron_s64_byte_size);
//
//     prefetch_to_gpu(network_inputs, neuron_f32_byte_size);
//     prefetch_to_gpu(membrane_potentials, neuron_f32_byte_size);
//     prefetch_to_gpu(last_spiked, neuron_s64_byte_size);
//     prefetch_to_gpu(last_tick_updated, neuron_s64_byte_size);
//     prefetch_to_gpu(active_neuron_indices, neuron_s32_byte_size);
//     prefetch_to_gpu(next_active_neuron_indices, neuron_s32_byte_size);
//     prefetch_to_gpu(active_neuron_count, sizeof(s32));
//     prefetch_to_gpu(next_active_neuron_count, sizeof(s32));
//     prefetch_to_gpu(active_generation, neuron_s32_byte_size);
//     prefetch_to_gpu(input_staging, neuron_f32_byte_size);
//     prefetch_to_gpu(override_staging, neuron_s64_byte_size);
//
//     logger->debug("SpikeEngine buffers allocated: neuron_f32_byte_size={} neuron_s32_byte_size={} "
//                   "neuron_s64_byte_size={} thread_count_per_block={} block_count={}",
//                   neuron_f32_byte_size, neuron_s32_byte_size, neuron_s64_byte_size,
//                   thread_count_per_block, block_count);
//     logger->info("SpikeEngine constructed: neuron_count={} resting_mp={} decay_rate={} learning_rate={}",
//                   neuron_count, resting_membrane_potential, decay_rate, learning_rate);
// }

// ── adjacency-list constructor ───────────────────────────────────────────────
// SpikeEngine::SpikeEngine(
//     vector<vector<s32>> *network,
//     const vector<s64> &shape,
//     s64 rank,
//     f32 resting_mp,
//     f32 decay_rate,
//     f32 learning_rate,
//     bool plasticity_enabled,
//     bool active_set_optimization_enabled
// )   : logger(make_logger())
//     , weights(*network, rank, true)
//     , neuron_count(shape[0] * shape[1])
//     , input_neuron_count(0)
//     , thread_count_per_block(256)
//     , block_count(0)
//     , resting_membrane_potential(resting_mp)
//     , decay_rate(decay_rate)
//     , learning_rate(learning_rate)
//     , spike_period(1)
//     , spike_threshold(1.0f)
//     , alive(true)
//     , active_set_optimization_enabled(active_set_optimization_enabled)
// {
//     if (!plasticity_enabled && learning_rate > 0.0f) {
//         throw std::runtime_error(
//                 "Spike engine cannot be initialized with learning rate > 0.0f"
//                 + " while plasticity is disabled.");
//     }
//
//     if (!plasticity_enabled) learning_rate = 0.0f;
//
//
//     block_count = (s32) ((neuron_count + thread_count_per_block - 1) / thread_count_per_block);
//
//     usize neuron_f32_byte_size = (usize) neuron_count * sizeof(f32);
//     usize neuron_s32_byte_size = (usize) neuron_count * sizeof(s32);
//     usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);
//
//     network_inputs = allocate<f32>(neuron_f32_byte_size);
//     memset(network_inputs.get_contents(), 0, neuron_f32_byte_size);
//
//     membrane_potentials = allocate<f32>(neuron_f32_byte_size);
//     std::fill(membrane_potentials.get_contents(),
//             membrane_potentials.get_contents() + neuron_count,
//             resting_membrane_potential);
//
//     last_spiked = allocate<s64>(neuron_s64_byte_size);
//     memset(last_spiked.get_contents(), 0, neuron_s64_byte_size);
//
//     last_tick_updated = allocate<s64>(neuron_s64_byte_size);
//     memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);
//
//     active_neuron_indices = allocate<s32>(neuron_s32_byte_size);
//     next_active_neuron_indices = allocate<s32>(neuron_s32_byte_size);
//
//     active_neuron_count = allocate<s32>(sizeof(s32));
//     next_active_neuron_count = allocate<s32>(sizeof(s32));
//     active_neuron_count.get_contents()[0] = 0;
//     next_active_neuron_count.get_contents()[0] = 0;
//
//     active_generation = allocate<s32>(neuron_s32_byte_size);
//     s32 *active_generation_data = active_generation.get_contents();
//     for (s64 neuron_index = 0; neuron_index < neuron_count; ++neuron_index)
//         active_generation_data[neuron_index] = -1;
//
//     input_staging = allocate<f32>(neuron_f32_byte_size);
//     override_staging = allocate<s64>(neuron_s64_byte_size);
//
//     prefetch_to_gpu(network_inputs, neuron_f32_byte_size);
//     prefetch_to_gpu(membrane_potentials, neuron_f32_byte_size);
//     prefetch_to_gpu(last_spiked, neuron_s64_byte_size);
//     prefetch_to_gpu(last_tick_updated, neuron_s64_byte_size);
//     prefetch_to_gpu(active_neuron_indices, neuron_s32_byte_size);
//     prefetch_to_gpu(next_active_neuron_indices, neuron_s32_byte_size);
// prefetch_to_gpu(active_neuron_count, sizeof(s32));
//     prefetch_to_gpu(next_active_neuron_count, sizeof(s32));
//     prefetch_to_gpu(active_generation, neuron_s32_byte_size);
//     prefetch_to_gpu(input_staging, neuron_f32_byte_size);
//     prefetch_to_gpu(override_staging, neuron_s64_byte_size);
//
//     logger->debug("SpikeEngine buffers allocated: neuron_f32_byte_size={} neuron_s32_byte_size={} "
//                   "neuron_s64_byte_size={} thread_count_per_block={} block_count={}",
//                   neuron_f32_byte_size, neuron_s32_byte_size, neuron_s64_byte_size,
//                   thread_count_per_block, block_count);
//     logger->info("SpikeEngine constructed: neuron_count={} resting_mp={} decay_rate={} learning_rate={}",
//                   neuron_count, resting_mp, decay_rate, learning_rate);
// }

// ── plasticity toggles / lifetime / input neurons / reset ────────────────────
// bool SpikeEngine::plasticity_enabled() {
//     return learning_rate > 0.0f;
// }
//
// void SpikeEngine::enable_plasticity(f32 _learning_rate) {
//     if (plasticity_enabled()) return;
//
//     learning_rate = _learning_rate;
// }
//
// void SpikeEngine::disable_plasticity() {
//     if (!plasticity_enabled()) return;
//
//     learning_rate = 0.0f;
// }
//
// void SpikeEngine::setup_lifetime(int lifetime_, bool allocate_logs, s64 max_log_bytes) {
//     logger->debug("setup_lifetime: lifetime={} allocate_logs={} max_log_bytes={}",
//                   lifetime_, allocate_logs, max_log_bytes);
//     lifetime = lifetime_;
//     if (lifetime < 0 || !allocate_logs) {
//         logger->info("Not allocating logs for run data.");
//         return;
//     }
//
//     s32 size_of_f32 = 4;
//     s64 required_bytes = neuron_count * lifetime * size_of_f32;
//     if (max_log_bytes < required_bytes) {
//         throw_runtime_error(*logger,
//             fmt::format("setup_lifetime: refusing to allocate membrane potential log "
//                         "({} neurons x {} ticks = {} bytes exceeds {}-byte budget; "
//                         "pass a larger max_log_bytes to enable recording)",
//                         neuron_count, lifetime, required_bytes, max_log_bytes));
//     }
//
//     cell_state_logs = new f32*[neuron_count];
//     for (s64 i = 0; i < neuron_count; ++i)
//         cell_state_logs[i] = new f32[lifetime];
//
//     logger->debug("setup_lifetime: allocated cell_state_logs for {} neurons x {} ticks",
//                   neuron_count, lifetime);
// }
//
// void SpikeEngine::set_input_neurons(const vector<s32> &input_neuron_list) {
//     logger->debug("set_input_neurons: input_neuron_count={}", input_neuron_list.size());
//     if (input_neuron_list.empty()) return;
//
//     s32 s32_byte_size = 4;
//     if (input_neuron_indices.pointer != nullptr) deallocate(std::move(input_neuron_indices));
//     input_neuron_indices = allocate<s32>(neuron_count * s32_byte_size);
//     memcpy(
//         input_neuron_indices.get_contents(),
//         input_neuron_list.data(),
//         input_neuron_list.size() * s32_byte_size
//     );
//     input_neuron_count = (s64) input_neuron_list.size();
//
//     prefetch_to_gpu(input_neuron_indices, (usize)neuron_count * s32_byte_size);
// }
//
// void SpikeEngine::reset_state(s64 last_spiked_value, s32 active_gen_value) {
//     logger->debug("reset_state: last_spiked_value={} active_gen_value={}",
//                   last_spiked_value, active_gen_value);
//     s32 f32_byte_size = 4;
//     usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);
//
//     memset(network_inputs.get_contents(), 0, (usize) neuron_count * f32_byte_size);
//     std::fill(membrane_potentials.get_contents(),
//               membrane_potentials.get_contents() + neuron_count,
//               resting_membrane_potential);
//     std::fill(last_spiked.get_contents(),
//               last_spiked.get_contents() + neuron_count,
//               last_spiked_value);
//     memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);
//     active_neuron_count.get_contents()[0] = 0;
//     next_active_neuron_count.get_contents()[0] = 0;
//     std::fill(active_generation.get_contents(),
//               active_generation.get_contents() + neuron_count,
//               active_gen_value);
// }

// ── the legacy multi-argument step_simulation ────────────────────────────────
// void SpikeEngine::step_simulation(
//     const vector<f32> &input_values,
//     s64 tick,
//     const vector<s64> &override_input_neurons,
//     bool decay_all_neurons
// ) {
//     if (input_values.empty()) {
//         log::throw_runtime_error(*logger,
//                 fmt::format("step_simulation: input_values is empty (tick={})", tick));
//     }
//
//     logger->trace("step_simulation: tick={} input_values.size={} override_input_neurons.size={}"
//                   + " decay_all_neurons={}",
//                   tick, input_values.size(), override_input_neurons.size(), decay_all_neurons);
//
//     MetalCommandBatch *batch = begin_command_batch();
//
//     if (decay_all_neurons) {
//         gpu_decay_all_neurons(
//             membrane_potentials.get_contents(),
//             last_tick_updated.get_contents(),
//             neuron_count,
//             tick,
//             resting_membrane_potential,
//             decay_rate,
//             batch);
//     }
//
//     memcpy(input_staging.get_contents(),
//            input_values.data(), input_values.size() * sizeof(f32));
//
//     // QUESTION: anyway to combine these two steps?
//     // what exactly is the merge_input_neurons for?
//     gpu_add_network_input(
//         membrane_potentials.get_contents(),
//         input_neuron_indices.get_contents(),
//         input_staging.get_contents(),
//         (s64)input_values.size(),
//         batch);
//
//     next_active_neuron_count.get_contents()[0] = 0;
//
//     if (!override_input_neurons.empty()) {
//         memcpy(override_staging.get_contents(),
//                override_input_neurons.data(), override_input_neurons.size() * sizeof(s64));
//
//         gpu_merge_input_neurons(
//             active_neuron_indices.get_contents(),
//             active_neuron_count.get_contents(),
//             override_staging.get_contents(),
//             (s64)override_input_neurons.size(),
//             batch);
//     }
//
//     gpu_step(
//         tick,
//         tick + 1,
//         spike_period,
//         spike_threshold,
//         learning_rate,
//         decay_rate,
//         resting_membrane_potential,
//         weights.U_matrix.get_contents(),
//         weights.V_matrix.get_contents(),
//         weights.rank_float4_stride,
//         use_constant_weight ? weights.constant_weight : 0.0,
//         weights.k2tree.internal_node_words.get_contents(),
//         weights.k2tree.leaf_node_words.get_contents(),
//         weights.k2tree.rank_superblock_table.get_contents(),
//         weights.k2tree.rank_subblock_table.get_contents(),
//         weights.k2tree.branching_factor,
//         weights.k2tree.superblock_size_words,
//         weights.k2tree.padded_node_count,
//         weights.k2tree.tree_height,
//         weights.k2tree.internal_bit_count,
//         neuron_count,
//         network_inputs.get_contents(),
//         membrane_potentials.get_contents(),
//         last_spiked.get_contents(),
//         last_tick_updated.get_contents(),
//         active_neuron_indices.get_contents(),
//         active_neuron_count.get_contents(),
//         next_active_neuron_indices.get_contents(),
//         next_active_neuron_count.get_contents(),
//         active_generation.get_contents(),
//         active_set_optimization_enabled,
//         thread_count_per_block,
//         block_count,
//         batch);
//
//     commit_command_batch(batch);
//
//     std::swap(active_neuron_indices, next_active_neuron_indices);
//     std::swap(active_neuron_count, next_active_neuron_count);
//
//     logger->trace("step_simulation: tick={} completed, active_neuron_count={}",
//                   tick, active_neuron_count.get_contents()[0]);
// }

// ── static recording, bifurcation estimates, reservoir features ──────────────
// void SpikeEngine::start_static_record(
//     const vector<vector<f32>> &input_spikes,
//     s64 lifetime,
//     const string &filename,
//     bool record_membrane,
//     s64 record_stride,
//     optional<string> compression,
//     optional<int> compression_level,
//     bool full_decay,
//     bool compression_async,
//     usize compression_queue_max,
//     usize compression_chunk_bytes
// ) {
//     if (lifetime < 0) {
//         log::throw_runtime_error(*logger,
//                 fmt::format("start_static_record: lifetime must be >= 0 (got {})", lifetime));
//     }
//     if (record_stride < 1) {
//         log::throw_runtime_error(*logger,
//             fmt::format("start_static_record: record_stride must be >= 1 (got {})",
//                         record_stride));
//     }
//     if (input_neuron_count <= 0) {
//         log::throw_runtime_error(*logger,
//             "start_static_record: no input neurons configured (call set_input_neurons first)");
//     }
//     if ((s64)input_spikes.size() < lifetime) {
//         log::throw_runtime_error(*logger,
//             fmt::format("start_static_record: input_spikes has {} " +
//                         "ticks but lifetime requires {}",
//                         input_spikes.size(), lifetime));
//     }
//
//     // Each tick's input row is positionally matched to input_neuron_indices, so
//     // it must contain exactly input_neuron_count values.
//     for (s64 tick = 0; tick < lifetime; ++tick) {
//         if ((s64)input_spikes[(usize)tick].size() != input_neuron_count) {
//             log::throw_runtime_error(*logger,
//                 fmt::format("start_static_record: input_spikes[{}] has {} values but there are {} input neurons",
//                             tick, input_spikes[(usize)tick].size(), input_neuron_count));
//         }
//     }
//
//     logger->info("start_static_record: lifetime={} filename={}", lifetime, filename);
//
//     SimulationRecorder recorder(
//         filename, neuron_count, compression, compression_level,
//         compression_async, compression_queue_max, compression_chunk_bytes);
//
//     // Forces input neurons into the active set every tick regardless of
//     // whether they're already active
//     vector<s64> override_input_neurons((usize)input_neuron_count);
//     const s32 *input_indices = input_neuron_indices.get_contents();
//     for (s64 i = 0; i < input_neuron_count; ++i)
//         override_input_neurons[(usize)i] = (s64)input_indices[i];
//
//     for (s64 tick = 0; tick < lifetime; ++tick) {
//         step_simulation(input_spikes[(usize)tick], tick, override_input_neurons, /*decay_all_neurons=*/false);
//
//         if (record_membrane && tick % record_stride == 0) {
//             // _decay_all(tick) runs after step() and only on recorded ticks,
//             // immediately before the membrane-potential snapshot
//             if (full_decay) {
//                 gpu_decay_all_neurons(
//                     membrane_potentials.get_contents(),
//                     last_tick_updated.get_contents(),
//                     neuron_count,
//                     tick,
//                     resting_membrane_potential,
//                     decay_rate);
//             }
//
//             synchronize_gpu_work();
//             prefetch_to_cpu(membrane_potentials, (usize)neuron_count * sizeof(f32));
//             recorder.record_frame(membrane_potentials.get_contents(), neuron_count);
//         }
//     }
//
//     recorder.finish();
//     logger->info("start_static_record: finished");
// }
//
// pair<f32, f32> SpikeEngine::estimate_bifurcation_weight(s32 input_period) const {
//     f32 decay_factor = std::pow(1.0f - decay_rate, (f32)input_period);
//     f32 w_accum = (spike_threshold - resting_membrane_potential) * (1.0f - decay_factor);
//     f32 w_instant = spike_threshold - resting_membrane_potential;
//     return make_pair(w_accum, w_instant);
// }
//
// void SpikeEngine::scale_uniform_weights_near_bifurcation(
//     f32 *target, f32 *w_accum, f32 *w_instant,
//     s32 input_period, f32 scale, bool freeze_learning, const bool *use_constant_weight_
// ) {
//     auto [w_accum_, w_instant_] = estimate_bifurcation_weight(input_period);
//     *w_accum = w_accum_;
//     *w_instant = w_instant_;
//     *target = *w_accum * scale;
//     weights.set_constant_weight(*target);
//
//     if (freeze_learning) {
//         learning_rate = 0.0f;
//     }
//
//     use_constant_weight = use_constant_weight_ != nullptr
//         ? *use_constant_weight_
//         : freeze_learning;
//
//     logger->debug("scale_uniform_weights_near_bifurcation: input_period={} scale={} target={} "
//                   "w_accum={} w_instant={} use_constant_weight={}",
//                   input_period, scale, *target, *w_accum, *w_instant, use_constant_weight);
// }
//
// ScaledReservoirResult SpikeEngine::scale_randomized_weights_near_bifurcation(
//     s32 input_period,
//     f32 scale,
//     bool freeze_learning
// ) {
//     auto [w_accum, w_instant] = estimate_bifurcation_weight(input_period);
//     f32 target = abs(w_accum * scale);
//     ScaleResult result = weights.scale_neighbor_weights_to_root_mean_square(target);
//
//     use_constant_weight = false;
//
//     if (freeze_learning) {
//         learning_rate = 0.0f;
//     }
//
//     logger->debug("scale_randomized_weights_near_bifurcation: " +
//                   "input_period={} scale={} target={} " +
//                   "w_accum={} w_instant={}",
//                   input_period, scale, target, w_accum, w_instant);
//
//     return ScaledReservoirResult{result,w_accum,w_instant};
// }
//
// void SpikeEngine::get_reservoir_features_vector(
//     s64 tick,
//     f32 spike_tau,
//     f32 voltage_scale,
//     GpuPointer<f32> output_buffer
// ) {
//     logger->trace("get_reservoir_features_vector: tick={} spike_tau={} voltage_scale={}",
//                   tick, spike_tau, voltage_scale);
//     if (spike_tau <= 0.0f) {
//         logger->warn("get_reservoir_features_vector: spike_tau was <= 0.0. Aborting.");
//         return;
//     }
//     if (voltage_scale <= 0.0f) {
//         logger->warn("get_reservoir_features_vector: voltage_scale was <= 0.0. Aborting.");
//         return;
//     }
//     gpu_reservoir_features(
//         neuron_count,
//         tick,
//         spike_tau,
//         voltage_scale,
//         membrane_potentials.get_contents(),
//         last_spiked.get_contents(),
//         last_tick_updated.get_contents(),
//         resting_membrane_potential,
//         decay_rate,
//         output_buffer.get_contents());
// }
