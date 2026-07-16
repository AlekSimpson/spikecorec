#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <variant>

#include "spikecorec/nml/output_recording.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

s32 population_index_containing(const ModelSpecification &model, s32 global_neuron_index) {
    for (s32 index = 0; index < (s32)model.populations.size(); ++index) {
        const PopulationEntry &population = model.populations[(usize)index];
        if (global_neuron_index >= population.neuron_index_begin && global_neuron_index < population.neuron_index_end) {
            return index;
        }
    }
    return -1;
}

// Position of `storage_name` among `program`'s own `.alloc` StateDirectives,
// in emission order -- matches allocate_model's own indexing of cell_state
// (allocator.cpp's boundary-offset walk counts StateDirectives in this same
// order). -1 if absent.
s32 state_directive_index_of(const IrProgram &program, const String &storage_name) {
    s32 index = 0;
    for (const auto &directive : program.alloc) {
        if (!std::holds_alternative<StateDirective>(directive)) continue;
        if (std::get<StateDirective>(directive).name == storage_name) return index;
        ++index;
    }
    return -1;
}

// Resolves `exposure_name` against `cell`'s own StateVariable/DerivedVariable
// declarations. `storage_name` is the StateVariableDecl's own internal name
// (DirectState only -- DerivedScratch is keyed by exposure_name itself,
// matching cell_lowering's own ExposeDirective naming).
struct ExposureLookup {
    bool found = false;
    bool is_derived = false;
    String storage_name;
};

ExposureLookup lookup_cell_exposure(const CellType &cell, const String &exposure_name) {
    for (const auto &state_variable : cell.state_variables) {
        if (state_variable.exposure == exposure_name) return ExposureLookup{true, false, state_variable.name};
    }
    for (const auto &derived_variable : cell.derived_variables) {
        if (derived_variable.exposure == exposure_name) return ExposureLookup{true, true, String()};
    }
    return ExposureLookup{};
}

} // namespace

Vector<RecordingStream> resolve_recording_streams(
    const ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs
) {
    if (type_library_ir_programs.size() != model.type_library.size()) {
        log::throw_runtime_error(log::logger(),
            "resolve_recording_streams: type_library_ir_programs.size() (" +
            std::to_string(type_library_ir_programs.size()) + ") must match model.type_library.size() (" +
            std::to_string(model.type_library.size()) + ")");
    }

    UnorderedMap<String, s32> stream_index_by_key; // "<population_index>:<exposure_name>:<is_event>"
    Vector<RecordingStream> streams;

    for (s32 recording_index = 0; recording_index < (s32)model.recordings.size(); ++recording_index) {
        const RecordingEntry &recording = model.recordings[(usize)recording_index];

        if (recording.target_neuron_index < 0) {
            log::logger().warn(
                "resolve_recording_streams: recording '{}' has an unresolved target_neuron_index -- skipped",
                recording.id);
            continue;
        }

        s32 population_index = population_index_containing(model, recording.target_neuron_index);
        if (population_index < 0) {
            log::logger().warn(
                "resolve_recording_streams: recording '{}' target_neuron_index {} names no cataloged population -- skipped",
                recording.id, recording.target_neuron_index);
            continue;
        }

        const PopulationEntry &population = model.populations[(usize)population_index];
        const TypeLibraryEntry &cell_entry = model.type_library[(usize)population.type_library_index];

        if (cell_entry.category != TypeLibraryCategory::Cell) {
            log::throw_runtime_error(log::logger(),
                "resolve_recording_streams: population '" + population.id + "' is bound to a non-Cell type "
                "library entry (recordings only ever resolve against a population, arch §4.1/§4.5)");
        }

        String stream_key = std::to_string(population_index) + ":" + recording.exposure_name + ":" +
            (recording.is_event_recording ? "1" : "0");

        auto existing_stream = stream_index_by_key.find(stream_key);
        if (existing_stream != stream_index_by_key.end()) {
            streams[(usize)existing_stream->second].recording_entry_indices.push_back(recording_index);
            continue;
        }

        RecordingStream stream;
        stream.population_index = population_index;
        stream.exposure_name = recording.exposure_name;
        stream.is_event_recording = recording.is_event_recording;

        if (recording.is_event_recording) {
            stream.source_kind = RecordingSourceKind::SpikeRaster;
        } else {
            const CellType &cell = std::get<CellType>(cell_entry.dynamics.flattened);
            ExposureLookup lookup = lookup_cell_exposure(cell, recording.exposure_name);
            if (!lookup.found) {
                log::logger().warn(
                    "resolve_recording_streams: recording '{}' exposure '{}' matches no StateVariable/DerivedVariable "
                    "on population '{}' cell type '{}' -- skipped",
                    recording.id, recording.exposure_name, population.id, cell_entry.component_type_name);
                continue;
            }

            if (lookup.is_derived) {
                stream.source_kind = RecordingSourceKind::DerivedScratch;
                stream.scratch_key = type_scoped_key(population.type_library_index, recording.exposure_name);
            } else {
                const IrProgram &cell_program = type_library_ir_programs[(usize)population.type_library_index];
                s32 state_variable_index = state_directive_index_of(cell_program, lookup.storage_name);
                if (state_variable_index < 0) {
                    log::throw_runtime_error(log::logger(),
                        "resolve_recording_streams: population '" + population.id + "' cell type '" +
                        cell_entry.component_type_name + "' StateVariable '" + lookup.storage_name +
                        "' has no matching .alloc StateDirective (malformed IrProgram)");
                }
                stream.source_kind = RecordingSourceKind::DirectState;
                stream.state_variable_index = state_variable_index;
            }
        }

        stream.recording_entry_indices.push_back(recording_index);
        stream_index_by_key[stream_key] = (s32)streams.size();
        streams.push_back(std::move(stream));
    }

    return streams;
}

void sample_recording_stream(
    const RecordingStream &stream, const ModelSpecification &model, const ModelAllocation &allocation,
    const s64 *last_spiked, s64 current_tick, Vector<f32> &out_values
) {
    const PopulationEntry &population = model.populations[(usize)stream.population_index];
    out_values.assign((usize)population.size, 0.0f);

    switch (stream.source_kind) {
        case RecordingSourceKind::DirectState: {
            const s64 *boundaries = allocation.cell_type_boundaries.get_contents();
            s64 chunk_base_offset = boundaries[stream.population_index];
            const f32 *cell_state = allocation.cell_state.get_contents();
            for (s32 local_index = 0; local_index < population.size; ++local_index) {
                s64 element_index = chunk_base_offset + (s64)stream.state_variable_index * population.size + local_index;
                out_values[(usize)local_index] = cell_state[element_index];
            }
            break;
        }
        case RecordingSourceKind::DerivedScratch: {
            auto scratch_entry = allocation.derived_exposure_scratch_buffers.find(stream.scratch_key);
            if (scratch_entry == allocation.derived_exposure_scratch_buffers.end()) {
                log::throw_runtime_error(log::logger(),
                    "sample_recording_stream: no derived_exposure_scratch_buffers entry for key '" +
                    stream.scratch_key + "'");
            }
            const f32 *scratch = scratch_entry->second.get_contents();
            for (s32 local_index = 0; local_index < population.size; ++local_index) {
                s64 global_index = population.neuron_index_begin + local_index;
                out_values[(usize)local_index] = scratch[global_index];
            }
            break;
        }
        case RecordingSourceKind::SpikeRaster: {
            if (last_spiked == nullptr) {
                log::throw_runtime_error(log::logger(),
                    "sample_recording_stream: a SpikeRaster stream needs a non-null last_spiked buffer");
            }
            for (s32 local_index = 0; local_index < population.size; ++local_index) {
                s64 global_index = population.neuron_index_begin + local_index;
                out_values[(usize)local_index] = (last_spiked[global_index] == current_tick) ? 1.0f : 0.0f;
            }
            break;
        }
    }
}

} // namespace spikecorec::nml
