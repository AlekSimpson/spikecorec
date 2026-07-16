#pragma once

#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── Recording / output selections -> recorder streams (ticket #59 [E2]; arch
// §3.3 ST6, §4.6) ──────────────────────────────────────────────────────────
//
// Ticket #7's ModelSpecification already parses OutputColumn/Line and
// EventSelection into flat RecordingEntry rows (one per NML element, each
// naming exactly one neuron's exposure or spike port -- see
// model_specification.h's own RecordingEntry doc comment). This module takes
// the next step: resolve each entry's exposure_name against its population's
// cell type (arch §4.1's widened per-population state layout, §4.5's
// derived-exposure scratch) into a concrete place to read the value from, and
// collapse entries naming the same (population, exposure, is_event_recording)
// triple into ONE recorder stream that samples that population's WHOLE
// neuron range each recorded tick (arch §4.6: "(population, exposure,
// stride) -> recorder streams" -- "stride" is the variable's stride within
// the population's widened state chunk, the same term §4.1 uses for that
// exact addressing quantity). The existing `.spire`/SimulationRecorder
// machinery (core/recording.h) already samples any per-neuron f32 array of a
// known length -- membrane-potential-style traces and every other exposure
// (and spike rasters, encoded as a 1.0/0.0 mask) all reuse it uniformly, one
// SimulationRecorder instance per resolved stream; no new file format.
//
// Scope boundary (arch §5, same as ticket #58's stimulus scope note):
// master-kernel assembly (#6) isn't merged yet, so nothing here drives a
// live per-tick sampling loop against a running generated kernel -- that's
// #61 (H1)'s job. This module only resolves the mapping and samples a
// snapshot given a tick number and the current buffer contents, both
// host-side; see this ticket's own tests for how a caller would drive it.

// How a resolved stream's value is actually stored/read (arch §4.1/§4.5/§4.6).
enum class RecordingSourceKind {
    DirectState,    // a stored `state` slot -- ModelAllocation::cell_state
    DerivedScratch, // an exposed DerivedVariable with no state slot -- ModelAllocation::derived_exposure_scratch_buffers
    SpikeRaster,    // an EventSelection -- read from last_spiked
};

// One recorder stream: every RecordingEntry naming the same (population,
// exposure_name, is_event_recording) triple collapses into this ONE stream.
struct RecordingStream {
    s32 population_index = -1;
    String exposure_name;
    bool is_event_recording = false;
    RecordingSourceKind source_kind = RecordingSourceKind::DirectState;

    // DirectState only: this variable's position among its population's cell
    // type's `.alloc` StateDirectives (the SoA stride index, arch §4.1) --
    // element location = cell_type_boundaries[population_index] +
    // state_variable_index * population.size + local_neuron_index.
    s32 state_variable_index = -1;

    // DerivedScratch only: the ModelAllocation::derived_exposure_scratch_buffers key.
    String scratch_key;

    // Every model.recordings[] index folded into this stream, for diagnostics.
    Vector<s32> recording_entry_indices;
};

// Resolves every entry in `model.recordings` into a deduplicated set of
// RecordingStreams, in first-seen order. `type_library_ir_programs` is the
// same parallel array allocate_model() takes (one IrProgram per
// model.type_library entry, in order) -- needed to find a StateVariable's
// actual storage slot index the same way the real allocator computes
// cell_state layout (allocator.cpp), rather than assuming it matches the raw
// CellType's own state_variables order. Throws std::runtime_error if
// `type_library_ir_programs.size() != model.type_library.size()`, or if a
// population resolves to a non-Cell type-library entry (recordings only
// ever resolve against a population, per allocator.h's own scratch-buffer
// scoping note).
//
// A RecordingEntry is silently skipped (logged as a warning) when: its
// target_neuron_index is unresolved (-1, see RecordingEntry's own comment --
// a diagnostic-only sentinel there, not fatal here either); or its
// exposure_name matches neither a StateVariable's nor a DerivedVariable's own
// `exposure` name on its population's cell type. A skipped entry produces no
// stream.
Vector<RecordingStream> resolve_recording_streams(
    const ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs);

// Samples `stream`'s current value for every neuron in its population (in
// local-index order) into `out_values` (resized to population.size).
// `last_spiked`/`current_tick` are only read for a SpikeRaster stream (1.0 if
// last_spiked[neuron] == current_tick, else 0.0) -- pass last_spiked ==
// nullptr for a model with no SpikeRaster streams. Throws std::runtime_error
// if a SpikeRaster stream is sampled with last_spiked == nullptr, or if a
// DerivedScratch stream's key is missing from `allocation` (a mismatched
// allocation/stream pairing).
void sample_recording_stream(
    const RecordingStream &stream, const ModelSpecification &model, const ModelAllocation &allocation,
    const s64 *last_spiked, s64 current_tick, Vector<f32> &out_values);

} // namespace spikecorec::nml
