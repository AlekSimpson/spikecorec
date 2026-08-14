//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine_allocator.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/dynamics_codegen.h"

using namespace std;
using namespace spikecorec::nml;

// spikecorec::log is qualified rather than pulled in: it re-declares Vector and String, so
// `using namespace spikecorec;` alongside `using namespace spikecorec::log;` makes every
// bare Vector ambiguous in any translation unit that includes this header.

namespace spikecorec {

    // What last_spiked holds for a neuron that has not fired yet. A refractory gate
    // compares tick - last_spiked against the cell's refractory time, so this has to read
    // as "long ago" rather than as tick 0.
    constexpr s64 NEVER_SPIKED_TICK = -((s64)1 << 32);

    // One spike, as the model asked for it to be recorded.
    struct RecordedSpike {
        f64 time_seconds = 0.0;
        s64 neuron_index = -1;
    };

    // A simulation built entirely from a LEMS/NeuroML document: the model decides the cell
    // and synapse dynamics, the network, the stimulus, how long to run and what to record.
    // Nothing about the dynamics is compiled into the engine — the tick kernel is generated
    // from the model's ComponentTypes at construction and compiled by the backend.
    class SpikeEngine {
    public:
        log::SharedPointer<log::EngineLogger> logger;

        // Two slabs carved into every buffer below. Declared before them because it owns
        // the storage they alias, so it must outlive them.
        EngineAllocator allocator;

        // Holds every per-edge quantity: the exact connection weight, the per-edge delay,
        // and one plane per synapse state variable — all indexed by the k^2-tree's own
        // adjacency, so only real edges are ever addressed. Default-constructed until the
        // model has been parsed, and left that way for a model whose populations are never
        // wired together: an empty weight matrix is a network with no connections, which
        // is an ordinary state rather than an absent one.
        WeightMatrix weights;

        NML_ParseResult network_details;
        ModelLayout layout;

        // ── model state, carved from the arena ────────────────────────────────────
        GpuPointer<void> cell_state;         // [layout.cell_state_length]
        GpuPointer<void> cell_parameters;    // [layout.cell_parameter_length]
        GpuPointer<void> synapse_parameters; // [layout.synapse_parameter_length]

        // Two rows, alternating by tick parity: a thread drains its slot in one row while
        // this tick's scatters accumulate into the other. That is what makes the synaptic
        // latency exactly one tick rather than one-or-two depending on thread order.
        GpuPointer<void> network_inputs;     // [2][total_neuron_count]

        // [spike_history_length][total_neuron_count]. A delayed arrival is answered by
        // asking whether the source spiked `delay` ticks ago, so every spike in flight is
        // remembered, not just the most recent one.
        GpuPointer<void> spike_history;
        GpuPointer<void> last_spiked;        // [total_neuron_count]

        // One element, bound in place of any per-edge plane the model never caused to be
        // allocated. A model with no connections registers no delay matrix and no per-edge
        // variable storage, and metal_dispatch binds an address it cannot resolve as raw
        // bytes rather than as a buffer -- so every argument must be a real address even
        // when the kernel provably never reads it.
        GpuPointer<void> edge_placeholder;

        // Host-side stimulus, applied before each dispatch.
        Vector<s64> continuous_injection_targets;
        Vector<f32> continuous_injection_amplitudes;
        Vector<s64> continuous_injection_start_ticks;
        Vector<s64> continuous_injection_end_ticks;

        // A scheduled spike train: one entry per (input profile, target) that carries
        // event times, plus a cursor into them. Kept sparse rather than expanded into a
        // dense [target][tick] array, which for a long run is almost all zeros.
        //
        // Each event injects `magnitude` for exactly one tick, so it delivers a charge of
        // magnitude * dt. That is what makes the amplitude a nanoamp-scale number rather
        // than the picoamps a sustained injector uses: to move a 100 pF membrane by 15 mV
        // in one 0.1 ms tick takes 15 nA.
        struct ScheduledSpikeTrain {
            s64 neuron_index = -1;
            f32 magnitude = 0.0f;
            Vector<s32> event_ticks;
            usize cursor = 0;
        };
        Vector<ScheduledSpikeTrain> scheduled_spike_trains;

        // ── recording ─────────────────────────────────────────────────────────────
        // Every neuron's spike count over the run, accumulated tick by tick. Kept
        // separately from recorded_spikes because the aliveness metrics need every
        // neuron's count whether or not the model asked for that neuron to be recorded.
        Vector<s64> spike_counts_per_neuron;
        Vector<RecordedSpike> recorded_spikes;
        // Row-major [recorded tick][traced quantity], parallel to traced_selections.
        Vector<f32> recorded_traces;
        Vector<RecordingSelection> traced_selections;
        Vector<f64> recorded_trace_times;

        // Set up by record_membrane_video(): a .spire recording of every neuron's
        // membrane potential, which is what the video renderer consumes. Separate from the
        // model's own OutputFiles because LEMS has no way to ask for "every neuron, every
        // Nth tick", and pretending one of its elements meant that would be inventing
        // semantics the format does not have.
        std::unique_ptr<SimulationRecorder> membrane_video_recorder;
        s64 membrane_video_frame_stride = 1;
        // Where each neuron's `v` sits in cell_state, precomputed so a frame is a gather
        // rather than a per-neuron name lookup.
        Vector<s64> membrane_offset_per_neuron;
        Vector<f32> membrane_frame_scratch;

        KernelHandle master_kernel{};
        String master_kernel_source;

        s64 total_neuron_count = 0;
        s64 lifetime = 0;
        f32 step_dt = 0.0f;
        u64 simulation_seed = 0;

        s32 thread_count_per_block = 256;
        s32 block_count = 0;
        bool alive = false;

        SpikeEngine() = delete;
        SpikeEngine(const SpikeEngine &) = delete;
        SpikeEngine &operator=(const SpikeEngine &) = delete;
        SpikeEngine(SpikeEngine &&) = delete;
        SpikeEngine &operator=(SpikeEngine &&) = delete;

        // Parses `lems_input_file`, allocates every buffer the model needs, fills them from
        // the model's starting parameters and OnStart initialisers, builds the weight matrix
        // from the model's connections, and compiles the generated tick kernel. Throws
        // naming the offending ComponentType if the model uses anything Phase 1 does not
        // simulate, rather than loading something it would run incorrectly.
        explicit SpikeEngine(const String &lems_input_file);

        ~SpikeEngine();

        // Runs the model for the number of ticks its Simulation asked for, recording as its
        // OutputFiles ask. Safe to call once.
        void run();

        // Advances exactly one tick. run() is this in a loop; separated so a test can step
        // and inspect state between ticks.
        void step_simulation(s64 tick);

        // The state variable named `variable_name` for one neuron, read back from the GPU.
        [[nodiscard]] f32 read_state_variable(s64 neuron_index, const String &variable_name) const;

        // Spikes per neuron per second over the whole run, and the fraction of neurons that
        // spiked at least once. What a demo has to clear to count as alive.
        [[nodiscard]] f64 mean_firing_rate_hertz() const;
        [[nodiscard]] f64 fraction_of_neurons_that_spiked() const;

        // Writes every recording profile the model declared to the files it named.
        void write_recordings();

        // Records every neuron's membrane potential into `path` as a .spire recording,
        // one frame every `frame_stride` ticks, for examples/render_membrane_video.py to
        // turn into a video. Call before run(); finished by write_recordings()/shutdown().
        void record_membrane_video(const String &path, s64 frame_stride = 1);

        void shutdown();

    private:
        void allocate_model_buffers();
        void initialize_cell_state();
        void build_weight_matrix();
        void collect_stimulus();
        void apply_stimulus(s64 tick);
        void *resolve_edge_plane(s64 matrix_index);
        void record_tick(s64 tick);
    };
} // namespace spikecorec
