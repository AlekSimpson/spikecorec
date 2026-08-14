//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

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
        // adjacency, so only real edges are ever addressed. Optional because there is no
        // network to build one from until a model has been parsed.
        Optional<WeightMatrix> weights;

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

        // Continuous injection is the only stimulus Phase 1 delivers. A spike train is
        // refused at construction rather than silently injecting nothing — see
        // collect_stimulus.

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
        void write_recordings() const;

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
