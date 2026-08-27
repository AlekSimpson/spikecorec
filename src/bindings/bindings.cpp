#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/core/topologies.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/weight_matrix.h"

namespace py = pybind11;
using namespace spikecorec;

namespace {

// Copies into a freshly allocated numpy array rather than viewing the engine's memory.
// A view would hand Python a pointer into a GPU slab whose lifetime is the engine's, and
// there is no sound cross-runtime ownership story for that -- the array would outlive the
// buffer the moment someone kept it past `del engine`.
template <typename ElementType>
py::array_t<ElementType> to_numpy(const ElementType *data, s64 count) {
    py::array_t<ElementType> result(static_cast<py::ssize_t>(count));
    if (count > 0) {
        std::memcpy(result.mutable_data(), data, static_cast<usize>(count) * sizeof(ElementType));
    }
    return result;
}

} // namespace

PYBIND11_MODULE(_spikecorec, m) {
    m.doc() = "spikecorec — a NeuroML-driven GPU spiking neural network engine";
    m.attr("__version__") = "0.1.0";

    // No GPU context to initialize. The backend is a member of each SpikeEngine now,
    // constructed and released with it, which also removes the teardown-ordering problem
    // the old global had against Python's finalization.

    m.def("set_log_level", [](const std::string &level) {
        spikecorec::log::logger().set_level(spdlog::level::from_str(level));
    }, py::arg("level"),
       "One of: trace, debug, info, warn, err, critical, off.");

    // ── topology generators ───────────────────────────────────────────────────────
    // Adjacency lists for the connectivity-in-code SpikeEngine constructor, for networks
    // too large to write out as <connection> elements.
    m.def("square_torus", &square_torus, py::arg("side_length"));
    m.def("small_world_torus", &small_world_torus,
          py::arg("side_length"), py::arg("random_fanout") = 4, py::arg("seed") = -1);
    m.def("random_fixed_outdegree", &random_fixed_outdegree,
          py::arg("side_length"), py::arg("fanout") = 8, py::arg("seed") = -1);

    // ── weight matrix ─────────────────────────────────────────────────────────────
    py::class_<WeightStats>(m, "WeightStats")
        .def_readonly("mean", &WeightStats::mean)
        .def_readonly("standard_deviation", &WeightStats::standard_deviation)
        .def_readonly("root_mean_square", &WeightStats::root_mean_square)
        .def_readonly("min_value", &WeightStats::min_value)
        .def_readonly("max_value", &WeightStats::max_value)
        .def("__repr__", [](const WeightStats &self) {
            return "<WeightStats mean=" + std::to_string(self.mean) +
                   " rms=" + std::to_string(self.root_mean_square) +
                   " min=" + std::to_string(self.min_value) +
                   " max=" + std::to_string(self.max_value) + ">";
        });

    // Non-copyable and non-movable from Python's side: it is owned by the engine and
    // exposed by reference, so the class is bound without any constructor.
    py::class_<WeightMatrix>(m, "WeightMatrix")
        .def_readonly("node_count", &WeightMatrix::node_count)
        .def_readonly("total_edge_count", &WeightMatrix::total_edge_count)
        .def_readonly("max_neighbor_count", &WeightMatrix::max_neighbor_count)
        .def_readonly("rank", &WeightMatrix::rank)

        // How faithfully the basis plus its corrections reproduce what the model declared.
        // The basis is a lossy projection and this is the price it charged -- worth reading
        // before trusting a result, because nothing refuses to run on account of it.
        .def_readonly("measured_weight_fit_error", &WeightMatrix::measured_weight_fit_error)
        .def_readonly("sparse_delta_capacity", &WeightMatrix::sparse_delta_capacity)
        .def_property_readonly("sparse_delta_occupancy_fraction",
                               &WeightMatrix::sparse_delta_occupancy_fraction)
        .def_readwrite("refit_occupancy_threshold_fraction",
                       &WeightMatrix::refit_occupancy_threshold_fraction)

        .def("get", &WeightMatrix::get,
             py::arg("source_node"), py::arg("target_node"),
             "The weight of one edge: the basis reconstruction plus its sparse correction.")
        .def("get_for_matrix", &WeightMatrix::get_for_matrix,
             py::arg("source_node"), py::arg("target_node"), py::arg("matrix_index"))
        .def("get_edge_delay_ticks", &WeightMatrix::get_edge_delay_ticks,
             py::arg("source_node"), py::arg("target_node"))
        .def("get_edge_synapse_prototype", &WeightMatrix::get_edge_synapse_prototype,
             py::arg("source_node"), py::arg("target_node"))
        .def("edge_ordinal", &WeightMatrix::edge_ordinal,
             py::arg("source_node"), py::arg("target_node"),
             "This edge's number in the canonical ordering, or None if it is not an edge.")

        .def("neighbors", [](const WeightMatrix &self, s64 node_index) {
            std::vector<s32> buffer(static_cast<usize>(std::max<s64>(self.max_neighbor_count, 1)));
            const s64 found = self.get_neighbors(node_index, buffer.data());
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"))
        .def("predecessors", [](const WeightMatrix &self, s64 node_index) {
            std::vector<s32> buffer(static_cast<usize>(std::max<s64>(self.max_neighbor_count, 1)));
            const s64 found = self.get_predecessors(node_index, buffer.data());
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"))

        // One value per REAL edge, indexed by edge ordinal. No padding rows and no
        // sentinels to filter out.
        .def("edge_weights", [](const WeightMatrix &self) {
            std::vector<f32> weights(static_cast<usize>(std::max<s64>(self.total_edge_count, 0)));
            if (!weights.empty()) self.neighbor_weights(weights.data());
            return to_numpy(weights.data(), self.total_edge_count);
        })
        .def("weight_stats", &WeightMatrix::neighbor_weight_stats)

        .def("accumulate_edge_delta", &WeightMatrix::accumulate_edge_delta,
             py::arg("matrix_index"), py::arg("source_node"), py::arg("target_node"),
             py::arg("delta"),
             "Queues a correction for one edge. Visible to reads immediately; folded into "
             "the basis by the next refit.")
        .def("compact_pending_deltas", &WeightMatrix::compact_pending_deltas)
        .def("refit", &WeightMatrix::refit,
             py::arg("sweep_count") = 4, py::arg("ridge_regularization") = 1e-3f,
             "Re-optimises the basis toward the values the corrections point at, then drops "
             "the ones it absorbed.")
        .def("is_refit_due", &WeightMatrix::is_refit_due)

        .def("save", [](const WeightMatrix &self, const std::string &path) {
            self.save(path.c_str());
        }, py::arg("path"))
        .def("load_from_disk", [](WeightMatrix &self, const std::string &path) {
            self.load_from_disk(path.c_str());
        }, py::arg("path"))

        .def("__repr__", [](const WeightMatrix &self) {
            return "<WeightMatrix nodes=" + std::to_string(self.node_count) +
                   " edges=" + std::to_string(self.total_edge_count) +
                   " rank=" + std::to_string(self.rank) + ">";
        });

    // ── recorded spikes ───────────────────────────────────────────────────────────
    py::class_<RecordedSpike>(m, "RecordedSpike")
        .def_readonly("time_seconds", &RecordedSpike::time_seconds)
        .def_readonly("neuron_index", &RecordedSpike::neuron_index)
        .def("__repr__", [](const RecordedSpike &self) {
            return "<RecordedSpike t=" + std::to_string(self.time_seconds) +
                   " neuron=" + std::to_string(self.neuron_index) + ">";
        });

    // ── the engine ────────────────────────────────────────────────────────────────
    py::class_<SpikeEngine>(m, "SpikeEngine")
        .def(py::init<const String &, bool>(),
             py::arg("lems_input_file"),
             py::arg("enable_hebbian_plasticity") = false,
             "Parses a LEMS document, allocates and fills every buffer the model needs, "
             "builds the weight matrix from its connections, and compiles the tick kernel.")
        .def(py::init<const String &, const std::vector<std::vector<s32>> &, const String &,
                      f64, f64, bool>(),
             py::arg("lems_input_file"), py::arg("adjacency"), py::arg("synapse_component_id"),
             py::arg("connection_weight") = 1.0,
             py::arg("connection_delay_seconds") = 0.0,
             py::arg("enable_hebbian_plasticity") = false,
             "The same, with connectivity supplied in code instead of in the document -- "
             "which is how a network too large to write as <connection> elements is built.")

        .def_readonly("total_neuron_count", &SpikeEngine::total_neuron_count)
        .def_readonly("lifetime", &SpikeEngine::lifetime)
        .def_readonly("step_dt", &SpikeEngine::step_dt)
        .def_readonly("alive", &SpikeEngine::alive)
        .def_readonly("hebbian_plasticity_enabled", &SpikeEngine::hebbian_plasticity_enabled)

        // Owned by the engine, so handed back by reference with the engine's lifetime
        // keeping it alive -- returning a copy is not an option and would not be wanted.
        .def_property_readonly("weights",
                               [](SpikeEngine &self) -> WeightMatrix & { return self.weights; },
                               py::return_value_policy::reference_internal)

        .def("run", &SpikeEngine::run,
             py::call_guard<py::gil_scoped_release>(),
             "Runs every tick the model's Simulation declared.")
        .def("step_simulation", &SpikeEngine::step_simulation, py::arg("tick"),
             py::call_guard<py::gil_scoped_release>(),
             "One tick, for callers driving their own loop -- mixing in custom stimulus "
             "between ticks, say.")

        .def("read_state_variable", &SpikeEngine::read_state_variable,
             py::arg("neuron_index"), py::arg("variable_name"),
             "One neuron's named state variable, read back from the GPU.")
        .def("state_variable_array", [](const SpikeEngine &self, const String &variable_name) {
            // A loop over read_state_variable rather than a slice of cell_state: state is
            // laid out per population and per variable, so there is no single contiguous
            // run holding one variable for every neuron.
            std::vector<f32> values(static_cast<usize>(self.total_neuron_count));
            for (s64 index = 0; index < self.total_neuron_count; index += 1) {
                values[static_cast<usize>(index)] = self.read_state_variable(index, variable_name);
            }
            return to_numpy(values.data(), self.total_neuron_count);
        }, py::arg("variable_name"),
           "The same variable for every neuron. Every cell type in the model has to declare "
           "it, or the read throws naming the one that does not.")

        .def_property_readonly("spike_counts", [](const SpikeEngine &self) {
            return to_numpy(self.spike_counts_per_neuron.data(),
                            static_cast<s64>(self.spike_counts_per_neuron.size()));
        }, "Spikes per neuron over the whole run.")
        .def_property_readonly("spike_times", [](const SpikeEngine &self) {
            // Two parallel arrays rather than a list of objects: a long run has millions of
            // spikes, and one Python object each is not a reasonable thing to build.
            std::vector<f64> times(self.recorded_spikes.size());
            std::vector<s64> neurons(self.recorded_spikes.size());
            for (usize index = 0; index < self.recorded_spikes.size(); index += 1) {
                times[index] = self.recorded_spikes[index].time_seconds;
                neurons[index] = self.recorded_spikes[index].neuron_index;
            }
            return py::make_tuple(to_numpy(times.data(), static_cast<s64>(times.size())),
                                  to_numpy(neurons.data(), static_cast<s64>(neurons.size())));
        }, "(times_seconds, neuron_indices), parallel arrays over every recorded spike.")

        .def("mean_firing_rate_hertz", &SpikeEngine::mean_firing_rate_hertz)
        .def("fraction_of_neurons_that_spiked", &SpikeEngine::fraction_of_neurons_that_spiked)

        .def("write_recordings", &SpikeEngine::write_recordings,
             "Writes every OutputFile and EventOutputFile the model declared.")
        .def("record_membrane_video", &SpikeEngine::record_membrane_video,
             py::arg("path"), py::arg("frame_stride") = 1,
             "Records every neuron's membrane potential to a .spire file for the video "
             "renderer. Call before run().")
        .def("write_spike_file", [](const SpikeEngine &self, const std::string &path) {
            self.write_spike_file(path);
        }, py::arg("path"), "Every spike of the run as 'time<tab>neuron'.")

        .def("shutdown", &SpikeEngine::shutdown)
        .def("__repr__", [](const SpikeEngine &self) {
            return "<SpikeEngine neurons=" + std::to_string(self.total_neuron_count) +
                   " ticks=" + std::to_string(self.lifetime) +
                   " dt=" + std::to_string(self.step_dt) + ">";
        });

    // ── recordings ────────────────────────────────────────────────────────────────
    // Decodes a .spire / .spire.gz / .spire.xz / .spire.bz2 recording into a
    // (frame_count, neuron_count) float32 array.
    m.def("read_spire_recording", [](const std::string &filename) {
        SpireRecording recording = read_spire_recording(filename);
        py::array_t<f32> result({static_cast<py::ssize_t>(recording.frame_count),
                                 static_cast<py::ssize_t>(recording.neuron_count)});
        // A header-only file decodes to zero frames, and memcpy(nullptr, nullptr, 0) is
        // undefined -- so skip the copy rather than relying on it being harmless.
        if (!recording.frames.empty()) {
            std::memcpy(result.mutable_data(), recording.frames.data(),
                        recording.frames.size() * sizeof(f32));
        }
        return result;
    }, py::arg("filename"));

    // The buffering/compression layer underneath record_membrane_video, for callers
    // driving their own per-tick recording loop.
    py::class_<SimulationRecorder>(m, "SimulationRecorder")
        .def(py::init<const std::string &, s64, std::optional<std::string>, std::optional<int>,
                      bool, usize, usize>(),
             py::arg("filename"), py::arg("neuron_count"),
             py::arg("compression") = std::optional<std::string>("auto"),
             py::arg("compression_level") = std::optional<int>{},
             // Named compression_async, not async: `async` has been a reserved keyword
             // since Python 3.7, so the obvious name would be a SyntaxError at the call.
             py::arg("compression_async") = false,
             py::arg("queue_max") = static_cast<usize>(8),
             py::arg("chunk_bytes") = static_cast<usize>(4 * 1024 * 1024))
        .def_property_readonly("neuron_count", &SimulationRecorder::neuron_count)
        .def("record_frame", [](SimulationRecorder &self,
                                py::array_t<f32, py::array::c_style | py::array::forcecast> frame) {
            // The length goes through so record_frame can reject a wrongly-sized array
            // rather than reading past its end.
            self.record_frame(frame.data(), frame.size());
        }, py::arg("membrane_potentials"))
        .def("finish", &SimulationRecorder::finish);
}

// ── TODO: reservoir surface, kept for when it comes back ──────────────────────────
//
// These bound a different engine -- the pre-NeuroML reservoir one, whose parameters were
// engine fields (spike_threshold, decay_rate, learning_rate) rather than things a LEMS
// document declares, and whose state was one flat membrane_potentials array per neuron.
// None of it compiles against the current SpikeEngine, and none of it was translated
// upward, because the concepts do not survive the move: a cell's threshold now belongs to
// its ComponentType, and state is laid out per population per variable.
//
// Left here rather than deleted so the surface is recoverable rather than archaeological.
// Whoever restores it will need the engine-side feature first; this is the shape it took.
//
//  .def(py::init([](const vector<vector<s32>> &network, s64 shape, s64 rank,
//                   f32 resting_mp, f32 decay_rate, f32 learning_rate) {
//           return std::make_unique<SpikeEngine>(&network, shape, rank,
//                                                resting_mp, decay_rate, learning_rate);
//       }), py::arg("network"), py::arg("shape"), py::arg("rank") = 1,
//          py::arg("resting_mp") = 0.1f, py::arg("decay_rate") = 0.01f,
//          py::arg("learning_rate") = 0.00222f)
//
//  .def("setup_lifetime", &SpikeEngine::setup_lifetime,
//       py::arg("lifetime"), py::arg("allocate_logs"),
//       py::arg("max_log_bytes") = SpikeEngine::DEFAULT_MAX_LOG_BYTES)
//  .def("set_input_neurons", &SpikeEngine::set_input_neurons, py::arg("input_neuron_list"))
//  .def("reset_state", &SpikeEngine::reset_state,
//       py::arg("last_spiked_value") = 0, py::arg("active_gen_value") = -1)
//  .def("is_alive", &SpikeEngine::is_alive)
//
//  // Bifurcation search: drove the network to the edge of runaway activity and reported
//  // the weight scale that got it there. Replaced by nothing yet.
//  .def("estimate_bifurcation_weight", &SpikeEngine::estimate_bifurcation_weight,
//       py::arg("input_period") = 1)
//  .def("scale_uniform_weights_near_bifurcation",
//       [](SpikeEngine &self, s32 input_period, f32 scale, bool freeze_learning) {
//           f32 target = 0.0f, w_accum = 0.0f, w_instant = 0.0f;
//           self.scale_uniform_weights_near_bifurcation(
//               &target, &w_accum, &w_instant, input_period, scale, freeze_learning, nullptr);
//           return py::make_tuple(target, w_accum, w_instant);
//       }, py::arg("input_period") = 1, py::arg("scale") = 1.2f,
//          py::arg("freeze_learning") = false)
//  .def("scale_randomized_weights_near_bifurcation",
//       &SpikeEngine::scale_randomized_weights_near_bifurcation,
//       py::arg("input_period") = 1, py::arg("scale") = 1.2f,
//       py::arg("freeze_learning") = false)
//
//  // Reservoir readout: spike traces, normalised voltages and a bias term, as one
//  // feature vector per tick.
//  .def("get_reservoir_features_vector",
//       [](SpikeEngine &self, s64 tick, f32 spike_tau, f32 voltage_scale) { ... },
//       py::arg("tick"), py::arg("spike_tau"), py::arg("voltage_scale"))
//
//  // All-in-one record loop. record_membrane_video covers the membrane half of this;
//  // driving the tick loop from the recorder's side does not exist any more.
//  .def("start_static_record", &SpikeEngine::start_static_record,
//       py::arg("input_spikes"), py::arg("lifetime"), py::arg("filename"),
//       py::arg("record_membrane") = true, py::arg("record_stride") = 1,
//       py::arg("compression") = std::optional<std::string>("auto"),
//       py::arg("compression_level") = std::optional<int>{},
//       py::arg("full_decay") = true, py::arg("compression_async") = false,
//       py::arg("compression_queue_max") = static_cast<usize>(8),
//       py::arg("compression_chunk_bytes") = static_cast<usize>(4 * 1024 * 1024),
//       py::call_guard<py::gil_scoped_release>())
//
//  // Flat per-neuron state reads. state_variable_array covers the membrane one; the
//  // active-set accessors have no successor, since the active-set optimisation is not
//  // part of the generated-kernel engine.
//  .def("get_membrane_potentials", ...)
//  .def("get_network_inputs", ...)
//  .def("get_last_spiked", ...)
//  .def("get_last_tick_updated", ...)
//  .def("get_active_neuron_indices", ...)
//  .def("get_active_neuron_count", ...)
//
//  .def_readonly("input_neuron_count", &SpikeEngine::input_neuron_count)
//  .def_readwrite("resting_membrane_potential", &SpikeEngine::resting_membrane_potential)
//  .def_readwrite("decay_rate", &SpikeEngine::decay_rate)
//  .def_readwrite("learning_rate", &SpikeEngine::learning_rate)
//  .def_readwrite("spike_period", &SpikeEngine::spike_period)
//  .def_readwrite("spike_threshold", &SpikeEngine::spike_threshold)
//  .def_readwrite("use_constant_weight", &SpikeEngine::use_constant_weight)
//  .def_readonly("running", &SpikeEngine::running)
