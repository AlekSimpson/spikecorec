#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/topologies.h"
#include "spikecorec/core/recording.h"

namespace py = pybind11;
using namespace spikecorec;

namespace {
    // Copies a GPU buffer's contents into a freshly-allocated numpy array — the
    // safe, simple read-back path (no raw GpuPointer<T> crosses the language
    // boundary; there's no sound cross-runtime ownership story for that).
    template<typename T>
    py::array_t<T> to_numpy(const T *data, s64 count) {
        py::array_t<T> result(static_cast<py::ssize_t>(count));
        std::memcpy(result.mutable_data(), data, static_cast<usize>(count) * sizeof(T));
        return result;
    }
}

PYBIND11_MODULE(_spikecorec, m) {
    m.doc() = "spikecorec C++/CUDA/Metal backend";
    m.attr("__version__") = "0.1.0";

    // The GPU context is process-global and has no safe teardown ordering with
    // respect to Python object finalization (SpikeEngine/WeightMatrix GPU buffers
    // may still be alive at interpreter shutdown) — initialize once at import and
    // let the OS reclaim GPU resources at process exit, mirroring how CUDA
    // contexts are typically left to the driver in Python extensions.
    initialize_gpu_context();

    m.def("square_torus", &square_torus, py::arg("k"));
    m.def("small_world_torus", &small_world_torus,
          py::arg("k"), py::arg("random_fanout") = 4, py::arg("seed") = -1);
    m.def("random_fixed_outdegree", &random_fixed_outdegree,
          py::arg("k"), py::arg("fanout") = 8, py::arg("seed") = -1);

    py::class_<WeightStats>(m, "WeightStats")
        .def_readonly("mean", &WeightStats::mean)
        .def_readonly("standard_deviation", &WeightStats::standard_deviation)
        .def_readonly("root_mean_square", &WeightStats::root_mean_square)
        .def_readonly("min_value", &WeightStats::min_value)
        .def_readonly("max_value", &WeightStats::max_value);

    py::class_<ScaleResult>(m, "ScaleResult")
        .def_readonly("target_root_mean_square", &ScaleResult::target_root_mean_square)
        .def_readonly("scale_factor", &ScaleResult::scale_factor)
        .def_readonly("before", &ScaleResult::before)
        .def_readonly("after", &ScaleResult::after);

    py::class_<ScaledReservoirResult>(m, "ScaledReservoirResult")
        .def_readonly("weight_scale_result", &ScaledReservoirResult::weight_scale_result)
        .def_readonly("target_root_mean_square", &ScaledReservoirResult::target_root_mean_square)
        .def_readonly("w_accum", &ScaledReservoirResult::w_accum)
        .def_readonly("w_instant", &ScaledReservoirResult::w_instant);

    // Minimal read-only WeightMatrix surface — enough for inspection/testing.
    // K2Tree itself isn't bound: nothing here needs to expose it directly.
    py::class_<WeightMatrix>(m, "WeightMatrix")
        .def_readonly("node_count", &WeightMatrix::node_count)
        .def_readonly("max_neighbor_count", &WeightMatrix::max_neighbor_count)
        .def_readonly("rank", &WeightMatrix::rank)
        .def_readonly("rank_float4_stride", &WeightMatrix::rank_float4_stride)
        .def_readonly("constant_weight", &WeightMatrix::constant_weight)
        .def_readonly("using_constant_weight", &WeightMatrix::using_constant_weight)
        .def("get", &WeightMatrix::get, py::arg("source_node"), py::arg("target_node"))
        .def("neighbor_weight_stats", &WeightMatrix::neighbor_weight_stats)
        .def("set_constant_weight", &WeightMatrix::set_constant_weight, py::arg("value"))
        .def("get_neighbors", [](const WeightMatrix &self, s64 node_index) {
            vector<s32> buffer(static_cast<usize>(self.max_neighbor_count));
            s64 written = self.get_neighbors(node_index, buffer.data());
            buffer.resize(static_cast<usize>(written));
            return buffer;
        }, py::arg("node_index"));

    py::class_<SpikeEngine>(m, "SpikeEngine")
        .def(py::init([](vector<vector<s32>> network, const vector<s64> &shape, s64 rank,
                         f32 resting_mp, f32 decay_rate, f32 learning_rate) {
            // SpikeEngine's ctor only dereferences `network` during the WeightMatrix
            // member-init (it doesn't retain the pointer) — the local copy's address
            // stays valid for that entire call, so no heap allocation is needed.
            return std::make_unique<SpikeEngine>(&network, shape, rank, resting_mp, decay_rate, learning_rate);
        }), py::arg("network"), py::arg("shape"), py::arg("rank") = 1,
            py::arg("resting_mp") = 0.1f, py::arg("decay_rate") = 0.01f, py::arg("learning_rate") = 0.00222f)

        .def("setup_lifetime", &SpikeEngine::setup_lifetime,
             py::arg("lifetime"), py::arg("allocate_logs"),
             py::arg("max_log_bytes") = SpikeEngine::DEFAULT_MAX_LOG_BYTES)
        .def("set_input_neurons", &SpikeEngine::set_input_neurons, py::arg("input_neuron_list"))
        .def("reset_state", &SpikeEngine::reset_state,
             py::arg("last_spiked_value") = 0, py::arg("active_gen_value") = -1)
        .def("step_simulation", &SpikeEngine::step_simulation,
             py::arg("input_values"), py::arg("tick"),
             py::arg("override_input_neurons") = vector<s64>{},
             py::arg("decay_all_neurons") = false)
        .def("is_alive", &SpikeEngine::is_alive)
        .def("estimate_bifurcation_weight", &SpikeEngine::estimate_bifurcation_weight,
             py::arg("input_period") = 1)
        .def("scale_uniform_weights_near_bifurcation",
             [](SpikeEngine &self, s32 input_period, f32 scale, bool freeze_learning) {
                 f32 target = 0.0f, w_accum = 0.0f, w_instant = 0.0f;
                 self.scale_uniform_weights_near_bifurcation(
                     &target, &w_accum, &w_instant, input_period, scale, freeze_learning, nullptr);
                 return py::make_tuple(target, w_accum, w_instant);
             }, py::arg("input_period") = 1, py::arg("scale") = 1.2f, py::arg("freeze_learning") = false)
        .def("scale_randomized_weights_near_bifurcation", &SpikeEngine::scale_randomized_weights_near_bifurcation,
             py::arg("input_period") = 1, py::arg("scale") = 1.2f, py::arg("freeze_learning") = false)
        .def("get_reservoir_features_vector",
             [](SpikeEngine &self, s64 tick, f32 spike_tau, f32 voltage_scale) {
                 s64 feature_count = 2 * self.neuron_count + 1;
                 GpuPointer<f32> output = allocate<f32>(static_cast<usize>(feature_count) * sizeof(f32));

                 // get_reservoir_features_vector takes its GpuPointer by value but only
                 // ever calls .get_contents() on it (engine.cpp:240-258) — it neither
                 // stores nor frees the handle. GpuPointer is move-only, so we hand it a
                 // borrowed duplicate of the raw handle and keep `output` as the sole
                 // owner responsible for the matching deallocate() below.
                 GpuPointer<f32> borrowed;
            #ifdef SPIKECOREC_CUDA
                 borrowed.pointer = output.pointer;
            #elif defined(SPIKECOREC_METAL)
                 borrowed.buffer = output.buffer;
            #endif
                 self.get_reservoir_features_vector(tick, spike_tau, voltage_scale, std::move(borrowed));

                 py::array_t<f32> result = to_numpy(output.get_contents(), feature_count);
                 deallocate(std::move(output));
                 return result;
             }, py::arg("tick"), py::arg("spike_tau"), py::arg("voltage_scale"))
        .def("start_static_record", &SpikeEngine::start_static_record,
             py::arg("input_spikes"), py::arg("lifetime"), py::arg("filename"),
             py::arg("record_membrane") = true,
             py::arg("record_stride") = 1,
             py::arg("compression") = std::optional<std::string>("auto"),
             py::arg("compression_level") = std::optional<int>{},
             py::arg("full_decay") = true,
             py::arg("compression_async") = false,
             py::arg("compression_queue_max") = static_cast<usize>(8),
             py::arg("compression_chunk_bytes") = static_cast<usize>(4 * 1024 * 1024),
             // Long-running call that may spin up a background compression
             // thread (compression_async=True) — release the GIL so that
             // thread (and other Python threads) can make progress while the
             // C++ tick loop runs. This is the *first* GIL release among these
             // bindings; every other bound method here is short/synchronous.
             py::call_guard<py::gil_scoped_release>())
        .def("shutdown", &SpikeEngine::shutdown)

        // Read-back accessors — copy GPU buffer contents into numpy arrays.
        // No raw GpuPointer<T> is exposed to Python (see to_numpy's note).
        .def("get_membrane_potentials", [](const SpikeEngine &self) {
            return to_numpy(self.membrane_potentials.get_contents(), self.neuron_count);
        })
        .def("get_network_inputs", [](const SpikeEngine &self) {
            return to_numpy(self.network_inputs.get_contents(), self.neuron_count);
        })
        .def("get_last_spiked", [](const SpikeEngine &self) {
            return to_numpy(self.last_spiked.get_contents(), self.neuron_count);
        })
        .def("get_last_tick_updated", [](const SpikeEngine &self) {
            return to_numpy(self.last_tick_updated.get_contents(), self.neuron_count);
        })
        .def("get_active_neuron_indices", [](const SpikeEngine &self) {
            s64 count = self.active_neuron_count.get_contents()[0];
            return to_numpy(self.active_neuron_indices.get_contents(), count);
        })
        .def("get_active_neuron_count", [](const SpikeEngine &self) {
            return self.active_neuron_count.get_contents()[0];
        })
        .def_property_readonly("weights", [](SpikeEngine &self) -> WeightMatrix & { return self.weights; },
                               py::return_value_policy::reference_internal)

        .def_readonly("neuron_count", &SpikeEngine::neuron_count)
        .def_readonly("input_neuron_count", &SpikeEngine::input_neuron_count)
        .def_readwrite("resting_membrane_potential", &SpikeEngine::resting_membrane_potential)
        .def_readwrite("decay_rate", &SpikeEngine::decay_rate)
        .def_readwrite("learning_rate", &SpikeEngine::learning_rate)
        .def_readwrite("spike_period", &SpikeEngine::spike_period)
        .def_readwrite("spike_threshold", &SpikeEngine::spike_threshold)
        .def_readwrite("use_constant_weight", &SpikeEngine::use_constant_weight)
        .def_readonly("running", &SpikeEngine::running);

    // Decodes a `.spire`/`.spire.gz`/`.spire.xz`/`.spire.bz2` recording (as
    // written by start_static_record / SimulationRecorder, or by the
    // spikecore Python reference — the format is byte-for-byte compatible)
    // into a (frame_count, neuron_count) float32 array.
    m.def("read_spire_recording", [](const string &filename) {
        SpireRecording recording = read_spire_recording(filename);
        py::array_t<f32> result({static_cast<py::ssize_t>(recording.frame_count),
                                 static_cast<py::ssize_t>(recording.neuron_count)});
        // A header-only file decodes to zero frames — both pointers may be null,
        // and memcpy(nullptr, nullptr, 0) is undefined, so skip the copy.
        if (!recording.frames.empty())
            std::memcpy(result.mutable_data(), recording.frames.data(),
                        recording.frames.size() * sizeof(f32));
        return result;
    }, py::arg("filename"));

    // Standalone buffering/compression/recording layer underlying
    // start_static_record — exposed for callers who want to drive their own
    // per-tick recording loops (e.g. mixing in custom stimulus logic between
    // engine.step_simulation calls) instead of the all-in-one method above.
    py::class_<SimulationRecorder>(m, "SimulationRecorder")
        .def(py::init<const string &, s64, std::optional<std::string>, std::optional<int>, bool, usize, usize>(),
             py::arg("filename"), py::arg("neuron_count"),
             py::arg("compression") = std::optional<std::string>("auto"),
             py::arg("compression_level") = std::optional<int>{},
             // Named `compression_async` (not `async`) on the Python side —
             // `async` has been a reserved keyword since Python 3.7, so
             // `SimulationRecorder(..., async=True)` would be a SyntaxError.
             // Matches start_static_record's naming for the same concept.
             py::arg("compression_async") = false,
             py::arg("queue_max") = static_cast<usize>(8),
             py::arg("chunk_bytes") = static_cast<usize>(4 * 1024 * 1024))
        .def_property_readonly("neuron_count", &SimulationRecorder::neuron_count)
        .def("record_frame", [](SimulationRecorder &self, py::array_t<f32, py::array::c_style | py::array::forcecast> membrane_potentials) {
            // Pass the array length so record_frame can reject a wrongly-sized
            // array instead of reading out of bounds past membrane_potentials.
            self.record_frame(membrane_potentials.data(), membrane_potentials.size());
        }, py::arg("membrane_potentials"))
        .def("finish", &SimulationRecorder::finish);
}
