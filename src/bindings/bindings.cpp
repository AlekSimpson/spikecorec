// ── spikecorec Python bindings ──────────────────────────────────────────────────────────
//
// The engine this binds is the NeuroML one: a SpikeEngine is built from a NeuroML/LEMS
// model file, advanced one tick at a time, and read back through its buffers and the
// `.spire` recordings the model's own <OutputFile>s produce. That is the whole surface.
//
// Nothing from the pre-NeuroML engine is bound. setup_lifetime, set_input_neurons,
// reset_state, the adjacency-list constructor, the reservoir/bifurcation helpers and the
// membrane_potentials/active-set accessors are all commented out in engine.h as legacy
// pending rework, and binding a name whose implementation no longer exists is how this file
// stopped compiling in the first place.
//
// Two deliberate omissions, both of which would otherwise read as ordinary knobs:
//
//   - the topology generators (square_torus and friends). They return an adjacency list,
//     and the only thing that ever consumed one from Python was the legacy SpikeEngine
//     constructor. With that gone there is nothing on this surface to hand one to; a model's
//     wiring comes from its <projection>s.
//
//   - SpikeEngine::use_constant_weight. It is live (dispatch_master_kernel forwards
//     weights.constant_weight to the kernel when it is set), but it is a SECOND constant-
//     weight flag independent of WeightMatrix::using_constant_weight, which is what the host
//     WeightMatrix::get() consults. Setting one from Python without the other makes the GPU
//     run on a weight the host read-back does not report, which is a wrong-numbers bug with
//     no way to see it. It has no NeuroML-path meaning either -- a model states its weights.

#include <cstring>
#include <string>
#include <vector>

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
#include "spikecorec/core/log.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/recording.h"

namespace py = pybind11;
using namespace spikecorec;

namespace {
    // Copies a buffer's contents into a freshly-allocated numpy array — the safe, simple
    // read-back path. No raw GpuPointer<T> crosses the language boundary: there is no sound
    // cross-runtime ownership story for one, and every engine buffer is an arena sub-range
    // that must never be freed by anything but the arena.
    template<typename ElementType>
    py::array_t<ElementType> to_numpy(const ElementType *data, s64 element_count) {
        py::array_t<ElementType> result(static_cast<py::ssize_t>(element_count));
        if (element_count > 0) {
            std::memcpy(result.mutable_data(), data,
                        static_cast<usize>(element_count) * sizeof(ElementType));
        }
        return result;
    }

    // A model whose cell types declare no state variables (or one with no wired synapse
    // prototypes) sizes the corresponding buffer at zero elements, which the arena answers
    // with a null handle — get_contents() would dereference it. Such a buffer reads back as
    // an empty array rather than crashing.
    template<typename ElementType>
    py::array_t<ElementType> buffer_to_numpy(const GpuPointer<ElementType> &buffer,
                                             s64 element_count) {
        if (buffer.pointer == nullptr || element_count <= 0) {
            return py::array_t<ElementType>(0);
        }

        prefetch_to_cpu(buffer, static_cast<usize>(element_count) * sizeof(ElementType));
        return to_numpy(buffer.get_contents(), element_count);
    }

    const nml::CellTypeSpecification &cell_type_of_neuron(const SpikeEngine &engine,
                                                          s64 neuron_index) {
        const s64 type_index =
                engine.network_details.neurons[static_cast<usize>(neuron_index)].cell_type_index;
        return engine.network_details.cell_types[static_cast<usize>(type_index)];
    }

    // Position of `name` in `names`, or -1. Cell types hold a handful of entries each, so a
    // scan is what the engine itself does everywhere this lookup appears.
    //
    // `Vector` is spelled out qualified here and below: spikecorec and spikecorec::log each
    // declare an alias template of that name and engine.h drags both into scope, so the bare
    // name is ambiguous at file scope. src/core/engine.cpp works around the same collision.
    s64 position_of(const spikecorec::Vector<String> &names, const String &name) {
        for (usize slot = 0; slot < names.size(); ++slot) {
            if (names[slot] == name) return static_cast<s64>(slot);
        }
        return -1;
    }

    // One named quantity for every neuron, by GLOBAL neuron index.
    //
    // cell_state and cell_parameters are sectioned by cell TYPE, so this gathers rather than
    // copying a range: neuron k's chunk starts at cell_state_base[k] / cell_parameter_base[k]
    // and the quantity's position inside that chunk is its position in its type's declaration
    // order. Two cell types may well put "v" in different slots, which is exactly why the
    // slot is resolved per neuron from the neuron's own type instead of being assumed.
    //
    // A neuron whose type does not declare `name` at all THROWS, naming the neuron, its type
    // and what that type does declare. Returning that neuron's first slot instead — which is
    // what SpikeEngine's own recording path falls back to — would answer a question about
    // "asc1" with a membrane potential.
    py::array_t<f32> gather_named_quantity(const SpikeEngine &engine, const String &name,
                                           bool from_parameters) {
        const GpuPointer<f32> &values = from_parameters ? engine.cell_parameters
                                                        : engine.cell_state;
        const GpuPointer<s32> &bases = from_parameters ? engine.cell_parameter_base
                                                       : engine.cell_state_base;
        if (values.pointer == nullptr || bases.pointer == nullptr) {
            throw std::runtime_error(
                    std::string("this model allocated no cell ") +
                    (from_parameters ? "parameters" : "state") + ", so '" + name +
                    "' cannot be read from it");
        }

        const f32 *value_contents = values.get_contents();
        const s32 *base_contents = bases.get_contents();

        py::array_t<f32> result(static_cast<py::ssize_t>(engine.total_neuron_count));
        f32 *destination = result.mutable_data();

        for (s64 neuron_index = 0; neuron_index < engine.total_neuron_count; ++neuron_index) {
            const nml::CellTypeSpecification &cell_type = cell_type_of_neuron(engine, neuron_index);
            const spikecorec::Vector<String> &names =
                    from_parameters ? cell_type.parameter_names : cell_type.state_variable_names;

            const s64 slot = position_of(names, name);
            if (slot < 0) {
                String declared;
                for (const String &declared_name : names) {
                    if (!declared.empty()) declared += ", ";
                    declared += declared_name;
                }
                throw std::runtime_error(
                        "neuron " + std::to_string(neuron_index) + " has cell type '" +
                        cell_type.name + "', which declares no " +
                        (from_parameters ? "parameter '" : "state variable '") + name +
                        "' (it declares: " + declared + ")");
            }

            destination[neuron_index] =
                    value_contents[base_contents[neuron_index] + slot];
        }

        return result;
    }
} // namespace

PYBIND11_MODULE(_spikecorec, m) {
    m.doc() = "spikecorec — NeuroML-driven spiking-network simulation on Metal/CUDA";
    m.attr("__version__") = "0.1.0";

    // The GPU context is process-global and has no safe teardown ordering with respect to
    // Python object finalization (a SpikeEngine's arena may still be alive at interpreter
    // shutdown) — initialize once at import and let the OS reclaim GPU resources at process
    // exit, mirroring how CUDA contexts are typically left to the driver in Python
    // extensions. release_gpu_resources() is deliberately never called.
    initialize_gpu_context();

    m.def("set_log_level", [](const std::string &level) {
        spikecorec::log::logger().set_level(spdlog::level::from_str(level));
    }, py::arg("level"),
       "Sets the engine's console log level: trace/debug/info/warn/err/critical/off.");

    py::class_<WeightStats>(m, "WeightStats")
        .def_readonly("mean", &WeightStats::mean)
        .def_readonly("standard_deviation", &WeightStats::standard_deviation)
        .def_readonly("root_mean_square", &WeightStats::root_mean_square)
        .def_readonly("min_value", &WeightStats::min_value)
        .def_readonly("max_value", &WeightStats::max_value);

    // Read-only, and reachable only through SpikeEngine.weights: the wiring comes from the
    // model, so there is nothing here to construct or mutate from Python. It is bound at all
    // because it is the only way to ask what the engine actually wired — a model that parses
    // cleanly but connects nothing is otherwise indistinguishable from one that connects
    // everything.
    py::class_<WeightMatrix>(m, "WeightMatrix")
        .def_readonly("node_count", &WeightMatrix::node_count)
        .def_readonly("max_neighbor_count", &WeightMatrix::max_neighbor_count)
        .def_readonly("rank", &WeightMatrix::rank)
        .def_readonly("using_exact_edge_weights", &WeightMatrix::using_exact_edge_weights)
        .def("get", &WeightMatrix::get, py::arg("source_node"), py::arg("target_node"),
             "Weight of the edge source_node -> target_node, 0.0 when there is no such edge.")
        .def("get_edge_delay_ticks", &WeightMatrix::get_edge_delay_ticks,
             py::arg("source_node"), py::arg("target_node"),
             "Conduction delay of that edge in whole ticks (always at least 1).")
        .def("neighbor_weight_stats", &WeightMatrix::neighbor_weight_stats,
             "mean/stddev/RMS/min/max over every edge weight.")
        .def("get_neighbors", [](const WeightMatrix &self, s64 node_index) {
            std::vector<s32> buffer(static_cast<usize>(self.max_neighbor_count));
            const s64 written = self.get_neighbors(node_index, buffer.data());
            buffer.resize(static_cast<usize>(written));
            return buffer;
        }, py::arg("node_index"),
           "Outgoing neighbours of node_index, in k^2-tree traversal order.");

    py::class_<SpikeEngine>(m, "SpikeEngine")
        // Constructed through a lambda rather than py::init<String &, bool>() for two
        // reasons. The engine's constructor takes its path by NON-CONST reference, so a
        // temporary cannot bind to it; and binding every argument by name through a lambda
        // means a new constructor parameter with a default (use_lazy_synapse_updates is
        // being added) neither breaks this file nor silently shifts what the existing
        // positional arguments mean.
        .def(py::init([](const std::string &model_path, bool enable_hebbian_learning) {
            String local_model_path = model_path;
            return std::make_unique<SpikeEngine>(local_model_path, enable_hebbian_learning);
        }), py::arg("model_path"), py::arg("enable_hebbian_learning") = false,
            "Builds an engine from a NeuroML/LEMS model file: parses and validates it, sizes\n"
            "and allocates every model buffer, runs the OnStart bodies, compiles the generated\n"
            "master kernel, and opens a recorder per <OutputFile> the model declares.\n"
            "\n"
            "enable_hebbian_learning currently allocates last_tick_updated and NOTHING ELSE --\n"
            "no kernel reads it and no weight update runs, so the run is bit-identical to one\n"
            "built without it. The engine warns about this at construction.")

        // Wrapped rather than bound straight through, for the negative-tick check alone. The
        // generated kernel resolves the delay-ring row as `tick % ring_depth` in SIGNED
        // arithmetic, so a negative tick indexes network_inputs BEFORE its first element --
        // an out-of-bounds device access, reachable from pure Python by a mistyped loop
        // bound. SpikeEngine::step_simulation does not check this itself; it should, and
        // this guard should then become redundant rather than load-bearing.
        .def("step_simulation", [](SpikeEngine &self, s64 tick) {
            if (tick < 0) {
                throw std::invalid_argument(
                        "step_simulation: tick must be non-negative, got " +
                        std::to_string(tick) + "; a negative tick indexes the delay ring out "
                        "of bounds");
            }
            self.step_simulation(tick);
        }, py::arg("tick"),
             "Advances every neuron by exactly one dt: delivers tick `tick`'s external\n"
             "stimulus, runs the generated master kernel plus the ring clear behind it, and\n"
             "writes one frame to every recorder.\n"
             "\n"
             "`tick` is the caller's -- the engine keeps no counter of its own and does not\n"
             "check it. It indexes the stimulus streams AND selects the delay-ring row, so a\n"
             "repeated, skipped or out-of-order tick silently produces a wrong simulation\n"
             "rather than an error. Step from 0 upwards by one, or use run().")

        .def("run", [](SpikeEngine &self, s64 tick_count, s64 first_tick) {
            if (first_tick < 0) {
                throw std::invalid_argument(
                        "run: first_tick must be non-negative, got " +
                        std::to_string(first_tick));
            }
            for (s64 index = 0; index < tick_count; ++index) {
                self.step_simulation(first_tick + index);
            }
        }, py::arg("tick_count"), py::arg("first_tick") = 0,
           // Thousands of ticks per call, none of which touch a Python object — release the
           // GIL so other Python threads (and any background .spire compression thread) keep
           // making progress meanwhile.
           py::call_guard<py::gil_scoped_release>(),
           "step_simulation over ticks first_tick .. first_tick + tick_count - 1, keeping the\n"
           "per-tick loop in C++.")

        .def("shutdown", &SpikeEngine::shutdown,
             "Finishes every recorder (which is what flushes the last buffered .spire frames\n"
             "to disk) and releases the compiled kernels. Idempotent; the destructor calls it\n"
             "if the caller did not.")

        // ── model description ────────────────────────────────────────────────────────
        .def_readonly("total_neuron_count", &SpikeEngine::total_neuron_count)
        .def_readonly("input_neuron_count", &SpikeEngine::input_neuron_count,
                      "How many (input, target) stimulus streams the model wired, NOT how many\n"
                      "distinct neurons are driven -- two inputs onto one neuron count twice.")
        .def_readonly("lifetime", &SpikeEngine::lifetime,
                      "Ticks the model's own <Simulation length/step> works out to.")
        .def_readonly("alive", &SpikeEngine::alive)
        .def_readonly("hebbian_learning_enabled", &SpikeEngine::hebbian_learning_enabled)
        .def_readonly("simulation_seed", &SpikeEngine::simulation_seed)
        .def_readonly("network_input_ring_depth", &SpikeEngine::network_input_ring_depth)
        .def_readonly("network_input_plane_count", &SpikeEngine::network_input_plane_count)
        .def_readonly("cell_state_element_count", &SpikeEngine::cell_state_element_count)
        .def_readonly("cell_parameter_element_count", &SpikeEngine::cell_parameter_element_count)
        .def_readonly("synapse_state_element_count", &SpikeEngine::synapse_state_element_count)

        .def_property_readonly("step_dt", [](const SpikeEngine &self) {
            return self.network_details.step_dt;
        }, "Seconds per tick, in SI, as the model's <Simulation step=...> resolved.")
        .def_property_readonly("simulation_duration", [](const SpikeEngine &self) {
            return self.network_details.simulation_duration;
        }, "Seconds the model's <Simulation length=...> resolved to.")

        .def_property_readonly("weights", [](SpikeEngine &self) -> WeightMatrix & {
            return self.weights;
        }, py::return_value_policy::reference_internal,
           "The wired adjacency: weights and per-edge delays, by global neuron index.")

        .def("cell_type_names", [](const SpikeEngine &self) {
            std::vector<std::string> names;
            for (const nml::CellTypeSpecification &cell_type : self.network_details.cell_types) {
                names.push_back(cell_type.name);
            }
            return names;
        }, "Cell type names, in the order their indices refer to.")

        .def("state_variable_names", [](const SpikeEngine &self, s64 cell_type_index) {
            if (cell_type_index < 0 ||
                cell_type_index >= (s64)self.network_details.cell_types.size()) {
                throw std::out_of_range("cell_type_index " + std::to_string(cell_type_index) +
                                        " names no cell type");
            }
            return std::vector<std::string>(
                    self.network_details.cell_types[(usize)cell_type_index]
                            .state_variable_names.begin(),
                    self.network_details.cell_types[(usize)cell_type_index]
                            .state_variable_names.end());
        }, py::arg("cell_type_index"),
           "That type's StateVariable names, in the order they occupy its state chunk.")

        .def("parameter_names", [](const SpikeEngine &self, s64 cell_type_index) {
            if (cell_type_index < 0 ||
                cell_type_index >= (s64)self.network_details.cell_types.size()) {
                throw std::out_of_range("cell_type_index " + std::to_string(cell_type_index) +
                                        " names no cell type");
            }
            return std::vector<std::string>(
                    self.network_details.cell_types[(usize)cell_type_index]
                            .parameter_names.begin(),
                    self.network_details.cell_types[(usize)cell_type_index]
                            .parameter_names.end());
        }, py::arg("cell_type_index"),
           "That type's Parameter names, in the order they occupy its parameter chunk.")

        .def("recording_output_filenames", [](const SpikeEngine &self) {
            std::vector<std::string> filenames;
            for (const RecordingConfig &profile : self.recording_profiles) {
                for (const String &filename : profile.output_filenames) {
                    filenames.push_back(filename);
                }
            }
            return filenames;
        }, "Every file the model's <OutputFile>/<EventOutputFile> declarations write, in the\n"
           "order the recorders were opened. Readable with read_spire_recording() after\n"
           "shutdown().")

        // ── read-back ────────────────────────────────────────────────────────────────
        .def("state_variable_values", [](const SpikeEngine &self, const std::string &name) {
            return gather_named_quantity(self, name, /*from_parameters=*/false);
        }, py::arg("name"),
           "One named StateVariable for every neuron, by global neuron index -- e.g.\n"
           "state_variable_values('v') is the membrane potential in VOLTS. Throws if any\n"
           "neuron's cell type does not declare that variable.")

        .def("parameter_values", [](const SpikeEngine &self, const std::string &name) {
            return gather_named_quantity(self, name, /*from_parameters=*/true);
        }, py::arg("name"),
           "One named Parameter for every neuron, by global neuron index, in SI -- a\n"
           "C=\"100pF\" reads back as 1e-10. Throws if any neuron's cell type does not\n"
           "declare that parameter.")

        .def("spike_flags", [](const SpikeEngine &self) {
            return buffer_to_numpy(self.spike_flags, self.total_neuron_count);
        }, "1 for every neuron that emitted on the tick just stepped, 0 otherwise. Lowered by\n"
           "the master kernel at the top of each tick, so this only ever describes the most\n"
           "recent step_simulation call.")

        .def("last_spiked", [](const SpikeEngine &self) {
            return buffer_to_numpy(self.last_spiked, self.total_neuron_count);
        }, "Tick each neuron last emitted on. Initialised to 0, NOT to -1, so a value of 0\n"
           "means either 'fired on tick 0' or 'has never fired' -- accumulate spike_flags()\n"
           "if you need to tell those apart.")

        .def("cell_state", [](const SpikeEngine &self) {
            return buffer_to_numpy(self.cell_state, self.cell_state_element_count);
        }, "The whole cell-state buffer, flat. Sectioned by cell TYPE, not by neuron index:\n"
           "use cell_state_base()/state_variable_values() to address it.")

        .def("cell_parameters", [](const SpikeEngine &self) {
            return buffer_to_numpy(self.cell_parameters, self.cell_parameter_element_count);
        }, "The whole cell-parameter buffer, flat, sectioned the same way as cell_state().")

        .def("cell_state_base", [](const SpikeEngine &self) {
            return buffer_to_numpy(self.cell_state_base, self.total_neuron_count);
        }, "Where each neuron's own chunk starts inside cell_state().")

        .def("cell_parameter_base", [](const SpikeEngine &self) {
            return buffer_to_numpy(self.cell_parameter_base, self.total_neuron_count);
        }, "Where each neuron's own chunk starts inside cell_parameters().")

        .def("cell_type_index", [](const SpikeEngine &self) {
            return buffer_to_numpy(self.cell_type_index, self.total_neuron_count);
        }, "Each neuron's cell type index, into cell_type_names().")

        .def("synapse_state", [](const SpikeEngine &self) {
            return buffer_to_numpy(self.synapse_state, self.synapse_state_element_count);
        }, "Per-(target neuron, wired synapse prototype) synapse state, flat.")

        .def("network_inputs", [](const SpikeEngine &self) {
            const s64 element_count = (s64)self.network_input_ring_depth *
                                      (s64)self.network_input_plane_count *
                                      self.total_neuron_count;
            return buffer_to_numpy(self.network_inputs, element_count);
        }, "The synaptic delay ring, flat. Reshape to\n"
           "(network_input_ring_depth, network_input_plane_count, total_neuron_count):\n"
           "row `tick % ring_depth` is what tick `tick` reads and what is cleared behind it,\n"
           "plane 0 is delivered current (in amperes) and plane 1+p holds arrivals awaiting\n"
           "wired synapse prototype p.");

    // Decodes a `.spire`/`.spire.gz`/`.spire.xz`/`.spire.bz2` recording — as written by the
    // recorders a model's <OutputFile>s open, or by SimulationRecorder below — into a
    // (frame_count, value_count) float32 array. One frame per tick; the columns are whatever
    // that output file selected, in selection order.
    m.def("read_spire_recording", [](const std::string &filename) {
        SpireRecording recording = read_spire_recording(filename);
        py::array_t<f32> result({static_cast<py::ssize_t>(recording.frame_count),
                                 static_cast<py::ssize_t>(recording.neuron_count)});
        // A header-only file decodes to zero frames — both pointers may be null, and
        // memcpy(nullptr, nullptr, 0) is undefined, so skip the copy.
        if (!recording.frames.empty()) {
            std::memcpy(result.mutable_data(), recording.frames.data(),
                        recording.frames.size() * sizeof(f32));
        }
        return result;
    }, py::arg("filename"));

    // The buffering/compression/recording layer the engine's own recorders are built on,
    // exposed for callers who want to write a `.spire` file from values they gathered
    // themselves between step_simulation calls, rather than from an <OutputFile> the model
    // declares.
    py::class_<SimulationRecorder>(m, "SimulationRecorder")
        .def(py::init<const std::string &, s64, std::optional<std::string>, std::optional<int>,
                      bool, usize, usize>(),
             py::arg("filename"), py::arg("value_count"),
             py::arg("compression") = std::optional<std::string>("auto"),
             py::arg("compression_level") = std::optional<int>{},
             // Named `compression_async` (not `async`) on the Python side — `async` has been
             // a reserved keyword since Python 3.7, so `SimulationRecorder(..., async=True)`
             // would be a SyntaxError.
             py::arg("compression_async") = false,
             py::arg("queue_max") = static_cast<usize>(8),
             py::arg("chunk_bytes") = static_cast<usize>(4 * 1024 * 1024))
        .def_property_readonly("value_count", &SimulationRecorder::neuron_count,
                               "Values per frame, fixed at construction.")
        .def("record_frame", [](SimulationRecorder &self,
                                py::array_t<f32, py::array::c_style | py::array::forcecast> values) {
            // Pass the array length so record_frame can reject a wrongly-sized array instead
            // of reading out of bounds past it.
            self.record_frame(values.data(), values.size());
        }, py::arg("values"))
        .def("finish", &SimulationRecorder::finish);
}
