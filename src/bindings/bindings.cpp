#include <algorithm>
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
#include "spikecorec/core/log.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/topologies.h"
#include "spikecorec/core/recording.h"

// ── NeuroML -> GPU codegen path (epic #1) -- ticket #133's own binding surface ──────────────────
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/synapse_lowering.h"
#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/master_kernel.h"
#include "spikecorec/nml/stimulus_schedule.h"

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

    // ── ticket #133: minimum-viable Python surface for the NML/GLIF codegen path ───────────────
    //
    // Before this, `SpikeEngine` (the legacy, hardcoded LIF-over-k²-tree engine) was the only
    // simulation object exposed to Python — the entire NeuroML -> GPU codegen epic (#1) was
    // unreachable from Python. Rather than binding every pipeline stage individually
    // (`NML_Parser`, `ResolvedModel`, `ModelSpecification`, `IrProgram`, `ModelAllocation`,
    // `AssembledModel`, each with their own Python-facing lifetime/ownership story), this wraps the
    // whole parse -> resolve -> lower -> allocate -> assemble -> step_tick pipeline (the exact
    // sequence every examples/*_torus_network_example.cpp already drives from C++ — see
    // examples/nml_pipeline_support.h / examples/glif_torus_network.h, example-only headers not
    // linked into this extension, hence the small amount of logic duplicated below) behind ONE
    // class. A Python caller only ever sees this class's constructor plus a handful of run/read-back
    // methods — everything else stays C++-internal. This is deliberately the MINIMUM surface that
    // satisfies "construct, run, and read back a GLIF network from Python" — not a binding of the
    // whole codegen pipeline.
    //
    // Scope boundary: only the Phase-1 stimulus shape `nml::build_stimulus_schedule` already covers
    // (`explicitInput` + `pulseGenerator`) is driven automatically. A model whose stimulus uses
    // `<inputList>` instead (a different, jLEMS-only-compatible XML shape the front end does not
    // lower to a `StimulusEntry` at all — see tests/exit_model_validation_tests.cpp's own header
    // comment) sees no automatic current injection here.
    class NmlNetworkRunner {
    public:
        NmlNetworkRunner(const string &nml_file_path, f32 dt_seconds)
            : model_(load_model(nml_file_path)),
              programs_(lower_type_library(model_)),
              allocation_(nml::allocate_model(model_, programs_)),
              weights_(build_weight_matrix(model_)),
              assembled_model_(model_, programs_),
              stimulus_schedule_(nml::build_stimulus_schedule(model_, (f64)dt_seconds)),
              dt_seconds_(dt_seconds) {
            seed_membrane_potentials_from_resting_parameter();

            const usize neuron_count = (usize)model_.total_neuron_count;

            network_inputs_ = allocate<f32>(neuron_count * sizeof(f32));
            std::memset(network_inputs_.get_contents(), 0, neuron_count * sizeof(f32));

            last_spiked_ = allocate<s64>(neuron_count * sizeof(s64));
            std::fill(last_spiked_.get_contents(), last_spiked_.get_contents() + model_.total_neuron_count, (s64)-1);

            next_active_neuron_indices_ = allocate<s32>(neuron_count * sizeof(s32));
            next_active_neuron_count_ = allocate<s32>(sizeof(s32));
            next_active_neuron_count_.get_contents()[0] = 0;

            active_generation_ = allocate<s32>(neuron_count * sizeof(s32));
            std::fill(active_generation_.get_contents(), active_generation_.get_contents() + model_.total_neuron_count, -1);

            buffers_.allocation = &allocation_;
            buffers_.weights = &weights_;
            buffers_.network_inputs = network_inputs_.get_contents();
            buffers_.last_spiked = last_spiked_.get_contents();
            buffers_.next_active_neuron_indices = next_active_neuron_indices_.get_contents();
            buffers_.next_active_neuron_count = next_active_neuron_count_.get_contents();
            buffers_.active_generation = active_generation_.get_contents();

            // Every emit port name a Cell-category type-in-use's IR fires `emit` on (every Phase-1
            // GLIF cell fires exactly one, "spike") — see nml::collect_emit_port_names's own doc
            // comment for why this is derived rather than hardcoded.
            spikecorec::Vector<String> emit_port_names = nml::collect_emit_port_names(model_, programs_);
            emit_port_flags_.reserve(emit_port_names.size());
            for (const String &port_name : emit_port_names) {
                GpuPointer<bool> flags = allocate<bool>(neuron_count * sizeof(bool));
                std::memset(flags.get_contents(), 0, neuron_count * sizeof(bool));
                buffers_.emit_port_flags[port_name] = flags.get_contents();
                emit_port_flags_.push_back(std::move(flags));
            }
        }

        NmlNetworkRunner(const NmlNetworkRunner &) = delete;
        NmlNetworkRunner &operator=(const NmlNetworkRunner &) = delete;

        // GpuPointer<T> has no RAII destructor anywhere in this codebase (deallocation is always
        // explicit — see ModelAllocation::~ModelAllocation for the same pattern); unlike a one-shot
        // example binary that leaks its own live buffers until process exit, a Python session may
        // construct/destroy many of these, so this explicitly frees every buffer this class itself
        // allocated (allocation_/weights_/assembled_model_ already free their own via their own
        // destructors, run afterward in the usual reverse-declaration-order).
        ~NmlNetworkRunner() {
            deallocate(std::move(network_inputs_));
            deallocate(std::move(last_spiked_));
            deallocate(std::move(next_active_neuron_indices_));
            deallocate(std::move(next_active_neuron_count_));
            deallocate(std::move(active_generation_));
            for (GpuPointer<bool> &flags : emit_port_flags_) deallocate(std::move(flags));
        }

        // Advances the simulation by exactly one tick: injects this tick's precomputed
        // pulseGenerator stimulus contribution (if any) into every neuron's network_inputs, then
        // dispatches one AssembledModel::step_tick call.
        void step() {
            for (s32 neuron_index = 0; neuron_index < model_.total_neuron_count; ++neuron_index) {
                buffers_.network_inputs[neuron_index] += (f32)stimulus_schedule_.current_at(neuron_index, tick_);
            }
            assembled_model_.step_tick(buffers_, dt_seconds_, tick_, tick_ + 1);
            ++tick_;
        }

        // Convenience loop over step() — keeps a multi-thousand-tick run's per-tick host loop in
        // C++ instead of paying Python call overhead once per tick.
        void run(s64 tick_count) {
            for (s64 index = 0; index < tick_count; ++index) step();
        }

        s64 neuron_count() const { return model_.total_neuron_count; }
        s64 current_tick() const { return tick_; }

        // Every neuron's membrane potential ("v", always cell-state slot 0 for every GLIF variant),
        // by GLOBAL neuron index. cell_state is structure-of-arrays within per-population chunks
        // (allocator.h §4.1), so this gathers across populations rather than a single memcpy.
        py::array_t<f32> membrane_potentials() const {
            py::array_t<f32> result(static_cast<py::ssize_t>(model_.total_neuron_count));
            f32 *destination = result.mutable_data();
            for (s32 population_index = 0; population_index < (s32)model_.populations.size(); ++population_index) {
                const nml::PopulationEntry &population = model_.populations[(usize)population_index];
                for (s32 local_index = 0; local_index < population.size; ++local_index) {
                    destination[population.neuron_index_begin + local_index] =
                        allocation_.cell_state.get_contents()[state_element_index(population_index, 0, local_index)];
                }
            }
            return result;
        }

        // Tick each neuron (global index) last spiked, or -1 if it never has.
        py::array_t<s64> last_spiked() const {
            prefetch_to_cpu(last_spiked_, (usize)model_.total_neuron_count * sizeof(s64));
            return to_numpy(last_spiked_.get_contents(), model_.total_neuron_count);
        }

    private:
        static nml::ModelSpecification load_model(const string &nml_file_path) {
            nml::NML_Parser parser;
            parser.parse(nml_file_path);
            nml::ResolvedModel resolved = nml::resolve_and_lower(parser);
            return nml::build_model_specification(resolved);
        }

        // One IrProgram per model.type_library entry, matching allocate_model/AssembledModel's own
        // parallel-array convention. Mirrors examples/nml_pipeline_support.h's own
        // lower_type_library_to_ir (an examples-only header not linked into this extension, hence
        // reimplemented here) with lower_inputs_on_device left at its Phase-1 default (false):
        // pulseGenerator is host-precomputed (build_stimulus_schedule), never lowered to a kernel.
        static spikecorec::Vector<nml::IrProgram> lower_type_library(const nml::ModelSpecification &model) {
            spikecorec::Vector<nml::IrProgram> programs;
            programs.reserve(model.type_library.size());
            for (const auto &entry : model.type_library) {
                if (entry.category == nml::TypeLibraryCategory::Cell) {
                    programs.push_back(nml::lower_cell_to_ir(entry));
                } else if (entry.category == nml::TypeLibraryCategory::Synapse) {
                    programs.push_back(nml::lower_synapse_to_ir(entry));
                } else {
                    nml::IrProgram placeholder;
                    placeholder.component_type_name = entry.component_type_name;
                    programs.push_back(std::move(placeholder));
                }
            }
            return programs;
        }

        // The exact-edge-set WeightMatrix AssembledModel::step_tick's ModelRuntimeBuffers::weights
        // requires (non-null regardless of whether the model has any real projections at all) —
        // mirrors examples/nml_pipeline_support.h's own build_weight_matrix.
        static WeightMatrix build_weight_matrix(const nml::ModelSpecification &model) {
            spikecorec::Vector<spikecorec::Vector<s32>> adjacency((usize)model.total_neuron_count);
            for (const auto &projection : model.projections) {
                for (const auto &connection : projection.connections) {
                    adjacency[(usize)connection.source_neuron_index].push_back(connection.target_neuron_index);
                }
            }
            return WeightMatrix(adjacency, /*rank=*/1);
        }

        s64 state_element_index(s32 population_index, s32 state_slot_index, s32 local_neuron_index) const {
            const nml::PopulationEntry &population = model_.populations[(usize)population_index];
            return allocation_.cell_type_boundaries.get_contents()[population_index]
                   + (s64)state_slot_index * population.size + local_neuron_index;
        }

        // allocate_model zero-initializes cell_state and never evaluates a cell type's own OnStart
        // (allocator.h/examples/nml_pipeline_support.h) — every population whose cell type bakes an
        // `EL` constant (every GLIF variant's OnStart is `v = EL`) gets that seeded into its own `v`
        // (state slot 0) by hand here; a population whose cell type has no `EL` (out of this class's
        // GLIF-shaped scope) is left at allocate_model's own zero default rather than throwing.
        void seed_membrane_potentials_from_resting_parameter() {
            for (s32 population_index = 0; population_index < (s32)model_.populations.size(); ++population_index) {
                const nml::PopulationEntry &population = model_.populations[(usize)population_index];
                const nml::TypeLibraryEntry &cell_entry = model_.type_library[(usize)population.type_library_index];
                auto resting_potential = cell_entry.baked_constants.find("EL");
                if (resting_potential == cell_entry.baked_constants.end()) continue;
                for (s32 local_index = 0; local_index < population.size; ++local_index) {
                    allocation_.cell_state.get_contents()[state_element_index(population_index, 0, local_index)] =
                        (f32)resting_potential->second;
                }
            }
        }

        nml::ModelSpecification model_;
        spikecorec::Vector<nml::IrProgram> programs_;
        nml::ModelAllocation allocation_;
        WeightMatrix weights_;
        nml::AssembledModel assembled_model_;
        nml::StimulusSchedule stimulus_schedule_;
        f32 dt_seconds_;

        GpuPointer<f32> network_inputs_;
        GpuPointer<s64> last_spiked_;
        GpuPointer<s32> next_active_neuron_indices_;
        GpuPointer<s32> next_active_neuron_count_;
        GpuPointer<s32> active_generation_;
        spikecorec::Vector<GpuPointer<bool>> emit_port_flags_;
        nml::ModelRuntimeBuffers buffers_;

        s64 tick_ = 0;
    };
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

    m.def("set_log_level", [](const string &level) {
        spikecorec::log::logger().set_level(spdlog::level::from_str(level));
    }, py::arg("level"));

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
        .def("is_alive", [](const SpikeEngine &self) { return self.alive; })
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
                 borrowed.pointer = output.pointer;
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
        // Each prefetches the buffer to the host first (SC-18) — a single bulk
        // async copy instead of the memcpy below faulting every page back from
        // the device one touch at a time.
        .def("get_membrane_potentials", [](const SpikeEngine &self) {
            prefetch_to_cpu(self.membrane_potentials, static_cast<usize>(self.neuron_count) * sizeof(f32));
            return to_numpy(self.membrane_potentials.get_contents(), self.neuron_count);
        })
        .def("get_network_inputs", [](const SpikeEngine &self) {
            prefetch_to_cpu(self.network_inputs, static_cast<usize>(self.neuron_count) * sizeof(f32));
            return to_numpy(self.network_inputs.get_contents(), self.neuron_count);
        })
        .def("get_last_spiked", [](const SpikeEngine &self) {
            prefetch_to_cpu(self.last_spiked, static_cast<usize>(self.neuron_count) * sizeof(s64));
            return to_numpy(self.last_spiked.get_contents(), self.neuron_count);
        })
        .def("get_last_tick_updated", [](const SpikeEngine &self) {
            prefetch_to_cpu(self.last_tick_updated, static_cast<usize>(self.neuron_count) * sizeof(s64));
            return to_numpy(self.last_tick_updated.get_contents(), self.neuron_count);
        })
        .def("get_active_neuron_indices", [](const SpikeEngine &self) {
            prefetch_to_cpu(self.active_neuron_count, sizeof(s32));
            s64 count = self.active_neuron_count.get_contents()[0];
            prefetch_to_cpu(self.active_neuron_indices, static_cast<usize>(count) * sizeof(s32));
            return to_numpy(self.active_neuron_indices.get_contents(), count);
        })
        .def("get_active_neuron_count", [](const SpikeEngine &self) {
            prefetch_to_cpu(self.active_neuron_count, sizeof(s32));
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
        .def_readonly("alive", &SpikeEngine::alive);

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

    // ── NeuroML -> GPU codegen path (epic #1), ticket #133 ────────────────────────────────────────
    // Runs an arbitrary Phase-1 NML model (GLIF-family cell populations, optionally wired through
    // real per-edge synapse projections, driven by pulseGenerator-based explicitInput stimuli) end
    // to end, without touching SpikeEngine. See NmlNetworkRunner's own class comment above for the
    // scope decision (this is the minimum viable binding surface, not a full pipeline exposure).
    py::class_<NmlNetworkRunner>(m, "NmlNetworkRunner")
        .def(py::init<const string &, f32>(), py::arg("nml_file_path"), py::arg("dt_seconds") = 1e-4f)
        .def("step", &NmlNetworkRunner::step)
        .def("run", &NmlNetworkRunner::run, py::arg("tick_count"),
             // May run thousands of ticks per call — release the GIL so other Python threads (and
             // any background compression thread from an unrelated SimulationRecorder) keep making
             // progress meanwhile, matching start_static_record's own established precedent.
             py::call_guard<py::gil_scoped_release>())
        .def("membrane_potentials", &NmlNetworkRunner::membrane_potentials)
        .def("last_spiked", &NmlNetworkRunner::last_spiked)
        .def_property_readonly("neuron_count", &NmlNetworkRunner::neuron_count)
        .def_property_readonly("current_tick", &NmlNetworkRunner::current_tick);
}
