#include <algorithm>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/k2tree.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/core/topologies.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/components.h"
#include "spikecorec/nml/dynamics.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/nml/parser.h"
#include "spikecorec/nml/utilities.h"

namespace py = pybind11;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

// Copies into a freshly allocated numpy array rather than viewing the engine's memory.
// A view would hand Python a pointer into a GPU slab whose lifetime is the engine's, and
// there is no sound cross-runtime ownership story for that -- the array would outlive the
// buffer the moment someone kept it past `del engine`.
template <typename ElementType>
py::array_t<ElementType> to_numpy(const ElementType *data, s64 count) {
    py::array_t<ElementType> result(static_cast<py::ssize_t>(std::max<s64>(count, 0)));
    if (count > 0 && data != nullptr) {
        std::memcpy(result.mutable_data(), data, static_cast<usize>(count) * sizeof(ElementType));
    }
    return result;
}

// The same, shaped [row_count][column_count] over row-major data.
template <typename ElementType>
py::array_t<ElementType> to_numpy_rows(const ElementType *data, s64 row_count, s64 column_count) {
    const s64 rows = std::max<s64>(row_count, 0);
    const s64 columns = std::max<s64>(column_count, 0);
    py::array_t<ElementType> result({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
    if (rows * columns > 0 && data != nullptr) {
        std::memcpy(result.mutable_data(), data, static_cast<usize>(rows * columns) * sizeof(ElementType));
    }
    return result;
}

// A device buffer's first `count` elements. Empty when the buffer was never allocated.
template <typename ElementType>
py::array_t<ElementType> device_buffer_to_numpy(const EnginePointer &pointer, s64 count) {
    if (pointer.is_empty()) return to_numpy<ElementType>(nullptr, 0);
    return to_numpy(pointer.get_contents_as<ElementType>(), count);
}

template <typename ElementType>
py::array_t<ElementType> device_buffer_to_numpy_rows(const EnginePointer &pointer, s64 row_count, s64 column_count) {
    if (pointer.is_empty()) return to_numpy_rows<ElementType>(nullptr, 0, 0);
    return to_numpy_rows(pointer.get_contents_as<ElementType>(), row_count, column_count);
}

// Python objects for values `owner` keeps, each keeping `owner` alive. A copy would carry
// pointers (a type's `extends`, an instance's `component_type`) into memory the owner frees.
template <typename ValueType>
py::object reference_or_none(const ValueType *value, py::handle owner) {
    if (value == nullptr) return py::none();
    return py::cast(value, py::return_value_policy::reference_internal, owner);
}

template <typename ValueType>
py::list references_in(const Vector<ValueType> &values, py::handle owner) {
    py::list result;
    for (const ValueType &value : values) result.append(reference_or_none(&value, owner));
    return result;
}

template <typename ValueType>
py::list references_to(const Vector<ValueType *> &values, py::handle owner) {
    py::list result;
    for (const ValueType *value : values) result.append(reference_or_none(value, owner));
    return result;
}

template <typename ValueType>
py::dict references_by_key(const UnorderedMap<String, ValueType> &values, py::handle owner) {
    py::dict result;
    for (const auto &[key, value] : values) result[py::str(key)] = reference_or_none(&value, owner);
    return result;
}

// Every engine buffer lives in the slab shutdown() releases.
void require_alive(const SpikeEngine &engine, const char *what) {
    if (!engine.alive) throw std::runtime_error(String(what) + ": the engine has been shut down");
}

} // namespace

PYBIND11_MODULE(_spikecorec, m) {
    m.doc() = "spikecorec — a NeuroML-driven GPU spiking neural network engine";
    m.attr("__version__") = "0.1.0";

    // The module follows the C++ namespaces: spikecorec here, spikecorec::nml in `nml` and
    // spikecorec::units in `units`, with every class and function under its C++ name.
    py::module_ nml_module = m.def_submodule("nml", "The parsed NeuroML/LEMS model (spikecorec::nml).");
    py::module_ units_module = m.def_submodule("units", "Unit conversion (spikecorec::units).");

    // ── logging ───────────────────────────────────────────────────────────────────
    m.def("set_log_level", [](const std::string &level) {
        spikecorec::log::logger().set_level(spdlog::level::from_str(level));
    }, py::arg("level"),
       "One of: trace, debug, info, warn, err, critical, off.");
    m.attr("LOG_PATH") = spikecorec::log::LOG_PATH;

    m.attr("NEVER_SPIKED_TICK") = NEVER_SPIKED_TICK;
    m.attr("MAX_RANK_FLOAT4_STRIDE") = MAX_RANK_FLOAT4_STRIDE;
    m.attr("DEFAULT_BRANCHING_FACTOR") = DEFAULT_BRANCHING_FACTOR;

    // ── units ─────────────────────────────────────────────────────────────────────
    py::class_<units::UnitDefinition>(units_module, "UnitDefinition",
            "A <Unit>: si = written magnitude * scale + offset.")
        .def(py::init<>())
        .def_readwrite("scale", &units::UnitDefinition::scale)
        .def_readwrite("offset", &units::UnitDefinition::offset)
        .def("__repr__", [](const units::UnitDefinition &self) {
            return "<UnitDefinition scale=" + std::to_string(self.scale) +
                   " offset=" + std::to_string(self.offset) + ">";
        });

    units_module.def("ms_to_seconds", &units::ms_to_seconds, py::arg("ms"));
    units_module.def("ms_to_ticks", &units::ms_to_ticks, py::arg("total_ms"), py::arg("ms_step"));
    units_module.def("seconds_to_ms", &units::seconds_to_ms, py::arg("seconds"));
    units_module.def("tick_to_ms", &units::tick_to_ms, py::arg("tick"), py::arg("total_ms"), py::arg("total_ticks"));
    units_module.def("tick_to_seconds", &units::tick_to_seconds,
                     py::arg("tick"), py::arg("total_seconds"), py::arg("total_ticks"));
    units_module.def("seconds_to_ticks", &units::seconds_to_ticks, py::arg("seconds"), py::arg("step_dt"));
    units_module.def("unit_suffix_scale", &units::unit_suffix_scale, py::arg("suffix"),
                     "SI scale for a NeuroML unit suffix, e.g. 'mV' -> 1e-3. Unknown suffixes scale by 1.");
    units_module.def("parse_quantity", &units::parse_quantity, py::arg("value"),
                     "'-60mV' -> -0.06 using only the built-in unit table. NML_Context.resolve_quantity "
                     "also knows the document's own units.");
    units_module.def("split_quantity", &units::split_quantity, py::arg("value"),
                     "'-60mV' -> (-60.0, 'mV').");

    // ── topology generators ───────────────────────────────────────────────────────
    // Adjacency lists for the connectivity-in-code SpikeEngine constructors, for networks
    // too large to write out as <connection> elements.
    m.def("square_torus", &square_torus, py::arg("side_length"));
    m.def("small_world_torus", &small_world_torus,
          py::arg("side_length"), py::arg("random_fanout") = 4, py::arg("seed") = -1);
    m.def("random_fixed_outdegree", &random_fixed_outdegree,
          py::arg("side_length"), py::arg("fanout") = 8, py::arg("seed") = -1);

    // ── recording configuration (spikecorec::RecordingConfig, as the parser fills it) ──
    py::enum_<OutputFileFormat>(m, "OutputFileFormat")
        .value("SPIRE", OutputFileFormat::SPIRE)
        .value("SPIREGZIP", OutputFileFormat::SPIREGZIP)
        .value("SPIREBZ2", OutputFileFormat::SPIREBZ2)
        .value("SPIREXZ", OutputFileFormat::SPIREXZ)
        .value("SPIKE_EVENTS", OutputFileFormat::SPIKE_EVENTS)
        .value("NML_STANDARD", OutputFileFormat::NML_STANDARD);

    py::class_<RecordingSelection>(m, "RecordingSelection")
        .def_readonly("quantity_path", &RecordingSelection::quantity_path, "Verbatim LEMS path, e.g. 'pop1[0]/v'.")
        .def_readonly("variable_name", &RecordingSelection::variable_name, "Trailing variable; '' for an event selection.")
        .def_readonly("event_port", &RecordingSelection::event_port, "EventSelection's port; '' for a column.")
        .def_readonly("neuron_index", &RecordingSelection::neuron_index,
                      "For a column, the variable's cell_state index; for an event, the cell's; -1 when none.")
        .def("__repr__", [](const RecordingSelection &self) {
            return "<RecordingSelection " + self.quantity_path + ">";
        });

    py::class_<RecordingConfig>(m, "RecordingConfig")
        .def_readonly("output_filenames", &RecordingConfig::output_filenames)
        .def_readonly("file_output_format", &RecordingConfig::file_output_format)
        .def_property_readonly("selections", [](py::object self) {
            return references_in(self.cast<const RecordingConfig &>().selections, self);
        })
        .def_readonly("recordings_count", &RecordingConfig::recordings_count)
        .def_readonly_static("DEFAULT_MAX_LOG_BYTES", &RecordingConfig::DEFAULT_MAX_LOG_BYTES);

    // ── .spire recordings ─────────────────────────────────────────────────────────
    py::enum_<SpireCompression>(m, "SpireCompression")
        .value("None_", SpireCompression::None)
        .value("Gzip", SpireCompression::Gzip)
        .value("Xz", SpireCompression::Xz)
        .value("Bz2", SpireCompression::Bz2);

    m.def("resolve_spire_compression", &resolve_spire_compression,
          py::arg("filename"), py::arg("requested") = std::optional<std::string>("auto"),
          "(compression, filename): 'auto' infers from the extension (.gz/.gzip, .xz/.lzma, .bz2); "
          "'none', 'gzip', 'xz' and 'bz2' ask for one explicitly.");

    // Decodes a .spire / .spire.gz / .spire.xz / .spire.bz2 recording into a
    // (frame_count, neuron_count) float32 array.
    m.def("read_spire_recording", [](const std::string &filename) {
        SpireRecording recording = read_spire_recording(filename);
        return to_numpy_rows(recording.frames.data(), recording.frame_count, recording.neuron_count);
    }, py::arg("filename"));

    py::class_<SpireWriter>(m, "SpireWriter", "Uncompressed .spire encoder, one frame at a time.")
        .def(py::init<const std::string &, s64>(), py::arg("filename"), py::arg("neuron_count"))
        .def("write_frame", [](SpireWriter &self,
                               py::array_t<f32, py::array::c_style | py::array::forcecast> frame) {
            // write_frame reads neuron_count values, so a shorter array would be read past its end.
            if (frame.size() != self.neuron_count()) {
                throw std::invalid_argument("write_frame: the frame has " + std::to_string(frame.size()) +
                                            " values, not " + std::to_string(self.neuron_count()));
            }
            self.write_frame(frame.data());
        }, py::arg("membrane_potentials"))
        .def_property_readonly("neuron_count", &SpireWriter::neuron_count);

    py::class_<SpireReader>(m, "SpireReader", "Uncompressed .spire decoder, one frame at a time.")
        .def(py::init<const std::string &>(), py::arg("filename"))
        .def("read_frame", [](SpireReader &self) -> py::object {
            std::vector<f32> frame(static_cast<usize>(self.neuron_count()));
            if (!self.read_frame(frame.data())) return py::none();
            return to_numpy(frame.data(), self.neuron_count());
        }, "The next frame, or None at the end of the file.")
        .def_property_readonly("neuron_count", &SpireReader::neuron_count);

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

    // ── nml: the parsed document tree ─────────────────────────────────────────────
    py::enum_<NML_DeclarationType>(nml_module, "NML_DeclarationType")
        .value("NamedDimension", NML_DeclarationType::NamedDimension)
        .value("Parameter", NML_DeclarationType::Parameter)
        .value("DerivedParameter", NML_DeclarationType::DerivedParameter)
        .value("Constant", NML_DeclarationType::Constant)
        .value("Requirement", NML_DeclarationType::Requirement)
        .value("Exposure", NML_DeclarationType::Exposure)
        .value("Property", NML_DeclarationType::Property)
        .value("Fixed", NML_DeclarationType::Fixed)
        .value("Text", NML_DeclarationType::Text)
        .value("EventPort", NML_DeclarationType::EventPort)
        .value("NamedTypeReference", NML_DeclarationType::NamedTypeReference)
        .value("Attachment", NML_DeclarationType::Attachment)
        .value("ComponentReference", NML_DeclarationType::ComponentReference)
        .value("Link", NML_DeclarationType::Link)
        .value("Children", NML_DeclarationType::Children)
        .value("Child", NML_DeclarationType::Child)
        .value("Path", NML_DeclarationType::Path)
        .value("StateVariable", NML_DeclarationType::StateVariable)
        .value("DerivedVariable", NML_DeclarationType::DerivedVariable)
        .value("ConditionalDerivedVariable", NML_DeclarationType::ConditionalDerivedVariable)
        .value("Case", NML_DeclarationType::Case)
        .value("TimeDerivative", NML_DeclarationType::TimeDerivative)
        .value("OnCondition", NML_DeclarationType::OnCondition)
        .value("OnEvent", NML_DeclarationType::OnEvent)
        .value("OnStart", NML_DeclarationType::OnStart)
        .value("OnEntry", NML_DeclarationType::OnEntry)
        .value("StateAssignment", NML_DeclarationType::StateAssignment)
        .value("EventOut", NML_DeclarationType::EventOut)
        .value("Transition", NML_DeclarationType::Transition)
        .value("Regime", NML_DeclarationType::Regime)
        .value("NOT_A_TYPE", NML_DeclarationType::NOT_A_TYPE);

    py::class_<NML_Tag>(nml_module, "NML_Tag", "One XML element's name and attributes.")
        .def_readonly("tag_name", &NML_Tag::tag_name)
        .def_readonly("tag_type", &NML_Tag::tag_type)
        .def_readonly("attributes", &NML_Tag::attributes)
        .def("has_attribute", &NML_Tag::has_attribute, py::arg("name"))
        .def("get_attribute", &NML_Tag::get_attribute, py::arg("name"))
        .def("get_attribute_or", &NML_Tag::get_attribute_or, py::arg("name"), py::arg("fallback"))
        .def("namespace_key", &NML_Tag::namespace_key,
             "Identity of this declaration within its ComponentType, e.g. '<namespace>:<name>'.")
        .def("is_declaration_type", &NML_Tag::is_declaration_type)
        .def("__repr__", [](const NML_Tag &self) { return "<NML_Tag " + self.tag_name + ">"; });

    // A node of the document tree the context owns. Only ever handed out by reference.
    py::class_<NML_Node>(nml_module, "NML_Node")
        .def_property_readonly("body", [](const NML_Node &self) -> const NML_Tag & { return self.body; })
        .def_property_readonly("children", [](py::object self) {
            return references_to(children_of(&self.cast<const NML_Node &>()), self);
        }, "The element children, in document order; empty for a leaf.")
        .def("__repr__", [](const NML_Node &self) { return "<NML_Node " + self.body.tag_name + ">"; });

    // ── nml: component types and instances ────────────────────────────────────────
    py::class_<NML_DynamicsExpression>(nml_module, "NML_DynamicsExpression",
            "One entry of a ComponentType's <Dynamics>, in source order.")
        .def_readonly("source_tag", &NML_DynamicsExpression::source_tag)
        .def_readonly("target", &NML_DynamicsExpression::target,
                      "Variable written, or exposure / port / regime name.")
        .def_readonly("expression", &NML_DynamicsExpression::expression, "value= / test= source, verbatim.")
        .def_readonly("regime_name", &NML_DynamicsExpression::regime_name, "Owning Regime; '' when regime-free.")
        .def_readonly("condition", &NML_DynamicsExpression::condition)
        .def_readonly("select", &NML_DynamicsExpression::select)
        .def_readonly("reduce", &NML_DynamicsExpression::reduce)
        .def("__repr__", [](const NML_DynamicsExpression &self) {
            return "<NML_DynamicsExpression " + self.target + " = " + self.expression + ">";
        });

    py::class_<NML_ComponentType>(nml_module, "NML_ComponentType")
        .def_readonly("name", &NML_ComponentType::name)
        .def_property_readonly("extends", [](py::object self) {
            return reference_or_none(self.cast<const NML_ComponentType &>().extends, self);
        })
        .def_readonly("source_file", &NML_ComponentType::source_file)
        .def_property_readonly("source_node", [](py::object self) {
            return reference_or_none(self.cast<const NML_ComponentType &>().source_node, self);
        }, "The <ComponentType> element in the document tree.")
        .def_property_readonly("dynamics", [](py::object self) {
            return references_in(self.cast<const NML_ComponentType &>().dynamics, self);
        })
        .def_readonly("state_variable_names", &NML_ComponentType::state_variable_names,
                      "Per-neuron (or, for a synapse, per-edge) state slots in order, inherited ones first.")
        .def("find_declaration", [](py::object self, const String &namespace_key) {
            return reference_or_none(self.cast<const NML_ComponentType &>().find_declaration(namespace_key), self);
        }, py::arg("namespace_key"),
           "The declaration this type resolves the key to, searching up the extends chain, or None.")
        .def("__repr__", [](const NML_ComponentType &self) { return "<NML_ComponentType " + self.name + ">"; });

    py::class_<NML_ComponentInstance>(nml_module, "NML_ComponentInstance")
        .def_readonly("id", &NML_ComponentInstance::id)
        .def_readonly("instance_data", &NML_ComponentInstance::instance_data,
                      "The element's attributes, verbatim (units unresolved).")
        .def_readonly("data_order", &NML_ComponentInstance::data_order, "Child instance ids, in source order.")
        .def_property_readonly("parent_instance", [](py::object self) {
            return reference_or_none(self.cast<const NML_ComponentInstance &>().parent_instance, self);
        })
        .def_property_readonly("component_type", [](py::object self) {
            return reference_or_none(self.cast<const NML_ComponentInstance &>().component_type, self);
        })
        .def("has_value", &NML_ComponentInstance::has_value, py::arg("key"))
        .def("value_or", &NML_ComponentInstance::value_or, py::arg("key"), py::arg("fallback") = String(""))
        .def("__repr__", [](const NML_ComponentInstance &self) {
            return "<NML_ComponentInstance " + self.id + " : " +
                   (self.component_type ? self.component_type->name : String("?")) + ">";
        });

    // ── nml: stimulus and connectivity ────────────────────────────────────────────
    py::class_<InputTarget>(nml_module, "InputTarget")
        .def_readonly("neuron_index", &InputTarget::neuron_index)
        .def_readonly("weight", &InputTarget::weight, "From <inputW weight=...>; 1.0 when unweighted.")
        .def_readonly("event_ticks", &InputTarget::event_ticks, "A spike train's ticks; empty for an injector.");

    py::class_<SimulationInputConfig>(nml_module, "SimulationInputConfig")
        .def_readonly("input_component_id", &SimulationInputConfig::input_component_id)
        .def_property_readonly("targets", [](py::object self) {
            return references_in(self.cast<const SimulationInputConfig &>().targets, self);
        })
        .def_readonly("amplitude", &SimulationInputConfig::amplitude)
        .def_readonly("rate", &SimulationInputConfig::rate)
        .def_readonly("start_tick", &SimulationInputConfig::start_tick)
        .def_readonly("end_tick", &SimulationInputConfig::end_tick, "0 runs to the end of the simulation.")
        .def_readonly("continuous_current_injection", &SimulationInputConfig::continuous_current_injection);

    py::class_<NML_NetworkEdge>(nml_module, "NML_NetworkEdge")
        .def_readonly("component_id", &NML_NetworkEdge::component_id, "The synapse this edge carries.")
        .def_readonly("weight", &NML_NetworkEdge::weight)
        .def_readonly("delay_ticks", &NML_NetworkEdge::delay_ticks)
        .def_readonly("parent", &NML_NetworkEdge::parent, "Source neuron.")
        .def_readonly("child", &NML_NetworkEdge::child, "Target neuron.")
        .def("__repr__", [](const NML_NetworkEdge &self) {
            return "<NML_NetworkEdge " + std::to_string(self.parent) + " -> " + std::to_string(self.child) +
                   " " + self.component_id + ">";
        });

    py::class_<AdjacencyList>(nml_module, "AdjacencyList", "The network's edges, one row per source neuron.")
        .def_property_readonly("list", [](py::object self) {
            py::list rows;
            for (const Vector<NML_NetworkEdge> &row : self.cast<const AdjacencyList &>().list) {
                rows.append(references_in(row, self));
            }
            return rows;
        })
        .def("in_network", &AdjacencyList::in_network, py::arg("node_index"))
        .def("is_parent", &AdjacencyList::is_parent, py::arg("parent"), py::arg("prospective_child"))
        .def("edge_arrays", [](const AdjacencyList &self) {
            // One object per edge does not scale to the networks this engine is for.
            std::vector<s64> parent;
            std::vector<s64> child;
            std::vector<f32> weight;
            std::vector<s64> delay_ticks;
            std::vector<std::string> component_id;
            for (const Vector<NML_NetworkEdge> &row : self.list) {
                for (const NML_NetworkEdge &edge : row) {
                    parent.push_back(edge.parent);
                    child.push_back(edge.child);
                    weight.push_back(edge.weight);
                    delay_ticks.push_back(edge.delay_ticks);
                    component_id.push_back(edge.component_id);
                }
            }
            py::dict result;
            result["parent"] = to_numpy(parent.data(), (s64)parent.size());
            result["child"] = to_numpy(child.data(), (s64)child.size());
            result["weight"] = to_numpy(weight.data(), (s64)weight.size());
            result["delay_ticks"] = to_numpy(delay_ticks.data(), (s64)delay_ticks.size());
            result["component_id"] = component_id;
            return result;
        }, "Every edge as parallel arrays: parent, child, weight, delay_ticks, component_id.")
        .def("__len__", [](const AdjacencyList &self) { return self.list.size(); });

    // ── nml: the context ──────────────────────────────────────────────────────────
    py::class_<NML_Context::NML_SimulationContext>(nml_module, "NML_SimulationContext")
        .def_readonly("simulation_component_id", &NML_Context::NML_SimulationContext::simulation_component_id)
        .def_readonly("target_network_id", &NML_Context::NML_SimulationContext::target_network_id)
        .def_readonly("step_dt", &NML_Context::NML_SimulationContext::step_dt, "Seconds per tick.")
        .def_readonly("simulation_duration", &NML_Context::NML_SimulationContext::simulation_duration, "Seconds.")
        .def_readonly("total_tick_count", &NML_Context::NML_SimulationContext::total_tick_count)
        .def_readonly("total_neuron_count", &NML_Context::NML_SimulationContext::total_neuron_count)
        .def_readonly("random_seed", &NML_Context::NML_SimulationContext::random_seed)
        .def_readonly("total_edge_count", &NML_Context::NML_SimulationContext::total_edge_count)
        .def_readonly("maximum_edge_delay", &NML_Context::NML_SimulationContext::maximum_edge_delay, "Ticks.")
        .def_readonly("population_base_indices", &NML_Context::NML_SimulationContext::population_base_indices,
                      "Population id -> where its cells start in cell_state.")
        .def_property_readonly("global_constants", [](const NML_Context::NML_SimulationContext &self) {
            std::unordered_map<std::string, f64> constants;
            for (const auto &[name, value] : self.global_constants) constants[name] = value.float64;
            return constants;
        }, "Name -> SI value.")
        .def_property_readonly("cell_instances", [](py::object self) {
            return references_in(self.cast<const NML_Context::NML_SimulationContext &>().cell_instances, self);
        })
        .def_property_readonly("synapse_instances", [](py::object self) {
            return references_in(self.cast<const NML_Context::NML_SimulationContext &>().synapse_instances, self);
        }, "In synapse prototype order: an edge's prototype index is its position here.")
        .def_property_readonly("network_data", [](py::object self) {
            return reference_or_none(&self.cast<const NML_Context::NML_SimulationContext &>().network_data, self);
        })
        .def_property_readonly("input_profiles", [](py::object self) {
            return references_in(self.cast<const NML_Context::NML_SimulationContext &>().input_profiles, self);
        })
        .def_property_readonly("recording_profiles", [](py::object self) {
            return references_in(self.cast<const NML_Context::NML_SimulationContext &>().recording_profiles, self);
        });

    py::class_<NML_Context>(nml_module, "NML_Context",
            "A parsed NeuroML/LEMS model. Build one with parse() to inspect a model without a GPU, or "
            "read SpikeEngine.context for the one an engine was built from. Don't call parse() or "
            "reset() on an engine's own context: the engine still reads it.")
        .def(py::init<>())
        .def("parse", &NML_Context::parse, py::arg("main_filepath"),
             "Parses the document, everything it includes, and the standard library it names.")
        .def("reset", &NML_Context::reset)
        .def("validate_lems_schema", &NML_Context::validate_lems_schema, py::arg("lems_filepath"),
             "XSD-validates a NeuroML document (a LEMS root is not validated and passes). The "
             "errors land in last_schema_validation_errors.")
        .def_property_readonly("simulation", [](py::object self) {
            return reference_or_none(&self.cast<const NML_Context &>().simulation, self);
        })
        .def_property_readonly("model_units", [](py::object self) {
            return references_by_key(self.cast<const NML_Context &>().model_units, self);
        })
        .def_property_readonly("model_constants", [](const NML_Context &self) {
            std::unordered_map<std::string, f64> constants;
            for (const auto &[name, value] : self.model_constants) constants[name] = value.float64;
            return constants;
        })
        .def_property_readonly("component_types", [](py::object self) {
            return references_by_key(self.cast<const NML_Context &>().component_types, self);
        })
        .def_property_readonly("component_instances", [](py::object self) {
            return references_by_key(self.cast<const NML_Context &>().component_instances, self);
        }, "Keyed by full path.")
        .def_property_readonly("document_roots", [](py::object self) {
            return references_to(self.cast<const NML_Context &>().document_roots, self);
        })
        .def_readonly("document_filepaths", &NML_Context::document_filepaths)
        .def_readonly("target_component_id", &NML_Context::target_component_id)
        .def_property_readonly("main_document_root", [](py::object self) {
            return reference_or_none(self.cast<const NML_Context &>().main_document_root, self);
        })
        .def_readonly("STANDARD_LIBRARY_PATH", &NML_Context::STANDARD_LIBRARY_PATH)
        .def_readonly("NML_SCHEMA_PATH", &NML_Context::NML_SCHEMA_PATH)
        .def_readonly("last_schema_validation_errors", &NML_Context::last_schema_validation_errors)
        .def("find_instance", [](py::object self, const String &instance_id) {
            return reference_or_none(self.cast<const NML_Context &>().find_instance(instance_id), self);
        }, py::arg("instance_id"))
        .def("is_instance_of", &NML_Context::is_instance_of, py::arg("instance"), py::arg("type_name"),
             "Whether the instance's type is type_name or extends it.")
        .def("resolve_quantity", &NML_Context::resolve_quantity, py::arg("value"),
             "'-60mV' -> -0.06, using the document's own <Unit> declarations as well as the built-in ones.")
        .def("get_cell_variable_count", &NML_Context::get_cell_variable_count, py::arg("cell_instance_index"))
        .def("get_cell_state_size", &NML_Context::get_cell_state_size)
        .def("get_population_size", &NML_Context::get_population_size, py::arg("population"))
        .def("neuron_index_of", &NML_Context::neuron_index_of, py::arg("cell_memory_index"),
             "The neuron whose cell occupies a cell_state index (a resolve_path result), or -1.")
        .def("resolve_path", &NML_Context::resolve_path, py::arg("path"), py::arg("current") = nullptr,
             "A LEMS path such as 'pop1[0]/v' -> its index in cell_state.");

    // ── nml: model functions ──────────────────────────────────────────────────────
    nml_module.def("component_parameter_values", &component_parameter_values,
                   py::arg("context"), py::arg("cell"),
                   "Every Parameter, Constant and DerivedParameter the instance's type resolves, in SI units.");
    nml_module.def("starting_values", &starting_values, py::arg("context"), py::arg("cell"),
                   py::arg("random_generator") = nullptr,
                   "The parameter values plus every state variable's starting value: OnStart, otherwise 0. "
                   "An OnStart that calls random() draws from random_generator.");
    nml_module.def("network_populations", [](py::object context) {
        return references_to(network_populations(context.cast<const NML_Context &>()), context);
    }, py::arg("context"), "The target network's populations, in document order.");
    nml_module.def("population_cell", [](py::object context, const NML_ComponentInstance &population) {
        return reference_or_none(&population_cell(context.cast<const NML_Context &>(), population), context);
    }, py::arg("context"), py::arg("population"), "The cell a population is made of.");
    nml_module.def("spike_history_length", &spike_history_length, py::arg("context"),
                   "Rows in the engine's spike_history ring.");
    nml_module.def("output_format_for_filename", &output_format_for_filename, py::arg("filename"));
    nml_module.def("evaluate_lems",
                   py::overload_cast<const String &, const UnorderedMap<String, f64> &, const String &, RandomGenerator *>(
                           &evaluate_lems),
                   py::arg("expression"), py::arg("values"), py::arg("owner_name") = String("python"),
                   py::arg("random_generator") = nullptr,
                   "A LEMS expression's value on the host; every name it reads must be in values. random() "
                   "draws from random_generator.");

    py::class_<RandomGenerator>(nml_module, "RandomGenerator")
        .def(py::init<u64>(), py::arg("seed") = 0)
        .def("uniform", &RandomGenerator::uniform, "Uniform on (0, 1), never 0 or 1 even as a float; LEMS random(1).")
        .def("normal", &RandomGenerator::normal, py::arg("mean"), py::arg("standard_deviation"))
        .def("exponential", &RandomGenerator::exponential, py::arg("rate"))
        .def("poisson", &RandomGenerator::poisson, py::arg("mean"));

    py::class_<LemsExpressionToken> lems_token(nml_module, "LemsExpressionToken");
    py::enum_<LemsExpressionToken::Kind>(lems_token, "Kind")
        .value("Number", LemsExpressionToken::Kind::Number)
        .value("Identifier", LemsExpressionToken::Kind::Identifier)
        .value("Operator", LemsExpressionToken::Kind::Operator)
        .value("OpenParen", LemsExpressionToken::Kind::OpenParen)
        .value("CloseParen", LemsExpressionToken::Kind::CloseParen)
        .value("Comma", LemsExpressionToken::Kind::Comma)
        .value("End", LemsExpressionToken::Kind::End);
    lems_token
        .def_readonly("lexeme", &LemsExpressionToken::lexeme)
        .def_readonly("kind", &LemsExpressionToken::kind)
        .def("__repr__", [](const LemsExpressionToken &self) { return "<LemsExpressionToken " + self.lexeme + ">"; });
    nml_module.def("tokenize_lems", &tokenize_lems, py::arg("expression"), py::arg("owner_name") = String("python"));

    // ── k^2-tree ──────────────────────────────────────────────────────────────────
    // Owned by the weight matrix and reached through it.
    py::class_<K2Tree>(m, "K2Tree", "The compressed adjacency: which (source, target) pairs are edges.")
        .def_readonly_static("ADJACENT_BATCH_QUERY_CAP", &K2Tree::ADJACENT_BATCH_QUERY_CAP)
        .def_readonly("branching_factor", &K2Tree::branching_factor)
        .def_readonly("superblock_size_words", &K2Tree::superblock_size_words)
        .def_readonly("node_count", &K2Tree::node_count)
        .def_readonly("padded_node_count", &K2Tree::padded_node_count)
        .def_readonly("tree_height", &K2Tree::tree_height)
        .def_readonly("internal_bit_count", &K2Tree::internal_bit_count)
        .def_readonly("internal_node_words_length", &K2Tree::internal_node_words_length)
        .def_readonly("leaf_node_words_length", &K2Tree::leaf_node_words_length)
        .def_readonly("rank_superblock_length", &K2Tree::rank_superblock_length)
        .def_readonly("rank_subblock_length", &K2Tree::rank_subblock_length)
        .def("adjacent", &K2Tree::adjacent, py::arg("source_node"), py::arg("target_node"),
             "1 if source_node -> target_node is an edge, otherwise 0.")
        .def("get_neighbors", [](const K2Tree &self, s32 node_index, s64 max_neighbor_count) {
            std::vector<s32> buffer(static_cast<usize>(std::max<s64>(max_neighbor_count, 1)));
            const s64 found = self.get_neighbors(node_index, buffer.data(), max_neighbor_count);
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"), py::arg("max_neighbor_count"))
        .def("get_predecessors", [](const K2Tree &self, s32 node_index, s64 max_neighbor_count) {
            std::vector<s32> buffer(static_cast<usize>(std::max<s64>(max_neighbor_count, 1)));
            const s64 found = self.get_predecessors(node_index, buffer.data(), max_neighbor_count);
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"), py::arg("max_neighbor_count"))
        .def("adjacent_batch", [](const K2Tree &self,
                                  py::array_t<s32, py::array::c_style | py::array::forcecast> source_indices,
                                  py::array_t<s32, py::array::c_style | py::array::forcecast> target_indices) {
            if (source_indices.size() != target_indices.size()) {
                throw std::invalid_argument("adjacent_batch: source_indices and target_indices differ in length");
            }
            std::vector<u8> output(static_cast<usize>(source_indices.size()));
            if (!output.empty()) {
                self.adjacent_batch(source_indices.data(), target_indices.data(), output.data(),
                                    static_cast<s32>(source_indices.size()));
            }
            return to_numpy(output.data(), (s64)output.size());
        }, py::arg("source_indices"), py::arg("target_indices"), "0 or 1 per (source, target) pair.")
        .def("trace", &K2Tree::trace, py::arg("source_node"), py::arg("target_node"),
             "Logs the tree walk for one pair and returns adjacent()'s answer.")
        .def("save", [](const K2Tree &self, const std::string &path) { self.save(path.c_str()); }, py::arg("path"));

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

    py::class_<ScaleResult>(m, "ScaleResult")
        .def_readonly("target_root_mean_square", &ScaleResult::target_root_mean_square)
        .def_readonly("scale_factor", &ScaleResult::scale_factor)
        .def_readonly("before", &ScaleResult::before)
        .def_readonly("after", &ScaleResult::after);

    // Owned by the engine and exposed by reference, so the class is bound without any
    // constructor: building one takes a GPU backend that only the engine holds.
    py::class_<WeightMatrix>(m, "WeightMatrix",
            "Every per-edge value of the network's synapses -- weight, delay and LEMS state -- as one "
            "plane each of M_k = U diag(Ck) V^T over the k^2-tree's edges, plus the sparse delta "
            "matrix S of updates since the last fit.")
        // Plane numbers and fitting constants.
        .def_readonly_static("WEIGHT_PLANE", &WeightMatrix::WEIGHT_PLANE)
        .def_readonly_static("DELAY_PLANE", &WeightMatrix::DELAY_PLANE)
        .def_readonly_static("FIRST_STATE_VARIABLE_PLANE", &WeightMatrix::FIRST_STATE_VARIABLE_PLANE)
        .def_readonly_static("LANE_GROUP", &WeightMatrix::LANE_GROUP)
        .def_readonly_static("DEFAULT_FIT_TOLERANCE", &WeightMatrix::DEFAULT_FIT_TOLERANCE)
        .def_readonly_static("DEFAULT_FIT_RIDGE", &WeightMatrix::DEFAULT_FIT_RIDGE)
        .def_readonly_static("FIT_STALL_IMPROVEMENT", &WeightMatrix::FIT_STALL_IMPROVEMENT)
        .def_readonly_static("FIT_STALL_WINDOW_SWEEP_COUNT", &WeightMatrix::FIT_STALL_WINDOW_SWEEP_COUNT)
        .def_readonly_static("MAXIMUM_FIT_SWEEP_COUNT", &WeightMatrix::MAXIMUM_FIT_SWEEP_COUNT)
        .def_readonly_static("DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION",
                             &WeightMatrix::DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION)

        // Shape.
        .def_readonly("node_count", &WeightMatrix::node_count)
        .def_readonly("total_edge_count", &WeightMatrix::total_edge_count)
        .def_readonly("max_neighbor_count", &WeightMatrix::max_neighbor_count)
        .def_readonly("matrix_count", &WeightMatrix::matrix_count,
                      "Planes: weight, delay, then the largest synapse type's state variables.")
        .def_readonly("updated_plane_count", &WeightMatrix::updated_plane_count,
                      "How many planes the generated kernel updates each tick.")
        .def_readonly("rank", &WeightMatrix::rank, "Lanes in the basis.")
        .def_readonly("rank_float4_stride", &WeightMatrix::rank_float4_stride)
        .def_readwrite("check_indexing", &WeightMatrix::check_indexing)
        .def_property_readonly("k2tree", [](WeightMatrix &self) -> K2Tree & { return self.k2tree; })

        // Fitting.
        .def_readwrite("fit_tolerance", &WeightMatrix::fit_tolerance,
                       "The worst relative error a refit may leave on any plane. The engine sets it from "
                       "SpikeEngine.fit_tolerance before each refit it runs.")
        .def_readonly("fit_rank_budget", &WeightMatrix::fit_rank_budget,
                      "The fixed rank the construction fit used, or -1 when it searched.")
        .def_readonly("measured_fit_error", &WeightMatrix::measured_fit_error,
                      "Per plane: the worst relative error the last fit left.")
        .def("worst_fit_error", &WeightMatrix::worst_fit_error, "The worst of measured_fit_error.")

        // The basis itself, copied out.
        .def_property_readonly("U_matrix", [](const WeightMatrix &self) {
            return device_buffer_to_numpy_rows<f32>(self.U_matrix, self.node_count, self.rank_float4_stride * WeightMatrix::LANE_GROUP);
        }, "[node_count][rank] source factors.")
        .def_property_readonly("V_matrix", [](const WeightMatrix &self) {
            return device_buffer_to_numpy_rows<f32>(self.V_matrix, self.node_count, self.rank_float4_stride * WeightMatrix::LANE_GROUP);
        }, "[node_count][rank] target factors.")
        .def_property_readonly("coefficients", [](const WeightMatrix &self) {
            return device_buffer_to_numpy_rows<f32>(self.coefficients, self.matrix_count, self.rank_float4_stride * WeightMatrix::LANE_GROUP);
        }, "[matrix_count][rank]: one coefficient row Ck per plane.")
        .def_property_readonly("edge_row_offset", [](const WeightMatrix &self) {
            return device_buffer_to_numpy<s64>(self.edge_row_offset, self.node_count + 1);
        }, "Prefix sum over out-degree: node n's edges are ordinals edge_row_offset[n] .. edge_row_offset[n + 1].")

        // Projection runs.
        .def_property_readonly("projection_first_edge_ordinal", [](const WeightMatrix &self) {
            return to_numpy(self.projection_first_edge_ordinal.data(), (s64)self.projection_first_edge_ordinal.size());
        })
        .def_property_readonly("projection_edge_count", [](const WeightMatrix &self) {
            return to_numpy(self.projection_edge_count.data(), (s64)self.projection_edge_count.size());
        })
        .def_property_readonly("projection_synapse_prototype", [](const WeightMatrix &self) {
            return to_numpy(self.projection_synapse_prototype.data(), (s64)self.projection_synapse_prototype.size());
        })

        // S, the sparse delta matrix, and its sizing.
        .def_readonly("sparse_delta_capacity", &WeightMatrix::sparse_delta_capacity, "Update slots per plane.")
        .def_readonly("sparse_delta_entry_count", &WeightMatrix::sparse_delta_entry_count,
                      "Updates each plane holds.")
        .def_readwrite("plasticity_reserve_entries", &WeightMatrix::plasticity_reserve_entries,
                       "Slots per plane kept for plasticity on top of the threshold's share. Applies at "
                       "the next resize.")
        .def_readonly("pending_delta_capacity", &WeightMatrix::pending_delta_capacity,
                      "Room in the device's per-tick update queue; fixed when the kernel is built.")
        .def("sparse_delta_occupancy_fraction", &WeightMatrix::sparse_delta_occupancy_fraction,
             "Fraction of the edge set holding an update, in the fullest plane.")
        .def("delta_capacity_for_threshold", &WeightMatrix::delta_capacity_for_threshold,
             py::arg("occupancy_threshold_fraction"),
             "The per-plane capacity a refit threshold asks for (every edge when it is 0), plus the "
             "plasticity reserve.")
        .def("resize_delta_capacity", &WeightMatrix::resize_delta_capacity, py::arg("new_capacity"),
             "Resizes S, keeping its updates; it never shrinks below them.")
        .def("sparse_deltas", [](const WeightMatrix &self, s64 matrix_index) {
            if (matrix_index < 0 || matrix_index >= self.matrix_count) {
                throw std::invalid_argument("sparse_deltas: no plane " + std::to_string(matrix_index));
            }
            const s64 entry_count = self.sparse_delta_entry_count[(usize)matrix_index];
            if (entry_count == 0 || self.sparse_delta_value.is_empty()) {
                return py::make_tuple(to_numpy<s64>(nullptr, 0), to_numpy<f32>(nullptr, 0));
            }
            const s64 offset = matrix_index * self.sparse_delta_capacity;
            return py::make_tuple(to_numpy(self.sparse_delta_edge_ordinal.get_contents_as<s64>() + offset, entry_count),
                                  to_numpy(self.sparse_delta_value.get_contents_as<f32>() + offset, entry_count));
        }, py::arg("matrix_index"), "(edge_ordinals, deltas): the updates one plane of S holds, by ordinal.")
        .def_property_readonly("sparse_delta_row_start", [](const WeightMatrix &self) {
            return device_buffer_to_numpy_rows<s32>(self.sparse_delta_row_start, self.matrix_count, self.node_count + 1);
        }, "S's CSR row starts, [matrix_count][node_count + 1].")
        .def_property_readonly("sparse_delta_edge_ordinal", [](const WeightMatrix &self) {
            return device_buffer_to_numpy_rows<s64>(self.sparse_delta_edge_ordinal, self.matrix_count, self.sparse_delta_capacity);
        }, "S's entries' edge ordinals, [matrix_count][sparse_delta_capacity]; only the first "
           "sparse_delta_entry_count of each row are live.")
        .def_property_readonly("sparse_delta_value", [](const WeightMatrix &self) {
            return device_buffer_to_numpy_rows<f32>(self.sparse_delta_value, self.matrix_count, self.sparse_delta_capacity);
        }, "S's entries' deltas, laid out like sparse_delta_edge_ordinal.")
        .def_property_readonly("pending_delta_count", [](const WeightMatrix &self) -> s64 {
            if (self.pending_delta_count.is_empty()) return 0;
            return *self.pending_delta_count.get_contents_as<s32>();
        }, "Updates the device queued since the last compact_pending_deltas.")
        .def("pending_deltas", [](const WeightMatrix &self) {
            s64 staged = 0;
            if (!self.pending_delta_count.is_empty()) {
                staged = std::min<s64>(*self.pending_delta_count.get_contents_as<s32>(), self.pending_delta_capacity);
            }
            return py::make_tuple(device_buffer_to_numpy<s64>(self.pending_delta_edge_ordinal, staged),
                                  device_buffer_to_numpy<f32>(self.pending_delta_value, staged),
                                  device_buffer_to_numpy<s32>(self.pending_delta_matrix_index, staged));
        }, "(edge_ordinals, deltas, matrix_indices): what the device queued and has not been merged.")

        // Reads.
        .def("get", &WeightMatrix::get, py::arg("source_node"), py::arg("target_node"),
             "The weight plane at one edge: the basis plus its update in S.")
        .def("get_for_matrix", &WeightMatrix::get_for_matrix,
             py::arg("source_node"), py::arg("target_node"), py::arg("matrix_index"),
             "Any plane at one edge: the basis plus its update in S.")
        .def("get_edge_delay_ticks", &WeightMatrix::get_edge_delay_ticks,
             py::arg("source_node"), py::arg("target_node"),
             "The delay plane rounded to whole ticks, at least one: how the kernel uses it.")
        .def("get_edge_synapse_prototype", &WeightMatrix::get_edge_synapse_prototype,
             py::arg("source_node"), py::arg("target_node"),
             "Index into the context's synapse_instances of the synapse this edge carries.")
        .def("edge_ordinal", &WeightMatrix::edge_ordinal, py::arg("source_node"), py::arg("target_node"),
             "This edge's number in the canonical ordering, or None if it is not an edge.")
        .def("check_index_inbounds", py::overload_cast<s32, s32>(&WeightMatrix::check_index_inbounds, py::const_),
             py::arg("source"), py::arg("target"))
        .def("check_index_inbounds", py::overload_cast<s32>(&WeightMatrix::check_index_inbounds, py::const_),
             py::arg("node_index"))
        .def("get_neighbors", [](const WeightMatrix &self, s64 node_index) {
            std::vector<s32> buffer(static_cast<usize>(std::max<s64>(self.max_neighbor_count, 1)));
            const s64 found = self.get_neighbors(node_index, buffer.data());
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"), "Targets of node_index's edges, in canonical (ordinal) order.")
        .def("get_predecessors", [](const WeightMatrix &self, s64 node_index) {
            std::vector<s32> buffer(static_cast<usize>(std::max<s64>(self.max_neighbor_count, 1)));
            const s64 found = self.get_predecessors(node_index, buffer.data());
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"), "Sources of the edges into node_index.")
        // One value per real edge, indexed by edge ordinal. No padding rows and no sentinels.
        .def("neighbor_weights", [](const WeightMatrix &self) {
            std::vector<f32> values(static_cast<usize>(std::max<s64>(self.total_edge_count, 0)));
            if (!values.empty()) self.neighbor_weights(values.data());
            return to_numpy(values.data(), self.total_edge_count);
        }, "Every edge's weight, by ordinal.")
        .def("neighbor_weights_for_matrix", [](const WeightMatrix &self, s64 matrix_index) {
            std::vector<f32> values(static_cast<usize>(std::max<s64>(self.total_edge_count, 0)));
            if (!values.empty()) self.neighbor_weights_for_matrix(values.data(), matrix_index);
            return to_numpy(values.data(), self.total_edge_count);
        }, py::arg("matrix_index"), "Every edge's value in one plane, by ordinal.")
        .def("neighbor_weight_stats", &WeightMatrix::neighbor_weight_stats)

        // Updates and refits.
        .def("accumulate_edge_delta", &WeightMatrix::accumulate_edge_delta,
             py::arg("matrix_index"), py::arg("source_node"), py::arg("target_node"), py::arg("delta"),
             "Adds delta to one edge's value in one plane through S. Visible to reads at once and "
             "folded into the basis by the next refit. S grows if the plane is full.")
        .def("compact_pending_deltas", &WeightMatrix::compact_pending_deltas,
             "Merges what the device queued into S, growing S if a plane would overflow.")
        .def("refit", &WeightMatrix::refit, py::arg("ridge_regularization") = WeightMatrix::DEFAULT_FIT_RIDGE,
             "Fits the basis to the current values (basis plus S) until every plane meets "
             "fit_tolerance, adding lanes if it must, then empties S.")
        .def("is_refit_due", &WeightMatrix::is_refit_due, py::arg("occupancy_threshold_fraction"),
             "True once the fullest plane of S holds updates on that fraction of the edge set.")
        .def("scale_neighbor_weights_to_root_mean_square", &WeightMatrix::scale_neighbor_weights_to_root_mean_square,
             py::arg("target_root_mean_square"), py::arg("epsilon") = 1e-12f,
             "Scales the weight plane (its coefficient row and its updates) to a target RMS.")

        .def("save", [](const WeightMatrix &self, const std::string &path) {
            self.save(path.c_str());
        }, py::arg("path"))
        .def("load_from_disk", [](WeightMatrix &self, const std::string &path) {
            self.load_from_disk(path.c_str());
        }, py::arg("path"))

        .def("__repr__", [](const WeightMatrix &self) {
            return "<WeightMatrix nodes=" + std::to_string(self.node_count) +
                   " edges=" + std::to_string(self.total_edge_count) +
                   " planes=" + std::to_string(self.matrix_count) +
                   " rank=" + std::to_string(self.rank) + ">";
        });

    // ── the engine ────────────────────────────────────────────────────────────────
    py::class_<SpikeEngine> engine_class(m, "SpikeEngine");

    py::class_<SpikeEngine::ScheduledSpikeTrain>(engine_class, "ScheduledSpikeTrain")
        .def_readonly("neuron_index", &SpikeEngine::ScheduledSpikeTrain::neuron_index)
        .def_readonly("magnitude", &SpikeEngine::ScheduledSpikeTrain::magnitude)
        .def_readonly("event_ticks", &SpikeEngine::ScheduledSpikeTrain::event_ticks)
        .def_readonly("cursor", &SpikeEngine::ScheduledSpikeTrain::cursor, "Next event_ticks entry to deliver.");

    engine_class
        .def(py::init<const String &, bool, f32>(),
             py::arg("lems_input_file"),
             py::arg("enable_hebbian_plasticity") = false,
             py::arg("fit_tolerance") = SpikeEngine::FIT_TOLERANCE_STANDARD,
             "Parses a LEMS document, allocates and fills every buffer the model needs, "
             "builds the weight matrix from its connections, and compiles the tick kernel.")
        .def(py::init<const String &, const std::vector<std::vector<s32>> &, const String &,
                      f64, f64, bool, f32>(),
             py::arg("lems_input_file"), py::arg("adjacency"), py::arg("synapse_component_id"),
             py::arg("connection_weight") = 1.0,
             py::arg("connection_delay_seconds") = 0.0,
             py::arg("enable_hebbian_plasticity") = false,
             py::arg("fit_tolerance") = SpikeEngine::FIT_TOLERANCE_STANDARD,
             "The same, with connectivity supplied in code instead of in the document -- "
             "which is how a network too large to write as <connection> elements is built. "
             "adjacency[source] lists source's targets; every edge carries the named synapse, "
             "connection_weight and connection_delay_seconds.")
        .def(py::init<const String &, const std::vector<std::vector<s32>> &,
                      const std::vector<String> &, const std::vector<f64> &,
                      f64, f64, bool, f32>(),
             py::arg("lems_input_file"), py::arg("adjacency"),
             py::arg("synapse_component_ids"),
             py::arg("synapse_proportions") = std::vector<f64>{},
             py::arg("connection_weight") = 1.0,
             py::arg("connection_delay_seconds") = 0.0,
             py::arg("enable_hebbian_plasticity") = false,
             py::arg("fit_tolerance") = SpikeEngine::FIT_TOLERANCE_STANDARD,
             "The same, with several synapses. The edges are split, in adjacency order, into one "
             "contiguous share per synapse sized by synapse_proportions (normalised; empty means "
             "equal shares). Nothing is random. A neuron's edges can straddle two shares; read "
             "weights.get_edge_synapse_prototype for any edge's synapse.")

        // What the engine was built from.
        .def_property_readonly("context", [](SpikeEngine &self) -> NML_Context & { return self.context; },
                               "The parsed model. Read-only in effect: don't parse() or reset() it.")
        .def_property_readonly("weights", [](SpikeEngine &self) -> WeightMatrix & { return self.weights; },
                               "Every per-edge value of the network, by reference.")
        .def_readonly("total_neuron_count", &SpikeEngine::total_neuron_count)
        .def_readonly("spike_history_row_count", &SpikeEngine::spike_history_row_count)
        .def_readonly("lifetime", &SpikeEngine::lifetime, "Ticks in the run.")
        .def_readonly("step_dt", &SpikeEngine::step_dt, "Seconds per tick.")
        .def_readonly("simulation_seed", &SpikeEngine::simulation_seed)
        .def_property_readonly("random_generator", [](SpikeEngine &self) -> RandomGenerator & { return self.random_generator; },
                               "Draws every random() the run calls, seeded with simulation_seed; by reference.")
        .def_readonly("random_values_count", &SpikeEngine::random_values_count,
                      "Slots in random_values: one per random() call per neuron that runs it.")
        .def_readonly("alive", &SpikeEngine::alive)
        .def_readonly("hebbian_plasticity_enabled", &SpikeEngine::hebbian_plasticity_enabled,
                      "The generated kernel stages no plasticity updates yet, so this changes nothing.")

        // The generated kernel.
        .def_readonly("master_kernel_source", &SpikeEngine::master_kernel_source,
                      "The generated master kernel exactly as compiled.")
        .def_readonly("kernel_parameter_names", &SpikeEngine::kernel_parameter_names,
                      "master_step's bound parameters, in binding order.")
        .def_readonly("synapse_active_ticks", &SpikeEngine::synapse_active_ticks,
                      "Ticks after a neuron spikes during which its outgoing synapses run (longest "
                      "delay plus the slowest synapse's settle time); -1 when every synapse runs every tick.")
        .def_readonly("projection_run_count", &SpikeEngine::projection_run_count)

        // Weight-matrix fit and refit settings.
        .def_readonly_static("FIT_TOLERANCE_PRECISE", &SpikeEngine::FIT_TOLERANCE_PRECISE)
        .def_readonly_static("FIT_TOLERANCE_ACCURATE", &SpikeEngine::FIT_TOLERANCE_ACCURATE)
        .def_readonly_static("FIT_TOLERANCE_STANDARD", &SpikeEngine::FIT_TOLERANCE_STANDARD)
        .def_readonly_static("FIT_TOLERANCE_COMPACT", &SpikeEngine::FIT_TOLERANCE_COMPACT)
        .def_readwrite("fit_tolerance", &SpikeEngine::fit_tolerance,
                       "Worst relative error a refit may leave on any per-edge value.")
        .def_readwrite("refit_every_n_ticks", &SpikeEngine::refit_every_n_ticks, "0 disables.")
        .def_property("refit_occupancy_threshold_fraction",
                      [](const SpikeEngine &self) { return self.refit_occupancy_threshold_fraction; },
                      &SpikeEngine::set_refit_occupancy_threshold_fraction,
                      "Refit once the fullest plane of S holds updates on this fraction of the edges; "
                      "0 disables. Also the size S is kept at per plane: setting it resizes S.")
        .def("set_refit_occupancy_threshold_fraction", &SpikeEngine::set_refit_occupancy_threshold_fraction,
             py::arg("fraction"))
        .def_readwrite("minimum_ticks_between_refits", &SpikeEngine::minimum_ticks_between_refits,
                       "Refits are at least this many ticks apart; S grows in between. 0 allows any spacing.")
        .def_readonly("last_refit_tick", &SpikeEngine::last_refit_tick, "The construction fit counts as tick 0.")
        .def_readonly("weight_fit_rank_budget", &SpikeEngine::weight_fit_rank_budget,
                      "The fixed rank the construction fit was asked for, or -1 to search.")

        // Plasticity settings.
        .def_readonly_static("DEFAULT_PLASTICITY_DELTA_CAPACITY", &SpikeEngine::DEFAULT_PLASTICITY_DELTA_CAPACITY)
        .def_readwrite("plasticity_fold_every_n_ticks", &SpikeEngine::plasticity_fold_every_n_ticks)
        .def_readwrite("plasticity_target_root_mean_square", &SpikeEngine::plasticity_target_root_mean_square,
                       "After each refit, with plasticity on, the weights are scaled back to this RMS; "
                       "negative disables.")

        // Stimulus, as collected from the document.
        .def_property_readonly("continuous_injection_targets", [](const SpikeEngine &self) {
            return to_numpy(self.continuous_injection_targets.data(), (s64)self.continuous_injection_targets.size());
        })
        .def_property_readonly("continuous_injection_amplitudes", [](const SpikeEngine &self) {
            return to_numpy(self.continuous_injection_amplitudes.data(), (s64)self.continuous_injection_amplitudes.size());
        })
        .def_property_readonly("continuous_injection_start_ticks", [](const SpikeEngine &self) {
            return to_numpy(self.continuous_injection_start_ticks.data(), (s64)self.continuous_injection_start_ticks.size());
        })
        .def_property_readonly("continuous_injection_end_ticks", [](const SpikeEngine &self) {
            return to_numpy(self.continuous_injection_end_ticks.data(), (s64)self.continuous_injection_end_ticks.size());
        })
        .def_property_readonly("scheduled_spike_trains", [](py::object self) {
            return references_in(self.cast<const SpikeEngine &>().scheduled_spike_trains, self);
        })

        // Device state, copied out.
        .def_property_readonly("cell_state", [](const SpikeEngine &self) {
            require_alive(self, "cell_state");
            return device_buffer_to_numpy<f32>(self.cell_state, self.context.get_cell_state_size());
        }, "Every cell's state variables; context.resolve_path gives a variable's index.")
        .def_property_readonly("network_inputs", [](const SpikeEngine &self) {
            require_alive(self, "network_inputs");
            return device_buffer_to_numpy_rows<f32>(self.network_inputs, 2, self.total_neuron_count);
        }, "[2][total_neuron_count], alternating by tick parity.")
        .def_property_readonly("spike_history", [](const SpikeEngine &self) {
            require_alive(self, "spike_history");
            return device_buffer_to_numpy_rows<u8>(self.spike_history, self.spike_history_row_count,
                                                   self.total_neuron_count);
        }, "[spike_history_row_count][total_neuron_count]; row tick % rows is that tick's spikes.")
        .def_property_readonly("last_spiked", [](const SpikeEngine &self) {
            require_alive(self, "last_spiked");
            return device_buffer_to_numpy<s64>(self.last_spiked, self.total_neuron_count);
        }, "Each neuron's last spike tick, or NEVER_SPIKED_TICK.")
        .def_property_readonly("random_values", [](const SpikeEngine &self) {
            require_alive(self, "random_values");
            return device_buffer_to_numpy<f32>(self.random_values, self.random_values_count);
        }, "The last tick's uniform draws, one per random() call per neuron that runs it.")

        // Running.
        .def("run", &SpikeEngine::run,
             py::call_guard<py::gil_scoped_release>(),
             "Runs every tick the model's Simulation declared.")
        .def("step_simulation", &SpikeEngine::step_simulation, py::arg("tick"),
             py::call_guard<py::gil_scoped_release>(),
             "One tick: stimulus, the kernel, merging S, a refit when one is due, recording.")

        // Reading state.
        .def("read_state_variable", [](const SpikeEngine &self, s64 neuron_index, const String &variable_name) {
            require_alive(self, "read_state_variable");
            return self.read_state_variable(neuron_index, variable_name);
        }, py::arg("neuron_index"), py::arg("variable_name"),
           "One neuron's named state variable, read back from the GPU.")
        .def("state_variable_array", [](const SpikeEngine &self, const String &variable_name) {
            require_alive(self, "state_variable_array");
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
        .def("synapse_state_variable_plane", &SpikeEngine::synapse_state_variable_plane,
             py::arg("synapse_component_id"), py::arg("variable_name"),
             "The weights plane holding one of a synapse's LEMS state variables on its edges.")

        // Results.
        .def_property_readonly("spike_counts_per_neuron", [](const SpikeEngine &self) {
            return to_numpy(self.spike_counts_per_neuron.data(), static_cast<s64>(self.spike_counts_per_neuron.size()));
        }, "Spikes per neuron over the run so far.")
        .def_property_readonly("recorded_spikes", [](const SpikeEngine &self) {
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
        }, "(time_seconds, neuron_index), parallel arrays over every spike so far.")
        .def_property_readonly("traced_selections", [](py::object self) {
            return references_in(self.cast<const SpikeEngine &>().traced_selections, self);
        }, "The document's OutputColumns, one per column of recorded_traces.")
        .def_property_readonly("recorded_trace_times", [](const SpikeEngine &self) {
            return to_numpy(self.recorded_trace_times.data(), static_cast<s64>(self.recorded_trace_times.size()));
        })
        .def_property_readonly("recorded_traces", [](const SpikeEngine &self) {
            return to_numpy_rows(self.recorded_traces.data(), static_cast<s64>(self.recorded_trace_times.size()),
                                 static_cast<s64>(self.traced_selections.size()));
        }, "[recorded tick][traced selection].")
        .def("mean_firing_rate_hertz", &SpikeEngine::mean_firing_rate_hertz)
        .def("fraction_of_neurons_that_spiked", &SpikeEngine::fraction_of_neurons_that_spiked)

        // Writing results.
        .def("write_recordings", &SpikeEngine::write_recordings,
             "Closes the membrane video and writes every OutputFile and EventOutputFile the model declared.")
        .def("record_membrane_video", &SpikeEngine::record_membrane_video,
             py::arg("path"), py::arg("frame_stride") = 1,
             "Records every neuron's `v` to a .spire file every frame_stride ticks. Call before run().")
        .def_readonly("membrane_video_frame_stride", &SpikeEngine::membrane_video_frame_stride)
        .def("write_spike_file", &SpikeEngine::write_spike_file, py::arg("path"),
             "Every spike of the run as 'time<tab>neuron'.")

        .def("shutdown", &SpikeEngine::shutdown)
        .def("__repr__", [](const SpikeEngine &self) {
            return "<SpikeEngine neurons=" + std::to_string(self.total_neuron_count) +
                   " ticks=" + std::to_string(self.lifetime) +
                   " dt=" + std::to_string(self.step_dt) + ">";
        });
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
