#include <algorithm>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "backend.h"
#include "engine.h"
#include "k2tree.h"
#include "log.h"
#include "recording.h"
#include "topologies.h"
#include "types.h"
#include "units.h"
#include "weight_matrix.h"
#include "components.h"
#include "dynamics.h"
#include "node.h"
#include "parser.h"
#include "utilities.h"

namespace py = pybind11;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

// Copies rather than views: a view into an engine buffer would dangle once the engine is freed.
template <typename Element_Type>
py::array_t<Element_Type> to_numpy(const Element_Type *data, s64 count) {
    py::array_t<Element_Type> result(static_cast<py::ssize_t>(std::max<s64>(count, 0)));
    if (count > 0 && data != nullptr) {
        std::memcpy(result.mutable_data(), data, static_cast<usize>(count) * sizeof(Element_Type));
    }
    return result;
}

// The same, shaped [row_count][column_count] over row-major data.
template <typename Element_Type>
py::array_t<Element_Type> to_numpy_rows(const Element_Type *data, s64 row_count, s64 column_count) {
    const s64 rows = std::max<s64>(row_count, 0);
    const s64 columns = std::max<s64>(column_count, 0);
    py::array_t<Element_Type> result({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
    if (rows * columns > 0 && data != nullptr) {
        std::memcpy(result.mutable_data(), data, static_cast<usize>(rows * columns) * sizeof(Element_Type));
    }
    return result;
}

// A device buffer's first `count` elements. Empty when the buffer was never allocated.
template <typename Element_Type>
py::array_t<Element_Type> device_buffer_to_numpy(const Engine_Pointer &pointer, s64 count) {
    if (pointer.is_empty()) return to_numpy<Element_Type>(nullptr, 0);
    return to_numpy(pointer.get_contents_as<Element_Type>(), count);
}

template <typename Element_Type>
py::array_t<Element_Type> device_buffer_to_numpy_rows(const Engine_Pointer &pointer, s64 row_count, s64 column_count) {
    if (pointer.is_empty()) return to_numpy_rows<Element_Type>(nullptr, 0, 0);
    return to_numpy_rows(pointer.get_contents_as<Element_Type>(), row_count, column_count);
}

// References that keep `owner` alive. A copy would hold pointers (a type's `extends`, an
// instance's `component_type`) into memory the owner frees.
template <typename Value_Type>
py::object reference_or_none(const Value_Type *value, py::handle owner) {
    if (value == nullptr) return py::none();
    return py::cast(value, py::return_value_policy::reference_internal, owner);
}

template <typename Value_Type>
py::list references_in(const Vector<Value_Type> &values, py::handle owner) {
    py::list result;
    for (const Value_Type &value : values) result.append(reference_or_none(&value, owner));
    return result;
}

template <typename Value_Type>
py::list references_to(const Vector<Value_Type *> &values, py::handle owner) {
    py::list result;
    for (const Value_Type *value : values) result.append(reference_or_none(value, owner));
    return result;
}

template <typename Value_Type>
py::dict references_by_key(const Unordered_Map<String, Value_Type> &values, py::handle owner) {
    py::dict result;
    for (const auto &[key, value] : values) result[py::str(key)] = reference_or_none(&value, owner);
    return result;
}

// Every engine buffer lives in the slab shutdown() releases.
void require_alive(const Spike_Engine &engine, const char *what) {
    if (!engine.alive) throw Runtime_Error(String(what) + ": the engine has been shut down");
}

// How often run() takes the GIL back to raise a pending Ctrl-C.
constexpr Milliseconds SIGNAL_CHECK_INTERVAL{100};

} // namespace

PYBIND11_MODULE(_spikecorec, m) {
    m.doc() = "spikecorec — a NeuroML-driven GPU spiking neural network engine";
    m.attr("__version__") = "0.1.0";

    // Mirrors the C++ namespaces (spikecorec, spikecorec::nml as `nml`, spikecorec::units as
    // `units`), with every class and function under its C++ name.
    py::module_ nml_module = m.def_submodule("nml", "The parsed NeuroML/LEMS model (spikecorec::nml).");
    py::module_ units_module = m.def_submodule("units", "Unit conversion (spikecorec::units).");

    // ── logging ───────────────────────────────────────────────────────────────────
    m.def("set_log_level", [](const String &level) {
        spikecorec::log::logger().set_level(spikecorec::log::level_from_string(level));
    }, py::arg("level"),
       "One of: trace, debug, info, warn, err, critical, off.");
    m.attr("LOG_PATH") = spikecorec::log::LOG_PATH;

    m.attr("NEVER_SPIKED_TICK") = NEVER_SPIKED_TICK;
    m.attr("MAX_RANK_FLOAT4_STRIDE") = MAX_RANK_FLOAT4_STRIDE;
    m.attr("DEFAULT_BRANCHING_FACTOR") = DEFAULT_BRANCHING_FACTOR;

    // ── units ─────────────────────────────────────────────────────────────────────
    py::class_<units::Unit_Definition>(units_module, "Unit_Definition",
            "A <Unit>: si = written magnitude * scale + offset.")
        .def(py::init<>())
        .def_readwrite("scale", &units::Unit_Definition::scale)
        .def_readwrite("offset", &units::Unit_Definition::offset)
        .def_readwrite("dimension", &units::Unit_Definition::dimension, "The <Unit>'s dimension; '' when it states none.")
        .def("__repr__", [](const units::Unit_Definition &self) {
            return "<Unit_Definition scale=" + std::to_string(self.scale) +
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
    units_module.def("unit_suffix_is_known", &units::unit_suffix_is_known, py::arg("suffix"),
                     "Whether the built-in table unit_suffix_scale reads has the suffix.");
    units_module.def("parse_quantity", &units::parse_quantity, py::arg("value"),
                     "'-60mV' -> -0.06 using only the built-in unit table. NML_Context.resolve_quantity "
                     "also knows the document's own units.");
    units_module.def("split_quantity", &units::split_quantity, py::arg("value"),
                     "'-60mV' -> (-60.0, 'mV').");

    // ── topology generators ───────────────────────────────────────────────────────
    // Adjacency lists for the Spike_Engine constructors that take connectivity in code.
    m.def("square_torus", &square_torus, py::arg("side_length"));
    m.def("small_world_torus", &small_world_torus,
          py::arg("side_length"), py::arg("random_fanout") = 4, py::arg("seed") = -1);
    m.def("random_fixed_outdegree", &random_fixed_outdegree,
          py::arg("side_length"), py::arg("fanout") = 8, py::arg("seed") = -1);

    // ── recording configuration (spikecorec::Recording_Config, as the parser fills it) ──
    py::enum_<Output_File_Format>(m, "Output_File_Format")
        .value("SPIRE", Output_File_Format::SPIRE)
        .value("SPIREGZIP", Output_File_Format::SPIREGZIP)
        .value("SPIREBZ2", Output_File_Format::SPIREBZ2)
        .value("SPIREXZ", Output_File_Format::SPIREXZ)
        .value("SPIKE_EVENTS", Output_File_Format::SPIKE_EVENTS)
        .value("NML_STANDARD", Output_File_Format::NML_STANDARD);

    py::class_<Recording_Selection>(m, "Recording_Selection")
        .def_readonly("quantity_path", &Recording_Selection::quantity_path, "Verbatim LEMS path, e.g. 'pop1[0]/v'.")
        .def_readonly("variable_name", &Recording_Selection::variable_name,
                      "The state variable a column records, e.g. 'v'; '' for an event selection.")
        .def_readonly("event_port", &Recording_Selection::event_port, "EventSelection's port; '' for a column.")
        .def_readonly("selection_id", &Recording_Selection::selection_id,
                      "The element's own id; an event file writes it beside each spike time.")
        .def_readonly("neuron_index", &Recording_Selection::neuron_index,
                      "For a column, the variable's cell_state index; for an event, the cell's; -1 when none.")
        .def("__repr__", [](const Recording_Selection &self) {
            return "<Recording_Selection " + self.quantity_path + ">";
        });

    py::class_<Recording_Config>(m, "Recording_Config")
        .def_readonly("output_filenames", &Recording_Config::output_filenames)
        .def_readonly("file_output_format", &Recording_Config::file_output_format)
        .def_readonly("output_directory", &Recording_Config::output_directory,
                      "Where a relative filename is written: the LEMS file's directory, or the element's path in it.")
        .def_readonly("event_format", &Recording_Config::event_format,
                      "An EventOutputFile's column order, 'TIME_ID' or 'ID_TIME'; '' for an OutputFile.")
        .def_property_readonly("selections", [](py::object self) {
            return references_in(self.cast<const Recording_Config &>().selections, self);
        })
        .def_readonly("recordings_count", &Recording_Config::recordings_count)
        .def_readonly_static("DEFAULT_MAX_LOG_BYTES", &Recording_Config::DEFAULT_MAX_LOG_BYTES);

    // ── .spire recordings ─────────────────────────────────────────────────────────
    py::enum_<Spire_Compression>(m, "Spire_Compression")
        .value("None_", Spire_Compression::None)
        .value("Gzip", Spire_Compression::Gzip)
        .value("Xz", Spire_Compression::Xz)
        .value("Bz2", Spire_Compression::Bz2);

    m.def("resolve_spire_compression", &resolve_spire_compression,
          py::arg("filename"), py::arg("requested") = Optional<String>("auto"),
          "(compression, filename): 'auto' infers from the extension (.gz/.gzip, .xz/.lzma, .bz2); "
          "'none', 'gzip', 'xz' and 'bz2' ask for one explicitly.");

    // A .spire recording, compressed or not, as a (frame_count, neuron_count) float32 array.
    m.def("read_spire_recording", [](const String &filename) {
        Spire_Recording recording = read_spire_recording(filename);
        return to_numpy_rows(recording.frames.data(), recording.frame_count, recording.neuron_count);
    }, py::arg("filename"));

    py::class_<Spire_Writer>(m, "Spire_Writer", "Uncompressed .spire encoder, one frame at a time.")
        .def(py::init<const String &, s64>(), py::arg("filename"), py::arg("neuron_count"))
        .def("write_frame", [](Spire_Writer &self,
                               py::array_t<f32, py::array::c_style | py::array::forcecast> frame) {
            // write_frame reads neuron_count values, so a shorter array would be read past its end.
            if (frame.size() != self.neuron_count()) {
                throw Invalid_Argument("write_frame: the frame has " + std::to_string(frame.size()) +
                                            " values, not " + std::to_string(self.neuron_count()));
            }
            self.write_frame(frame.data());
        }, py::arg("membrane_potentials"))
        .def("flush", &Spire_Writer::flush, "Pushes the frames written so far to the file.")
        .def("close", &Spire_Writer::close, "Flushes and closes the file; later calls do nothing.")
        .def("__enter__", [](py::object self) { return self; })
        .def("__exit__", [](Spire_Writer &self, const py::object &, const py::object &, const py::object &) {
            self.close();
        })
        .def_property_readonly("neuron_count", &Spire_Writer::neuron_count);

    py::class_<Spire_Reader>(m, "Spire_Reader", "Uncompressed .spire decoder, one frame at a time.")
        .def(py::init<const String &>(), py::arg("filename"))
        .def("read_frame", [](Spire_Reader &self) -> py::object {
            Vector<f32> frame(static_cast<usize>(self.neuron_count()));
            if (!self.read_frame(frame.data())) return py::none();
            return to_numpy(frame.data(), self.neuron_count());
        }, "The next frame, or None at the end of the file.")
        .def_property_readonly("neuron_count", &Spire_Reader::neuron_count);

    // The buffering and compression layer under record_membrane_video, for custom record loops.
    py::class_<Simulation_Recorder>(m, "Simulation_Recorder")
        .def(py::init<const String &, s64, Optional<String>, Optional<int>,
                      bool, usize, usize>(),
             py::arg("filename"), py::arg("neuron_count"),
             py::arg("compression") = Optional<String>("auto"),
             py::arg("compression_level") = Optional<int>{},
             // Not `async`: that is a reserved word in Python.
             py::arg("compression_async") = false,
             py::arg("queue_max") = static_cast<usize>(8),
             py::arg("chunk_bytes") = static_cast<usize>(4 * 1024 * 1024))
        .def_property_readonly("neuron_count", &Simulation_Recorder::neuron_count)
        .def("record_frame", [](Simulation_Recorder &self,
                                py::array_t<f32, py::array::c_style | py::array::forcecast> frame) {
            // Passes the length so record_frame rejects a wrongly sized array.
            self.record_frame(frame.data(), frame.size());
        }, py::arg("membrane_potentials"))
        .def("finish", &Simulation_Recorder::finish);

    // ── nml: the parsed document tree ─────────────────────────────────────────────
    py::enum_<NML_Declaration_Type>(nml_module, "NML_Declaration_Type")
        .value("NamedDimension", NML_Declaration_Type::NamedDimension)
        .value("Parameter", NML_Declaration_Type::Parameter)
        .value("DerivedParameter", NML_Declaration_Type::DerivedParameter)
        .value("Constant", NML_Declaration_Type::Constant)
        .value("Requirement", NML_Declaration_Type::Requirement)
        .value("Exposure", NML_Declaration_Type::Exposure)
        .value("Property", NML_Declaration_Type::Property)
        .value("Fixed", NML_Declaration_Type::Fixed)
        .value("Text", NML_Declaration_Type::Text)
        .value("EventPort", NML_Declaration_Type::EventPort)
        .value("NamedTypeReference", NML_Declaration_Type::NamedTypeReference)
        .value("Attachment", NML_Declaration_Type::Attachment)
        .value("ComponentReference", NML_Declaration_Type::ComponentReference)
        .value("Link", NML_Declaration_Type::Link)
        .value("Children", NML_Declaration_Type::Children)
        .value("Child", NML_Declaration_Type::Child)
        .value("Path", NML_Declaration_Type::Path)
        .value("StateVariable", NML_Declaration_Type::StateVariable)
        .value("DerivedVariable", NML_Declaration_Type::DerivedVariable)
        .value("ConditionalDerivedVariable", NML_Declaration_Type::ConditionalDerivedVariable)
        .value("Case", NML_Declaration_Type::Case)
        .value("TimeDerivative", NML_Declaration_Type::TimeDerivative)
        .value("OnCondition", NML_Declaration_Type::OnCondition)
        .value("OnEvent", NML_Declaration_Type::OnEvent)
        .value("OnStart", NML_Declaration_Type::OnStart)
        .value("OnEntry", NML_Declaration_Type::OnEntry)
        .value("StateAssignment", NML_Declaration_Type::StateAssignment)
        .value("EventOut", NML_Declaration_Type::EventOut)
        .value("Transition", NML_Declaration_Type::Transition)
        .value("Regime", NML_Declaration_Type::Regime)
        .value("ChildInstance", NML_Declaration_Type::ChildInstance)
        .value("With", NML_Declaration_Type::With)
        .value("EventConnection", NML_Declaration_Type::EventConnection)
        .value("NOT_A_TYPE", NML_Declaration_Type::NOT_A_TYPE);

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

    // A node of the document tree the context owns; handed out by reference only.
    py::class_<NML_Node>(nml_module, "NML_Node")
        .def_property_readonly("body", [](const NML_Node &self) -> const NML_Tag & { return self.body; })
        .def_property_readonly("children", [](py::object self) {
            return references_to(children_of(&self.cast<const NML_Node &>()), self);
        }, "The element children, in document order; empty for a leaf.")
        .def("__repr__", [](const NML_Node &self) { return "<NML_Node " + self.body.tag_name + ">"; });

    // ── nml: component types and instances ────────────────────────────────────────
    py::class_<NML_Dynamics_Expression>(nml_module, "NML_Dynamics_Expression",
            "One entry of a ComponentType's <Dynamics>, in source order.")
        .def_readonly("source_tag", &NML_Dynamics_Expression::source_tag)
        .def_readonly("target", &NML_Dynamics_Expression::target,
                      "Variable written, or exposure / port / regime name.")
        .def_readonly("expression", &NML_Dynamics_Expression::expression, "value= / test= source, verbatim.")
        .def_readonly("regime_name", &NML_Dynamics_Expression::regime_name, "Owning Regime; '' when regime-free.")
        .def_readonly("condition", &NML_Dynamics_Expression::condition)
        .def_readonly("select", &NML_Dynamics_Expression::select)
        .def_readonly("reduce", &NML_Dynamics_Expression::reduce)
        .def("__repr__", [](const NML_Dynamics_Expression &self) {
            return "<NML_Dynamics_Expression " + self.target + " = " + self.expression + ">";
        });

    py::class_<NML_Structure_Entry>(nml_module, "NML_Structure_Entry",
            "One entry of a ComponentType's <Structure>, in source order.")
        .def_readonly("source_tag", &NML_Structure_Entry::source_tag)
        .def_readonly("component", &NML_Structure_Entry::component,
                      "ChildInstance: the ComponentReference instantiated per instance.")
        .def_readonly("instance", &NML_Structure_Entry::instance, "With: 'this', 'parent', or a Path or child name.")
        .def_readonly("alias", &NML_Structure_Entry::alias, "With: the name it is bound to.")
        .def_readonly("source", &NML_Structure_Entry::source, "EventConnection: the alias in its from=.")
        .def_readonly("target", &NML_Structure_Entry::target, "EventConnection: the alias in its to=.")
        .def_readonly("receiver", &NML_Structure_Entry::receiver)
        .def_readonly("source_port", &NML_Structure_Entry::source_port)
        .def_readonly("target_port", &NML_Structure_Entry::target_port)
        .def("__repr__", [](const NML_Structure_Entry &self) {
            return "<NML_Structure_Entry " + self.component + self.instance + self.source + " " + self.alias +
                   self.target + ">";
        });

    py::class_<NML_Component_Type>(nml_module, "NML_Component_Type")
        .def_readonly("name", &NML_Component_Type::name)
        .def_property_readonly("extends", [](py::object self) {
            return reference_or_none(self.cast<const NML_Component_Type &>().extends, self);
        })
        .def_readonly("source_file", &NML_Component_Type::source_file)
        .def_property_readonly("source_node", [](py::object self) {
            return reference_or_none(self.cast<const NML_Component_Type &>().source_node, self);
        }, "The <ComponentType> element in the document tree.")
        .def_property_readonly("dynamics", [](py::object self) {
            return references_in(self.cast<const NML_Component_Type &>().dynamics, self);
        })
        .def_property_readonly("structure", [](py::object self) {
            return references_in(self.cast<const NML_Component_Type &>().structure, self);
        }, "The nearest <Structure> in the extends chain, as NML_Structure_Entry items.")
        .def_readonly("state_variable_names", &NML_Component_Type::state_variable_names,
                      "Per-neuron (or, for a synapse, per-edge) state slots in order, inherited ones first.")
        .def("find_declaration", [](py::object self, const String &namespace_key) {
            return reference_or_none(self.cast<const NML_Component_Type &>().find_declaration(namespace_key), self);
        }, py::arg("namespace_key"),
           "The declaration this type resolves the key to, searching up the extends chain, or None.")
        .def("__repr__", [](const NML_Component_Type &self) { return "<NML_Component_Type " + self.name + ">"; });

    py::class_<NML_Component_Instance>(nml_module, "NML_Component_Instance")
        .def_readonly("id", &NML_Component_Instance::id)
        .def_readonly("instance_data", &NML_Component_Instance::instance_data,
                      "Declared attributes and <Fixed> values, verbatim (units unresolved).")
        .def_readonly("data_order", &NML_Component_Instance::data_order, "Child instance ids, in source order.")
        .def_property_readonly("parent_instance", [](py::object self) {
            return reference_or_none(self.cast<const NML_Component_Instance &>().parent_instance, self);
        })
        .def_property_readonly("component_type", [](py::object self) {
            return reference_or_none(self.cast<const NML_Component_Instance &>().component_type, self);
        })
        .def("has_value", &NML_Component_Instance::has_value, py::arg("key"))
        .def("value_or", &NML_Component_Instance::value_or, py::arg("key"), py::arg("fallback") = String(""))
        .def("__repr__", [](const NML_Component_Instance &self) {
            return "<NML_Component_Instance " + self.id + " : " +
                   (self.component_type ? self.component_type->name : String("?")) + ">";
        });

    // ── nml: stimulus and connectivity ────────────────────────────────────────────
    py::class_<Input_Target>(nml_module, "Input_Target")
        .def_readonly("neuron_index", &Input_Target::neuron_index)
        .def_readonly("weight", &Input_Target::weight, "From <inputW weight=...>; 1.0 when unweighted.")
        .def_readonly("destination", &Input_Target::destination,
                      "The target cell's attachment the input goes to; '' means 'synapses'.");

    py::class_<Simulation_Input_Config>(nml_module, "Simulation_Input_Config")
        .def_readonly("input_component_id", &Simulation_Input_Config::input_component_id)
        .def_property_readonly("targets", [](py::object self) {
            return references_in(self.cast<const Simulation_Input_Config &>().targets, self);
        });

    py::class_<NML_Network_Edge>(nml_module, "NML_Network_Edge")
        .def_readonly("component_id", &NML_Network_Edge::component_id, "The synapse this edge carries.")
        .def_readonly("weight", &NML_Network_Edge::weight)
        .def_readonly("delay_ticks", &NML_Network_Edge::delay_ticks)
        .def_readonly("parent", &NML_Network_Edge::parent, "Source neuron.")
        .def_readonly("child", &NML_Network_Edge::child, "Target neuron.")
        .def_readonly("destination", &NML_Network_Edge::destination,
                      "The target's attachment the synapse goes to; '' means 'synapses'.")
        .def("__repr__", [](const NML_Network_Edge &self) {
            return "<NML_Network_Edge " + std::to_string(self.parent) + " -> " + std::to_string(self.child) +
                   " " + self.component_id + ">";
        });

    py::class_<Adjacency_List>(nml_module, "Adjacency_List", "The network's edges, one row per source neuron.")
        .def_property_readonly("list", [](py::object self) {
            py::list rows;
            for (const Vector<NML_Network_Edge> &row : self.cast<const Adjacency_List &>().list) {
                rows.append(references_in(row, self));
            }
            return rows;
        })
        .def("in_network", &Adjacency_List::in_network, py::arg("node_index"))
        .def("is_parent", &Adjacency_List::is_parent, py::arg("parent"), py::arg("prospective_child"))
        .def("edge_arrays", [](const Adjacency_List &self) {
            // Parallel arrays: one Python object per edge does not scale.
            Vector<s64> parent;
            Vector<s64> child;
            Vector<f32> weight;
            Vector<s64> delay_ticks;
            Vector<String> component_id;
            for (const Vector<NML_Network_Edge> &row : self.list) {
                for (const NML_Network_Edge &edge : row) {
                    parent.push_back(edge.parent);
                    child.push_back(edge.child);
                    weight.push_back(edge.weight);
                    delay_ticks.push_back(edge.delay_ticks);
                    component_id.push_back(edge.component_id);
                }
            }
            py::dict result;
            result["parent"] = to_numpy(parent.data(), static_cast<s64>(parent.size()));
            result["child"] = to_numpy(child.data(), static_cast<s64>(child.size()));
            result["weight"] = to_numpy(weight.data(), static_cast<s64>(weight.size()));
            result["delay_ticks"] = to_numpy(delay_ticks.data(), static_cast<s64>(delay_ticks.size()));
            result["component_id"] = component_id;
            return result;
        }, "Every edge as parallel arrays: parent, child, weight, delay_ticks, component_id.")
        .def("__len__", [](const Adjacency_List &self) { return self.list.size(); });

    // ── nml: the context ──────────────────────────────────────────────────────────
    py::class_<NML_Context::NML_Simulation_Context>(nml_module, "NML_Simulation_Context")
        .def_readonly("simulation_component_id", &NML_Context::NML_Simulation_Context::simulation_component_id)
        .def_readonly("target_network_id", &NML_Context::NML_Simulation_Context::target_network_id)
        .def_readonly("step_dt", &NML_Context::NML_Simulation_Context::step_dt, "Seconds per tick.")
        .def_readonly("simulation_duration", &NML_Context::NML_Simulation_Context::simulation_duration, "Seconds.")
        .def_readonly("total_tick_count", &NML_Context::NML_Simulation_Context::total_tick_count)
        .def_readonly("total_neuron_count", &NML_Context::NML_Simulation_Context::total_neuron_count)
        .def_readonly("random_seed", &NML_Context::NML_Simulation_Context::random_seed)
        .def_readonly("total_edge_count", &NML_Context::NML_Simulation_Context::total_edge_count)
        .def_readonly("maximum_edge_delay", &NML_Context::NML_Simulation_Context::maximum_edge_delay, "Ticks.")
        .def_readonly("population_base_indices", &NML_Context::NML_Simulation_Context::population_base_indices,
                      "Population id -> where its cells start in cell_state.")
        .def_readonly("instance_positions", &NML_Context::NML_Simulation_Context::instance_positions,
                      "A populationList <instance>'s full id -> its position among the list's instances.")
        .def_property_readonly("global_constants", [](const NML_Context::NML_Simulation_Context &self) {
            Unordered_Map<String, f64> constants;
            for (const auto &[name, value] : self.global_constants) constants[name] = value.float64;
            return constants;
        }, "Name -> SI value.")
        .def_property_readonly("cell_instances", [](py::object self) {
            return references_in(self.cast<const NML_Context::NML_Simulation_Context &>().cell_instances, self);
        })
        .def_property_readonly("synapse_instances", [](py::object self) {
            return references_in(self.cast<const NML_Context::NML_Simulation_Context &>().synapse_instances, self);
        }, "In synapse prototype order: an edge's prototype index is its position here.")
        .def_property_readonly("network_data", [](py::object self) {
            return reference_or_none(&self.cast<const NML_Context::NML_Simulation_Context &>().network_data, self);
        })
        .def_property_readonly("input_profiles", [](py::object self) {
            return references_in(self.cast<const NML_Context::NML_Simulation_Context &>().input_profiles, self);
        })
        .def_property_readonly("recording_profiles", [](py::object self) {
            return references_in(self.cast<const NML_Context::NML_Simulation_Context &>().recording_profiles, self);
        });

    py::class_<NML_Context>(nml_module, "NML_Context",
            "A parsed NeuroML/LEMS model. parse() one to inspect a model without a GPU, or read "
            "Spike_Engine.context. Don't parse() or reset() an engine's context: it is in use.")
        .def(py::init<>())
        .def("parse", &NML_Context::parse, py::arg("main_filepath"),
             "Parses the whole standard library, then the document and everything it includes.")
        .def("reset", &NML_Context::reset)
        .def("validate_lems_schema", &NML_Context::validate_lems_schema, py::arg("lems_filepath"),
             "XSD-validates a NeuroML document; any other root passes unchecked. Errors go to "
             "last_schema_validation_errors.")
        .def_property_readonly("simulation", [](py::object self) {
            return reference_or_none(&self.cast<const NML_Context &>().simulation, self);
        })
        .def_property_readonly("model_units", [](py::object self) {
            return references_by_key(self.cast<const NML_Context &>().model_units, self);
        })
        .def_property_readonly("model_constants", [](const NML_Context &self) {
            Unordered_Map<String, f64> constants;
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
             "'-60mV' -> -0.06, using the document's own <Unit> declarations as well as the built-in ones. "
             "Raises on text that is not a number with an optional unit, and on an unknown unit.")
        .def("resolve_quantity_of_dimension", &NML_Context::resolve_quantity_of_dimension,
             py::arg("value"), py::arg("dimension"),
             "resolve_quantity, also raising when the value's unit is not of `dimension`, or it has no "
             "unit and `dimension` is not 'none'. '' or '*' checks nothing.")
        .def("get_cell_variable_count", &NML_Context::get_cell_variable_count, py::arg("cell_instance_index"),
             "The state variables a cell declares; cell_slot_count gives the slots it takes.")
        .def_static("cell_slot_count", &NML_Context::cell_slot_count, py::arg("cell_type"),
                    "Slots one cell takes in cell_state: one per state variable, and one for a cell with none.")
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
    nml_module.def("population_cell", [](py::object context, const NML_Component_Instance &population) {
        return reference_or_none(&population_cell(context.cast<const NML_Context &>(), population), context);
    }, py::arg("context"), py::arg("population"), "The cell a population is made of.");
    nml_module.def("spike_history_length", &spike_history_length, py::arg("context"),
                   "Rows in the engine's spike_history ring.");
    nml_module.def("output_format_for_filename", &output_format_for_filename, py::arg("filename"));
    nml_module.def("evaluate_lems",
                   py::overload_cast<const String &, const Unordered_Map<String, f64> &, const String &, Random_Generator *>(
                           &evaluate_lems),
                   py::arg("expression"), py::arg("values"), py::arg("owner_name") = String("python"),
                   py::arg("random_generator") = nullptr,
                   "A LEMS expression's value on the host; every name it reads must be in values. random() "
                   "draws from random_generator.");

    py::class_<Random_Generator>(nml_module, "Random_Generator")
        .def(py::init<u64>(), py::arg("seed") = 0)
        .def("uniform", &Random_Generator::uniform, "Uniform on (0, 1), never 0 or 1 even as a float; LEMS random(1).")
        .def("normal", &Random_Generator::normal, py::arg("mean"), py::arg("standard_deviation"))
        .def("exponential", &Random_Generator::exponential, py::arg("rate"))
        .def("poisson", &Random_Generator::poisson, py::arg("mean"));

    py::class_<Lems_Expression_Token> lems_token(nml_module, "Lems_Expression_Token");
    py::enum_<Lems_Expression_Token::Kind>(lems_token, "Kind")
        .value("Number", Lems_Expression_Token::Kind::Number)
        .value("Identifier", Lems_Expression_Token::Kind::Identifier)
        .value("Operator", Lems_Expression_Token::Kind::Operator)
        .value("OpenParen", Lems_Expression_Token::Kind::OpenParen)
        .value("CloseParen", Lems_Expression_Token::Kind::CloseParen)
        .value("Comma", Lems_Expression_Token::Kind::Comma)
        .value("End", Lems_Expression_Token::Kind::End);
    lems_token
        .def_readonly("lexeme", &Lems_Expression_Token::lexeme)
        .def_readonly("kind", &Lems_Expression_Token::kind)
        .def("__repr__", [](const Lems_Expression_Token &self) { return "<Lems_Expression_Token " + self.lexeme + ">"; });
    nml_module.def("tokenize_lems", &tokenize_lems, py::arg("expression"), py::arg("owner_name") = String("python"));

    // ── k^2-tree ──────────────────────────────────────────────────────────────────
    // Owned by the weight matrix and reached through it.
    py::class_<K2_Tree>(m, "K2_Tree", "The compressed adjacency: which (source, target) pairs are edges.")
        .def_readonly_static("MAX_SUPERBLOCK_SIZE_WORDS", &K2_Tree::MAX_SUPERBLOCK_SIZE_WORDS)
        .def_readonly("branching_factor", &K2_Tree::branching_factor)
        .def_readonly("superblock_size_words", &K2_Tree::superblock_size_words)
        .def_readonly("node_count", &K2_Tree::node_count)
        .def_readonly("padded_node_count", &K2_Tree::padded_node_count)
        .def_readonly("tree_height", &K2_Tree::tree_height)
        .def_readonly("internal_bit_count", &K2_Tree::internal_bit_count)
        .def_readonly("internal_node_words_length", &K2_Tree::internal_node_words_length)
        .def_readonly("leaf_node_words_length", &K2_Tree::leaf_node_words_length)
        .def_readonly("rank_superblock_length", &K2_Tree::rank_superblock_length)
        .def_readonly("rank_subblock_length", &K2_Tree::rank_subblock_length)
        .def("adjacent", &K2_Tree::adjacent, py::arg("source_node"), py::arg("target_node"),
             "1 if source_node -> target_node is an edge, otherwise 0.")
        .def("get_neighbors", [](const K2_Tree &self, u32 node_index, s64 max_neighbor_count) {
            Vector<u32> buffer(static_cast<usize>(std::max<s64>(max_neighbor_count, 1)));
            const s64 found = self.get_neighbors(node_index, buffer.data(), max_neighbor_count);
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"), py::arg("max_neighbor_count"))
        .def("get_predecessors", [](const K2_Tree &self, u32 node_index, s64 max_neighbor_count) {
            Vector<u32> buffer(static_cast<usize>(std::max<s64>(max_neighbor_count, 1)));
            const s64 found = self.get_predecessors(node_index, buffer.data(), max_neighbor_count);
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"), py::arg("max_neighbor_count"))
        .def("trace", &K2_Tree::trace, py::arg("source_node"), py::arg("target_node"),
             "Logs the tree walk for one pair and returns adjacent()'s answer.")
        .def("save", [](const K2_Tree &self, const String &path) { self.save(path.c_str()); }, py::arg("path"));

    // ── weight matrix ─────────────────────────────────────────────────────────────
    py::class_<Weight_Stats>(m, "Weight_Stats")
        .def_readonly("mean", &Weight_Stats::mean)
        .def_readonly("standard_deviation", &Weight_Stats::standard_deviation)
        .def_readonly("root_mean_square", &Weight_Stats::root_mean_square)
        .def_readonly("min_value", &Weight_Stats::min_value)
        .def_readonly("max_value", &Weight_Stats::max_value)
        .def("__repr__", [](const Weight_Stats &self) {
            return "<Weight_Stats mean=" + std::to_string(self.mean) +
                   " rms=" + std::to_string(self.root_mean_square) +
                   " min=" + std::to_string(self.min_value) +
                   " max=" + std::to_string(self.max_value) + ">";
        });

    py::class_<Scale_Result>(m, "Scale_Result")
        .def_readonly("target_root_mean_square", &Scale_Result::target_root_mean_square)
        .def_readonly("scale_factor", &Scale_Result::scale_factor)
        .def_readonly("before", &Scale_Result::before)
        .def_readonly("after", &Scale_Result::after);

    // Bound without a constructor: building one takes the engine's GPU backend.
    py::class_<Weight_Matrix> weight_matrix_class(m, "Weight_Matrix",
            "Every per-edge value (weight, delay, synapse state), one plane each: M_k = "
            "U diag(Ck) V^T over the k^2-tree's edges, plus S, the updates since the last fit.");

    py::enum_<Weight_Matrix::Fit_Method>(weight_matrix_class, "Fit_Method",
            "The two candidate fits. Both fit one plane's own lanes in f32 to the same weighted, "
            "ridge-regularized squared error.")
        .value("LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO",
               Weight_Matrix::Fit_Method::LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO)
        .value("MASKED_ALTERNATING_LEAST_SQUARES", Weight_Matrix::Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES);

    weight_matrix_class
        // Plane numbers and fitting constants.
        .def_readonly_static("WEIGHT_PLANE", &Weight_Matrix::WEIGHT_PLANE)
        .def_readonly_static("DELAY_PLANE", &Weight_Matrix::DELAY_PLANE)
        .def_readonly_static("FIRST_STATE_VARIABLE_PLANE", &Weight_Matrix::FIRST_STATE_VARIABLE_PLANE)
        .def_readonly_static("LANE_GROUP", &Weight_Matrix::LANE_GROUP)
        .def_readonly_static("DEFAULT_FIT_TOLERANCE", &Weight_Matrix::DEFAULT_FIT_TOLERANCE)
        .def_readonly_static("DEFAULT_FIT_RIDGE", &Weight_Matrix::DEFAULT_FIT_RIDGE)
        .def_readonly_static("FIT_STALL_IMPROVEMENT", &Weight_Matrix::FIT_STALL_IMPROVEMENT)
        .def_readonly_static("MASKED_ALTERNATING_LEAST_SQUARES_STALL_WINDOW_SWEEP_COUNT",
                             &Weight_Matrix::MASKED_ALTERNATING_LEAST_SQUARES_STALL_WINDOW_SWEEP_COUNT)
        .def_readonly_static("MASKED_ALTERNATING_LEAST_SQUARES_MAXIMUM_SWEEP_COUNT",
                             &Weight_Matrix::MASKED_ALTERNATING_LEAST_SQUARES_MAXIMUM_SWEEP_COUNT)
        .def_readonly_static("LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_STALL_WINDOW_ITERATION_COUNT",
                             &Weight_Matrix::LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_STALL_WINDOW_ITERATION_COUNT)
        .def_readonly_static("LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_MAXIMUM_ITERATION_COUNT",
                             &Weight_Matrix::LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_MAXIMUM_ITERATION_COUNT)
        .def_readonly_static("LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_HISTORY_LENGTH",
                             &Weight_Matrix::LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_HISTORY_LENGTH)
        .def_readonly_static("WOLFE_SUFFICIENT_DECREASE", &Weight_Matrix::WOLFE_SUFFICIENT_DECREASE)
        .def_readonly_static("WOLFE_CURVATURE", &Weight_Matrix::WOLFE_CURVATURE)
        .def_readonly_static("LINE_SEARCH_MAXIMUM_EVALUATION_COUNT", &Weight_Matrix::LINE_SEARCH_MAXIMUM_EVALUATION_COUNT)
        .def_readonly_static("DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION", &Weight_Matrix::DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION)
        .def_readonly_static("DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION",
                             &Weight_Matrix::DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION)

        // Shape.
        .def_readonly("node_count", &Weight_Matrix::node_count)
        .def_readonly("total_edge_count", &Weight_Matrix::total_edge_count)
        .def_readonly("max_neighbor_count", &Weight_Matrix::max_neighbor_count)
        .def_readonly("max_predecessor_count", &Weight_Matrix::max_predecessor_count)
        .def_readonly("matrix_count", &Weight_Matrix::matrix_count,
                      "Planes: weight, delay, then the largest synapse type's state variables.")
        .def_readonly("updated_plane_count", &Weight_Matrix::updated_plane_count,
                      "How many planes the generated kernel updates each tick.")
        .def_readonly("rank", &Weight_Matrix::rank, "Lanes in the basis.")
        .def_readonly("rank_float4_stride", &Weight_Matrix::rank_float4_stride)
        .def_readwrite("check_indexing", &Weight_Matrix::check_indexing)
        .def_property_readonly("k2tree", [](Weight_Matrix &self) -> K2_Tree & { return self.k2tree; })

        // Fitting.
        .def_readwrite("fit_tolerance", &Weight_Matrix::fit_tolerance,
                       "The worst relative error a fit may leave on any plane. The engine sets it "
                       "from Spike_Engine.fit_tolerance before each refit.")
        .def_readwrite("fit_method", &Weight_Matrix::fit_method,
                       "The candidate fit refits use. The engine sets it from Spike_Engine.fit_method "
                       "before each refit.")
        .def_readwrite("lane_search_fit_method", &Weight_Matrix::lane_search_fit_method,
                       "The candidate fit the lane searches use (declare_runs and calibrate_lane_budget), "
                       "which set each plane's lanes.")
        .def_readonly("measured_fit_error", &Weight_Matrix::measured_fit_error,
                      "Per plane: the worst relative error the last fit left.")
        .def_readwrite("expected_active_edge_fraction", &Weight_Matrix::expected_active_edge_fraction,
                       "The expected fraction of edges whose values change between refits (0 to 1), which "
                       "sizes every plane's fixed lane budget when the runs are declared.")
        .def("worst_fit_error", &Weight_Matrix::worst_fit_error, "The worst of measured_fit_error.")

        // The basis itself, copied out.
        .def_property_readonly("U_matrix", [](const Weight_Matrix &self) {
            return device_buffer_to_numpy_rows<f32>(self.U_matrix, self.node_count, self.rank_float4_stride * Weight_Matrix::LANE_GROUP);
        }, "[node_count][rank] source factors.")
        .def_property_readonly("V_matrix", [](const Weight_Matrix &self) {
            return device_buffer_to_numpy_rows<f32>(self.V_matrix, self.node_count, self.rank_float4_stride * Weight_Matrix::LANE_GROUP);
        }, "[node_count][rank] target factors.")
        .def_property_readonly("coefficients", [](const Weight_Matrix &self) {
            return device_buffer_to_numpy_rows<f32>(self.coefficients, self.matrix_count, self.rank_float4_stride * Weight_Matrix::LANE_GROUP);
        }, "[matrix_count][rank]: one coefficient row Ck per plane.")
        .def_property_readonly("edge_row_offset", [](const Weight_Matrix &self) {
            return device_buffer_to_numpy<s64>(self.edge_row_offset, self.node_count + 1);
        }, "Out-degree prefix sum: node n's edges have ordinals edge_row_offset[n] up to [n + 1].")

        // Runs.
        .def_property_readonly("run_first_edge_ordinal", [](const Weight_Matrix &self) {
            return to_numpy(self.run_first_edge_ordinal.data(), static_cast<s64>(self.run_first_edge_ordinal.size()));
        })
        .def_property_readonly("run_edge_count", [](const Weight_Matrix &self) {
            return to_numpy(self.run_edge_count.data(), static_cast<s64>(self.run_edge_count.size()));
        })
        .def_property_readonly("run_synapse_prototype", [](const Weight_Matrix &self) {
            return to_numpy(self.run_synapse_prototype.data(), static_cast<s64>(self.run_synapse_prototype.size()));
        })

        // S, the sparse update buffer, and its sizing.
        .def_readonly("sparse_delta_capacity", &Weight_Matrix::sparse_delta_capacity,
                      "Update slots in S's pool, shared by every plane.")
        .def_readonly("sparse_delta_entry_count", &Weight_Matrix::sparse_delta_entry_count,
                      "Updates each plane holds.")
        .def_readwrite("plasticity_reserve_entries", &Weight_Matrix::plasticity_reserve_entries,
                       "Extra slots in S and the queue for plasticity; applies at the next resize.")
        .def_readonly("pending_delta_capacity", &Weight_Matrix::pending_delta_capacity,
                      "Room in the device's per-tick update queue; fixed when the kernel is built.")
        .def_readonly("updated_plane_count", &Weight_Matrix::updated_plane_count,
                      "Planes the kernel updates each tick, as codegen reports.")
        .def("sparse_delta_occupancy_fraction", &Weight_Matrix::sparse_delta_occupancy_fraction,
             "Fraction of the edge set holding an update, in the fullest plane.")
        .def("delta_capacity_for_threshold", &Weight_Matrix::delta_capacity_for_threshold,
             py::arg("occupancy_threshold_fraction"),
             "S's pool capacity a refit threshold asks for: that fraction of the edges (every edge when it "
             "is 0) for each plane the kernel updates, plus the plasticity reserve.")
        .def("resize_delta_capacity", &Weight_Matrix::resize_delta_capacity, py::arg("new_capacity"),
             "Resizes S, keeping its updates; it never shrinks below them.")
        .def("resize_pending_delta_capacity", &Weight_Matrix::resize_pending_delta_capacity, py::arg("new_capacity"),
             "Resizes the device's update queue, emptying it. The engine's kernel bakes the size in, so "
             "only a matrix outside an engine can use this.")
        .def("pending_capacity_for_threshold", &Weight_Matrix::pending_capacity_for_threshold,
             py::arg("occupancy_threshold_fraction"),
             "The queue size for a refit threshold: like S, and at least one neuron's walk.")
        .def("sparse_deltas", [](const Weight_Matrix &self, s64 matrix_index) {
            if (matrix_index < 0 || matrix_index >= self.matrix_count) {
                throw Invalid_Argument("sparse_deltas: no plane " + std::to_string(matrix_index));
            }
            const s64 entry_count = self.sparse_delta_entry_count[static_cast<usize>(matrix_index)];
            if (entry_count == 0 || self.sparse_delta_value.is_empty()) {
                return py::make_tuple(to_numpy<u32>(nullptr, 0), to_numpy<f32>(nullptr, 0));
            }
            const s64 plane_start = self.sparse_delta_row_start.get_contents_as<s32>()[matrix_index * (self.node_count + 1)];
            return py::make_tuple(to_numpy(self.sparse_delta_edge_ordinal.get_contents_as<u32>() + plane_start, entry_count),
                                  to_numpy(self.sparse_delta_value.get_contents_as<f32>() + plane_start, entry_count));
        }, py::arg("matrix_index"), "(edge_ordinals, deltas): the updates one plane of S holds, by ordinal.")
        .def_property_readonly("sparse_delta_row_start", [](const Weight_Matrix &self) {
            return device_buffer_to_numpy_rows<s32>(self.sparse_delta_row_start, self.matrix_count, self.node_count + 1);
        }, "S's CSR row starts as pool positions, [matrix_count][node_count + 1].")
        .def_property_readonly("sparse_delta_edge_ordinal", [](const Weight_Matrix &self) {
            return device_buffer_to_numpy<u32>(self.sparse_delta_edge_ordinal, self.sparse_delta_capacity);
        }, "S's pool of edge ordinals, [sparse_delta_capacity]: every plane's entries, plane after plane; "
           "sparse_delta_row_start gives each plane's and row's positions.")
        .def_property_readonly("sparse_delta_value", [](const Weight_Matrix &self) {
            return device_buffer_to_numpy<f32>(self.sparse_delta_value, self.sparse_delta_capacity);
        }, "S's pool of deltas, laid out like sparse_delta_edge_ordinal.")
        .def_property_readonly("pending_delta_count", &Weight_Matrix::pending_delta_count,
                               "Updates the device queued since the last compact_pending_deltas.")
        .def("pending_deltas", [](const Weight_Matrix &self) {
            s64 staged = 0;
            staged = std::min<s64>(self.pending_delta_count(), self.pending_delta_capacity);
            return py::make_tuple(device_buffer_to_numpy<u32>(self.pending_delta_edge_ordinal, staged),
                                  device_buffer_to_numpy<f32>(self.pending_delta_value, staged),
                                  device_buffer_to_numpy<u8>(self.pending_delta_matrix_index, staged));
        }, "(edge_ordinals, deltas, matrix_indices): what the device queued and has not been merged.")

        // Reads.
        .def("get", &Weight_Matrix::get, py::arg("source_node"), py::arg("target_node"),
             "The weight plane at one edge: the basis plus its update in S; 0.0 for a pair that is not an edge.")
        .def("get_for_matrix", &Weight_Matrix::get_for_matrix,
             py::arg("source_node"), py::arg("target_node"), py::arg("matrix_index"),
             "Any plane at one edge: the basis plus its update in S; 0.0 for a pair that is not an edge.")
        .def("get_edge_delay_ticks", &Weight_Matrix::get_edge_delay_ticks,
             py::arg("source_node"), py::arg("target_node"),
             "The delay plane rounded to whole ticks, at least one: how the kernel uses it; 0 for a pair that is not an edge.")
        .def("get_edge_synapse_prototype", &Weight_Matrix::get_edge_synapse_prototype,
             py::arg("source_node"), py::arg("target_node"),
             "Index into the context's synapse_instances of the synapse this edge carries.")
        .def("edge_ordinal", &Weight_Matrix::edge_ordinal, py::arg("source_node"), py::arg("target_node"),
             "The edge's ordinal, or None when the pair is not an edge.")
        .def("check_index_inbounds", py::overload_cast<s32, s32>(&Weight_Matrix::check_index_inbounds, py::const_),
             py::arg("source"), py::arg("target"))
        .def("check_index_inbounds", py::overload_cast<s32>(&Weight_Matrix::check_index_inbounds, py::const_),
             py::arg("node_index"))
        .def("get_neighbors", [](const Weight_Matrix &self, s64 node_index) {
            Vector<u32> buffer(static_cast<usize>(std::max<s64>(self.max_neighbor_count, 1)));
            const s64 found = self.get_neighbors(node_index, buffer.data());
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"), "Targets of node_index's edges, in ordinal order.")
        .def("get_predecessors", [](const Weight_Matrix &self, s64 node_index) {
            Vector<u32> buffer(static_cast<usize>(std::max<s64>(self.max_predecessor_count, 1)));
            const s64 found = self.get_predecessors(node_index, buffer.data());
            return to_numpy(buffer.data(), found);
        }, py::arg("node_index"), "Sources of the edges into node_index.")
        .def("neighbor_weights", [](const Weight_Matrix &self) {
            Vector<f32> values(static_cast<usize>(std::max<s64>(self.total_edge_count, 0)));
            if (!values.empty()) self.neighbor_weights(values.data());
            return to_numpy(values.data(), self.total_edge_count);
        }, "Every edge's weight, by ordinal.")
        .def("neighbor_weights_for_matrix", [](const Weight_Matrix &self, s64 matrix_index) {
            Vector<f32> values(static_cast<usize>(std::max<s64>(self.total_edge_count, 0)));
            if (!values.empty()) self.neighbor_weights_for_matrix(values.data(), matrix_index);
            return to_numpy(values.data(), self.total_edge_count);
        }, py::arg("matrix_index"), "Every edge's value in one plane, by ordinal.")
        .def("neighbor_weight_stats", &Weight_Matrix::neighbor_weight_stats)

        // Updates and refits.
        .def("accumulate_edge_delta", &Weight_Matrix::accumulate_edge_delta,
             py::arg("matrix_index"), py::arg("source_node"), py::arg("target_node"), py::arg("delta"),
             "Adds delta to one edge's value in one plane, through S. Reads see it at once; "
             "the next refit folds it into the basis. S grows if the plane is full.")
        .def("compact_pending_deltas", &Weight_Matrix::compact_pending_deltas,
             "Merges what the device queued into S, growing S if a plane would overflow.")
        .def("planes_awaiting_lane_budget", &Weight_Matrix::planes_awaiting_lane_budget,
             "Planes that hold no lanes yet: those that started at zero.")
        .def("plane_updates", [](const Weight_Matrix &self, s64 matrix_index) {
            const Vector<Pair<s64, f32>> updates = self.plane_updates(matrix_index);
            Vector<s64> ordinals;
            Vector<f32> deltas;
            for (const Pair<s64, f32> &update : updates) {
                ordinals.push_back(update.first);
                deltas.push_back(update.second);
            }
            return py::make_tuple(to_numpy(ordinals.data(), static_cast<s64>(ordinals.size())),
                                  to_numpy(deltas.data(), static_cast<s64>(deltas.size())));
        }, py::arg("matrix_index"), "(edge_ordinals, deltas): one plane's updates in S, by ordinal.")
        .def("plane_update_sum_of_squares", &Weight_Matrix::plane_update_sum_of_squares, py::arg("matrix_index"),
             "The sum of squares of one plane's updates in S.")
        .def("calibrate_lane_budget", [](Weight_Matrix &self, s64 matrix_index,
                                         py::array_t<f32, py::array::c_style | py::array::forcecast> calibration_values) {
            const Vector<f32> values(calibration_values.data(), calibration_values.data() + calibration_values.size());
            self.calibrate_lane_budget(matrix_index, values);
        }, py::arg("matrix_index"), py::arg("calibration_values"),
             "Sets the lane budget of a plane with no lanes from values it is expected to hold (one per "
             "edge), searched with lane_search_fit_method; the plane keeps the fitted V rows and reads "
             "zero until its first refit.")
        .def("discard_updates", &Weight_Matrix::discard_updates, py::arg("occupancy_threshold_fraction"),
             "Empties S and the queue, dropping every update, and sizes S for the threshold.")
        .def("refit", &Weight_Matrix::refit, py::arg("ridge_regularization") = Weight_Matrix::DEFAULT_FIT_RIDGE,
             "Refits every plane with updates in S to the current values (basis plus S) with "
             "fit_method, each on lanes of its own at their current count, moving only the rows at "
             "either end of an updated edge; a plane with no lanes yet gets its count once, by the "
             "construction search. Then empties S.")
        .def("is_refit_due", &Weight_Matrix::is_refit_due, py::arg("occupancy_threshold_fraction"),
             "True once the fullest plane of S holds updates on that fraction of the edge set.")
        .def("scale_neighbor_weights_to_root_mean_square", &Weight_Matrix::scale_neighbor_weights_to_root_mean_square,
             py::arg("target_root_mean_square"), py::arg("epsilon") = 1e-12f,
             "Scales the weight plane (its coefficient row and its updates) to a target RMS.")

        .def("save", [](const Weight_Matrix &self, const String &path) {
            self.save(path.c_str());
        }, py::arg("path"))
        .def("load_from_disk", [](Weight_Matrix &self, const String &path) {
            self.load_from_disk(path.c_str());
        }, py::arg("path"))

        .def("__repr__", [](const Weight_Matrix &self) {
            return "<Weight_Matrix nodes=" + std::to_string(self.node_count) +
                   " edges=" + std::to_string(self.total_edge_count) +
                   " planes=" + std::to_string(self.matrix_count) +
                   " rank=" + std::to_string(self.rank) + ">";
        });

    // ── the engine ────────────────────────────────────────────────────────────────
    py::class_<Spike_Engine> engine_class(m, "Spike_Engine");

    engine_class
        .def(py::init<const String &, bool, f32, Weight_Matrix::Fit_Method, f32>(),
             py::arg("lems_input_file"),
             py::arg("enable_hebbian_plasticity") = false,
             py::arg("fit_tolerance") = Spike_Engine::FIT_TOLERANCE_STANDARD,
             py::arg("fit_method") = Weight_Matrix::Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES,
             py::arg("expected_active_edge_fraction") = Weight_Matrix::DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION,
             "Parses a LEMS document, fills every model buffer, builds the weight matrix from its "
             "connections, and compiles the tick kernel.")
        .def(py::init<const String &, const Vector<Vector<s32>> &, const String &,
                      f64, f64, bool, f32, Weight_Matrix::Fit_Method, f32>(),
             py::arg("lems_input_file"), py::arg("adjacency"), py::arg("synapse_component_id"),
             py::arg("connection_weight") = 1.0,
             py::arg("connection_delay_seconds") = 0.0,
             py::arg("enable_hebbian_plasticity") = false,
             py::arg("fit_tolerance") = Spike_Engine::FIT_TOLERANCE_STANDARD,
             py::arg("fit_method") = Weight_Matrix::Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES,
             py::arg("expected_active_edge_fraction") = Weight_Matrix::DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION,
             "The same, with adjacency replacing the document's connections: adjacency[source] "
             "lists source's targets, and every edge carries the named synapse, connection_weight "
             "and connection_delay_seconds.")
        .def(py::init<const String &, const Vector<Vector<s32>> &,
                      const Vector<String> &, const Vector<f64> &,
                      f64, f64, bool, f32, Weight_Matrix::Fit_Method, f32>(),
             py::arg("lems_input_file"), py::arg("adjacency"),
             py::arg("synapse_component_ids"),
             py::arg("synapse_proportions") = Vector<f64>{},
             py::arg("connection_weight") = 1.0,
             py::arg("connection_delay_seconds") = 0.0,
             py::arg("enable_hebbian_plasticity") = false,
             py::arg("fit_tolerance") = Spike_Engine::FIT_TOLERANCE_STANDARD,
             py::arg("fit_method") = Weight_Matrix::Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES,
             py::arg("expected_active_edge_fraction") = Weight_Matrix::DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION,
             "The same, with several synapses: the edges are split, in adjacency order, into one "
             "contiguous share per synapse, sized by synapse_proportions (normalised; empty means "
             "equal). weights.get_edge_synapse_prototype gives any edge's synapse.")

        // What the engine was built from.
        .def_property_readonly("context", [](Spike_Engine &self) -> NML_Context & { return self.context; },
                               "The parsed model. Read-only in effect: don't parse() or reset() it.")
        .def_property_readonly("weights", [](Spike_Engine &self) -> Weight_Matrix & { return self.weights; },
                               "Every per-edge value of the network, by reference.")
        .def_readonly("total_neuron_count", &Spike_Engine::total_neuron_count)
        .def_readonly("spike_history_row_count", &Spike_Engine::spike_history_row_count)
        .def_readonly("lifetime", &Spike_Engine::lifetime, "Ticks in the run.")
        .def_property_readonly("ticks_stepped", [](const Spike_Engine &self) {
            const Lock_Guard<Mutex> results_lock(self.results_mutex);
            return self.ticks_stepped;
        }, "Ticks stepped so far, which is also the next tick step_simulation accepts.")
        .def_readonly("step_dt", &Spike_Engine::step_dt, "Seconds per tick.")
        .def_readonly("simulation_seed", &Spike_Engine::simulation_seed)
        .def_property_readonly("random_generator", [](Spike_Engine &self) -> Random_Generator & { return self.random_generator; },
                               "Draws every random() the run calls, seeded with simulation_seed; by reference.")
        .def_readonly("random_values_count", &Spike_Engine::random_values_count,
                      "Slots in random_values: one per random() call per neuron that runs it.")
        .def_readonly("alive", &Spike_Engine::alive)
        .def_readonly("hebbian_plasticity_enabled", &Spike_Engine::hebbian_plasticity_enabled,
                      "No effect on weights yet: the generated kernel stages no plasticity updates.")

        // The generated kernel.
        .def_readonly("master_kernel_source", &Spike_Engine::master_kernel_source,
                      "The generated master kernel exactly as compiled.")
        .def_readonly("kernel_parameter_names", &Spike_Engine::kernel_parameter_names,
                      "master_step's bound parameters, in binding order.")
        .def_readonly("synapse_active_ticks", &Spike_Engine::synapse_active_ticks,
                      "Ticks after a neuron spikes during which its outgoing synapses run (longest "
                      "delay plus the slowest synapse's settle time); -1 when every synapse runs every tick.")
        .def_readonly("run_count", &Spike_Engine::run_count)
        .def_readonly("deferred_walk_pass_count", &Spike_Engine::deferred_walk_pass_count,
                      "Extra walk passes the run has needed because the update queue was full.")
        .def_readonly("calibration_tick_count", &Spike_Engine::calibration_tick_count,
                      "Ticks construction's calibration ran after a spike on every neuron to size the lane "
                      "budgets of the planes that start at zero; 0 when none needed it.")
        .def_readonly_static("CALIBRATION_TICK_LIMIT", &Spike_Engine::CALIBRATION_TICK_LIMIT)
        .def_readonly_static("CALIBRATION_SPIKE_SPREAD_DIVISOR", &Spike_Engine::CALIBRATION_SPIKE_SPREAD_DIVISOR)

        // Weight-matrix fit and refit settings.
        .def_readonly_static("FIT_TOLERANCE_PRECISE", &Spike_Engine::FIT_TOLERANCE_PRECISE)
        .def_readonly_static("FIT_TOLERANCE_ACCURATE", &Spike_Engine::FIT_TOLERANCE_ACCURATE)
        .def_readonly_static("FIT_TOLERANCE_STANDARD", &Spike_Engine::FIT_TOLERANCE_STANDARD)
        .def_readonly_static("FIT_TOLERANCE_COMPACT", &Spike_Engine::FIT_TOLERANCE_COMPACT)
        .def_readwrite("fit_tolerance", &Spike_Engine::fit_tolerance,
                       "Worst relative error a refit may leave on any per-edge value.")
        .def_readwrite("fit_method", &Spike_Engine::fit_method,
                       "The candidate fit every refit uses; also a constructor argument, for the construction fit.")
        .def_readonly("expected_active_edge_fraction", &Spike_Engine::expected_active_edge_fraction,
                      "The expected fraction of edges whose values change between refits, which sized the lane "
                      "budgets at construction; a constructor argument.")
        .def_readwrite("refit_every_n_ticks", &Spike_Engine::refit_every_n_ticks, "0 disables.")
        .def_property("refit_occupancy_threshold_fraction",
                      [](const Spike_Engine &self) { return self.refit_occupancy_threshold_fraction; },
                      &Spike_Engine::set_refit_occupancy_threshold_fraction,
                      "Refit once the fullest plane of S holds updates on this fraction of the edges; "
                      "0 disables. Also the size S is kept at per plane: setting it resizes S.")
        .def("set_refit_occupancy_threshold_fraction", &Spike_Engine::set_refit_occupancy_threshold_fraction,
             py::arg("fraction"))
        .def_readwrite("minimum_ticks_between_refits", &Spike_Engine::minimum_ticks_between_refits,
                       "Refits are at least this many ticks apart; S grows in between. 0 allows any spacing.")
        .def_readonly("last_refit_tick", &Spike_Engine::last_refit_tick, "The construction fit counts as tick 0.")

        // Plasticity settings.
        .def_readonly_static("DEFAULT_PLASTICITY_DELTA_CAPACITY", &Spike_Engine::DEFAULT_PLASTICITY_DELTA_CAPACITY)
        .def_readwrite("plasticity_fold_every_n_ticks", &Spike_Engine::plasticity_fold_every_n_ticks)
        .def_readwrite("plasticity_target_root_mean_square", &Spike_Engine::plasticity_target_root_mean_square,
                       "After each refit, with plasticity on, the weights are scaled back to this RMS; "
                       "negative disables.")

        // Inputs, run by the kernel from their own dynamics.
        .def_readonly("input_entry_count", &Spike_Engine::input_entry_count, "One entry per input per target cell.")
        .def_readonly("input_value_stride", &Spike_Engine::input_value_stride,
                      "Floats per entry in input_values: its weight, then the largest input's state.")
        .def_readonly("input_spike_count_size", &Spike_Engine::input_spike_count_size)
        .def_property_readonly("input_row_start", [](const Spike_Engine &self) {
            require_alive(self, "input_row_start");
            return device_buffer_to_numpy<s64>(self.input_row_start, self.total_neuron_count + 1);
        }, "[total_neuron_count + 1]; neuron n's entries are input_row_start[n] up to input_row_start[n + 1].")
        .def_property_readonly("input_entry_prototype", [](const Spike_Engine &self) {
            require_alive(self, "input_entry_prototype");
            return device_buffer_to_numpy<s32>(self.input_entry_prototype, self.input_entry_count);
        }, "Which input each entry runs; inputs are numbered in order of first use.")
        .def_property_readonly("input_values", [](const Spike_Engine &self) {
            require_alive(self, "input_values");
            return device_buffer_to_numpy_rows<f32>(self.input_values, self.input_entry_count, self.input_value_stride);
        }, "[input_entry_count][input_value_stride]: each entry's weight, then its parts' state.")
        .def_property_readonly("input_spike_counts", [](const Spike_Engine &self) {
            require_alive(self, "input_spike_counts");
            return device_buffer_to_numpy<u8>(self.input_spike_counts, self.input_spike_count_size);
        }, "Every spike train's spikes per tick, one train after another.")

        // Device state, copied out.
        .def_property_readonly("cell_state", [](const Spike_Engine &self) {
            require_alive(self, "cell_state");
            return device_buffer_to_numpy<f32>(self.cell_state, self.context.get_cell_state_size());
        }, "Every cell's state variables; context.resolve_path gives a variable's index.")
        .def_property_readonly("network_inputs", [](const Spike_Engine &self) {
            require_alive(self, "network_inputs");
            return device_buffer_to_numpy_rows<f32>(self.network_inputs, 2, self.total_neuron_count);
        }, "[2][total_neuron_count], alternating by tick parity.")
        .def_property_readonly("event_arrival_count", [](const Spike_Engine &self) {
            require_alive(self, "event_arrival_count");
            // Two rows only when some cell type has an OnEvent; otherwise the buffer is a placeholder.
            const u64 two_rows_bytes = sizeof(u32) * 2 * static_cast<u64>(self.total_neuron_count);
            const s64 row_count = self.event_arrival_count.total_bytes >= two_rows_bytes ? 2 : 0;
            return device_buffer_to_numpy_rows<u32>(self.event_arrival_count, row_count, self.total_neuron_count);
        }, "[2][total_neuron_count] spikes that reached each cell, alternating by tick parity; "
           "[0][total_neuron_count] when no cell type has an OnEvent.")
        .def_property_readonly("spike_history", [](const Spike_Engine &self) {
            require_alive(self, "spike_history");
            return device_buffer_to_numpy_rows<u8>(self.spike_history, self.spike_history_row_count,
                                                   self.total_neuron_count);
        }, "[spike_history_row_count][total_neuron_count]; row tick % rows is that tick's spikes.")
        .def_property_readonly("last_spiked", [](const Spike_Engine &self) {
            require_alive(self, "last_spiked");
            return device_buffer_to_numpy<s64>(self.last_spiked, self.total_neuron_count);
        }, "Each neuron's last spike tick, or NEVER_SPIKED_TICK.")
        .def_property_readonly("random_values", [](const Spike_Engine &self) {
            require_alive(self, "random_values");
            return device_buffer_to_numpy<f32>(self.random_values, self.random_values_count);
        }, "The last tick's uniform draws, one per random() call per neuron that runs it.")

        // Running.
        .def("run", [](Spike_Engine &self, const py::object &after_each_tick) {
            // The ticks run without the GIL. It is taken back between ticks only to call
            // after_each_tick and, every SIGNAL_CHECK_INTERVAL, to raise a pending Ctrl-C.
            const bool calls_back = !after_each_tick.is_none();
            Steady_Clock::time_point last_signal_check = Steady_Clock::now();
            const py::gil_scoped_release release;
            self.run([&](s64 tick) {
                const bool signal_check_due = Steady_Clock::now() - last_signal_check >= SIGNAL_CHECK_INTERVAL;
                if (!calls_back && !signal_check_due) return;

                const py::gil_scoped_acquire acquire;
                if (calls_back) after_each_tick(tick);
                if (signal_check_due) {
                    last_signal_check = Steady_Clock::now();
                    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
                }
            });
        }, py::arg("after_each_tick") = py::none(),
           "Runs every tick the model's Simulation declared. after_each_tick, when given, is called "
           "with each tick once it is done; an exception it raises, or Ctrl-C, stops the run.")
        .def("step_simulation", &Spike_Engine::step_simulation, py::arg("tick"),
             py::call_guard<py::gil_scoped_release>(),
             "One tick: random draws, the kernel, the merge into S, a refit if due, and recording.")

        // Reading state.
        .def("read_state_variable", [](const Spike_Engine &self, s64 neuron_index, const String &variable_name) {
            require_alive(self, "read_state_variable");
            return self.read_state_variable(neuron_index, variable_name);
        }, py::arg("neuron_index"), py::arg("variable_name"),
           "One neuron's named state variable, read back from the GPU.")
        .def("state_variable_array", [](const Spike_Engine &self, const String &variable_name) {
            require_alive(self, "state_variable_array");
            // Not a slice of cell_state: state is laid out per population, so one variable is
            // not contiguous across neurons.
            Vector<f32> values(static_cast<usize>(self.total_neuron_count));
            for (s64 index = 0; index < self.total_neuron_count; index += 1) {
                values[static_cast<usize>(index)] = self.read_state_variable(index, variable_name);
            }
            return to_numpy(values.data(), self.total_neuron_count);
        }, py::arg("variable_name"),
           "The variable for every neuron; throws naming any cell type that lacks it.")
        .def("synapse_state_variable_plane", &Spike_Engine::synapse_state_variable_plane,
             py::arg("synapse_component_id"), py::arg("variable_name"),
             "The weights plane holding one of a synapse's LEMS state variables on its edges.")

        // Results. A run on another thread appends to these, so each copy is taken under
        // results_mutex.
        .def_property_readonly("spike_counts_per_neuron", [](const Spike_Engine &self) {
            const Lock_Guard<Mutex> results_lock(self.results_mutex);
            return to_numpy(self.spike_counts_per_neuron.data(), static_cast<s64>(self.spike_counts_per_neuron.size()));
        }, "Spikes per neuron over the run so far.")
        .def_property_readonly("recorded_spikes", [](const Spike_Engine &self) {
            // Parallel arrays: a long run has millions of spikes, too many for one object each.
            Vector<f64> times;
            Vector<s64> neurons;
            {
                const Lock_Guard<Mutex> results_lock(self.results_mutex);
                times.resize(self.recorded_spikes.size());
                neurons.resize(self.recorded_spikes.size());
                for (usize index = 0; index < self.recorded_spikes.size(); index += 1) {
                    times[index] = self.recorded_spikes[index].time_seconds;
                    neurons[index] = self.recorded_spikes[index].neuron_index;
                }
            }
            return py::make_tuple(to_numpy(times.data(), static_cast<s64>(times.size())),
                                  to_numpy(neurons.data(), static_cast<s64>(neurons.size())));
        }, "(time_seconds, neuron_index), parallel arrays over every spike so far.")
        .def_property_readonly("traced_selections", [](py::object self) {
            return references_in(self.cast<const Spike_Engine &>().traced_selections, self);
        }, "The document's OutputColumns, one per column of recorded_traces.")
        .def_property_readonly("recorded_trace_times", [](const Spike_Engine &self) {
            const Lock_Guard<Mutex> results_lock(self.results_mutex);
            return to_numpy(self.recorded_trace_times.data(), static_cast<s64>(self.recorded_trace_times.size()));
        })
        .def_property_readonly("recorded_traces", [](const Spike_Engine &self) {
            const Lock_Guard<Mutex> results_lock(self.results_mutex);
            return to_numpy_rows(self.recorded_traces.data(), static_cast<s64>(self.recorded_trace_times.size()),
                                 static_cast<s64>(self.traced_selections.size()));
        }, "[recorded tick][traced selection].")
        .def("mean_firing_rate_hertz", &Spike_Engine::mean_firing_rate_hertz)
        .def("fraction_of_neurons_that_spiked", &Spike_Engine::fraction_of_neurons_that_spiked)

        // Writing results.
        .def("write_recordings", &Spike_Engine::write_recordings,
             "Closes the membrane video and writes every OutputFile and EventOutputFile the model declared.")
        .def("record_membrane_video", &Spike_Engine::record_membrane_video,
             py::arg("path"), py::arg("frame_stride") = 1,
             "Records every neuron's `v` to a .spire file every frame_stride ticks. Call before run().")
        .def_readonly("membrane_video_frame_stride", &Spike_Engine::membrane_video_frame_stride)
        .def("write_spike_file", &Spike_Engine::write_spike_file, py::arg("path"),
             "Every spike of the run as 'time<tab>neuron'.")

        .def("shutdown", &Spike_Engine::shutdown)
        .def("__repr__", [](const Spike_Engine &self) {
            return "<Spike_Engine neurons=" + std::to_string(self.total_neuron_count) +
                   " ticks=" + std::to_string(self.lifetime) +
                   " dt=" + std::to_string(self.step_dt) + ">";
        });
}

// ── TODO: reservoir surface ───────────────────────────────────────────────────────
//
// Bindings for the pre-NeuroML reservoir engine. None of it compiles against Spike_Engine:
// a cell's threshold belongs to its ComponentType, and state is laid out per population per
// variable. Each binding needs its engine-side feature first.
//
//  .def(py::init([](const vector<vector<s32>> &network, s64 shape, s64 rank,
//                   f32 resting_mp, f32 decay_rate, f32 learning_rate) {
//           return std::make_unique<Spike_Engine>(&network, shape, rank,
//                                                resting_mp, decay_rate, learning_rate);
//       }), py::arg("network"), py::arg("shape"), py::arg("rank") = 1,
//          py::arg("resting_mp") = 0.1f, py::arg("decay_rate") = 0.01f,
//          py::arg("learning_rate") = 0.00222f)
//
//  .def("setup_lifetime", &Spike_Engine::setup_lifetime,
//       py::arg("lifetime"), py::arg("allocate_logs"),
//       py::arg("max_log_bytes") = Spike_Engine::DEFAULT_MAX_LOG_BYTES)
//  .def("set_input_neurons", &Spike_Engine::set_input_neurons, py::arg("input_neuron_list"))
//  .def("reset_state", &Spike_Engine::reset_state,
//       py::arg("last_spiked_value") = 0, py::arg("active_gen_value") = -1)
//  .def("is_alive", &Spike_Engine::is_alive)
//
//  // Bifurcation search: the weight scale at the edge of runaway activity.
//  .def("estimate_bifurcation_weight", &Spike_Engine::estimate_bifurcation_weight,
//       py::arg("input_period") = 1)
//  .def("scale_uniform_weights_near_bifurcation",
//       [](Spike_Engine &self, s32 input_period, f32 scale, bool freeze_learning) {
//           f32 target = 0.0f, w_accum = 0.0f, w_instant = 0.0f;
//           self.scale_uniform_weights_near_bifurcation(
//               &target, &w_accum, &w_instant, input_period, scale, freeze_learning, nullptr);
//           return py::make_tuple(target, w_accum, w_instant);
//       }, py::arg("input_period") = 1, py::arg("scale") = 1.2f,
//          py::arg("freeze_learning") = false)
//  .def("scale_randomized_weights_near_bifurcation",
//       &Spike_Engine::scale_randomized_weights_near_bifurcation,
//       py::arg("input_period") = 1, py::arg("scale") = 1.2f,
//       py::arg("freeze_learning") = false)
//
//  // Reservoir readout: spike traces, normalised voltages and a bias, one vector per tick.
//  .def("get_reservoir_features_vector",
//       [](Spike_Engine &self, s64 tick, f32 spike_tau, f32 voltage_scale) { ... },
//       py::arg("tick"), py::arg("spike_tau"), py::arg("voltage_scale"))
//
//  // All-in-one record loop; record_membrane_video covers its membrane half.
//  .def("start_static_record", &Spike_Engine::start_static_record,
//       py::arg("input_spikes"), py::arg("lifetime"), py::arg("filename"),
//       py::arg("record_membrane") = true, py::arg("record_stride") = 1,
//       py::arg("compression") = std::optional<std::string>("auto"),
//       py::arg("compression_level") = std::optional<int>{},
//       py::arg("full_decay") = true, py::arg("compression_async") = false,
//       py::arg("compression_queue_max") = static_cast<usize>(8),
//       py::arg("compression_chunk_bytes") = static_cast<usize>(4 * 1024 * 1024),
//       py::call_guard<py::gil_scoped_release>())
//
//  // Flat per-neuron state reads; state_variable_array covers the membrane one. The
//  // generated-kernel engine has no active set.
//  .def("get_membrane_potentials", ...)
//  .def("get_network_inputs", ...)
//  .def("get_last_spiked", ...)
//  .def("get_last_tick_updated", ...)
//  .def("get_active_neuron_indices", ...)
//  .def("get_active_neuron_count", ...)
//
//  .def_readonly("input_neuron_count", &Spike_Engine::input_neuron_count)
//  .def_readwrite("resting_membrane_potential", &Spike_Engine::resting_membrane_potential)
//  .def_readwrite("decay_rate", &Spike_Engine::decay_rate)
//  .def_readwrite("learning_rate", &Spike_Engine::learning_rate)
//  .def_readwrite("spike_period", &Spike_Engine::spike_period)
//  .def_readwrite("spike_threshold", &Spike_Engine::spike_threshold)
//  .def_readwrite("use_constant_weight", &Spike_Engine::use_constant_weight)
//  .def_readonly("running", &Spike_Engine::running)
