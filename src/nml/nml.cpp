#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>

#include "spikecorec/nml/nml.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/units.h"

#include <algorithm>
#include <stdexcept>

using namespace std;
using namespace spikecorec;
using namespace std::filesystem;

namespace spikecorec::nml {

// NML_Node implementation
bool NML_Node::is_declaration_type() const {
    return string_to_declaration_type(tag_name) != NML_DeclarationType::NOT_A_TYPE;
}

bool NML_Node::has_attribute(const String &name) const {
    return attributes.find(name) != attributes.end();
}

void NML_Node::add_attribute(const String &name, String value) {
    attributes[name] = std::move(value);
}

void NML_Node::nest(NML_Node component) {
    body.push_back(component);
}

String NML_Node::get_attribute(const String &name) const {
    auto entry = attributes.find(name);
    if (entry == attributes.end()) return "";
    return entry->second;
}

NML_Declaration NML_Node::to_declaration() const {
    NML_Declaration declaration(string_to_declaration_type(tag_name), attributes);

    for (const NML_Node &child : body) {
        if (!child.is_declaration_type()) continue;
        declaration.children.push_back(child.to_declaration());
    }

    return declaration;
}

// NML Declaration related implementations
bool NML_Declaration::has_value(const String &key) const {
    return datavalues.find(key) != datavalues.end();
}

String NML_Declaration::get_value(const String &key) const {
    auto entry = datavalues.find(key);
    if (entry == datavalues.end()) {
        String message = "Could not find key: " + key + " in neuroml declaration.";
        throw out_of_range(message);
    }

    return entry->second;
}

void DeclarationList::insert(const NML_Declaration &declaration) {
    ordered_declaration_type_counts[declaration.tag_type] += 1;

    auto comparator = [](const NML_Declaration &current, const NML_Declaration &next) {
        return current.tag_type < next.tag_type;
    };
    auto index = std::lower_bound(
            declarations.begin(), declarations.end(),
            declaration,
            comparator);

    declarations.insert(index, declaration);
}


// Only these declarations are supplied by an instance's XML attributes. StateVariables are
// runtime state initialized by OnStart, and Exposure/Children/EventPort/dynamics tags carry
// no instance value at all -- binding them would store empty strings for every one.
static bool is_instance_bound_declaration(NML_DeclarationType type) {
    switch (type) {
        case NML_DeclarationType::Parameter:
        case NML_DeclarationType::Property:
        case NML_DeclarationType::Text:
        case NML_DeclarationType::Path:
        case NML_DeclarationType::ComponentReference:
            return true;
        default:
            return false;
    }
}

void NML_Parser::bind_instance_data(
    NML_Node *instance_node,
    const ComponentType &component_type,
    ComponentInstance &instance
) {
    for (const auto &declaration : component_type.declarations.for_all()) {
        if (!is_instance_bound_declaration(declaration.tag_type)) continue;
        if (!declaration.has_value("name")) continue;

        String declaration_name = declaration.get_value("name");

        // Absent is not empty: a Fixed-pinned Parameter or a Property with a defaultValue
        // legitimately never appears on the instance tag.
        if (!instance_node->has_attribute(declaration_name)) continue;

        instance.instance_data[declaration_name] =
                instance_node->get_attribute(declaration_name);
    }
}


// "-60mV" -> -0.06. Splits the numeric prefix from the unit suffix and scales to SI.
f64 parse_quantity(const String &value) {
    if (value.empty()) return 0.0;

    usize cursor = 0;
    while (cursor < value.size() && isspace(static_cast<unsigned char>(value[cursor]))) cursor += 1;

    usize number_start = cursor;
    if (cursor < value.size() && (value[cursor] == '-' || value[cursor] == '+')) cursor += 1;
    while (cursor < value.size() &&
           (isdigit(static_cast<unsigned char>(value[cursor])) || value[cursor] == '.')) {
        cursor += 1;
    }

    // Exponent, e.g. "1.5e-3mV".
    if (cursor < value.size() && (value[cursor] == 'e' || value[cursor] == 'E')) {
        usize exponent_cursor = cursor + 1;
        if (exponent_cursor < value.size() &&
            (value[exponent_cursor] == '-' || value[exponent_cursor] == '+')) {
            exponent_cursor += 1;
        }
        if (exponent_cursor < value.size() &&
            isdigit(static_cast<unsigned char>(value[exponent_cursor]))) {
            cursor = exponent_cursor;
            while (cursor < value.size() &&
                   isdigit(static_cast<unsigned char>(value[cursor]))) cursor += 1;
        }
    }

    if (cursor == number_start) return 0.0;

    f64 magnitude = 0.0;
    try {
        magnitude = std::stod(value.substr(number_start, cursor - number_start));
    } catch (const std::exception &) {
        return 0.0;
    }

    while (cursor < value.size() && isspace(static_cast<unsigned char>(value[cursor]))) cursor += 1;

    return magnitude * units::unit_suffix_scale(value.substr(cursor));
}


// Dynamics extraction -- walks a ComponentType's declarations and flattens them into
// stage-tagged instructions. Nesting supplies the context a flat list can't: the owning
// Regime, and the OnCondition/OnEvent whose firing gates a StateAssignment or EventOut.
static DynamicsStage stage_for_declaration(NML_DeclarationType type) {
    switch (type) {
        case NML_DeclarationType::OnStart:         return DynamicsStage::Initialize;
        case NML_DeclarationType::TimeDerivative:
        case NML_DeclarationType::DerivedVariable:
        case NML_DeclarationType::ConditionalDerivedVariable:
        case NML_DeclarationType::Case:            return DynamicsStage::Integrate;
        case NML_DeclarationType::OnCondition:     return DynamicsStage::Detect;
        case NML_DeclarationType::StateAssignment: return DynamicsStage::Reset;
        case NML_DeclarationType::EventOut:        return DynamicsStage::Emit;
        case NML_DeclarationType::OnEvent:         return DynamicsStage::Arrival;
        default:                                   return DynamicsStage::RegimeEntry;
    }
}

static String first_present_value(const NML_Declaration &declaration,
                                  const Vector<String> &keys) {
    for (const String &key : keys) {
        if (declaration.has_value(key)) return declaration.get_value(key);
    }
    return "";
}

static void collect_dynamics_instructions(
    const NML_Declaration &declaration,
    const String &regime_name,
    const String &condition,
    Vector<DynamicsInstruction> &return_value
) {
    if (!is_dynamic_type(declaration.tag_type)) return;

    DynamicsInstruction instruction(
            stage_for_declaration(declaration.tag_type), declaration.tag_type);

    // The written name lives under a different attribute per tag: `variable` on
    // TimeDerivative/StateAssignment, `name` on DerivedVariable/Regime, `port` on
    // OnEvent/EventOut, `regime` on Transition.
    instruction.target =
            first_present_value(declaration, {"variable", "name", "port", "regime"});
    instruction.expression = first_present_value(declaration, {"value", "test", "select"});
    instruction.regime_name = regime_name;
    instruction.condition = condition;

    return_value.push_back(instruction);

    // A Regime scopes everything under it; a condition or event handler gates everything
    // under it. Both propagate down as context for the children.
    String child_regime_name = regime_name;
    if (declaration.tag_type == NML_DeclarationType::Regime) {
        child_regime_name = instruction.target;
    }

    String child_condition = condition;
    if (declaration.tag_type == NML_DeclarationType::OnCondition ||
        declaration.tag_type == NML_DeclarationType::OnEvent ||
        declaration.tag_type == NML_DeclarationType::OnStart ||
        declaration.tag_type == NML_DeclarationType::OnEntry) {
        child_condition = instruction.expression;
    }

    for (const NML_Declaration &child : declaration.children) {
        collect_dynamics_instructions(child, child_regime_name, child_condition, return_value);
    }
}

static Vector<DynamicsInstruction> extract_dynamics_program(const ComponentType &component_type) {
    Vector<DynamicsInstruction> program;
    for (const auto &declaration : component_type.declarations.for_all()) {
        collect_dynamics_instructions(declaration, "", "", program);
    }
    return program;
}

static Vector<String> extract_state_variable_names(const ComponentType &component_type) {
    Vector<String> names;
    for (const auto &declaration : component_type.declarations.for_all()) {
        if (declaration.tag_type != NML_DeclarationType::StateVariable) continue;
        if (!declaration.has_value("name")) continue;
        names.push_back(declaration.get_value("name"));
    }
    return names;
}

// Conductance-based synapses carry a driving force g*(erev - v), so they need the
// postsynaptic voltage; current-based ones do not.
static bool is_conductance_based(const ComponentType &component_type) {
    bool declares_erev = false;
    bool requires_voltage = false;

    for (const auto &declaration : component_type.declarations.for_all()) {
        if (!declaration.has_value("name")) continue;
        String name = declaration.get_value("name");

        if (declaration.tag_type == NML_DeclarationType::Parameter && name == "erev") {
            declares_erev = true;
        }
        if (declaration.tag_type == NML_DeclarationType::Requirement && name == "v") {
            requires_voltage = true;
        }
    }

    return declares_erev && requires_voltage;
}

// A synapse needs per-edge storage when its state does not superpose across converging
// edges. In NeuroML that is signalled compositionally: a plasticity or block mechanism
// child makes arrival multiplicative/state-dependent, so the same base synapse type flips
// category on its children rather than on its extends chain.
static bool requires_per_edge_state(const ComponentType &component_type) {
    static const Set<String> non_superposing_child_types = {
        "basePlasticityMechanism", "baseBlockMechanism"
    };

    for (const auto &declaration : component_type.declarations.for_all()) {
        if (declaration.tag_type != NML_DeclarationType::Children) continue;
        if (!declaration.has_value("type")) continue;

        if (non_superposing_child_types.count(declaration.get_value("type")) > 0) return true;
    }

    return false;
}


// Output format inferred from the file extension, matching resolve_spire_compression's
// rules. A non-.spire extension means the NML-standard column matrix.
static OutputFileFormat output_format_for_filename(const String &filename) {
    auto ends_with = [&filename](const String &suffix) {
        return filename.size() >= suffix.size() &&
               filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    if (ends_with(".spire.gz") || ends_with(".spire.gzip")) return OutputFileFormat::SPIREGZIP;
    if (ends_with(".spire.xz") || ends_with(".spire.lzma")) return OutputFileFormat::SPIREXZ;
    if (ends_with(".spire.bz2"))                           return OutputFileFormat::SPIREBZ2;
    if (ends_with(".spire"))                               return OutputFileFormat::SPIRE;

    return OutputFileFormat::NML_STANDARD;
}

// Every <Constant> a ComponentType declares, namespaced by its owning type so that
// hindmarshRose1984Cell's MSEC and fitzHughNagumoCell's SEC cannot collide.
static void collect_component_type_constants(
    const NML_Declaration &declaration,
    const String &component_type_name,
    UnorderedMap<String, Real> &return_value
) {
    if (declaration.tag_type == NML_DeclarationType::Constant &&
        declaration.has_value("name") && declaration.has_value("value")) {
        Real resolved;
        resolved.float64 = parse_quantity(declaration.get_value("value"));
        return_value[component_type_name + "." + declaration.get_value("name")] = resolved;
    }

    for (const NML_Declaration &child : declaration.children) {
        collect_component_type_constants(child, component_type_name, return_value);
    }
}

// Instance ids are scoped by parent ("net1.pop1"); model paths name the bare id.
static String leaf_id(const String &scoped_id) {
    usize separator = scoped_id.rfind('.');
    if (separator == String::npos) return scoped_id;
    return scoped_id.substr(separator + 1);
}

// Resolves a connection endpoint to a global neuron index. Three spellings occur:
// "../pre[0]" and "pre[0]" on connection/synapticConnection, and "../pre/0/cell1" when
// the population was declared as a populationList. Returns -1 if unresolvable.
static s64 resolve_cell_path(
    const String &path,
    const UnorderedMap<String, s64> &population_first_neuron_index,
    const UnorderedMap<String, s64> &population_size
) {
    String remaining = path;
    while (remaining.rfind("../", 0) == 0) remaining = remaining.substr(3);

    String population_name;
    String index_text;

    usize bracket = remaining.find('[');
    if (bracket != String::npos) {
        usize closing = remaining.find(']', bracket);
        if (closing == String::npos) return -1;

        population_name = remaining.substr(0, bracket);
        index_text = remaining.substr(bracket + 1, closing - bracket - 1);
    } else {
        usize first_slash = remaining.find('/');
        if (first_slash == String::npos) return -1;

        population_name = remaining.substr(0, first_slash);

        usize second_slash = remaining.find('/', first_slash + 1);
        index_text = (second_slash == String::npos)
                ? remaining.substr(first_slash + 1)
                : remaining.substr(first_slash + 1, second_slash - first_slash - 1);
    }

    auto base_index = population_first_neuron_index.find(population_name);
    if (base_index == population_first_neuron_index.end()) return -1;

    s64 local_index = 0;
    try {
        local_index = static_cast<s64>(std::stoll(index_text));
    } catch (const std::exception &) {
        return -1;
    }

    auto size = population_size.find(population_name);
    if (size != population_size.end() && local_index >= size->second) return -1;

    return base_index->second + local_index;
}


// Reads a Simulation instance: run length, dt, seed, and the recording selections its
// Display/OutputFile/EventOutputFile children describe.
static void apply_simulation_node(const NML_Node &simulation_node, NML_ParseResult &return_value) {
    if (simulation_node.has_attribute("step")) {
        return_value.step_dt = parse_quantity(simulation_node.get_attribute("step"));
    }
    if (simulation_node.has_attribute("length")) {
        return_value.simulation_duration = parse_quantity(simulation_node.get_attribute("length"));
    }
    if (return_value.step_dt > 0.0) {
        return_value.total_tick_count =
                static_cast<s64>(return_value.simulation_duration / return_value.step_dt);
    }

    if (simulation_node.has_attribute("seed")) {
        try {
            return_value.random_seed =
                    static_cast<u64>(std::stoull(simulation_node.get_attribute("seed")));
        } catch (const std::exception &) {
            // A non-numeric seed is not worth failing the whole parse over.
        }
    }

    for (const NML_Node &child : simulation_node.body) {
        // OutputFile/EventOutputFile name a destination; Display is on-screen only and
        // carries no file, so it contributes no recording profile.
        if (child.tag_name != "OutputFile" && child.tag_name != "EventOutputFile") continue;

        String filename = child.get_attribute("fileName");
        if (filename.empty()) continue;

        RecordingConfig recording_profile{};
        recording_profile.output_filenames.push_back(filename);
        recording_profile.file_output_format.push_back(
                child.tag_name == "EventOutputFile"
                        ? OutputFileFormat::SPIKE_EVENTS
                        : output_format_for_filename(filename));

        // One column / event selection per recorded quantity.
        recording_profile.recordings_count = 0;
        for (const NML_Node &selection : child.body) {
            if (selection.tag_name == "OutputColumn" ||
                selection.tag_name == "EventSelection") {
                recording_profile.recordings_count += 1;
            }
        }

        return_value.recording_profiles.push_back(std::move(recording_profile));
    }
}


// parser code

NML_ParseResult NML_Parser::export_model_details_to_engine(NML_Node *lems_root) {
    NML_ParseResult return_value;

    // four categories of things that need to be allocated:
    // 1. cell specific state and metadata about the cells
    // 2. synapse specific state + connections
    // 3. network input buffer
    // 4. general network constants
    //
    // additionally we will also need the parse result to carry with it info on the units specified
    //
    // then we also need to extract information related to dynamics
    // and kernel compilation
    // Pass 0 -- document scope. Runs first because dt has to be known before the input
    // pass can express delays in ticks.
    if (lems_root) {
        for (const NML_Node &child : lems_root->body) {
            if (child.tag_name == "Constant") {
                if (!child.has_attribute("name")) continue;

                Real resolved;
                resolved.float64 = parse_quantity(child.get_attribute("value"));
                return_value.global_constants[child.get_attribute("name")] = resolved;
                continue;
            }

            // The Simulation instance is a component instance at document scope, not a
            // language element -- matched here by the ComponentType it instantiates.
            if (child.tag_name == "Simulation") {
                apply_simulation_node(child, return_value);
            }
        }
    }

    // Every ComponentType-scoped <Constant>, namespaced by its owning type.
    for (const auto &[type_name, component_type] : declared_component_types) {
        for (const auto &declaration : component_type.declarations.for_all()) {
            collect_component_type_constants(
                    declaration, type_name, return_value.global_constants);
        }
    }

    // Pass 1 -- bucket every instance by category. A single unordered walk cannot do the
    // whole export: the adjacency list needs population index ranges, which are only known
    // once every population has been assigned one.
    Vector<String> cell_instance_ids;
    Vector<String> synapse_instance_ids;
    Vector<String> input_instance_ids;
    Vector<String> graph_instance_ids;
    Vector<String> general_instance_ids;

    for (const auto &[id, instance] : instance_table) {
        if (!instance.component_type) continue;

        switch (instance.component_type->runtime_category) {
            case RuntimeCategory::Cell:
                cell_instance_ids.push_back(id);
                break;
            case RuntimeCategory::Synapse:
                synapse_instance_ids.push_back(id);
                break;
            case RuntimeCategory::InputScheme:
                input_instance_ids.push_back(id);
                break;
            case RuntimeCategory::Graph:
                graph_instance_ids.push_back(id);
                break;
            case RuntimeCategory::General:
            default:
                general_instance_ids.push_back(id);
                break;
        }
    }

    // General -- everything that is neither dynamics nor graph. Any scalar an instance
    // here carries is model-wide, so it lands in global_constants namespaced by instance
    // id; the Simulation instance additionally supplies dt / length / seed if the
    // document-scope pass did not already find it.
    for (const String &instance_id : general_instance_ids) {
        const ComponentInstance &instance = instance_table.at(instance_id);

        if (instance.component_type->name == "Simulation") {
            if (return_value.step_dt <= 0.0) {
                auto step = instance.instance_data.find("step");
                if (step != instance.instance_data.end()) {
                    return_value.step_dt = parse_quantity(step->second);
                }

                auto length = instance.instance_data.find("length");
                if (length != instance.instance_data.end()) {
                    return_value.simulation_duration = parse_quantity(length->second);
                }

                if (return_value.step_dt > 0.0) {
                    return_value.total_tick_count = static_cast<s64>(
                            return_value.simulation_duration / return_value.step_dt);
                }
            }

            if (!return_value.random_seed.has_value()) {
                auto seed = instance.instance_data.find("seed");
                if (seed != instance.instance_data.end()) {
                    try {
                        return_value.random_seed =
                                static_cast<u64>(std::stoull(seed->second));
                    } catch (const std::exception &) {
                        // Non-numeric seed; leave unset rather than failing the parse.
                    }
                }
            }
            continue;
        }

        for (const auto &[value_name, value_text] : instance.instance_data) {
            Real resolved;
            resolved.float64 = parse_quantity(value_text);
            return_value.global_constants[leaf_id(instance.id) + "." + value_name] = resolved;
        }
    }

    // Pass 2 -- cell types: state vector shape, starting parameters, dynamics program.
    UnorderedMap<String, s64> cell_type_indices;

    for (const String &instance_id : cell_instance_ids) {
        const ComponentInstance &instance = instance_table.at(instance_id);
        const ComponentType &component_type = *instance.component_type;

        if (cell_type_indices.find(component_type.name) == cell_type_indices.end()) {
            s64 type_index = static_cast<s64>(cell_type_indices.size());
            cell_type_indices[component_type.name] = type_index;

            Vector<String> state_variable_names = extract_state_variable_names(component_type);

            return_value.cell_state_offsets[component_type.name] = static_cast<s32>(type_index);
            return_value.cell_state_size.push_back(
                    static_cast<s64>(state_variable_names.size()));
            return_value.cell_state_variable_names[component_type.name] =
                    std::move(state_variable_names);
            return_value.cell_dynamics[component_type.name] =
                    extract_dynamics_program(component_type);
        }

        Vector<Real> starting_parameters;
        for (const auto &[parameter_name, parameter_value] : instance.instance_data) {
            (void)parameter_name;
            Real resolved;
            resolved.float64 = parse_quantity(parameter_value);
            starting_parameters.push_back(resolved);
        }
        return_value.cell_starting_parameters.push_back(std::move(starting_parameters));
    }

    return_value.cell_type_count = static_cast<s64>(cell_type_indices.size());

    // Pass 3 -- synapse types: same shape as cells, plus the two storage classifications
    // the engine needs before it can size the weight matrix.
    UnorderedMap<String, s64> synapse_type_indices;

    for (const String &instance_id : synapse_instance_ids) {
        const ComponentInstance &instance = instance_table.at(instance_id);
        const ComponentType &component_type = *instance.component_type;

        if (synapse_type_indices.find(component_type.name) == synapse_type_indices.end()) {
            s64 type_index = static_cast<s64>(synapse_type_indices.size());
            synapse_type_indices[component_type.name] = type_index;

            return_value.synapse_state_size.push_back(
                    static_cast<s64>(extract_state_variable_names(component_type).size()));
            return_value.synapse_dynamics[component_type.name] =
                    extract_dynamics_program(component_type);

            if (is_conductance_based(component_type)) {
                return_value.conductance_based_synapse_types.insert(type_index);
            }
            if (requires_per_edge_state(component_type)) {
                return_value.per_edge_synapse_projection_types.insert(type_index);
            }
        }

        Vector<Real> starting_parameters;
        for (const auto &[parameter_name, parameter_value] : instance.instance_data) {
            (void)parameter_name;
            Real resolved;
            resolved.float64 = parse_quantity(parameter_value);
            starting_parameters.push_back(resolved);
        }
        return_value.synapse_starting_parameters.push_back(std::move(starting_parameters));
    }

    return_value.synapse_type_count = static_cast<s64>(synapse_type_indices.size());

    // Pass 4 -- graph: populations first, so each gets a contiguous global neuron range,
    // then projections whose endpoints resolve against those ranges.
    UnorderedMap<String, s64> population_first_neuron_index;
    UnorderedMap<String, s64> population_size;

    for (const String &instance_id : graph_instance_ids) {
        const ComponentInstance &instance = instance_table.at(instance_id);

        // networkWithTemperature carries a model-wide temperature that every
        // temperature-dependent channel reads.
        auto temperature = instance.instance_data.find("temperature");
        if (temperature != instance.instance_data.end()) {
            Real resolved;
            resolved.float64 = parse_quantity(temperature->second);
            return_value.global_constants[leaf_id(instance.id) + ".temperature"] = resolved;
        }

        if (instance.component_type->name != "population" &&
            instance.component_type->name != "populationList") {
            continue;
        }

        // A sized population is one instance covering `size` neurons; a populationList
        // instead enumerates them as child instances.
        s64 size = static_cast<s64>(instance.structured_instance_data.size());
        auto size_attribute = instance.instance_data.find("size");
        if (size == 0 && size_attribute != instance.instance_data.end()) {
            size = static_cast<s64>(parse_quantity(size_attribute->second));
        }

        // Keyed on the bare id, since that is what connection paths name.
        String population_name = leaf_id(instance.id);
        population_first_neuron_index[population_name] = return_value.total_cell_count;
        population_size[population_name] = size;
        return_value.total_cell_count += size;
    }

    return_value.network_adjacency_list.assign(
            static_cast<usize>(return_value.total_cell_count), Vector<s64>());

    for (const String &instance_id : graph_instance_ids) {
        const ComponentInstance &instance = instance_table.at(instance_id);
        const String &type_name = instance.component_type->name;

        if (type_name != "connection" && type_name != "connectionWD" &&
            type_name != "synapticConnection") {
            continue;
        }

        auto pre_endpoint = instance.instance_data.find(
                type_name == "synapticConnection" ? "from" : "preCellId");
        auto post_endpoint = instance.instance_data.find(
                type_name == "synapticConnection" ? "to" : "postCellId");

        if (pre_endpoint == instance.instance_data.end()) continue;
        if (post_endpoint == instance.instance_data.end()) continue;

        s64 pre_neuron_index = resolve_cell_path(
                pre_endpoint->second, population_first_neuron_index, population_size);
        s64 post_neuron_index = resolve_cell_path(
                post_endpoint->second, population_first_neuron_index, population_size);

        if (pre_neuron_index < 0 || post_neuron_index < 0) continue;

        return_value.network_adjacency_list[static_cast<usize>(pre_neuron_index)]
                .push_back(post_neuron_index);
    }

    // Pass 5 -- inputs.
    for (const String &instance_id : input_instance_ids) {
        const ComponentInstance &instance = instance_table.at(instance_id);

        SimulationInputConfig input_profile{};
        input_profile.continuous_current_injection =
                instance.instance_data.find("amplitude") != instance.instance_data.end();

        auto amplitude = instance.instance_data.find("amplitude");
        if (amplitude != instance.instance_data.end()) {
            input_profile.amplitude = parse_quantity(amplitude->second);
        }

        auto rate = instance.instance_data.find("rate");
        if (rate != instance.instance_data.end()) {
            input_profile.rate = parse_quantity(rate->second);
        }

        // Times are held in ticks so the engine never converts; dt has to be known first.
        if (return_value.step_dt > 0.0) {
            auto delay = instance.instance_data.find("delay");
            if (delay != instance.instance_data.end()) {
                input_profile.start_tick =
                        static_cast<s64>(parse_quantity(delay->second) / return_value.step_dt);
            }

            auto duration = instance.instance_data.find("duration");
            if (duration != instance.instance_data.end()) {
                input_profile.end_tick =
                        input_profile.start_tick +
                        static_cast<s64>(parse_quantity(duration->second) / return_value.step_dt);
            }
        }

        return_value.input_profiles.push_back(input_profile);
    }

    return return_value;
}

RuntimeCategory get_component_runtime_category(const String &component_name, UnorderedMap<String, Vector<String>> &dependency_map) {
    static const Vector<Pair<String, RuntimeCategory>> category_roots = {
        {"baseCell",              RuntimeCategory::Cell},
        {"baseSynapse",           RuntimeCategory::Synapse},
        {"baseGradedSynapse",     RuntimeCategory::Synapse},
        {"basePointCurrent",      RuntimeCategory::InputScheme},
        {"baseSpikeSource",       RuntimeCategory::InputScheme},
        {"network",               RuntimeCategory::Graph},
        {"basePopulation",        RuntimeCategory::Graph},
        {"projection",            RuntimeCategory::Graph},
        {"electricalProjection",  RuntimeCategory::Graph},
        {"continuousProjection",  RuntimeCategory::Graph},
        {"explicitConnection",    RuntimeCategory::Graph},
        {"inputList",             RuntimeCategory::Graph},
        {"explicitInput",         RuntimeCategory::Graph},
    };

    for (const auto &[root_name, category] : category_roots) {
        if (component_name == root_name) return category;

        Vector<String> search_frontier = {root_name};
        Set<String> visited;

        while (!search_frontier.empty()) {
            String current_type = search_frontier.back();
            search_frontier.pop_back();

            if (!visited.insert(current_type).second) continue;

            auto entry = dependency_map.find(current_type);
            if (entry == dependency_map.end()) continue;

            for (const String &descendant : entry->second) {
                if (descendant == component_name) return category;
                search_frontier.push_back(descendant);
            }
        }
    }

    return RuntimeCategory::General;
}

void NML_Parser::load_component_types_in_file(NML_Node *file_root) {
    // 1. figure out dependency order
    UnorderedMap<String, NML_Node *> node_catalogue;
    UnorderedMap<String, Vector<String>> component_type_dependencies; // parent type -> child types
    NML_Node *current_node;
    Vector<NML_Node *> nodes;
    nodes.push_back(file_root);

    while (!nodes.empty()) {
        current_node = nodes.back();
        nodes.pop_back();

        for (NML_Node child_node : current_node->body) {
            nodes.push_back(&child_node);
        }

        if (current_node->tag_name != "ComponentType") continue;

        String component_type_name = current_node->get_attribute("name");

        if (node_catalogue.find(component_type_name) != node_catalogue.end()) {
            throw runtime_error(
                    "Invalid redefintion of componenet type: " + component_type_name);
        }

        node_catalogue[component_type_name] = current_node;

        if (component_type_dependencies.find(component_type_name) == component_type_dependencies.end()) {
            // add new type entry to map
            component_type_dependencies[component_type_name];
        } 

        if (current_node->has_attribute("extends")) {
            // this node is a child of a parent type
            String parent_type_name = current_node->get_attribute("extends");
            component_type_dependencies[parent_type_name].push_back(component_type_name);
        }
    }

    // 2. instantiate all component types in order of extensions
    auto instantiation_order = Vector<Pair<String, Vector<String>>>(
        component_type_dependencies.begin(), component_type_dependencies.end()
    );
    sort(instantiation_order.begin(), instantiation_order.end(),
         [](const auto& extension_list_a, const auto& extension_list_b) {
        return extension_list_a.second.size() < extension_list_b.second.size();
    });
    for (const auto &mapping_pair : instantiation_order) {
        const String &parent_type_name = mapping_pair.first;
        const Vector<String> &child_type_names = mapping_pair.second;

        NML_Node *current_node = node_catalogue[parent_type_name];

        DeclarationList component_declarations;
        extract_all_declarations_nested_in_node(current_node, component_declarations);

        if (declared_component_types.find(parent_type_name) != declared_component_types.end()) {
            RuntimeCategory category = get_component_runtime_category(parent_type_name, component_type_dependencies);
            declared_component_types[parent_type_name] = ComponentType(parent_type_name, component_declarations, category);
        }

        ComponentType &parent_component_type = declared_component_types[parent_type_name];
        for (const auto &child_name : child_type_names) {
            NML_Node *child_node = node_catalogue[child_name];

            DeclarationList child_declarations = parent_component_type.declarations;
            extract_all_declarations_nested_in_node(child_node, child_declarations);
            RuntimeCategory category = get_component_runtime_category(child_name, component_type_dependencies);
            declared_component_types[child_name] = ComponentType(child_name, child_declarations, category);
        }
    }
}

NML_ParseResult NML_Parser::parse_lems(const String &lems_main_file) {
    if (!exists(STANDARD_LIBRARY_PATH)) {
        log::logger().error("NML standard library path does not exist: {}", STANDARD_LIBRARY_PATH);
        return NML_ParseResult{};
    }

    // load all base neuroml component types
    for (const auto &entry : directory_iterator(STANDARD_LIBRARY_PATH)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".xml" && entry.path().extension() != ".nml") continue;
        
        String filepath = entry.path().string();
        parse_neuroml(filepath);
    }

    xmlNodePtr lems_root_xml = get_xml_root(lems_main_file);
    NML_Node main_file_root = xml_node_to_nml_node(lems_root_xml);

    Set<String> parsed_lems_files;
    for (NML_Node &child : main_file_root.body) {
        if (child.tag_name == "Target") {} // note: only one target can be found
        else if (child.tag_name == "Include") {
            // can include another LEMS file 
            // OR can include a neuroml file, 
            // parse dispatch will be condtional on this
        }
        else if (child.tag_name == "Dimension") {}
        else if (child.tag_name == "Unit") {}
        else if (child.tag_name == "Constant") {}
        else if (child.tag_name == "ComponentType") {}
        else if (child.tag_name == "Component") {}
        else {} // its an instance?
                // note: we find "Simulation tags" as a Simulation instance here
    }

    return export_model_details_to_engine(&main_file_root);
}

void NML_Parser::parse_neuroml(const String &nml_file_path) {
    if (parsed_neuroml_files.find(nml_file_path) != parsed_neuroml_files.end()) {
        return;
    }

    xmlNodePtr root = get_xml_root(nml_file_path);

    // todo: validate against xsd

    NML_Node neuroml_root = xml_node_to_nml_node(root);

    // 1. parse all included files
    Vector<NML_Node *> found_instance_nodes;
    for (auto &child : neuroml_root.body) {
        if (child.tag_name == "include") {
            String included_filepath = child.get_attribute("href");
            parse_neuroml(included_filepath);
            continue;
        }

        if (child.tag_name == "ComponentType") continue;

        // then this must be a component instance
        found_instance_nodes.push_back(&child);
    }

    // 2. load all component types defined in the file
    load_component_types_in_file(&neuroml_root);

    // 3. instantiate all ComponentType instances in file
    for (auto &child : neuroml_root.body) {
        if (child.tag_name == "include" || child.tag_name == "ComponentType") continue;
        if (declared_component_types.find(child.tag_name) == declared_component_types.end()) {
            continue;
        }

        instantiate(&child);
    }

    parsed_neuroml_files.insert(nml_file_path);
}

void NML_Parser::instantiate(NML_Node *instance_node) {
    String component_type_name = instance_node->tag_name;
    Vector<NML_Node> &instance_body = instance_node->body;

    auto type_entry = declared_component_types.find(component_type_name);
    if (type_entry == declared_component_types.end()) return;
    const ComponentType &component_to_instantiate = type_entry->second;

    String instance_id = instance_node->get_attribute("id");

    instance_table[instance_id];
    ComponentInstance &instance = instance_table[instance_id];
    instance.id = instance_id;
    instance.component_type = &component_to_instantiate;

    bind_instance_data(instance_node, component_to_instantiate, instance);

    if (instance_body.empty()) return;

    // instantiate all child instances
    for (auto &child_instance_node : instance_body) {
        instantiate(&child_instance_node, instance_id);
    }
}

void NML_Parser::instantiate(NML_Node *instance_node, String &parent_instance_id) {
    String component_type_name = instance_node->tag_name;
    Vector<NML_Node> &instance_body = instance_node->body;

    auto type_entry = declared_component_types.find(component_type_name);
    if (type_entry == declared_component_types.end()) return;
    const ComponentType &component_to_instantiate = type_entry->second;

    String instance_id = parent_instance_id + "." + instance_node->get_attribute("id");

    instance_table[instance_id];
    ComponentInstance &instance = instance_table[instance_id];
    instance.id = instance_id;
    instance.component_type = &component_to_instantiate;

    bind_instance_data(instance_node, component_to_instantiate, instance);

    if (instance_body.empty()) return;

    // instantiate all child instances
    for (auto &child_instance_node : instance_body) {
        instantiate(&child_instance_node, instance_id);
    }

    instance_table[parent_instance_id].structured_instance_data.push_back(instance_id);
}

NML_Node NML_Parser::xml_node_to_nml_node(xmlNodePtr node) {
    String tag_name = reinterpret_cast<const char *>(node->name);
    auto component = NML_Node(tag_name);

    for (xmlAttrPtr attribute = node->properties; attribute; attribute = attribute->next) {
        xmlChar *value = xmlGetProp(node, attribute->name);
        if (!value) continue;

        component.add_attribute(reinterpret_cast<const char *>(attribute->name),
                                String(reinterpret_cast<const char *>(value)));
        xmlFree(value);
    }

    for (xmlNodePtr child = node->children; child; child = child->next) {
        if (child->type != XML_ELEMENT_NODE) continue;
        component.nest(xml_node_to_nml_node(child));
    }

    return component;
}

void NML_Parser::extract_all_declarations_nested_in_node(
    NML_Node *node,
    DeclarationList &return_value
) {
    Vector<NML_Node> nodes_to_extract;
    nodes_to_extract.push_back(*node);

    while (!nodes_to_extract.empty()) {
        // Taken by value and popped first: appending current_node's body below would
        // otherwise read from the same vector it grows, and a reallocation mid-insert
        // would leave the source range dangling.
        NML_Node current_node = nodes_to_extract.back();
        nodes_to_extract.pop_back();

        // to_declaration() recurses into nested declarations itself, so a declaration
        // node is consumed whole -- its body must not be walked again here.
        if (current_node.is_declaration_type()) {
            return_value.insert(current_node.to_declaration());
            continue;
        }

        // Not a declaration (Dynamics, Structure, the ComponentType itself, ...) --
        // descend into it looking for the declarations underneath.
        nodes_to_extract.insert(
                nodes_to_extract.end(), current_node.body.begin(), current_node.body.end());
    }
}

// Accumulates each schema validation error's line number and message (libxml2's messages
// already name the offending element, e.g. "Element 'thisTagDoesNotExistInSchema': No
// matching global declaration available for the validation root.") into `*user_data`,
// joined by " | " -- see validate_against_schema's own comment for why this replaces
// libxml2's default, stderr-only error handler.
void collect_schema_validation_error(void *user_data, xmlErrorPtr error) {
    if (!error || !error->message) return;

    String message(error->message);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();

    String *destination = static_cast<String *>(user_data);
    if (!destination->empty()) *destination += " | ";
    *destination += "line " + std::to_string(error->line) + ": " + message;
}

bool NML_Parser::validate_against_schema(const String &nml_file_path) {
    xmlSchemaParserCtxtPtr parser_context = xmlSchemaNewParserCtxt(NML_SCHEMA_PATH.c_str());
    if (!parser_context) {
        log::logger().error("Could not create XSD parser context for schema {}", NML_SCHEMA_PATH);
        return false;
    }

    xmlSchemaPtr schema = xmlSchemaParse(parser_context);
    xmlSchemaFreeParserCtxt(parser_context);
    if (!schema) {
        log::logger().error("Could not parse XSD schema {}", NML_SCHEMA_PATH);
        return false;
    }

    xmlSchemaValidCtxtPtr valid_context = xmlSchemaNewValidCtxt(schema);
    if (!valid_context) {
        log::logger().error(
                "Could not create XSD validation context for schema {}", NML_SCHEMA_PATH);
        xmlSchemaFree(schema);
        return false;
    }

    // Route every validation error through collect_schema_validation_error instead of libxml2's
    // default handler (which only prints to stderr, invisible to both the caller's exception and
    // this codebase's own logger) -- populates last_schema_validation_errors with each error's line
    // number and message (already naming the offending element) so ingest_file's thrown exception
    // can report exactly what failed and where.
    last_schema_validation_errors.clear();
    xmlSchemaSetValidStructuredErrors(
            valid_context, collect_schema_validation_error, &last_schema_validation_errors);

    // xmlSchemaValidateFile: 0 = valid, >0 = validation errors (captured above, by element and line
    // number), <0 = internal/API error.
    int result = xmlSchemaValidateFile(valid_context, nml_file_path.c_str(), 0);

    xmlSchemaFreeValidCtxt(valid_context);
    xmlSchemaFree(schema);

    return result == 0;
}

xmlNodePtr NML_Parser::get_xml_root(const String &filepath) {
    xmlDocPtr document = xmlReadFile(filepath.c_str(), nullptr, XML_PARSE_NOBLANKS);
    if (!document) {
        log::logger().error("Could not parse NML standard library file {}", filepath);
        return nullptr;
    }

    xmlNodePtr root = xmlDocGetRootElement(document);
    if (!root) {
        log::logger().error("NML standard library file {} has no root element", filepath);
        xmlFreeDoc(document);
        return nullptr;
    }

    return root;
}





String &ComponentInstance::get_value(String &key) {
    // Static so the miss case can still hand back a reference; a literal would dangle.
    static String missing_value = "";

    if (instance_data.find(key) == instance_data.end()) {
        return missing_value;
    }

    return instance_data[key];
}

RuntimeCategory ComponentInstance::get_runtime_category() {
    return component_type->runtime_category;
}




}
