#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemas.h>

#include <algorithm>
#include <filesystem>

#include "log.h"
#include "units.h"
#include "types.h"
#include "recording.h"

#include "parser.h"
#include "node.h"
#include "components.h"
#include "dynamics.h"
#include "utilities.h"

using namespace spikecorec;
using namespace std;
using namespace std::filesystem;


namespace spikecorec::nml {

// A count or index written as a whole, non-negative number ("3", "3.0"); nullopt for anything else
// ("2.6", "two", "-1", "3 cells").
static Optional<s64> parse_whole_number(const String &text) {
    const usize first = text.find_first_not_of(" \t\n\r");
    const usize last = text.find_last_not_of(" \t\n\r");
    if (first == String::npos) return nullopt;
    const String trimmed = text.substr(first, last - first + 1);
    if (!isdigit(static_cast<unsigned char>(trimmed.front()))) return nullopt;

    f64 value = 0.0;
    usize consumed = 0;
    try {
        value = std::stod(trimmed, &consumed);
    } catch (const Exception &) {
        return nullopt;
    }
    if (consumed != trimmed.size() || !std::isfinite(value) || value != std::floor(value) ||
        value > static_cast<f64>(Numeric_Limits<s32>::max())) {
        return nullopt;
    }
    return static_cast<s64>(value);
}

// One attribute's quantity, checked against its dimension, with the element named in any error.
static f64 resolve_attribute(const NML_Context &context, const NML_Component_Instance &element,
                             const String &attribute, const String &dimension) {
    try {
        return context.resolve_quantity_of_dimension(element.value_or(attribute), dimension);
    } catch (const Runtime_Error &error) {
        throw Runtime_Error("'" + element.id + "' " + attribute + ": " + error.what());
    }
}

// The name of the state variable at one cell-memory index, or "" when no cell holds it.
static String variable_name_at(const NML_Context &context, s64 cell_memory_index) {
    for (const NML_Component_Instance *population : network_populations(context)) {
        const NML_Component_Type &cell_type = *population_cell(context, *population).component_type;
        const s64 slot_count = NML_Context::cell_slot_count(cell_type);
        const s64 base = context.simulation.population_base_indices.at(population->id);
        if (cell_memory_index < base || cell_memory_index >= base + context.get_population_size(population) * slot_count) {
            continue;
        }
        const usize offset = static_cast<usize>((cell_memory_index - base) % slot_count);
        return offset < cell_type.state_variable_names.size() ? cell_type.state_variable_names[offset] : "";
    }
    return "";
}

bool is_document_scope_directive(const String &tag_name) {
    static const Set<String> directives = {
        "Include", "include", "ComponentType", "Unit", "Dimension", "Constant", "Target"
    };
    return directives.count(tag_name) != 0;
}

// One entry per StateAssignment, EventOut and Transition inside an event handler.
void collect_event_actions(const NML_Node *handler, const String &regime_name, const String &condition,
                           Vector<NML_Dynamics_Expression> &dynamics) {
    for (const NML_Node *action : children_of(handler)) {
        const NML_Tag &tag = action->body;
        NML_Dynamics_Expression entry(tag.tag_type);
        entry.regime_name = regime_name;
        entry.condition = condition;

        if (tag.tag_type == NML_Declaration_Type::StateAssignment) {
            entry.target = tag.get_attribute("variable");
            entry.expression = tag.get_attribute("value");
        } else if (tag.tag_type == NML_Declaration_Type::EventOut) {
            entry.target = tag.get_attribute("port");
        } else if (tag.tag_type == NML_Declaration_Type::Transition) {
            entry.target = tag.get_attribute("regime");
        } else {
            continue;
        }
        dynamics.push_back(std::move(entry));
    }
}

// Flattens a <Structure> element into structure entries, in document order.
void collect_structure(const NML_Node *element, Vector<NML_Structure_Entry> &structure) {
    for (const NML_Node *child : children_of(element)) {
        const NML_Tag &tag = child->body;
        NML_Structure_Entry entry(tag.tag_type);

        switch (tag.tag_type) {
            case NML_Declaration_Type::ChildInstance:
                entry.component = tag.get_attribute("component");
                break;
            case NML_Declaration_Type::With:
                entry.instance = tag.get_attribute("instance");
                entry.alias = tag.get_attribute("as");
                break;
            case NML_Declaration_Type::EventConnection:
                entry.source = tag.get_attribute("from");
                entry.target = tag.get_attribute("to");
                entry.receiver = tag.get_attribute_or("receiver", "");
                entry.source_port = tag.get_attribute_or("sourcePort", "");
                entry.target_port = tag.get_attribute_or("targetPort", "");
                break;
            default:
                continue;
        }
        structure.push_back(std::move(entry));
    }
}

// Flattens a <Dynamics> or <Regime> element into dynamics entries, in document order.
// Each OnStart assignment is an OnStart entry; an event handler's entry precedes its actions.
void collect_dynamics(const NML_Node *element, const String &regime_name,
                      Vector<NML_Dynamics_Expression> &dynamics) {
    for (const NML_Node *child : children_of(element)) {
        const NML_Tag &tag = child->body;
        NML_Dynamics_Expression entry(tag.tag_type);
        entry.regime_name = regime_name;

        switch (tag.tag_type) {
            case NML_Declaration_Type::DerivedVariable:
                entry.target = tag.get_attribute("name");
                entry.expression = tag.get_attribute("value");
                entry.select = tag.get_attribute("select");
                entry.reduce = tag.get_attribute("reduce");
                dynamics.push_back(std::move(entry));
                break;

            case NML_Declaration_Type::ConditionalDerivedVariable:
                for (const NML_Node *case_node : children_of(child)) {
                    if (case_node->body.tag_type != NML_Declaration_Type::Case) continue;
                    NML_Dynamics_Expression case_entry(NML_Declaration_Type::Case);
                    case_entry.regime_name = regime_name;
                    case_entry.target = tag.get_attribute("name");
                    case_entry.condition = case_node->body.get_attribute("condition");
                    case_entry.expression = case_node->body.get_attribute("value");
                    dynamics.push_back(std::move(case_entry));
                }
                break;

            case NML_Declaration_Type::TimeDerivative:
                entry.target = tag.get_attribute("variable");
                entry.expression = tag.get_attribute("value");
                dynamics.push_back(std::move(entry));
                break;

            case NML_Declaration_Type::OnStart:
                for (const NML_Node *assignment : children_of(child)) {
                    if (assignment->body.tag_type != NML_Declaration_Type::StateAssignment) continue;
                    NML_Dynamics_Expression start_entry(NML_Declaration_Type::OnStart);
                    start_entry.target = assignment->body.get_attribute("variable");
                    start_entry.expression = assignment->body.get_attribute("value");
                    dynamics.push_back(std::move(start_entry));
                }
                break;

            case NML_Declaration_Type::OnCondition:
                entry.expression = tag.get_attribute("test");
                dynamics.push_back(entry);
                collect_event_actions(child, regime_name, entry.expression, dynamics);
                break;

            case NML_Declaration_Type::OnEntry:
                dynamics.push_back(entry);
                collect_event_actions(child, regime_name, "", dynamics);
                break;

            case NML_Declaration_Type::OnEvent:
                entry.target = tag.get_attribute("port");
                dynamics.push_back(entry);
                collect_event_actions(child, regime_name, "", dynamics);
                break;

            case NML_Declaration_Type::Regime:
                entry.target = tag.get_attribute("name");
                entry.expression = tag.get_attribute("initial");
                dynamics.push_back(std::move(entry));
                collect_dynamics(child, tag.get_attribute("name"), dynamics);
                break;

            default:
                break;
        }
    }
}

NML_Node *NML_Context::parse_neuroml_file(const String &filepath, Vector<String> &files_to_parse) {
    Error_Code path_error;
    Path canonical = weakly_canonical(Path(filepath), path_error);
    String canonical_text = path_error ? filepath : canonical.string();

    xmlNodePtr xml_root = get_xml_root(filepath);
    if (!xml_root) return nullptr;

    NML_Node *root = build_nml_tree(xml_root);
    xmlFreeDoc(xml_root->doc);

    if (!root) return nullptr;

    Path containing_directory = Path(canonical_text).parent_path();

    for (const NML_Node *child : children_of(root)) {
        const NML_Tag &tag = child->body;

        if (tag.tag_name != "Include" && tag.tag_name != "include") continue;

        String reference = tag.has_attribute("file")
                ? tag.get_attribute("file")
                : tag.get_attribute("href");
        if (reference.empty()) continue;

        Path included = Path(reference);
        if (included.is_relative()) {
            included = containing_directory / included;
        }

        if (!exists(included) && !STANDARD_LIBRARY_PATH.empty()) {
            const Path from_standard_library = Path(STANDARD_LIBRARY_PATH) / Path(reference).filename();
            if (exists(from_standard_library)) {
                included = from_standard_library;
            }
        }

        files_to_parse.push_back(included.string());
    }

    return root;
}

void NML_Context::reset() {
    component_instances.clear();
    component_types.clear();
    for (NML_Node *root : document_roots) delete root;
    document_roots.clear();
    document_filepaths.clear();
    main_document_root = nullptr;
    model_units.clear();
    model_constants.clear();
    target_component_id.clear();
    simulation = NML_Simulation_Context();
}

void NML_Context::parse(const String &main_filepath) {
    if (STANDARD_LIBRARY_PATH.empty() || !exists(STANDARD_LIBRARY_PATH)) {
        log::logger().error("NML standard library path does not exist: {}",
                            STANDARD_LIBRARY_PATH);
        return;
    }
    reset();

    Vector<String> files_to_parse;
    for (const auto &entry : Directory_Iterator(STANDARD_LIBRARY_PATH)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".xml" && entry.path().extension() != ".nml") continue;

        files_to_parse.push_back(entry.path().string());
    }

    std::sort(files_to_parse.begin(), files_to_parse.end());

    files_to_parse.push_back(main_filepath);

    Set<String> parsed_files;

    Error_Code main_path_error;
    const Path canonical_main = weakly_canonical(Path(main_filepath), main_path_error);
    const String canonical_main_file = main_path_error ? main_filepath : canonical_main.string();

    // Resolved once every file is read: a <Constant> may use a <Unit> that a later file declares.
    Vector<const NML_Node *> constant_nodes;
    String main_target_component_id;
    String included_target_component_id;

    // A failed parse leaves no half-built context behind for a later query to walk.
    try {
        String current_file;
        while (!files_to_parse.empty()) {
            current_file = files_to_parse.front();

            Error_Code path_error;
            const Path canonical = weakly_canonical(Path(current_file), path_error);
            const String canonical_file = path_error ? current_file : canonical.string();

            if (parsed_files.count(canonical_file) != 0) {
                files_to_parse.erase(files_to_parse.begin());
                continue;
            }
            parsed_files.insert(canonical_file);

            auto *root = parse_neuroml_file(current_file, files_to_parse);
            files_to_parse.erase(files_to_parse.begin());
            if (!root) continue;

            for (NML_Node *child : children_of(root)) {
                const NML_Tag &tag = child->body;
                if (tag.tag_name == "Unit") {
                    String symbol = tag.get_attribute("symbol");
                    if (symbol.empty()) continue;

                    // LEMS: si = raw * scale * 10^power + offset; power is folded into scale.
                    units::Unit_Definition definition;
                    definition.scale = 1.0;
                    definition.offset = 0.0;
                    definition.dimension = tag.get_attribute("dimension");

                    if (tag.has_attribute("scale")) {
                        auto [scale_magnitude, scale_suffix] = units::split_quantity(tag.get_attribute("scale"));
                        static_cast<void>(scale_suffix);
                        if (scale_magnitude != 0.0) definition.scale = scale_magnitude;
                    }
                    if (tag.has_attribute("power")) {
                        auto [power_magnitude, power_suffix] = units::split_quantity(tag.get_attribute("power"));
                        static_cast<void>(power_suffix);
                        definition.scale *= std::pow(10.0, power_magnitude);
                    }
                    if (tag.has_attribute("offset")) {
                        auto [offset_magnitude, offset_suffix] =
                                units::split_quantity(tag.get_attribute("offset"));
                        static_cast<void>(offset_suffix);
                        definition.offset = offset_magnitude;
                    }

                    model_units[symbol] = definition;
                    continue;
                }

                if (tag.tag_name == "Constant") {
                    constant_nodes.push_back(child);
                    continue;
                }

                // The main file's <Target> wins; without one, the first an included file declares.
                if (tag.tag_name == "Target") {
                    if (canonical_file == canonical_main_file) {
                        main_target_component_id = tag.get_attribute("component");
                    } else if (included_target_component_id.empty()) {
                        included_target_component_id = tag.get_attribute("component");
                    }
                    continue;
                }
            }

            document_roots.push_back(root);
            document_filepaths.push_back(canonical_file);
            if (canonical_file == canonical_main_file) {
                main_document_root = root;
            }
        }
        target_component_id = main_target_component_id.empty() ? included_target_component_id : main_target_component_id;

        for (const NML_Node *constant_node : constant_nodes) {
            const String constant_name = constant_node->body.get_attribute("name");
            if (constant_name.empty()) continue;

            Real resolved;
            resolved.float64 = resolve_quantity(constant_node->body.get_attribute("value"));
            model_constants[constant_name] = resolved;
        }

        parse_component_types();
        parse_component_instances();
        parse_simulation_details(main_document_root);
    } catch (...) {
        reset();
        throw;
    }
}

// Records type_node in component_types; nullptr when it has no name. extends stays unset
// because the parent may not be recorded yet; parse_component_types links it.
NML_Component_Type *NML_Context::create_component_type(
        const NML_Node *type_node, const String &source_filepath) {
    const String type_name = type_node->body.get_attribute("name");
    if (type_name.empty()) {
        log::logger().warn("Ignoring <ComponentType> with no name in {}", source_filepath);
        return nullptr;
    }

    auto existing = component_types.find(type_name);
    if (existing != component_types.end()) {
        throw Runtime_Error(
                "Duplicate ComponentType '" + type_name + "' declared in both " +
                existing->second.source_file + " and " + source_filepath);
    }

    NML_Component_Type component_type(type_name);
    component_type.source_file = source_filepath;
    component_type.source_node = type_node;

    return &component_types.emplace(type_name, std::move(component_type)).first->second;
}

void NML_Context::parse_component_types() {
    for (usize document_index = 0; document_index < document_roots.size(); document_index += 1) {
        const String &filepath = document_filepaths[document_index];

        for (const NML_Node *child : children_of(document_roots[document_index])) {
            if (child->body.tag_name != "ComponentType") continue;

            create_component_type(child, filepath);
        }
    }

    // Sorted, so an unresolved or cyclic extends chain is reported the same way on every run.
    Vector<String> type_names;
    type_names.reserve(component_types.size());
    for (const auto &[type_name, component_type] : component_types) {
        static_cast<void>(component_type);
        type_names.push_back(type_name);
    }
    std::sort(type_names.begin(), type_names.end());

    // Link order is free: a link is only an address, and component_types never moves an element.
    for (const String &type_name : type_names) {
        NML_Component_Type &component_type = component_types.at(type_name);

        const String parent_name = component_type.source_node->body.get_attribute("extends");
        if (parent_name.empty()) continue;

        auto parent = component_types.find(parent_name);
        if (parent == component_types.end()) {
            throw Runtime_Error(
                    "Unresolved ComponentType '" + parent_name +
                    "' referenced from extends chain: " + type_name + " -> " + parent_name);
        }

        component_type.extends = &parent->second;
    }

    // A cycle would make every walk up an extends chain loop forever, so each chain is checked.
    for (const String &type_name : type_names) {
        String chain;
        Set<String> visited_type_names;

        for (const NML_Component_Type *type = &component_types.at(type_name); type;
             type = type->extends) {
            if (!chain.empty()) chain += " -> ";
            chain += type->name;

            if (!visited_type_names.insert(type->name).second) {
                throw Runtime_Error("Cyclic ComponentType extends chain: " + chain);
            }
        }
    }

    // Ancestors' state variables come first, so a subtype keeps its parent's slot order.
    for (const String &type_name : type_names) {
        NML_Component_Type &component_type = component_types.at(type_name);

        Vector<const NML_Component_Type *> chain;
        for (const NML_Component_Type *type = &component_type; type; type = type->extends) {
            chain.push_back(type);
        }

        Set<String> seen_names;
        for (usize chain_index = chain.size(); chain_index > 0; chain_index -= 1) {
            Vector<const NML_Node *> pending_nodes = {chain[chain_index - 1]->source_node};
            while (!pending_nodes.empty()) {
                const NML_Node *node = pending_nodes.back();
                pending_nodes.pop_back();

                const Vector<NML_Node *> &child_nodes = children_of(node);
                for (usize index = child_nodes.size(); index > 0; index -= 1) {
                    pending_nodes.push_back(child_nodes[index - 1]);
                }

                if (node->body.tag_type != NML_Declaration_Type::StateVariable) continue;

                // A subtype that redeclares the name as something else shadows it.
                const NML_Node *declaration =
                        component_type.find_declaration(node->body.namespace_key());
                if (!declaration || declaration->body.tag_type != NML_Declaration_Type::StateVariable) {
                    continue;
                }

                const String name = node->body.get_attribute("name");
                if (seen_names.insert(name).second) {
                    component_type.state_variable_names.push_back(name);
                }
            }
        }
    }

    // A type's own <Dynamics> replaces its ancestors'; without one it inherits the nearest.
    for (const String &type_name : type_names) {
        NML_Component_Type &component_type = component_types.at(type_name);

        const NML_Node *dynamics_node = nullptr;
        for (const NML_Component_Type *type = &component_type; type && !dynamics_node; type = type->extends) {
            for (const NML_Node *child : children_of(type->source_node)) {
                if (child->body.tag_name != "Dynamics") continue;
                dynamics_node = child;
                break;
            }
        }

        component_type.dynamics.clear();
        if (dynamics_node) collect_dynamics(dynamics_node, "", component_type.dynamics);

        // Likewise the nearest <Structure>.
        const NML_Node *structure_node = nullptr;
        for (const NML_Component_Type *type = &component_type; type && !structure_node; type = type->extends) {
            for (const NML_Node *child : children_of(type->source_node)) {
                if (child->body.tag_name != "Structure") continue;
                structure_node = child;
                break;
            }
        }

        component_type.structure.clear();
        if (structure_node) collect_structure(structure_node, component_type.structure);
    }
}

// Creates the instance that instance_node declares, keyed by its full path ("net1/pop0/0").
// parent_instance is nullptr at document scope.
NML_Component_Instance *NML_Context::create_component_instance(
        const NML_Component_Type *component_type, const NML_Node *instance_node,
        NML_Component_Instance *parent_instance) {
    static const Set<NML_Declaration_Type> instance_bound_declarations = {
        NML_Declaration_Type::Parameter,
        NML_Declaration_Type::Property,
        NML_Declaration_Type::Text,
        NML_Declaration_Type::Path,
        NML_Declaration_Type::ComponentReference
    };

    const NML_Tag &tag = instance_node->body;
    const String prefix = parent_instance ? parent_instance->id + "/" : "";

    // An element with no id is named by its type and an ordinal, e.g. "member0".
    String name = tag.get_attribute("id");
    if (name.empty()) {
        usize ordinal = 0;
        while (component_instances.count(prefix + component_type->name + std::to_string(ordinal)) != 0) {
            ordinal += 1;
        }
        name = component_type->name + std::to_string(ordinal);
    }

    const String instance_id = prefix + name;

    if (component_instances.count(instance_id) != 0) {
        throw Runtime_Error("Duplicate component instance '" + instance_id + "'");
    }

    NML_Component_Instance &instance = component_instances[instance_id];
    instance.id = instance_id;
    instance.parent_instance = parent_instance;
    instance.component_type = component_type;

    // Bind only attributes the type declares as instance-set values.
    for (const auto &[attribute_name, attribute_value] : tag.attributes) {
        const NML_Node *declaration = component_type->find_declaration("var:" + attribute_name);
        if (!declaration) {
            declaration = component_type->find_declaration("pathsegment:" + attribute_name);
        }
        if (!declaration) continue;
        if (instance_bound_declarations.count(declaration->body.tag_type) == 0) continue;

        instance.instance_data[attribute_name] = attribute_value;
    }

    // <Fixed> overrides the tag's value; the nearest type's pin wins.
    Set<String> pinned_parameters;
    for (const NML_Component_Type *type = component_type; type; type = type->extends) {
        for (const NML_Node *type_element : children_of(type->source_node)) {
            if (type_element->body.tag_type != NML_Declaration_Type::Fixed) continue;

            const String parameter_name = type_element->body.get_attribute("parameter");
            if (parameter_name.empty()) continue;
            if (!pinned_parameters.insert(parameter_name).second) continue;

            const NML_Node *pinned = component_type->find_declaration("var:" + parameter_name);
            if (!pinned || pinned->body.tag_type != NML_Declaration_Type::Parameter) continue;

            instance.instance_data[parameter_name] = type_element->body.get_attribute("value");
        }
    }

    if (parent_instance) parent_instance->data_order.push_back(instance_id);

    return &instance;
}

// Instantiates every component the documents declare, nested children included.
void NML_Context::parse_component_instances() {
    // Rebuilt, not appended to: a second pass would append every child to its parent twice.
    component_instances.clear();

    struct Pending_Instance {
        const NML_Node *node;
        NML_Component_Instance *parent_instance;
    };

    // Depth first in document order, so elements are pushed in reverse.
    Vector<Pending_Instance> pending;

    auto lowercase = [](String text) {
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        return text;
    };
    Unordered_Map<String, String> lowercase_type_names;
    for (const auto &[type_name, component_type] : component_types) {
        static_cast<void>(component_type);
        lowercase_type_names[lowercase(type_name)] = type_name;
    }

    for (const NML_Node *root : document_roots) {
        const Vector<NML_Node *> &top_level_elements = children_of(root);
        for (usize index = top_level_elements.size(); index > 0; index -= 1) {
            const NML_Node *element = top_level_elements[index - 1];
            if (is_document_scope_directive(element->body.tag_name)) continue;
            pending.push_back({element, nullptr});
        }

        while (!pending.empty()) {
            const Pending_Instance current = pending.back();
            pending.pop_back();

            // <Component type="spike" .../> is the generic spelling of <spike .../>.
            const String &tag_name = current.node->body.tag_name;
            const String type_name = tag_name == "Component" ? current.node->body.get_attribute("type") : tag_name;
            auto type_entry = component_types.find(type_name);
            if (type_entry == component_types.end()) {
                if (tag_name == "Component") {
                    throw Runtime_Error("<Component id=\"" + current.node->body.get_attribute("id") + "\"> names type '" +
                                        type_name + "', which no ComponentType declares");
                }
                // An annotation is skipped, but a ComponentType's name in the wrong case is a typo
                // that would silently drop the element.
                auto case_typo = lowercase_type_names.find(lowercase(tag_name));
                if (case_typo != lowercase_type_names.end()) {
                    throw Runtime_Error("<" + tag_name + "> is no ComponentType; ComponentType names are case "
                                        "sensitive, and this one is '" + case_typo->second + "'");
                }
                log::logger().warn("Ignoring <{}>: no such ComponentType is declared", tag_name);
                continue;
            }

            NML_Component_Instance *instance = create_component_instance(
                    &type_entry->second, current.node, current.parent_instance);

            const Vector<NML_Node *> &child_elements = children_of(current.node);
            for (usize index = child_elements.size(); index > 0; index -= 1) {
                pending.push_back({child_elements[index - 1], instance});
            }
        }
    }
}

const NML_Component_Instance *NML_Context::find_instance(const String &instance_id) const {
    auto entry = component_instances.find(instance_id);
    if (entry == component_instances.end()) return nullptr;
    return &entry->second;
}

bool NML_Context::is_instance_of(const NML_Component_Instance *instance,
                                const String &type_name) const {
    if (!instance) return false;
    for (const NML_Component_Type *type = instance->component_type; type; type = type->extends) {
        if (type->name == type_name) return true;
    }
    return false;
}

s64 NML_Context::resolve_path(
    const String &path, 
    const NML_Component_Instance *current
) const {
    usize start = 0;
    usize end = path.find('/');
    if (end == String::npos) end = path.size();

    const NML_Component_Instance *walk = current == nullptr
        ? find_instance(simulation.target_network_id)
        : current;
    const NML_Component_Instance *population = nullptr;
    s64 component_index = -1;
    // "pop0/3/cellId/v" names the cell's id after an <instance>; "pop0[3]/v" does not.
    bool indexed_by_instance = false;

    while (walk && start < path.size()) {
        const String token = path.substr(start, end - start);

        if (token == "..") {
            walk = walk->parent_instance;
            if (walk == nullptr) return -1;
        } else if (!token.empty() && token != ".") {
            String name = token;
            const usize bracket = token.find('[');
            if (bracket != String::npos) {
                // The index in "pop[3]" must be a whole number closing the segment.
                if (token.back() != ']') return -1;
                const Optional<s64> index = parse_whole_number(token.substr(bracket + 1, token.size() - bracket - 2));
                if (!index.has_value()) return -1;
                component_index = *index;
                name = token.substr(0, bracket);
            }

            // A segment that names no child ends the walk; the rest names a variable.
            const NML_Component_Instance *child = find_instance(walk->id + "/" + name);
            if (child == nullptr) break;

            if (is_instance_of(child, "population")) population = child;
            if (is_instance_of(child, "instance")) {
                // An instance's id is only a name; its place among the population's instances is the index.
                auto position = simulation.instance_positions.find(child->id);
                if (position == simulation.instance_positions.end()) return -1;
                component_index = position->second;
                indexed_by_instance = true;
            }
            walk = child;
        }

        start = end + 1;
        end = path.find('/', start);
        if (end == String::npos) end = path.size();
    }

    if (population == nullptr || component_index < 0) return -1;
    // An index past the end of its population names no cell.
    if (component_index >= get_population_size(population)) return -1;

    const String cell_id = population->value_or("component");
    const Vector<String> &variable_names = find_instance(cell_id)->component_type->state_variable_names;
    const s64 cell_index = simulation.population_base_indices.at(population->id) +
            component_index * cell_slot_count(*find_instance(cell_id)->component_type);

    // TODO: classify each path segment up front so the redundant cell id needs no special case.
  
    // What is left names a variable; "pop0/3/cellId/v" repeats the cell's id first. A lone cell id
    // names the cell, unless the bracket form left it, where it can only be a variable of that name.
    String remainder = start < path.size() ? path.substr(start) : "";
    const bool cell_id_is_a_variable =
            std::find(variable_names.begin(), variable_names.end(), cell_id) != variable_names.end();
    if (remainder.rfind(cell_id + "/", 0) == 0) {
        remainder.erase(0, cell_id.size() + 1);
    } else if (remainder == cell_id && (indexed_by_instance || !cell_id_is_a_variable)) {
        remainder.clear();
    }
    if (remainder.empty()) return cell_index;

    const auto variable = std::find(variable_names.begin(), variable_names.end(), remainder);
    if (variable == variable_names.end()) return -1;
    return cell_index + static_cast<s64>(variable - variable_names.begin());
}

void NML_Context::parse_simulation_details(NML_Node *lems_root) {
    for (const auto &[constant_name, constant_value] : model_constants) {
        simulation.global_constants[constant_name] = constant_value;
    }

    // ── simulation data ──────────────────────────────────────────────────────────────
    // The <Target> component, otherwise the main document's first <Simulation>.
    const NML_Component_Instance *simulation_instance = nullptr;
    if (!target_component_id.empty()) {
        simulation_instance = find_instance(target_component_id);
        if (!simulation_instance) {
            throw Runtime_Error("<Target> names '" + target_component_id +
                                "', which no instance declares");
        }
    } else if (lems_root) {
        for (const NML_Node *node : children_of(lems_root)) {
            if (node->body.tag_name != "Simulation") continue;
            simulation_instance = find_instance(node->body.get_attribute("id"));
            break;
        }
    }

    if (!simulation_instance) {
        log::logger().warn("No Simulation instance found; the context has no step, duration or network");
        return;
    }

    simulation.simulation_component_id = simulation_instance->id;
    simulation.target_network_id = simulation_instance->value_or("target");
    if (simulation_instance->has_value("step")) {
        simulation.step_dt = resolve_attribute(*this, *simulation_instance, "step", "time");
    }
    if (!simulation_instance->has_value("length")) {
        throw Runtime_Error("Simulation '" + simulation_instance->id + "' states no length");
    }
    simulation.simulation_duration = resolve_attribute(*this, *simulation_instance, "length", "time");
    // Every time converts to ticks by dividing by the step.
    if (simulation.step_dt <= 0.0) {
        throw Runtime_Error("Simulation '" + simulation_instance->id + "' has no usable step (parsed " +
                            std::to_string(simulation.step_dt) + " s)");
    }
    simulation.total_tick_count =
            units::seconds_to_ticks(simulation.simulation_duration, simulation.step_dt);

    if (simulation_instance->has_value("seed")) {
        try {
            simulation.random_seed = static_cast<u64>(std::stoull(simulation_instance->value_or("seed")));
        } catch (const Exception &) {
            log::logger().warn("Simulation '{}' has a non-numeric seed '{}'; ignoring",
                               simulation_instance->id, simulation_instance->value_or("seed"));
        }
    }

    // ── network data ──────────────────────────────────────────────────────────────
    const NML_Component_Instance *network = find_instance(simulation.target_network_id);
    if (!network) {
        throw Runtime_Error("Simulation '" + simulation_instance->id + "' targets '" +
                            simulation.target_network_id + "', which no instance declares");
    }

    // ── populations ──────────────────────────────────────────────────────────────
    s64 current_cell_base = 0;
    for (const String &child_id : network->data_order) {
        const NML_Component_Instance *population = find_instance(child_id);
        if (!is_instance_of(population, "population")) continue;

        // A populationList's <instance> children are its cells, in document order, whatever their
        // ids; a plain population states a size.
        s64 instance_position = 0;
        for (const String &member_id : population->data_order) {
            if (!is_instance_of(find_instance(member_id), "instance")) continue;
            simulation.instance_positions[member_id] = instance_position;
            instance_position += 1;
        }
        const s64 neuron_count = get_population_size(population);

        const String component_id = population->value_or("component");
        const NML_Component_Instance *cell = find_instance(component_id);
        if (!cell || !cell->component_type) {
            throw Runtime_Error("Population '" + population->id + "' references component '" +
                                component_id + "', which no instance declares");
        }

        simulation.cell_instances.push_back(*cell);
        simulation.population_base_indices[population->id] = current_cell_base;
        current_cell_base += neuron_count * cell_slot_count(*cell->component_type);
        simulation.total_neuron_count += neuron_count;
    }

    if (simulation.cell_instances.empty()) {
        log::logger().warn("Network '{}' declares no populations", network->id);
        return;
    }

    // ── projections ──────────────────────────────────────────────────────────────
    simulation.network_data = Adjacency_List(simulation.total_neuron_count, 0);
    Set<String> registered_synapse_ids;

    for (const String &child_id : network->data_order) {
        const NML_Component_Instance *projection = find_instance(child_id);
        // Connections come only from a <projection>; these would contribute no edges.
        for (const char *unread_type : {"explicitConnection", "electricalProjection", "continuousProjection"}) {
            if (!is_instance_of(projection, unread_type)) continue;
            throw Runtime_Error("Network '" + network->id + "' holds '" + projection->id + "', a " +
                                projection->component_type->name + ", which the engine does not simulate: it "
                                "takes connections only from a <projection> of event-driven synapses");
        }
        if (!is_instance_of(projection, "projection")) continue;

        const String synapse_id = projection->value_or("synapse");
        const NML_Component_Instance *synapse = find_instance(synapse_id);
        if (!synapse) {
            throw Runtime_Error("Projection '" + projection->id + "' references synapse '" +
                                synapse_id + "', which no instance declares");
        }
        if (registered_synapse_ids.insert(synapse_id).second) {
            simulation.synapse_instances.push_back(*synapse);
        }

        for (const String &connection_id : projection->data_order) {
            const NML_Component_Instance *connection = find_instance(connection_id);
            if (!is_instance_of(connection, "connection") &&
                !is_instance_of(connection, "connectionWD")) {
                continue;
            }

            const String presynaptic_path = connection->value_or("preCellId");
            const String postsynaptic_path = connection->value_or("postCellId");

            NML_Network_Edge edge;
            edge.component_id = synapse_id;
            edge.destination = connection->value_or("destination");
            edge.parent = neuron_index_of(resolve_path(presynaptic_path, projection));
            edge.child = neuron_index_of(resolve_path(postsynaptic_path, projection));

            if (edge.parent < 0 || edge.child < 0) {
                throw Runtime_Error("Connection '" + connection->id + "' in projection '" +
                                    projection->id + "' has an endpoint that resolves to no "
                                    "neuron: pre='" + presynaptic_path + "' post='" +
                                    postsynaptic_path + "'");
            }

            // A plain <connection> states neither; a connectionWD must state both.
            const String described_connection = "Connection '" + connection->id + "' in projection '" + projection->id + "'";
            if (is_instance_of(connection, "connectionWD") &&
                (!connection->has_value("weight") || !connection->has_value("delay"))) {
                throw Runtime_Error(described_connection + " is a connectionWD, which must state both its weight and delay");
            }
            edge.weight = connection->has_value("weight")
                    ? static_cast<f32>(resolve_attribute(*this, *connection, "weight", "none"))
                    : 1.0f;
            const f64 delay_seconds = connection->has_value("delay") ? resolve_attribute(*this, *connection, "delay", "time") : 0.0;
            if (!std::isfinite(delay_seconds) || delay_seconds < 0.0) {
                throw Runtime_Error(described_connection + " has the delay '" + connection->value_or("delay") +
                                    "'; a delay must be finite and not negative");
            }
            edge.delay_ticks = units::seconds_to_ticks(delay_seconds, simulation.step_dt);

            simulation.network_data.add(edge);
            simulation.total_edge_count += 1;
            simulation.maximum_edge_delay = std::max(simulation.maximum_edge_delay, edge.delay_ticks);
            simulation.minimum_edge_delay = simulation.total_edge_count == 1
                    ? edge.delay_ticks : std::min(simulation.minimum_edge_delay, edge.delay_ticks);
        }
    }

    // ── inputs ───────────────────────────────────────────────────────────────────
    // Only the input's id is kept: the kernel runs the input component's own dynamics.
    auto build_input_profile = [&](const String &input_component_id,
                                   Vector<Input_Target> targets) {
        const NML_Component_Instance *input = find_instance(input_component_id);
        if (!input || !input->component_type) {
            throw Runtime_Error("Input wiring references component '" + input_component_id +
                                "', which no instance declares");
        }

        Simulation_Input_Config profile;
        profile.input_component_id = input->id;
        profile.targets = std::move(targets);
        simulation.input_profiles.push_back(std::move(profile));
    };

    for (const String &child_id : network->data_order) {
        const NML_Component_Instance *child = find_instance(child_id);

        if (is_instance_of(child, "explicitInput")) {
            const String target_path = child->value_or("target");

            Input_Target target;
            target.destination = child->value_or("destination");
            target.neuron_index = resolve_path(target_path, network);
            if (target.neuron_index < 0) {
                throw Runtime_Error("explicitInput '" + child->id + "' targets '" +
                                    target_path + "', which resolves to no neuron");
            }

            build_input_profile(child->value_or("input"), {target});
            continue;
        }

        if (!is_instance_of(child, "inputList")) continue;

        Vector<Input_Target> targets;
        for (const String &input_id : child->data_order) {
            const NML_Component_Instance *input = find_instance(input_id);
            if (!is_instance_of(input, "input") && !is_instance_of(input, "inputW")) continue;

            const String target_path = input->value_or("target");

            Input_Target target;
            target.destination = input->value_or("destination");
            target.neuron_index = resolve_path(target_path, child);
            if (target.neuron_index < 0) {
                throw Runtime_Error("Input '" + input->id + "' in inputList '" + child->id +
                                    "' targets '" + target_path +
                                    "', which resolves to no neuron");
            }
            if (input->has_value("weight")) {
                target.weight = resolve_attribute(*this, *input, "weight", "none");
            }
            targets.push_back(std::move(target));
        }

        if (!targets.empty()) build_input_profile(child->value_or("component"), std::move(targets));
    }

    // ── recordings ───────────────────────────────────────────────────────────────
    // A relative fileName is relative to the LEMS file, or to the element's path inside it.
    Path lems_directory;
    for (usize index = 0; index < document_roots.size(); index += 1) {
        if (document_roots[index] == lems_root) lems_directory = Path(document_filepaths[index]).parent_path();
    }

    for (const String &output_id : simulation_instance->data_order) {
        const NML_Component_Instance *output = find_instance(output_id);

        // Display is on-screen only and names no file, so it contributes no profile.
        const bool is_event_file = is_instance_of(output, "EventOutputFile");
        if (!is_event_file && !is_instance_of(output, "OutputFile")) continue;

        const String filename = output->value_or("fileName");
        if (filename.empty()) continue;

        Recording_Config profile{};
        profile.output_filenames.push_back(filename);
        profile.file_output_format.push_back(
                is_event_file ? Output_File_Format::SPIKE_EVENTS
                              : output_format_for_filename(filename));
        profile.recordings_count = 0;

        const Path output_path = output->value_or("path");
        profile.output_directory = (output_path.empty() ? lems_directory
                                    : output_path.is_absolute() ? output_path
                                    : lems_directory / output_path).string();

        if (is_event_file) {
            profile.event_format = output->has_value("format") ? output->value_or("format") : "TIME_ID";
            if (profile.event_format != "TIME_ID" && profile.event_format != "ID_TIME") {
                throw Runtime_Error("EventOutputFile '" + output->id + "' has format '" + profile.event_format +
                                    "'; it must be TIME_ID or ID_TIME");
            }
        }

        for (const String &selection_id : output->data_order) {
            const NML_Component_Instance *selection = find_instance(selection_id);

            Recording_Selection recorded;
            const bool is_column = is_instance_of(selection, "OutputColumn");
            if (is_column) {
                recorded.quantity_path = selection->value_or("quantity");
            } else if (is_instance_of(selection, "EventSelection")) {
                recorded.quantity_path = selection->value_or("select");
                recorded.event_port = selection->value_or("eventPort");
            } else {
                continue;
            }
            recorded.selection_id = selection->id.substr(selection->id.rfind('/') + 1);

            // The engine refuses a selection that resolves to nothing it can record.
            recorded.neuron_index = resolve_path(recorded.quantity_path);
            if (recorded.neuron_index < 0) {
                log::logger().warn("Recording '{}' in {} resolves to no neuron; recorded "
                                   "with index -1", recorded.quantity_path, filename);
            } else if (is_column) {
                recorded.variable_name = variable_name_at(*this, recorded.neuron_index);
            }

            profile.selections.push_back(std::move(recorded));
            profile.recordings_count += 1;
        }

        simulation.recording_profiles.push_back(std::move(profile));
    }
}

// Appends "line N: message" for each schema error to *user_data, joined by " | ".
// libxml2 2.12 changed the error parameter to `const xmlError *`.
// static: utilities.cpp defines an external function of the same name.
#if LIBXML_VERSION >= 21200
static void collect_schema_validation_error(void *user_data, const xmlError *error) {
#else
static void collect_schema_validation_error(void *user_data, xmlErrorPtr error) {
#endif
    if (!error || !error->message) return;

    String message(error->message);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();

    auto *destination = static_cast<String *>(user_data);
    if (!destination->empty()) *destination += " | ";
    *destination += "line " + std::to_string(error->line) + ": " + message;
}

bool NML_Context::validate_lems_schema(const String &lems_filepath) {
    xmlDocPtr document = xmlReadFile(lems_filepath.c_str(), nullptr, 0);
    if (!document) {
        last_schema_validation_errors = "could not read " + lems_filepath;
        return false;
    }

    xmlNodePtr root = xmlDocGetRootElement(document);
    const bool is_neuroml_document =
            root && root->name &&
            String(reinterpret_cast<const char *>(root->name)) == "neuroml";
    xmlFreeDoc(document);

    if (!is_neuroml_document) {
        log::logger().debug("{} is not a NeuroML document root; skipping XSD validation",
                            lems_filepath);
        last_schema_validation_errors.clear();
        return true;
    }

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

    last_schema_validation_errors.clear();
    xmlSchemaSetValidStructuredErrors(
            valid_context, collect_schema_validation_error, &last_schema_validation_errors);

    // 0 = valid, > 0 = schema errors (collected above), < 0 = internal error.
    int result = xmlSchemaValidateFile(valid_context, lems_filepath.c_str(), 0);

    xmlSchemaFreeValidCtxt(valid_context);
    xmlSchemaFree(schema);

    return result == 0;
}

xmlNodePtr NML_Context::get_xml_root(const String &filepath) {
    xmlDocPtr document = xmlReadFile(filepath.c_str(), nullptr, XML_PARSE_NOBLANKS);
    if (!document) {
        log::logger().error("Could not parse NML/LEMS file {}", filepath);
        return nullptr;
    }

    xmlNodePtr root = xmlDocGetRootElement(document);
    if (!root) {
        log::logger().error("NML/LEMS file {} has no root element", filepath);
        xmlFreeDoc(document);
        return nullptr;
    }

    return root;
}

NML_Context::~NML_Context() {
    for (NML_Node *root : document_roots) delete root;
}

// Splits "-60 mV" into its number and unit symbol. False unless the whole text is one finite
// number, optionally followed by a symbol: "-60mV;", "3 mV extra", "1.2.3mV", "abc" and "1e400"
// are all refused.
static bool split_strict_quantity(const String &value, f64 &magnitude, String &suffix) {
    usize cursor = value.find_first_not_of(" \t\n\r");
    if (cursor == String::npos) return false;
    const usize number_start = cursor;
    auto is_digit_at = [&value](usize position) {
        return position < value.size() && isdigit(static_cast<unsigned char>(value[position]));
    };

    if (value[cursor] == '-' || value[cursor] == '+') cursor += 1;
    usize digit_count = 0;
    while (is_digit_at(cursor)) { cursor += 1; digit_count += 1; }
    if (cursor < value.size() && value[cursor] == '.') {
        cursor += 1;
        while (is_digit_at(cursor)) { cursor += 1; digit_count += 1; }
    }
    if (digit_count == 0) return false;
    if (cursor < value.size() && (value[cursor] == 'e' || value[cursor] == 'E')) {
        usize exponent = cursor + 1;
        if (exponent < value.size() && (value[exponent] == '-' || value[exponent] == '+')) exponent += 1;
        if (is_digit_at(exponent)) {
            cursor = exponent;
            while (is_digit_at(cursor)) cursor += 1;
        }
    }

    try {
        magnitude = std::stod(value.substr(number_start, cursor - number_start));
    } catch (const Exception &) {
        return false;
    }
    if (!std::isfinite(magnitude)) return false;

    const usize suffix_start = value.find_first_not_of(" \t\n\r", cursor);
    const usize suffix_end = value.find_last_not_of(" \t\n\r");
    suffix = suffix_start == String::npos ? "" : value.substr(suffix_start, suffix_end - suffix_start + 1);
    if (suffix.empty()) return true;
    if (!isalpha(static_cast<unsigned char>(suffix.front())) && suffix.front() != '_') return false;
    return std::all_of(suffix.begin(), suffix.end(), [](char character) {
        return isalnum(static_cast<unsigned char>(character)) || character == '_';
    });
}

f64 NML_Context::resolve_quantity(const String &value) const {
    f64 magnitude = 0.0;
    String suffix;
    if (!split_strict_quantity(value, magnitude, suffix)) {
        throw Runtime_Error("'" + value + "' is not a quantity: a quantity is a finite number, optionally followed by a unit");
    }
    if (suffix.empty()) return magnitude;

    // A declared <Unit> wins over the built-in table; only it can carry an offset (degC -> K).
    auto declared = model_units.find(suffix);
    if (declared != model_units.end()) {
        return magnitude * declared->second.scale + declared->second.offset;
    }
    if (!units::unit_suffix_is_known(suffix)) {
        throw Runtime_Error("'" + value + "' has the unit '" + suffix + "', which neither the document nor the "
                            "NeuroML standard units declare");
    }

    return magnitude * units::unit_suffix_scale(suffix);
}

f64 NML_Context::resolve_quantity_of_dimension(const String &value, const String &dimension) const {
    const f64 resolved = resolve_quantity(value);
    // "*" accepts any dimension; with no dimension stated there is nothing to check.
    if (dimension.empty() || dimension == "*") return resolved;

    f64 magnitude = 0.0;
    String suffix;
    split_strict_quantity(value, magnitude, suffix);
    if (suffix.empty()) {
        if (dimension == "none") return resolved;
        throw Runtime_Error("'" + value + "' has no unit, but it is a " + dimension);
    }
    auto declared = model_units.find(suffix);
    if (declared != model_units.end() && !declared->second.dimension.empty() && declared->second.dimension != dimension) {
        throw Runtime_Error("'" + value + "' is a " + declared->second.dimension + ", but it should be a " + dimension);
    }
    return resolved;
}

s64 NML_Context::cell_slot_count(const NML_Component_Type &cell_type) {
    return std::max<s64>(1, static_cast<s64>(cell_type.state_variable_names.size()));
}

// The number of state variables a cell declares; cell_slot_count gives the slots it takes.
s64 NML_Context::get_cell_variable_count(s64 cell_instance_index) {
    const NML_Component_Type *cell_type =
            simulation.cell_instances.at(static_cast<usize>(cell_instance_index)).component_type;
    return static_cast<s64>(cell_type->state_variable_names.size());
}

s64 NML_Context::neuron_index_of(s64 cell_memory_index) const {
    s64 first_neuron = 0;
    for (const NML_Component_Instance *population : network_populations(*this)) {
        const s64 population_size = get_population_size(population);
        const s64 slot_count = cell_slot_count(*population_cell(*this, *population).component_type);
        const s64 base = simulation.population_base_indices.at(population->id);

        if (cell_memory_index >= base && cell_memory_index < base + population_size * slot_count) {
            return first_neuron + (cell_memory_index - base) / slot_count;
        }
        first_neuron += population_size;
    }
    return -1;
}

// A populationList's <instance> children are its cells; a plain population states a size.
s64 NML_Context::get_population_size(const NML_Component_Instance *population) const {
    if (population == nullptr) throw Invalid_Argument("get_population_size: no population was given");
    s64 population_size = 0;
    for (const String &member_id : population->data_order) {
        if (is_instance_of(find_instance(member_id), "instance")) population_size += 1;
    }
    if (population_size == 0 && population->has_value("size")) {
        const Optional<s64> size = parse_whole_number(population->value_or("size"));
        if (!size.has_value()) {
            throw Runtime_Error("Population '" + population->id + "' has the size '" + population->value_or("size") +
                                "', which is not a whole number of cells");
        }
        population_size = *size;
    }
    return population_size;
}

// The number of values in cell memory: every state variable of every cell, over all populations.
s64 NML_Context::get_cell_state_size() const {
    const NML_Component_Instance *network = find_instance(simulation.target_network_id);
    if (!network) return 0;

    s64 cell_state_size = 0;
    for (const String &child_id : network->data_order) {
        const NML_Component_Instance *population = find_instance(child_id);
        if (!is_instance_of(population, "population")) continue;

        const NML_Component_Instance *cell = find_instance(population->value_or("component"));
        if (cell == nullptr || cell->component_type == nullptr) continue;
        cell_state_size += get_population_size(population) * cell_slot_count(*cell->component_type);
    }
    return cell_state_size;
}









}











