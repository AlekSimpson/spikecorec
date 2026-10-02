#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemas.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>

#include "spikecorec/core/log.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

#include "spikecorec/nml/parser.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/nml/components.h"
#include "spikecorec/nml/dynamics.h"
#include "spikecorec/nml/utilities.h"

using namespace spikecorec;
using namespace std;
using namespace std::filesystem;


namespace spikecorec::nml {

bool is_document_scope_directive(const String &tag_name) {
    static const Set<String> directives = {
        "Include", "include", "ComponentType", "Unit", "Dimension", "Constant", "Target"
    };
    return directives.count(tag_name) != 0;
}

// One entry per StateAssignment, EventOut and Transition inside an event handler.
void collect_event_actions(const NML_Node *handler, const String &regime_name, const String &condition,
                           Vector<NML_DynamicsExpression> &dynamics) {
    for (const NML_Node *action : children_of(handler)) {
        const NML_Tag &tag = action->body;
        NML_DynamicsExpression entry(tag.tag_type);
        entry.regime_name = regime_name;
        entry.condition = condition;

        if (tag.tag_type == NML_DeclarationType::StateAssignment) {
            entry.target = tag.get_attribute("variable");
            entry.expression = tag.get_attribute("value");
        } else if (tag.tag_type == NML_DeclarationType::EventOut) {
            entry.target = tag.get_attribute("port");
        } else if (tag.tag_type == NML_DeclarationType::Transition) {
            entry.target = tag.get_attribute("regime");
        } else {
            continue;
        }
        dynamics.push_back(std::move(entry));
    }
}

// Flattens a <Dynamics> or <Regime> element into dynamics entries, in document order.
// OnStart assignments are entries of their own (source_tag OnStart); OnCondition, OnEntry
// and OnEvent are followed by the entries of their actions.
void collect_dynamics(const NML_Node *element, const String &regime_name,
                      Vector<NML_DynamicsExpression> &dynamics) {
    for (const NML_Node *child : children_of(element)) {
        const NML_Tag &tag = child->body;
        NML_DynamicsExpression entry(tag.tag_type);
        entry.regime_name = regime_name;

        switch (tag.tag_type) {
            case NML_DeclarationType::DerivedVariable:
                entry.target = tag.get_attribute("name");
                entry.expression = tag.get_attribute("value");
                entry.select = tag.get_attribute("select");
                entry.reduce = tag.get_attribute("reduce");
                dynamics.push_back(std::move(entry));
                break;

            case NML_DeclarationType::ConditionalDerivedVariable:
                for (const NML_Node *case_node : children_of(child)) {
                    if (case_node->body.tag_type != NML_DeclarationType::Case) continue;
                    NML_DynamicsExpression case_entry(NML_DeclarationType::Case);
                    case_entry.regime_name = regime_name;
                    case_entry.target = tag.get_attribute("name");
                    case_entry.condition = case_node->body.get_attribute("condition");
                    case_entry.expression = case_node->body.get_attribute("value");
                    dynamics.push_back(std::move(case_entry));
                }
                break;

            case NML_DeclarationType::TimeDerivative:
                entry.target = tag.get_attribute("variable");
                entry.expression = tag.get_attribute("value");
                dynamics.push_back(std::move(entry));
                break;

            case NML_DeclarationType::OnStart:
                for (const NML_Node *assignment : children_of(child)) {
                    if (assignment->body.tag_type != NML_DeclarationType::StateAssignment) continue;
                    NML_DynamicsExpression start_entry(NML_DeclarationType::OnStart);
                    start_entry.target = assignment->body.get_attribute("variable");
                    start_entry.expression = assignment->body.get_attribute("value");
                    dynamics.push_back(std::move(start_entry));
                }
                break;

            case NML_DeclarationType::OnCondition:
                entry.expression = tag.get_attribute("test");
                dynamics.push_back(entry);
                collect_event_actions(child, regime_name, entry.expression, dynamics);
                break;

            case NML_DeclarationType::OnEntry:
                dynamics.push_back(entry);
                collect_event_actions(child, regime_name, "", dynamics);
                break;

            case NML_DeclarationType::OnEvent:
                entry.target = tag.get_attribute("port");
                dynamics.push_back(entry);
                collect_event_actions(child, regime_name, "", dynamics);
                break;

            case NML_DeclarationType::Regime:
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

NML_Node *NML_Context::parse_neuroml(const String &filepath) {
    xmlNodePtr xml_root = get_xml_root(filepath);
    if (!xml_root) return nullptr;

    NML_Node *root = build_nml_tree(xml_root);
    xmlFreeDoc(xml_root->doc);
    return root;
}

NML_Node *NML_Context::parse_neuroml_file(const String &filepath, Vector<String> &files_to_parse) {
    std::error_code path_error;
    path canonical = weakly_canonical(path(filepath), path_error);
    String canonical_text = path_error ? filepath : canonical.string();

    NML_Node *root = parse_neuroml(filepath);
    if (!root) return nullptr;

    path containing_directory = path(canonical_text).parent_path();

    for (const NML_Node *child : children_of(root)) {
        const NML_Tag &tag = child->body;

        if (tag.tag_name != "Include" && tag.tag_name != "include") continue;

        String reference = tag.has_attribute("file")
                ? tag.get_attribute("file")
                : tag.get_attribute("href");
        if (reference.empty()) continue;

        path included = path(reference);
        if (included.is_relative()) {
            included = containing_directory / included;
        }

        if (!exists(included) && !STANDARD_LIBRARY_PATH.empty()) {
            const path from_standard_library = path(STANDARD_LIBRARY_PATH) / path(reference).filename();
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
    simulation = NML_SimulationContext();
}

void NML_Context::parse(const String &main_filepath) {
    if (STANDARD_LIBRARY_PATH.empty() || !exists(STANDARD_LIBRARY_PATH)) {
        log::logger().error("NML standard library path does not exist: {}",
                            STANDARD_LIBRARY_PATH);
        return;
    }
    reset();

    Vector<String> files_to_parse;
    for (const auto &entry : directory_iterator(STANDARD_LIBRARY_PATH)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".xml" && entry.path().extension() != ".nml") continue;

        files_to_parse.push_back(entry.path().string());
    }

    std::sort(files_to_parse.begin(), files_to_parse.end());

    files_to_parse.push_back(main_filepath);

    Set<String> parsed_files;

    std::error_code main_path_error;
    const path canonical_main = weakly_canonical(path(main_filepath), main_path_error);
    const String canonical_main_file = main_path_error ? main_filepath : canonical_main.string();

    // Resolved after every document is read: a <Constant> may be written in a <Unit> that
    // a later file in the queue, such as one its own document includes, declares.
    Vector<const NML_Node *> constant_nodes;

    String current_file;
    while (files_to_parse.size() > 0) {
        current_file = files_to_parse.front();

        std::error_code path_error;
        const path canonical = weakly_canonical(path(current_file), path_error);
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

                // LEMS: si = raw * scale * 10^power + offset. power is folded into scale here
                // so resolution stays a single multiply-add.
                units::UnitDefinition definition;
                definition.scale = 1.0;
                definition.offset = 0.0;

                if (tag.has_attribute("scale")) {
                    auto [scale_magnitude, scale_suffix] = units::split_quantity(tag.get_attribute("scale"));
                    (void)scale_suffix;
                    if (scale_magnitude != 0.0) definition.scale = scale_magnitude;
                }
                if (tag.has_attribute("power")) {
                    auto [power_magnitude, power_suffix] = units::split_quantity(tag.get_attribute("power"));
                    (void)power_suffix;
                    definition.scale *= std::pow(10.0, power_magnitude);
                }
                if (tag.has_attribute("offset")) {
                    auto [offset_magnitude, offset_suffix] =
                            units::split_quantity(tag.get_attribute("offset"));
                    (void)offset_suffix;
                    definition.offset = offset_magnitude;
                }

                model_units[symbol] = definition;
                continue;
            }

            if (tag.tag_name == "Constant") {
                constant_nodes.push_back(child);
                continue;
            }

            if (tag.tag_name == "Target") {
                target_component_id = tag.get_attribute("component");
                continue;
            }
        }

        document_roots.push_back(root);
        document_filepaths.push_back(canonical_file);
        if (canonical_file == canonical_main_file) {
            main_document_root = root;
        }
    }

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
}

// Records the <ComponentType> element type_node, read from source_filepath, in
// component_types and returns its entry. Its extends link is left unset because the parent
// may not be recorded yet; parse_component_types links every type once all are recorded.
// nullptr when the element has no name.
NML_ComponentType *NML_Context::create_component_type(
        const NML_Node *type_node, const String &source_filepath) {
    const String type_name = type_node->body.get_attribute("name");
    if (type_name.empty()) {
        log::logger().warn("Ignoring <ComponentType> with no name in {}", source_filepath);
        return nullptr;
    }

    auto existing = component_types.find(type_name);
    if (existing != component_types.end()) {
        throw runtime_error(
                "Duplicate ComponentType '" + type_name + "' declared in both " +
                existing->second.source_file + " and " + source_filepath);
    }

    NML_ComponentType component_type(type_name);
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
        (void)component_type;
        type_names.push_back(type_name);
    }
    std::sort(type_names.begin(), type_names.end());

    // a parent need not be linked before its children: each link is only an address, and
    // component_types never moves an element once it is inserted.
    for (const String &type_name : type_names) {
        NML_ComponentType &component_type = component_types.at(type_name);

        const String parent_name = component_type.source_node->body.get_attribute("extends");
        if (parent_name.empty()) continue;

        auto parent = component_types.find(parent_name);
        if (parent == component_types.end()) {
            throw runtime_error(
                    "Unresolved ComponentType '" + parent_name +
                    "' referenced from extends chain: " + type_name + " -> " + parent_name);
        }

        component_type.extends = &parent->second;
    }

    // a cycle would send every walk up the chain, find_declaration's included, around it
    // forever, so each chain is walked once here to its root.
    for (const String &type_name : type_names) {
        String chain;
        Set<String> visited_type_names;

        for (const NML_ComponentType *type = &component_types.at(type_name); type;
             type = type->extends) {
            if (!chain.empty()) chain += " -> ";
            chain += type->name;

            if (!visited_type_names.insert(type->name).second) {
                throw runtime_error("Cyclic ComponentType extends chain: " + chain);
            }
        }
    }

    // Ancestors' state variables come first, so a subtype keeps its parent's slot order.
    for (const String &type_name : type_names) {
        NML_ComponentType &component_type = component_types.at(type_name);

        Vector<const NML_ComponentType *> chain;
        for (const NML_ComponentType *type = &component_type; type; type = type->extends) {
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

                if (node->body.tag_type != NML_DeclarationType::StateVariable) continue;

                // A subtype that redeclares the name as something else shadows it.
                const NML_Node *declaration =
                        component_type.find_declaration(node->body.namespace_key());
                if (!declaration || declaration->body.tag_type != NML_DeclarationType::StateVariable) {
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
        NML_ComponentType &component_type = component_types.at(type_name);

        const NML_Node *dynamics_node = nullptr;
        for (const NML_ComponentType *type = &component_type; type && !dynamics_node; type = type->extends) {
            for (const NML_Node *child : children_of(type->source_node)) {
                if (child->body.tag_name != "Dynamics") continue;
                dynamics_node = child;
                break;
            }
        }

        component_type.dynamics.clear();
        if (dynamics_node) collect_dynamics(dynamics_node, "", component_type.dynamics);
    }
}

// Creates the instance that instance_node declares, keyed by its full path ("net1/pop0/0").
// parent_instance is nullptr at document scope.
NML_ComponentInstance *NML_Context::create_component_instance(
        const NML_ComponentType *component_type, const NML_Node *instance_node,
        NML_ComponentInstance *parent_instance) {
    static const Set<NML_DeclarationType> instance_bound_declarations = {
        NML_DeclarationType::Parameter,
        NML_DeclarationType::Property,
        NML_DeclarationType::Text,
        NML_DeclarationType::Path,
        NML_DeclarationType::ComponentReference
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
        throw runtime_error("Duplicate component instance '" + instance_id + "'");
    }

    NML_ComponentInstance &instance = component_instances[instance_id];
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
    for (const NML_ComponentType *type = component_type; type; type = type->extends) {
        for (const NML_Node *type_element : children_of(type->source_node)) {
            if (type_element->body.tag_type != NML_DeclarationType::Fixed) continue;

            const String parameter_name = type_element->body.get_attribute("parameter");
            if (parameter_name.empty()) continue;
            if (!pinned_parameters.insert(parameter_name).second) continue;

            const NML_Node *pinned = component_type->find_declaration("var:" + parameter_name);
            if (!pinned || pinned->body.tag_type != NML_DeclarationType::Parameter) continue;

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

    struct PendingInstance {
        const NML_Node *node;
        NML_ComponentInstance *parent_instance;
    };

    // Depth first in document order, so elements are pushed in reverse.
    Vector<PendingInstance> pending;

    for (const NML_Node *root : document_roots) {
        const Vector<NML_Node *> &top_level_elements = children_of(root);
        for (usize index = top_level_elements.size(); index > 0; index -= 1) {
            const NML_Node *element = top_level_elements[index - 1];
            if (is_document_scope_directive(element->body.tag_name)) continue;
            pending.push_back({element, nullptr});
        }

        while (!pending.empty()) {
            const PendingInstance current = pending.back();
            pending.pop_back();

            const String &tag_name = current.node->body.tag_name;
            auto type_entry = component_types.find(tag_name);
            if (type_entry == component_types.end()) {
                log::logger().warn("Ignoring <{}>: no such ComponentType is declared", tag_name);
                continue;
            }

            NML_ComponentInstance *instance = create_component_instance(
                    &type_entry->second, current.node, current.parent_instance);

            const Vector<NML_Node *> &child_elements = children_of(current.node);
            for (usize index = child_elements.size(); index > 0; index -= 1) {
                pending.push_back({child_elements[index - 1], instance});
            }
        }
    }
}

const NML_ComponentInstance *NML_Context::find_instance(const String &instance_id) const {
    auto entry = component_instances.find(instance_id);
    if (entry == component_instances.end()) return nullptr;
    return &entry->second;
}

bool NML_Context::is_instance_of(const NML_ComponentInstance *instance,
                                const String &type_name) const {
    return instance && instance->component_type && instance->component_type->name == type_name;
}

s64 NML_Context::resolve_path(
    const String &path, 
    const NML_ComponentInstance *current
) const {
    usize start = 0;
    usize end = path.find('/');
    if (end == String::npos) end = path.size();

    const NML_ComponentInstance *walk = current == nullptr
        ? find_instance(simulation.target_network_id)
        : current;
    const NML_ComponentInstance *population = nullptr;
    s64 component_index = -1;

    while (walk && start < path.size()) {
        const String token = path.substr(start, end - start);

        if (token == "..") {
            walk = walk->parent_instance;
            if (walk == nullptr) return -1;
        } else if (!token.empty() && token != ".") {
            String name = token;
            const usize bracket = token.find('[');
            if (bracket != String::npos) {
                component_index = stoll(token.substr(bracket + 1));
                name = token.substr(0, bracket);
            }

            // A segment that names no child ends the walk; the rest names a variable.
            const NML_ComponentInstance *child = find_instance(walk->id + "/" + name);
            if (child == nullptr) break;

            if (is_instance_of(child, "population")) population = child;
            if (is_instance_of(child, "instance")) component_index = stoll(name);
            walk = child;
        }

        start = end + 1;
        end = path.find('/', start);
        if (end == String::npos) end = path.size();
    }

    if (population == nullptr || component_index < 0) return -1;

    const String cell_id = population->value_or("component");
    const Vector<String> &variable_names = find_instance(cell_id)->component_type->state_variable_names;
    const s64 cell_index = simulation.population_base_indices.at(population->id) +
            component_index * static_cast<s64>(variable_names.size());

    // TODO: we dont have to do this case handling if we could identify ahead of time what kind of token each token is 
    // because for things like pop0/3/cellId/v, cellId is redundant
  
    // What is left names a variable; "pop0/3/cellId/v" repeats the cell's id first.
    String remainder = start < path.size() ? path.substr(start) : "";
    if (remainder == cell_id || remainder.rfind(cell_id + "/", 0) == 0) {
        remainder.erase(0, cell_id.size() + 1);
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
    // get target component for simulation and capture simulation data
    const NML_ComponentInstance *simulation_instance = nullptr;
    if (!target_component_id.empty()) {
        simulation_instance = find_instance(target_component_id);
        if (!simulation_instance) {
            throw runtime_error("<Target> names '" + target_component_id +
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
        simulation.step_dt = resolve_quantity(simulation_instance->value_or("step"));
    }
    if (simulation_instance->has_value("length")) {
        simulation.simulation_duration = resolve_quantity(simulation_instance->value_or("length"));
    }
    // Every time in the model is converted to ticks with the step, and the conversion
    // divides by it.
    if (simulation.step_dt <= 0.0) {
        throw runtime_error("Simulation '" + simulation_instance->id + "' has no usable step (parsed " +
                            std::to_string(simulation.step_dt) + " s)");
    }
    simulation.total_tick_count =
            units::seconds_to_ticks(simulation.simulation_duration, simulation.step_dt);

    if (simulation_instance->has_value("seed")) {
        try {
            simulation.random_seed = static_cast<u64>(std::stoull(simulation_instance->value_or("seed")));
        } catch (const std::exception &) {
            log::logger().warn("Simulation '{}' has a non-numeric seed '{}'; ignoring",
                               simulation_instance->id, simulation_instance->value_or("seed"));
        }
    }

    // ── network data ──────────────────────────────────────────────────────────────
    const NML_ComponentInstance *network = find_instance(simulation.target_network_id);
    if (!network) {
        throw runtime_error("Simulation '" + simulation_instance->id + "' targets '" +
                            simulation.target_network_id + "', which no instance declares");
    }

    // ── populations ──────────────────────────────────────────────────────────────
    s64 current_cell_base = 0;
    for (const String &child_id : network->data_order) {
        const NML_ComponentInstance *population = find_instance(child_id);
        if (!is_instance_of(population, "population")) continue;

        // A populationList enumerates its members as <instance> children, and that count
        // is authoritative; a plain population states a size.
        s64 neuron_count = 0;
        for (const String &member_id : population->data_order) {
            if (!is_instance_of(find_instance(member_id), "instance")) continue;
            neuron_count += 1;
        }
        if (neuron_count == 0 && population->has_value("size")) {
            neuron_count = static_cast<s64>(
                    std::llround(resolve_quantity(population->value_or("size"))));
        }
        if (neuron_count < 0) {
            throw runtime_error("Population '" + population->id + "' has a negative size");
        }

        const String component_id = population->value_or("component");
        const NML_ComponentInstance *cell = find_instance(component_id);
        if (!cell || !cell->component_type) {
            throw runtime_error("Population '" + population->id + "' references component '" +
                                component_id + "', which no instance declares");
        }

        simulation.cell_instances.push_back(*cell);
        s64 cell_state_size = cell->component_type->state_variable_names.size();
        simulation.population_base_indices[population->id] = current_cell_base; 
        current_cell_base += neuron_count * cell_state_size;
        simulation.total_neuron_count += neuron_count;
    }

    if (simulation.cell_instances.empty()) {
        log::logger().warn("Network '{}' declares no populations", network->id);
        return;
    }

    // ── projections ──────────────────────────────────────────────────────────────
    simulation.network_data = AdjacencyList(simulation.total_neuron_count, 0);
    Set<String> registered_synapse_ids;

    for (const String &child_id : network->data_order) {
        const NML_ComponentInstance *projection = find_instance(child_id);
        if (!is_instance_of(projection, "projection")) continue;

        const String synapse_id = projection->value_or("synapse");
        const NML_ComponentInstance *synapse = find_instance(synapse_id);
        if (!synapse) {
            throw runtime_error("Projection '" + projection->id + "' references synapse '" +
                                synapse_id + "', which no instance declares");
        }
        if (registered_synapse_ids.insert(synapse_id).second) {
            simulation.synapse_instances.push_back(*synapse);
        }

        for (const String &connection_id : projection->data_order) {
            const NML_ComponentInstance *connection = find_instance(connection_id);
            if (!is_instance_of(connection, "connection") &&
                !is_instance_of(connection, "connectionWD")) {
                continue;
            }

            const String presynaptic_path = connection->value_or("preCellId");
            const String postsynaptic_path = connection->value_or("postCellId");

            NML_NetworkEdge edge;
            edge.component_id = synapse_id;
            edge.parent = neuron_index_of(resolve_path(presynaptic_path, projection));
            edge.child = neuron_index_of(resolve_path(postsynaptic_path, projection));

            if (edge.parent < 0 || edge.child < 0) {
                throw runtime_error("Connection '" + connection->id + "' in projection '" +
                                    projection->id + "' has an endpoint that resolves to no "
                                    "neuron: pre='" + presynaptic_path + "' post='" +
                                    postsynaptic_path + "'");
            }

            // A plain <connection> states neither; a connectionWD states both.
            edge.weight = connection->has_value("weight")
                    ? static_cast<f32>(resolve_quantity(connection->value_or("weight")))
                    : 1.0f;
            edge.delay_ticks = connection->has_value("delay")
                    ? units::seconds_to_ticks(resolve_quantity(connection->value_or("delay")),
                                              simulation.step_dt)
                    : 0;

            simulation.network_data.add(edge);
            simulation.total_edge_count += 1;
            simulation.maximum_edge_delay = std::max(simulation.maximum_edge_delay, edge.delay_ticks);
        }
    }

    // ── inputs ───────────────────────────────────────────────────────────────────
    // The wiring (explicitInput, inputList) names the targets; the input component it
    // references carries the amplitude, the timing and any spike train.
    auto build_input_profile = [&](const String &input_component_id,
                                   Vector<InputTarget> targets) {
        const NML_ComponentInstance *input = find_instance(input_component_id);
        if (!input) {
            throw runtime_error("Input wiring references component '" + input_component_id +
                                "', which no instance declares");
        }

        SimulationInputConfig profile;
        profile.input_component_id = input->id;

        if (input->has_value("amplitude")) {
            profile.amplitude = resolve_quantity(input->value_or("amplitude"));
        }
        if (input->has_value("rate")) {
            profile.rate = resolve_quantity(input->value_or("rate"));
        }
        if (input->has_value("delay")) {
            profile.start_tick = units::seconds_to_ticks(
                    resolve_quantity(input->value_or("delay")), simulation.step_dt);
        }
        if (input->has_value("duration")) {
            profile.end_tick = profile.start_tick + units::seconds_to_ticks(
                    resolve_quantity(input->value_or("duration")), simulation.step_dt);
        }

        // A spikeArray carries its train as <spike time="..."/> children; every target
        // receives the same train.
        Vector<s32> spike_ticks;
        for (const String &spike_id : input->data_order) {
            const NML_ComponentInstance *spike = find_instance(spike_id);
            if (!is_instance_of(spike, "spike") || !spike->has_value("time")) continue;

            spike_ticks.push_back(static_cast<s32>(
                    units::seconds_to_ticks(resolve_quantity(spike->value_or("time")),
                                            simulation.step_dt)));
        }
        std::sort(spike_ticks.begin(), spike_ticks.end());
        for (InputTarget &target : targets) target.event_ticks = spike_ticks;

        // A component carrying a spike train is an event source whatever else it declares;
        // only one with no train and an amplitude injects current continuously.
        profile.continuous_current_injection =
                spike_ticks.empty() && input->has_value("amplitude");

        profile.targets = std::move(targets);
        simulation.input_profiles.push_back(std::move(profile));
    };

    for (const String &child_id : network->data_order) {
        const NML_ComponentInstance *child = find_instance(child_id);

        if (is_instance_of(child, "explicitInput")) {
            const String target_path = child->value_or("target");

            InputTarget target;
            target.neuron_index = resolve_path(target_path, network);
            if (target.neuron_index < 0) {
                throw runtime_error("explicitInput '" + child->id + "' targets '" +
                                    target_path + "', which resolves to no neuron");
            }

            build_input_profile(child->value_or("input"), {target});
            continue;
        }

        if (!is_instance_of(child, "inputList")) continue;

        Vector<InputTarget> targets;
        for (const String &input_id : child->data_order) {
            const NML_ComponentInstance *input = find_instance(input_id);
            if (!is_instance_of(input, "input") && !is_instance_of(input, "inputW")) continue;

            const String target_path = input->value_or("target");

            InputTarget target;
            target.neuron_index = resolve_path(target_path, child);
            if (target.neuron_index < 0) {
                throw runtime_error("Input '" + input->id + "' in inputList '" + child->id +
                                    "' targets '" + target_path +
                                    "', which resolves to no neuron");
            }
            if (input->has_value("weight")) {
                target.weight = resolve_quantity(input->value_or("weight"));
            }
            targets.push_back(std::move(target));
        }

        if (!targets.empty()) build_input_profile(child->value_or("component"), std::move(targets));
    }

    // ── recordings ───────────────────────────────────────────────────────────────
    for (const String &output_id : simulation_instance->data_order) {
        const NML_ComponentInstance *output = find_instance(output_id);

        // Display is on-screen only and names no file, so it contributes no profile.
        const bool is_event_file = is_instance_of(output, "EventOutputFile");
        if (!is_event_file && !is_instance_of(output, "OutputFile")) continue;

        const String filename = output->value_or("fileName");
        if (filename.empty()) continue;

        RecordingConfig profile{};
        profile.output_filenames.push_back(filename);
        profile.file_output_format.push_back(
                is_event_file ? OutputFileFormat::SPIKE_EVENTS
                              : output_format_for_filename(filename));
        profile.recordings_count = 0;

        for (const String &selection_id : output->data_order) {
            const NML_ComponentInstance *selection = find_instance(selection_id);

            RecordingSelection recorded;
            if (is_instance_of(selection, "OutputColumn")) {
                recorded.quantity_path = selection->value_or("quantity");
            } else if (is_instance_of(selection, "EventSelection")) {
                recorded.quantity_path = selection->value_or("select");
                recorded.event_port = selection->value_or("eventPort");
            } else {
                continue;
            }

            recorded.neuron_index = resolve_path(recorded.quantity_path);

            if (recorded.neuron_index < 0) {
                log::logger().warn("Recording '{}' in {} resolves to no neuron; recorded "
                                   "with index -1", recorded.quantity_path, filename);
            }

            profile.selections.push_back(std::move(recorded));
            profile.recordings_count += 1;
        }

        simulation.recording_profiles.push_back(std::move(profile));
    }
}

// Accumulates each schema validation error's line number and message (libxml2's messages
// already name the offending element, e.g. "Element 'thisTagDoesNotExistInSchema': No
// matching global declaration available for the validation root.") into `*user_data`,
// joined by " | " -- see validate_against_schema's own comment for why this replaces
// libxml2's default, stderr-only error handler.
// libxml2 2.12 made xmlStructuredErrorFunc take a `const xmlError *`; before that it took
// a mutable xmlErrorPtr. This machine has both 2.9 (pkg-config, what the Makefile picks)
// and 2.13 (xml2-config) installed, so the signature is selected rather than assumed.
// static: utilities.cpp defines a function of the same name, and two external definitions
// would collide at link time.
#if LIBXML_VERSION >= 21200
static void collect_schema_validation_error(void *user_data, const xmlError *error) {
#else
static void collect_schema_validation_error(void *user_data, xmlErrorPtr error) {
#endif
    if (!error || !error->message) return;

    String message(error->message);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();

    String *destination = static_cast<String *>(user_data);
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

    // xmlSchemaValidateFile: 0 = valid, >0 = validation errors (captured above, by element and line
    // number), <0 = internal/API error.
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

f64 NML_Context::resolve_quantity(const String &value) const {
    auto [magnitude, suffix] = units::split_quantity(value);
    if (suffix.empty()) return magnitude;

    // A <Unit> the document declared wins: it is authoritative for this model, and it is
    // the only source that can express an offset (degC -> K).
    auto declared = model_units.find(suffix);
    if (declared != model_units.end()) {
        return magnitude * declared->second.scale + declared->second.offset;
    }

    return magnitude * units::unit_suffix_scale(suffix);
}

// The number of per-neuron state slots a cell needs.
s64 NML_Context::get_cell_variable_count(s64 cell_instance_index) {
    const NML_ComponentType *cell_type =
            simulation.cell_instances.at(static_cast<usize>(cell_instance_index)).component_type;
    return static_cast<s64>(cell_type->state_variable_names.size());
}

// A populationList's <instance> children are its cells; a plain population states a size.
s64 NML_Context::neuron_index_of(s64 cell_memory_index) const {
    s64 first_neuron = 0;
    for (const NML_ComponentInstance *population : network_populations(*this)) {
        const s64 population_size = get_population_size(population);
        const s64 variable_count =
                (s64)population_cell(*this, *population).component_type->state_variable_names.size();
        const s64 base = simulation.population_base_indices.at(population->id);

        if (variable_count > 0 && cell_memory_index >= base &&
            cell_memory_index < base + population_size * variable_count) {
            return first_neuron + (cell_memory_index - base) / variable_count;
        }
        first_neuron += population_size;
    }
    return -1;
}

s64 NML_Context::get_population_size(const NML_ComponentInstance *population) const {
    s64 population_size = 0;
    for (const String &member_id : population->data_order) {
        if (is_instance_of(find_instance(member_id), "instance")) population_size += 1;
    }
    if (population_size == 0 && population->has_value("size")) {
        population_size = std::llround(resolve_quantity(population->value_or("size")));
    }
    return population_size;
}

// The number of values in cell memory: every state variable of every cell, over all populations.
s64 NML_Context::get_cell_state_size() const {
    const NML_ComponentInstance *network = find_instance(simulation.target_network_id);
    if (!network) return 0;

    s64 cell_state_size = 0;
    for (const String &child_id : network->data_order) {
        const NML_ComponentInstance *population = find_instance(child_id);
        if (!is_instance_of(population, "population")) continue;

        // A populationList's <instance> children are its cells; a plain population states a size.
        s64 population_size = 0;
        for (const String &member_id : population->data_order) {
            if (is_instance_of(find_instance(member_id), "instance")) population_size += 1;
        }
        if (population_size == 0 && population->has_value("size")) {
            population_size = std::llround(resolve_quantity(population->value_or("size")));
        }

        const NML_ComponentInstance *cell = find_instance(population->value_or("component"));
        cell_state_size += population_size * static_cast<s64>(cell->component_type->state_variable_names.size());
    }
    return cell_state_size;
}









}











