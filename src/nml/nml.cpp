#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>

#include "spikecorec/nml/nml.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/units.h"

#include <algorithm>
#include <cmath>
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
    body.push_back(std::move(component));
}

String NML_Node::get_attribute(const String &name) const {
    auto entry = attributes.find(name);
    if (entry == attributes.end()) return "";
    return entry->second;
}

const NML_Node *NML_Node::find_child(const String &child_tag_name) const {
    for (const NML_Node &child : body) {
        if (child.tag_name == child_tag_name) return &child;
    }
    return nullptr;
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

String NML_Declaration::value_or(const String &key, const String &fallback) const {
    auto entry = datavalues.find(key);
    if (entry == datavalues.end()) return fallback;
    return entry->second;
}

String NML_Declaration::namespace_key() const {
    // The namespaces a ComponentType keeps names unique within, per
    // docs/nml_general_notes.md. Seven tags share `variables`, so a subtype declaring
    // StateVariable "tau" overrides an inherited Parameter "tau".
    switch (tag_type) {
        case NML_DeclarationType::Parameter:
        case NML_DeclarationType::DerivedParameter:
        case NML_DeclarationType::Constant:
        case NML_DeclarationType::Requirement:
        case NML_DeclarationType::Property:
        case NML_DeclarationType::Text:
        case NML_DeclarationType::StateVariable:
        case NML_DeclarationType::DerivedVariable:
        case NML_DeclarationType::ConditionalDerivedVariable:
            return "var:" + value_or("name");

        case NML_DeclarationType::Exposure:
            return "exposure:" + value_or("name");

        case NML_DeclarationType::EventPort:
            return "port:" + value_or("name");

        case NML_DeclarationType::Child:
        case NML_DeclarationType::Children:
        case NML_DeclarationType::Attachment:
        case NML_DeclarationType::ComponentReference:
        case NML_DeclarationType::Link:
        case NML_DeclarationType::Path:
            return "pathsegment:" + value_or("name");

        case NML_DeclarationType::Regime:
            return "regime:" + value_or("name");

        // A subtype restating a derivative for the same variable replaces it rather than
        // integrating the variable twice.
        case NML_DeclarationType::TimeDerivative:
            return "derivative:" + value_or("variable");

        // Everything else accumulates: a subtype adding an OnCondition adds a condition,
        // it does not replace the parent's.
        default:
            return "";
    }
}

void DeclarationList::insert(const NML_Declaration &declaration) {
    ordered_declaration_type_counts[declaration.tag_type] += 1;

    auto comparator = [](const NML_Declaration &current, const NML_Declaration &next) {
        return current.tag_type < next.tag_type;
    };
    // upper_bound, not lower_bound: lower_bound returns the first element of an equal
    // tag_type group, so inserting there puts each new declaration ahead of its
    // predecessors and reverses source order within the group. Parameter order is the
    // column order of the exported starting-parameter rows, so that ordering is
    // load-bearing.
    auto index = std::upper_bound(
            declarations.begin(), declarations.end(),
            declaration,
            comparator);

    declarations.insert(index, declaration);
}

void DeclarationList::overlay(const NML_Declaration &declaration) {
    String key = declaration.namespace_key();

    // An empty key means this tag accumulates rather than overrides.
    if (key.empty()) {
        insert(declaration);
        return;
    }

    for (usize index = 0; index < declarations.size(); index += 1) {
        if (declarations[index].namespace_key() != key) continue;

        // Same tag_type keeps the list's grouping intact, so the entry is replaced where
        // it sits and the ancestor's position in declaration order is inherited. A
        // different tag_type (a StateVariable overriding a Parameter) has to move groups.
        if (declarations[index].tag_type == declaration.tag_type) {
            declarations[index] = declaration;
            return;
        }

        ordered_declaration_type_counts[declarations[index].tag_type] -= 1;
        declarations.erase(declarations.begin() + static_cast<s64>(index));
        insert(declaration);
        return;
    }

    insert(declaration);
}

const NML_Declaration *DeclarationList::find(NML_DeclarationType type,
                                             const String &name) const {
    for (const NML_Declaration &declaration : declarations) {
        if (declaration.tag_type != type) continue;
        if (declaration.value_or("name") == name) return &declaration;
    }
    return nullptr;
}

Vector<const NML_Declaration *> DeclarationList::find_all(NML_DeclarationType type) const {
    Vector<const NML_Declaration *> matches;
    for (const NML_Declaration &declaration : declarations) {
        if (declaration.tag_type != type) continue;
        matches.push_back(&declaration);
    }
    return matches;
}

bool is_declaration_type(s32 raw_type_value) {
    return raw_type_value >= 0 &&
           raw_type_value < static_cast<s32>(NML_DeclarationType::NOT_A_TYPE);
}

bool is_dynamic_type(NML_DeclarationType type) {
    return DYNAMICS_TYPES.find(type) != DYNAMICS_TYPES.end();
}

Vector<String> ComponentType::ordered_parameter_names() const {
    Vector<String> names;

    // Parameter and Property only. Text carries metadata strings ("synapses",
    // "TIME_ID"), and DerivedParameter is computed from other parameters rather than
    // supplied per instance, so neither belongs in a numeric starting-parameter row.
    for (const NML_Declaration &declaration : declarations.for_all()) {
        if (declaration.tag_type != NML_DeclarationType::Parameter &&
            declaration.tag_type != NML_DeclarationType::Property) {
            continue;
        }
        if (!declaration.has_value("name")) continue;
        names.push_back(declaration.get_value("name"));
    }

    return names;
}

Vector<String> ComponentType::ordered_parameter_dimensions() const {
    Vector<String> dimensions;

    // Walked with exactly the same filter as ordered_parameter_names so the two stay
    // index-for-index aligned; a parameter that declares no dimension contributes an
    // empty entry rather than being skipped.
    for (const NML_Declaration &declaration : declarations.for_all()) {
        if (declaration.tag_type != NML_DeclarationType::Parameter &&
            declaration.tag_type != NML_DeclarationType::Property) {
            continue;
        }
        if (!declaration.has_value("name")) continue;
        dimensions.push_back(declaration.value_or("dimension"));
    }

    return dimensions;
}

Vector<String> ComponentType::ordered_state_variable_names() const {
    Vector<String> names;
    for (const NML_Declaration &declaration : declarations.for_all()) {
        if (declaration.tag_type != NML_DeclarationType::StateVariable) continue;
        if (!declaration.has_value("name")) continue;
        names.push_back(declaration.get_value("name"));
    }
    return names;
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


// Splits "-60mV" into its numeric magnitude and its unit suffix. Shared by parse_quantity
// and NML_Parser::resolve_quantity so both scan numbers identically.
Pair<f64, String> split_quantity(const String &value) {
    if (value.empty()) return {0.0, ""};

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

    if (cursor == number_start) return {0.0, ""};

    f64 magnitude = 0.0;
    try {
        magnitude = std::stod(value.substr(number_start, cursor - number_start));
    } catch (const std::exception &) {
        return {0.0, ""};
    }

    while (cursor < value.size() && isspace(static_cast<unsigned char>(value[cursor]))) cursor += 1;

    String suffix = value.substr(cursor);
    while (!suffix.empty() && isspace(static_cast<unsigned char>(suffix.back()))) suffix.pop_back();

    return {magnitude, suffix};
}

// "-60mV" -> -0.06 against the built-in table only.
f64 parse_quantity(const String &value) {
    auto [magnitude, suffix] = split_quantity(value);
    return magnitude * units::unit_suffix_scale(suffix);
}

f64 NML_Parser::resolve_quantity(const String &value) const {
    auto [magnitude, suffix] = split_quantity(value);
    if (suffix.empty()) return magnitude;

    // A <Unit> the document declared wins: it is authoritative for this model, and it is
    // the only source that can express an offset (degC -> K).
    auto declared = declared_units.find(suffix);
    if (declared != declared_units.end()) {
        return magnitude * declared->second.scale + declared->second.offset;
    }

    return magnitude * units::unit_suffix_scale(suffix);
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
    Optional<DynamicsStage> handler_stage,
    Vector<DynamicsInstruction> &return_value
) {
    if (!is_dynamic_type(declaration.tag_type)) return;

    // A StateAssignment's stage is decided by the handler it sits in, not by its own tag.
    // The same <StateAssignment variable="v"> means "initialise v" under OnStart and
    // "reset v" under OnCondition; iafCell contains both. Tagging the OnStart one as
    // Reset would have the assembled tick re-run it every tick, pinning v to its initial
    // value forever.
    DynamicsStage stage = stage_for_declaration(declaration.tag_type);
    if (handler_stage.has_value() &&
        (declaration.tag_type == NML_DeclarationType::StateAssignment ||
         declaration.tag_type == NML_DeclarationType::EventOut)) {
        stage = *handler_stage;
    }

    DynamicsInstruction instruction(stage, declaration.tag_type);

    // The written name lives under a different attribute per tag: `variable` on
    // TimeDerivative/StateAssignment, `name` on DerivedVariable/Regime, `port` on
    // OnEvent/EventOut, `regime` on Transition.
    instruction.target =
            first_present_value(declaration, {"variable", "name", "port", "regime"});
    instruction.expression = first_present_value(declaration, {"value", "test", "select"});
    instruction.regime_name = regime_name;
    instruction.condition = condition;

    if (declaration.tag_type == NML_DeclarationType::Regime) {
        instruction.is_initial_regime = declaration.value_or("initial") == "true";
    }

    return_value.push_back(instruction);

    // A Regime scopes everything under it; a condition or event handler gates everything
    // under it. Both propagate down as context for the children.
    String child_regime_name = regime_name;
    if (declaration.tag_type == NML_DeclarationType::Regime) {
        child_regime_name = instruction.target;
    }

    // What gates the children. An OnCondition gates on its test; an OnEvent gates on the
    // arrival of an event at its port, so the port name is the gate. OnStart and OnEntry
    // are unconditional -- they are located by stage instead.
    String child_condition = condition;
    Optional<DynamicsStage> child_handler_stage = handler_stage;

    switch (declaration.tag_type) {
        case NML_DeclarationType::OnCondition:
            child_condition = instruction.expression;
            child_handler_stage = std::nullopt;
            break;
        case NML_DeclarationType::OnEvent:
            child_condition = instruction.target;
            child_handler_stage = DynamicsStage::Arrival;
            break;
        case NML_DeclarationType::OnStart:
            child_condition = "";
            child_handler_stage = DynamicsStage::Initialize;
            break;
        case NML_DeclarationType::OnEntry:
            child_condition = "";
            child_handler_stage = DynamicsStage::RegimeEntry;
            break;
        default:
            break;
    }

    for (const NML_Declaration &child : declaration.children) {
        collect_dynamics_instructions(
                child, child_regime_name, child_condition, child_handler_stage, return_value);
    }
}

static Vector<DynamicsInstruction> extract_dynamics_program(const ComponentType &component_type) {
    Vector<DynamicsInstruction> program;
    for (const auto &declaration : component_type.declarations.for_all()) {
        collect_dynamics_instructions(declaration, "", "", std::nullopt, program);
    }
    return program;
}

// Conductance-based synapses carry a driving force g*(erev - v), so they need the
// postsynaptic voltage; current-based ones do not.
static bool is_conductance_based(const ComponentType &component_type) {
    bool declares_erev =
            component_type.declarations.find(NML_DeclarationType::Parameter, "erev") != nullptr;
    bool requires_voltage =
            component_type.declarations.find(NML_DeclarationType::Requirement, "v") != nullptr;

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

// Seconds -> ticks. Rounded, never truncated: dt is a binary-inexact decimal, so a
// delay that is an exact multiple of dt still lands just under it -- 10ms / 0.01ms
// evaluates to 999.9999999999999, which truncation turns into 999 ticks.
static s64 tick_count_from_seconds(f64 seconds, f64 step_dt) {
    if (step_dt <= 0.0) return 0;
    return static_cast<s64>(std::llround(seconds / step_dt));
}

// Instance ids are scoped by parent ("net1.pop1"); model paths name the bare id.
static String leaf_id(const String &scoped_id) {
    usize separator = scoped_id.rfind('.');
    if (separator == String::npos) return scoped_id;
    return scoped_id.substr(separator + 1);
}


// ── path resolution ──────────────────────────────────────────────────────────────
//
// One routine for every cell-addressing spelling NeuroML uses, so connections, inputs and
// recording quantities all resolve identically:
//
//     ../pop1[0]        connection preCellId / postCellId
//     pop1[0]           synapticConnection from / to, explicitInput target
//     ../pop1/0/cell1   populationList form, naming the component type
//     pop1[0]/v         OutputColumn quantity, with a trailing variable
struct ResolvedCellPath {
    s64 neuron_index = -1;
    String population_name;
    s64 local_index = -1;
    String trailing; // whatever followed the cell selector, e.g. "v"
};

static bool text_is_integer(const String &text) {
    if (text.empty()) return false;
    for (usize index = 0; index < text.size(); index += 1) {
        if (!isdigit(static_cast<unsigned char>(text[index]))) return false;
    }
    return true;
}

static Vector<String> split_on_slash(const String &text) {
    Vector<String> segments;
    usize start = 0;
    while (start <= text.size()) {
        usize next = text.find('/', start);
        if (next == String::npos) {
            segments.push_back(text.substr(start));
            break;
        }
        segments.push_back(text.substr(start, next - start));
        start = next + 1;
    }
    return segments;
}

// `population_cell_type_name` disambiguates "pop1/0/x": x is the component type name in
// the populationList spelling, and a trailing variable otherwise. Empty when unknown, in
// which case a single trailing segment is read as a variable.
static ResolvedCellPath parse_cell_path(const String &path,
                                        const String &population_cell_type_name) {
    ResolvedCellPath resolved;

    String remaining = path;
    while (remaining.rfind("../", 0) == 0) remaining = remaining.substr(3);
    if (remaining.rfind("./", 0) == 0) remaining = remaining.substr(2);

    usize bracket = remaining.find('[');
    if (bracket != String::npos) {
        usize closing = remaining.find(']', bracket);
        if (closing == String::npos) return resolved;

        resolved.population_name = remaining.substr(0, bracket);
        String index_text = remaining.substr(bracket + 1, closing - bracket - 1);
        if (!text_is_integer(index_text)) return resolved;
        resolved.local_index = static_cast<s64>(std::stoll(index_text));

        String rest = remaining.substr(closing + 1);
        if (!rest.empty() && rest.front() == '/') rest = rest.substr(1);
        resolved.trailing = rest;
        return resolved;
    }

    Vector<String> segments = split_on_slash(remaining);
    if (segments.size() < 2) return resolved;
    if (!text_is_integer(segments[1])) return resolved;

    resolved.population_name = segments[0];
    resolved.local_index = static_cast<s64>(std::stoll(segments[1]));

    usize trailing_start = 2;
    if (segments.size() > 2 && !population_cell_type_name.empty() &&
        segments[2] == population_cell_type_name) {
        trailing_start = 3;
    }

    for (usize index = trailing_start; index < segments.size(); index += 1) {
        if (!resolved.trailing.empty()) resolved.trailing += "/";
        resolved.trailing += segments[index];
    }

    return resolved;
}


// ── ingest ───────────────────────────────────────────────────────────────────────

// The conversion reads nothing but the libxml node, so it is a free function; the
// NML_Parser method of the same name delegates here.
static NML_Node convert_xml_node(xmlNodePtr node) {
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
        component.nest(convert_xml_node(child));
    }

    return component;
}

// Reads and converts a document, freeing the libxml document once the tree has been
// copied into NML_Nodes. Returns false when the file could not be read.
static bool read_document_root(const String &filepath, NML_Node &return_value) {
    xmlDocPtr document = xmlReadFile(filepath.c_str(), nullptr, XML_PARSE_NOBLANKS);
    if (!document) {
        log::logger().error("Could not parse NML/LEMS file {}", filepath);
        return false;
    }

    xmlNodePtr root = xmlDocGetRootElement(document);
    if (!root) {
        log::logger().error("NML/LEMS file {} has no root element", filepath);
        xmlFreeDoc(document);
        return false;
    }

    // convert_xml_node deep-copies into owned Strings, so the document can be freed as
    // soon as the conversion returns -- unlike get_xml_root, which hands back a pointer
    // into a document it never frees.
    return_value = convert_xml_node(root);
    xmlFreeDoc(document);
    return true;
}

void NML_Parser::ingest_document(const String &file_path) {
    // Dedupe on the canonical path, not the raw href: Cells.xml and Synapses.xml both
    // include NeuroMLCoreDimensions.xml, and LEMS makes includes idempotent. Deduping on
    // the spelling would let the same file through twice and turn every type it declares
    // into a spurious redefinition.
    std::error_code path_error;
    path canonical = weakly_canonical(path(file_path), path_error);
    String canonical_text = path_error ? file_path : canonical.string();

    if (parsed_neuroml_files.find(canonical_text) != parsed_neuroml_files.end()) return;

    // Marked before recursing, so an include cycle terminates instead of recursing until
    // the stack runs out.
    parsed_neuroml_files.insert(canonical_text);

    NML_Node root;
    if (!read_document_root(canonical_text, root)) return;

    path containing_directory = path(canonical_text).parent_path();

    for (NML_Node &child : root.body) {
        // LEMS spells it <Include file="..."/>; NeuroML spells it <include href="..."/>.
        if (child.tag_name == "Include" || child.tag_name == "include") {
            String reference = child.has_attribute("file")
                    ? child.get_attribute("file")
                    : child.get_attribute("href");
            if (reference.empty()) continue;

            // Relative to the including file's own directory, not the process cwd.
            path included = path(reference);
            if (included.is_relative()) included = containing_directory / included;

            // A model that says <Include file="Cells.xml"/> means the standard library's
            // Cells.xml, not one sitting beside itself. Falling back keeps such a document
            // portable instead of only loadable from the directory it was written in --
            // and parse_lems has already ingested the standard library, so the fallback
            // path is normally deduped away rather than re-read.
            if (!exists(included) && !STANDARD_LIBRARY_PATH.empty()) {
                const path from_standard_library =
                        path(STANDARD_LIBRARY_PATH) / path(reference).filename();
                if (exists(from_standard_library)) included = from_standard_library;
            }

            ingest_document(included.string());
            continue;
        }

        if (child.tag_name == "ComponentType") {
            String type_name = child.get_attribute("name");
            if (type_name.empty()) {
                log::logger().warn("Ignoring <ComponentType> with no name in {}", canonical_text);
                continue;
            }

            auto existing = component_type_source_files.find(type_name);
            if (existing != component_type_source_files.end() &&
                existing->second != canonical_text) {
                // Survived canonical-path deduping, so this is a genuine redefinition
                // across two different files. Silently letting one win would make the
                // model depend on filesystem iteration order.
                throw runtime_error(
                        "Duplicate ComponentType '" + type_name + "' declared in both " +
                        existing->second + " and " + canonical_text);
            }

            // Moved, not copied: `root` is a local that dies when this call returns, and
            // a ComponentType node carries its whole declaration subtree.
            component_type_catalogue[type_name] = std::move(child);
            component_type_source_files[type_name] = canonical_text;
            continue;
        }

        if (child.tag_name == "Unit") {
            String symbol = child.get_attribute("symbol");
            if (symbol.empty()) continue;

            // LEMS: si = raw * scale * 10^power + offset. power is folded into scale here
            // so resolution stays a single multiply-add.
            UnitDefinition definition;
            definition.scale = 1.0;
            definition.offset = 0.0;

            if (child.has_attribute("scale")) {
                auto [scale_magnitude, scale_suffix] = split_quantity(child.get_attribute("scale"));
                (void)scale_suffix;
                if (scale_magnitude != 0.0) definition.scale = scale_magnitude;
            }
            if (child.has_attribute("power")) {
                auto [power_magnitude, power_suffix] = split_quantity(child.get_attribute("power"));
                (void)power_suffix;
                definition.scale *= std::pow(10.0, power_magnitude);
            }
            if (child.has_attribute("offset")) {
                auto [offset_magnitude, offset_suffix] =
                        split_quantity(child.get_attribute("offset"));
                (void)offset_suffix;
                definition.offset = offset_magnitude;
            }

            declared_units[symbol] = definition;
            continue;
        }

        if (child.tag_name == "Dimension") continue; // dimensional analysis is not modelled

        if (child.tag_name == "Constant") {
            String constant_name = child.get_attribute("name");
            if (constant_name.empty()) continue;

            Real resolved;
            resolved.float64 = resolve_quantity(child.get_attribute("value"));
            document_constants[constant_name] = resolved;
            continue;
        }

        if (child.tag_name == "Target") {
            target_component_id = child.get_attribute("component");
            continue;
        }

        // Anything else is a component instance. It cannot be instantiated yet: the type
        // it names may live in a file not read so far.
        document_instance_nodes.push_back(std::move(child));
    }
}


// ── ComponentType resolution ─────────────────────────────────────────────────────

// Collects a node's declarations in source order. Replaces a LIFO-stack walk, which
// reversed siblings and so destroyed the Parameter order the export depends on.
static void collect_declarations_in_source_order(const NML_Node &node,
                                                 Vector<NML_Declaration> &return_value) {
    for (const NML_Node &child : node.body) {
        if (child.is_declaration_type()) {
            // to_declaration() consumes the whole subtree (a Regime brings its
            // TimeDerivatives with it), so the body must not be walked again here.
            return_value.push_back(child.to_declaration());
            continue;
        }

        // Not a declaration (Dynamics, Structure, Simulation, ...) -- descend looking for
        // the declarations underneath.
        collect_declarations_in_source_order(child, return_value);
    }
}

void NML_Parser::extract_all_declarations_nested_in_node(
    NML_Node *node,
    DeclarationList &return_value
) {
    if (!node) return;

    Vector<NML_Declaration> collected;
    collect_declarations_in_source_order(*node, collected);

    for (const NML_Declaration &declaration : collected) {
        return_value.insert(declaration);
    }
}

RuntimeCategory NML_Parser::category_for_component_type(const String &type_name) const {
    // Roots of each runtime category. The walk is upward from the type through its
    // `extends` chain, so a type is categorised by its nearest categorised ancestor.
    // connection, input, instance and location have no `extends` at all and so have to
    // appear here in their own right.
    static const UnorderedMap<String, RuntimeCategory> category_roots = {
        {"baseCell",                     RuntimeCategory::Cell},
        {"baseSynapse",                  RuntimeCategory::Synapse},
        {"baseGradedSynapse",            RuntimeCategory::Synapse},
        {"basePointCurrent",             RuntimeCategory::InputScheme},
        {"baseSpikeSource",              RuntimeCategory::InputScheme},
        {"network",                      RuntimeCategory::Graph},
        {"networkWithTemperature",       RuntimeCategory::Graph},
        {"basePopulation",               RuntimeCategory::Graph},
        {"population",                   RuntimeCategory::Graph},
        {"populationList",               RuntimeCategory::Graph},
        {"instance",                     RuntimeCategory::Graph},
        {"location",                     RuntimeCategory::Graph},
        {"projection",                   RuntimeCategory::Graph},
        {"electricalProjection",         RuntimeCategory::Graph},
        {"continuousProjection",         RuntimeCategory::Graph},
        {"explicitConnection",           RuntimeCategory::Graph},
        {"connection",                   RuntimeCategory::Graph},
        {"inputList",                    RuntimeCategory::Graph},
        {"input",                        RuntimeCategory::Graph},
        {"explicitInput",                RuntimeCategory::Graph},
    };

    String current = type_name;
    Set<String> visited;

    while (!current.empty()) {
        if (!visited.insert(current).second) break; // cycle; resolution reports it

        auto root = category_roots.find(current);
        if (root != category_roots.end()) return root->second;

        auto catalogued = component_type_catalogue.find(current);
        if (catalogued == component_type_catalogue.end()) break;

        current = catalogued->second.get_attribute("extends");
    }

    return RuntimeCategory::General;
}

const ComponentType &NML_Parser::resolve_component_type(const String &type_name,
                                                        Vector<String> &resolution_stack) {
    auto already_resolved = declared_component_types.find(type_name);
    if (already_resolved != declared_component_types.end()) return already_resolved->second;

    auto catalogued = component_type_catalogue.find(type_name);
    if (catalogued == component_type_catalogue.end()) {
        String chain;
        for (const String &entry : resolution_stack) chain += entry + " -> ";
        throw runtime_error(
                "Unresolved ComponentType '" + type_name +
                "' referenced from extends chain: " + chain + type_name);
    }

    for (const String &entry : resolution_stack) {
        if (entry != type_name) continue;

        String chain;
        for (const String &stack_entry : resolution_stack) chain += stack_entry + " -> ";
        throw runtime_error("Cyclic ComponentType extends chain: " + chain + type_name);
    }

    resolution_stack.push_back(type_name);

    // A reference, not a copy: resolution only ever writes to declared_component_types,
    // never to the catalogue, so this stays valid across the recursive parent resolve.
    const NML_Node &type_node = catalogued->second;
    String parent_name = type_node.get_attribute("extends");

    DeclarationList declarations;
    if (!parent_name.empty()) {
        // Ancestors first: the parent's declarations form the base that this type's own
        // declarations overlay, so most-derived wins.
        const ComponentType &parent = resolve_component_type(parent_name, resolution_stack);
        declarations = parent.declarations;
    }

    Vector<NML_Declaration> own_declarations;
    collect_declarations_in_source_order(type_node, own_declarations);
    for (const NML_Declaration &declaration : own_declarations) {
        declarations.overlay(declaration);
    }

    // <Fixed parameter="tau" value="10ms"/> pins an inherited Parameter to a constant.
    // Applied after the overlay so it can reach a Parameter this type never declared.
    for (const NML_Declaration &declaration : declarations.for_all()) {
        if (declaration.tag_type != NML_DeclarationType::Fixed) continue;

        String parameter_name = declaration.value_or("parameter");
        if (parameter_name.empty()) continue;

        for (NML_Declaration &target : declarations.declarations) {
            if (target.tag_type != NML_DeclarationType::Parameter) continue;
            if (target.value_or("name") != parameter_name) continue;

            target.datavalues["value"] = declaration.value_or("value");
            break;
        }
    }

    resolution_stack.pop_back();

    ComponentType resolved(type_name, std::move(declarations),
                           category_for_component_type(type_name));
    resolved.extends = parent_name;

    declared_component_types[type_name] = std::move(resolved);
    return declared_component_types[type_name];
}

void NML_Parser::resolve_all_component_types() {
    // Sorted so resolution order -- and therefore any error surfaced by it -- does not
    // depend on hash iteration order.
    Vector<String> type_names;
    type_names.reserve(component_type_catalogue.size());
    for (const auto &[type_name, node] : component_type_catalogue) {
        (void)node;
        type_names.push_back(type_name);
    }
    std::sort(type_names.begin(), type_names.end());

    for (const String &type_name : type_names) {
        Vector<String> resolution_stack;
        resolve_component_type(type_name, resolution_stack);
    }
}

void NML_Parser::load_component_types_in_file(NML_Node *file_root) {
    if (!file_root) return;

    for (NML_Node &child : file_root->body) {
        if (child.tag_name != "ComponentType") continue;

        String type_name = child.get_attribute("name");
        if (type_name.empty()) continue;

        component_type_catalogue[type_name] = child;
    }

    resolve_all_component_types();
}


// ── instantiation ────────────────────────────────────────────────────────────────

void NML_Parser::instantiate(NML_Node *instance_node) {
    String no_parent;
    instantiate(instance_node, no_parent);
}

void NML_Parser::instantiate(NML_Node *instance_node, String &parent_instance_id) {
    if (!instance_node) return;

    String component_type_name = instance_node->tag_name;

    auto type_entry = declared_component_types.find(component_type_name);
    if (type_entry == declared_component_types.end()) {
        // Not a declared ComponentType. Documents are full of metadata that is not
        // modelled -- notes, annotation, and the ~50 RDF/Dublin Core wrappers -- and none
        // of it affects the simulation, so this warns rather than failing the parse.
        // References that genuinely cannot resolve (a population's component, a
        // connection endpoint) are thrown on where they are read.
        log::logger().warn("Ignoring <{}>: no such ComponentType is declared",
                           component_type_name);
        return;
    }
    const ComponentType &component_to_instantiate = type_entry->second;

    String own_id = instance_node->get_attribute("id");
    if (own_id.empty()) {
        // A <connection> inside a projection routinely carries no id. Without a synthetic
        // one every such child collapses onto the same empty key and all but the last is
        // lost.
        usize ordinal = 0;
        if (!parent_instance_id.empty()) {
            auto parent = instance_table.find(parent_instance_id);
            if (parent != instance_table.end()) {
                ordinal = parent->second.structured_instance_data.size();
            }
        }
        own_id = component_type_name + "[" + std::to_string(ordinal) + "]";
    }

    String instance_id = parent_instance_id.empty()
            ? own_id
            : parent_instance_id + "." + own_id;

    if (instance_table.find(instance_id) != instance_table.end()) {
        log::logger().warn("Duplicate component instance id '{}'; the later one wins",
                           instance_id);
    }

    ComponentInstance &instance = instance_table[instance_id];
    instance.id = instance_id;
    instance.component_type_name = component_type_name;
    instance.parent_instance_id = parent_instance_id;
    instance.component_type = &component_to_instantiate;

    bind_instance_data(instance_node, component_to_instantiate, instance);

    // Registered on the parent before recursing, so structured_instance_data lists
    // children in document order.
    if (!parent_instance_id.empty()) {
        instance_table[parent_instance_id].structured_instance_data.push_back(instance_id);
    }

    for (NML_Node &child_instance_node : instance_node->body) {
        instantiate(&child_instance_node, instance_id);
    }
}


// ── entry points ─────────────────────────────────────────────────────────────────

NML_ParseResult NML_Parser::parse_lems(const String &lems_main_file) {
    if (!STANDARD_LIBRARY_PATH.empty() && exists(STANDARD_LIBRARY_PATH)) {
        // Sorted: directory_iterator order is unspecified, and ingest order decides which
        // file a duplicate ComponentType is blamed on.
        Vector<String> standard_library_files;
        for (const auto &entry : directory_iterator(STANDARD_LIBRARY_PATH)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".xml" && entry.path().extension() != ".nml") continue;

            standard_library_files.push_back(entry.path().string());
        }
        std::sort(standard_library_files.begin(), standard_library_files.end());

        for (const String &filepath : standard_library_files) {
            ingest_document(filepath);
        }
    } else {
        log::logger().error("NML standard library path does not exist: {}",
                            STANDARD_LIBRARY_PATH);
    }

    ingest_document(lems_main_file);

    // Every document is in, so `extends` can finally be resolved by name across all of
    // them, and only then can instances be bound to their types.
    resolve_all_component_types();

    // Rebuilt from scratch rather than added to. instantiate() appends each child onto
    // its parent's structured_instance_data, so instantiating twice over the same nodes
    // would duplicate every population and connection in the network.
    instance_table.clear();
    for (NML_Node &instance_node : document_instance_nodes) {
        instantiate(&instance_node);
    }

    std::error_code path_error;
    path canonical = weakly_canonical(path(lems_main_file), path_error);
    read_document_root(path_error ? lems_main_file : canonical.string(), main_document_root);

    return export_model_details_to_engine(&main_document_root);
}

void NML_Parser::parse_neuroml(const String &nml_file_path) {
    ingest_document(nml_file_path);
    resolve_all_component_types();

    // As in parse_lems: rebuilt, not appended to, so calling this more than once (or
    // after parse_lems) cannot duplicate a network's children.
    instance_table.clear();
    for (NML_Node &instance_node : document_instance_nodes) {
        instantiate(&instance_node);
    }
}


// ── export ───────────────────────────────────────────────────────────────────────

// Every <Constant> a ComponentType declares, namespaced by its owning type so that
// hindmarshRose1984Cell's MSEC and fitzHughNagumoCell's SEC cannot collide.
static void collect_component_type_constants(
    const NML_Parser &parser,
    const NML_Declaration &declaration,
    const String &component_type_name,
    UnorderedMap<String, Real> &return_value
) {
    if (declaration.tag_type == NML_DeclarationType::Constant &&
        declaration.has_value("name") && declaration.has_value("value")) {
        Real resolved;
        resolved.float64 = parser.resolve_quantity(declaration.get_value("value"));
        return_value[component_type_name + "." + declaration.get_value("name")] = resolved;
    }

    for (const NML_Declaration &child : declaration.children) {
        collect_component_type_constants(parser, child, component_type_name, return_value);
    }
}

NML_ParseResult NML_Parser::export_model_details_to_engine(NML_Node *lems_root) {
    NML_ParseResult return_value;

    // Name -> index and instance -> prototype lookups are parser bookkeeping, so they
    // stay local. The result carries entities, not the indices used to assemble them.
    UnorderedMap<String, s64> cell_type_indices;
    UnorderedMap<String, s64> synapse_type_indices;
    UnorderedMap<String, s64> cell_prototype_indices;
    UnorderedMap<String, s64> synapse_prototype_indices;

    // ── Pass 0: document scope ───────────────────────────────────────────────────
    // dt has to be known before anything else, because delays are converted to ticks.
    for (const auto &[constant_name, constant_value] : document_constants) {
        return_value.global_constants[constant_name] = constant_value;
    }

    for (const auto &[type_name, component_type] : declared_component_types) {
        for (const auto &declaration : component_type.declarations.for_all()) {
            collect_component_type_constants(
                    *this, declaration, type_name, return_value.global_constants);
        }
    }

    // The Simulation is a component instance, not a language element. <Target> names it;
    // failing that, the first Simulation instance in document order is used.
    const ComponentInstance *simulation = nullptr;
    if (!target_component_id.empty()) {
        auto targeted = instance_table.find(target_component_id);
        if (targeted != instance_table.end()) simulation = &targeted->second;
    }
    if (!simulation) {
        for (const NML_Node &node : document_instance_nodes) {
            auto candidate = instance_table.find(node.get_attribute("id"));
            if (candidate == instance_table.end()) continue;
            if (candidate->second.component_type_name != "Simulation") continue;

            simulation = &candidate->second;
            break;
        }
    }

    if (simulation) {
        return_value.simulation_component_id = simulation->id;
        return_value.target_network_id = simulation->value_or("target");

        if (simulation->has_value("step")) {
            return_value.step_dt = resolve_quantity(simulation->value_or("step"));
        }
        if (simulation->has_value("length")) {
            return_value.simulation_duration = resolve_quantity(simulation->value_or("length"));
        }
        return_value.total_tick_count = tick_count_from_seconds(
                return_value.simulation_duration, return_value.step_dt);

        if (simulation->has_value("seed")) {
            try {
                return_value.random_seed =
                        static_cast<u64>(std::stoull(simulation->value_or("seed")));
            } catch (const std::exception &) {
                log::logger().warn("Simulation '{}' has a non-numeric seed '{}'; ignoring",
                                   simulation->id, simulation->value_or("seed"));
            }
        }
    } else if (lems_root) {
        log::logger().warn("No Simulation instance found in {}", lems_root->tag_name);
    }

    // ── Pass 1: types and prototypes ─────────────────────────────────────────────
    // Two things are registered here and they are deliberately separate. Everything
    // structural -- state variables, dynamics, parameter column order -- is a property of
    // the ComponentType. The parameter VALUES are a property of the instance: a population
    // names "component=cellA", and cellA and cellB may both be iafCell yet differ.
    auto register_type =
        [&](const ComponentType &component_type, bool is_cell) -> s64 {
            UnorderedMap<String, s64> &indices = is_cell ? cell_type_indices
                                                         : synapse_type_indices;

            auto known = indices.find(component_type.name);
            if (known != indices.end()) return known->second;

            if (is_cell) {
                s64 type_index = static_cast<s64>(return_value.cell_types.size());
                indices[component_type.name] = type_index;

                CellTypeSpecification specification;
                specification.name = component_type.name;
                specification.state_variable_names =
                        component_type.ordered_state_variable_names();
                specification.parameter_names = component_type.ordered_parameter_names();
                specification.parameter_dimensions =
                        component_type.ordered_parameter_dimensions();
                specification.dynamics = extract_dynamics_program(component_type);

                // Where this type's state begins in the engine's cell-state buffer: the
                // running sum of the preceding types' state variable counts.
                specification.state_offset = 0;
                for (const CellTypeSpecification &preceding : return_value.cell_types) {
                    specification.state_offset +=
                            static_cast<s32>(preceding.state_variable_names.size());
                }

                return_value.cell_types.push_back(std::move(specification));
                return type_index;
            }

            s64 type_index = static_cast<s64>(return_value.synapse_types.size());
            indices[component_type.name] = type_index;

            SynapseTypeSpecification specification;
            specification.name = component_type.name;
            specification.state_variable_names = component_type.ordered_state_variable_names();
            specification.parameter_names = component_type.ordered_parameter_names();
            specification.dynamics = extract_dynamics_program(component_type);
            specification.is_conductance_based = is_conductance_based(component_type);
            specification.requires_per_edge_state = requires_per_edge_state(component_type);

            return_value.synapse_types.push_back(std::move(specification));
            return type_index;
        };

    auto register_prototype =
        [&](const ComponentInstance &instance, bool is_cell) -> s64 {
            UnorderedMap<String, s64> &prototypes = is_cell ? cell_prototype_indices
                                                            : synapse_prototype_indices;

            auto known = prototypes.find(instance.id);
            if (known != prototypes.end()) return known->second;

            const ComponentType &component_type = *instance.component_type;
            s64 type_index = register_type(component_type, is_cell);

            const Vector<String> &parameter_names = is_cell
                    ? return_value.cell_types[static_cast<usize>(type_index)].parameter_names
                    : return_value.synapse_types[static_cast<usize>(type_index)].parameter_names;

            // This instance's values, in the type's declared parameter order. An absent
            // attribute falls back to the declaration's own value -- a Fixed pin or a
            // Property defaultValue -- and only then to zero.
            ComponentPrototype prototype;
            prototype.instance_id = instance.id;
            prototype.type_index = type_index;
            prototype.starting_parameters.reserve(parameter_names.size());

            for (const String &parameter_name : parameter_names) {
                String text = instance.value_or(parameter_name);
                if (text.empty()) {
                    const NML_Declaration *declared = component_type.declarations.find(
                            NML_DeclarationType::Parameter, parameter_name);
                    if (!declared) {
                        declared = component_type.declarations.find(
                                NML_DeclarationType::Property, parameter_name);
                    }
                    if (declared) {
                        text = declared->value_or("value", declared->value_or("defaultValue"));
                    }
                }

                Real resolved;
                resolved.float64 = text.empty() ? 0.0 : resolve_quantity(text);
                prototype.starting_parameters.push_back(resolved);
            }

            Vector<ComponentPrototype> &destination = is_cell
                    ? return_value.cell_prototypes
                    : return_value.synapse_prototypes;

            s64 prototype_index = static_cast<s64>(destination.size());
            prototypes[instance.id] = prototype_index;
            destination.push_back(std::move(prototype));

            return prototype_index;
        };

    for (const NML_Node &node : document_instance_nodes) {
        auto entry = instance_table.find(node.get_attribute("id"));
        if (entry == instance_table.end()) continue;

        const ComponentInstance &instance = entry->second;
        if (!instance.component_type) continue;

        RuntimeCategory category = instance.component_type->runtime_category;
        if (category != RuntimeCategory::Cell && category != RuntimeCategory::Synapse) continue;

        register_prototype(instance, category == RuntimeCategory::Cell);
    }

    // ── Pass 2: populations and neurons ──────────────────────────────────────────
    // Walked from the target network through structured_instance_data, so populations are
    // numbered in the order the document declares them.
    const ComponentInstance *network = nullptr;
    if (!return_value.target_network_id.empty()) {
        auto targeted = instance_table.find(return_value.target_network_id);
        if (targeted != instance_table.end()) network = &targeted->second;
    }
    if (!network) {
        for (const NML_Node &node : document_instance_nodes) {
            auto candidate = instance_table.find(node.get_attribute("id"));
            if (candidate == instance_table.end()) continue;
            if (candidate->second.component_type_name != "network" &&
                candidate->second.component_type_name != "networkWithTemperature") {
                continue;
            }

            network = &candidate->second;
            return_value.target_network_id = candidate->second.id;
            break;
        }
    }

    if (network) {
        auto temperature = network->instance_data.find("temperature");
        if (temperature != network->instance_data.end()) {
            Real resolved;
            resolved.float64 = resolve_quantity(temperature->second);
            return_value.global_constants[leaf_id(network->id) + ".temperature"] = resolved;
        }

        for (const String &child_id : network->structured_instance_data) {
            auto child_entry = instance_table.find(child_id);
            if (child_entry == instance_table.end()) continue;

            const ComponentInstance &child = child_entry->second;
            if (child.component_type_name != "population" &&
                child.component_type_name != "populationList") {
                continue;
            }

            // A populationList enumerates its members as <instance> children; a sized
            // population states a count.
            s64 size = 0;
            for (const String &member_id : child.structured_instance_data) {
                auto member = instance_table.find(member_id);
                if (member == instance_table.end()) continue;
                if (member->second.component_type_name != "instance") continue;
                size += 1;
            }
            if (size == 0 && child.has_value("size")) {
                size = static_cast<s64>(std::llround(resolve_quantity(child.value_or("size"))));
            }

            String component_reference = child.value_or("component");
            if (component_reference.empty()) {
                throw runtime_error(
                        "Population '" + child.id + "' names no component to instantiate");
            }

            auto cell_instance = instance_table.find(component_reference);
            if (cell_instance == instance_table.end()) {
                throw runtime_error(
                        "Population '" + child.id + "' references component '" +
                        component_reference + "', which no instance declares");
            }
            if (!cell_instance->second.component_type) {
                throw runtime_error(
                        "Population '" + child.id + "' references component '" +
                        component_reference + "', which resolved to no ComponentType");
            }

            const String &cell_type_name = cell_instance->second.component_type->name;
            if (cell_instance->second.component_type->runtime_category != RuntimeCategory::Cell) {
                throw runtime_error(
                        "Population '" + child.id + "' references component '" +
                        component_reference + "' of type '" + cell_type_name +
                        "', which is not classified as a cell");
            }

            // The population names an instance, so the parameters are that instance's.
            // Registered on demand: a prototype declared inside the network rather than
            // at document scope has not been walked yet.
            s64 prototype_index = register_prototype(cell_instance->second, true);
            s64 cell_type_index = cell_type_indices.at(cell_type_name);
            s64 population_index = static_cast<s64>(return_value.populations.size());

            PopulationLayout layout;
            layout.population_name = leaf_id(child.id);
            layout.instance_id = child.id;
            layout.component_instance_id = cell_instance->second.id;
            layout.first_neuron_index = static_cast<s64>(return_value.neurons.size());
            layout.neuron_count = size;
            layout.cell_type_index = cell_type_index;
            layout.prototype_index = prototype_index;

            return_value.populations.push_back(std::move(layout));

            for (s64 member = 0; member < size; member += 1) {
                Neuron neuron;
                neuron.cell_type_index = cell_type_index;
                neuron.prototype_index = prototype_index;
                neuron.population_index = population_index;
                return_value.neurons.push_back(std::move(neuron));
            }
        }
    }

    UnorderedMap<String, const PopulationLayout *> population_by_name;
    for (const PopulationLayout &layout : return_value.populations) {
        population_by_name[layout.population_name] = &layout;
    }

    // Resolves a path against the population layout, or -1. The population's own cell type
    // name disambiguates the populationList spelling "pop/0/cellType".
    auto resolve_neuron_index = [&](const String &path) -> s64 {
        ResolvedCellPath partial = parse_cell_path(path, "");
        if (partial.population_name.empty()) return -1;

        auto population = population_by_name.find(partial.population_name);
        if (population == population_by_name.end()) return -1;

        String cell_type_name;
        s64 cell_type_index = population->second->cell_type_index;
        if (cell_type_index >= 0 &&
            cell_type_index < static_cast<s64>(return_value.cell_types.size())) {
            cell_type_name = return_value.cell_types[static_cast<usize>(cell_type_index)].name;
        }

        ResolvedCellPath resolved = parse_cell_path(path, cell_type_name);
        if (resolved.local_index < 0) return -1;
        if (resolved.local_index >= population->second->neuron_count) return -1;

        return population->second->first_neuron_index + resolved.local_index;
    };

    // ── Pass 3: connections ──────────────────────────────────────────────────────
    static const Set<String> connection_tags = {
        "connection", "connectionWD",
        "synapticConnection", "synapticConnectionWD",
        "explicitConnection"
    };

    if (network) {
        for (const String &child_id : network->structured_instance_data) {
            auto projection_entry = instance_table.find(child_id);
            if (projection_entry == instance_table.end()) continue;

            const ComponentInstance &projection = projection_entry->second;
            if (projection.component_type_name != "projection") continue;

            // The projection names the synapse prototype every one of its edges uses.
            // Resolves to a prototype as well as a type, since two projections may share
            // a synapse ComponentType with different parameter values.
            auto resolve_synapse = [&](const String &reference, const String &owner_id,
                                       s64 &type_index, s64 &prototype_index) {
                auto synapse_instance = instance_table.find(reference);
                if (synapse_instance == instance_table.end() ||
                    !synapse_instance->second.component_type) {
                    throw runtime_error(
                            "Projection '" + owner_id + "' references synapse '" +
                            reference + "', which no instance declares");
                }
                if (synapse_instance->second.component_type->runtime_category !=
                    RuntimeCategory::Synapse) {
                    throw runtime_error(
                            "Projection '" + owner_id + "' references synapse '" + reference +
                            "' of type '" + synapse_instance->second.component_type->name +
                            "', which is not classified as a synapse");
                }

                prototype_index = register_prototype(synapse_instance->second, false);
                type_index = synapse_type_indices.at(
                        synapse_instance->second.component_type->name);
            };

            s64 projection_synapse_type = -1;
            s64 projection_synapse_prototype = -1;
            String synapse_reference = projection.value_or("synapse");
            if (!synapse_reference.empty()) {
                resolve_synapse(synapse_reference, projection.id,
                                projection_synapse_type, projection_synapse_prototype);
            }

            for (const String &connection_id : projection.structured_instance_data) {
                auto connection_entry = instance_table.find(connection_id);
                if (connection_entry == instance_table.end()) continue;

                const ComponentInstance &connection = connection_entry->second;
                if (connection_tags.find(connection.component_type_name) ==
                    connection_tags.end()) {
                    continue;
                }

                // connection/connectionWD spell their endpoints preCellId/postCellId; the
                // explicitConnection family spells them from/to.
                String pre_path = connection.value_or("preCellId",
                                                      connection.value_or("from"));
                String post_path = connection.value_or("postCellId",
                                                       connection.value_or("to"));

                s64 pre_neuron_index = resolve_neuron_index(pre_path);
                s64 post_neuron_index = resolve_neuron_index(post_path);

                if (pre_neuron_index < 0 || post_neuron_index < 0) {
                    throw runtime_error(
                            "Connection '" + connection.id + "' in projection '" +
                            projection.id + "' has an endpoint that resolves to no neuron: "
                            "pre='" + pre_path + "' post='" + post_path + "'");
                }

                NetworkEdge edge;
                edge.target_neuron_index = post_neuron_index;
                edge.synapse_type_index = projection_synapse_type;
                edge.synapse_prototype_index = projection_synapse_prototype;

                // A synapticConnection names its own synapse, overriding the projection's.
                String own_synapse = connection.value_or("synapse");
                if (!own_synapse.empty()) {
                    resolve_synapse(own_synapse, connection.id,
                                    edge.synapse_type_index, edge.synapse_prototype_index);
                }

                if (connection.has_value("weight")) {
                    edge.weight = resolve_quantity(connection.value_or("weight"));
                }
                if (connection.has_value("delay")) {
                    edge.delay_tick_count = tick_count_from_seconds(
                            resolve_quantity(connection.value_or("delay")),
                            return_value.step_dt);
                }

                return_value.neurons[static_cast<usize>(pre_neuron_index)]
                        .outgoing_edges.push_back(edge);
            }
        }
    }

    // ── Pass 4: inputs ───────────────────────────────────────────────────────────
    // One profile per input component that is actually wired to a target. The wiring
    // lives on explicitInput / inputList; the amplitudes and rates live on the input
    // component the wiring names.
    auto build_input_profile =
        [&](const String &input_component_id, Vector<InputTarget> targets) {
            auto input_instance = instance_table.find(input_component_id);
            if (input_instance == instance_table.end()) {
                throw runtime_error(
                        "Input wiring references component '" + input_component_id +
                        "', which no instance declares");
            }

            const ComponentInstance &input = input_instance->second;

            SimulationInputConfig profile;
            profile.input_component_id = input.id;
            profile.input_component_type_name = input.component_type_name;

            if (input.has_value("amplitude")) {
                profile.amplitude = resolve_quantity(input.value_or("amplitude"));
            }
            if (input.has_value("rate")) {
                profile.rate = resolve_quantity(input.value_or("rate"));
            }

            // Times are held in ticks so the engine never converts.
            if (return_value.step_dt > 0.0) {
                if (input.has_value("delay")) {
                    profile.start_tick = tick_count_from_seconds(
                            resolve_quantity(input.value_or("delay")), return_value.step_dt);
                }
                if (input.has_value("duration")) {
                    profile.end_tick = profile.start_tick + tick_count_from_seconds(
                            resolve_quantity(input.value_or("duration")), return_value.step_dt);
                }

                // A spikeArray carries its train as <spike time="..."/> children. Every
                // target receives the same train.
                Vector<s32> spike_ticks;
                for (const String &spike_id : input.structured_instance_data) {
                    auto spike = instance_table.find(spike_id);
                    if (spike == instance_table.end()) continue;
                    if (spike->second.component_type_name != "spike") continue;
                    if (!spike->second.has_value("time")) continue;

                    spike_ticks.push_back(static_cast<s32>(tick_count_from_seconds(
                            resolve_quantity(spike->second.value_or("time")),
                            return_value.step_dt)));
                }
                std::sort(spike_ticks.begin(), spike_ticks.end());

                if (!spike_ticks.empty()) {
                    for (InputTarget &target : targets) target.event_ticks = spike_ticks;
                }

                // A component carrying an explicit spike train is an event source, whatever
                // else it declares; only one with no train and an amplitude injects current
                // continuously. Deciding this on `amplitude` alone misreads every spiking
                // source that also carries one.
                profile.continuous_current_injection =
                        spike_ticks.empty() && input.has_value("amplitude");
            }

            profile.targets = std::move(targets);
            return_value.input_profiles.push_back(std::move(profile));
        };

    if (network) {
        for (const String &child_id : network->structured_instance_data) {
            auto child_entry = instance_table.find(child_id);
            if (child_entry == instance_table.end()) continue;

            const ComponentInstance &child = child_entry->second;

            if (child.component_type_name == "explicitInput") {
                String target_path = child.value_or("target");
                s64 neuron_index = resolve_neuron_index(target_path);
                if (neuron_index < 0) {
                    throw runtime_error(
                            "explicitInput '" + child.id + "' targets '" + target_path +
                            "', which resolves to no neuron");
                }

                InputTarget target;
                target.neuron_index = neuron_index;
                build_input_profile(child.value_or("input"), {target});
                continue;
            }

            if (child.component_type_name != "inputList") continue;

            Vector<InputTarget> targets;
            for (const String &input_id : child.structured_instance_data) {
                auto input_entry = instance_table.find(input_id);
                if (input_entry == instance_table.end()) continue;

                const ComponentInstance &input = input_entry->second;
                if (input.component_type_name != "input" &&
                    input.component_type_name != "inputW") {
                    continue;
                }

                String target_path = input.value_or("target");
                s64 neuron_index = resolve_neuron_index(target_path);
                if (neuron_index < 0) {
                    throw runtime_error(
                            "Input '" + input.id + "' in inputList '" + child.id +
                            "' targets '" + target_path + "', which resolves to no neuron");
                }

                InputTarget target;
                target.neuron_index = neuron_index;
                if (input.has_value("weight")) {
                    target.weight = resolve_quantity(input.value_or("weight"));
                }
                targets.push_back(std::move(target));
            }

            if (targets.empty()) continue;

            build_input_profile(child.value_or("component"), std::move(targets));
        }
    }

    // ── Pass 5: recordings ───────────────────────────────────────────────────────
    if (simulation) {
        for (const String &output_id : simulation->structured_instance_data) {
            auto output_entry = instance_table.find(output_id);
            if (output_entry == instance_table.end()) continue;

            const ComponentInstance &output = output_entry->second;

            // Display is on-screen only and names no file, so it contributes no profile.
            bool is_event_file = output.component_type_name == "EventOutputFile";
            if (output.component_type_name != "OutputFile" && !is_event_file) continue;

            String filename = output.value_or("fileName");
            if (filename.empty()) continue;

            RecordingConfig recording_profile{};
            recording_profile.output_filenames.push_back(filename);
            recording_profile.file_output_format.push_back(
                    is_event_file ? OutputFileFormat::SPIKE_EVENTS
                                  : output_format_for_filename(filename));
            recording_profile.recordings_count = 0;

            for (const String &selection_id : output.structured_instance_data) {
                auto selection_entry = instance_table.find(selection_id);
                if (selection_entry == instance_table.end()) continue;

                const ComponentInstance &selection = selection_entry->second;

                RecordingSelection recorded;
                if (selection.component_type_name == "OutputColumn") {
                    recorded.quantity_path = selection.value_or("quantity");
                } else if (selection.component_type_name == "EventSelection") {
                    recorded.quantity_path = selection.value_or("select");
                    recorded.event_port = selection.value_or("eventPort");
                } else {
                    continue;
                }

                recorded.neuron_index = resolve_neuron_index(recorded.quantity_path);
                if (recorded.neuron_index < 0) {
                    log::logger().warn(
                            "Recording '{}' in {} resolves to no neuron; recorded with "
                            "index -1", recorded.quantity_path, filename);
                } else {
                    String cell_type_name;
                    ResolvedCellPath partial = parse_cell_path(recorded.quantity_path, "");
                    auto population = population_by_name.find(partial.population_name);
                    if (population != population_by_name.end()) {
                        s64 cell_type_index = population->second->cell_type_index;
                        if (cell_type_index >= 0 &&
                            cell_type_index <
                                    static_cast<s64>(return_value.cell_types.size())) {
                            cell_type_name = return_value.cell_types[
                                    static_cast<usize>(cell_type_index)].name;
                        }
                    }
                    recorded.variable_name =
                            parse_cell_path(recorded.quantity_path, cell_type_name).trailing;
                }

                recording_profile.selections.push_back(std::move(recorded));
                recording_profile.recordings_count += 1;
            }

            return_value.recording_profiles.push_back(std::move(recording_profile));
        }
    }

    return return_value;
}

Vector<Vector<s64>> build_adjacency_list(const NML_ParseResult &parse_result) {
    Vector<Vector<s64>> adjacency(parse_result.neurons.size());

    for (usize source = 0; source < parse_result.neurons.size(); source += 1) {
        const Vector<NetworkEdge> &edges = parse_result.neurons[source].outgoing_edges;
        adjacency[source].reserve(edges.size());

        for (const NetworkEdge &edge : edges) {
            adjacency[source].push_back(edge.target_neuron_index);
        }
    }

    return adjacency;
}


// ── libxml plumbing ──────────────────────────────────────────────────────────────

NML_Node NML_Parser::xml_node_to_nml_node(xmlNodePtr node) {
    return convert_xml_node(node);
}

// Accumulates each schema validation error's line number and message (libxml2's messages
// already name the offending element, e.g. "Element 'thisTagDoesNotExistInSchema': No
// matching global declaration available for the validation root.") into `*user_data`,
// joined by " | " -- see validate_against_schema's own comment for why this replaces
// libxml2's default, stderr-only error handler.
// libxml2 2.12 made xmlStructuredErrorFunc take a `const xmlError *`; before that it took
// a mutable xmlErrorPtr. This machine has both 2.9 (pkg-config, what the Makefile picks)
// and 2.13 (xml2-config) installed, so the signature is selected rather than assumed.
#if LIBXML_VERSION >= 21200
void collect_schema_validation_error(void *user_data, const xmlError *error) {
#else
void collect_schema_validation_error(void *user_data, xmlErrorPtr error) {
#endif
    if (!error || !error->message) return;

    String message(error->message);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();

    String *destination = static_cast<String *>(user_data);
    if (!destination->empty()) *destination += " | ";
    *destination += "line " + std::to_string(error->line) + ": " + message;
}

bool NML_Parser::validate_against_schema(const String &nml_file_path) {
    // The schema describes NeuroML documents. A LEMS wrapper -- <Lems><Include .../>
    // <Target .../></Lems>, which is what drives a simulation and therefore what gets
    // handed to the engine -- is a different language with no XSD here, and validating one
    // against the NeuroML schema reports every element in it as an error. There is nothing
    // to check, so say so rather than failing a perfectly good file.
    xmlDocPtr document = xmlReadFile(nml_file_path.c_str(), nullptr, 0);
    if (!document) {
        last_schema_validation_errors = "could not read " + nml_file_path;
        return false;
    }

    xmlNodePtr root = xmlDocGetRootElement(document);
    const bool is_neuroml_document =
            root && root->name &&
            String(reinterpret_cast<const char *>(root->name)) == "neuroml";
    xmlFreeDoc(document);

    if (!is_neuroml_document) {
        log::logger().debug("{} is not a NeuroML document root; skipping XSD validation",
                            nml_file_path);
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


// ── ComponentInstance ────────────────────────────────────────────────────────────

String &ComponentInstance::get_value(String &key) {
    // Static so the miss case can still hand back a reference; a literal would dangle.
    static String missing_value = "";

    if (instance_data.find(key) == instance_data.end()) {
        return missing_value;
    }

    return instance_data[key];
}

bool ComponentInstance::has_value(const String &key) const {
    return instance_data.find(key) != instance_data.end();
}

String ComponentInstance::value_or(const String &key, const String &fallback) const {
    auto entry = instance_data.find(key);
    if (entry == instance_data.end()) return fallback;
    return entry->second;
}

RuntimeCategory ComponentInstance::get_runtime_category() {
    return component_type->runtime_category;
}

}
