#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {
enum class NML_DeclarationType {
    NamedDimension,
    Parameter,
    DerivedParameter,
    Constant,
    Requirement,
    Exposure,
    Property,
    Fixed,
    Text,
    EventPort,
    NamedTypeReference,
    Attachment,
    ComponentReference,
    Link,
    Children,
    Child,
    Path,
    StateVariable,
    DerivedVariable,
    ConditionalDerivedVariable,
    Case,
    TimeDerivative,
    OnCondition,
    OnEvent,
    OnStart,
    OnEntry,
    StateAssignment,
    EventOut,
    Transition,
    Regime,

    NOT_A_TYPE
};

struct NML_Declaration {
    NML_DeclarationType tag_type;

    UnorderedMap<String, String> datavalues;

    Vector<NML_Declaration> children;

    NML_Declaration(NML_DeclarationType type) : tag_type(type) {};
    NML_Declaration(NML_DeclarationType type, UnorderedMap<String, String> values)
        : tag_type(type), datavalues(std::move(values)) {};

    void add_values(const UnorderedMap<String, String> &data) { datavalues = data; }

    String get_value(const String &key) const;
    bool has_value(const String &key) const;
    String value_or(const String &key, const String &fallback = "") const;

    // Identity of this declaration within its ComponentType, ex: "<namespace>:<name>".
    String namespace_key() const;
};

struct NML_DeclarationList {
    Vector<NML_Declaration> declarations;

    NML_DeclarationList() = default;
    ~NML_DeclarationList() = default;

    NML_DeclarationList(const NML_DeclarationList &other) = default;
    NML_DeclarationList &operator=(const NML_DeclarationList &other) = default;

    NML_DeclarationList(NML_DeclarationList &&other) noexcept = default;
    NML_DeclarationList &operator=(NML_DeclarationList &&other) noexcept = default;

    const Vector<NML_Declaration> &for_all() const { return declarations; }

    // Appends, keeping `declarations` grouped by tag_type and preserving source order
    // within each group.
    void insert(const NML_Declaration &declaration);

    // Inheritance overlay: replaces the declaration sharing `declaration`'s
    // namespace_key() if one is present, otherwise inserts. This is what makes a subtype
    // redeclaring an ancestor's Parameter an override rather than a duplicate.
    void overlay(const NML_Declaration &declaration);

    const NML_Declaration *find_first(NML_DeclarationType type, const String &name) const;
    Vector<const NML_Declaration *> find_all(NML_DeclarationType type) const;
};


};
