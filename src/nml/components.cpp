#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>

#include "spikecorec/nml/components.h"
#include "spikecorec/nml/dynamics.h"
#include "spikecorec/nml/declarations.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace spikecorec;
using namespace std::filesystem;

namespace spikecorec::nml {

String &NML_ComponentInstance::get_value(String &key) {
    if (instance_data.find(key) == instance_data.end()) {
        static String missing_value = "";
        return missing_value;
    }

    return instance_data[key];
}

bool NML_ComponentInstance::has_value(const String &key) const {
    return instance_data.find(key) != instance_data.end();
}

String NML_ComponentInstance::value_or(const String &key, const String &fallback) const {
    auto entry = instance_data.find(key);
    if (entry == instance_data.end()) return fallback;
    return entry->second;
}

RuntimeCategory NML_ComponentInstance::get_runtime_category() {
    return component_type->runtime_category;
}





}
