#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>

#include "spikecorec/nml/utilities.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/units.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace spikecorec;
using namespace std::filesystem;

namespace spikecorec::nml {


// Output format inferred from the file extension, matching resolve_spire_compression's
// rules. A non-.spire extension means the NML-standard column matrix.
OutputFileFormat output_format_for_filename(const String &filename) {
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




}




