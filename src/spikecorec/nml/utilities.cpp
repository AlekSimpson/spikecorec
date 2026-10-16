#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>

#include "utilities.h"
#include "log.h"
#include "units.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace spikecorec;
using namespace std::filesystem;

namespace spikecorec::nml {


// The output format a filename's extension selects; any non-.spire name is the NML column format.
Output_File_Format output_format_for_filename(const String &filename) {
    auto ends_with = [&filename](const String &suffix) {
        return filename.size() >= suffix.size() &&
               filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    if (ends_with(".spire.gz") || ends_with(".spire.gzip")) return Output_File_Format::SPIREGZIP;
    if (ends_with(".spire.xz") || ends_with(".spire.lzma")) return Output_File_Format::SPIREXZ;
    if (ends_with(".spire.bz2"))                           return Output_File_Format::SPIREBZ2;
    if (ends_with(".spire"))                               return Output_File_Format::SPIRE;

    return Output_File_Format::NML_STANDARD;
}


// Unused: parser.cpp has its own static copy, which is the one it calls.
// // Appends "line N: message" for each schema error to *user_data, joined by " | ".
// // libxml2 2.12 changed the error parameter to `const xmlError *`.
// #if LIBXML_VERSION >= 21200
// void collect_schema_validation_error(void *user_data, const xmlError *error) {
// #else
// void collect_schema_validation_error(void *user_data, xmlErrorPtr error) {
// #endif
//     if (!error || !error->message) return;
//
//     String message(error->message);
//     while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();
//
//     String *destination = static_cast<String *>(user_data);
//     if (!destination->empty()) *destination += " | ";
//     *destination += "line " + std::to_string(error->line) + ": " + message;
// }




}




