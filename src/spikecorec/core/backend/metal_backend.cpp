//
// Created by Alek Simpson on 5/30/26.
//


#include <string>

#include "metal.h"
#include "../log.h"
#include "../types.h"

using namespace std;

namespace spikecorec {

// #ifdef SPIKECOREC_CUDA


#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#include <dlfcn.h>

u64 aligned_partition_offset(u64 cursor, u64 alignment) {
    return (cursor + alignment - 1) & ~(alignment - 1);
}


MTL::Library *load_default_metal_library(MTL::Device *device) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&load_default_metal_library), &info) && info.dli_fname) {
        String binary_path(info.dli_fname);
        const usize last_slash = binary_path.find_last_of('/');
        const String directory = (last_slash == String::npos) ? "." : binary_path.substr(0, last_slash);
        const String metallib_path = directory + "/default.metallib";

        NS::String *path_string = NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
        NS::URL *url = NS::URL::fileURLWithPath(path_string);
        NS::Error *error = nullptr;
        if (MTL::Library *library = device->newLibrary(url, &error)) return library;
    }
    return device->newDefaultLibrary();
}

String describe_metal_error(NS::Error *error) {
    return error
        ? String(error->localizedDescription()->utf8String())
        : String("unknown error");
}

// Metal returns command buffers, encoders, errors and strings autoreleased: the caller does not
// own them. This pool frees them when the scope ends, on whatever thread runs it, instead of in
// the thread's own pool (which on the main thread never drains).
struct Autorelease_Pool_Scope {
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    Autorelease_Pool_Scope() = default;
    Autorelease_Pool_Scope(const Autorelease_Pool_Scope &) = delete;
    Autorelease_Pool_Scope &operator=(const Autorelease_Pool_Scope &) = delete;
    ~Autorelease_Pool_Scope() { pool->release(); }
};

#endif
}