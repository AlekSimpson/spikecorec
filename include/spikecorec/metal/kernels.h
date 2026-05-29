#pragma once

#include "spikecorec/core/types.h"
#include "Metal/Metal.hpp"

namespace spikecorec::metal {

struct Context {
    MTL::Device*       device  = nullptr;
    MTL::CommandQueue* queue   = nullptr;
    MTL::Library*      library = nullptr;
};

Context make_context();
void    release_context(Context& ctx);

} // namespace spikecorec::metal
