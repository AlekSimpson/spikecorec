#pragma once

#include "spikecorec/core/types.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

namespace spikecorec::metal {

struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
};

Context make_context();

} // namespace spikecorec::metal
#endif // __OBJC__
