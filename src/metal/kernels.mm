#include "spikecorec/metal/kernels.hpp"

#ifdef __OBJC__

namespace spikecorec::metal {

Context make_context() {
    Context ctx;
    ctx.device  = MTLCreateSystemDefaultDevice();
    ctx.queue   = [ctx.device newCommandQueue];
    ctx.library = [ctx.device newDefaultLibrary];
    return ctx;
}

} // namespace spikecorec::metal

#endif // __OBJC__
