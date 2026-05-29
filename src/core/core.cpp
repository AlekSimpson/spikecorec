#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"

#ifdef SPIKECOREC_METAL
#include "spikecorec/metal/kernels.h"

// ── Metal backend implementation ──────────────────────────────
// CPU-side Metal setup and kernel dispatch lives here.
// GPU kernel source is in src/metal/shaders/

namespace spikecorec::metal {

Context make_context() {
    Context ctx;
    ctx.device  = MTL::CreateSystemDefaultDevice();
    ctx.queue   = ctx.device->newCommandQueue();
    ctx.library = ctx.device->newDefaultLibrary();
    return ctx;
}

void release_context(Context& ctx) {
    if (ctx.library) { ctx.library->release(); ctx.library = nullptr; }
    if (ctx.queue)   { ctx.queue->release();   ctx.queue   = nullptr; }
    if (ctx.device)  { ctx.device->release();  ctx.device  = nullptr; }
}

} // namespace spikecorec::metal

namespace spikecorec::backend {

static metal::Context g_ctx;

void init() {
    g_ctx = metal::make_context();
}

void shutdown() {
    metal::release_context(g_ctx);
}

// Add kernel dispatch functions here, e.g.:
// void run_step(const f32* input, f32* output, usize n) {
//     auto* cmd = g_ctx.queue->commandBuffer();
//     auto* enc = cmd->computeCommandEncoder();
//     // set pipeline, buffers, dispatch...
//     enc->endEncoding();
//     cmd->commit();
//     cmd->waitUntilCompleted();
// }

} // namespace spikecorec::backend

#endif // SPIKECOREC_METAL
