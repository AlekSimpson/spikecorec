#include <metal_stdlib>
using namespace metal;

// Example compute kernel — replace with your own.
kernel void passthrough(
    device const float* in  [[ buffer(0) ]],
    device       float* out [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    out[gid] = in[gid];
}
