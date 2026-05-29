#import <Metal/Metal.h>
#include <cassert>
#include <cstdio>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    assert(device != nil && "No Metal device found");

    printf("metal tests passed (device: %s)\n",
           [[device name] UTF8String]);
    return 0;
}
