#include <Metal/Metal.hpp>
#include <cassert>
#include <cstdio>

int main() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    assert(device != nullptr && "No Metal device found");

    printf("metal tests passed (device: %s)\n", device->name()->utf8String());
    device->release();
    return 0;
}
