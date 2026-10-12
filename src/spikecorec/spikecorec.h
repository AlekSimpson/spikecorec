#pragma once

#include "types.h"

#if defined(SPIKECOREC_CUDA)
#  include "kernels.cuh"
#elif defined(SPIKECOREC_METAL)
#  include "kernels.h"
#endif



namespace spikecorec {

String combine(const Vector<String> &values) {
    String result = "";
    for (String string_ : values) result += string_;
    return result;
}

String combine(String &filler, const Vector<String> &values) {
    String result = "";
    for (String string_: values) result += string_ + filler;
    return result;
}


};
