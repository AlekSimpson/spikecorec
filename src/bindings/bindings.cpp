#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "spikecorec/core/types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_spikecorec, m) {
    m.doc() = "spikecorec C++/CUDA/Metal backend";
    m.attr("__version__") = "0.1.0";

    // Add bindings here as the library grows.
    // Example:
    //   py::class_<spikecorec::MyClass>(m, "MyClass")
    //       .def(py::init<>())
    //       .def("method", &spikecorec::MyClass::method);
}
