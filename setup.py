"""
Build the _spikecorec extension module.

  pip install -e .                  # auto-detect backend
  SPIKECOREC_BACKEND=cuda pip install -e .
  SPIKECOREC_BACKEND=metal pip install -e .
"""

import os
import sys
import platform
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import pybind11


class CudaBuildExt(_build_ext):
    """build_ext that routes .cu sources through nvcc, since setuptools has no CUDA rule."""

    def build_extensions(self):
        cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        nvcc = os.path.join(cuda_path, "bin", "nvcc")
        if not os.path.exists(nvcc):
            nvcc = shutil.which("nvcc") or nvcc
        arch = os.environ.get("SPIKECOREC_CUDA_ARCH")  # e.g. "sm_87"; unset lets nvcc choose

        self.compiler.src_extensions.append(".cu")
        default_compile = self.compiler._compile

        def cuda_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                cmd = [nvcc, "-std=c++17", "-O2", "--expt-relaxed-constexpr",
                       "-Xcompiler", "-fPIC"]
                if arch:
                    cmd += ["-arch=" + arch]
                # pp_opts holds the -I and -D flags; extra_postargs loses its host-only -f flags.
                cmd += pp_opts
                cmd += [a for a in extra_postargs if not a.startswith("-f")]
                cmd += ["-c", src, "-o", obj]
                self.compiler.spawn(cmd)
            else:
                default_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = cuda_compile
        try:
            super().build_extensions()
        finally:
            self.compiler._compile = default_compile


def _pkg_config(*flags, package):
    """Returns pkg-config output for `package`, or None if it isn't available."""
    if shutil.which("pkg-config") is None:
        return None
    try:
        out = subprocess.check_output(["pkg-config", *flags, package], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return out.decode().split()

# ── Backend selection ────────────────────────────────────────
_is_mac = platform.system() == "Darwin"
_default_backend = "metal" if _is_mac else "cuda"
BACKEND = os.environ.get("SPIKECOREC_BACKEND", _default_backend).lower()

if BACKEND not in ("cuda", "metal"):
    raise ValueError(f"SPIKECOREC_BACKEND must be 'cuda' or 'metal', got '{BACKEND}'")

# ── Common settings ──────────────────────────────────────────
INC = ["src", "src/spikecorec/core", "src/spikecorec/nml", "src/spikecorec/cuda", "src/spikecorec/metal",
       pybind11.get_include(), "third_party/spdlog/include"]
SRCS = [
    "src/spikecorec/bindings/bindings.cpp",
    "src/spikecorec/core/backend.cpp",
    "src/spikecorec/core/engine.cpp",
    "src/spikecorec/core/k2tree.cpp",
    "src/spikecorec/core/log.cpp",
    "src/spikecorec/core/recording.cpp",
    "src/spikecorec/core/topologies.cpp",
    "src/spikecorec/core/types.cpp",
    "src/spikecorec/core/units.cpp",
    "src/spikecorec/core/weight_matrix.cpp",
    # The NML front-end and kernel codegen, which engine.cpp links against.
    "src/spikecorec/nml/components.cpp",
    "src/spikecorec/nml/dynamics.cpp",
    "src/spikecorec/nml/kernel_boilerplate.cpp",
    "src/spikecorec/nml/kernel_parser.cpp",
    "src/spikecorec/nml/kernel_tokenizer.cpp",
    "src/spikecorec/nml/lems_parser.cpp",
    "src/spikecorec/nml/lems_tokenizer.cpp",
    "src/spikecorec/nml/node.cpp",
    "src/spikecorec/nml/parser.cpp",
    "src/spikecorec/nml/random_generator.cpp",
    "src/spikecorec/nml/utilities.cpp",
]
EXTRA_COMPILE_ARGS = ["-std=c++17", "-O2"]
EXTRA_LINK_ARGS = []
DEFINE_MACROS = [("SPDLOG_ACTIVE_LEVEL", "SPDLOG_LEVEL_TRACE")]  # spdlog is header-only by default

# ── Compression library detection (.spire codec — gzip/xz/bz2) ──────────────
# Optional, as in the Makefile: without a codec's library, raw .spire still works and that
# compressed format throws at runtime.
for _macro, _package in (
    ("SPIKECOREC_HAVE_ZLIB", "zlib"),
    ("SPIKECOREC_HAVE_LZMA", "liblzma"),
    ("SPIKECOREC_HAVE_BZ2", "bzip2"),
):
    _libs = _pkg_config("--libs", package=_package)
    if _libs is None:
        continue
    DEFINE_MACROS.append((_macro, None))
    for _flag in _pkg_config("--cflags", package=_package) or []:
        if _flag.startswith("-I"):
            INC.append(_flag[2:])
        else:
            EXTRA_COMPILE_ARGS.append(_flag)
    for _flag in _libs:
        EXTRA_LINK_ARGS.append(_flag)

# ── XML parsing (libxml2 — the NML/LEMS front-end) ────────────────────────────
# Required by the NML sources. On macOS the extension links with -undefined dynamic_lookup,
# so a missing -lxml2 fails at import, not at build.
_libxml2_libs = _pkg_config("--libs", package="libxml-2.0")
if _libxml2_libs is None:
    raise RuntimeError("libxml2 not found through pkg-config (libxml-2.0); the NML front-end needs it.")
DEFINE_MACROS.append(("SPIKECOREC_HAVE_LIBXML2", None))
for _flag in _pkg_config("--cflags", package="libxml-2.0") or []:
    if _flag.startswith("-I"):
        INC.append(_flag[2:])
    else:
        EXTRA_COMPILE_ARGS.append(_flag)
EXTRA_LINK_ARGS += _libxml2_libs

# ── CUDA backend ─────────────────────────────────────────────
if BACKEND == "cuda":
    CUDA_PATH = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    INC.append(f"{CUDA_PATH}/include")
    # backend.cpp also uses the driver API (cuInit, cuModuleLoadData) and NVRTC. The driver
    # stub satisfies the link; the real libcuda loads at runtime.
    EXTRA_LINK_ARGS += [f"-L{CUDA_PATH}/lib64", f"-L{CUDA_PATH}/lib64/stubs",
                        "-lcudart", "-lcuda", "-lnvrtc"]
    DEFINE_MACROS.append(("SPIKECOREC_CUDA", None))

# ── Metal backend ────────────────────────────────────────────
elif BACKEND == "metal":
    if not _is_mac:
        raise RuntimeError("Metal backend requires macOS.")
    SRCS.append("src/spikecorec/metal/metal_cpp_impl.cpp")
    INC.append("third_party/metal-cpp")
    EXTRA_LINK_ARGS += [
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "QuartzCore",
    ]
    DEFINE_MACROS.append(("SPIKECOREC_METAL", None))

# ── Compile-time paths ───────────────────────────────────────
# Source-tree paths the engine reads at runtime: the NeuroML standard library, the XSD, and
# (Metal) the directory of k2tree_device.metalinc, which the codegen parses into each kernel.
# Without them the extension imports but cannot load a model.
_source_root = os.path.abspath(os.path.dirname(__file__))
DEFINE_MACROS.append(
    ("SPIKECOREC_NML_STD_LIB_DIR",
     '"' + os.path.join(_source_root, "third_party", "neuroml2", "std_lib") + '"'))
DEFINE_MACROS.append(
    ("SPIKECOREC_NML_SCHEMA_PATH",
     '"' + os.path.join(_source_root, "third_party", "neuroml2", "schema", "NeuroML_v2.3.xsd") + '"'))
if BACKEND == "metal":
    DEFINE_MACROS.append(
        ("SPIKECOREC_METAL_DEVICE_DIR",
         '"' + os.path.join(_source_root, "src", "spikecorec", "metal") + '"'))

# ── Extension ────────────────────────────────────────────────
ext = Extension(
    name="spikecorec._spikecorec",
    sources=SRCS,
    include_dirs=INC,
    define_macros=DEFINE_MACROS,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    language="c++",
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": CudaBuildExt} if BACKEND == "cuda" else {},
)
