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
    """build_ext that compiles .cu sources with nvcc and everything else with the
    default host compiler. setuptools/distutils has no native CUDA rule, so we
    intercept per-file compilation and route .cu files through nvcc."""

    def build_extensions(self):
        cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        nvcc = os.path.join(cuda_path, "bin", "nvcc")
        if not os.path.exists(nvcc):
            nvcc = shutil.which("nvcc") or nvcc
        arch = os.environ.get("SPIKECOREC_CUDA_ARCH")  # e.g. "sm_87"; default lets nvcc pick + JIT

        self.compiler.src_extensions.append(".cu")
        default_compile = self.compiler._compile

        def cuda_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                cmd = [nvcc, "-std=c++17", "-O2", "--expt-relaxed-constexpr",
                       "-Xcompiler", "-fPIC"]
                if arch:
                    cmd += ["-arch=" + arch]
                # pp_opts carries the -I include dirs and -D macros; keep nvcc-safe
                # entries from extra_postargs (-std/-O...) and drop host-only -f flags.
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
INC = ["include", pybind11.get_include(), "third_party/spdlog/include"]
SRCS = [
    "src/bindings/bindings.cpp",
    "src/core/backend.cpp",
    "src/core/engine.cpp",
    "src/core/k2tree.cpp",
    "src/core/log.cpp",
    "src/core/recording.cpp",
    "src/core/time_utilities.cpp",
    "src/core/topologies.cpp",
    "src/core/types.cpp",
    "src/core/weight_matrix.cpp",
    # NeuroML/LEMS -> GPU codegen front+back end (epic #1) — ticket #133 exposes this to Python via
    # bindings.cpp's own NmlNetworkRunner, so it must actually be compiled/linked into the extension
    # (previously it was not: the Python surface was limited to SpikeEngine/WeightMatrix).
    "src/nml/allocator.cpp",
    "src/nml/cell_lowering.cpp",
    "src/nml/delay_ring.cpp",
    "src/nml/discrete_spike_input.cpp",
    "src/nml/expression_lowering.cpp",
    "src/nml/gpu_source.cpp",
    "src/nml/inputs_lowering.cpp",
    "src/nml/ir.cpp",
    "src/nml/master_kernel.cpp",
    "src/nml/model_specification.cpp",
    "src/nml/nml.cpp",
    "src/nml/output_recording.cpp",
    "src/nml/plasticity_wiring.cpp",
    "src/nml/resolve.cpp",
    "src/nml/stimulus_schedule.cpp",
    "src/nml/synapse_lowering.cpp",
]
EXTRA_COMPILE_ARGS = ["-std=c++17", "-O2"]
EXTRA_LINK_ARGS = []
DEFINE_MACROS = [("SPDLOG_ACTIVE_LEVEL", "SPDLOG_LEVEL_TRACE")]  # spdlog is header-only by default

# ── NeuroML2 vendored bundle paths (std-lib + XSD schema, ticket #133) ───────
# Baked in as absolute path macros, mirroring the Makefile's own NML_STD_LIB_DIR/NML_SCHEMA_PATH
# ($(abspath ...)) — NML_Parser::STANDARD_LIBRARY_PATH/NML_SCHEMA_PATH (resolve.h) resolve from these
# at runtime regardless of the interpreter's current working directory.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_NML_STD_LIB_DIR = os.path.join(_REPO_ROOT, "third_party", "neuroml2", "std_lib")
_NML_SCHEMA_PATH = os.path.join(_REPO_ROOT, "third_party", "neuroml2", "schema", "NeuroML_v2.3.xsd")
DEFINE_MACROS.append(("SPIKECOREC_NML_STD_LIB_DIR", '"%s"' % _NML_STD_LIB_DIR))
DEFINE_MACROS.append(("SPIKECOREC_NML_SCHEMA_PATH", '"%s"' % _NML_SCHEMA_PATH))

# ── XML parsing (libxml2 — NML/LEMS front-end) ───────────────────────────────
# nml.h/resolve.h include <libxml/parser.h>/<libxml/tree.h>/<libxml/xmlschemastypes.h>
# unconditionally (no #ifdef guard) — every NML/GLIF source file above needs libxml2's headers on
# the include path to compile at all. Detected the same way as the compression codecs below (via
# pkg-config), mirroring the Makefile's own HAS_LIBXML2 probe.
_libxml2_libs = _pkg_config("--libs", package="libxml-2.0")
if _libxml2_libs is not None:
    for _flag in _pkg_config("--cflags", package="libxml-2.0") or []:
        if _flag.startswith("-I"):
            INC.append(_flag[2:])
        else:
            EXTRA_COMPILE_ARGS.append(_flag)
    for _flag in _libxml2_libs:
        EXTRA_LINK_ARGS.append(_flag)

# ── Compression library detection (.spire codec — gzip/xz/bz2) ──────────────
# Mirrors the Makefile's HAS_ZLIB/HAS_LZMA/HAS_BZ2 probing: query pkg-config
# for cflags/libs and define SPIKECOREC_HAVE_* only when found. Missing
# libraries degrade gracefully — SpireWriter/Reader throw a clear
# runtime_error for compressed variants whose library wasn't available here.
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

# ── CUDA backend ─────────────────────────────────────────────
if BACKEND == "cuda":
    CUDA_PATH = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    INC.append(f"{CUDA_PATH}/include")
    SRCS.append("src/cuda/kernels.cu")       # compiled with nvcc by CudaBuildExt
    # backend.cpp uses the driver API (cuInit/cuCtxCreate/cuModuleLoadData) and
    # NVRTC in addition to the runtime API, so link -lcuda and -lnvrtc too. The
    # driver stub satisfies the link; the real libcuda loads at runtime.
    EXTRA_LINK_ARGS += [f"-L{CUDA_PATH}/lib64", f"-L{CUDA_PATH}/lib64/stubs",
                        "-lcudart", "-lcuda", "-lnvrtc"]
    DEFINE_MACROS.append(("SPIKECOREC_CUDA", None))

# ── Metal backend ────────────────────────────────────────────
elif BACKEND == "metal":
    if not _is_mac:
        raise RuntimeError("Metal backend requires macOS.")
    SRCS.append("src/metal/metal_cpp_impl.cpp")
    INC.append("third_party/metal-cpp")
    EXTRA_LINK_ARGS += [
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "QuartzCore",
    ]
    DEFINE_MACROS.append(("SPIKECOREC_METAL", None))

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
