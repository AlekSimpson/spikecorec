"""
Build the _spikecorec extension module.

  pip install -e .                  # auto-detect backend
  SPIKECOREC_BACKEND=cuda pip install -e .
  SPIKECOREC_BACKEND=metal pip install -e .
"""

import glob
import os
import platform
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import pybind11

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


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

# Globbed, not listed by hand. A hand-maintained list is how this extension came to be
# missing src/core/engine_allocator.cpp, src/core/units.cpp and the whole of src/nml/ while
# the Makefile (which globs) built them fine — `make python` then failed at link time with
# undefined symbols that named nothing about the real cause. The Makefile's own CORE_SRCS /
# NML_SRCS are the same two wildcards.
SRCS = (
    ["src/bindings/bindings.cpp"]
    + sorted(glob.glob("src/core/*.cpp"))
    + sorted(glob.glob("src/nml/*.cpp"))
)
EXTRA_COMPILE_ARGS = ["-std=c++17", "-O2"]
EXTRA_LINK_ARGS = []
DEFINE_MACROS = [
    ("SPDLOG_ACTIVE_LEVEL", "SPDLOG_LEVEL_TRACE"),  # spdlog is header-only by default
    # Baked in as absolute paths, exactly as the Makefile's $(abspath ...) does, so the
    # vendored NeuroML/LEMS standard library and the XSD schema resolve no matter what
    # directory the interpreter importing this extension was started from.
    ("SPIKECOREC_NML_STD_LIB_DIR",
     '"' + os.path.join(REPO_ROOT, "third_party", "neuroml2", "std_lib") + '"'),
    ("SPIKECOREC_NML_SCHEMA_PATH",
     '"' + os.path.join(REPO_ROOT, "third_party", "neuroml2", "schema", "NeuroML_v2.3.xsd") + '"'),
]

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

# ── XML parsing (libxml2 — the NML/LEMS front-end) ──────────────────────────
# Not optional the way the compression codecs are: src/nml/nml.cpp includes
# <libxml/parser.h> unconditionally and the engine cannot read a model without it.
# Probed the same way, but a miss is a hard error rather than a graceful degrade.
_libxml2_libs = _pkg_config("--libs", package="libxml-2.0")
if _libxml2_libs is None:
    raise RuntimeError(
        "libxml2 not found via pkg-config. The NeuroML/LEMS front-end needs it; "
        "install libxml2 (macOS: `brew install libxml2`, and make sure its .pc file is on "
        "PKG_CONFIG_PATH)."
    )
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
