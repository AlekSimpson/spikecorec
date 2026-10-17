# ============================================================
# spikecorec — Makefile
# Supports CUDA and Metal backends + Python bindings
# Usage:
#   make              — auto-detect backend and build
#   make cuda         — build CUDA backend
#   make metal        — build Metal backend
#   make python       — build Python extension (editable install into .venv, via uv)
#   make test         — build and run C++ tests
#   make test-python  — run the Python tests in tests/python (pytest, in .venv)
#   make examples     — build examples
#   make cli          — build main.cpp into build/cli
#   make metal-debug, cuda-debug, cli-debug — the same with debug info, into build/debug
#   make clean        — remove build artifacts
#   make info         — show detected platform/toolchain
# ============================================================

PROJECT    := spikecorec
VERSION    := 0.1.0

# Toolchain
CXX        := g++
NVCC       := nvcc
AR         := ar

# DEBUG=1 (or a *-debug target) builds with -g -O0 into build/debug, apart from optimised ones.
DEBUG      ?= 0
ifeq ($(DEBUG),1)
  OPTIMIZATION_FLAGS := -g -O0
  BUILD_DIR          := build/debug
else
  OPTIMIZATION_FLAGS := -O2
  BUILD_DIR          := build
endif

# Directories
INC_DIR    := src
SRC_DIR    := src/spikecorec
TEST_DIR   := tests
EX_DIR     := examples

# Compiler flags
# -MMD -MP writes a .d file of header dependencies beside each .o (included at the bottom),
# so editing a header rebuilds every object that uses it.
# Headers sit beside their sources and are included by file name alone, so every source
# directory is an include directory.
INCLUDE_FLAGS := -I$(INC_DIR) $(addprefix -I$(SRC_DIR)/,core nml cuda metal)
CXXFLAGS   := -std=c++17 $(OPTIMIZATION_FLAGS) -Wall -Wextra -MMD -MP $(INCLUDE_FLAGS)
# The precompiled CUDA kernels (src/spikecorec/cuda/*.cu) are built for CUDA_ARCH, by default the
# GPUs of the machine that builds them (for example sm_121 on a GB10). On a GPU they were not built
# for they do not load, and the fits run their host versions.
CUDA_ARCH  ?= native
NVCCFLAGS  := -std=c++17 $(OPTIMIZATION_FLAGS) -MMD -MP $(INCLUDE_FLAGS) --expt-relaxed-constexpr -arch=$(CUDA_ARCH)
ifeq ($(DEBUG),1)
  NVCCFLAGS += -G
endif
ARFLAGS    := rcs

# ── Platform detection ───────────────────────────────────────
UNAME_S    := $(shell uname -s)
UNAME_M    := $(shell uname -m)

CUDA_PATH  ?= /usr/local/cuda
HAS_NVCC   := $(shell command -v nvcc 2>/dev/null && echo yes || echo no)

# ── Compression library detection (.spire codec — gzip/xz/bz2) ───────────────
# Optional: without a codec's library, raw .spire still works and that compressed format
# throws at runtime.
HAS_ZLIB   := $(shell pkg-config --exists zlib   && echo yes || echo no)
HAS_LZMA   := $(shell pkg-config --exists liblzma && echo yes || echo no)
HAS_BZ2    := $(shell pkg-config --exists bzip2  && echo yes || echo no)

COMPRESSION_LIBS :=

ifeq ($(HAS_ZLIB),yes)
  CXXFLAGS         += -DSPIKECOREC_HAVE_ZLIB $(shell pkg-config --cflags zlib)
  COMPRESSION_LIBS += $(shell pkg-config --libs zlib)
endif
ifeq ($(HAS_LZMA),yes)
  CXXFLAGS         += -DSPIKECOREC_HAVE_LZMA $(shell pkg-config --cflags liblzma)
  COMPRESSION_LIBS += $(shell pkg-config --libs liblzma)
endif
ifeq ($(HAS_BZ2),yes)
  CXXFLAGS         += -DSPIKECOREC_HAVE_BZ2 $(shell pkg-config --cflags bzip2)
  COMPRESSION_LIBS += $(shell pkg-config --libs bzip2)
endif

# ── XML parsing (libxml2 — NML/LEMS front-end) ───────────────────────────────
# Required: the NML sources use it to parse and XSD-validate NeuroML/LEMS documents.
HAS_LIBXML2 := $(shell pkg-config --exists libxml-2.0 && echo yes || echo no)

ifeq ($(HAS_LIBXML2),yes)
  CXXFLAGS     += -DSPIKECOREC_HAVE_LIBXML2 $(shell pkg-config --cflags libxml-2.0)
  LIBXML2_LIBS := $(shell pkg-config --libs libxml-2.0)
endif

# ── NeuroML/LEMS standard-library bundle path (vendored) ─────────────────────
# Absolute, so NML_Context::STANDARD_LIBRARY_PATH works from any working directory.
NML_STD_LIB_DIR := $(abspath third_party/neuroml2/std_lib)
CXXFLAGS        += -DSPIKECOREC_NML_STD_LIB_DIR=\"$(NML_STD_LIB_DIR)\"

# ── NeuroML2 XSD schema path (vendored) ──────────────────────────────────────
NML_SCHEMA_PATH := $(abspath third_party/neuroml2/schema/NeuroML_v2.3.xsd)
CXXFLAGS         += -DSPIKECOREC_NML_SCHEMA_PATH=\"$(NML_SCHEMA_PATH)\"

# ── Device-code includes shared with runtime-compiled kernels ────────────────
# A runtime-compiled Metal kernel has no include path, so the codegen reads
# k2tree_device.metalinc from here and parses it into the kernel. Absolute, like the above.
METAL_DEVICE_DIR := $(abspath $(SRC_DIR)/metal)
CXXFLAGS         += -DSPIKECOREC_METAL_DEVICE_DIR=\"$(METAL_DEVICE_DIR)\"

# ── Logging (spdlog, header-only, vendored submodule) ─────────────────────────
# spdlog is header-only by default; only the compile-time log level is set here.
CXXFLAGS += -Ithird_party/spdlog/include -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE

# ── GoogleTest (compiled from submodule source) ───────────────────────────────
GTEST_DIR  := third_party/googletest/googletest
GTEST_OBJ  := $(BUILD_DIR)/gtest-all.o
CXXFLAGS   += -I$(GTEST_DIR)/include

ifeq ($(UNAME_S),Darwin)
  HAS_METAL     := yes
  CXX           := clang++
  METALC        := xcrun -sdk macosx metal
  METALLIB      := xcrun -sdk macosx metallib
  METALFLAGS    := $(if $(filter 1,$(DEBUG)),-gline-tables-only -frecord-sources,-O2)
  METAL_LDFLAGS := -framework Metal -framework Foundation -framework QuartzCore
  CXXFLAGS      += -Ithird_party/metal-cpp -DSPIKECOREC_METAL
else
  HAS_METAL  := no
  # Host sources include the CUDA headers, so the host compiler needs CUDA's include path.
  CXXFLAGS   += -DSPIKECOREC_CUDA -I$(CUDA_PATH)/include
endif

# Auto-select default backend
ifeq ($(UNAME_S),Darwin)
  BACKEND ?= metal
else
  BACKEND ?= cuda
endif

# ── Source files ─────────────────────────────────────────────
CORE_SRCS      := $(wildcard $(SRC_DIR)/core/*.cpp)
NML_SRCS       := $(wildcard $(SRC_DIR)/nml/*.cpp)
CUDA_SRCS      := $(wildcard $(SRC_DIR)/cuda/*.cu)
METAL_SRCS     := $(wildcard $(SRC_DIR)/metal/*.cpp)
METAL_SHADERS  := $(wildcard $(SRC_DIR)/metal/*.metal)

TEST_CORE_SRCS  := $(wildcard $(TEST_DIR)/*.cpp)
TEST_CUDA_SRCS  := $(wildcard $(TEST_DIR)/cuda/*.cpp)
TEST_METAL_SRCS := $(wildcard $(TEST_DIR)/metal/*.cpp)

EX_SRCS        := $(wildcard $(EX_DIR)/*.cpp)

# ── Object / artifact paths ──────────────────────────────────
CORE_OBJS   := $(patsubst $(SRC_DIR)/%.cpp,  $(BUILD_DIR)/%.o,   $(CORE_SRCS))
NML_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,  $(BUILD_DIR)/%.o,   $(NML_SRCS))
CUDA_OBJS   := $(patsubst $(SRC_DIR)/%.cu,   $(BUILD_DIR)/%.o,   $(CUDA_SRCS))
METAL_OBJS  := $(patsubst $(SRC_DIR)/%.cpp,  $(BUILD_DIR)/%.o,   $(METAL_SRCS))
AIR_FILES   := $(patsubst $(SRC_DIR)/metal/%.metal, \
                           $(BUILD_DIR)/metal/%.air, $(METAL_SHADERS))

CUDA_LIB    := $(BUILD_DIR)/lib$(PROJECT)_cuda.a
METAL_LIB   := $(BUILD_DIR)/lib$(PROJECT)_metal.a

# ── Python toolchain ─────────────────────────────────────────
# The extension builds into a uv-managed virtual environment, .venv, made on first use.
UV              ?= uv
VENV            := .venv
PYTHON_VERSION  ?= 3.13
PYTHON          := $(VENV)/bin/python
PYTHON_PACKAGES := pybind11 numpy setuptools matplotlib

# ── Top-level targets ────────────────────────────────────────
.PHONY: demos all cuda metal python test examples clean info cli metal-debug cuda-debug cli-debug

all: $(BACKEND)

cuda: check-cuda $(CUDA_LIB)
	@echo "[spikecorec] CUDA library built → $(CUDA_LIB)"

metal: check-metal $(METAL_LIB) $(BUILD_DIR)/default.metallib
	@echo "[spikecorec] Metal library built → $(METAL_LIB)"

# ── Checks ───────────────────────────────────────────────────
check-cuda:
ifeq ($(HAS_NVCC),no)
	$(error CUDA not found. Install the CUDA toolkit or set CUDA_PATH.)
endif

check-metal:
ifneq ($(HAS_METAL),yes)
	$(error Metal is only available on macOS.)
endif

# ── Static libraries ─────────────────────────────────────────
$(CUDA_LIB): $(CORE_OBJS) $(NML_OBJS) $(CUDA_OBJS)
	@mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $^

$(METAL_LIB): $(CORE_OBJS) $(NML_OBJS) $(METAL_OBJS)
	@mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $^

# ── Core objects (platform-agnostic C++) ─────────────────────
$(BUILD_DIR)/core/%.o: $(SRC_DIR)/core/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ── NML/LEMS front-end objects (platform-agnostic C++) ───────
$(BUILD_DIR)/nml/%.o: $(SRC_DIR)/nml/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ── CUDA objects (.cu) ───────────────────────────────────────
# The precompiled kernels Engine_Backend::load_precompiled_function finds by name.
$(BUILD_DIR)/cuda/%.o: $(SRC_DIR)/cuda/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ── Metal objects (.cpp via metal-cpp) ───────────────────────
$(BUILD_DIR)/metal/%.o: $(SRC_DIR)/metal/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ── Metal shaders (.metal → .air → .metallib) ────────────────
$(BUILD_DIR)/metal/%.air: $(SRC_DIR)/metal/%.metal $(SRC_DIR)/metal/k2tree_device.metalinc
	@mkdir -p $(@D)
	$(METALC) $(METALFLAGS) -I$(SRC_DIR)/metal -c $< -o $@

$(BUILD_DIR)/default.metallib: $(AIR_FILES)
	@mkdir -p $(@D)
	$(METALLIB) $(AIR_FILES) -o $@

# ── Python extension ─────────────────────────────────────────
# An editable install into .venv; needs uv. The package is always rebuilt, since uv only
# notices changes to pyproject.toml and setup.py.
python:
	@command -v $(UV) >/dev/null || { echo "uv not found: see https://docs.astral.sh/uv/"; exit 1; }
	$(UV) venv --quiet --allow-existing --python $(PYTHON_VERSION) $(VENV)
	$(UV) pip install --quiet --python $(PYTHON) $(PYTHON_PACKAGES)
	SPIKECOREC_BACKEND=$(BACKEND) $(UV) pip install --quiet --python $(PYTHON) --no-build-isolation \
	    --reinstall-package spikecorec -e .
ifeq ($(BACKEND),metal)
	$(MAKE) $(BUILD_DIR)/default.metallib
	cp $(BUILD_DIR)/default.metallib python/spikecorec/default.metallib
endif
	@echo "[spikecorec] Python extension installed (backend=$(BACKEND))"

# ── Tests ────────────────────────────────────────────────────
# `make test` builds every tests/*.cpp into one GoogleTest runner; `make test-python` runs
# tests/python. TEST_ARGUMENTS passes through, e.g. TEST_ARGUMENTS=--gtest_filter='Units*'.
.PHONY: test test-cuda test-metal test-python

test: test-$(BACKEND)

CUDA_LINK := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs -lcudart -lcuda -lnvrtc

$(GTEST_OBJ): $(GTEST_DIR)/src/gtest-all.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) -c $< -o $@

# One object per test file. The fixture directory is absolute, so any working directory works.
TEST_CXXFLAGS := $(CXXFLAGS) -I$(TEST_DIR) -DSPIKECOREC_TEST_FIXTURE_DIR=\"$(abspath $(TEST_DIR))/fixtures\"
TEST_OBJS     := $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/tests/%.o, $(TEST_CORE_SRCS))

$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@

test-cuda: check-cuda $(CUDA_LIB) $(GTEST_OBJ) $(TEST_OBJS)
	$(CXX) $(TEST_OBJS) $(GTEST_OBJ) \
	    -L$(BUILD_DIR) -l$(PROJECT)_cuda $(CUDA_LINK) $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread \
	    -o $(BUILD_DIR)/test_runner_cuda
	$(BUILD_DIR)/test_runner_cuda $(TEST_ARGUMENTS)

test-metal: check-metal $(METAL_LIB) $(BUILD_DIR)/default.metallib $(GTEST_OBJ) $(TEST_OBJS)
	$(CXX) $(TEST_OBJS) $(GTEST_OBJ) \
	    -L$(BUILD_DIR) -l$(PROJECT)_metal $(METAL_LDFLAGS) $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread \
	    -o $(BUILD_DIR)/test_runner_metal
	$(BUILD_DIR)/test_runner_metal $(TEST_ARGUMENTS)

test-python: python
	$(UV) pip install --quiet --python $(PYTHON) pytest
	$(PYTHON) -m pytest $(TEST_DIR)/python $(TEST_ARGUMENTS)

# ── Command-line program ─────────────────────────────────────
# Builds CLI_SOURCE into $(BUILD_DIR)/cli, beside default.metallib, where the engine looks for it.
CLI_SOURCE  ?= main.cpp
CLI_PROGRAM := $(BUILD_DIR)/cli

ifeq ($(BACKEND),metal)
  CLI_LIBRARY := $(METAL_LIB) $(BUILD_DIR)/default.metallib
  CLI_LINK    := -l$(PROJECT)_metal $(METAL_LDFLAGS)
else
  CLI_LIBRARY := $(CUDA_LIB)
  CLI_LINK    := -l$(PROJECT)_cuda $(CUDA_LINK)
endif

cli: check-$(BACKEND) $(CLI_PROGRAM)
	@echo "[spikecorec] command-line program built → $(CLI_PROGRAM)"

$(CLI_PROGRAM): $(CLI_SOURCE) $(CLI_LIBRARY)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CLI_SOURCE) \
	    -L$(BUILD_DIR) $(CLI_LINK) $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread \
	    -o $@

# ── Debug builds ─────────────────────────────────────────────
# The optimised target of the same name, rebuilt with DEBUG=1 into build/debug.
metal-debug cuda-debug cli-debug:
	$(MAKE) DEBUG=1 $(@:-debug=)

# ── Examples ─────────────────────────────────────────────────
examples: examples-$(BACKEND)

examples-cuda: check-cuda $(CUDA_LIB)
	@echo "[spikecorec] examples are Metal-only: the example build rule links the Metal library."
	@false

# Every .cpp directly in examples/ is one example program.
EX_PROGRAMS := $(patsubst $(EX_DIR)/%.cpp, $(BUILD_DIR)/examples/%, $(wildcard $(EX_DIR)/*.cpp))

examples-metal: check-metal $(EX_PROGRAMS)
	@echo "[spikecorec] examples built → $(BUILD_DIR)/examples/"

# ── Demos ────────────────────────────────────────────────────
# Runs every examples/demo_*.py with the .venv Python.
PYTHON_DEMOS := $(sort $(wildcard $(EX_DIR)/demo_*.py))

demos: python
	@for demo in $(PYTHON_DEMOS); do \
	    echo "[spikecorec] $$demo"; \
	    $(PYTHON) $$demo || exit 1; \
	done
	@echo "[spikecorec] demos rendered → $(BUILD_DIR)/demos/"

# `make build/examples/<name>` builds examples/<name>.cpp alone. Every examples/ header is a
# dependency of every example, so no example runs stale header values.
EX_HEADERS := $(wildcard $(EX_DIR)/*.h)

$(BUILD_DIR)/examples/%: $(EX_DIR)/%.cpp $(EX_HEADERS) $(METAL_LIB) $(BUILD_DIR)/default.metallib
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< \
	    -L$(BUILD_DIR) -l$(PROJECT)_metal $(METAL_LDFLAGS) $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread \
	    -o $@

# ── Utilities ────────────────────────────────────────────────
info:
	@echo "Platform   : $(UNAME_S) $(UNAME_M)"
	@echo "CUDA found : $(HAS_NVCC)"
	@echo "Metal found: $(HAS_METAL)"
	@echo "zlib found : $(HAS_ZLIB)"
	@echo "lzma found : $(HAS_LZMA)"
	@echo "bzip2 found: $(HAS_BZ2)"
	@echo "libxml2    : $(HAS_LIBXML2)"
	@echo "Backend    : $(BACKEND)"
	@echo "CXX        : $(CXX)"

compdb:
	bear -- make $(BACKEND) 2>/dev/null; true
	@echo "[spikecorec] compile_commands.json updated"

clean:
	rm -rf $(BUILD_DIR)

# ── Header dependencies ──────────────────────────────────────
# The .d files from -MMD -MP. `-include`, since a clean tree has none yet.
DEPENDENCY_FILES := $(CORE_OBJS:.o=.d) $(NML_OBJS:.o=.d) $(METAL_OBJS:.o=.d) $(CUDA_OBJS:.o=.d) $(TEST_OBJS:.o=.d) \
                    $(CLI_PROGRAM).d
-include $(DEPENDENCY_FILES)
