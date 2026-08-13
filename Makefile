# ============================================================
# spikecorec — Makefile
# Supports CUDA and Metal backends + Python bindings
# Usage:
#   make              — auto-detect backend and build
#   make cuda         — build CUDA backend
#   make metal        — build Metal backend
#   make python       — build Python extension (pip editable install)
#   make python-test  — build the extension AND run python/tests (needs PYTHON=, see below)
#   make test         — build and run C++ tests
#   make examples     — build examples
#   make run-examples — build AND run every example, failing on any non-zero exit
#   make check        — the full gate: make test + make run-examples
#   make demos        — build the demo programs (examples/demos/)
#   make run-demos    — build AND run every demo, recording but not rendering
#   make demo-videos  — build, run AND render every demo to a video (needs DEMO_PYTHON)
#   make clean        — remove build artifacts
#   make info         — show detected platform/toolchain
# ============================================================

PROJECT    := spikecorec
VERSION    := 0.1.0

# Toolchain
CXX        := g++
NVCC       := nvcc
AR         := ar

# Directories
INC_DIR    := include
SRC_DIR    := src
BUILD_DIR  := build
TEST_DIR   := tests
EX_DIR     := examples

# Compiler flags
CXXFLAGS   := -std=c++17 -O2 -Wall -Wextra -I$(INC_DIR)
NVCCFLAGS  := -std=c++17 -O2 -I$(INC_DIR) --expt-relaxed-constexpr
ARFLAGS    := rcs

# ── Platform detection ───────────────────────────────────────
UNAME_S    := $(shell uname -s)
UNAME_M    := $(shell uname -m)

CUDA_PATH  ?= /usr/local/cuda
HAS_NVCC   := $(shell command -v nvcc 2>/dev/null && echo yes || echo no)

# ── Compression library detection (.spire codec — gzip/xz/bz2) ───────────────
# Mirrors the HAS_NVCC pattern: probe via pkg-config, define SPIKECOREC_HAVE_*
# and append link flags only when found. Missing libraries degrade gracefully —
# raw .spire always works; SpireWriter/Reader throw a clear runtime_error for
# compressed variants whose library wasn't available at build time.
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

# ── XML parsing (libxml2 — NML/LEMS front-end, ticket #2 [A3]) ───────────────
# Parse + XSD-validate NeuroML/LEMS XML (third_party/neuroml2/schema, std_lib).
# Detected the same way as the compression codecs above, via pkg-config.
HAS_LIBXML2 := $(shell pkg-config --exists libxml-2.0 && echo yes || echo no)

ifeq ($(HAS_LIBXML2),yes)
  CXXFLAGS     += -DSPIKECOREC_HAVE_LIBXML2 $(shell pkg-config --cflags libxml-2.0)
  LIBXML2_LIBS := $(shell pkg-config --libs libxml-2.0)
endif

# ── NeuroML/LEMS standard-library bundle path (vendored, ticket #3 [A1]) ─────
# Baked in as an absolute path at compile time (via $(abspath), a GNU Make
# built-in) so NML_StandardLibrary::STANDARD_LIBRARY_PATH resolves correctly
# no matter what directory the binary is later run from.
NML_STD_LIB_DIR := $(abspath third_party/neuroml2/std_lib)
CXXFLAGS        += -DSPIKECOREC_NML_STD_LIB_DIR=\"$(NML_STD_LIB_DIR)\"

# ── NeuroML2 XSD schema path (vendored, ticket #8 [A2]) ──────────────────────
NML_SCHEMA_PATH := $(abspath third_party/neuroml2/schema/NeuroML_v2.3.xsd)
CXXFLAGS         += -DSPIKECOREC_NML_SCHEMA_PATH=\"$(NML_SCHEMA_PATH)\"

# ── Test fixture data directory ──────────────────────────────────────────────
# Baked in as an absolute path at compile time, same convention as
# NML_STD_LIB_DIR above, so tests can locate their checked-in .nml fixtures /
# captured pyneuroml reference data no matter what directory the test binary is
# run from.
TEST_FIXTURES_DIR := $(abspath tests/fixtures)
CXXFLAGS          += -DSPIKECOREC_TEST_FIXTURES_DIR=\"$(TEST_FIXTURES_DIR)\"

# ── Logging (spdlog, header-only, vendored submodule) ─────────────────────────
# Header-only mode is spdlog's default (it self-defines SPDLOG_HEADER_ONLY
# whenever SPDLOG_COMPILED_LIB isn't set) — only the active-level gate is ours to set.
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
  METALFLAGS    := -O2
  METAL_LDFLAGS := -framework Metal -framework Foundation -framework QuartzCore
  CXXFLAGS      += -Ithird_party/metal-cpp -DSPIKECOREC_METAL
else
  HAS_METAL  := no
  # Core .cpp files (compiled with g++) include <cuda_runtime.h>/<cuda.h>, so the
  # CUDA headers must be on the include path for the host compiler too — not just nvcc.
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

# ── Header dependency tracking ───────────────────────────────
# Without this, editing a header does not rebuild the .o files that include it,
# while tests ARE recompiled every run — so new test code links against a stale
# library. That mismatch shows up as a segfault in an unrelated test rather than
# as a compile error, which is exactly as confusing as it sounds. -MMD emits a
# .d file of each object's header prerequisites alongside it; -MP adds phony
# targets so a deleted header does not wedge the build with a missing-prereq
# error. The generated .d files are included below, once the object lists exist.
DEPFLAGS    := -MMD -MP
ALL_DEP_FILES := $(patsubst %.o,%.d,$(CORE_OBJS) $(NML_OBJS) $(METAL_OBJS))

# ── Python toolchain ─────────────────────────────────────────
# Building the extension needs pybind11; running python/tests additionally needs numpy and
# pytest. None of the three is needed by the C++/Metal/CUDA build, so point this at whatever
# interpreter has them:
#     make python PYTHON=/path/to/venv/bin/python
#     make python-test PYTHON=/path/to/venv/bin/python
PYTHON     ?= python3
PY_INC     := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
PYBIND_INC := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)

# ── Top-level targets ────────────────────────────────────────
.PHONY: all cuda metal python python-test test examples run-examples check clean info

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
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# ── NML/LEMS front-end objects (platform-agnostic C++) ───────
$(BUILD_DIR)/nml/%.o: $(SRC_DIR)/nml/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# ── CUDA objects (.cu) ───────────────────────────────────────
$(BUILD_DIR)/cuda/%.o: $(SRC_DIR)/cuda/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ── Metal objects (.cpp via metal-cpp) ───────────────────────
$(BUILD_DIR)/metal/%.o: $(SRC_DIR)/metal/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# ── Metal shaders (.metal → .air → .metallib) ────────────────
$(BUILD_DIR)/metal/%.air: $(SRC_DIR)/metal/%.metal
	@mkdir -p $(@D)
	$(METALC) $(METALFLAGS) -c $< -o $@

$(BUILD_DIR)/default.metallib: $(AIR_FILES)
	@mkdir -p $(@D)
	$(METALLIB) $(AIR_FILES) -o $@

# ── Python extension ─────────────────────────────────────────
# Builds an editable install so `import spikecorec` works from the repo root.
# pybind11 must be installed: pip install pybind11
python:
ifeq ($(PY_INC),)
	$(error Python headers not found. Run: pip install pybind11)
endif
ifeq ($(PYBIND_INC),)
	$(error pybind11 not found. Run: pip install pybind11)
endif
	SPIKECOREC_BACKEND=$(BACKEND) $(PYTHON) -m pip install -e . --no-build-isolation -q
ifeq ($(BACKEND),metal)
	$(MAKE) $(BUILD_DIR)/default.metallib
	cp $(BUILD_DIR)/default.metallib python/spikecorec/default.metallib
endif
	@echo "[spikecorec] Python extension installed (backend=$(BACKEND))"

# Builds the extension and runs python/tests against it. A few seconds: the fixture is an
# 8-neuron ring run for a few thousand ticks.
#
# Deliberately NOT wired into `make check`. `make check` has to stay runnable with no
# arguments, and this cannot be: PYTHON defaults to `python3`, and whether that particular
# interpreter has pybind11, numpy and pytest is a property of the machine rather than of the
# repository. A `make check` that failed on a missing pytest would be reporting the
# environment instead of the code. Run it explicitly:
#     make python-test PYTHON=/path/to/venv/bin/python
python-test: python
	$(PYTHON) -m pytest python/tests

# ── Tests ────────────────────────────────────────────────────
.PHONY: test test-cuda test-metal

test: test-$(BACKEND)

CUDA_LINK := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs -lcudart -lcuda -lnvrtc

$(GTEST_OBJ): $(GTEST_DIR)/src/gtest-all.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) -c $< -o $@

test-cuda: check-cuda $(CUDA_LIB) $(GTEST_OBJ)
	$(CXX) $(CXXFLAGS) $(TEST_CORE_SRCS) $(GTEST_OBJ) \
	    -L$(BUILD_DIR) -l$(PROJECT)_cuda $(CUDA_LINK) $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread \
	    -o $(BUILD_DIR)/test_runner_cuda
	$(BUILD_DIR)/test_runner_cuda

test-metal: check-metal $(METAL_LIB) $(BUILD_DIR)/default.metallib $(GTEST_OBJ)
	$(CXX) $(CXXFLAGS) $(TEST_CORE_SRCS) $(GTEST_OBJ) \
	    -L$(BUILD_DIR) -l$(PROJECT)_metal $(METAL_LDFLAGS) $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread \
	    -o $(BUILD_DIR)/test_runner_metal
	$(BUILD_DIR)/test_runner_metal

# ── Examples ─────────────────────────────────────────────────
# One binary per examples/*.cpp, into build/examples/. Programs under
# examples/unsupported/ are deliberately NOT picked up by EX_SRCS' wildcard — each
# needs an engine capability that does not exist yet, and each says which in its own
# header comment. See examples/README.md.
EX_BINS := $(patsubst $(EX_DIR)/%.cpp, $(BUILD_DIR)/examples/%, $(EX_SRCS))
EX_HDRS := $(wildcard $(EX_DIR)/*.h)

examples: examples-$(BACKEND)
	@echo "[spikecorec] examples built → $(BUILD_DIR)/examples/"

examples-cuda: check-cuda $(CUDA_LIB) $(EX_BINS)

examples-metal: check-metal $(METAL_LIB) $(BUILD_DIR)/default.metallib $(EX_BINS)

ifeq ($(BACKEND),cuda)
$(BUILD_DIR)/examples/%: $(EX_DIR)/%.cpp $(EX_HDRS) | $(CUDA_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< \
	    -L$(BUILD_DIR) -l$(PROJECT)_cuda $(CUDA_LINK) \
	    $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread -o $@
else
$(BUILD_DIR)/examples/%: $(EX_DIR)/%.cpp $(EX_HDRS) | $(METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< \
	    -L$(BUILD_DIR) -l$(PROJECT)_metal $(METAL_LDFLAGS) \
	    $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread -o $@
endif

# ── Running the examples ─────────────────────────────────────
# `make test` compiles and runs the test suite and nothing else -- not one line of
# examples/ is even compiled by it, which is how every example in this directory can be
# broken while the suite reports a full pass. `make run-examples` closes that hole: it
# builds every example AND runs it, and fails if any of them exits non-zero.
#
# Each example is run with its recordings pointed into build/example_runs/<name>/ so a
# check never writes into the working tree, and its console output is kept alongside in
# <name>.log so a failure can be read after the fact.
EXAMPLE_RUN_DIR := $(BUILD_DIR)/example_runs

# The full gate. Split across two sub-makes rather than named as prerequisites so the
# tests always run before the examples, whatever -j the caller passed.
check:
	$(MAKE) test
	$(MAKE) run-examples

run-examples: examples
	@rm -rf $(EXAMPLE_RUN_DIR)
	@mkdir -p $(EXAMPLE_RUN_DIR)
	@failed_examples=""; \
	for example_binary in $(EX_BINS); do \
	    example_name=`basename $$example_binary`; \
	    printf '[spikecorec] %-32s ' "$$example_name"; \
	    if $$example_binary --record-dir $(abspath $(EXAMPLE_RUN_DIR))/$$example_name \
	            > $(EXAMPLE_RUN_DIR)/$$example_name.log 2>&1; then \
	        echo "ok"; \
	    else \
	        echo "FAILED"; \
	        tail -12 $(EXAMPLE_RUN_DIR)/$$example_name.log | sed 's/^/                 | /'; \
	        failed_examples="$$failed_examples $$example_name"; \
	    fi; \
	done; \
	if [ -n "$$failed_examples" ]; then \
	    echo "[spikecorec] examples that did not exit 0:$$failed_examples"; \
	    exit 1; \
	fi; \
	echo "[spikecorec] all $(words $(EX_BINS)) examples ran to completion"

# ── Demos ────────────────────────────────────────────────────
# `make demo-videos` builds the demo programs, runs every GLIF variant and renders each run's
# recordings to a playable video, in one command. That is the whole point of the target: the
# videos are a build product, regenerated from scratch whenever the model underneath them
# changes, and never checked in (they land under build/, which .gitignore already covers).
#
# Kept out of EX_SRCS' wildcard deliberately -- these are examples/demos/*.cpp, not
# examples/*.cpp -- because `make check` runs every example on every invocation and a demo is
# far too slow for that: each renders several hundred video frames through matplotlib.
# `make demos` builds them; `make run-demos` runs them without rendering, which IS quick.
DEMO_DIR        := $(EX_DIR)/demos
DEMO_SRCS       := $(wildcard $(DEMO_DIR)/*.cpp)
DEMO_BINS       := $(patsubst $(DEMO_DIR)/%.cpp, $(BUILD_DIR)/demos/%, $(DEMO_SRCS))
DEMO_HDRS       := $(wildcard $(DEMO_DIR)/*.h) $(EX_HDRS)
DEMO_OUTPUT_DIR := $(BUILD_DIR)/demo_videos
DEMO_VARIANTS   := glif1 glif2 glif3 glif4 glif5
DEMO_SIDE       := 48

# render_spire_video.py sizes its spike markers to stay inside one grid cell, which on a
# 48-wide grid is small enough that the spikes read as specks against the membrane heatmap
# behind them. A demo wants the opposite emphasis -- the spikes ARE the subject -- so the
# markers are widened to about a cell. Retune this alongside DEMO_SIDE.
DEMO_SPIKE_SIZE := 34

# Rendering needs numpy + matplotlib, which the C++/Metal/CUDA build itself does not. Point this
# at any interpreter that has them:
#     make demo-videos DEMO_PYTHON=/path/to/venv/bin/python
DEMO_PYTHON     ?= python3

.PHONY: demos demos-cuda demos-metal run-demos demo-videos

demos: demos-$(BACKEND)
	@echo "[spikecorec] demos built → $(BUILD_DIR)/demos/"

demos-cuda: check-cuda $(CUDA_LIB) $(DEMO_BINS)

demos-metal: check-metal $(METAL_LIB) $(BUILD_DIR)/default.metallib $(DEMO_BINS)

ifeq ($(BACKEND),cuda)
$(BUILD_DIR)/demos/%: $(DEMO_DIR)/%.cpp $(DEMO_HDRS) | $(CUDA_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< \
	    -L$(BUILD_DIR) -l$(PROJECT)_cuda $(CUDA_LINK) \
	    $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread -o $@
else
$(BUILD_DIR)/demos/%: $(DEMO_DIR)/%.cpp $(DEMO_HDRS) | $(METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< \
	    -L$(BUILD_DIR) -l$(PROJECT)_metal $(METAL_LDFLAGS) \
	    $(COMPRESSION_LIBS) $(LIBXML2_LIBS) -lpthread -o $@
endif

# Every variant simulated and recorded, no rendering. Seconds, not minutes.
run-demos: demos
	@rm -rf $(DEMO_OUTPUT_DIR)
	@mkdir -p $(DEMO_OUTPUT_DIR)
	@set -e; for demo_variant in $(DEMO_VARIANTS); do \
	    run_directory=$(abspath $(DEMO_OUTPUT_DIR))/$$demo_variant; \
	    mkdir -p $$run_directory; \
	    echo "[spikecorec] running $$demo_variant demo"; \
	    if ! $(BUILD_DIR)/demos/glif_family_demo --variant $$demo_variant \
	            --side $(DEMO_SIDE) --record-dir $$run_directory \
	            > $$run_directory/run.log 2>&1; then \
	        cat $$run_directory/run.log; \
	        echo "[spikecorec] $$demo_variant demo did not exit 0"; \
	        exit 1; \
	    fi; \
	    cat $$run_directory/run.log; \
	done
	@echo "[spikecorec] demo recordings → $(DEMO_OUTPUT_DIR)/<variant>/"

# The one command that rebuilds every demo and regenerates every video from scratch.
demo-videos: run-demos
	@set -e; for demo_variant in $(DEMO_VARIANTS); do \
	    run_directory=$(abspath $(DEMO_OUTPUT_DIR))/$$demo_variant; \
	    echo "[spikecorec] rendering $$demo_variant demo"; \
	    $(DEMO_PYTHON) $(EX_DIR)/render_spire_video.py \
	        $$run_directory/$${demo_variant}_demo_spikes.spire \
	        --membrane $$run_directory/$${demo_variant}_demo_membrane.spire \
	        --side $(DEMO_SIDE) --spike-size $(DEMO_SPIKE_SIZE) \
	        --output $(abspath $(DEMO_OUTPUT_DIR))/$${demo_variant}_demo.gif; \
	done
	@echo "[spikecorec] demo videos → $(DEMO_OUTPUT_DIR)/<variant>_demo.gif"

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

# Header prerequisites discovered by -MMD; must come after the object lists.
-include $(ALL_DEP_FILES)
