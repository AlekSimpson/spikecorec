# ============================================================
# spikecorec — Makefile
# Supports CUDA and Metal backends + Python bindings
# Usage:
#   make              — auto-detect backend and build
#   make cuda         — build CUDA backend
#   make metal        — build Metal backend
#   make python       — build Python extension (pip editable install)
#   make test         — build and run C++ tests
#   make examples     — build examples
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
  CXXFLAGS   += -DSPIKECOREC_CUDA
endif

# Auto-select default backend
ifeq ($(UNAME_S),Darwin)
  BACKEND ?= metal
else
  BACKEND ?= cuda
endif

# ── Source files ─────────────────────────────────────────────
CORE_SRCS      := $(wildcard $(SRC_DIR)/core/*.cpp)
CUDA_SRCS      := $(wildcard $(SRC_DIR)/cuda/*.cu)
METAL_SRCS     := $(wildcard $(SRC_DIR)/metal/*.cpp)
METAL_SHADERS  := $(wildcard $(SRC_DIR)/metal/*.metal)

TEST_CORE_SRCS  := $(wildcard $(TEST_DIR)/*.cpp)
TEST_CUDA_SRCS  := $(wildcard $(TEST_DIR)/cuda/*.cpp)
TEST_METAL_SRCS := $(wildcard $(TEST_DIR)/metal/*.cpp)

EX_SRCS        := $(wildcard $(EX_DIR)/*.cpp)

# ── Object / artifact paths ──────────────────────────────────
CORE_OBJS   := $(patsubst $(SRC_DIR)/%.cpp,  $(BUILD_DIR)/%.o,   $(CORE_SRCS))
CUDA_OBJS   := $(patsubst $(SRC_DIR)/%.cu,   $(BUILD_DIR)/%.o,   $(CUDA_SRCS))
METAL_OBJS  := $(patsubst $(SRC_DIR)/%.cpp,  $(BUILD_DIR)/%.o,   $(METAL_SRCS))
AIR_FILES   := $(patsubst $(SRC_DIR)/metal/%.metal, \
                           $(BUILD_DIR)/metal/%.air, $(METAL_SHADERS))

CUDA_LIB    := $(BUILD_DIR)/lib$(PROJECT)_cuda.a
METAL_LIB   := $(BUILD_DIR)/lib$(PROJECT)_metal.a

# ── Python toolchain ─────────────────────────────────────────
PYTHON     ?= python3
PY_INC     := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
PYBIND_INC := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)

# ── Top-level targets ────────────────────────────────────────
.PHONY: all cuda metal python test examples clean info

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
$(CUDA_LIB): $(CORE_OBJS) $(CUDA_OBJS)
	@mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $^

$(METAL_LIB): $(CORE_OBJS) $(METAL_OBJS)
	@mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $^

# ── Core objects (platform-agnostic C++) ─────────────────────
$(BUILD_DIR)/core/%.o: $(SRC_DIR)/core/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ── CUDA objects (.cu) ───────────────────────────────────────
$(BUILD_DIR)/cuda/%.o: $(SRC_DIR)/cuda/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ── Metal objects (.cpp via metal-cpp) ───────────────────────
$(BUILD_DIR)/metal/%.o: $(SRC_DIR)/metal/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

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

# ── Tests ────────────────────────────────────────────────────
.PHONY: test test-cuda test-metal

test: test-$(BACKEND)

test-cuda: check-cuda $(CUDA_LIB)
	$(CXX) $(CXXFLAGS) $(TEST_CORE_SRCS) $(TEST_CUDA_SRCS) \
	    -L$(BUILD_DIR) -l$(PROJECT)_cuda -lcudart -L$(CUDA_PATH)/lib64 \
	    -o $(BUILD_DIR)/test_runner_cuda
	$(BUILD_DIR)/test_runner_cuda

test-metal: check-metal $(METAL_LIB)
	$(CXX) $(CXXFLAGS) $(TEST_CORE_SRCS) $(TEST_METAL_SRCS) \
	    -L$(BUILD_DIR) -l$(PROJECT)_metal $(METAL_LDFLAGS) \
	    -o $(BUILD_DIR)/test_runner_metal
	$(BUILD_DIR)/test_runner_metal

# ── Examples ─────────────────────────────────────────────────
examples: examples-$(BACKEND)

examples-cuda: check-cuda $(CUDA_LIB)
	$(NVCC) $(NVCCFLAGS) $(EX_DIR)/cuda_example.cpp \
	    -L$(BUILD_DIR) -l$(PROJECT)_cuda \
	    -o $(BUILD_DIR)/cuda_example

examples-metal: check-metal $(METAL_LIB)
	$(CXX) $(CXXFLAGS) $(EX_DIR)/metal_example.cpp \
	    -L$(BUILD_DIR) -l$(PROJECT)_metal $(METAL_LDFLAGS) \
	    -o $(BUILD_DIR)/metal_example

# ── Utilities ────────────────────────────────────────────────
info:
	@echo "Platform   : $(UNAME_S) $(UNAME_M)"
	@echo "CUDA found : $(HAS_NVCC)"
	@echo "Metal found: $(HAS_METAL)"
	@echo "Backend    : $(BACKEND)"
	@echo "CXX        : $(CXX)"

compdb:
	bear -- make $(BACKEND) 2>/dev/null; true
	@echo "[spikecorec] compile_commands.json updated"

clean:
	rm -rf $(BUILD_DIR)
