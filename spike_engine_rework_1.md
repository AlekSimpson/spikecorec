# Finish the backend rework on `agent_integration_2`

## Context

The old GPU memory layer (`GpuPointer<T>`, `allocate<T>()`, `EngineAllocator`, `metal_dispatch`, and the
whole `src/metal/` + `src/cuda/` trees) has been deleted and replaced with `include/spikecorec/core/backend.h`
+ `src/core/backend.cpp`: an `EnginePointer` value handle plus `MetalBackend`/`CudaBackend` that partition a
slab and dispatch through `create_function` / `run_function`. The headers for `K2Tree` and `WeightMatrix` were
converted; their `.cpp` files were not. Nothing compiles today, and 11 TODO comments mark the open questions.

This plan does three things:

1. **Answers and closes every TODO** (except `backend.cpp:162`, deliberately skipped), with the code that
   goes in.
2. **Finishes the port** to a compiling Metal build and a green C++ test suite.
3. **Removes per-edge value storage from `WeightMatrix` entirely.** Today weights, delays, synapse prototype
   indices and synapse state variables all live in padded `[node_count × max_neighbor_count]` f32 planes,
   uncompressed. Section C removes that, with no fallback that restores it.

### Working rules for this plan

- **No new structs.** Where several values travel together, use parallel arrays (structure-of-arrays), which
  is also what the GPU wants — an array of pairs forces a strided load of one field at a time.
- **Lean on `WeightMatrix`.** Anything per-edge is its job, through the mechanisms it already has: the
  k²-tree for which pairs exist, the shared `U`/`V` basis with per-matrix `Ck` for their values.
- **No exact/raw-storage mode.** `enable_exact_edge_weights` and its machinery are deleted, not gated.
- **No new backend types.** `partition`/`allocate` stay methods on `MetalBackend`/`CudaBackend`;
  `AbstractBackend` is not restructured; `EnginePointer` gains no owner pointer.
- Scalars reach the kernel as `EnginePointer` wrappers around host variables with `offset = 0`.
- `EngineBackend &` is threaded explicitly into `K2Tree` and `WeightMatrix`.
- The ahead-of-time `default.metallib` pipeline is restored; scratch buffers are preallocated.
- **Hebbian plasticity is the engine's existing built-in rule, off by default**, switched on by one
  construction flag. Nothing is lowered from LEMS, and "off" costs nothing rather than costing a branch.

### What this supersedes in `CLAUDE.md`

Treat these as stale once this plan lands, and update `CLAUDE.md` at the end:

| Claim | Status |
|---|---|
| #53 (D3) "sparse delta buffers `Sk`" as per-edge planes | The planes were never sparse. C5 makes the delta store bounded and SoA. |
| #54 (D4) periodic `refit()` | Deleted (C6). With no stored ground truth there is nothing to fit against. |
| #57 (D1) "aggregated accumulators" as a future ticket | Becomes the core of C1, not a later optimization. |
| #64 (F3) per-edge delay as a registered matrix "for a future consumer" | Delay is a live matrix in the family now (C3). |
| #66 (F5) "plasticity wiring" as Phase 2 | The delta path is designed here (C5), using the engine's existing built-in Hebbian rule, off by default behind a construction flag. Nothing is lowered from LEMS. |
| "`network_inputs` is *the* per-neuron accumulator" | Becomes one accumulator per (neuron, prototype) (C1). |
| "U/V factorization = memory savings, the engine does not train on tasks" | **Still true and still load-bearing.** Nothing here changes it. |

---

# A. Blockers that gate everything else

Not TODOs, but nothing below can be tested until they land. In this order.

### A1. `backend.h` is missing its closing brace

`namespace spikecorec::backend {` opens at line 14 and never closes — brace balance is 22 open, 21 close.
Every header included after `backend.h` lands *inside* that namespace, which is why `engine.cpp` reports
hundreds of `no member named 'memcpy' in namespace 'spikecorec::backend::std'` errors. Fix this before
reading any other diagnostic; most of them are phantoms of this one.

Nothing anywhere does `using namespace spikecorec::backend`, so `EnginePointer`, `EngineBackend`,
`EngineFunction` and the `EngineDatatype` enumerators are unresolvable in `engine.h`, `k2tree.h` and
`weight_matrix.h`. **Recommendation: drop the nested namespace** and declare these in `namespace spikecorec`
alongside every other core type. The project convention is `spikecorec::metal` / `spikecorec::cuda` for
backend-*specific* code, not a generic `backend` layer.

### A2. `backend.h` fixes

**`EngineDatatype`** — add the two missing types. `UNSIGNED16` is required because
`K2Tree::rank_subblock_table` is `u16`; `FLOAT32X4` is required for `U_matrix` / `V_matrix` (D2).

```cpp
enum class EngineDatatype {
    SIGNED32, SIGNED64, FLOAT32, FLOAT64, FLOAT32X4,
    UNSIGNED8, UNSIGNED16, UNSIGNED32, UNSIGNED64, BOOLEAN,
};
```

**`AbstractBackend`** — the only change is qualifying the enumerators in the `type_alignments` initializer,
which is a hard compile error as written (`{SIGNED32, 4}` on a scoped enum). Structure, fields and their
meanings are untouched:

```cpp
    UnorderedMap<EngineDatatype, u64> type_alignments = {
        {EngineDatatype::SIGNED32,   4},
        {EngineDatatype::SIGNED64,   8},
        {EngineDatatype::FLOAT32,    4},
        {EngineDatatype::FLOAT64,    8},
        {EngineDatatype::FLOAT32X4, 16},
        {EngineDatatype::UNSIGNED8,  1},
        {EngineDatatype::UNSIGNED16, 2},
        {EngineDatatype::UNSIGNED32, 4},
        {EngineDatatype::UNSIGNED64, 8},
        {EngineDatatype::BOOLEAN,    1},
    };
```

`last_offset`, `max_bytes`, `memory_has_been_allocated` and `base_pointer` stay exactly as they are — B
finally uses all four for their evident purpose. `threads_per_block` needs no declaration change; the
constructors just have to stop initializing it in a derived member-init list (A3).

**`EnginePointer`** — same four fields, plus one flag, plus a typed accessor. No owner pointer.

```cpp
// A non-owning handle to a byte range.
//
// base_pointer is the platform's own handle to the allocation, NOT the address of the
// bytes: MTL::Buffer * on Metal, the device address on CUDA. Metal binds buffer objects
// rather than addresses, so run_function needs the buffer itself and the offset has to
// travel beside it -- get_contents() is what reaches the bytes through the pair.
//
// The exception is inline_scalar: a kernel argument bound by value rather than as a
// buffer, where base_pointer is the host address of the value and offset is 0.
struct EnginePointer {
    void *base_pointer  = nullptr;
    s64   offset        = 0;
    u64   total_bytes   = 0;
    u64   alignment     = 1;
    bool  inline_scalar = false;

    [[nodiscard]] void *get_contents() const {
        if (base_pointer == nullptr) return nullptr;
        if (inline_scalar) return base_pointer;
#ifdef SPIKECOREC_METAL
        return static_cast<u8 *>(static_cast<MTL::Buffer *>(base_pointer)->contents()) + offset;
#else
        return static_cast<u8 *>(base_pointer) + offset;
#endif
    }

    template <typename ElementType>
    [[nodiscard]] ElementType *get_contents_as() const {
        return static_cast<ElementType *>(get_contents());
    }

    [[nodiscard]] bool is_empty() const { return base_pointer == nullptr || total_bytes == 0; }
};
```

This is the shape the deleted `GpuPointer` already proved on both backends — `MTL::Buffer *` plus
`offset_bytes` on Metal, a bare device address on CUDA — minus the move-only ownership discipline, which
`EnginePointer` does not need because it owns nothing.

Every field needs its default initializer: `K2Tree() = default` and `WeightMatrix() = default` currently
produce indeterminate `offset` / `total_bytes` / `alignment`.

**Remaining header fixes:**

- `base_pointer + offset` is arithmetic on `void *` — ill-formed; the `static_cast<u8 *>` above fixes it.
- `MetalBackend` and `CudaBackend` are both defined unconditionally while naming platform-only types
  (`CUdevice`, `MTL::Device`). Wrap each in its own `#ifdef`.
- `device = MTL::CreateSystemDefaultDevice()` as a default member initializer means every `MetalBackend`
  construction grabs a device, and `~MetalBackend` releases it with no deleted copy constructor — a
  double-release. Make the backend non-copyable and create the device in the constructor body.
- `MetalBackend` needs a cached `MTL::CommandQueue *queue`, created once in the constructor. Today
  `run_function` calls `newCommandQueue()` per dispatch and never releases it (D7).
- `base_memory_pointer` becomes `Vector<MTL::Buffer *> slabs` (Metal) / `Vector<void *> slabs` (CUDA), so
  one backend can hold the several chunks section B is about. Both are derived members.
- `SpikeEngine` declares `EngineBackend gpu;` with no default constructor available. Give
  `threads_per_block` a default of 256 (matching the deleted `thread_count_per_block`) or construct `gpu`
  in `SpikeEngine`'s member-init list.

### A3. `backend.cpp` signature and typo fixes

Mechanical but numerous: `return this;` from reference-returning `partition` (×4) → `return *this;`;
`void *MetalBackend::allocate()` vs. the header's `EnginePointer allocate()`; `EngineFunction
create_function` vs. the header's `Optional<EngineFunction>` (keep `Optional`, and make `engine.cpp:123`
handle absence); `element_count` / `threads_per_group` undeclared in `run_function` (meant `job_count` /
`threads_per_block`); `new_partitions` typo in `CudaBackend::partition`, which also never pushes into
`partitions` nor increments `max_bytes`; `CudaBackend::allocate` returning `nullptr` from an `EnginePointer`
function; `CUfuntion` typo; `s32 job_count` vs. the header's `s64`; `combine(name, ".cu")` matches no
overload (`spikecorec.h` has `combine(const Vector<String>&)` and `combine(String&, const Vector<String>&)`,
and that header includes the deleted `kernels.cuh` anyway — use `(name + ".cu").c_str()`);
`nvrtcCreateProgram` / `cuModuleGetFunction` need `.c_str()`; `<cuda_runtime.h>` is missing despite
`cudaMallocManaged` / `cudaFree`; and the local `Vector<char> log(log_size)` shadows the `log` namespace two
lines above a `logger()` call. Both constructors must assign `threads_per_block` in the body rather than in
the member-init list, since it is a base-class member.

### A4. Restore the three deleted `src/metal` files

```bash
mkdir -p src/metal
git show 841df1b:src/metal/metal_cpp_impl.cpp     > src/metal/metal_cpp_impl.cpp
git show 841df1b:src/metal/k2tree_device.metalinc > src/metal/k2tree_device.metalinc
git show 841df1b:src/metal/kernels.metal          > src/metal/kernels.metal
```

- `metal_cpp_impl.cpp` is the single translation unit defining `NS_/MTL_/CA_PRIVATE_IMPLEMENTATION`.
  Without it every `MTL::` / `NS::` symbol is undefined at link.
- `k2tree_device.metalinc` (185 lines) is read at *runtime* by `read_device_include` and prepended to every
  generated kernel (`src/nml/dynamics_codegen.cpp:1139`). It defines `MAX_K2TREE_HEIGHT` and
  `k2t_next_neighbor`, both used by the generated propagate stage. Without it **every `SpikeEngine`
  construction throws** at `engine.cpp:120`, before dispatch is ever reached.
- `kernels.metal` is the AOT shader source behind `gpu_neighbor_weights`, `gpu_scale_uv`,
  `gpu_weight_update`, `gpu_k2tree_adjacent_batch` and `gpu_k2tree_get_neighbors_batch`. Its
  `weight_update_kernel` is also the arithmetic C5 reuses.

### A5. `make clean` before trusting any build

`build/libspikecorec_metal.a` still contains `engine_allocator.o` and `metal_cpp_impl.o` from the Aug 13
build. `ar rcs` replaces named members but never evicts unnamed ones, so a link may appear to succeed
against months-old object code that matches no source. Note `make clean` is `rm -rf build`, which also
deletes the generated demo artifacts under `build/demos/` — check whether any are wanted first.

### A6. Makefile and setup.py

A4 restores `src/metal/`, so `METAL_SRCS`, `METAL_SHADERS` and `METAL_DEVICE_DIR` all resolve and the
`.metal → .air → .metallib` rules have inputs again. Remaining: delete the dead `$(BUILD_DIR)/cuda/%.o`
rule (`src/cuda/` stays deleted) and the `examples/cuda_example.cpp` reference in `examples-cuda`, which
never existed. `setup.py:117` still lists `src/cuda/kernels.cu` — remove it; `setup.py:129`'s
`src/metal/metal_cpp_impl.cpp` is fixed by A4. `setup.py:76-85` also omits `src/core/units.cpp`,
`src/nml/nml.cpp` and `src/nml/dynamics_codegen.cpp` from `SRCS`, a pre-existing link gap.

---

# B. Making `partition` + `allocate` reusable

> **TODO** `include/spikecorec/core/backend.h:85` — *"for both cuda and metal make the partition + allocate
> pattern be reusable so that we can allocate multiple different chunks for distinct parts of the engine
> like WeightMatrix/K2Tree/SpikeEngine"*

Everything stays inside `MetalBackend` / `CudaBackend`. No new type, no change to `AbstractBackend`'s
structure — the four fields it already carries are exactly what this needs, and `last_offset` finally gets
used (it is currently declared and never read).

### Two bugs to fix in the same edit

- **`partition` reads the slab before it exists.** `MetalBackend::partition` does
  `new_partition.base_pointer = base_memory_pointer->get_contents()` while `engine.cpp:96-104` partitions
  first and allocates second, so every handle is filled from an uninitialized `MTL::Buffer *`. Partitions
  must record offsets only, and `allocate()` fills in the handle.
- **`alignment` is stored and never applied.** `offset = previous.offset + previous.total_bytes`. The
  deleted `EngineAllocator` guaranteed 16-byte alignment for exactly the `float4` reason.

### The change

`allocate()` takes the same partition vector so it can fill the handles in:

```cpp
    MetalBackend &partition(u64 bytes, EngineDatatype datatype, Vector<EnginePointer> &partitions);
    EnginePointer allocate(Vector<EnginePointer> &partitions);
    void deallocate_slab(const EnginePointer &slab);
    void deallocate_all();
```

```cpp
// Metal wants a 256-byte offset for a buffer bound to a `constant` kernel parameter, and
// float4 reads need 16. One floor covers both, and with a handful of partitions per chunk
// the padding is irrelevant.
static constexpr u64 PARTITION_ALIGNMENT = 256;

MetalBackend &MetalBackend::partition(
    u64 bytes, EngineDatatype datatype, Vector<EnginePointer> &partitions
) {
    // A finished allocate() means the next partition() opens a FRESH chunk rather than
    // trying to extend a slab that already exists. That is what makes the pattern
    // reusable: SpikeEngine, WeightMatrix and K2Tree each run their own
    // partition -> allocate round against the same backend, and each gets its own slab.
    if (memory_has_been_allocated) {
        memory_has_been_allocated = false;
        last_offset = 0;
        max_bytes   = 0;
    }

    const u64 alignment      = std::max<u64>(type_alignments[datatype], PARTITION_ALIGNMENT);
    const s64 aligned_offset = (s64)(((u64)last_offset + alignment - 1) & ~(alignment - 1));

    // base_pointer stays null until allocate() runs -- the slab does not exist yet, and
    // reading it here is what made every handle point at garbage.
    //
    // A zero-byte request is legal (a cell type with no parameters, a graph with no edges)
    // and yields an empty handle rather than a range nothing may be written to.
    partitions.push_back(
        bytes == 0 ? EnginePointer{}
                   : EnginePointer{nullptr, aligned_offset, bytes, alignment, false});

    last_offset = aligned_offset + (s64)bytes;
    max_bytes   = (u64)last_offset;
    return *this;
}
```

```cpp
EnginePointer MetalBackend::allocate(Vector<EnginePointer> &partitions) {
    if (memory_has_been_allocated) return base_pointer;

    memory_has_been_allocated = true;
    if (max_bytes == 0) return EnginePointer{};

    MTL::Buffer *slab = device->newBuffer(max_bytes, MTL::ResourceStorageModeShared);

    // newBuffer returns null when the device cannot satisfy the request, and every caller
    // writes through get_contents() immediately -- an unchecked null surfaces as a
    // segfault inside a memset rather than as an allocation failure. Exhaustion is the
    // expected failure mode at the network sizes this engine targets.
    if (slab == nullptr) {
        log::throw_runtime_error(logger(),
            fmt::format("MetalBackend::allocate: the GPU could not allocate {} bytes ({:.2f} MiB)",
                        max_bytes, (f64)max_bytes / (1024.0 * 1024.0)));
    }
    slabs.push_back(slab);

    // Only handles from THIS chunk are still null, so a caller reusing one vector across
    // chunks does not get its earlier handles re-aimed at the new slab.
    for (EnginePointer &partition : partitions) {
        if (partition.base_pointer != nullptr || partition.total_bytes == 0) continue;
        partition.base_pointer = slab;
    }

    base_pointer = EnginePointer{slab, 0, max_bytes, PARTITION_ALIGNMENT, false};
    return base_pointer;
}
```

`CudaBackend` is the same function with `cudaMallocManaged(&slab, max_bytes)` and a `void *` slab.

`deallocate_slab` releases one chunk so `K2Tree` and `WeightMatrix` can free their own storage in their
destructors; `deallocate_all` walks `slabs`. Each owner keeps the whole-chunk handle `allocate()` returned:

```cpp
void MetalBackend::deallocate_slab(const EnginePointer &slab) {
    if (slab.base_pointer == nullptr) return;
    MTL::Buffer *buffer = static_cast<MTL::Buffer *>(slab.base_pointer);
    slabs.erase(std::remove(slabs.begin(), slabs.end(), buffer), slabs.end());
    buffer->release();
}
```

Call sites read the same as today, with the vector passed to `allocate` as well:

```cpp
    Vector<EnginePointer> data_partitions;
    gpu.partition(sizeof(f32) * layout.cell_state_length, EngineDatatype::FLOAT32, data_partitions)
       .partition(sizeof(f32) * layout.cell_parameter_length, EngineDatatype::FLOAT32, data_partitions)
       // ...
    model_slab = gpu.allocate(data_partitions);
```

---

# C. Removing per-edge value storage from `WeightMatrix`

> **TODO** `include/spikecorec/core/weight_matrix.h:230` — *"what is this for? This isn't actually just storing
> all the per edge state in memory like we are NOT supposed to be is it?"*

**It is.** And not only `per_edge_variable_values` — every `sparse_delta_buffers` entry has the same shape.

## C0. What is happening today

`set_edge_weight` calls `enable_exact_edge_weights()` (`weight_matrix.cpp:619` → `:578`), which pins the
default matrix's Ck to all-zero in every lane. `set_edge_delay_ticks` does the same at `:536`, and
`configure_per_edge_variable_count` at `:337`. `build_weight_matrix` (`engine.cpp:234-264`) calls all three
on **every edge of every model**.

So on the NeuroML path every matrix is in exact mode: `U·Ck·Vᵀ ≡ 0`, and every weight, delay, prototype
index and synapse state variable is purely its dense Sk entry. U and V are allocated and carry no
information about any edge value. Footprint is
`(2 + per_edge_variable_count) × node_count × max_neighbor_count × 4` bytes — about **1.6 GB** at 1M
neurons, max degree 100, one synapse state variable, padded to the widest row in the graph.

The f32 argument that motivated exact mode (`weight_matrix.h:141-158`) is correct as far as it goes: a
5e-10 conductance cannot survive as a delta against an order-1 reconstruction. But the conclusion drawn
from it was wrong. U and V are seeded `N(0,1)` and never rescaled to the data. **The problem is the scale
of the basis, not the representation.**

Everything below removes the dense storage. There is no mode, flag or fallback that restores it.

**The governing invariant for all of section C.** The k²-tree is the source of truth for which `(i, j)` pairs
exist, and nothing in the engine ever asks for a value at a pair that is not an edge. So:

- **Nothing is sized by `node_count × max_neighbor_count` or by `node_count²`.** Anything per-edge is sized
  `total_edge_count` and indexed by the canonical edge ordinal (C7); anything per-node is sized
  `node_count`.
- **No fit, residual, or loss is ever evaluated off the support.** Every one of them gathers its rows
  through `get_neighbors()`. This is not only correctness — it is what makes the construction-time fit
  `O(total_edge_count · r²)` instead of `O(node_count² · r²)`.
- **Rank is bounded by the edge count, not the matrix dimensions.** See C4.

Where a claim below cites a rank or a footprint, it is derived on the sparse support. Any number that
implicitly assumed a dense `node_count × node_count` target would be wrong by orders of magnitude, and is a
bug in this plan rather than a limit of the design.

## C1. Synapse state variables — collapse to per-target accumulators

This removes `per_edge_variable_values` planes `1..k` outright.

**The argument.** `generate_synapse_body` (`dynamics_codegen.cpp:743-757`) throws on conductance-based
synapses and on synapses composing plasticity or block mechanisms. So every Phase-1 synapse has:

- linear state dynamics — a `TimeDerivative` handled by `emit_integrate_stage`;
- a current `i` that is a `DerivedVariable` linear in that state;
- **all parameters per-prototype**, read from `synapse_parameters[synapse_parameter_base + slot]`;
- exactly one per-edge quantity, `weight`, entering only through the `OnEvent` `StateAssignment`, linearly
  (`g = g + gbase*weight`).

For a target neuron `t` and prototype `p`, with `S_e` the state of edge `e`:

```
d/dt (Σ_e S_e) = Σ_e f(S_e) = f(Σ_e S_e)      f is linear and shared per prototype
     Σ_e S_e  += gbase · w_e                   the arrival increment is linear in w_e
     Σ_e i_e   = i(Σ_e S_e)                    i is linear in S
```

The sum obeys the same ODE as each term, so **one accumulator per (target, prototype, state variable)**
reproduces the per-edge computation *exactly* — not approximately.

**Sizing:** `synapse_state[prototype_count][state_variable_count][node_count]` f32 — 8 MB at 1M neurons,
one prototype, two state variables, against ~400 MB per plane today.

**Race discipline** — generalize the existing `network_inputs` double-buffering rather than inventing one:

```
synapse_arrivals[2][prototype_count][node_count]              // atomically scattered into the NEXT row
synapse_state[prototype_count][state_count][node_count]       // touched only by the target's own thread
```

- propagate (thread owns source `s`): `atomic_fetch_add(&synapse_arrivals[next_row][p][target], gbase*w_e)`
- integrate (thread owns target `t`): drain `synapse_arrivals[current_row][p][t]` into
  `synapse_state[p][·][t]`, advance one `dt`, clear the drained slot.

No thread reads a slot another is writing — the property the current `current_row`/`next_row` split
already buys.

**This is also a large speedup.** Today the propagate stage loads and stores every state variable on every
edge on every tick, whether or not anything arrived — the comment at `dynamics_codegen.cpp:928-931` says so
explicitly. Afterwards an edge touches memory only when a spike actually arrives (~0.1% of ticks at 1-10 Hz
with dt = 0.1 ms), and the decay runs `node_count × prototype_count` times per tick instead of
`total_edge_count` times.

**Codegen changes.** In `generate_synapse_body`, replace the `edge_variables[...]` load/store pair
(`dynamics_codegen.cpp:766-770` and `:820-824`) with a scatter of the arrival increment:

```cpp
    // No per-edge state left to read or write: the arrival increment is linear in the
    // edge's weight, so it lands directly in the target's per-prototype accumulator and
    // the decay happens once per target rather than once per edge.
    source << indent << "if (arrived) {\n"
           << indent << "    device atomic_float *arrival_slot = (device atomic_float *)\n"
           << indent << "        (synapse_arrivals + (next_row * PROTOTYPE_COUNT + synapse_prototype)\n"
           << indent << "                            * neuron_count + target);\n"
           << indent << "    atomic_fetch_add_explicit(arrival_slot, "
           << translate_expression(arrival_increment, symbols, synapse_type.name)
           << ", memory_order_relaxed);\n"
           << indent << "}\n";
```

and move `emit_integrate_stage` for the synapse into the *cell* body, where the target's thread runs it
once per prototype:

```cpp
    // stage 2, per target neuron: one decay per prototype, not one per incoming edge
    for (int prototype = 0; prototype < PROTOTYPE_COUNT; ++prototype) {
        const int state_slot   = prototype * neuron_count + neuron_index;
        const int arrival_slot = (current_row * PROTOTYPE_COUNT + prototype) * neuron_count + neuron_index;

        float state_0 = synapse_state[state_slot] + synapse_arrivals[arrival_slot];
        synapse_arrivals[arrival_slot] = 0.0f;

        state_0 += step_dt * (-state_0 / synapse_parameters[synapse_parameter_base[prototype] + TAU_SLOT]);
        synapse_state[state_slot] = state_0;

        network_input += state_0;   // the `i` exposure, summed across prototypes
    }
```

**Where aggregation stops.** A synapse whose current is nonlinear in its state (NMDA's Mg block), or one
with genuinely per-edge parameters (`blockingPlasticSynapse`), does not factor. Both are already rejected
by `generate_synapse_body`; keep the rejection and say in the throw message that aggregation is the reason,
so the constraint is discoverable when someone picks up nonlinear synapses.

## C2. Synapse prototype index — projection runs, as parallel arrays

Plane 0 of `per_edge_variable_values` stores each edge's `synapse_prototype_index` as a float. A NeuroML
`<projection>` names one synapse for all its connections (`src/nml/nml.cpp:1524`), with only
`<synapticConnection synapse="...">` overriding per-connection (`:1527-1531`). So the assignment is a
partition of the edge set into a handful of runs — usually one.

Prototype index is the one per-edge quantity that must **not** go through the shared basis: it selects a
`switch` case in the kernel, so a reconstruction error is a wrong synapse type rather than a small numeric
error, and kernel control flow should not depend on a fit.

Three parallel arrays, not an array of triples — this is also the layout `emit_integer_table` already
emits, so they bake into the kernel as `constant int[]` tables directly:

```cpp
// Which synapse prototype each edge uses, as runs over the canonical edge ordering (C7).
// Parallel arrays rather than one array of records: emit_integer_table bakes each as its
// own `constant int[]`, and the device reads one of them per lookup.
//
// A NeuroML projection names one synapse for every connection it declares, so this is
// O(projections) -- one entry for the overwhelmingly common single-projection network.
// Lookup is a binary search on projection_first_edge_ordinal.
Vector<s64> projection_first_edge_ordinal;
Vector<s64> projection_edge_count;
Vector<s32> projection_synapse_prototype;
```

**Degeneration guard.** A network wired connection-by-connection with mixed synapses would produce one run
per edge, which is worse than the 4-byte plane it replaces. When the run count exceeds a fraction of
`total_edge_count`, fall back to a packed `ceil(log2(prototype_count))`-bit selector over
`total_edge_count` — 2 bits per edge for three prototypes, which is near the information-theoretic floor
for the choice rather than raw storage. Log which representation was chosen.

## C3. Delay — a matrix in the shared basis, rounded

`<connectionWD delay="..."/>` is read off the **connection**, not the projection
(`src/nml/nml.cpp:1535-1539`), so per-edge delay variation is a first-class NeuroML feature. A run table
would degenerate to one run per edge on exactly the models that need it.

Delay is what `delay_matrix_index` was always meant to be: another matrix in the shared-basis family,
sharing `U`/`V` with its own `Ck`. The bug was pinning that `Ck` to zero. Unpinned, it works — and delay has
a property weights do not: **it is a small integer, so `round()` absorbs fit error.** A basis that captures
the projection block structure reproduces per-projection-constant delay exactly, and rounding keeps it
correct under small perturbation.

```cpp
// Reconstructed like any other value field, then rounded: delay is a whole number of
// ticks, so the rounding boundary is what makes a small fit error harmless where the
// same error in a weight would not be.
const s32 delay_ticks = (s32)lround(reconstruct_entry(DELAY_MATRIX_INDEX, source, target));
```

`constant_delay_ticks` stays as the fast path — the topology constructor takes a single
`connection_delay_seconds` for every edge, so that path is uniform by construction and needs no
reconstruction at all.

**The fidelity check for delay is exact-integer, not a tolerance** (C4): a delay off by one tick indexes the
wrong row of the spike-history ring, which is a simulation error rather than a rounding artefact. If exact
recovery fails at the maximum rank, construction throws naming the projection.

**What sharing one basis across two matrices actually costs.** One `U`/`V` serves every matrix, with only a
per-matrix `Ck` between them. Counting on the sparse support: `m` matrices contribute `m · total_edge_count`
constraints against `2·node_count·r + m·r` parameters, so

```
2 · node_count · r  ≳  m · total_edge_count        →       r  ≳  m · average_degree / 2
```

Two matrices (weight and delay) need **twice** the rank of weights alone — not "weight and delay must share
structure", which is what a dense-target reading would suggest. It is a factor of two on a number that is
already small for real networks, and C4's derivation discovers the actual value rather than assuming
either bound.

## C4. Weight — the shared basis, and deriving the rank

With C1-C3 done, `weight` is the only real-valued per-edge quantity left, and it is **static**: the
simulation kernel never writes it (plasticity does, through C5). That is precisely what `U·Ck·Vᵀ` exists for.

**Delete exact mode.** `enable_exact_edge_weights`, `using_exact_edge_weights` and
`pin_coefficient_vector_to_zero` all go, along with every branch that reads them
(`weight_matrix.cpp:337, 536, 578-581, 619, 727-734, 995-1000, 1189-1192`). Ck is never pinned; the
low-rank term is always the value.

### How much rank is actually needed

This matters because it decides whether the representation is honest, and the sparse support changes the
answer completely.

A rank-`r` factorization has `2·node_count·r` free parameters (less `r²` of gauge freedom, since
`U → UA`, `V → VA⁻ᵀ` leaves the product unchanged). **The constraints are only the edges that exist** —
`total_edge_count` of them, not `node_count²` — because the k²-tree is what says which pairs are real and
nothing ever asks for a value at a pair that is not an edge. So a solution generically exists when:

```
2 · node_count · r  −  r²   ≥   total_edge_count
                      r     ≳   average_degree / 2
```

For a degree-8 torus, `r ≈ 4`. For a degree-100 cortical-style network, `r ≈ 50`. `MAX_RANK_FLOAT4_STRIDE`
is 64, i.e. 256 lanes, so average degree up to about 512 is representable in principle. **This is a far
better result than a dense-matrix intuition suggests, and it is the sparse support that buys it.**

**`average_degree / 2` is a ceiling, not a target.** It is the rank at which an *arbitrary* field — every
edge independently chosen — becomes fittable. No real document is arbitrary: a uniform weight is rank 1, and
a per-projection-constant weight is rank ≤ the projection count, both of which are far below the ceiling and
independent of degree. The ceiling's only job is to bound the search and to say when a failure to fit is a
property of the field rather than of the rank.

**The fit ranges only over edges, and that is a complexity result as well as a correctness one.** The loss
is `Σ over existing edges` — nothing is evaluated, penalised or even visited at a pair the k²-tree says is
not an edge. So an alternating-least-squares solve for row `U[i]` is a least-squares problem in
`degree(i)` equations, not `node_count`, and one full sweep costs `O(total_edge_count · r²)` rather than
`O(node_count² · r²)`. At a million nodes that is the difference between a construction step and an
impossibility. Every fit and residual routine below must gather its rows through `get_neighbors()`, never
by scanning a row of the matrix.

Three caveats worth carrying:

- The count is a dimension argument — necessary, and generically sufficient. Degenerate supports can still
  fail: a complete bipartite block `K(a,b)` needs `r ≥ min(a,b)` on its own regardless of the global count,
  because the block is dense within itself. This is the one place density genuinely matters, and it is a
  property of the sub-block, not of the matrix.
- The fit is nonconvex, so a solution existing does not mean alternating least squares finds it. **The
  measured residual is the authority, never the formula.**
- Only at the ceiling does the basis cost `2·node_count·r·4 ≈ 4·total_edge_count` bytes, i.e. break even
  with one f32 per edge. Everywhere below it — which is where real models sit — the basis is
  `2·node_count·r·4` with `r` a small constant, so compression scales with the network while the edge count
  does not enter at all. That is what "rank controls compression fidelity" has always meant, stated in
  numbers.

**A subtlety in the current code that must be fixed while doing this:** `rank_float4_stride = (rank+3)/4`,
and the reconstruction sums every one of `rank_float4_stride * 4` lanes — all of which are seeded `N(0,1)`
(`weight_matrix.cpp:107-114`). So today's `rank = 1` is **physically rank 4**. The `rank` parameter is
lying by up to 3. Either round `rank` up to a multiple of 4 and say so, or zero the padding lanes so the
logical rank is the real one. Rounding up is better here — the lanes are already paid for.

### Deriving it

```cpp
// The model's declared weights are the only ground truth there is: nothing stores them
// afterwards. Climb from the cheapest representation that could work, stopping at the
// first rank whose MEASURED residual clears tolerance -- so the rank is a property of
// this model's weight field rather than a constant someone guessed.
//
// The direction of the search is the whole point. average_degree/2 is where an arbitrary
// field becomes fittable; it is the ceiling. Every document anyone actually writes sits
// far below it and its rank has nothing to do with degree, so the search starts at the
// bottom and the ceiling only bounds how far it is worth climbing.
s64 WeightMatrix::derive_rank(const Vector<f32> &declared_weights, f32 relative_tolerance) {
    // Closed forms first -- these need no fit at all and cover the common document.
    // A uniform field is rank 1; a field constant on each projection run is exactly
    // representable at rank == run count, one latent lane per run.
    if (weights_are_uniform(declared_weights)) return LANE_GROUP;      // rank 1, one lane group
    const s64 run_rank = round_up_to_lane_group(projection_run_count());
    if (run_rank <= MAX_RANK_FLOAT4_STRIDE * LANE_GROUP) {
        fit_basis_at_rank(run_rank, declared_weights);
        if (measure_worst_relative_error(declared_weights) <= relative_tolerance) return run_rank;
    }

    // Otherwise climb. Past the ceiling the free parameters no longer outnumber the
    // edges, so a failure there is the field's property and not the rank's.
    const s64 arbitrary_field_ceiling = std::min<s64>(
            MAX_RANK_FLOAT4_STRIDE * LANE_GROUP,
            round_up_to_lane_group((total_edge_count / node_count + 1) / 2));

    for (s64 candidate = LANE_GROUP; candidate <= arbitrary_field_ceiling; candidate *= 2) {
        fit_basis_at_rank(candidate, declared_weights);
        const f32 worst_relative_error = measure_worst_relative_error(declared_weights);
        if (worst_relative_error <= relative_tolerance) {
            log::logger().info(
                "WeightMatrix: rank {} reproduces every declared weight to {:.2e} relative "
                "({} edges, {} nodes, basis is {} bytes vs {} for one f32 per edge)",
                candidate, worst_relative_error, total_edge_count, node_count,
                2 * node_count * candidate * 4, total_edge_count * 4);
            return candidate;
        }
    }

    log::throw_runtime_error(log::logger(), fmt::format(
        "WeightMatrix: no rank up to {} reproduces this model's weights to {:.2e} relative. "
        "The weight field is not low-rank on this network's edge set, so the engine would "
        "simulate weights the model did not specify.",
        MAX_RANK_FLOAT4_STRIDE * 4, relative_tolerance));
}
```

**Fail loudly rather than silently mis-simulate.** With no exact-storage fallback, a model whose weights the
basis cannot express is a model this engine cannot run correctly, and it should say so at construction
rather than produce plausible-looking spike trains from weights nobody asked for.

Two shortcuts before any fitting happens: a uniform weight field is rank-1 in closed form
(`set_constant_weight`, `weight_matrix.cpp:255`, already handles the padding-lane correction at `:262-267`),
and a field that is constant per projection is exactly representable at a rank equal to the projection
count. Both are checked first, so the common model never runs a fit at all.

**Where the declared weights come from.** `build_weight_matrix` has every edge's weight from the parse
result before anything is stored. The fit consumes a transient host array of `total_edge_count` f32s, which
is discarded when construction finishes — a build-time cost, not resident storage. At 100M edges that is a
400 MB transient; if that becomes a problem, the fit minibatches over projections instead.

## C5. Plasticity deltas — off by default, a bounded SoA store when on

**Scope.** The engine's existing built-in Hebbian rule, nothing lowered from LEMS — NeuroML has very little
to say about plasticity, and this is not a priority. It is **off by default** and switched on by a single
construction flag, and "off" means it costs nothing rather than costing a branch: the codegen does not emit
the block at all, and `WeightMatrix` allocates zero bytes for the delta buffers.

```cpp
// Off by default. Hebbian updates change what the simulation computes, and no NeuroML
// document asks for them, so a model runs without them unless the caller says otherwise.
explicit SpikeEngine(const String &lems_input_file,
                     bool enable_hebbian_plasticity = false);

SpikeEngine(const String &lems_input_file,
            const vector<vector<s32>> &adjacency,
            const String &synapse_component_id,
            f64 connection_weight = 1.0,
            f64 connection_delay_seconds = 0.0,
            bool enable_hebbian_plasticity = false);
```

The flag is known before `generate_master_kernel` runs (`engine.cpp:120`, after `build_weight_matrix` at
`:117`), so it reaches the codegen as a defaulted parameter rather than a new field — which also leaves the
four existing `generate_master_kernel` call sites in `engine_tests.cpp` unchanged:

```cpp
String generate_master_kernel(const NML_ParseResult &parse_result,
                              const ModelLayout &layout,
                              bool enable_hebbian_plasticity = false);
```

The one flag drives everything downstream: it sets `WeightMatrix`'s `plasticity_delta_capacity` (0 when
off, which is already the "no buffers" case in D2's partition round), and it decides whether the propagate
stage emits the block below at all.

**What has to happen when it is on.** A Hebbian update produces a per-edge weight change. There is nowhere
to put it per-edge, so it has to end up in `U`/`V`. Doing that inline is not safe: a rank-1 nudge is an
alternating least-squares solve over `U[source]` and `V[target]` (`weight_update_kernel`,
`src/metal/kernels.metal:125-175`), with threadgroup reductions across lanes. Many propagate threads hitting
one target's `V` row concurrently cannot run that correctly.

**So: the propagate thread appends, and a fold pass applies.** Appending is a single atomic increment; the
fold runs once per interval, single-pass, with no contention.

```cpp
// Per-edge weight deltas awaiting the next fold into U/V. Parallel arrays, not an array
// of pairs: the fold reads ordinals and values as separate typed buffers, and a struct
// would make each a strided load.
//
// Capacity is a fraction of total_edge_count, not one slot per edge. A propagate thread
// that finds the store full triggers the fold instead of growing it, so per-edge storage
// never becomes a function of edge count.
EnginePointer plasticity_edge_ordinals;   // s64[plasticity_delta_capacity]
EnginePointer plasticity_delta_values;    // f32[plasticity_delta_capacity]
EnginePointer plasticity_delta_count;     // s32[1], bumped atomically on device
s64 plasticity_delta_capacity = 0;
```

Device side, emitted into the k²-tree walk the propagate stage already runs — the thread owns
`(source, target)` and both spike times are already resident. **This whole block is absent from the
generated source when the flag is off**, so a model that does not ask for plasticity pays nothing per edge
per tick:

```cpp
    // emitted only when enable_hebbian_plasticity
    const float delta = hebbian_delta(tick, last_spiked[neuron_index],
                                      last_spiked[target], step_dt);
    if (delta != 0.0f) {
        device atomic_int *count = (device atomic_int *)plasticity_delta_count;
        const int slot = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        if (slot < plasticity_delta_capacity) {
            plasticity_edge_ordinals[slot] = edge_base;
            plasticity_delta_values[slot]  = delta;
        }
        // Overflow is not silently dropped: the host reads the count after the tick and
        // folds early when it exceeded capacity, replaying nothing -- the deltas past
        // capacity were never written, so the fold interval is what has to shorten, and a
        // persistently overflowing interval is a logged warning.
    }

    // The host, after the tick:
    //   if (count >= capacity || ticks_since_fold >= fold_every_n_ticks) fold_edge_deltas();
```

`hebbian_delta` reads `last_spiked` for both endpoints, so it must honour D6's sentinel: a negative value
means the endpoint has never fired, which yields no delta rather than an enormous one.

The fold is the arithmetic `weight_update_kernel` already implements, batched over the store instead of one
dispatch per edge — `WeightMatrix::update()`'s per-edge dispatch-plus-sync is what made plasticity unusable
per-tick, not the math:

```cpp
// One dispatch for the whole store rather than one per edge. Deltas targeting the same
// V row are folded by the same thread group, which is what keeps the least-squares step
// well-defined -- the reason this cannot happen inline in the propagate walk.
void WeightMatrix::fold_edge_deltas(f32 learning_rate, f32 l2_regularization, s32 iterations);
```

`WeightMatrix::update()` stays as the host-side single-edge entry point, since tests and tooling use it.

`hebbian_delta` is emitted as a substituted snippet rather than hardcoded arithmetic. That costs nothing
now and leaves the seam if a LEMS-lowered rule is ever wanted — `generate_synapse_body` currently rejects
composed plasticity mechanisms outright (`dynamics_codegen.cpp:750-756`), so lowering one would be real new
work that nothing is asking for.

## C6. Delete `refit()`

`refit()` (~400 lines including a hand-rolled Cholesky solve, `weight_matrix.cpp:1229-1560`) fits `U`/`V`
to the dense Sk point cloud. Once nothing stores per-edge values there is no ground truth to fit against —
the basis *is* the weights, so a re-fit against values derived from it is an identity.

Deleted with it: the Cholesky helpers, `refit_every_n_ticks`, `refit_occupancy_threshold_fraction`,
`ticks_since_last_refit`, `advance_tick()`, `max_sparse_delta_occupancy_fraction()`, and the tests covering
them. This is the largest single deletion in the plan.

What survives, because it addresses a real failure mode of C5: **magnitude renormalization.** Many small
rank-1 nudges can let the basis drift in scale, and `scale_neighbor_weights_to_root_mean_square()` /
`gpu_scale_uv` already exist to correct it. Keep them, and run one after each fold — which means, with
plasticity off by default, they never run during an ordinary simulation either. The basis only drifts
because something nudged it.

The initial fit in C4 is a *construction-time* operation against the model's declared weights, and is not
`refit()` returning under another name — it has real ground truth, and it runs exactly once.

## C7. Edge ordinals

Everything above is indexed by a **canonical edge ordinal**, not a padded slot:

```cpp
// Prefix sum over real out-degree: edge_row_offset[n] is the ordinal of node n's first
// outgoing edge, and edge_row_offset[node_count] == total_edge_count. Computed at
// construction by the same get_neighbors() walk that already computes total_edge_count.
//
// This is what removes the max_neighbor_count padding: anything indexed by edge is sized
// total_edge_count, not node_count * max_neighbor_count. The k^2-tree still owns the
// adjacency -- this numbers the edges it enumerates, in its own traversal order.
Vector<s64> edge_row_offset;   // [node_count + 1]
```

Device-side it is one line in the propagate walk:

```diff
- const int edge_base = neuron_index * max_neighbor_count + slot;
+ const int edge_base = edge_row_offset[neuron_index] + slot;
```

Cost: 8 bytes per node.

## C8. Kernel changes

Buffers 20-22 of `KERNEL_SIGNATURE` (`edge_weights`, `edge_delays`, `edge_variables`) are replaced:

| Was | Becomes |
|---|---|
| `edge_weights[edge_base]` | `Σ_k U[source][k]·Ck[k]·V[target][k]`, reconstructed in the walk |
| `edge_delays[edge_base]` | the same reconstruction on the delay matrix, then `lround` |
| `edge_variables[edge_base]` (plane 0) | a `constant` projection-run table, binary-searched |
| `edge_variables[(k+1)*stride + edge_base]` | gone — C1's per-target accumulators |

New buffers: `U`, `V`, `coefficients`, `edge_row_offset`, `synapse_arrivals`, `synapse_state`, and the three
plasticity arrays. New `constant` scalar: `rank_float4_stride`.

The reconstruction already exists in both a host and a device form and should be reused rather than
rewritten: `weight_matrix.cpp:468-477` (host) and `src/metal/kernels.metal:49-76` (device, restored by A4).
Since the walk already holds `source` and `target` in registers, it adds arithmetic but no memory traffic —
`rank_float4_stride` fused multiply-adds per edge in place of one load.

`MAX_RANK_FLOAT4_STRIDE == 64` (`weight_matrix.h:14`, checked at `weight_matrix.cpp:87-94`) must now stay in
sync across three places: the restored `kernels.metal:122`, the generated master kernel, and the host check.
Make it one shared constant rather than the current "must match" comment.

## C9. What the engine calls

No new public interface. `build_weight_matrix` (`engine.cpp:222-270`) stops calling `set_edge_weight` /
`set_edge_delay_ticks` / `set_edge_variable` once per edge and instead declares one run per projection,
which is the shape the model already has:

```cpp
void SpikeEngine::build_weight_matrix() {
    const Vector<Vector<s32>> network = build_adjacency_list(network_details);
    // rank=-1 means "derive it from what the declarations below actually contain" (C4)
    // rather than the hardcoded 1. A zero delta capacity is what turns the plasticity
    // block off end to end -- no buffers here, and no emitted block in the kernel.
    weights = WeightMatrix(gpu, network, /*rank=*/-1, /*check_indexing=*/true,
                           /*max_neighbor_count=*/-1, /*weight_seed=*/(s64)simulation_seed,
                           /*plasticity_delta_capacity=*/enable_hebbian_plasticity
                                   ? default_plasticity_delta_capacity(total_edge_count) : 0);

    // One declaration per projection rather than per edge.
    weights.declare_projections(collect_projection_runs(network_details));
}
```

`collect_projection_runs` walks `network_details.neurons[*].outgoing_edges` in canonical edge order and
coalesces runs sharing a `(synapse_prototype_index, weight, delay_tick_count)` triple, returning parallel
arrays. Model load time then scales with projections rather than edges.

`configure_per_edge_variable_count`, `set_edge_variable`, `per_edge_variable_matrix_base`,
`per_edge_variable_values`, `sparse_delta_buffers` and `sparse_delta_touched` are all deleted.

## C10. Stale comments to correct while in here

- `weight_matrix.h:110-118`, `:152-158` — assert that delay avoids dense per-edge storage; it does not,
  until C3.
- `weight_matrix.h:203-229` — the `per_edge_variable_values` block argues for contiguity on kernel
  argument-table grounds. That is an argument about one buffer versus many, not dense versus sparse, and
  the member goes away in C1.
- `weight_matrix.h:86-95` — describes Sk as "the old map's absent-key → 0 semantics" over a dense array;
  rewrite for C5.
- `weight_matrix.h:130` — `rank_float4_stride` is documented as `ceil(rank/4)` without saying that all
  `stride*4` lanes participate, which is what makes `rank=1` mean 4 (C4).
- `weight_matrix.h:272-277` — "since its default constructor is deleted" is false; `weight_matrix.h:264`
  has `WeightMatrix() = default`.
- `weight_matrix.h:604-610`, `weight_matrix.cpp:1352-1359` — claim the engine binds
  `coefficient_vectors[DEFAULT_MATRIX_INDEX]` as the generated kernel's `edge_weight_coefficients`
  argument. No such argument exists today; C8 makes it true.
- Every `GpuPointer` reference in both headers (`weight_matrix.h:45-46, 60, 94, 229, 273, 368`;
  `k2tree.h:31-35`) — the move-only discipline they describe is gone (D1).

## C11. Sequencing within C

C7 first (cheap, and C2/C3/C5 index by it), then C1 (the item that matters, and self-contained), then C2,
then C3/C4/C8 together since they change the kernel signature as a unit, then C5/C6, then C9/C10.
C1 + C7 alone take the 1.6 GB example down to a run table plus `node_count × prototype_count` accumulators.

---

# D. The remaining TODOs

## D1. `k2tree.h:18` — K2Tree on `EnginePointer` + `EngineBackend`

> *"need to make K2Tree impl use EnginePointer and initialize with EngineBackend correctly"*

The header is converted; `k2tree.cpp` was never touched. Four member buffers are allocated in two places
(`k2tree.cpp:315-318` in `make_k2tree_from_arrays`, `:480-483` in `load`), freed in the destructor
(`:392-395`), plus three per-call scratch buffers in `adjacent_batch` (`:648-650`).

Factories take the backend explicitly:

```cpp
static optional<K2Tree> from_adjacency_list(
    EngineBackend &backend,
    const vector<vector<s32>> &adjacency_list,
    s32 node_count = -1,
    s32 branching_factor = DEFAULT_BRANCHING_FACTOR,
    s32 superblock_size = 1024);
```

and `make_k2tree_from_arrays` becomes one partition round:

```cpp
    Vector<EnginePointer> partitions;
    backend.partition(internal_node_words_length * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
           .partition(leaf_node_words_length     * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
           .partition(rank_superblock_length     * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
           .partition(rank_subblock_length       * sizeof(u16), EngineDatatype::UNSIGNED16, partitions)
           // Staging for adjacent_batch, preallocated: partition/allocate hands out one
           // chunk per round, so a per-call allocation is not available. Queries beyond
           // the cap run in chunks.
           .partition(ADJACENT_BATCH_QUERY_CAP * sizeof(s32), EngineDatatype::SIGNED32, partitions)
           .partition(ADJACENT_BATCH_QUERY_CAP * sizeof(s32), EngineDatatype::SIGNED32, partitions)
           .partition(ADJACENT_BATCH_QUERY_CAP * sizeof(u8),  EngineDatatype::UNSIGNED8, partitions);
    const EnginePointer owning_slab = backend.allocate(partitions);

    memcpy(partitions[0].get_contents(), internal_node_words_host.data(), ...);
    // ... three more memcpys ...
```

`K2Tree` keeps `EngineBackend *owning_backend` and the whole-chunk `EnginePointer owning_slab`, and its
destructor releases the chunk:

```cpp
K2Tree::~K2Tree() {
    // The four bit arrays and the query staging buffers are all sub-ranges of one slab;
    // releasing the slab is what frees them, and no sub-range owns anything.
    if (owning_backend != nullptr) owning_backend->deallocate_slab(owning_slab);
}
```

Every `get_contents()` call site in `k2tree.cpp` needs a type — `internal_node_words.get_contents_as<u32>()`
— in five groups (`:534-537, 554-557, 602-605, 621-624, 695-698`), all inside `const` methods, which is
why `get_contents()` is `const` in A2.

**Scratch cap.** `ADJACENT_BATCH_QUERY_CAP` (suggest 65536, ~576 KB) with `adjacent_batch` looping in
chunks. Its only caller is `tests/k2tree_tests.cpp:186`, so the cap sits on no hot path.

**Move semantics.** `K2Tree(K2Tree&&) = default` now works — `EnginePointer` is a trivially copyable
non-owning value. The move must null the source's `owning_backend` so the destructor does not
double-release; that is the one member needing a hand-written move. The placement-new hack at
`weight_matrix.cpp:196-201` (`k2tree.~K2Tree(); new (&k2tree) K2Tree(std::move(other.k2tree));`) exists
only to dodge `GpuPointer`'s assert and should go, with the comment above it.

**Delete `tests/k2tree_tests.cpp:293-318`** — `building_without_a_gpu_context_reports_a_clear_error` tests
a process-global GPU context that no longer exists. The RAII backend makes the failure it guards against
unrepresentable.

## D2. `weight_matrix.h:44` — float4 support

> *"need to add float4 support for model backend"*

Three parts:

1. **`EngineDatatype::FLOAT32X4`, alignment 16** (A2). `PARTITION_ALIGNMENT` already satisfies it, but the
   datatype should still state its own requirement.
2. **`get_contents_as<float4>()`** (A2) so the ~15 indexing sites read naturally:
   `const float4 *u_row = U_matrix.get_contents_as<float4>() + source_node * rank_float4_stride;`
   This matters more than it looks: on a raw `f32 *` the same expression advances a quarter as far, and it
   compiles clean. Keeping the pointer typed is what makes the stride arithmetic fail loudly.
3. **`WeightMatrix` runs its own partition round** in the constructor:

```cpp
    const u64 matrix_byte_size = (u64)node_count * (u64)rank_float4_stride * sizeof(float4);

    Vector<EnginePointer> partitions;
    backend.partition(matrix_byte_size, EngineDatatype::FLOAT32X4, partitions)          // U
           .partition(matrix_byte_size, EngineDatatype::FLOAT32X4, partitions)          // V
           .partition((u64)(node_count + 1) * sizeof(s64), EngineDatatype::SIGNED64, partitions)
           .partition((u64)matrix_count * coefficient_lane_count * sizeof(f32),
                      EngineDatatype::FLOAT32, partitions)                              // every Ck
           // plasticity_delta_capacity is 0 unless the engine was constructed with
           // enable_hebbian_plasticity, and partition() hands back an empty handle for a
           // zero-byte request -- so the default build allocates nothing for these three.
           .partition((u64)plasticity_delta_capacity * sizeof(s64), EngineDatatype::SIGNED64, partitions)
           .partition((u64)plasticity_delta_capacity * sizeof(f32), EngineDatatype::FLOAT32, partitions)
           .partition(plasticity_delta_capacity > 0 ? sizeof(s32) : 0,
                      EngineDatatype::SIGNED32, partitions)                             // delta count
           // Preallocated output for dispatch_neighbor_weights, which allocated per call.
           // Sized by edges, not by node_count * max_neighbor_count: a padded scratch here
           // would reintroduce the exact allocation C7 exists to remove (400 MB at 1M
           // nodes and degree 100), and it would be resident for the object's whole life.
           .partition((u64)total_edge_count * sizeof(f32), EngineDatatype::FLOAT32, partitions);
    owning_slab = backend.allocate(partitions);
```

That re-sizing changes `neighbor_weights()`'s output contract from a padded
`[node_count][max_neighbor_count]` grid with sentinel `-1` rows (`weight_matrix.h:126-128`) to a flat
edge-ordinal-indexed array. Its callers are `neighbor_weight_stats()` and roughly ten tests in
`weight_matrix_tests.cpp`, none of them on a simulation path, so the churn is contained — and the padded
form's sentinel slots existed only to describe padding that no longer exists.

**Invariant to check when this lands:** after C7, `max_neighbor_count` sizes *no allocation anywhere*. It
survives only as the caller-supplied output cap on `K2Tree::get_neighbors()`. A `grep` for
`max_neighbor_count` next to a `partition` or a `sizeof` should come back empty.

Because a chunk is sized once, the Ck block is sized up front from `matrix_count` (weights + delay, known
at construction) rather than pushed lazily. `load_from_disk` reallocating on a dimension change
(`weight_matrix.cpp:1580-1625`) releases its slab and runs a fresh round.

Note C4's rank derivation runs **before** this partition round, since `rank_float4_stride` sizes `U`/`V`.
The fit happens on host arrays and only its result is uploaded.

## D3. `recording.h:38` — the `RecordedValue` union

> *"this is not final and is just filler code really until we use it and think of something better"*

`RecordedValue` and `RecordingConfig::recored_data` (note the typo) are **dead** — nothing in the repo reads
or writes either. Real recorded data flows through `SpikeEngine::recorded_traces` (`Vector<f32>`),
`recorded_spikes` and `SimulationRecorder`.

**Delete both.** A bare union with no discriminant cannot be read safely, so it is not a foundation for
anything; when a consumer exists the right shape is a tagged value, and that is a five-line change then.
While in the file, `s32 recordings_count;` is uninitialized — give it `= 0`.

```diff
-    union RecordedValue {
-        // TODO: this is not final and is just filler code really until we
-        // use it and think of something better
-        f64 float64;
-        s64 int64;
-        f32 float32;
-        s32 int32;
-    };
-
     struct RecordingConfig {
         Vector<String> output_filenames;
         Vector<OutputFileFormat> file_output_format;
-        Vector<RecordedValue> recored_data;
```

## D4. `engine.cpp:106` — `std::move` on the partition assignments?

> *"do we need to add std::move around RHS here? Or does the '=' copy the value out"*

**No.** `EnginePointer` is an aggregate of scalars with no user-declared move operations, so its move and
its copy are the same operation. `std::move` would change nothing.

The reason is worth a comment, because it is the invariant that replaced `GpuPointer`'s move-only
discipline:

```cpp
// EnginePointer is a non-owning value: copying one produces a second name for the same
// range, and the backend's slab is what owns the storage. That is deliberate -- it is why
// K2Tree and WeightMatrix can go back to defaulted move assignment, which GpuPointer's
// null-destination assert made impossible.
```

The positional `data_partitions[6]` indexing is fragile enough to name:

```cpp
enum ModelPartition : usize {
    CELL_STATE = 0, CELL_PARAMETERS, SYNAPSE_PARAMETERS, NETWORK_INPUTS,
    SPIKE_HISTORY, LAST_SPIKED, EMPTY_EDGE_PLANE, MODEL_PARTITION_COUNT
};
cell_state = data_partitions[CELL_STATE];
```

Also in this function: `engine.h` still declares `allocate_model_buffers()` while `engine.cpp:144` defines
`initialize_model_buffers()`, and `gpu_bytes` at `engine.cpp:180` is a local that went away with the old
allocator — the log line should read the slab handle's `total_bytes`.

## D5. `engine.cpp:113` — what `edge_placeholder` is

> *"figure out what this edge_placeholder thing is again"*

A one-float GPU range bound in place of any per-edge plane the model never caused to be allocated. Buffers
20-22 of the generated kernel must always bind to something real: a model with no connections registers no
delay matrix and allocates no per-edge planes, and the old `metal_dispatch` treated an address it could not
resolve in its registry as *scalar bytes* — so a null would have been silently `setBytes`'d as eight bytes
of pointer into a `device float *` slot.

The registry is gone, but the need survives in a different form: `run_function` calls `setBuffer` with
`parameter.base_pointer`, and a default-constructed `EnginePointer` has no buffer to bind. Metal also
errors on binding nil to a slot the kernel declares.

Keep it, rename it, and give it the comment it never had. It also lost its declaration — the member is not
in `engine.h` at all any more.

```cpp
// Bound to any per-edge kernel argument the model has no plane for. A model with no
// connections registers none of them, and there is no way to bind "nothing" to a
// `device float *` the kernel declares. One float is enough: the kernel never reads it,
// because a neuron with no adjacency row exits the propagate walk before its first load.
EnginePointer empty_edge_plane;
```

`resolve_edge_plane` returns an `EnginePointer` rather than a `void *`, since that is what `run_function`
takes. After C8, `U`, `V` and `coefficients` are always allocated, so the only argument still needing a
placeholder is `synapse_arrivals` on a model with no synapses at all — check whether that case is reachable
before keeping the member.

## D6. `engine.cpp:149` — can `last_spiked` be `-1`?

> *"can't we just set this to -1 instead of NEVER_SPIKED_TICK?"*

**Not as the code reads today, but yes after a two-line kernel change — and that is the better fix.**

The gate is `dynamics_codegen.cpp:715`:

```cpp
const float time_since_spike = (float)(tick - last_spiked[neuron_index]) * step_dt;
if (time_since_spike >= <refractory duration>) { ... }
```

With `last_spiked = -1` at `tick = 0`, `time_since_spike` is `1 · dt` = 0.1 ms. A GLIF refractory period is
3-5 ms, i.e. 30-50 ticks — so every cell in the model starts held and stays held for the first ~30-50
ticks. That is the same class of bug the existing comment describes for `0`, just less visible.

`NEVER_SPIKED_TICK = -(1 << 32)` works by making the difference enormous, but it encodes an assumption
about the longest refractory period the engine will ever simulate. Make the sentinel explicit in the gate
instead:

```diff
- << indent << "const float time_since_spike = (float)(tick - last_spiked["
- << "neuron_index]) * step_dt;\n"
+ << indent << "// last_spiked is negative for a cell that has never fired, which is not\n"
+ << indent << "// the same as one that fired long ago -- a never-fired cell is never held,\n"
+ << indent << "// whatever the refractory duration is.\n"
+ << indent << "const float time_since_spike = last_spiked[neuron_index] < 0\n"
+ << indent << "    ? INFINITY\n"
+ << indent << "    : (float)(tick - last_spiked[neuron_index]) * step_dt;\n"
```

```diff
- std::fill(last_spiked_data, last_spiked_data + total_neuron_count, NEVER_SPIKED_TICK);
+ // Negative means "has never fired" -- see the refractory gate in dynamics_codegen.
+ std::fill(last_spiked_data, last_spiked_data + total_neuron_count, (s64)-1);
```

`NEVER_SPIKED_TICK` is commented out at `engine.h:24` while `engine.cpp:155` still uses it, so this TODO
has to be resolved one way or the other for the build.

C5's `plasticity_delta()` reads `last_spiked` for both endpoints, so it must use the same sentinel
convention — a never-fired endpoint produces no delta rather than an enormous one.

**Test to add:** a GLIF cell with a long refractory period must fire on the first tick it crosses
threshold, including tick 0 — precisely the case `-1` alone would break.

## D7. `backend.cpp:112` — Metal dispatch error logging

> *"do error logging here and stuff"*

The current check is inverted (`if (error == nullptr) return false;` reports failure on success), and
`error` is only ever written by `newComputePipelineState` — the command buffer never touches it. Three
other problems in the same function belong in the same edit:

- `newCommandQueue()` is called **per dispatch** and never released. At 10k ticks that is 10k leaked
  queues. Cache one on the backend (A2).
- `newComputePipelineState` is called **per dispatch**. Build it once in `create_function` and store it in
  `EngineFunction`, as the deleted `KernelHandle` did.
- `encoder` and `command_buffer` are never released.

```cpp
bool MetalBackend::run_function(
    EngineFunction &function, Vector<EnginePointer> &parameters, s64 job_count
) {
    MTL::CommandBuffer *command_buffer = queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = command_buffer->computeCommandEncoder();
    encoder->setComputePipelineState(function.pipeline_state);

    for (usize index = 0; index < parameters.size(); index += 1) {
        const EnginePointer &parameter = parameters[index];
        if (parameter.inline_scalar) {
            // `constant T &x [[buffer(N)]]` accepts inline constant data as readily as a
            // buffer binding, which is what lets a scalar travel as an EnginePointer.
            encoder->setBytes(parameter.base_pointer, parameter.total_bytes, index);
        } else {
            encoder->setBuffer(static_cast<MTL::Buffer *>(parameter.base_pointer),
                               (NS::UInteger)parameter.offset, index);
        }
    }

    const s64 threads = std::max<s64>(job_count, 1);
    encoder->dispatchThreads(MTL::Size::Make((NS::UInteger)threads, 1, 1),
                             MTL::Size::Make((NS::UInteger)threads_per_block, 1, 1));
    encoder->endEncoding();
    encoder->release();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    if (command_buffer->status() == MTL::CommandBufferStatusError) {
        NS::Error *execution_error = command_buffer->error();
        const String description = execution_error
                ? execution_error->localizedDescription()->utf8String()
                : "unknown error";
        logger().critical("run_function: command buffer failed: {} (job_count={}, parameters={})",
                          description, job_count, parameters.size());
        command_buffer->release();
        return false;
    }

    logger().trace("run_function: dispatched {} threads, {} parameters", threads, parameters.size());
    command_buffer->release();
    return true;
}
```

`create_function` needs the matching error handling it lacks — it leaks the library, never checks
`newLibrary` or `newFunction` for null, and builds no pipeline:

```cpp
Optional<EngineFunction> MetalBackend::create_function(const String &name, const String &source_code) {
    NS::Error *error = nullptr;
    MTL::Library *library = device->newLibrary(
            NS::String::string(source_code.c_str(), NS::UTF8StringEncoding), nullptr, &error);
    if (library == nullptr) {
        logger().critical("create_function: newLibrary failed: {}",
                          error ? error->localizedDescription()->utf8String() : "unknown error");
        return std::nullopt;
    }

    MTL::Function *function = library->newFunction(
            NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    library->release();
    if (function == nullptr) {
        logger().critical("create_function: '{}' not found in the compiled library", name);
        return std::nullopt;
    }

    error = nullptr;
    MTL::ComputePipelineState *pipeline = device->newComputePipelineState(function, &error);
    function->release();
    if (pipeline == nullptr) {
        logger().critical("create_function: newComputePipelineState failed: {}",
                          error ? error->localizedDescription()->utf8String() : "unknown error");
        return std::nullopt;
    }

    return EngineFunction{pipeline};
}
```

`EngineFunction` holds the pipeline state, not the `MTL::Function` — the pipeline is the thing worth
caching across ticks, and it retains what it needs from the function.

**Autorelease pools:** the old code used none and released every object explicitly. `NS::String::string`
returns an autoreleased object with no pool to catch it, so those leak by design — bounded, and
per-compile rather than per-tick. Keeping `NS::String::string` out of `run_function`, as above, is what
makes that safe.

The AOT path also needs a `load_precompiled_kernel` equivalent on `MetalBackend`, since the five kernels in
the restored `kernels.metal` come from `default.metallib` rather than from `create_function`. Carry back
the `dladdr`-based `load_default_metal_library` from `841df1b:src/core/backend.cpp:81-96` — a command-line
tool has no app bundle for `newDefaultLibrary()` to search, and that helper is what solved it.

## D8. `backend.cpp:224` — CUDA launch result logging

> *"add logging around the launch result for both error and success scenarios"*

The parameter marshalling has a bug that must be fixed alongside it: `parameters[index] = ...` writes into
the caller's `Vector<EnginePointer>` instead of the launch array `parameters_`, leaving slots `0..n-1`
uninitialized.

```cpp
bool CudaBackend::run_function(
    EngineFunction &function, Vector<EnginePointer> &parameters, s64 job_count
) {
    const s64 block_count = (job_count + threads_per_block - 1) / threads_per_block;

    // cuLaunchKernel takes pointers TO each argument's value. For a slab range the value
    // is the device address, so it needs a stable cell to point at; for an inline scalar
    // the caller's own storage already is that cell.
    Vector<void *> argument_values(parameters.size(), nullptr);
    Vector<void *> argument_pointers(parameters.size(), nullptr);

    for (usize index = 0; index < parameters.size(); index += 1) {
        const EnginePointer &parameter = parameters[index];
        if (parameter.inline_scalar) {
            argument_pointers[index] = parameter.base_pointer;
        } else {
            argument_values[index]   = parameter.get_contents();
            argument_pointers[index] = &argument_values[index];
        }
    }

    constexpr u32 USE_DYNAMIC_SHARED_MEMORY = 0;
    const CUresult launch_result = cuLaunchKernel(
        function.cuda_function,
        (u32)block_count, 1, 1,
        (u32)threads_per_block, 1, 1,
        USE_DYNAMIC_SHARED_MEMORY, nullptr,
        argument_pointers.data(), nullptr);

    if (launch_result != CUDA_SUCCESS) {
        const char *error_name = nullptr;
        const char *error_string = nullptr;
        cuGetErrorName(launch_result, &error_name);
        cuGetErrorString(launch_result, &error_string);
        logger().critical("run_function: cuLaunchKernel failed: {} ({}) "
                          "-- grid={} block={} job_count={} parameters={}",
                          error_name ? error_name : "unknown",
                          error_string ? error_string : "unknown",
                          block_count, threads_per_block, job_count, parameters.size());
        return false;
    }

    const cudaError_t execution_result = cudaDeviceSynchronize();
    if (execution_result != cudaSuccess) {
        logger().critical("run_function: cudaDeviceSynchronize failed after launch: {}",
                          cudaGetErrorString(execution_result));
        return false;
    }

    logger().trace("run_function: launched grid={} block={} job_count={} parameters={}",
                   block_count, threads_per_block, job_count, parameters.size());
    return true;
}
```

This function *is* the generic positional-argument launcher the generated master kernel needs on CUDA. But
note `dynamics_codegen.cpp` emits Metal only — there is no `__global__` path in the file at all — so a CUDA
build still cannot run a generated kernel. Its predecessor `src/nml/kernel_codegen.cpp` emitted both;
re-adding the CUDA branch is separate work, and the throw that used to guard this ("no simulation can run on
this backend") should be reinstated so a CUDA build fails loudly rather than reporting successful ticks that
ran no dynamics.

> `backend.cpp:162` (the NVRTC `--gpu-architecture` TODO) is **left as-is** per your instruction. Note the
> current call is `nvrtcCompileProgram(program, 1, options)` against an empty `options[]`, which is
> undefined behaviour — worth passing `0` for the count until the TODO is picked up.

---

# E. The dispatch call site

`engine.cpp:546-602` builds `const void *arguments[]` / `argument_sizes[]` for the deleted `metal_dispatch`.
Rewritten for `run_function`, with scalars as `EnginePointer` wrappers at offset 0:

```cpp
void SpikeEngine::step_simulation(s64 tick) {
    apply_stimulus(tick);

    // Scalars live on the stack for the duration of the dispatch and reach the kernel as
    // inline constant data: an EnginePointer with inline_scalar set means "these bytes",
    // not "this range of a slab", and run_function is what tells the two apart.
    const s32 neuron_count_argument         = (s32)total_neuron_count;
    const s32 spike_history_length_argument = (s32)layout.spike_history_length;
    const s32 max_neighbor_count_argument   = (s32)weights.max_neighbor_count;

    auto inline_scalar = [](const auto &value) {
        return EnginePointer{(void *)&value, 0, sizeof(value), alignof(decltype(value)), true};
    };

    const K2Tree &tree = weights.k2tree;
    Vector<EnginePointer> parameters = {
        inline_scalar(tick),                            // 0
        inline_scalar(step_dt),                         // 1
        inline_scalar(neuron_count_argument),           // 2
        inline_scalar(spike_history_length_argument),   // 3
        inline_scalar(max_neighbor_count_argument),     // 4
        cell_state,                                     // 5
        cell_parameters,                                // 6
        synapse_parameters,                             // 7
        network_inputs,                                 // 8
        spike_history,                                  // 9
        last_spiked,                                    // 10
        tree.internal_node_words,                       // 11
        tree.leaf_node_words,                           // 12
        tree.rank_superblock_table,                     // 13
        tree.rank_subblock_table,                       // 14
        inline_scalar(tree.branching_factor),           // 15
        inline_scalar(tree.superblock_size_words),      // 16
        inline_scalar(tree.padded_node_count),          // 17
        inline_scalar(tree.tree_height),                // 18
        inline_scalar(tree.internal_bit_count),         // 19
        resolve_edge_plane(WeightMatrix::DEFAULT_MATRIX_INDEX),  // 20
        resolve_edge_plane(weights.delay_matrix_index),          // 21
        weights.per_edge_variable_values.is_empty()              // 22
            ? empty_edge_plane : weights.per_edge_variable_values,
    };

    if (!gpu.run_function(kernel_function, parameters, total_neuron_count)) {
        log::throw_runtime_error(*logger,
                "SpikeEngine: tick " + to_string(tick) + " failed on the GPU");
    }

    record_tick(tick);
}
```

Two behaviour changes to be deliberate about:

- **`job_count` is `total_neuron_count`, not `block_count`.** The old path called `dispatchThreadgroups`
  with `grid_size = block_count`; `run_function` calls `dispatchThreads`, which takes total threads.
  Passing `block_count` would run a `1/256` slice of the network and report success.
- **`synchronize_gpu_work()` is deleted** from the call site. It was already a no-op on Metal, and
  `run_function` blocks on `waitUntilCompleted`.

Arguments 20-22 change shape once section C lands, and **argument 4 disappears entirely**. The kernel uses
`max_neighbor_count` in exactly two places today — `edge_plane_stride = neuron_count * max_neighbor_count`
(`dynamics_codegen.cpp:911`) and the walk's `if (slot >= max_neighbor_count) break;` (`:979`). C7 replaces
the first with the edge ordinal and the second with the node's actual degree,
`edge_row_offset[n+1] - edge_row_offset[n]`, which is a tighter bound as well as a correct one. This is the
pre-C form, so the port can be tested before the storage rework starts.

---

# F. Sequencing

1. **A1-A6** — braces, header structure, `backend.cpp` signatures, restore the three `src/metal` files,
   `make clean`, Makefile/setup.py. *Checkpoint: `src/core/backend.cpp` compiles.*
2. **B** — reusable partition/allocate on both backends. *Checkpoint: the `SpikeEngine` constructor
   compiles and its handles resolve to real addresses.*
3. **D1, D2** — `k2tree.cpp` and `weight_matrix.cpp` ported, backend threaded through explicitly.
   *Checkpoint: `make metal` links.*
4. **E, D5, D6, D7, D8** — dispatch, `empty_edge_plane`, the `-1` sentinel, the logging TODOs.
   *Checkpoint: `make test` green; a demo runs and produces a membrane video.*
5. **D3, D4** — the two comment/dead-code TODOs; trivial, any time.
6. **C7, C1** — edge ordinals, then aggregated accumulators. *Checkpoint: spike trains identical to step 4
   on every fixture, with per-edge synapse state gone.*
7. **C2** — projection runs for the prototype index.
8. **C3, C4, C8** — delay into the basis, rank derivation, kernel reconstruction. *Checkpoint: derived rank
   and residual logged for every fixture; footprint measured.*
9. **C5, C6** — plasticity delta store and fold, `refit()` deleted.
10. **C9, C10** — engine call site and the stale-comment sweep, then update `CLAUDE.md`.

Steps 1-5 restore the branch. Steps 6-10 are the invariant repair, and can be a separate PR onto the same
branch once 1-5 are green — which also gives step 6 a known-good baseline to diff spike trains against.

---

# G. Verification

**Build**

```bash
make clean && make metal          # must link; watch for undefined NS::/MTL:: symbols (A4)
make test                         # tests/*.cpp, 6465 lines, single compile + link + run
make examples && make demos
```

**Runtime smoke** — `generate_master_kernel` throwing on a missing `k2tree_device.metalinc` is the first
thing that will bite, and it happens in the `SpikeEngine` constructor before any dispatch:

```bash
./build/examples/iaf_single_cell_example        # smallest path through construct -> run -> record
./build/demos/glif1_network_demo                # exercises the k²-tree walk and propagate stage
```

**Numerical checks for section C** — this is where real adversarial review is needed, because a
per-edge-to-aggregate refactor is exactly the shape of a silent wrong-number defect:

- Capture spike trains and membrane traces from the step-4 build for every fixture in
  `tests/fixtures/nml/`, then assert bit-equality after C1. **C1 is an exact transformation; anything that
  moves is a bug**, which makes it the strongest checkpoint in the plan.
- A dedicated two-edge test: one target, two sources, different weights, different delays, arrivals on
  different ticks. Assert against a hand-computed expectation, not just the previous build — a shared bug
  in both paths would otherwise pass.
- A three-prototype test, since aggregation is per-prototype and a single-prototype test cannot distinguish
  correct grouping from no grouping at all.
- A never-fired cell with a long refractory period must fire on the first tick it crosses threshold (D6).
- **Delay must round-trip exactly.** Assert `get_edge_delay_ticks == declared` for every edge of every
  fixture after C3 — this is an integer equality, not a tolerance, and it is the check that catches a rank
  too low for the delay field.
- **After C4, weight readback is exact to the derived rank's fidelity rather than bit-exact.** Every test
  asserting `EXPECT_FLOAT_EQ` on `get()` after a weight assignment becomes a tolerance assertion tied to
  the tolerance the fit was told to meet. A tolerance wide enough to hide a fit failure is the risk to
  scrutinise in review — prefer asserting the *derived rank and measured residual* directly, which is a
  sharper check than any per-edge tolerance.
- **Plasticity off is the default and must be free (C5):** assert that a default-constructed engine's
  generated kernel source contains no `hebbian_delta` and no `plasticity_delta_count`, and that
  `plasticity_delta_capacity` is 0 so the three buffers are empty handles. This is a source-substring and
  a size assertion, which is cheap and catches the block being emitted unconditionally.
- **Plasticity on, fold equivalence (C5):** apply N deltas through `update()` one at a time, and the same N
  through the store plus one `fold_edge_deltas()`. The reconstructions must agree to fit tolerance. Also
  assert the store never exceeds capacity and that an overflowing interval logs.
- **Plasticity on must not change an unstimulated run:** with no spikes there are no deltas, so a
  plasticity-enabled engine and a default one must produce identical traces. Separates "the flag is wired"
  from "the rule fires when it should".

**Footprint** — log each chunk's `total_bytes` at construction and record it for `glif1_million_demo`
before and after section C. The claims to check: nothing scales with `node_count × max_neighbor_count`;
synapse state scales with `node_count × prototype_count`; prototype storage scales with projection count;
the basis scales with `node_count × derived_rank`.

**Existing test debt to resolve while porting**

- `tests/test_core.cpp:13,16` — `initialize_gpu_context()` / `release_gpu_resources()` in `main()`. It is
  the only `main()`, so nothing links until this is fixed.
- `tests/k2tree_tests.cpp:293-318` — delete (D1).
- `tests/weight_matrix_tests.cpp:578-600, 2176` — direct `GpuPointer` / `allocate<>` use plus a `.pointer`
  field check.
- `tests/weight_matrix_tests.cpp:1016-1100` — the Sk round-trip tests; rewritten against C5's store, not
  deleted, since the round-trip property still holds in the new representation.
- The `refit()` tests go with C6.
- 19 files under `examples/` call `initialize_gpu_context()` / `release_gpu_resources()` — two lines each.
  `tests/engine_tests.cpp:21` includes `examples/demos/network_model.h`, so breaking the examples tree
  breaks the test build.

**Out of scope** — `src/bindings/bindings.cpp` references `SpikeEngine::membrane_potentials`,
`neuron_count`, `last_tick_updated` and `active_neuron_count`, none of which exist on the current engine.
It is stale against a much older engine independent of this rework and needs a rewrite rather than a port,
so `make python` will keep failing until that is scheduled.
