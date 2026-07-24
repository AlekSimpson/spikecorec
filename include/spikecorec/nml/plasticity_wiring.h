#pragma once

#include <optional>

#include "spikecorec/core/engine.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── NML STDP -> stage-7 WeightMatrix::update wiring (ticket #66 [F5]; arch §3.3 D6) ────────────
//
// Per CLAUDE.md's own "U/V factorization is a memory optimization, NOT learning" section: what
// this file wires up is LOCAL SYNAPTIC PLASTICITY (spike-timing-dependent updates to individual
// edge weights) -- a biological simulation feature, not task learning, gradient descent, or
// anything related to the U/V rank/compression parameter.
//
// ── What the target kernel actually consumes (read before changing anything here) ──────────────
//
// The ticket names "the existing stage-7 WeightMatrix::update path" as the wiring target. Reading
// src/metal/kernels.metal (mirrored exactly in src/cuda/kernels.cu, verified there only by
// structural reading -- no CUDA toolchain on this machine) turns up TWO, unrelated
// `learning_rate`-consuming kernels, and only one of them is what actually executes during a
// simulation tick:
//   - `weight_update_kernel` (backend::gpu_weight_update, WeightMatrix::update) -- a standalone,
//     directly-callable rank-1 Hebbian nudge of one (source, target) pair. Grepping the whole tree
//     shows `WeightMatrix::update` is never called from anywhere in the engine's own tick loop
//     (src/core/engine.cpp) -- it is a manually-invokable API (e.g. for a caller doing its own
//     reservoir-style weight nudges), not what runs during `step_simulation`.
//   - `step_apply_hebbian_update`, called from inside `step`/`step_no_active_optimization`
//     (src/metal/kernels.metal, src/cuda/kernels.cu) -- THIS is the real, always-running stage-7
//     plasticity path: every tick a neuron spikes and propagates to a downstream neighbor, if that
//     neighbor has already spiked at some earlier, different tick, the kernel computes
//     `decay_delta = -learning_rate * pow(|tick - child_last_spiked|, -3)` and folds it into a
//     rank-1 nudge of U[source]/V[target]. `learning_rate` (SpikeEngine::learning_rate,
//     enable_plasticity/disable_plasticity's own parameter, SC-11) is the ONLY externally-supplied
//     constant this path consumes -- the `pow(x, -3)` recency shape and the two internal constants
//     step_apply_hebbian_update is invoked with (0.5f "learning_rate", 1.0f "l2_regularization")
//     are BAKED INTO THE KERNEL ITSELF, not passed in. This confirms the ticket's own framing:
//     "already hardcoded with its own decay/update shape; map NML params into baked constants this
//     kernel already consumes, do NOT rewrite the kernel's own update math."
//   - Sign convention (verified by reading the code, not assumed): `pow(tick_delta, -3)` is always
//     positive for `tick_delta > 0` (the only case the `!(child_last_spiked == 0 || ... == tick)`
//     guard lets through), so `decay_delta` is always <= 0. There is no branch anywhere that ever
//     makes this positive -- `step_apply_hebbian_update` is invoked from exactly ONE call site in
//     each kernel, always with this same negative `decay_delta`. So the ONLY plasticity that ever
//     actually runs during a tick is a one-signed DEPRESSION, scaled larger the more RECENTLY the
//     postsynaptic neuron fired relative to now -- not a two-sided
//     potentiate-if-pre-before-post/depress-if-post-before-pre window the way textbook STDP is
//     usually described. tests/plasticity_wiring_tests.cpp's
//     `PlasticityWiring.stdp_direction_matches_kernel_sign_convention` drives this through a real
//     simulation with controlled spike timing and confirms this empirically (real before/after
//     weight numbers), rather than assuming it.
//
// ── Mapping an NML STDP spec onto that one baked constant ───────────────────────────────────────
//
// third_party/neuroml2/std_lib/Synapses.xml's own `stdpSynapse` ComponentType (the only
// ComponentType in the vendored std-lib with "stdp" in its name) is explicitly marked
// "NOTE: EXAMPLE NOT YET WORKING!!!!" in its own description, and its Dynamics declare no
// tauPlus/tauMinus/aPlus/aMinus (or equivalent) parameters at all -- its `M`/`P` state variables
// are initialized in OnStart and never otherwise assigned. Grepping the whole vendored
// third_party/neuroml2 tree for tauPlus/tauMinus/aPlus/aMinus (or common `snake_case`/underscore
// spellings) turns up nothing anywhere. So there is no complete, working STDP ComponentType to
// lower verbatim; `find_stdp_spec` below instead detects an STDP-shaped synapse STRUCTURALLY, by
// parameter name, exactly the same generic, name-driven spirit codegen already uses everywhere
// else ("codegen is generic: it emits one small IR program per ComponentType, never a
// hand-written kernel") -- any synapse ComponentType (vendored or user-defined) that bakes all
// four of `tauPlus`/`tauMinus`/`aPlus`/`aMinus` counts as "has an STDP spec", matching the
// standard NeuroML/LEMS community naming convention for a custom STDP ComponentType (see
// tests/plasticity_wiring_tests.cpp's own `TestStdpSynapse` fixture, built the same way this
// codebase's other tests build a stand-in for a real-but-gapped std-lib type, e.g.
// synapse_lowering_tests.cpp's own `TestExpTwoSynapse`).
//
// Since the kernel truly has only ONE dial, only one of the four spec values can actually reach
// it. `map_stdp_spec_to_learning_rate` uses `aMinus` (the standard STDP depression-amplitude
// parameter): the kernel's own, empirically-confirmed real behavior is a pure depression process
// (see above), so `aMinus` -- not `aPlus` -- is the parameter whose real-world meaning actually
// matches what this kernel computes. `tauPlus`/`tauMinus`/`aPlus` are still required for
// PRESENCE detection (so "has an STDP spec" is a real structural check, not a single-parameter
// guess) but cannot otherwise influence this kernel's math -- its `pow(tick_delta, -3)` recency
// shape is fixed in the kernel source, unrelated to any tau, and its sign is fixed to depression,
// unrelated to aPlus. This is a real, documented limitation of the TARGET KERNEL (out of this
// ticket's scope to change -- the ticket is explicit that rewriting the kernel's own update math
// is not the job here), not something this wiring layer papers over.

struct StdpSpec {
    f32 tau_plus;  // seconds (SI; already unit-resolved by the resolve/lower pipeline)
    f32 tau_minus; // seconds
    f32 a_plus;    // dimensionless
    f32 a_minus;   // dimensionless
};

// Looks for `tauPlus`/`tauMinus`/`aPlus`/`aMinus` among `synapse_entry.baked_constants` (Phase-1
// always bakes Parameters/Properties as literals, see model_specification.h). Returns nullopt if
// `synapse_entry` isn't a Synapse-category entry, or if any of the four names is absent.
std::optional<StdpSpec> find_stdp_spec(const TypeLibraryEntry &synapse_entry);

// Maps `spec` onto the minus-side (depression) scalar the real stage-7 kernel path consumes (see
// this header's own doc comment for the derivation): `spec.a_minus`, unchanged.
f32 map_stdp_spec_to_learning_rate(const StdpSpec &spec);

// Maps `spec` onto the plus-side (potentiation) scalar the real bidirectional-STDP path consumes:
// `spec.a_plus`, the standard STDP potentiation-amplitude parameter. This is the LTP counterpart of
// map_stdp_spec_to_learning_rate above — it drives the causal (pre-before-post) predecessor-edge
// potentiation both stage-7 call sites now apply (the legacy GPU `step`/`step_no_active_optimization`
// kernels and SpikeEngine::apply_nml_stdp_plasticity), scaled by the same pow(tick_delta, -3) recency
// shape the depression side uses, just positive. Throws if a_plus is negative (a potentiation
// amplitude is always >= 0, matching map_stdp_spec_to_learning_rate's own aMinus guard).
f32 map_stdp_spec_to_learning_rate_plus(const StdpSpec &spec);

// Scans every Synapse-category entry in `model.type_library` for an StdpSpec (`find_stdp_spec`).
// Presence -> `engine.enable_plasticity(map_stdp_spec_to_learning_rate(spec),
// map_stdp_spec_to_learning_rate_plus(spec))` (both the minus-side depression rate and the plus-side
// potentiation rate now reach the engine, for real bidirectional STDP); absence ->
// `engine.disable_plasticity()` -- exactly the presence/absence rule the ticket specifies, reusing
// SC-11's already-toggleable engine API rather than building a new one. If more than one
// Synapse-category entry has an STDP spec and they map to different learning rates, the first
// found (in `model.type_library` order) wins and a warning is logged -- Phase 1's engine has
// exactly one global `SpikeEngine::learning_rate` scalar (engine.h), so per-synapse-type STDP
// parameters cannot all be represented simultaneously; this is a real, documented Phase-1
// limitation, not a silent pick.
//
// Note (existing SC-11 behavior, not changed here): `SpikeEngine::enable_plasticity` is a no-op if
// plasticity is already enabled (`learning_rate > 0.0f`) -- it will NOT overwrite an
// already-enabled engine's learning rate with this model's mapped value. Call this on a
// freshly-constructed engine (or one with `disable_plasticity()` already called) if the model's
// own STDP spec should take effect.
void apply_stdp_wiring(const ModelSpecification &model, SpikeEngine &engine);

} // namespace spikecorec::nml
