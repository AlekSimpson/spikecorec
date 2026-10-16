# Resilience review

A review of defects in the engine, made on 2026-10-12 and 2026-10-13 (branch
`agent_integration_2`, commit `04d1b3d`). It confirmed 51 findings, each reproduced with a scratch
program or document, or traced through the code. All but the ones below were fixed in October
2026; their fixes and the tests that cover them are in the code.

This page keeps only what is still open and what was left as it is on purpose. The numbers are
the review's own.

## Open

### 14. Included NeuroML files are never validated against the schema (high)

`validate_lems_schema` checks only the file passed to the engine, and skips it when its root is
not `<neuroml>`. The schema does not allow `<Simulation>` inside `<neuroml>`, so every runnable
model has a LEMS root, and its included `.nml` files are never validated.

- **Done so far:** a tag that matches a ComponentType but for case (`<connectionWd>`) is refused,
  and unit typos are refused by the strict `resolve_quantity` (finding 15).
- **Still open:** validating included `.nml` files against the XSD would refuse documents that
  declare or use their own ComponentTypes, which most test models and the shipped examples do.
  How to validate them needs a decision.
- **Location:** `src/spikecorec/core/engine.cpp` (the validation call);
  `src/spikecorec/nml/parser.cpp` (`validate_lems_schema` and the include queue).

### 12. Regular generators drift in float32 (low)

The time `t` is computed on the host in f64 and passed to the kernel as `simulation_time` (f32
on Metal, f64 on CUDA). A `spikeGenerator` still advances `tnext = tnext + period` in f32 state,
and each addition rounds to the ulp of `t`. Because the period is not a multiple of that ulp, the
error is systematic: in [16, 32) s, a 1 ms period becomes 0.99945 ms.

- **Measured:** 1 ms period, dt 0.1 ms, 30 s: 30,004 spikes instead of 29,999; the last lands at
  tick 299,992 instead of 300,040.
- **Location:** the generated `tnext` update; cell state is f32.

## Left as it is

### 11. Recorded traces are labelled one step early, and the initial state is not recorded (low)

`record_tick` stores the state after the kernel has advanced it to (tick + 1) · dt, but labels the
row tick · dt. Left as it is (user decision, 2026-10-13).
