"""
nondeterminism_demo.py — shows that the Metal spike engine's *exact* spike
pattern is nondeterministic run-to-run, even with the seed, inputs, and all
parameters held fixed.

Why this happens (see src/metal/kernels.metal, kernel `step`):
  * step() runs one GPU thread per active neuron, concurrently.
  * Spikes propagate into downstream neurons with float `atomic_fetch_add`
    (network_inputs[child]). Float addition is NOT associative, so the
    scheduler-dependent order of those atomic adds changes the low-order bits
    of each neuron's accumulated input.
  * Membrane potentials sit near the spike threshold (rest 0.1, threshold 1.0),
    so those tiny differences flip individual spikes, and the flips cascade
    over ticks.

The engine ALWAYS spikes — only *which* neurons spike and the exact potentials
wobble. That's why spike_validation.py asserts "at least one spike" rather than
an exact count.

Run from the repo root:  python3.10 notebooks/nondeterminism_demo.py
"""
from spikecorec import SpikeEngine
import spikecorec as spc
import numpy as np
import os
import tempfile

SEED = 1042
SIDE = 4
LIFETIME = 10
INPUTS = [1, 4, 12]
RUNS = 25


def run_once():
    """Build a fresh engine with identical seed + inputs + parameters, simulate
    `LIFETIME` ticks, and return (last_spiked, membrane_recording). Nothing is
    randomized between calls — any difference in the output comes purely from the
    GPU, not from the Python side."""
    network = spc.small_world_torus(SIDE, seed=SEED)
    engine = SpikeEngine(network, [SIDE, SIDE], rank=1,
                         resting_mp=0.1, decay_rate=0.22, learning_rate=0.00222)
    engine.set_input_neurons(INPUTS)
    rng = np.random.default_rng(SEED)                       # fixed seed every run
    stimulus = rng.normal(size=(LIFETIME, len(INPUTS))).tolist()
    path = os.path.join(tempfile.gettempdir(), "nondet_demo.spire")
    engine.start_static_record(stimulus, LIFETIME, path, compression=None)
    last_spiked = engine.get_last_spiked().copy()
    recording = spc.read_spire_recording(path).copy()       # (LIFETIME, neurons)
    os.remove(path)
    return last_spiked, recording


# ── collect many identically-configured runs ─────────────────────────────────
last_spikes, recordings = [], []
for _ in range(RUNS):
    ls, rec = run_once()
    last_spikes.append(ls)
    recordings.append(rec)
recordings = np.stack(recordings)            # (RUNS, LIFETIME, neuron_count)
last_spikes = np.stack(last_spikes)          # (RUNS, neuron_count)
spike_counts = (last_spikes != 0).sum(axis=1)
neuron_count = recordings.shape[2]

print("=" * 72)
print(f"Same seed ({SEED}), same inputs {INPUTS}, same parameters — run {RUNS}x")
print("=" * 72)

# [1] spike counts vary across runs
print("\n[1] Neurons that spiked, per run:")
print("    ", spike_counts.tolist())
print(f"     min={spike_counts.min()}  max={spike_counts.max()}  "
      f"mean={spike_counts.mean():.1f}   (out of {neuron_count} neurons)")

# [2] essentially every run is a different pattern
distinct = len({tuple(row.tolist()) for row in last_spikes})
print(f"\n[2] Distinct last_spiked patterns: {distinct} of {RUNS} runs"
      f"  ({'ALL DIFFERENT' if distinct == RUNS else 'some repeat'})")

# [3] divergence stays at float-rounding noise, then explodes once a spike flips
per_tick_spread = recordings.std(axis=0).max(axis=1)   # max-over-neurons std-over-runs
peak = float(per_tick_spread.max()) or 1.0
# Early ticks differ only at the ULP level (~1e-7) — the atomic-add ordering is
# already nondeterministic, but the wobble is too small to flip a spike yet. Once
# it does (here, tick 4), the spread jumps by ~7 orders of magnitude.
NEGLIGIBLE = 1e-3
print("\n[3] Membrane spread across runs, per recorded tick")
print("    (max over neurons of the per-run std-dev; ~1e-7 = identical up to float rounding):")
for t, s in enumerate(per_tick_spread):
    bar = "#" * int(round(40 * float(s) / peak))
    note = "   <- float-rounding noise only" if float(s) < NEGLIGIBLE else ""
    print(f"     tick {t:2d} | {float(s):10.6f} | {bar}{note}")
first_div = next((t for t, s in enumerate(per_tick_spread) if float(s) >= NEGLIGIBLE), None)
if first_div is not None:
    print(f"     -> within float rounding (~1e-7) for ticks 0..{first_div - 1}, "
          f"then diverges macroscopically from tick {first_div}.")

# [4] concrete: trace the most-divergent neuron across several runs
focus = int(recordings.std(axis=0).sum(axis=0).argmax())
print(f"\n[4] Membrane trace of neuron {focus} (most divergent), first 6 runs:")
print("     run |", "  ".join(f"t{t:<2d}" for t in range(LIFETIME)))
for r in range(min(6, RUNS)):
    vals = " ".join(f"{v:5.2f}" for v in recordings[r, :, focus])
    print(f"     {r:3d} | {vals}")
print("     ^ note the identical early columns, then the runs disagree.")

print("\nConclusion: the engine reliably spikes, but the exact spike set and")
print("potentials are not reproducible — float atomic_fetch_add ordering in step()")
print("(kernels.metal) amplified by near-threshold dynamics. Hence the validator")
print("checks for 'at least one spike', not a fixed count.")

# ── optional figure, only if matplotlib happens to be installed ───────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(range(RUNS), spike_counts)
    ax[0].set(title="spikes per run (identical seed)", xlabel="run", ylabel="neurons spiked")
    ax[1].plot(per_tick_spread, marker="o")
    ax[1].set(title="cross-run membrane spread", xlabel="tick", ylabel="max std over neurons")
    for r in range(RUNS):
        ax[2].plot(recordings[r, :, focus], alpha=0.5)
    ax[2].set(title=f"neuron {focus}: every run", xlabel="tick", ylabel="membrane")
    out = os.path.join(os.path.dirname(__file__), "nondeterminism_demo.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nSaved figure -> {out}")
except Exception as exc:  # noqa: BLE001 - demo is text-first; figure is a bonus
    print(f"\n(matplotlib unavailable — skipped the figure: {type(exc).__name__})")
