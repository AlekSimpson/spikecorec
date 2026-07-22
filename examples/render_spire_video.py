#!/usr/bin/env python3
"""render_spire_video.py — play back a `.spire` recording as an mp4/gif.

Completes the visualization deferral `include/spikecorec/core/recording.h`'s own
`read_spire_recording` doc comment states explicitly: "Ports the file-loading half of
`play_spire_recording` (frame decoding only -- visualization stays out of scope)." This
script is that visualization step (ticket #138).

Reads one or two `.spire` recordings written by `NetworkActivityRecorder`
(examples/nml_pipeline_support.h) via the already-bound `spikecorec.read_spire_recording()`
(see src/bindings/bindings.cpp) -- a spike-raster stream (required, a 0.0/1.0 mask per neuron
per tick) and, optionally, a membrane-potential stream (same shape) -- and renders an animated
grid: each neuron placed at its own `(row, column)` position on a `side x side` torus (the
SAME `row * side_length + column` flat-index convention every torus example in this directory
generates its network with, see examples/glif_torus_network.h's own header comment), membrane
potential as a per-tick heatmap and spikes overlaid as bright markers on the tick they occur.

For a non-torus recording (no natural 2D grid -- e.g. a small hand-built network), pass
`--side 1` to fall back to a single-row layout instead of guessing a square grid.

Usage:
    ./examples/render_spire_video.py recordings/glif3_torus_spikes.spire \\
        --side 8 --membrane recordings/glif3_torus_membrane.spire

    ./examples/render_spire_video.py recordings/glif3_torus_spikes.spire \\
        --side 8 --output glif3_torus.gif

Dependencies (not part of spikecorec's own core C++/CUDA/Metal build):
    numpy, matplotlib -- both already listed as optional Python dependencies
    (pyproject.toml's `[project.optional-dependencies].dev`, extended here to also cover this
    script under a new `examples` group). Install with:

        pip install -e ".[examples]"

    mp4 output additionally needs an `ffmpeg` binary on PATH (matplotlib's `FFMpegWriter`); with
    no `ffmpeg` available, pass `--output foo.gif` instead (Pillow, a matplotlib dependency
    already, writes gif directly -- no extra install).

    `spikecorec` itself must be built and importable first: `make python` (see this repo's
    top-level README/Makefile).
"""

import argparse
import os
import sys

import numpy as np


def parse_arguments(argument_list):
    parser = argparse.ArgumentParser(
        description="Render a .spire spike-raster (+ optional membrane-potential) recording as "
                    "a playable mp4/gif, one neuron per its real network grid position.")
    parser.add_argument(
        "spike_recording_path", metavar="SPIKE_RECORDING",
        help="path to a .spire spike-raster recording (0.0/1.0 mask per neuron per tick, e.g. "
             "recordings/glif3_torus_spikes.spire)")
    parser.add_argument(
        "--membrane", dest="membrane_recording_path", default=None,
        help="path to the matching .spire membrane-potential recording (same neuron count/frame "
             "count as SPIKE_RECORDING). Rendered as a per-tick heatmap behind the spike overlay; "
             "omit to render spikes alone against a blank grid.")
    parser.add_argument(
        "--side", type=int, default=None,
        help="torus side length (the population holds side^2 neurons at row*side+column, "
             "matching examples/glif_torus_network.h's own generated layout). Defaults to "
             "round(sqrt(neuron_count)) if that is an exact square; pass --side 1 for a "
             "non-square/non-torus recording (a single-row layout).")
    parser.add_argument(
        "--output", default=None,
        help="output video path (.mp4 or .gif, chosen by extension). Defaults to "
             "SPIKE_RECORDING's own path with its .spire extension replaced by .mp4.")
    parser.add_argument(
        "--fps", type=int, default=30, help="playback frames per second (default: 30)")
    parser.add_argument(
        "--stride", type=int, default=None,
        help="render only every Nth simulated tick, to keep a long run's video a reasonable "
             "length. Defaults to whatever keeps the rendered frame count at or below "
             "--max-frames.")
    parser.add_argument(
        "--max-frames", type=int, default=400,
        help="target rendered frame count when --stride is not given explicitly (default: 400)")
    parser.add_argument(
        "--dt", type=float, default=1e-4,
        help="seconds per simulated tick, for the on-screen time readout (default: 0.0001, "
             "matching every example's own --dt default)")
    return parser.parse_args(argument_list)


def infer_grid_side_length(neuron_count, requested_side_length):
    if requested_side_length is not None:
        if requested_side_length * requested_side_length > neuron_count:
            raise ValueError(
                f"--side {requested_side_length} implies {requested_side_length * requested_side_length} "
                f"neurons, more than this recording's own {neuron_count}")
        return requested_side_length

    square_root = int(round(neuron_count ** 0.5))
    if square_root * square_root == neuron_count:
        return square_root
    raise ValueError(
        f"neuron_count={neuron_count} is not a perfect square, so a torus grid side length "
        f"cannot be inferred -- pass --side explicitly (--side 1 for a single-row layout)")


def default_output_path(spike_recording_path):
    base_path, _ = os.path.splitext(spike_recording_path)
    return base_path + ".mp4"


def choose_stride(frame_count, requested_stride, max_frames):
    if requested_stride is not None:
        return max(requested_stride, 1)
    if frame_count <= max_frames:
        return 1
    return -(-frame_count // max_frames)  # ceiling division


def render_spire_video(
    spike_frames, membrane_frames, side_length, output_path, fps, stride, dt_seconds
):
    # Local import: matplotlib is only needed once we actually render, so a caller hitting an
    # earlier validation error (e.g. a non-square neuron_count) never pays its import cost.
    import matplotlib
    matplotlib.use("Agg")  # headless -- this script only ever writes a file, never shows a window
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    frame_count, neuron_count = spike_frames.shape
    row_count = -(-neuron_count // side_length)  # ceiling division -- covers a non-square total

    def frame_to_grid(values_by_neuron):
        grid = np.full((row_count * side_length,), np.nan, dtype=np.float32)
        grid[:neuron_count] = values_by_neuron
        return grid.reshape(row_count, side_length)

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.set_xticks([])
    axes.set_yticks([])

    if membrane_frames is not None:
        membrane_minimum = float(np.nanmin(membrane_frames))
        membrane_maximum = float(np.nanmax(membrane_frames))
        if membrane_maximum - membrane_minimum < 1e-12:
            membrane_maximum = membrane_minimum + 1e-12
        heatmap = axes.imshow(
            frame_to_grid(membrane_frames[0]), cmap="viridis", vmin=membrane_minimum,
            vmax=membrane_maximum, interpolation="nearest")
        figure.colorbar(heatmap, ax=axes, label="membrane potential (V)", fraction=0.046, pad=0.04)
    else:
        heatmap = axes.imshow(
            frame_to_grid(np.zeros(neuron_count, dtype=np.float32)), cmap="gray", vmin=0.0,
            vmax=1.0, interpolation="nearest")

    # Spikes overlaid as a separate scatter layer so they pop visually against the membrane-
    # potential heatmap (or a blank grid) underneath, rather than being folded into the same
    # color scale as the continuous trace.
    neuron_rows, neuron_columns = np.divmod(np.arange(neuron_count), side_length)
    spike_scatter = axes.scatter([], [], s=120, c="red", marker="o", linewidths=0)

    title = axes.set_title("")

    def update(rendered_frame_index):
        tick = rendered_frame_index * stride
        if membrane_frames is not None:
            heatmap.set_data(frame_to_grid(membrane_frames[tick]))

        spiking_neuron_mask = spike_frames[tick] > 0.5
        spike_scatter.set_offsets(
            np.column_stack((neuron_columns[spiking_neuron_mask], neuron_rows[spiking_neuron_mask])))

        title.set_text(f"tick {tick}   t={tick * dt_seconds * 1000.0:.2f}ms   "
                        f"spikes this tick: {int(spiking_neuron_mask.sum())}")
        return (heatmap, spike_scatter, title) if membrane_frames is not None else (spike_scatter, title)

    rendered_frame_count = -(-frame_count // stride)  # ceiling division
    animation_object = animation.FuncAnimation(
        figure, update, frames=rendered_frame_count, interval=1000.0 / fps, blit=False)

    _, output_extension = os.path.splitext(output_path)
    if output_extension.lower() == ".gif":
        animation_object.save(output_path, writer=animation.PillowWriter(fps=fps))
    else:
        animation_object.save(output_path, writer=animation.FFMpegWriter(fps=fps))

    plt.close(figure)


def main(argument_list=None):
    arguments = parse_arguments(sys.argv[1:] if argument_list is None else argument_list)

    # Imported here, not at module scope: everything above this point (argument parsing,
    # grid-size inference) needs neither spikecorec nor a built extension module, so a caller
    # exploring `--help` or hitting a usage error never needs `make python` to have succeeded.
    import spikecorec

    spike_frames = spikecorec.read_spire_recording(arguments.spike_recording_path)
    membrane_frames = None
    if arguments.membrane_recording_path is not None:
        membrane_frames = spikecorec.read_spire_recording(arguments.membrane_recording_path)
        if membrane_frames.shape != spike_frames.shape:
            raise ValueError(
                f"--membrane recording shape {membrane_frames.shape} does not match "
                f"SPIKE_RECORDING's own shape {spike_frames.shape} (frame_count, neuron_count)")

    frame_count, neuron_count = spike_frames.shape
    side_length = infer_grid_side_length(neuron_count, arguments.side)
    output_path = arguments.output or default_output_path(arguments.spike_recording_path)
    stride = choose_stride(frame_count, arguments.stride, arguments.max_frames)

    print(f"[render_spire_video] {neuron_count} neurons, {frame_count} ticks, "
          f"{side_length}-wide grid, rendering every {stride} tick(s) "
          f"({-(-frame_count // stride)} frames) -> {output_path}")

    render_spire_video(
        spike_frames, membrane_frames, side_length, output_path, arguments.fps, stride, arguments.dt)

    print(f"[render_spire_video] wrote {output_path}")


if __name__ == "__main__":
    main()
