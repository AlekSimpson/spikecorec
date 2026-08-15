#!/usr/bin/env python3
"""Render a .spire membrane recording to a video.

A .spire file is what SpikeEngine::record_membrane_video() writes: a 4-byte big-endian
neuron count, then one frame per recorded tick, each frame being `neuron_count` native
float32 membrane potentials. This turns that into an animation of the population laid out
as a grid, one cell per neuron, coloured by membrane potential, with a spike raster and a
population-rate trace underneath so the network's activity is visible as a whole and not
just as a flickering grid.

    python3 examples/render_membrane_video.py build/glif3_network_membrane.spire

Writes alongside the recording with a .mp4 extension unless --output says otherwise.
mp4 needs ffmpeg on PATH; --output something.gif uses Pillow instead.

Requires numpy and matplotlib. On this machine: /opt/homebrew/bin/python3.10.
"""

import argparse
import math
import os
import struct
import sys

import numpy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot
from matplotlib import animation, gridspec


SPIRE_HEADER_BYTES = 4


def parse_arguments(argument_list):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", help="path to the .spire membrane recording")
    parser.add_argument("--spikes", default=None,
                        help="optional TIME_ID spike file to draw as a raster. Defaults to "
                             "the recording's own path with _membrane.spire replaced by "
                             "_spikes.dat, when that file exists.")
    parser.add_argument("--output", default=None,
                        help="output path; .mp4 (ffmpeg) or .gif (Pillow). Defaults to the "
                             "recording path with a .mp4 extension.")
    parser.add_argument("--frames-per-second", type=int, default=30)
    parser.add_argument("--seconds-per-frame", type=float, default=None,
                        help="simulated seconds each recorded frame covers, used to label "
                             "the time axis. Inferred from --duration when given.")
    parser.add_argument("--duration", type=float, default=None,
                        help="simulated duration of the whole recording, in seconds")
    parser.add_argument("--max-frames", type=int, default=600,
                        help="subsample the recording down to at most this many frames")
    parser.add_argument("--title", default=None)
    return parser.parse_args(argument_list)


def read_spire_recording(path):
    """Returns (frames, neuron_count) with frames shaped [frame_count][neuron_count]."""
    with open(path, "rb") as recording_file:
        header = recording_file.read(SPIRE_HEADER_BYTES)
        if len(header) < SPIRE_HEADER_BYTES:
            raise SystemExit(f"{path}: too short to contain a .spire header")

        neuron_count = struct.unpack(">I", header)[0]
        if neuron_count == 0:
            raise SystemExit(f"{path}: header says zero neurons")

        payload = numpy.frombuffer(recording_file.read(), dtype=numpy.float32)

    frame_count = payload.size // neuron_count
    if frame_count == 0:
        raise SystemExit(f"{path}: header says {neuron_count} neurons but there are no "
                         f"complete frames after it")

    leftover = payload.size - frame_count * neuron_count
    if leftover:
        print(f"{path}: dropping a partial trailing frame ({leftover} of {neuron_count} "
              f"values) - the writer was interrupted mid-frame", file=sys.stderr)

    frames = payload[:frame_count * neuron_count].reshape(frame_count, neuron_count)
    return frames, neuron_count


def read_spike_file(path):
    """Returns (times, neuron_indices) from a TIME_ID two-column file."""
    times = []
    indices = []
    with open(path) as spike_file:
        for line in spike_file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            times.append(float(parts[0]))
            indices.append(int(parts[1]))
    return numpy.array(times), numpy.array(indices)


def grid_side_length(neuron_count):
    """Squarest grid that holds every neuron."""
    side = int(math.ceil(math.sqrt(neuron_count)))
    return side


def default_spike_path(recording_path):
    if recording_path.endswith("_membrane.spire"):
        candidate = recording_path[: -len("_membrane.spire")] + "_spikes.dat"
        if os.path.exists(candidate):
            return candidate
    return None


def main(argument_list):
    arguments = parse_arguments(argument_list)

    frames, neuron_count = read_spire_recording(arguments.recording)
    frame_count = frames.shape[0]

    # Subsample rather than render every frame: a 2 s run recorded every 10 ticks is 2,000
    # frames, and a 600-frame video at 30 fps is already 20 seconds of watching.
    if frame_count > arguments.max_frames:
        selected = numpy.linspace(0, frame_count - 1, arguments.max_frames).astype(int)
        frames = frames[selected]
        frame_indices = selected
    else:
        frame_indices = numpy.arange(frame_count)

    duration = arguments.duration
    seconds_per_frame = arguments.seconds_per_frame
    if seconds_per_frame is None and duration is not None:
        seconds_per_frame = duration / float(frame_count)
    frame_times = (frame_indices * seconds_per_frame
                   if seconds_per_frame is not None else frame_indices.astype(float))
    time_label = "time (s)" if seconds_per_frame is not None else "recorded frame"

    spike_path = arguments.spikes or default_spike_path(arguments.recording)
    spike_times, spike_indices = (read_spike_file(spike_path) if spike_path
                                  else (numpy.array([]), numpy.array([])))

    output_path = arguments.output
    if output_path is None:
        base = arguments.recording
        for suffix in (".spire.gz", ".spire.xz", ".spire.bz2", ".spire"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        output_path = base + ".mp4"

    side = grid_side_length(neuron_count)
    padded = numpy.full((frames.shape[0], side * side), numpy.nan, dtype=numpy.float32)
    padded[:, :neuron_count] = frames
    grid_frames = padded.reshape(frames.shape[0], side, side)

    # Colour scale from the data itself, clipped to sane membrane potentials so one
    # numerical excursion does not flatten the whole scale.
    finite = frames[numpy.isfinite(frames)]
    if finite.size == 0:
        raise SystemExit(f"{arguments.recording}: every recorded value is non-finite")
    low = float(numpy.percentile(finite, 1))
    high = float(numpy.percentile(finite, 99.5))
    if high - low < 1e-6:
        high = low + 1e-6

    # The grid panel is about 40% of the figure's width, so the figure is sized to give it
    # roughly one pixel per neuron rather than squeezing a 1024x1024 sheet into 450 px and
    # letting the resampler decide what survives. Floored so small grids still get a
    # readable figure, capped so a very large one stays a sane video size.
    grid_panel_pixels = min(side, 1100)
    figure_width_pixels = int(min(max(grid_panel_pixels / 0.40, 1100), 2600))
    figure_width_inches = figure_width_pixels / 100.0

    figure = pyplot.figure(figsize=(figure_width_inches, figure_width_inches * 0.636),
                           facecolor="#0e1416")
    # Generous wspace: the grid's colour bar sits between the two top panels, and a
    # tighter column gap puts its tick labels on top of the trace axis's.
    layout = gridspec.GridSpec(2, 2, height_ratios=[3, 1.4], width_ratios=[1.15, 1],
                               hspace=0.34, wspace=0.34,
                               left=0.06, right=0.96, top=0.90, bottom=0.09)

    grid_axes = figure.add_subplot(layout[0, 0])
    trace_axes = figure.add_subplot(layout[0, 1])
    raster_axes = figure.add_subplot(layout[1, :])

    for axes in (grid_axes, trace_axes, raster_axes):
        axes.set_facecolor("#121a1c")
        for spine in axes.spines.values():
            spine.set_color("#37474a")
        axes.tick_params(colors="#94a5a8", labelsize=8)
        axes.xaxis.label.set_color("#94a5a8")
        axes.yaxis.label.set_color("#94a5a8")

    # The grid is the squarest rectangle that holds every neuron, so the last row is
    # usually short. Those cells are NaN and are painted as background rather than as the
    # bottom of the colour scale, which would read as a block of resting neurons.
    colour_map = matplotlib.colormaps["magma"].copy()
    colour_map.set_bad("#121a1c")

    image = grid_axes.imshow(grid_frames[0], vmin=low, vmax=high, cmap=colour_map,
                             interpolation="nearest")
    grid_axes.set_xticks([])
    grid_axes.set_yticks([])
    grid_axes.set_title(f"membrane potential, {neuron_count} neurons",
                        color="#dce5e4", fontsize=10)

    colour_bar = figure.colorbar(image, ax=grid_axes, fraction=0.046, pad=0.06)
    # As a title rather than a rotated axis label: the rotated glyph lands on top of
    # the tick labels at this figure size.
    colour_bar.ax.set_title("volts", color="#94a5a8", fontsize=7, pad=6)
    colour_bar.ax.tick_params(colors="#94a5a8", labelsize=7)
    colour_bar.outline.set_edgecolor("#37474a")

    # A handful of individual traces, so single-cell behaviour is visible next to the
    # population view.
    traced = numpy.linspace(0, neuron_count - 1, min(6, neuron_count)).astype(int)
    for offset, neuron_index in enumerate(traced):
        trace_axes.plot(frame_times, frames[:, neuron_index], linewidth=0.8,
                        color=pyplot.cm.viridis(offset / max(1, len(traced) - 1)),
                        alpha=0.9)
    trace_axes.set_xlim(frame_times[0], frame_times[-1])
    trace_axes.set_xlabel(time_label, fontsize=8)
    trace_axes.set_ylabel("V", fontsize=8)
    trace_axes.set_title("six membrane traces", color="#dce5e4", fontsize=10)
    trace_marker = trace_axes.axvline(frame_times[0], color="#e0824e", linewidth=1.0)

    if spike_times.size:
        raster_axes.scatter(spike_times, spike_indices, s=0.6, c="#63b3c2",
                            marker=".", linewidths=0)
        raster_axes.set_ylim(-1, neuron_count)
        raster_axes.set_ylabel("neuron", fontsize=8)
        raster_axes.set_title(
            f"spike raster - {spike_times.size} spikes", color="#dce5e4", fontsize=10)
        raster_axes.set_xlim(frame_times[0], frame_times[-1])
    else:
        raster_axes.text(0.5, 0.5, "no spike file", transform=raster_axes.transAxes,
                         ha="center", va="center", color="#6b7c80")
        raster_axes.set_xlim(frame_times[0], frame_times[-1])
    raster_axes.set_xlabel(time_label, fontsize=8)
    raster_marker = raster_axes.axvline(frame_times[0], color="#e0824e", linewidth=1.0)

    title = arguments.title or os.path.basename(arguments.recording)
    heading = figure.suptitle(f"{title}    {frame_times[0]:.3f}",
                             color="#dce5e4", fontsize=13)

    def draw_frame(frame_index):
        image.set_data(grid_frames[frame_index])
        position = frame_times[frame_index]
        trace_marker.set_xdata([position, position])
        raster_marker.set_xdata([position, position])
        suffix = " s" if seconds_per_frame is not None else ""
        heading.set_text(f"{title}    {position:.3f}{suffix}")
        return image, trace_marker, raster_marker, heading

    animator = animation.FuncAnimation(figure, draw_frame, frames=grid_frames.shape[0],
                                       interval=1000 / arguments.frames_per_second,
                                       blit=False)

    if output_path.endswith(".gif"):
        writer = animation.PillowWriter(fps=arguments.frames_per_second)
    else:
        writer = animation.FFMpegWriter(fps=arguments.frames_per_second, bitrate=2400)

    animator.save(output_path, writer=writer, savefig_kwargs={"facecolor": "#0e1416"})
    pyplot.close(figure)

    print(f"{arguments.recording}: {frame_count} frames of {neuron_count} neurons "
          f"-> {output_path} ({grid_frames.shape[0]} rendered at "
          f"{arguments.frames_per_second} fps, {side}x{side} grid in a "
          f"{figure_width_pixels}px figure)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
