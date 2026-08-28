"""Recording and rendering helpers shared by the demos.

This is the one module a demo imports. Everything else a demo needs is written out in
the demo's own function, so each one reads top to bottom without following a call into
another file.

A .spire file is what SpikeEngine.record_membrane_video() writes: a 4-byte big-endian
neuron count, then one frame per recorded tick, each frame being `neuron_count` native
float32 membrane potentials. render_membrane_video() turns that into an animation of the
population as a grid, one cell per neuron, with membrane traces and a spike raster
beside it.

Requires numpy and matplotlib. Writing .mp4 needs ffmpeg on PATH; an .gif output path
uses Pillow instead.
"""

import math
import os
import struct
from pathlib import Path

import numpy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot
from matplotlib import animation, gridspec

__all__ = [
    "Path",
    "demo_path",
    "demo_directory",
    "read_spire_recording",
    "read_spike_file",
    "render_membrane_video",
]

SPIRE_HEADER_BYTES = 4

# Membrane potentials are volts in the engine and millivolts everywhere a person reads
# them. Every number this module displays is converted once, at the point it is read.
VOLTS_TO_MILLIVOLTS = 1000.0

# The axis a membrane trace is drawn against, in millivolts. Fixed rather than fitted to
# the data so a trace reads as a membrane potential at a glance and so two demos are
# comparable side by side. Widened, never narrowed, when a model leaves it.
DEFAULT_MILLIVOLT_RANGE = (-80.0, -40.0)

_GROUND = "#0e1416"
_PANEL = "#121a1c"
_EDGE = "#37474a"
_LABEL = "#94a5a8"
_TEXT = "#dce5e4"
_MARKER = "#e0824e"
_TRACE_COLOURS = ["#63b3c2", "#e0824e", "#9ccf7a", "#c98bd0"]


def demo_directory():
    """Where demo artefacts are written, created on first use."""
    directory = Path(__file__).resolve().parent.parent / "build" / "demos"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def demo_path(filename):
    """A path inside the demo output directory, as a string the engine can take."""
    return str(demo_directory() / filename)


def read_spire_recording(path):
    """Returns (frames, neuron_count) with frames shaped [frame_count][neuron_count]."""
    with open(path, "rb") as recording_file:
        header = recording_file.read(SPIRE_HEADER_BYTES)
        if len(header) < SPIRE_HEADER_BYTES:
            raise ValueError(f"{path}: too short to contain a .spire header")

        neuron_count = struct.unpack(">I", header)[0]
        if neuron_count == 0:
            raise ValueError(f"{path}: header says zero neurons")

        payload = numpy.frombuffer(recording_file.read(), dtype=numpy.float32)

    frame_count = payload.size // neuron_count
    if frame_count == 0:
        raise ValueError(f"{path}: header says {neuron_count} neurons but there are no "
                         f"complete frames after it")

    frames = payload[:frame_count * neuron_count].reshape(frame_count, neuron_count)
    return frames, neuron_count


def read_spike_file(path):
    """Returns (times, neuron_indices) from a TIME_ID two-column file."""
    times = []
    indices = []
    with open(path) as spike_file:
        for line in spike_file:
            parts = line.split()
            if len(parts) < 2:
                continue
            times.append(float(parts[0]))
            indices.append(int(parts[1]))
    return numpy.array(times), numpy.array(indices)


def _grid_side_length(neuron_count):
    """Squarest grid that holds every neuron."""
    return int(math.ceil(math.sqrt(neuron_count)))


def _millivolt_range(frames_millivolts, requested):
    """The display range, widened to hold the data but never narrower than requested."""
    finite = frames_millivolts[numpy.isfinite(frames_millivolts)]
    if finite.size == 0:
        raise ValueError("every recorded value is non-finite")

    low, high = requested
    low = min(low, float(numpy.percentile(finite, 0.5)))
    high = max(high, float(numpy.percentile(finite, 99.5)))
    if high - low < 1.0:
        high = low + 1.0
    return low, high


def render_membrane_video(recording_path,
                          output_path=None,
                          spikes_path=None,
                          duration=None,
                          trace_neurons=None,
                          frames_per_second=30,
                          max_frames=300,
                          millivolt_range=DEFAULT_MILLIVOLT_RANGE,
                          title=None):
    """Renders a .spire recording to a video and returns the path written.

    trace_neurons selects which cells get a membrane trace, as (neuron_index, label)
    pairs -- normally one excitatory cell and one inhibitory one. Two traces read as
    two cells; six read as a thicket. Defaults to the first neuron alone.

    duration is the simulated length of the whole recording in seconds, used to label
    the time axis. Without it the axis counts recorded frames.

    spikes_path is a TIME_ID spike file. Its spikes fill the raster panel, and the ones
    belonging to a traced neuron are marked above that neuron's trace -- a spike is
    followed immediately by a reset, so the trace alone dips where a person expects a
    peak.
    """
    recording_path = str(recording_path)
    frames, neuron_count = read_spire_recording(recording_path)
    frames = frames * VOLTS_TO_MILLIVOLTS
    frame_count = frames.shape[0]

    seconds_per_frame = duration / float(frame_count) if duration is not None else None
    time_label = "time (s)" if seconds_per_frame is not None else "recorded frame"

    def times_for(indices):
        return (indices * seconds_per_frame if seconds_per_frame is not None
                else indices.astype(float))

    # The traces are drawn from every recorded frame; only the animated grid is
    # subsampled. Plotting the subsampled frames instead would sample a 10 ms sawtooth
    # every 1.7 ms and draw peaks that wander between -56 and -52 mV, which is an
    # artefact of the sampling rather than anything the cell did.
    trace_times = times_for(numpy.arange(frame_count))

    # Subsample the animation: a 500 ms run recorded every 5 ticks is 1,000 frames, and
    # 300 frames at 30 fps is already 10 seconds of watching.
    if frame_count > max_frames:
        selected = numpy.linspace(0, frame_count - 1, max_frames).astype(int)
        grid_source = frames[selected]
        frame_indices = selected
    else:
        grid_source = frames
        frame_indices = numpy.arange(frame_count)
    frame_times = times_for(frame_indices)

    if spikes_path is not None and os.path.exists(str(spikes_path)):
        spike_times, spike_indices = read_spike_file(str(spikes_path))
    else:
        spike_times, spike_indices = numpy.array([]), numpy.array([])

    if output_path is None:
        base = recording_path
        for suffix in (".spire.gz", ".spire.xz", ".spire.bz2", ".spire"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        output_path = base + ".mp4"
    output_path = str(output_path)

    if not trace_neurons:
        trace_neurons = [(0, "neuron 0")]

    low, high = _millivolt_range(frames, millivolt_range)

    side = _grid_side_length(neuron_count)
    padded = numpy.full((grid_source.shape[0], side * side), numpy.nan, dtype=numpy.float32)
    padded[:, :neuron_count] = grid_source
    grid_frames = padded.reshape(grid_source.shape[0], side, side)

    # The grid panel is about 40% of the figure's width, so the figure is sized to give
    # it roughly one pixel per neuron rather than letting the resampler decide what
    # survives. Floored so small grids still get a readable figure, capped so a very
    # large one stays a sane video size.
    grid_panel_pixels = min(side, 1100)
    figure_width_pixels = int(min(max(grid_panel_pixels / 0.40, 1100), 2600))
    figure_width_inches = figure_width_pixels / 100.0

    figure = pyplot.figure(figsize=(figure_width_inches, figure_width_inches * 0.636),
                           facecolor=_GROUND)
    # Generous wspace: the grid's colour bar sits between the two top panels, and a
    # tighter column gap puts its tick labels on top of the trace axis's.
    layout = gridspec.GridSpec(2, 2, height_ratios=[3, 1.4], width_ratios=[1.15, 1],
                               hspace=0.34, wspace=0.34,
                               left=0.06, right=0.96, top=0.90, bottom=0.09)

    grid_axes = figure.add_subplot(layout[0, 0])
    trace_axes = figure.add_subplot(layout[0, 1])
    raster_axes = figure.add_subplot(layout[1, :])

    for axes in (grid_axes, trace_axes, raster_axes):
        axes.set_facecolor(_PANEL)
        for spine in axes.spines.values():
            spine.set_color(_EDGE)
        axes.tick_params(colors=_LABEL, labelsize=8)
        axes.xaxis.label.set_color(_LABEL)
        axes.yaxis.label.set_color(_LABEL)

    # The grid is the squarest rectangle that holds every neuron, so the last row is
    # usually short. Those cells are NaN and are painted as background rather than as
    # the bottom of the colour scale, which would read as resting neurons.
    colour_map = matplotlib.colormaps["magma"].copy()
    colour_map.set_bad(_PANEL)

    image = grid_axes.imshow(grid_frames[0], vmin=low, vmax=high, cmap=colour_map,
                             interpolation="nearest")
    grid_axes.set_xticks([])
    grid_axes.set_yticks([])
    grid_axes.set_title(f"membrane potential, {neuron_count} neurons",
                        color=_TEXT, fontsize=10)

    colour_bar = figure.colorbar(image, ax=grid_axes, fraction=0.046, pad=0.06)
    # As a title rather than a rotated axis label: the rotated glyph lands on top of the
    # tick labels at this figure size.
    colour_bar.ax.set_title("mV", color=_LABEL, fontsize=7, pad=6)
    colour_bar.ax.tick_params(colors=_LABEL, labelsize=7)
    colour_bar.outline.set_edgecolor(_EDGE)

    # One trace per named cell, and a row of spike markers above the traces. The reset
    # is what the trace shows at a spike, so without the markers the moment a cell fires
    # reads as a downward step rather than an event.
    spike_row_height = (high - low) * 0.04
    for offset, (neuron_index, label) in enumerate(trace_neurons):
        colour = _TRACE_COLOURS[offset % len(_TRACE_COLOURS)]
        trace_axes.plot(trace_times, frames[:, neuron_index], linewidth=1.1,
                        color=colour, alpha=0.95, label=label)

        if spike_times.size:
            own_spikes = spike_times[spike_indices == neuron_index]
            if own_spikes.size:
                marker_height = high - spike_row_height * (offset + 0.6)
                trace_axes.plot(own_spikes,
                                numpy.full(own_spikes.shape, marker_height),
                                linestyle="none", marker="|", markersize=5,
                                markeredgewidth=1.0, color=colour)

    trace_axes.set_xlim(trace_times[0], trace_times[-1])
    trace_axes.set_ylim(low, high)
    trace_axes.set_xlabel(time_label, fontsize=8)
    trace_axes.set_ylabel("membrane potential (mV)", fontsize=8)
    trace_axes.set_title("membrane traces", color=_TEXT, fontsize=10)
    legend = trace_axes.legend(loc="lower right", fontsize=7, framealpha=0.25,
                               facecolor=_PANEL, edgecolor=_EDGE)
    for entry in legend.get_texts():
        entry.set_color(_LABEL)
    trace_marker = trace_axes.axvline(frame_times[0], color=_MARKER, linewidth=1.0)

    if spike_times.size:
        # A size tuned for 20,000 spikes is invisible at 30. Scaled to the count so a
        # small network's raster is legible and a large one does not become a solid
        # block.
        marker_size = float(numpy.clip(2000.0 / max(spike_times.size, 1), 0.6, 12.0))
        raster_axes.scatter(spike_times, spike_indices, s=marker_size, c="#63b3c2",
                            marker=".", linewidths=0)
        raster_axes.set_ylim(-1, neuron_count)
        raster_axes.set_ylabel("neuron", fontsize=8)
        raster_axes.set_title(f"spike raster - {spike_times.size} spikes",
                              color=_TEXT, fontsize=10)
    else:
        raster_axes.text(0.5, 0.5, "no spike file", transform=raster_axes.transAxes,
                         ha="center", va="center", color="#6b7c80")
    raster_axes.set_xlim(frame_times[0], frame_times[-1])
    raster_axes.set_xlabel(time_label, fontsize=8)
    raster_marker = raster_axes.axvline(frame_times[0], color=_MARKER, linewidth=1.0)

    heading_text = title or os.path.basename(recording_path)
    heading = figure.suptitle(f"{heading_text}    {frame_times[0]:.3f}",
                              color=_TEXT, fontsize=13)

    def draw_frame(frame_index):
        image.set_data(grid_frames[frame_index])
        position = frame_times[frame_index]
        trace_marker.set_xdata([position, position])
        raster_marker.set_xdata([position, position])
        suffix = " s" if seconds_per_frame is not None else ""
        heading.set_text(f"{heading_text}    {position:.3f}{suffix}")
        return image, trace_marker, raster_marker, heading

    animator = animation.FuncAnimation(figure, draw_frame, frames=grid_frames.shape[0],
                                       interval=1000 / frames_per_second, blit=False)

    if output_path.endswith(".gif"):
        writer = animation.PillowWriter(fps=frames_per_second)
    else:
        writer = animation.FFMpegWriter(fps=frames_per_second, bitrate=2400)

    animator.save(output_path, writer=writer, savefig_kwargs={"facecolor": _GROUND})
    pyplot.close(figure)

    print(f"{os.path.basename(recording_path)}: {frame_count} frames of {neuron_count} "
          f"neurons -> {output_path} ({grid_frames.shape[0]} rendered at "
          f"{frames_per_second} fps)")
    return output_path
