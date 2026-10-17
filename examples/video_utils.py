"""Recording and rendering helpers for the demo notebook.

Spike_Engine.record_membrane_video() writes a .spire file: a 4-byte big-endian neuron
count, then one frame of `neuron_count` native float32 membrane potentials per recorded
tick. render_membrane_video() animates it as a grid with traces and a spike raster.

Requires numpy and matplotlib. An .mp4 needs ffmpeg on PATH; a .gif uses Pillow.
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

# The engine records volts; this module displays millivolts.
VOLTS_TO_MILLIVOLTS = 1000.0

# Trace axis in millivolts. Fixed so demos are comparable; widened, never narrowed, to fit the data.
DEFAULT_MILLIVOLT_RANGE = (-80.0, -40.0)

_GROUND = "#0e1416"
_PANEL = "#121a1c"
_EDGE = "#37474a"
_LABEL = "#94a5a8"
_TEXT = "#dce5e4"
_MARKER = "#e0824e"
_TRACE_COLOURS = ["#63b3c2", "#e0824e", "#9ccf7a", "#c98bd0"]


def demo_directory():
    """build/demos under the repository root, created on first use."""
    directory = Path(__file__).resolve().parent.parent / "build" / "demos"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def demo_path(filename):
    """A path inside the demo output directory, as a string the engine can take."""
    return str(demo_directory() / filename)


def read_spire_recording(path):
    """Returns (frames, neuron_count), frames [frame_count][neuron_count]. Uncompressed only."""
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

    trace_neurons: (neuron_index, label) pairs that get a membrane trace; default neuron 0.
    duration: the recording's simulated length in seconds; without it the time axis counts
    frames. spikes_path: a TIME_ID spike file for the raster and the markers over each trace.
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

    # Traces use every recorded frame; only the animated grid is subsampled, to at most
    # max_frames. Subsampled traces would alias the spike sawtooth.
    trace_times = times_for(numpy.arange(frame_count))

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

    # The grid panel is about 40% of the figure's width; size the figure (1100 to 2600
    # pixels wide) to give it about one pixel per neuron.
    grid_panel_pixels = min(side, 1100)
    figure_width_pixels = int(min(max(grid_panel_pixels / 0.40, 1100), 2600))
    figure_width_inches = figure_width_pixels / 100.0

    figure = pyplot.figure(figsize=(figure_width_inches, figure_width_inches * 0.636),
                           facecolor=_GROUND)
    # Wide column gap so the colour bar's tick labels clear the trace axis.
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

    # Padding cells at the end of the grid are NaN; paint them as background, not as the
    # low end of the scale, where they would look like resting neurons.
    colour_map = matplotlib.colormaps["magma"].copy()
    colour_map.set_bad(_PANEL)

    image = grid_axes.imshow(grid_frames[0], vmin=low, vmax=high, cmap=colour_map,
                             interpolation="nearest")
    grid_axes.set_xticks([])
    grid_axes.set_yticks([])
    grid_axes.set_title(f"membrane potential, {neuron_count} neurons",
                        color=_TEXT, fontsize=10)

    colour_bar = figure.colorbar(image, ax=grid_axes, fraction=0.046, pad=0.06)
    # A title, because a rotated axis label overlaps the tick labels at this size.
    colour_bar.ax.set_title("mV", color=_LABEL, fontsize=7, pad=6)
    colour_bar.ax.tick_params(colors=_LABEL, labelsize=7)
    colour_bar.outline.set_edgecolor(_EDGE)

    # Spike markers above the traces, because at a spike a trace shows only the reset.
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
        # Marker size shrinks as the spike count grows, so small and large rasters stay legible.
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
