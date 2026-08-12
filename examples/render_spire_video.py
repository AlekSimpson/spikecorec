#!/usr/bin/env python3
"""render_spire_video.py — play back a `.spire` recording as an mp4/gif.

Completes the visualization deferral `include/spikecorec/core/recording.h`'s own
`read_spire_recording` doc comment states explicitly: "Ports the file-loading half of
`play_spire_recording` (frame decoding only -- visualization stays out of scope)." This
script is that visualization step.

Reads one or two `.spire` recordings written by `SimulationRecorder`, which SpikeEngine drives
automatically for every `<OutputFile>`/`<EventOutputFile>` a model declares -- a spike-raster
stream (required, a 0.0/1.0 mask per neuron per tick, what an `<EventOutputFile>` produces)
and, optionally, a membrane-potential stream (same shape, what an `<OutputFile>` produces) --
and renders an animated grid: each neuron placed at its own `(row, column)` position on a
`side x side` torus (the SAME `row * side_length + column` flat-index convention every torus
example in this directory generates its network with, see examples/glif_torus_network.h's own
header comment), membrane potential as a per-tick heatmap and spikes overlaid as bright markers
on the tick they occur.

For a non-torus recording (no natural 2D grid -- e.g. a small hand-built network), pass
`--side 1` to fall back to a single-row layout instead of guessing a square grid.

Large recordings are read lazily rather than materialized whole, which a big run makes
impossible (a 1000x1000 torus recorded for 15000 ticks is 56 GiB per stream, so the spike +
membrane pair together would need 112 GiB). This script imports only numpy and matplotlib --
it never imports `spikecorec`, so the Python extension does not need to have been built:

  * An *uncompressed* `.spire` file is a 4-byte big-endian neuron_count followed by fixed-size
    native-float32 frames (see recording.h's SpireEncoder), so frame N sits at a computable
    offset. Such a recording is memory-mapped and only the strided ticks actually drawn are ever
    touched.
  * A *compressed* recording (`.spire.gz`/`.xz`/`.bz2`) has no fixed per-frame offset -- reaching
    frame N means decoding everything before it -- so it is walked sequentially instead, one frame
    in memory at a time (StreamingSpireReader). Every tick is decoded, but only the strided ones
    are drawn. This costs one extra pass up front, since a compressed stream's frame count is not
    derivable from its file size the way an uncompressed one's is.

Usage:
    ./examples/render_spire_video.py recordings/glif3_torus_spikes.spire \\
        --side 8 --membrane recordings/glif3_torus_membrane.spire

    ./examples/render_spire_video.py recordings/glif3_torus_spikes.spire \\
        --side 8 --output glif3_torus.gif

Dependencies (not part of spikecorec's own core C++/CUDA/Metal build):
    numpy, matplotlib. Install with:

        pip install numpy matplotlib

    mp4 output additionally needs an `ffmpeg` binary on PATH (matplotlib's `FFMpegWriter`); with
    no `ffmpeg` available, pass `--output foo.gif` instead (Pillow, a matplotlib dependency
    already, writes gif directly -- no extra install).

    The `spikecorec` extension module is NOT needed: this script reads the `.spire` byte format
    itself.
"""

import argparse
import os
import struct
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
        "--ticks-per-second", dest="ticks_per_second_path", default=None,
        help="path to a plain-text ticks-per-second telemetry file (one float per simulated tick). "
             "By default this is auto-detected next to SPIKE_RECORDING (its own path with "
             "'_spikes.spire' replaced by '_ticks_per_second.txt', the file every recording example "
             "writes automatically) and silently skipped if that file does not exist; pass this "
             "explicitly to point at a differently-named file, or 'none' to disable the lookup "
             "entirely.")
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
    parser.add_argument(
        "--record-stride", type=int, default=1,
        help="how many simulated ticks each RECORDED frame advanced, i.e. whatever --record-stride "
             "the example itself ran with (default: 1, every tick recorded). Only affects the "
             "on-screen readout: frame N of the recording is labelled tick N*record_stride and "
             "timestamped N*record_stride*--dt, so a strided recording reports its real sim tick "
             "instead of its frame index. Pass the example's own --dt unchanged alongside this "
             "rather than pre-multiplying it.")
    parser.add_argument(
        "--cmap", default="cividis",
        help="matplotlib colormap for the membrane-potential heatmap (default: cividis -- a "
             "dark-blue-to-yellow scale with no red/orange in it, so red spike markers stay "
             "visually distinct at every membrane-potential level, unlike viridis's own bright "
             "yellow high end or plasma/inferno/magma's red-orange range)")
    parser.add_argument(
        "--spike-size", type=float, default=None,
        help="spike marker area in points^2 (matplotlib scatter 's'). Defaults to a size scaled "
             "down as the grid gets denser (large tori would otherwise render spikes as "
             "overlapping blobs bigger than a single grid cell) -- pass this to override.")
    parser.add_argument(
        "--spike-color", default="red",
        help="matplotlib color for the spike markers (default: red, which reads clearly against the "
             "default cividis heatmap). A colormap containing red or orange -- seismic, inferno, "
             "plasma -- hides red markers wherever the membrane sits in that band, so pass a hue "
             "the chosen --cmap does not contain (e.g. lime, whose nearest color in each of those "
             "three maps is ~0.95 away in RGB, against red's 0.006 in seismic).")
    parser.add_argument(
        "--no-spikes", action="store_true",
        help="skip the spike-marker overlay entirely, rendering only the membrane-potential "
             "heatmap (only meaningful together with --membrane).")
    return parser.parse_args(argument_list)


def default_spike_marker_size(side_length):
    # Tuned so a marker's on-screen diameter stays a modest fraction of one grid cell's own width
    # at this script's fixed figsize=(6, 6) -- without this, a fixed marker size that looks right
    # on an 8-wide torus renders as oversized, overlapping blobs on a 71-wide one (grid cells
    # shrink roughly as 1/side_length, but a fixed scatter size does not).
    return max(6.0, 2500.0 / (side_length ** 1.5))


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


# Every compressed variant TorusExampleOptions::record_extension can produce (recording.h's
# SimulationRecorder infers the codec from exactly these suffixes) -- longest first, so
# ".spire.gz" matches before a naive os.path.splitext would strip only ".gz" and leave ".spire"
# dangling on the end of a derived base name.
_SPIRE_SUFFIXES = (".spire.gzip", ".spire.gz", ".spire.lzma", ".spire.xz", ".spire.bz2", ".spire")


def strip_spire_suffix(path):
    for suffix in _SPIRE_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    base, _ = os.path.splitext(path)
    return base


def memory_map_spire_frames(recording_path):
    """Lazily map an uncompressed `.spire` recording as a (frame_count, neuron_count) float32 view.

    Mirrors SpireDecoder's own read side (recording.h): a 4-byte big-endian neuron_count header,
    then back-to-back frames of `neuron_count` native-order float32 values. Because every frame is
    the same size, frame `tick` starts at a computable offset and numpy can map the file instead of
    reading it -- indexing the result pages in only the 4 * neuron_count bytes of that one frame.
    """
    with open(recording_path, "rb") as recording_file:
        header_bytes = recording_file.read(4)
    if len(header_bytes) != 4:
        raise ValueError(
            f"'{recording_path}' is too short to hold a .spire header (got {len(header_bytes)} "
            f"bytes, expected at least 4)")
    neuron_count = struct.unpack(">I", header_bytes)[0]
    if neuron_count == 0:
        raise ValueError(f"'{recording_path}' declares a neuron_count of 0")

    payload_byte_count = os.path.getsize(recording_path) - 4
    frame_byte_count = neuron_count * 4
    frame_count = payload_byte_count // frame_byte_count
    if frame_count == 0:
        raise ValueError(
            f"'{recording_path}' declares {neuron_count} neurons ({frame_byte_count} bytes per "
            f"frame) but holds only {payload_byte_count} payload bytes -- no complete frame")

    # A partial trailing frame means the writer was interrupted mid-frame; drop it and say so
    # rather than reshaping garbage, matching play_spire_recording's own "Truncated recording"
    # handling (recording.h).
    trailing_byte_count = payload_byte_count % frame_byte_count
    if trailing_byte_count != 0:
        print(f"[render_spire_video] warning: '{recording_path}' ends with a partial frame "
              f"({trailing_byte_count} trailing bytes) -- ignoring it and rendering the "
              f"{frame_count} complete frames")

    return np.memmap(
        recording_path, dtype=np.float32, mode="r", offset=4, shape=(frame_count, neuron_count))


# The compression suffixes resolve_spire_compression (recording.h) itself infers a codec from --
# the same set SpireSink writes, so a recording named by any of them decodes with the matching
# stdlib module (gzip/lzma/bz2 all read the standard containers zlib/liblzma/libbz2 produce).
_COMPRESSED_SUFFIXES = (".gz", ".gzip", ".xz", ".lzma", ".bz2")


def is_compressed_spire_path(recording_path):
    return recording_path.lower().endswith(_COMPRESSED_SUFFIXES)


def open_spire_byte_stream(recording_path):
    """Open `recording_path` as a binary stream, transparently decompressing if it is compressed."""
    lowered_path = recording_path.lower()
    if lowered_path.endswith((".gz", ".gzip")):
        import gzip
        return gzip.open(recording_path, "rb")
    if lowered_path.endswith((".xz", ".lzma")):
        import lzma
        return lzma.open(recording_path, "rb")  # FORMAT_AUTO -- reads .xz and legacy .lzma alike
    if lowered_path.endswith(".bz2"):
        import bz2
        return bz2.open(recording_path, "rb")
    return open(recording_path, "rb")


class StreamingSpireReader:
    """Reads a `.spire` recording one frame at a time, in order, in bounded memory.

    A compressed recording cannot be memory-mapped: its frames have no fixed file offset to seek
    to, so the only way to reach frame N is to decode frames 0..N-1 on the way there. This walks
    the stream sequentially instead, holding exactly one frame at a time -- which is what makes a
    56 GiB-decompressed recording renderable at all, and without a built extension module.

    Works on an uncompressed `.spire` too, so a mixed pair (one stream compressed, one not) needs
    no special casing; memory_map_spire_frames stays the faster path for random access.
    """

    def __init__(self, recording_path):
        self.recording_path = recording_path
        self._stream = open_spire_byte_stream(recording_path)
        header_bytes = self._stream.read(4)
        if len(header_bytes) != 4:
            self._stream.close()
            raise ValueError(
                f"'{recording_path}' is too short to hold a .spire header (got {len(header_bytes)} "
                f"bytes, expected at least 4)")
        self.neuron_count = struct.unpack(">I", header_bytes)[0]
        if self.neuron_count == 0:
            self._stream.close()
            raise ValueError(f"'{recording_path}' declares a neuron_count of 0")
        self._frame_byte_count = self.neuron_count * 4

    def read_frame(self):
        """Return the next frame as a read-only float32 view, or None once the stream is spent.

        A short final read means the writer was interrupted mid-frame -- reported and treated as
        end-of-stream rather than reshaped into garbage, matching memory_map_spire_frames's own
        partial-trailing-frame handling.
        """
        payload_bytes = self._stream.read(self._frame_byte_count)
        if not payload_bytes:
            return None
        if len(payload_bytes) < self._frame_byte_count:
            print(f"[render_spire_video] warning: '{self.recording_path}' ends with a partial frame "
                  f"({len(payload_bytes)} of {self._frame_byte_count} bytes) -- treating it as "
                  f"end-of-stream")
            return None
        return np.frombuffer(payload_bytes, dtype=np.float32)

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()
        return False


def scan_streamed_recording(recording_path, track_value_range):
    """One sequential pass over `recording_path`: count its frames, optionally its min/max too.

    A compressed recording's frame count is not derivable from its file size the way an
    uncompressed one's is (and gzip's own ISIZE trailer only stores the uncompressed length modulo
    2^32, which wraps for a stream this large), so counting means decoding. Since that pass has to
    touch every frame anyway, it collects the membrane color-scale range at the same time rather
    than paying for a second pass.
    """
    frame_count = 0
    minimum_value = np.inf
    maximum_value = -np.inf
    with StreamingSpireReader(recording_path) as reader:
        while True:
            frame = reader.read_frame()
            if frame is None:
                break
            frame_count += 1
            if track_value_range:
                minimum_value = min(minimum_value, float(np.nanmin(frame)))
                maximum_value = max(maximum_value, float(np.nanmax(frame)))
    return frame_count, minimum_value, maximum_value


def resolve_membrane_value_range(minimum_value, maximum_value):
    """Clamp a scanned membrane range into something imshow can use as vmin/vmax."""
    if not np.isfinite(minimum_value) or not np.isfinite(maximum_value):
        # Every scanned frame was all-NaN -- fall back to a unit scale so imshow still renders.
        return 0.0, 1.0
    if maximum_value - minimum_value < 1e-12:
        return minimum_value, minimum_value + 1e-12
    return minimum_value, maximum_value


def make_memory_mapped_frame_supplier(spike_frames, membrane_frames, frame_count, stride):
    """Build a fresh-generator factory yielding (tick, spike_frame, membrane_frame) from memmaps."""
    def frame_supplier():
        for tick in range(0, frame_count, stride):
            membrane_frame = membrane_frames[tick] if membrane_frames is not None else None
            yield tick, spike_frames[tick], membrane_frame
    return frame_supplier


def make_streamed_frame_supplier(
    spike_recording_path, membrane_recording_path, frame_count, stride
):
    """Build a fresh-generator factory that walks both recordings sequentially, in lockstep.

    Every tick is decoded (there is no way to skip ahead in a compressed stream) but only the
    strided ones are yielded, so the caller still renders `frame_count / stride` frames while
    memory stays at one frame per stream.
    """
    def frame_supplier():
        spike_reader = StreamingSpireReader(spike_recording_path)
        membrane_reader = (
            StreamingSpireReader(membrane_recording_path)
            if membrane_recording_path is not None else None)
        try:
            for tick in range(frame_count):
                spike_frame = spike_reader.read_frame()
                if spike_frame is None:
                    raise ValueError(
                        f"'{spike_recording_path}' ended after {tick} frames, expected {frame_count}")
                membrane_frame = None
                if membrane_reader is not None:
                    membrane_frame = membrane_reader.read_frame()
                    if membrane_frame is None:
                        raise ValueError(
                            f"'{membrane_recording_path}' ended after {tick} frames, but "
                            f"'{spike_recording_path}' still has frames -- the two recordings do "
                            f"not cover the same run")
                if tick % stride == 0:
                    yield tick, spike_frame, membrane_frame
        finally:
            spike_reader.close()
            if membrane_reader is not None:
                membrane_reader.close()
    return frame_supplier


def default_output_path(spike_recording_path):
    return strip_spire_suffix(spike_recording_path) + ".mp4"


def default_ticks_per_second_path(spike_recording_path):
    # An optional wall-clock telemetry file, "<same dir>/<base_name>_ticks_per_second.txt" next
    # to "<base_name>_spikes.spire[.gz|.xz|.bz2]". No example in this directory writes one at
    # present, so the overlay is normally skipped; the reader stays because a caller timing its
    # own run can drop the file next to its recording and get the overlay for free. The
    # telemetry file is always plain text, uncompressed, regardless of what --compress the
    # spike/membrane recordings themselves used.
    directory, filename = os.path.split(spike_recording_path)
    base = strip_spire_suffix(filename)
    base_name = base[: -len("_spikes")] if base.endswith("_spikes") else base
    return os.path.join(directory, base_name + "_ticks_per_second.txt")


def choose_stride(frame_count, requested_stride, max_frames):
    if requested_stride is not None:
        return max(requested_stride, 1)
    if frame_count <= max_frames:
        return 1
    return -(-frame_count // max_frames)  # ceiling division


def render_spire_video(
    frame_supplier, rendered_frame_count, neuron_count, side_length, output_path, fps, dt_seconds,
    membrane_value_range=None, cmap="cividis", spike_size=None, spike_color="red",
    show_spikes=True, ticks_per_second_values=None, record_stride=1
):
    """Render frames pulled from `frame_supplier` to `output_path`.

    `frame_supplier` is a zero-argument factory returning a FRESH generator of
    (tick, spike_frame, membrane_frame) tuples -- a factory rather than a plain generator because
    FuncAnimation calls new_frame_seq() more than once (_init_draw consumes one frame before save()
    walks the sequence), and a streamed generator cannot be rewound. `membrane_value_range` is the
    (vmin, vmax) for the heatmap, or None to render spikes against a blank grid.
    """
    # Local import: matplotlib is only needed once we actually render, so a caller hitting an
    # earlier validation error (e.g. a non-square neuron_count) never pays its import cost.
    import matplotlib
    matplotlib.use("Agg")  # headless -- this script only ever writes a file, never shows a window
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    row_count = -(-neuron_count // side_length)  # ceiling division -- covers a non-square total

    def frame_to_grid(values_by_neuron):
        grid = np.full((row_count * side_length,), np.nan, dtype=np.float32)
        grid[:neuron_count] = values_by_neuron
        return grid.reshape(row_count, side_length)

    # constrained_layout (rather than a fixed subplot rect) is what keeps the colorbar's own
    # "membrane potential (V)" label from being clipped off the right edge of the rendered
    # frame -- FuncAnimation never calls tight_layout() per frame, so a fixed-rect figure only
    # ever gets the margins right for whatever was on-screen at figure-creation time.
    figure, axes = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axes.set_xticks([])
    axes.set_yticks([])

    if membrane_value_range is not None:
        # Range computed by the caller, never by a whole-array np.nanmin/np.nanmax here: the
        # membrane source may be a lazy memmap or a sequential decode of a file far larger than RAM
        # (56 GiB for a 1000x1000 torus over 15000 ticks), and a whole-array reduction would pull
        # every byte of it through memory.
        membrane_minimum, membrane_maximum = membrane_value_range
        heatmap = axes.imshow(
            frame_to_grid(np.zeros(neuron_count, dtype=np.float32)), cmap=cmap,
            vmin=membrane_minimum, vmax=membrane_maximum, interpolation="nearest")
        colorbar = figure.colorbar(heatmap, ax=axes, label="membrane potential (V)", fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=9)
        colorbar.set_label("membrane potential (V)", fontsize=9)
    else:
        heatmap = axes.imshow(
            frame_to_grid(np.zeros(neuron_count, dtype=np.float32)), cmap="gray", vmin=0.0,
            vmax=1.0, interpolation="nearest")

    # Spikes overlaid as a separate scatter layer so they pop visually against the membrane-
    # potential heatmap (or a blank grid) underneath, rather than being folded into the same
    # color scale as the continuous trace. A thin black edge keeps a marker readable as a distinct
    # dot rather than blending into a similarly-lit heatmap cell underneath it.
    neuron_rows, neuron_columns = np.divmod(np.arange(neuron_count), side_length)
    resolved_spike_size = spike_size if spike_size is not None else default_spike_marker_size(side_length)
    spike_scatter = axes.scatter(
        [], [], s=resolved_spike_size, c=spike_color, marker="o",
        linewidths=0.6 if show_spikes else 0, edgecolors="black" if show_spikes else "none")

    title = axes.set_title("", fontsize=10)

    def update(frame_payload):
        recording_frame_index, spike_frame, membrane_frame = frame_payload
        # The recording's own frame index is not the simulated tick when the run was recorded with a
        # stride -- frame N advanced N*record_stride ticks of dynamics. Telemetry stays indexed by
        # frame (TicksPerSecondTelemetry writes one line per RECORDED frame, not per tick).
        tick = recording_frame_index * record_stride
        if membrane_value_range is not None and membrane_frame is not None:
            heatmap.set_data(frame_to_grid(membrane_frame))

        spiking_neuron_mask = spike_frame > 0.5
        if show_spikes:
            spike_scatter.set_offsets(
                np.column_stack((neuron_columns[spiking_neuron_mask], neuron_rows[spiking_neuron_mask])))
        spike_count_this_tick = int(spiking_neuron_mask.sum())

        line_one = f"tick {tick}   t={tick * dt_seconds * 1000.0:.2f}ms   neurons: {neuron_count}"
        line_two = f"spikes this tick: {spike_count_this_tick}"
        if ticks_per_second_values is not None:
            line_two += f"   sim speed: {ticks_per_second_values[recording_frame_index]:,.0f} ticks/s"
        title.set_text(line_one + "\n" + line_two)
        return (heatmap, spike_scatter, title) if membrane_value_range is not None else (spike_scatter, title)

    # `frames=frame_supplier` (a callable) rather than a generator object, plus
    # cache_frame_data=False: FuncAnimation tees/caches a plain iterable to support repeat, which
    # would retain every 4 MB frame it has seen and undo the whole point of streaming. A callable is
    # re-invoked for a fresh sequence instead, and repeat=False keeps it from being wrapped anyway.
    animation_object = animation.FuncAnimation(
        figure, update, frames=frame_supplier, save_count=rendered_frame_count,
        cache_frame_data=False, repeat=False, interval=1000.0 / fps, blit=False)

    _, output_extension = os.path.splitext(output_path)
    if output_extension.lower() == ".gif":
        animation_object.save(output_path, writer=animation.PillowWriter(fps=fps))
    else:
        animation_object.save(output_path, writer=animation.FFMpegWriter(fps=fps))

    plt.close(figure)


def main(argument_list=None):
    arguments = parse_arguments(sys.argv[1:] if argument_list is None else argument_list)

    # Either recording being compressed forces the sequential path for BOTH (StreamingSpireReader
    # handles an uncompressed stream too, so a mixed pair needs no special case). Neither path
    # imports `spikecorec`: materializing an entire recording in RAM is exactly what is
    # impossible at this scale.
    use_streaming = (
        is_compressed_spire_path(arguments.spike_recording_path)
        or (arguments.membrane_recording_path is not None
            and is_compressed_spire_path(arguments.membrane_recording_path)))

    spike_frames = None
    membrane_frames = None
    membrane_value_range = None

    if use_streaming:
        with StreamingSpireReader(arguments.spike_recording_path) as spike_reader:
            neuron_count = spike_reader.neuron_count
        if arguments.membrane_recording_path is not None:
            with StreamingSpireReader(arguments.membrane_recording_path) as membrane_reader:
                if membrane_reader.neuron_count != neuron_count:
                    raise ValueError(
                        f"--membrane recording declares {membrane_reader.neuron_count} neurons, "
                        f"but SPIKE_RECORDING declares {neuron_count}")

        # The counting pass. Run against the membrane recording when there is one so the same decode
        # yields the color-scale range too; the spike recording's own frame count is then checked
        # against it while streaming (make_streamed_frame_supplier raises if they disagree).
        scanned_path = arguments.membrane_recording_path or arguments.spike_recording_path
        print(f"[render_spire_video] compressed recording -- scanning '{scanned_path}' sequentially "
              f"to count frames"
              f"{' and measure the membrane range' if arguments.membrane_recording_path else ''}")
        frame_count, scanned_minimum, scanned_maximum = scan_streamed_recording(
            scanned_path, arguments.membrane_recording_path is not None)
        if frame_count == 0:
            raise ValueError(f"'{scanned_path}' holds no complete frames")
        if arguments.membrane_recording_path is not None:
            membrane_value_range = resolve_membrane_value_range(scanned_minimum, scanned_maximum)
    else:
        spike_frames = memory_map_spire_frames(arguments.spike_recording_path)
        if arguments.membrane_recording_path is not None:
            membrane_frames = memory_map_spire_frames(arguments.membrane_recording_path)
            if membrane_frames.shape != spike_frames.shape:
                raise ValueError(
                    f"--membrane recording shape {membrane_frames.shape} does not match "
                    f"SPIKE_RECORDING's own shape {spike_frames.shape} (frame_count, neuron_count)")
        frame_count, neuron_count = spike_frames.shape

    ticks_per_second_values = None
    ticks_per_second_path = arguments.ticks_per_second_path
    if ticks_per_second_path is not None and ticks_per_second_path.lower() == "none":
        ticks_per_second_path = None  # explicit opt-out -- skip auto-detection too
    elif ticks_per_second_path is None:
        # Not passed explicitly -- auto-detect the file every recording example writes next to its
        # own spike recording, and silently render without the overlay if it isn't there (an older
        # recording, or one from a non-recording example that never had this telemetry).
        guessed_path = default_ticks_per_second_path(arguments.spike_recording_path)
        ticks_per_second_path = guessed_path if os.path.exists(guessed_path) else None

    if ticks_per_second_path is not None:
        ticks_per_second_values = np.loadtxt(ticks_per_second_path)
        if ticks_per_second_values.shape != (frame_count,):
            raise ValueError(
                f"ticks-per-second file '{ticks_per_second_path}' has {ticks_per_second_values.shape} "
                f"entries, expected exactly {frame_count} (one per tick, matching SPIKE_RECORDING's "
                f"own frame count)")
    side_length = infer_grid_side_length(neuron_count, arguments.side)
    output_path = arguments.output or default_output_path(arguments.spike_recording_path)
    stride = choose_stride(frame_count, arguments.stride, arguments.max_frames)

    rendered_frame_count = -(-frame_count // stride)  # ceiling division
    print(f"[render_spire_video] {neuron_count} neurons, {frame_count} ticks, "
          f"{side_length}-wide grid, rendering every {stride} tick(s) "
          f"({rendered_frame_count} frames) -> {output_path}")
    print(f"[render_spire_video] frame source: "
          f"{'sequential streaming decode' if use_streaming else 'memory-mapped random access'}")
    print(f"[render_spire_video] sim-speed telemetry: "
          f"{ticks_per_second_path if ticks_per_second_values is not None else 'none found, skipping overlay'}")

    if use_streaming:
        frame_supplier = make_streamed_frame_supplier(
            arguments.spike_recording_path, arguments.membrane_recording_path, frame_count, stride)
    else:
        if membrane_frames is not None:
            # Scoped to the drawn ticks only, so the memmap path never faults in the whole file --
            # the streaming path measures every frame instead, since its counting pass decodes them
            # all regardless. With any realistic stride the two ranges agree to the pixel.
            scanned_minimum = np.inf
            scanned_maximum = -np.inf
            for scanned_tick in range(0, frame_count, stride):
                scanned_frame = membrane_frames[scanned_tick]
                scanned_minimum = min(scanned_minimum, float(np.nanmin(scanned_frame)))
                scanned_maximum = max(scanned_maximum, float(np.nanmax(scanned_frame)))
            membrane_value_range = resolve_membrane_value_range(scanned_minimum, scanned_maximum)
        frame_supplier = make_memory_mapped_frame_supplier(
            spike_frames, membrane_frames, frame_count, stride)

    render_spire_video(
        frame_supplier, rendered_frame_count, neuron_count, side_length, output_path, arguments.fps,
        arguments.dt, membrane_value_range=membrane_value_range, cmap=arguments.cmap,
        spike_size=arguments.spike_size, spike_color=arguments.spike_color,
        show_spikes=not arguments.no_spikes, ticks_per_second_values=ticks_per_second_values,
        record_stride=max(arguments.record_stride, 1))

    print(f"[render_spire_video] wrote {output_path}")


if __name__ == "__main__":
    main()
