"""One GLIF1 cell under a current step."""

import math

import spikecorec as spc

from video_utils import *


def single_cell_demo():
    simulation_seconds = 0.5
    video_frame_stride = 5

    model = Path(__file__).resolve().parent / "models" / "LEMS_single_cell.xml"

    # No adjacency: one cell has nothing to connect to, so the engine takes the model
    # and nothing else.
    engine = spc.SpikeEngine(str(model))

    engine.record_membrane_video(demo_path("single_cell_membrane.spire"),
                                 video_frame_stride)
    engine.run()
    engine.write_recordings()
    engine.write_spike_file(demo_path("single_cell_spikes.dat"))

    # The model drives the cell at 2.5x rheobase, so its interspike interval is the 5 ms
    # refractory period plus tau * ln(50/30) = 5.1 ms. Checking the measurement against
    # the closed form is what makes this a test of the engine and not just a picture.
    spike_times, _ = engine.spike_times
    measured_interval = float(spike_times[1] - spike_times[0]) if spike_times.size > 1 else 0.0
    analytic_interval = 0.005 + 0.010 * math.log(50.0 / 30.0)

    print("spikes                  : %d" % spike_times.size)
    print("mean firing rate        : %.2f Hz" % engine.mean_firing_rate_hertz())
    print("measured interspike gap : %.4f s" % measured_interval)
    print("analytic interspike gap : %.4f s" % analytic_interval)
    print("relative error          : %.3f %%"
          % (100.0 * abs(measured_interval - analytic_interval) / analytic_interval))
    print("final membrane potential: %.2f mV"
          % (1000.0 * engine.read_state_variable(0, "v")))

    engine.shutdown()

    render_membrane_video(demo_path("single_cell_membrane.spire"),
                          spikes_path=demo_path("single_cell_spikes.dat"),
                          duration=simulation_seconds,
                          trace_neurons=[(0, "the cell")],
                          title="GLIF1 under a 500 pA step")


if __name__ == "__main__":
    single_cell_demo()
