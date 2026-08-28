import os
import tempfile
import spikecorec as spc

from video_utils import *


def demo_function():
    side_length = 10
    video_frame_stride = 5

    model = Path("/neuroml/file/path")
    topology = spc.square_torus(side_length)
    engine = spc.SpikeEngine(model, topology, "synapseTypeName", 
                             connection_weight=1.0, 
                             connection_delay_seconds=1e-3)
    engine.record_membrane_video(demo_path("example.spire"), video_frame_stride)
    engine.run()
    engine.write_recordings()
    engine.write_spike_file(demo_path("spikes.dat"))
    engine.shutdown()






