import spikecorec
network = spikecorec.square_torus(8)
engine = spikecorec.SpikeEngine(network, shape=[8, 8], rank=4)
engine.set_input_neurons([0, 1, 2])
engine.step_simulation([2.0, 2.0, 2.0], tick=0)
print(engine.neuron_count, engine.is_alive())
engine.shutdown()
