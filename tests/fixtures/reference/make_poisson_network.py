"""Writes tests/fixtures/nml/pynn_poisson_network.nml: 100 Poisson sources at 20 Hz driving 50 IF_curr_exp
cells through expCurrSynapse and 50 IF_curr_alpha cells through alphaCurrSynapse, each cell hearing 25
sources chosen at random. Deterministic: rerunning it writes the same file.

    python3 tests/fixtures/reference/make_poisson_network.py
"""
import random
from pathlib import Path

SOURCE_COUNT = 100
TARGET_COUNT = 50
SOURCES_PER_TARGET = 25
EXP_WEIGHT = 0.3
ALPHA_WEIGHT = 0.11   # an alpha arrival carries e times the charge of an exp one of the same weight

generator = random.Random(20261006)
lines = [
    '<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="PynnPoissonNetwork">',
    '  <!-- Written by tests/fixtures/reference/make_poisson_network.py; do not edit by hand. -->',
    '  <expCurrSynapse id="expSynapse" tau_syn="5"/>',
    '  <alphaCurrSynapse id="alphaSynapse" tau_syn="5"/>',
    '  <IF_curr_exp id="expCell" cm="1" i_offset="0" tau_m="20" tau_refrac="5" v_init="-65" v_reset="-70"'
    ' v_rest="-65" v_thresh="-50" tau_syn_E="5" tau_syn_I="5"/>',
    '  <IF_curr_alpha id="alphaCell" cm="1" i_offset="0" tau_m="20" tau_refrac="5" v_init="-65" v_reset="-70"'
    ' v_rest="-65" v_thresh="-50" tau_syn_E="5" tau_syn_I="5"/>',
    '  <spikeGeneratorPoisson id="poissonSource" averageRate="20 Hz"/>',
    '  <network id="poissonNetwork">',
    f'    <population id="sources" component="poissonSource" size="{SOURCE_COUNT}"/>',
    f'    <population id="expTargets" component="expCell" size="{TARGET_COUNT}"/>',
    f'    <population id="alphaTargets" component="alphaCell" size="{TARGET_COUNT}"/>',
]
for population, synapse, weight in [("expTargets", "expSynapse", EXP_WEIGHT), ("alphaTargets", "alphaSynapse", ALPHA_WEIGHT)]:
    lines.append(f'    <projection id="{population}Projection" presynapticPopulation="sources" '
                 f'postsynapticPopulation="{population}" synapse="{synapse}">')
    connection_id = 0
    for target in range(TARGET_COUNT):
        for source in sorted(generator.sample(range(SOURCE_COUNT), SOURCES_PER_TARGET)):
            lines.append(f'      <connectionWD id="{connection_id}" preCellId="../sources[{source}]" '
                         f'postCellId="../{population}[{target}]" weight="{weight}" delay="2 ms"/>')
            connection_id += 1
    lines.append('    </projection>')
lines += ['  </network>', '</neuroml>', '']
output = Path(__file__).resolve().parent.parent / "nml" / "pynn_poisson_network.nml"
output.write_text("\n".join(lines))
print(f"wrote {output}")
