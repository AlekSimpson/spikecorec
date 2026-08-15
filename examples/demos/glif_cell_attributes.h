#pragma once

// The GLIF family's electrical parameters, in one place because every demo uses the same
// ones and the comparisons between demos only mean something if they do.
//
// C = 100 pF and gL = 10 nS give a membrane time constant of 10 ms and a rheobase of
// gL * (vth - EL) = 10 nS * 20 mV = 200 pA. A 5 ms refractory period caps any cell at
// 200 Hz. GLIF4 and GLIF5 declare no vth at all: their threshold is the state variable
// theta, which starts at thetaInf and climbs with every spike.

#include <stdexcept>
#include <string>

#include "spikecorec/core/types.h"

namespace spikecorec::demos {

// The ComponentType name, as declared in tests/fixtures/nml/glif_cell_types.nml.
inline String glif_cell_type_name(s32 glif_index) {
    return "GLIF" + std::to_string(glif_index) + "Cell";
}

// `refractory_period` is overridable because it sets how long a cell stays unavailable
// after firing, and on a recurrent sheet that decides whether activity travels as fronts or
// simply saturates: a front has to leave the neighbourhood before the cells behind it
// recover, or it re-enters and every cell just fires at its ceiling.
inline String glif_cell_attributes(s32 glif_index,
                                   const String &refractory_period = "5ms") {
    const String shared = String(R"( C="100pF" gL="10nS" EL="-70mV" vreset="-70mV" t_ref=")")
                          + refractory_period + R"(")";

    switch (glif_index) {
        case 1: return shared + R"( vth="-50mV")";
        case 2: return shared + R"( vth="-50mV" resetScale="0.3")";
        case 3: return shared + R"( vth="-50mV" tauAsc1="100ms" tauAsc2="10ms")"
                                R"( ascAdd1="-60pA" ascAdd2="-120pA")";
        case 4: return shared + R"( thetaInf="-50mV" tauTheta="50ms" thetaSpikeAdd="3mV")";
        case 5: return shared + R"( thetaInf="-50mV" tauTheta="50ms" thetaSpikeAdd="3mV")"
                                R"( tauAsc1="100ms" tauAsc2="10ms")"
                                R"( ascAdd1="-60pA" ascAdd2="-120pA")";
        default:
            throw std::runtime_error("glif_cell_attributes: GLIF index must be 1..5, got " +
                                     std::to_string(glif_index));
    }
}

// What each type adds, for the demo to print. Worth stating because two of them behave
// identically under gentle drive and that surprises people.
inline String glif_description(s32 glif_index) {
    switch (glif_index) {
        case 1: return "leaky integrate-and-fire with a refractory period";
        case 2: return "GLIF1 plus a voltage-dependent reset, vreset + resetScale*(v - vth). "
                       "Under gentle drive the threshold overshoot is a fraction of a "
                       "millivolt, so this fires identically to GLIF1";
        case 3: return "GLIF1 plus two after-spike currents that decay and are bumped at "
                       "every spike, which suppresses sustained firing";
        case 4: return "GLIF1 plus an adapting threshold theta that relaxes to thetaInf and "
                       "jumps on every spike";
        case 5: return "both the after-spike currents and the adapting threshold";
        default: return "";
    }
}

// Where the ComponentTypes themselves live. None of GLIF1-5 are in the NeuroML standard
// library, so every model includes this file by absolute path.
inline String glif_cell_types_path() {
    return "tests/fixtures/nml/glif_cell_types.nml";
}

} // namespace spikecorec::demos
