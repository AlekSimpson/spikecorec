# Vendored NeuroML2/LEMS version pin (ticket #3 [A1])

These files are plain vendored copies (not a git submodule) of the upstream NeuroML2 standard
library and schema. Pinned versions, read directly from the vendored files themselves:

- **NeuroML2 XML schema**: v2.3 (`schema/NeuroML_v2.3.xsd`, `targetNamespace`
  `http://www.neuroml.org/schema/neuroml2`)
- **LEMS schema**: v0.7.6 (`std_lib/NeuroML2CoreTypes.xml`'s own namespace declaration,
  `http://www.neuroml.org/lems/0.7.6`)
- **std_lib bundle**: `Cells.xml`, `Channels.xml`, `Inputs.xml`, `Networks.xml`,
  `NeuroMLCoreCompTypes.xml`, `NeuroMLCoreDimensions.xml`, `PyNN.xml`, `Simulation.xml`,
  `Synapses.xml`, aggregated by `NeuroML2CoreTypes.xml`'s own `<Include>` directives.

Upgrading this bundle means replacing every file under `schema/` and `std_lib/` with a newer
upstream NeuroML2/LEMS release and updating the versions recorded above — the parser/resolver
(`src/nml/nml.cpp`, `src/nml/resolve.cpp`) are written against the NeuroML v2.3 / LEMS v0.7.6
element and attribute surface, not any specific vendoring date.
