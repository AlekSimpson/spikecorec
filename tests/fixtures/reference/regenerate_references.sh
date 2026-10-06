#!/usr/bin/env bash
# Regenerates the jNeuroML reference outputs the C++ tests compare the engine against.
# Needs jnml: `uv tool install pyneuroml` puts it in ~/.local/bin.
#
#   tests/fixtures/reference/regenerate_references.sh
set -euo pipefail

reference_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_directory="$(cd "${reference_directory}/../nml" && pwd)"
jnml_command="$(command -v jnml || echo "${HOME}/.local/bin/jnml")"
if [[ ! -x "${jnml_command}" ]]; then
    echo "jnml not found; install it with: uv tool install pyneuroml" >&2
    exit 1
fi

# jNeuroML writes its outputs beside the LEMS file, so it runs on a copy.
work_directory="$(mktemp -d)"
trap 'rm -rf "${work_directory}"' EXIT

# pynn_poisson_network.nml itself is written by make_poisson_network.py.
for model in pynn_synapses pynn_poisson_network; do
    cp "${model_directory}/${model}.nml" "${model_directory}/LEMS_${model}.xml" "${work_directory}/"
    (cd "${work_directory}" && "${jnml_command}" "LEMS_${model}.xml" -nogui > "${model}_jnml.log")
    for output in "${work_directory}/${model}"_*.dat; do
        cp "${output}" "${reference_directory}/"
        echo "wrote ${reference_directory}/$(basename "${output}")"
    done
done
"${jnml_command}" -v 2>/dev/null | head -1 > "${reference_directory}/jnml_version.txt" || true
