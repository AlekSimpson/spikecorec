// ── Example: driving a network with literal 0/1 spike arrays ────────────────────────────────────
//
// The other examples drive their networks from NeuroML: a `pulseGenerator` bound by `explicitInput`,
// precomputed on the host into a continuous current window. This one uses the third driving
// mechanism — a literal, host-provided bit sequence, one value per tick per input neuron, with no
// NeuroML generator ComponentType behind it at all:
//
//     [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, ...]
//
// That is `DiscreteSpikeInputSchedule` (nml/discrete_spike_input.h). It is deliberately thin — a
// struct plus one method — because `AssembledModel::step_tick` already reads `network_inputs` as an
// ordinary writable array every tick, so injecting a bit-driven current needs no engine change.
//
// ── The three ways to drive a model, for comparison ─────────────────────────────────────────────
//   pulseGenerator + explicitInput   declared in NeuroML, host-precomputed into a continuous window
//                                    → glif3_torus_network_example
//   on-device generator              lowered to real device code, evaluated per tick on the GPU
//                                    → poisson_population_example
//   discrete spike array             THIS FILE — no NeuroML surface, no ComponentType, just bits
//
// Use the last when the input comes from outside the model entirely: recorded data, an encoder
// upstream, a dataset, another simulation.
//
// ── How a bit becomes a spike ───────────────────────────────────────────────────────────────────
// Each set bit injects `--amplitude` amperes for exactly ONE tick. `network_inputs` is drained every
// tick, so bits never accumulate — the default amplitude (60 nA) is calibrated so one bit drives a
// rested GLIF3 cell past threshold on its own. One bit in, one spike out.
//
// A bit can fail to produce a spike in two distinct ways, and the run counts them separately rather
// than lumping them together, because they call for opposite fixes:
//
//   refractory     the bit arrived while the cell was still inside its 5 ms refractory period.
//                  Fix by slowing the input down (`--bit-ticks`).
//   sub-threshold  the cell was free to fire, but the impulse did not reach threshold. GLIF3's
//                  after-spike currents hold `v` roughly 10-15 mV BELOW rest for a while after each
//                  spike, so a bit that would fire a rested cell may not fire a recently-fired one.
//                  Fix by raising `--amplitude`.
//
// The example makes this concrete by feeding the SAME pattern to two neurons at two rates: one
// slower than the refractory period (every bit gets through, exactly 1:1) and one faster (roughly
// half are lost, split across both causes).
//
// Each element of the pattern occupies a slot of `--bit-ticks` ticks, and a `1` injects on the first
// tick of its slot. So `--pattern 1010 --bit-ticks 100` means "spike, wait 10ms, nothing, wait 10ms,
// spike, …" — the slot width is how you set the input's timescale.
//
// ── Lateral coupling is OFF by default ──────────────────────────────────────────────────────────
// `--weight` defaults to 0 here, unlike the other torus examples, so each driven neuron's output
// depends ONLY on its own bit array and the input→output relationship is unambiguous. Pass
// `--weight 2.5e-8` to switch the torus coupling back on; the driven neurons then also fire from
// their neighbours, and the one-bit-one-spike correspondence no longer holds.
//
// Run:  ./build/examples/discrete_spike_input_example [--pattern 0001001010] [--bit-ticks 100]
//                                                     [--amplitude 6e-8] [--weight 2.5e-8]

// See glif3_torus_network_example.cpp — the backend header must precede anything that pulls a
// `String` alias into global scope, or it collides with metal-cpp's own `NS::String`.
#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <iostream>

#include "spikecorec/nml/discrete_spike_input.h"

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

namespace {

// One named input stream: a bit pattern, a slot width, and the neuron it drives.
struct InputStream {
    String name;
    s32 target_neuron_index = 0;
    s64 slot_ticks = 100;      // ticks each pattern element occupies
    Vector<u8> pattern_bits;   // the caller's array, tiled across the run
    Vector<u8> expanded_bits;  // per-tick: 1 only on the first tick of a set element's slot
};

// Parses a bit string like "0001001010". Commas, spaces and brackets are ignored, so
// "[0,0,0,1,0,0,1,0,1,0]" parses identically.
Vector<u8> parse_bit_pattern(const String &text) {
    Vector<u8> bits;
    for (char character : text) {
        if (character == '0') bits.push_back(0);
        else if (character == '1') bits.push_back(1);
        else if (character == ',' || character == ' ' || character == '[' || character == ']') continue;
        else throw std::runtime_error(String("--pattern accepts only 0s and 1s, found '") + character + "'");
    }
    if (bits.empty()) throw std::runtime_error("--pattern is empty");
    return bits;
}

// Expands `pattern_bits` across `tick_count`: element `k` owns ticks [k*slot, (k+1)*slot), and a set
// element injects on the FIRST tick of its slot only — one impulse per set bit, not a held current.
// The pattern tiles once it runs out.
Vector<u8> expand_pattern(const Vector<u8> &pattern_bits, s64 slot_ticks, s64 tick_count) {
    Vector<u8> expanded((usize)tick_count, 0);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick % slot_ticks != 0) continue;
        const s64 element_index = (tick / slot_ticks) % (s64)pattern_bits.size();
        expanded[(usize)tick] = pattern_bits[(usize)element_index];
    }
    return expanded;
}

// Renders a per-tick bit array as a raster row. One column covers several ticks; a column shows '|'
// if any tick in its window is set, so a sparse pattern never vanishes.
void print_tick_raster(const String &label, const Vector<u8> &bits_by_tick, s64 tick_count,
                       s32 column_count = 86) {
    const s32 columns = (s32)std::min<s64>(column_count, std::max<s64>(tick_count, 1));
    Vector<char> row((usize)columns, ' ');
    for (s64 tick = 0; tick < (s64)bits_by_tick.size() && tick < tick_count; ++tick) {
        if (bits_by_tick[(usize)tick] == 0) continue;
        row[(usize)std::min<s64>(tick * columns / std::max<s64>(tick_count, 1), columns - 1)] = '|';
    }
    std::cout << "    " << std::left << std::setw(14) << label << std::right << "│";
    for (char cell : row) std::cout << cell;
    std::cout << "\n";
}

// Same shape, from a list of spike ticks, so input and output rasters line up column for column.
void print_spike_raster_row(const String &label, const Vector<s64> &spike_ticks, s64 tick_count,
                            s32 column_count = 86) {
    Vector<u8> bits_by_tick((usize)tick_count, 0);
    for (s64 spike_tick : spike_ticks) {
        if (spike_tick >= 0 && spike_tick < tick_count) bits_by_tick[(usize)spike_tick] = 1;
    }
    print_tick_raster(label, bits_by_tick, tick_count, column_count);
}

// Why each set bit did or did not produce a spike.
struct BitOutcomeTally {
    s64 set_bit_count = 0;
    s64 produced_a_spike = 0;
    s64 blocked_by_refractory = 0; // the cell was still refractory from an earlier spike
    s64 subthreshold = 0;          // the cell was free to fire but the impulse did not reach threshold
};

// Classifies every set bit rather than assuming a single cause. A bit counts as having produced a
// spike if one appears within `delivery_window_ticks` of it (the engine's own >=1-tick latency means
// the spike need not land on the exact same tick).
BitOutcomeTally classify_bit_outcomes(
    const Vector<u8> &expanded_bits, const Vector<s64> &spike_ticks, s64 refractory_ticks,
    s64 delivery_window_ticks = 2
) {
    BitOutcomeTally tally;
    for (s64 tick = 0; tick < (s64)expanded_bits.size(); ++tick) {
        if (expanded_bits[(usize)tick] == 0) continue;
        ++tally.set_bit_count;

        bool spike_followed = false;
        s64 most_recent_earlier_spike = -1;
        for (s64 spike_tick : spike_ticks) {
            if (spike_tick >= tick && spike_tick <= tick + delivery_window_ticks) spike_followed = true;
            if (spike_tick < tick) most_recent_earlier_spike = spike_tick;
        }

        if (spike_followed) ++tally.produced_a_spike;
        else if (most_recent_earlier_spike >= 0 && tick - most_recent_earlier_spike <= refractory_ticks) {
            ++tally.blocked_by_refractory;
        } else ++tally.subthreshold;
    }
    return tally;
}

} // namespace

int main(int argument_count, char **argument_values) {
    // Reuses the shared torus flags, but note the weight default: 0, so lateral coupling is off and
    // each driven neuron's output depends only on its own bits (see this file's header comment).
    TorusExampleOptions options = parse_torus_example_options(
        argument_count, argument_values, TorusExampleOptions{{4000, 1e-4f, false, false}, 8, 0.0f});

    String pattern_text = "0001001010";
    s64 slow_slot_ticks = 100; // 10ms per element — comfortably longer than the 5ms refractory period
    f32 bit_amplitude_amperes = 6.0e-8f;
    for (int argument_index = 1; argument_index < argument_count; ++argument_index) {
        String argument = argument_values[argument_index];
        if (argument_index + 1 >= argument_count) continue;
        if (argument == "--pattern") pattern_text = argument_values[argument_index + 1];
        else if (argument == "--bit-ticks") slow_slot_ticks = std::strtoll(argument_values[argument_index + 1], nullptr, 10);
        else if (argument == "--amplitude") bit_amplitude_amperes = std::strtof(argument_values[argument_index + 1], nullptr);
    }
    const Vector<u8> pattern_bits = parse_bit_pattern(pattern_text);
    // Deliberately shorter than the 5ms (50-tick) refractory period, so bits get swallowed.
    const s64 fast_slot_ticks = 20;

    GpuContextScope gpu_context_scope;

    // ── 1. A torus network with NO NeuroML stimulus at all ──────────────────────────────────────
    // `stimulated_neuron_indices` is left empty, so the generated document declares no
    // pulseGenerator and no explicitInput. Every bit of drive comes from the host array.
    TorusNetworkOptions network_options;
    network_options.variant = GlifVariant::Glif3;
    network_options.side_length = options.side_length;
    network_options.stimulated_neuron_indices = {};

    ModelSpecification model =
        load_generated_model("discrete_input_torus", generate_glif_torus_network_nml(network_options));

    Vector<IrProgram> programs = lower_type_library_to_ir(model);
    print_model_summary(model, programs);
    if (options.base.print_ir) print_ir_programs(model, programs);

    std::cout << "\n  Note `stimuli : 0` above — this model declares no pulseGenerator and no\n"
              << "  explicitInput. All drive comes from the host-provided bit arrays below.\n";

    ModelAllocation allocation = allocate_model(model, programs);
    seed_glif_initial_state(allocation, model, GlifVariant::Glif3);

    WeightMatrix weights = build_weight_matrix(model);
    weights.set_constant_weight(options.scattered_weight);

    AssembledModel assembled_model(model, programs);
    LiveModelBuffers live = make_live_model_buffers(allocation, weights, model.total_neuron_count);

    // ── 2. The same pattern at two rates ────────────────────────────────────────────────────────
    const s64 tick_count = options.base.tick_count;

    Vector<InputStream> streams;
    {
        InputStream slow;
        slow.name = "slow";
        slow.target_neuron_index = 0;
        slow.slot_ticks = slow_slot_ticks;
        slow.pattern_bits = pattern_bits;
        slow.expanded_bits = expand_pattern(pattern_bits, slow_slot_ticks, tick_count);
        streams.push_back(std::move(slow));

        InputStream fast;
        fast.name = "fast";
        fast.target_neuron_index = 42;
        fast.slot_ticks = fast_slot_ticks;
        fast.pattern_bits = pattern_bits;
        fast.expanded_bits = expand_pattern(pattern_bits, fast_slot_ticks, tick_count);
        streams.push_back(std::move(fast));
    }

    // ── 3. Building the schedule ────────────────────────────────────────────────────────────────
    // spike_bits is tick-major: spike_bits[tick][index] is the bit for target_neuron_indices[index].
    DiscreteSpikeInputSchedule discrete_schedule;
    discrete_schedule.current_amplitude_amperes = bit_amplitude_amperes;
    for (const InputStream &stream : streams) {
        discrete_schedule.target_neuron_indices.push_back(stream.target_neuron_index);
    }

    discrete_schedule.spike_bits.resize((usize)tick_count);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        Vector<u8> bits_this_tick;
        bits_this_tick.reserve(streams.size());
        for (const InputStream &stream : streams) bits_this_tick.push_back(stream.expanded_bits[(usize)tick]);
        discrete_schedule.spike_bits[(usize)tick] = std::move(bits_this_tick);
    }

    print_heading("Discrete spike input");
    std::cout << "  pattern         : [";
    for (usize bit_index = 0; bit_index < pattern_bits.size(); ++bit_index) {
        std::cout << (bit_index > 0 ? "," : "") << (int)pattern_bits[bit_index];
    }
    std::cout << "]  (" << pattern_bits.size() << " elements, tiled)\n"
              << "  bit amplitude   : " << bit_amplitude_amperes * 1e9f
              << " nA injected for exactly one tick per set bit\n"
              << "  refractory      : 5ms = 50 ticks\n"
              << "  lateral coupling: " << (options.scattered_weight == 0.0f
                                                ? "off (--weight 0) — each neuron reflects only its own bits"
                                                : "ON — driven neurons also fire from their neighbours")
              << "\n\n"
              << "    slow → neuron " << std::setw(3) << streams[0].target_neuron_index << "   slot "
              << slow_slot_ticks << " ticks (" << format_seconds((f64)slow_slot_ticks * options.base.dt_seconds)
              << " per element) — longer than the refractory period\n"
              << "    fast → neuron " << std::setw(3) << streams[1].target_neuron_index << "   slot "
              << fast_slot_ticks << " ticks (" << format_seconds((f64)fast_slot_ticks * options.base.dt_seconds)
              << " per element) — shorter than the refractory period\n";

    // ── 4. Tick loop ────────────────────────────────────────────────────────────────────────────
    // One line, exactly where a pulseGenerator window would otherwise be applied.
    print_heading("Simulating");
    std::cout << "  " << tick_count << " ticks × " << options.base.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)tick_count * options.base.dt_seconds) << "\n";

    Vector<Vector<s64>> spike_ticks_by_neuron((usize)model.total_neuron_count);
    Vector<s64> spike_count_by_neuron((usize)model.total_neuron_count, 0);

    for (s64 tick = 0; tick < tick_count; ++tick) {
        discrete_schedule.apply_to_network_inputs(live.buffers.network_inputs, tick);

        assembled_model.step_tick(live.buffers, options.base.dt_seconds, tick, tick + 1);

        for (s32 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
            if (live.buffers.last_spiked[neuron_index] != tick) continue;
            spike_ticks_by_neuron[(usize)neuron_index].push_back(tick);
            ++spike_count_by_neuron[(usize)neuron_index];
        }
    }

    // ── 5. Input versus output, aligned ─────────────────────────────────────────────────────────
    print_heading("Input bits vs. output spikes");

    const s64 refractory_ticks = (s64)std::llround(0.005 / (f64)options.base.dt_seconds); // t_ref = 5ms

    for (const InputStream &stream : streams) {
        const Vector<s64> &output_spike_ticks = spike_ticks_by_neuron[(usize)stream.target_neuron_index];
        const BitOutcomeTally tally = classify_bit_outcomes(stream.expanded_bits, output_spike_ticks, refractory_ticks);

        std::cout << "\n  \033[1m" << stream.name << "\033[0m — neuron " << stream.target_neuron_index
                  << ", one pattern element every " << stream.slot_ticks << " ticks\n";
        print_tick_raster("input bits", stream.expanded_bits, tick_count);
        print_spike_raster_row("output spikes", output_spike_ticks, tick_count);

        std::cout << "                   " << tally.set_bit_count << " bits in → " << output_spike_ticks.size()
                  << " spikes out\n";
        if (tally.produced_a_spike == tally.set_bit_count) {
            std::cout << "                   ✓ every bit produced exactly one spike\n";
        } else {
            // Two genuinely different failure modes — reported separately rather than lumped
            // together, because they call for opposite fixes (slow the input down vs. raise
            // --amplitude).
            if (tally.blocked_by_refractory > 0) {
                std::cout << "                   " << tally.blocked_by_refractory
                          << " arrived while the cell was still refractory\n";
            }
            if (tally.subthreshold > 0) {
                std::cout << "                   " << tally.subthreshold
                          << " landed on a non-refractory cell but did not reach threshold\n"
                          << "                     (GLIF3's after-spike currents hold v below rest for a\n"
                          << "                      while after each spike — raise --amplitude to overcome it)\n";
            }
        }
    }

    // ── 6. What the rest of the network did ─────────────────────────────────────────────────────
    s64 total_spike_count = 0;
    s32 neurons_that_fired = 0;
    for (s64 count : spike_count_by_neuron) {
        total_spike_count += count;
        if (count > 0) ++neurons_that_fired;
    }

    print_heading("Network response");
    std::cout << "  total spikes       " << total_spike_count << "\n"
              << "  neurons that fired " << neurons_that_fired << " / " << model.total_neuron_count << "\n\n";
    print_torus_grid("Spikes per neuron", spike_count_by_neuron, options.side_length, "spike count");

    if (options.scattered_weight == 0.0f) {
        std::cout << "\n  Only the two driven neurons fired: lateral coupling is off by default here so\n"
                  << "  the bit-to-spike mapping above is unambiguous. Re-run with --weight 2.5e-8 to\n"
                  << "  couple the torus and watch the activity spread from both input sites.\n";
    }

    return 0;
}
