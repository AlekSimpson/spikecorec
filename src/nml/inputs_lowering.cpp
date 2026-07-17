#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/nml/inputs_lowering.h"

#include "spikecorec/nml/expression_lowering.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// Reads a required resolved constant off `entry.baked_constants` -- mirrors
// stimulus_schedule.cpp's own `required_baked_constant` helper exactly, just returning the
// already-formatted `.tick`/`.alloc` literal text (expression_lowering.h's `format_literal`)
// instead of the raw f64, since every caller here only ever uses it to build a
// `ParamConstantDirective`.
String required_baked_literal(const TypeLibraryEntry &entry, const String &name) {
    auto value = entry.baked_constants.find(name);
    if (value == entry.baked_constants.end()) {
        log::throw_runtime_error(log::logger(),
            "inputs_lowering: '" + entry.bound_instance_id + "' (" + entry.component_type_name +
            ") has no resolved '" + name + "' value");
    }
    return format_literal(value->second);
}

// `weight` (a `Property`, NeuroML-documented `defaultValue="1"`) -- Property default values are not
// yet applied at resolve time (arch §3.1), so an absent `weight` defaults to 1.0 here, matching
// stimulus_schedule.cpp's own identical precedent for `pulseGenerator`'s `weight`.
String baked_literal_or_default(const TypeLibraryEntry &entry, const String &name, f64 fallback_value) {
    auto value = entry.baked_constants.find(name);
    return format_literal(value == entry.baked_constants.end() ? fallback_value : value->second);
}

// `t0 = tick * dt` -- the current absolute simulated time, computed once per tick. None of the
// four supported generators is given absolute time by the engine directly (only `dt`/`tick` are
// reserved, ir_spec.md §3.1), so every one of them needs this as its very first `@integrate`
// instruction.
BinaryInstruction current_time_instruction() { return BinaryInstruction{BinaryOpcode::Mul, "t0", "tick", "dt"}; }

// The recurring "before delay / within [delay, delay+duration) / after" three-way split every
// deterministic generator below uses (`sineGenerator`/`rampGenerator`/`voltageClamp`'s own
// `OnCondition` triplet, collapsed into one equivalent `if/elif/else` -- see this file's header
// comment on why an if/elif/else replaces LEMS's independent-and-overwriting OnConditions here).
// Appends the two boolean instructions computing `before_start_name`/`within_window_name` into
// `output`; `elif_upper_bound_is_inclusive` selects `<=` (voltageClamp's own third OnCondition uses
// `.gt.` on the upper bound, i.e. the window itself is inclusive of the boundary) vs `<`
// (sineGenerator/rampGenerator's own `.lt.` upper bound).
void append_delay_window_conditions(Vector<TickInstruction> &output, bool elif_upper_bound_is_inclusive) {
    output.push_back(current_time_instruction());
    output.push_back(BinaryInstruction{BinaryOpcode::Lt, "before_start", "t0", "delay"});
    output.push_back(BinaryInstruction{BinaryOpcode::Add, "t1", "delay", "duration"});
    output.push_back(BinaryInstruction{
        elif_upper_bound_is_inclusive ? BinaryOpcode::Le : BinaryOpcode::Lt, "within_window", "t0", "t1"});
}

// ── sineGenerator ────────────────────────────────────────────────────────────────────────────────
//
// i = 0                                                     , t <  delay
// i = weight * amplitude * sin(phase + 2*pi*(t-delay)/period), delay <= t < delay+duration
// i = 0                                                     , t >= delay+duration
IrProgram build_sine_generator_program(const TypeLibraryEntry &entry) {
    IrProgram program;
    program.component_type_name = entry.component_type_name;
    program.alloc = {
        StateDirective{"i", "f32", std::nullopt},
        ParamConstantDirective{"weight", baked_literal_or_default(entry, "weight", 1.0)},
        ParamConstantDirective{"phase", required_baked_literal(entry, "phase")},
        ParamConstantDirective{"delay", required_baked_literal(entry, "delay")},
        ParamConstantDirective{"duration", required_baked_literal(entry, "duration")},
        ParamConstantDirective{"amplitude", required_baked_literal(entry, "amplitude")},
        ParamConstantDirective{"period", required_baked_literal(entry, "period")},
        ExposeDirective{"i"},
    };

    Vector<TickInstruction> body;
    append_delay_window_conditions(body, /*elif_upper_bound_is_inclusive=*/false);

    IfInstruction dispatch;
    dispatch.condition = "before_start";
    dispatch.then_body = {MoveInstruction{"i", "0"}};
    ElseIfBranch window_branch;
    window_branch.condition = "within_window";
    window_branch.body = {
        BinaryInstruction{BinaryOpcode::Sub, "t2", "t0", "delay"},
        BinaryInstruction{BinaryOpcode::Mul, "t2", "t2", "6.283185307179586"}, // 2*pi
        BinaryInstruction{BinaryOpcode::Div, "t2", "t2", "period"},
        BinaryInstruction{BinaryOpcode::Add, "t2", "phase", "t2"},
        UnaryInstruction{UnaryOpcode::Sin, "t2", "t2"},
        BinaryInstruction{BinaryOpcode::Mul, "t2", "amplitude", "t2"},
        BinaryInstruction{BinaryOpcode::Mul, "i", "weight", "t2"},
    };
    dispatch.else_if_branches = {window_branch};
    dispatch.else_body = Vector<TickInstruction>{MoveInstruction{"i", "0"}};
    body.push_back(dispatch);
    body.push_back(BinaryInstruction{BinaryOpcode::Add, "network_inputs", "network_inputs", "i"});

    program.tick.integrate = body;
    return program;
}

// ── rampGenerator ────────────────────────────────────────────────────────────────────────────────
//
// i = weight * baselineAmplitude                                                     , t <  delay
// i = weight * (startAmplitude + (finishAmplitude-startAmplitude)*(t-delay)/duration) , in-window
// i = weight * baselineAmplitude                                                     , t >= delay+duration
IrProgram build_ramp_generator_program(const TypeLibraryEntry &entry) {
    IrProgram program;
    program.component_type_name = entry.component_type_name;
    program.alloc = {
        StateDirective{"i", "f32", std::nullopt},
        ParamConstantDirective{"weight", baked_literal_or_default(entry, "weight", 1.0)},
        ParamConstantDirective{"delay", required_baked_literal(entry, "delay")},
        ParamConstantDirective{"duration", required_baked_literal(entry, "duration")},
        ParamConstantDirective{"startAmplitude", required_baked_literal(entry, "startAmplitude")},
        ParamConstantDirective{"finishAmplitude", required_baked_literal(entry, "finishAmplitude")},
        ParamConstantDirective{"baselineAmplitude", required_baked_literal(entry, "baselineAmplitude")},
        ExposeDirective{"i"},
    };

    Vector<TickInstruction> body;
    append_delay_window_conditions(body, /*elif_upper_bound_is_inclusive=*/false);

    IfInstruction dispatch;
    dispatch.condition = "before_start";
    dispatch.then_body = {BinaryInstruction{BinaryOpcode::Mul, "i", "weight", "baselineAmplitude"}};
    ElseIfBranch window_branch;
    window_branch.condition = "within_window";
    window_branch.body = {
        BinaryInstruction{BinaryOpcode::Sub, "t2", "finishAmplitude", "startAmplitude"},
        BinaryInstruction{BinaryOpcode::Sub, "t3", "t0", "delay"},
        BinaryInstruction{BinaryOpcode::Mul, "t2", "t2", "t3"},
        BinaryInstruction{BinaryOpcode::Div, "t2", "t2", "duration"},
        BinaryInstruction{BinaryOpcode::Add, "t2", "startAmplitude", "t2"},
        BinaryInstruction{BinaryOpcode::Mul, "i", "weight", "t2"},
    };
    dispatch.else_if_branches = {window_branch};
    dispatch.else_body = Vector<TickInstruction>{BinaryInstruction{BinaryOpcode::Mul, "i", "weight", "baselineAmplitude"}};
    body.push_back(dispatch);
    body.push_back(BinaryInstruction{BinaryOpcode::Add, "network_inputs", "network_inputs", "i"});

    program.tick.integrate = body;
    return program;
}

// ── voltageClamp ─────────────────────────────────────────────────────────────────────────────────
//
// i = 0                                            , t <  delay
// i = weight * (targetVoltage - v) / simpleSeriesResistance , delay <= t <= delay+duration
// i = 0                                            , t >  delay+duration
//
// See this file's header comment for the KNOWN LIMITATION: `require v` is not resolvable by
// master_kernel.cpp's current per-neuron dispatch, so this lowering is exercised at the
// IR/GPU-source level only, not through AssembledModel.
IrProgram build_voltage_clamp_program(const TypeLibraryEntry &entry) {
    IrProgram program;
    program.component_type_name = entry.component_type_name;
    program.alloc = {
        StateDirective{"i", "f32", std::nullopt},
        RequireDirective{"v", "postsynaptic"},
        ParamConstantDirective{"weight", baked_literal_or_default(entry, "weight", 1.0)},
        ParamConstantDirective{"delay", required_baked_literal(entry, "delay")},
        ParamConstantDirective{"duration", required_baked_literal(entry, "duration")},
        ParamConstantDirective{"targetVoltage", required_baked_literal(entry, "targetVoltage")},
        ParamConstantDirective{"simpleSeriesResistance", required_baked_literal(entry, "simpleSeriesResistance")},
        ExposeDirective{"i"},
    };

    Vector<TickInstruction> body;
    append_delay_window_conditions(body, /*elif_upper_bound_is_inclusive=*/true);

    IfInstruction dispatch;
    dispatch.condition = "before_start";
    dispatch.then_body = {MoveInstruction{"i", "0"}};
    ElseIfBranch window_branch;
    window_branch.condition = "within_window";
    window_branch.body = {
        BinaryInstruction{BinaryOpcode::Sub, "t2", "targetVoltage", "v"},
        BinaryInstruction{BinaryOpcode::Div, "t2", "t2", "simpleSeriesResistance"},
        BinaryInstruction{BinaryOpcode::Mul, "i", "weight", "t2"},
    };
    dispatch.else_if_branches = {window_branch};
    dispatch.else_body = Vector<TickInstruction>{MoveInstruction{"i", "0"}};
    body.push_back(dispatch);
    body.push_back(BinaryInstruction{BinaryOpcode::Add, "network_inputs", "network_inputs", "i"});

    program.tick.integrate = body;
    return program;
}

// ── SpikeSourcePoisson (PyNN.xml) ───────────────────────────────────────────────────────────────
//
// A genuine on-device Poisson spike source (ir_spec.md §3.3/§6's own "rand ... covers ...
// SpikeSourcePoisson"). See this file's header comment for the documented simplifications
// (two state variables instead of four; a plain boolean AND window instead of the original's
// `H()`-based trick). Firing rule, exactly the original's own ISI recurrence:
//   due  = (t > tnext) AND (t >= start) AND (t <= start+duration)
//   on due: emit spike; tnext <- tnext + (-log(max(rand(),1e-7)) / rate); tsince <- 0
// (`max(rand(),1e-7)` before `log` matches gpu_source.cpp's own rng preamble's identical guard
// against `log(0)` -- xorshift32's uniform draw is exactly 0.0 on a vanishingly rare but nonzero
// fraction of steps.)
IrProgram build_poisson_spike_source_program(const TypeLibraryEntry &entry) {
    IrProgram program;
    program.component_type_name = entry.component_type_name;
    program.alloc = {
        StateDirective{"tsince", "f32", std::nullopt},
        StateDirective{"tnext", "f32", std::nullopt},
        ParamConstantDirective{"start", required_baked_literal(entry, "start")},
        ParamConstantDirective{"duration", required_baked_literal(entry, "duration")},
        ParamConstantDirective{"rate", required_baked_literal(entry, "rate")},
        ExposeDirective{"tsince"},
    };

    program.tick.integrate = {BinaryInstruction{BinaryOpcode::Add, "tsince", "tsince", "dt"}};

    program.tick.detect = {
        current_time_instruction(),
        BinaryInstruction{BinaryOpcode::Gt, "due", "t0", "tnext"},
        BinaryInstruction{BinaryOpcode::Ge, "after_start", "t0", "start"},
        BinaryInstruction{BinaryOpcode::And, "due", "due", "after_start"},
        BinaryInstruction{BinaryOpcode::Add, "t1", "start", "duration"},
        BinaryInstruction{BinaryOpcode::Le, "within_duration", "t0", "t1"},
        BinaryInstruction{BinaryOpcode::And, "due", "due", "within_duration"},
    };

    program.tick.emit = {IfInstruction{"due", {EmitInstruction{"spike"}}, {}, std::nullopt}};

    Vector<TickInstruction> reset_body = {
        RandomInstruction{RandomOpcode::Rand, "t2"},
        BinaryInstruction{BinaryOpcode::Max, "t2", "t2", "0.0000001"},
        UnaryInstruction{UnaryOpcode::Log, "t2", "t2"},
        UnaryInstruction{UnaryOpcode::Neg, "t2", "t2"},
        BinaryInstruction{BinaryOpcode::Div, "t2", "t2", "rate"},
        BinaryInstruction{BinaryOpcode::Add, "tnext", "tnext", "t2"},
        MoveInstruction{"tsince", "0"},
    };
    program.tick.reset = {IfInstruction{"due", reset_body, {}, std::nullopt}};

    return program;
}

} // namespace

IrProgram lower_inputs_to_ir(const TypeLibraryEntry &inputs_entry) {
    if (inputs_entry.category != TypeLibraryCategory::Inputs) {
        log::throw_runtime_error(log::logger(),
            "inputs_lowering: '" + inputs_entry.component_type_name + "' is not an Inputs-category TypeLibraryEntry");
    }

    if (inputs_entry.component_type_name == "sineGenerator") return build_sine_generator_program(inputs_entry);
    if (inputs_entry.component_type_name == "rampGenerator") return build_ramp_generator_program(inputs_entry);
    if (inputs_entry.component_type_name == "voltageClamp") return build_voltage_clamp_program(inputs_entry);
    if (inputs_entry.component_type_name == "SpikeSourcePoisson") return build_poisson_spike_source_program(inputs_entry);

    log::throw_runtime_error(log::logger(),
        "inputs_lowering: on-device lowering for Inputs ComponentType '" + inputs_entry.component_type_name +
        "' is not implemented (ticket #65 [F4] only lowers sineGenerator/rampGenerator/voltageClamp/"
        "SpikeSourcePoisson -- poissonFiringSynapse/compoundInput need generic child-component "
        "composition, a construct docs/nml_ir_spec.md §6 defers to Phase 3, so they are out of this "
        "ticket's scope; see inputs_lowering.h's own header comment)");
}

Vector<IrProgram> lower_all_inputs_to_ir(const ModelSpecification &model) {
    Vector<IrProgram> programs;
    for (const auto &entry : model.type_library) {
        if (entry.category != TypeLibraryCategory::Inputs) continue;
        programs.push_back(lower_inputs_to_ir(entry));
    }
    return programs;
}

} // namespace spikecorec::nml
