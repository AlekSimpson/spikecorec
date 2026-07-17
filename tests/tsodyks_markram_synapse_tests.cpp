#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/synapse_lowering.h"
#include "spikecorec/nml/gpu_source.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Tsodyks-Markram short-term plasticity: "already realized by D2/D3 + B3" (ticket #66 [F5]) ──
//
// The ticket's own framing: an STP synapse's facilitation/depression state (u/x in the standard
// Tsodyks-Markram formalism) is "just ordinary per-edge state" -- already fully handled by the
// existing per-edge shared-basis WeightMatrix machinery (#52/#53) and synapse lowering (#51), no
// new IR/allocator/WeightMatrix code needed. This file documents exactly how far that's true, with
// two real, load-bearing findings from actually tracing the code (not read from this comment alone
// -- every claim below is backed by one of the tests further down):
//
// 1. The real vendored short-term-plasticity ComponentTypes are `tsodyksMarkramDepMechanism`
//    (depression-only) and `tsodyksMarkramDepFacMechanism` (full dep+fac, Tsodyks/Pawelzik/Markram
//    1998) in third_party/neuroml2/std_lib/Synapses.xml -- both extend `basePlasticityMechanism`,
//    NOT `baseSynapse`, and are normally composed into a real `blockingPlasticSynapse` as a
//    `<Children>` sub-component via a `select`/`reduce="multiply"` DerivedVariable
//    (`plasticityFactor`). synapse_lowering.h/.cpp ALREADY documents that composition shape
//    ("Children-based sub-mechanism composition... out of Phase-1 scope") as unsupported by
//    `lower_synapse_to_ir` -- so the real vendored ComponentTypes, composed the way the std-lib
//    actually uses them, cannot go through the existing pipeline verbatim. This is a pre-existing,
//    already-documented #51 scope boundary, not a new finding of this ticket.
//
// 2. The genuinely new finding this ticket's investigation turned up: EVEN taken in isolation
//    (ignoring the Children/select/reduce issue), the Tsodyks-Markram OnEvent update itself is
//    unsupported by TWO independent, already-merged layers:
//      - `tsodyksMarkramDepMechanism`'s OnEvent is `R = R * (1 - U)` (`U` a fixed release-probability
//        constant) and `tsodyksMarkramDepFacMechanism`'s is `R = R * (1 - U)` / `U = U + initReleaseProb
//        * (1 - U)` (`U` itself a state variable) -- EVERY real TM OnEvent update is multiplicative in
//        the assigned variable's own CURRENT value. `synapse_lowering.cpp`'s `lower_on_events` only
//        accepts a "purely additive self-increment" StateAssignment (`var = var + <increment>`), but
//        that restriction is purely SYNTACTIC (checked by AST shape, not by what `<increment>` reads)
//        -- it does not, by itself, reject a self-referencing increment like `R + (0 - R * U)`.
//      - The REAL restriction lives one layer downstream, in `gpu_source.cpp` (ticket #55):
//        its operand scan unconditionally fails ("peredge quantity '<name>' used as a plain operand
//        -- only loadedge/accedge may reference a peredge variable") on ANY bare reference to a
//        `peredge` variable outside `loadedge`/`accedge`, and `lower_on_events` (#51) never emits a
//        `LoadEdgeInstruction` for an onevent body -- so an
//        OnEvent increment that reads ANY peredge variable's current value (itself or another) always
//        fails, just one call deeper than the front-end's own check.
//    `TsodyksMarkramGap.*` below proves both halves of this precisely (front-end accepts it, IR->GPU
//    source rejects it with the specific message), rather than asserting it from reading alone.
//
// Given (2), this ticket's own "no new machinery" instruction, and CLAUDE.md's directive to flag a
// real gap rather than silently work around it: `TsodyksMarkram.*` below instead demonstrates the
// part of the claim that IS true today, with no changes to synapse_lowering.cpp/gpu_source.cpp/
// allocator.cpp/weight_matrix.*: a per-edge synapse with TWO Tsodyks-Markram-named state variables
// (`R`, `U`), using the REAL vendored TimeDerivative relaxation equations verbatim
// (`(1-R)/tauRec`, `(initReleaseProb-U)/tauFac`) -- fully supported, no approximation -- plus a
// spike-triggered ADDITIVE (not multiplicative) bump for each, an honest, explicitly-labeled
// substitute for the real OnEvent jump that (2) shows is not lowerable today. The additive bump is
// still real short-term depression/facilitation (a per-edge value that steps down/up at each
// presynaptic spike and relaxes back toward baseline between spikes) -- just not literally
// `R_new = R_old * (1 - U)`.

namespace {

bool approx(f32 first, f32 second, f32 epsilon = 1e-3f) {
    return std::fabs(first - second) <= epsilon * (1.0f + std::fabs(second));
}

f32 decay_toward_target(f32 old_value, f32 target, f32 tau, f32 dt) {
    return target + (old_value - target) * std::exp(-dt / tau);
}

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

const TypeLibraryEntry &type_library_entry_for(const ModelSpecification &specification, const String &bound_instance_id) {
    for (const auto &entry : specification.type_library) {
        if (entry.bound_instance_id == bound_instance_id) return entry;
    }
    throw std::runtime_error("no type library entry for '" + bound_instance_id + "'");
}

const String DUMMY_CELL_COMPONENT_TYPE =
    "  <ComponentType name=\"DummyCell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <TimeDerivative variable=\"v\" value=\"network_inputs / C\"/>"
    "    </Dynamics>"
    "  </ComponentType>";

// Mirrors synapse_lowering_tests.cpp's own build_synapse_type_library_entry (kept as its own copy
// here, matching how every *_tests.cpp file in this tree keeps its own fixture code self-contained).
TypeLibraryEntry build_synapse_type_library_entry(
    const String &fixture_id, const String &synapse_component_type_xml,
    const String &synapse_tag, const String &synapse_instance_attributes
) {
    write_temp_file("spikecorec_tsodyks_markram_" + fixture_id + "_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TsodyksMarkram" + fixture_id + "Content\">"
        + DUMMY_CELL_COMPONENT_TYPE +
        "  <DummyCell id=\"dummyCellInstance\" C=\"1.0e-10\"/>"
        + synapse_component_type_xml +
        "  <" + synapse_tag + " id=\"synapseInstance\" " + synapse_instance_attributes + "/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"dummyCellInstance\" size=\"2\"/>"
        "    <projection id=\"Proj\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" synapse=\"synapseInstance\">"
        "      <connection id=\"0\" preCellId=\"Pop/0/dummyCellInstance\" postCellId=\"Pop/1/dummyCellInstance\"/>"
        "    </projection>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_tsodyks_markram_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TsodyksMarkram" + fixture_id + "Top\">"
        "  <include href=\"spikecorec_tsodyks_markram_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    ModelSpecification specification = build_model_specification(resolved);
    return type_library_entry_for(specification, "synapseInstance");
}

// The achievable, honest substitute (see this file's own header comment): real vendored TM
// TimeDerivative equations for R/U verbatim, additive-only spike-triggered bump.
const String TEST_TSODYKS_MARKRAM_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"TestTsodyksMarkramSynapse\" extends=\"baseConductanceBasedSynapse\">"
    "    <Parameter name=\"initReleaseProb\" dimension=\"none\"/>"
    "    <Parameter name=\"tauRec\" dimension=\"time\"/>"
    "    <Parameter name=\"tauFac\" dimension=\"time\"/>"
    "    <Parameter name=\"depressionDelta\" dimension=\"none\"/>"
    "    <Parameter name=\"facilitationDelta\" dimension=\"none\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"R\" dimension=\"none\"/>"
    "      <StateVariable name=\"U\" dimension=\"none\"/>"
    "      <DerivedVariable name=\"g\" dimension=\"conductance\" exposure=\"g\" value=\"gbase * R * U\"/>"
    "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g * (erev - v)\"/>"
    "      <TimeDerivative variable=\"R\" value=\"(1 - R) / tauRec\"/>"
    "      <TimeDerivative variable=\"U\" value=\"(initReleaseProb - U) / tauFac\"/>"
    "      <OnEvent port=\"in\">"
    "        <StateAssignment variable=\"R\" value=\"R + depressionDelta\"/>"
    "        <StateAssignment variable=\"U\" value=\"U + facilitationDelta\"/>"
    "      </OnEvent>"
    "    </Dynamics>"
    "  </ComponentType>";

// Reproduces the REAL vendored OnEvent semantics exactly (R's update multiplies by (1-U), U itself
// a peredge state variable) -- used ONLY to prove where/how this is rejected, not expected to lower
// successfully end to end.
const String TEST_BROKEN_TSODYKS_MARKRAM_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"TestBrokenTsodyksMarkramSynapse\" extends=\"baseConductanceBasedSynapse\">"
    "    <Dynamics>"
    "      <StateVariable name=\"R\" dimension=\"none\"/>"
    "      <StateVariable name=\"U\" dimension=\"none\"/>"
    "      <DerivedVariable name=\"g\" dimension=\"conductance\" exposure=\"g\" value=\"gbase * R\"/>"
    "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g * (erev - v)\"/>"
    "      <OnEvent port=\"in\">"
    "        <StateAssignment variable=\"R\" value=\"R + (0 - R * U)\"/>"
    "      </OnEvent>"
    "    </Dynamics>"
    "  </ComponentType>";

// ── minimal, test-local dispatch-argument builder (mirrors master_kernel.cpp's own
// DispatchArgumentBuilder -- kept as its own copy here, matching this tree's established
// per-file-self-contained-fixture convention) ────────────────────────────────────────────────────
class DispatchArgumentBuilder {
public:
    void add_pointer(const void *pointer) { add_bytes(&pointer, sizeof(void *)); }
    void add_f32(f32 value) { add_bytes(&value, sizeof(f32)); }
    void add_s64(s64 value) { add_bytes(&value, sizeof(s64)); }
    void add_s32(s32 value) { add_bytes(&value, sizeof(s32)); }

    void dispatch(KernelHandle handle, LaunchConfig config) const {
        Vector<const void *> args(slots_.size());
        Vector<usize> sizes(slots_.size());
        for (usize index = 0; index < slots_.size(); ++index) {
            args[index] = slots_[index].data();
            sizes[index] = slots_[index].size();
        }
#ifdef SPIKECOREC_METAL
        metal_dispatch(handle, config, args.data(), sizes.data(), (u32)slots_.size());
#elif defined(SPIKECOREC_CUDA)
        cuda_dispatch(handle, config, args.data(), sizes.data(), (u32)slots_.size());
#endif
    }

private:
    void add_bytes(const void *value, usize size) {
        Vector<u8> storage(size);
        std::memcpy(storage.data(), value, size);
        slots_.push_back(std::move(storage));
    }

    Vector<Vector<u8>> slots_;
};

// Generic by-name resolver for whichever of TestTsodyksMarkramSynapse's two generated functions
// (`_tick`, `_deliver_in`) is being dispatched -- driven off GpuFunctionSignature's own
// parameter_names_in_order (ticket #55's documented calling convention), so this test does not need
// to hand-derive/guess the exact parameter order gpu_source.cpp emits.
struct DispatchContext {
    f32 dt_value = 0.0f;
    f32 *network_inputs = nullptr;
    f32 *v_buffer = nullptr;
    f32 *g_buffer = nullptr;
    f32 *i_buffer = nullptr;
    WeightMatrix *weights = nullptr;
    s64 matrix_index_r = -1;
    s64 matrix_index_u = -1;
    s64 neuron_count = 0;
    s32 *source_node_indices = nullptr;
    s32 *target_node_indices = nullptr;
    s32 *edge_slot_indices = nullptr;
    s64 event_count = 0;
};

void append_argument(DispatchArgumentBuilder &builder, const String &name, const DispatchContext &context) {
    if (name == "dt") { builder.add_f32(context.dt_value); return; }
    if (name == "network_inputs") { builder.add_pointer(context.network_inputs); return; }
    if (name == "v") { builder.add_pointer(context.v_buffer); return; }
    if (name == "g") { builder.add_pointer(context.g_buffer); return; }
    if (name == "i") { builder.add_pointer(context.i_buffer); return; }
    if (name == "internal_node_words") { builder.add_pointer(context.weights->k2tree.internal_node_words.get_contents()); return; }
    if (name == "leaf_node_words") { builder.add_pointer(context.weights->k2tree.leaf_node_words.get_contents()); return; }
    if (name == "rank_superblock_table") { builder.add_pointer(context.weights->k2tree.rank_superblock_table.get_contents()); return; }
    if (name == "rank_subblock_table") { builder.add_pointer(context.weights->k2tree.rank_subblock_table.get_contents()); return; }
    if (name == "branching_factor") { builder.add_s32(context.weights->k2tree.branching_factor); return; }
    if (name == "superblock_size_words") { builder.add_s32(context.weights->k2tree.superblock_size_words); return; }
    if (name == "padded_node_count") { builder.add_s32(context.weights->k2tree.padded_node_count); return; }
    if (name == "tree_height") { builder.add_s32(context.weights->k2tree.tree_height); return; }
    if (name == "internal_bit_count") { builder.add_s32(context.weights->k2tree.internal_bit_count); return; }
    if (name == "node_count") { builder.add_s64(context.weights->node_count); return; }
    if (name == "max_neighbor_count") { builder.add_s64(context.weights->max_neighbor_count); return; }
    if (name == "rank_float4_stride") { builder.add_s64(context.weights->rank_float4_stride); return; }
    if (name == "U") { builder.add_pointer(context.weights->U_matrix.get_contents()); return; }
    if (name == "V") { builder.add_pointer(context.weights->V_matrix.get_contents()); return; }
    if (name == "coefficients_R") { builder.add_pointer(context.weights->coefficient_vectors[context.matrix_index_r].get_contents()); return; }
    if (name == "sparse_delta_R") { builder.add_pointer(context.weights->sparse_delta_buffers[context.matrix_index_r].get_contents()); return; }
    if (name == "coefficients_U") { builder.add_pointer(context.weights->coefficient_vectors[context.matrix_index_u].get_contents()); return; }
    if (name == "sparse_delta_U") { builder.add_pointer(context.weights->sparse_delta_buffers[context.matrix_index_u].get_contents()); return; }
    if (name == "neuron_count") { builder.add_s64(context.neuron_count); return; }
    if (name == "source_node_indices") { builder.add_pointer(context.source_node_indices); return; }
    if (name == "target_node_indices") { builder.add_pointer(context.target_node_indices); return; }
    if (name == "edge_slot_indices") { builder.add_pointer(context.edge_slot_indices); return; }
    if (name == "event_count") { builder.add_s64(context.event_count); return; }
    throw std::runtime_error("tsodyks_markram_synapse_tests: unhandled dispatch parameter '" + name + "'");
}

const String &source_text_for_this_backend(const GpuSource &source) {
#ifdef SPIKECOREC_CUDA
    return source.cuda_source;
#else
    return source.msl_source;
#endif
}

LaunchConfig launch_config_for(s64 element_count) {
    constexpr u32 threads_per_block = 32;
    if (element_count <= 0) return LaunchConfig{0, threads_per_block};
    u32 grid = (u32)((element_count + threads_per_block - 1) / threads_per_block);
    return LaunchConfig{grid, threads_per_block};
}

} // namespace

// ── the two-layer gap (see this file's own header comment) ──────────────────────────────────────

TEST(TsodyksMarkramGap, front_end_synapse_lowering_accepts_a_peredge_self_referencing_onevent_increment) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "gap_front_end", TEST_BROKEN_TSODYKS_MARKRAM_SYNAPSE_COMPONENT_TYPE, "TestBrokenTsodyksMarkramSynapse",
        "gbase=\"1nS\" erev=\"0mV\"");

    // synapse_lowering.cpp's own `is_additive`/`left_is_self` check is purely syntactic (top-level
    // `+` with one bare-identifier self operand) -- it does not inspect what the OTHER operand
    // reads, so `R + (0 - R * U)` passes it even though the increment reads R and U's own current
    // per-edge values. This does NOT throw here.
    IrProgram program;
    ASSERT_NO_THROW(program = lower_synapse_to_ir(entry));
    EXPECT_EQ(program.component_type_name, "TestBrokenTsodyksMarkramSynapse");
}

TEST(TsodyksMarkramGap, ir_to_gpu_source_rejects_the_same_program_with_a_specific_message) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "gap_gpu_source", TEST_BROKEN_TSODYKS_MARKRAM_SYNAPSE_COMPONENT_TYPE, "TestBrokenTsodyksMarkramSynapse",
        "gbase=\"1nS\" erev=\"0mV\"");
    IrProgram program = lower_synapse_to_ir(entry);

    try {
        lower_ir_program_to_gpu_source(program);
        FAIL() << "expected lower_ir_program_to_gpu_source to reject a peredge-variable-referencing "
                  "onevent increment (gpu_source.cpp's resolve_operand fatally rejects any bare "
                  "peredge reference outside loadedge/accedge, and synapse_lowering.cpp's "
                  "lower_on_events never emits a LoadEdgeInstruction for an onevent body)";
    } catch (const std::runtime_error &error) {
        String message = error.what();
        EXPECT_NE(message.find("peredge quantity"), String::npos) << message;
        EXPECT_NE(message.find("plain operand"), String::npos) << message;
    }
}

// ── the achievable substitute: real TM TimeDerivative equations + additive spike-triggered bump ──

TEST(TsodyksMarkram, lowers_through_the_existing_synapse_lowering_pipeline_with_no_new_machinery) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "lowering", TEST_TSODYKS_MARKRAM_SYNAPSE_COMPONENT_TYPE, "TestTsodyksMarkramSynapse",
        "gbase=\"1nS\" erev=\"0mV\" initReleaseProb=\"0.3\" tauRec=\"2s\" tauFac=\"3s\" "
        "depressionDelta=\"-0.25\" facilitationDelta=\"0.15\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_TRUE(entry.is_conductance_based);

    IrProgram program = lower_synapse_to_ir(entry);
    String rendered = print_ir_program(program);

    // .alloc: R and U are ordinary peredge slots (arch §4.3), same as any other per-edge synapse
    // state variable already tested in synapse_lowering_tests.cpp.
    EXPECT_NE(rendered.find("peredge R"), String::npos);
    EXPECT_NE(rendered.find("peredge U"), String::npos);
    EXPECT_NE(rendered.find("param tauRec"), String::npos);
    EXPECT_NE(rendered.find("param tauFac"), String::npos);
    EXPECT_NE(rendered.find("param initReleaseProb"), String::npos);

    // .tick @deliver: a plain additive accedge bump for each -- the EXACT same shape
    // expOneSynapse's own real `accedge g@edge, weight` uses, just two variables instead of one.
    usize deliver_start = rendered.find("@deliver");
    usize integrate_start = rendered.find("@integrate");
    ASSERT_NE(deliver_start, String::npos);
    ASSERT_NE(integrate_start, String::npos);
    String deliver_section = rendered.substr(deliver_start, integrate_start - deliver_start);
    EXPECT_NE(deliver_section.find("accedge R@edge, depressionDelta"), String::npos) << deliver_section;
    EXPECT_NE(deliver_section.find("accedge U@edge, facilitationDelta"), String::npos) << deliver_section;

    // .tick @integrate: both R and U decay via the recognized non-zero-target linear-decay shape
    // (read-decay-writeback-delta, arch §4.3) -- the REAL vendored TimeDerivative equations,
    // verbatim, with no approximation.
    String integrate_section = rendered.substr(integrate_start);
    EXPECT_NE(integrate_section.find("loadedge edge_R_old, R@edge"), String::npos) << integrate_section;
    EXPECT_NE(integrate_section.find("expdecay edge_R, edge_R, tauRec"), String::npos) << integrate_section;
    EXPECT_NE(integrate_section.find("loadedge edge_U_old, U@edge"), String::npos) << integrate_section;
    EXPECT_NE(integrate_section.find("expdecay edge_U, edge_U, tauFac"), String::npos) << integrate_section;
}

// The strongest form of "just works via the existing machinery": lowers, generates real MSL,
// ACTUALLY compiles it through the real Metal runtime compiler (backend::compile_kernel -- stronger
// evidence than an `xcrun -c` syntax check, since a failed compile here throws), and dispatches the
// real generated `_deliver_in`/`_tick` functions against a real WeightMatrix's per-edge Sk storage
// across repeated simulated presynaptic spikes -- reading back R/U's actual reconstructed per-edge
// value after each step and comparing it to a host-side closed-form reference computed from the
// SAME equations (decay-toward-target + additive bump), not just checking "didn't crash".
TEST(TsodyksMarkram, per_edge_state_evolves_correctly_across_repeated_presynaptic_spikes) {
    const f32 init_release_prob = 0.3f;
    const f32 tau_rec = 2.0f;
    const f32 tau_fac = 3.0f;
    const f32 depression_delta = -0.25f;
    const f32 facilitation_delta = 0.15f;
    const f32 dt = 1.0f;

    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "dispatch", TEST_TSODYKS_MARKRAM_SYNAPSE_COMPONENT_TYPE, "TestTsodyksMarkramSynapse",
        "gbase=\"1nS\" erev=\"0mV\" initReleaseProb=\"0.3\" tauRec=\"2s\" tauFac=\"3s\" "
        "depressionDelta=\"-0.25\" facilitationDelta=\"0.15\"");
    IrProgram program = lower_synapse_to_ir(entry);
    GpuSource source = lower_ir_program_to_gpu_source(program);
    ASSERT_EQ(source.functions.size(), 2u);

    const GpuFunctionSignature *tick_signature = nullptr;
    const GpuFunctionSignature *deliver_signature = nullptr;
    for (const auto &signature : source.functions) {
        if (signature.function_name == "TestTsodyksMarkramSynapse_tick") tick_signature = &signature;
        if (signature.function_name == "TestTsodyksMarkramSynapse_deliver_in") deliver_signature = &signature;
    }
    ASSERT_NE(tick_signature, nullptr);
    ASSERT_NE(deliver_signature, nullptr);

    const String &backend_source = source_text_for_this_backend(source);
    KernelHandle tick_handle = compile_kernel(backend_source.c_str(), tick_signature->function_name.c_str());
    KernelHandle deliver_handle = compile_kernel(backend_source.c_str(), deliver_signature->function_name.c_str());

    // node 0 -> node 1 (the only connection this fixture's own NML network declares); rank kept
    // tiny since only R/U's own per-edge Sk storage (not the reconstructed weight) is under test.
    vector<vector<s32>> network = {{1}, {}};
    WeightMatrix weights(network, /*rank=*/4);
    s64 matrix_index_r = weights.add_coefficient_vector(vector<f32>(4, 0.0f));
    s64 matrix_index_u = weights.add_coefficient_vector(vector<f32>(4, 0.0f));
    // get_for_matrix()/lookup_sparse_delta() skip reading Sk entirely (fast path, returns 0) until
    // accumulate_edge_delta() has been called at least once for a given matrix_index -- this test
    // writes Sk via the real compiled `accedge`/`expdecay`-writeback GPU kernels instead of that
    // host-side method, so it must set the same "has been touched" bookkeeping accumulate_edge_delta
    // would have set, or every read below would silently observe the untouched-fast-path's hardcoded
    // 0 instead of what the kernels actually wrote.
    weights.sparse_delta_touched[(usize)matrix_index_r] = true;
    weights.sparse_delta_touched[(usize)matrix_index_u] = true;

    s32 neighbor_buffer[8];
    s64 neighbor_count = weights.get_neighbors(0, neighbor_buffer);
    ASSERT_EQ(neighbor_count, 1);
    ASSERT_EQ(neighbor_buffer[0], 1);
    s32 edge_slot = 0;

    GpuPointer<s32> source_node_indices = allocate<s32>(sizeof(s32));
    GpuPointer<s32> target_node_indices = allocate<s32>(sizeof(s32));
    GpuPointer<s32> edge_slot_indices = allocate<s32>(sizeof(s32));
    source_node_indices.get_contents()[0] = 0;
    target_node_indices.get_contents()[0] = 1;
    edge_slot_indices.get_contents()[0] = edge_slot;

    const s64 neuron_count = 2;
    GpuPointer<f32> network_inputs = allocate<f32>(sizeof(f32) * neuron_count);
    GpuPointer<f32> v_buffer = allocate<f32>(sizeof(f32) * neuron_count);
    GpuPointer<f32> g_buffer = allocate<f32>(sizeof(f32) * neuron_count);
    GpuPointer<f32> i_buffer = allocate<f32>(sizeof(f32) * neuron_count);
    std::memset(network_inputs.get_contents(), 0, sizeof(f32) * neuron_count);
    std::memset(v_buffer.get_contents(), 0, sizeof(f32) * neuron_count);
    std::memset(g_buffer.get_contents(), 0, sizeof(f32) * neuron_count);
    std::memset(i_buffer.get_contents(), 0, sizeof(f32) * neuron_count);

    DispatchContext context;
    context.dt_value = dt;
    context.network_inputs = network_inputs.get_contents();
    context.v_buffer = v_buffer.get_contents();
    context.g_buffer = g_buffer.get_contents();
    context.i_buffer = i_buffer.get_contents();
    context.weights = &weights;
    context.matrix_index_r = matrix_index_r;
    context.matrix_index_u = matrix_index_u;
    context.neuron_count = neuron_count;
    context.source_node_indices = source_node_indices.get_contents();
    context.target_node_indices = target_node_indices.get_contents();
    context.edge_slot_indices = edge_slot_indices.get_contents();
    context.event_count = 1;

    auto dispatch_deliver = [&]() {
        DispatchArgumentBuilder builder;
        for (const String &name : deliver_signature->parameter_names_in_order) append_argument(builder, name, context);
        builder.dispatch(deliver_handle, launch_config_for(context.event_count));
    };
    auto dispatch_tick = [&]() {
        DispatchArgumentBuilder builder;
        for (const String &name : tick_signature->parameter_names_in_order) append_argument(builder, name, context);
        builder.dispatch(tick_handle, launch_config_for(neuron_count));
    };

    f32 expected_r = 0.0f; // Phase-1 per-edge state defaults to 0 (OnStart values not encoded, see
    f32 expected_u = 0.0f; // synapse_lowering.h's own documented scope boundary)

    // Presynaptic spike 1.
    dispatch_deliver();
    expected_r += depression_delta;
    expected_u += facilitation_delta;
    EXPECT_TRUE(approx(weights.get_for_matrix(0, 1, matrix_index_r), expected_r))
        << "R after spike 1: got " << weights.get_for_matrix(0, 1, matrix_index_r) << " want " << expected_r;
    EXPECT_TRUE(approx(weights.get_for_matrix(0, 1, matrix_index_u), expected_u))
        << "U after spike 1: got " << weights.get_for_matrix(0, 1, matrix_index_u) << " want " << expected_u;

    // Relax for one dt between spikes.
    dispatch_tick();
    expected_r = decay_toward_target(expected_r, 1.0f, tau_rec, dt);
    expected_u = decay_toward_target(expected_u, init_release_prob, tau_fac, dt);
    EXPECT_TRUE(approx(weights.get_for_matrix(0, 1, matrix_index_r), expected_r))
        << "R after 1 tick of relaxation: got " << weights.get_for_matrix(0, 1, matrix_index_r) << " want " << expected_r;
    EXPECT_TRUE(approx(weights.get_for_matrix(0, 1, matrix_index_u), expected_u))
        << "U after 1 tick of relaxation: got " << weights.get_for_matrix(0, 1, matrix_index_u) << " want " << expected_u;

    // Presynaptic spike 2 -- proves the bump repeats correctly on top of the relaxed value (not
    // just a single one-shot jump).
    dispatch_deliver();
    expected_r += depression_delta;
    expected_u += facilitation_delta;
    EXPECT_TRUE(approx(weights.get_for_matrix(0, 1, matrix_index_r), expected_r))
        << "R after spike 2: got " << weights.get_for_matrix(0, 1, matrix_index_r) << " want " << expected_r;
    EXPECT_TRUE(approx(weights.get_for_matrix(0, 1, matrix_index_u), expected_u))
        << "U after spike 2: got " << weights.get_for_matrix(0, 1, matrix_index_u) << " want " << expected_u;

    // Relax again.
    dispatch_tick();
    expected_r = decay_toward_target(expected_r, 1.0f, tau_rec, dt);
    expected_u = decay_toward_target(expected_u, init_release_prob, tau_fac, dt);
    EXPECT_TRUE(approx(weights.get_for_matrix(0, 1, matrix_index_r), expected_r))
        << "R after 2 ticks of relaxation: got " << weights.get_for_matrix(0, 1, matrix_index_r) << " want " << expected_r;
    EXPECT_TRUE(approx(weights.get_for_matrix(0, 1, matrix_index_u), expected_u))
        << "U after 2 ticks of relaxation: got " << weights.get_for_matrix(0, 1, matrix_index_u) << " want " << expected_u;
}
