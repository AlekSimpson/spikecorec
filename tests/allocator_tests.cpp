#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/synapse_lowering.h"
#include "spikecorec/nml/allocator.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── `.alloc` interpretation -> allocator tests (ticket #5 [C2]) ──────────
//
// Exercises allocate_model() against a real, two-population GLIF network
// (reusing cell_lowering_tests.cpp's own cell_lowering-compatible GLIF1Cell
// fixture -- the plain model_specification_tests.cpp GLIF1Cell can't be
// reused here since it references `iSyn` without declaring it as a
// DerivedVariable, which lower_cell_to_ir would reject) wired to three
// synapse types, all lowering per-edge (the aggregatable/per-edge
// distinction was dropped -- see synapse_lowering.h): expOneSynapse (real,
// conductance-based, one `require v`, one `peredge` variable "g"),
// alphaCurrentSynapse (real, current-based, two `peredge` variables "I"/"J",
// whose alpha-shaped decay isn't Phase-1-recognized so it isn't lowered into
// `.tick`, but the peredge slots are still declared), and TestNmdaSynapse
// (synapse_lowering_tests.cpp's own per-edge fixture, conductance-based,
// `require v`, one `peredge` variable "g"). The GLIF1Cell fixture's `iSyn`
// DerivedVariable (`select`/`reduce="add"`, exposed but never a
// `state`/`accum` slot) doubles as the derived-exposure-scratch acceptance
// case (arch §4.1/§4.5).

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

s32 type_library_index_for(const ModelSpecification &specification, const String &bound_instance_id) {
    for (s32 index = 0; index < (s32)specification.type_library.size(); ++index) {
        if (specification.type_library[(usize)index].bound_instance_id == bound_instance_id) return index;
    }
    throw std::runtime_error("no type library entry for '" + bound_instance_id + "'");
}

// Parallel to `model.type_library`: lowers each Cell/Synapse entry through
// the real #50/#51 lowering; an Inputs entry (no IR-lowering ticket yet,
// see allocator.h's own header comment) gets an empty placeholder program.
Vector<IrProgram> build_type_library_ir_programs(const ModelSpecification &model) {
    Vector<IrProgram> programs;
    programs.reserve(model.type_library.size());
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) {
            programs.push_back(lower_cell_to_ir(entry));
        } else if (entry.category == TypeLibraryCategory::Synapse) {
            programs.push_back(lower_synapse_to_ir(entry));
        } else {
            IrProgram placeholder;
            placeholder.component_type_name = entry.component_type_name;
            programs.push_back(std::move(placeholder));
        }
    }
    return programs;
}

// The cell_lowering_tests.cpp GLIF1Cell fixture verbatim (2 state variables
// `v`/`refractoryTimeElapsed`, a refractory Regime pair, and an exposed-but-
// never-stored `iSyn` DerivedVariable) plus a custom per-edge NMDA synapse
// (synapse_lowering_tests.cpp's TestNmdaSynapse fixture verbatim) and the two
// real aggregatable synapse types (expOneSynapse/alphaCurrentSynapse) wired
// into a 2-population, 3-projection network.
String write_allocator_fixture() {
    write_temp_file("spikecorec_allocator_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"AllocatorContent\">"
        "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
        "    <Parameter name=\"t_ref\" dimension=\"time\"/>"
        "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <StateVariable name=\"refractoryTimeElapsed\" dimension=\"time\"/>"
        "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\" select=\"synapses[*]/i\" reduce=\"add\"/>"
        "      <OnStart>"
        "        <StateAssignment variable=\"v\" value=\"EL\"/>"
        "        <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
        "      </OnStart>"
        "      <Regime name=\"integrating\" initial=\"true\">"
        "        <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
        "        <OnCondition test=\"v .gt. vth\">"
        "          <EventOut port=\"spike\"/>"
        "          <StateAssignment variable=\"v\" value=\"vreset\"/>"
        "          <Transition regime=\"refractory\"/>"
        "        </OnCondition>"
        "      </Regime>"
        "      <Regime name=\"refractory\">"
        "        <OnEntry>"
        "          <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
        "        </OnEntry>"
        "        <TimeDerivative variable=\"refractoryTimeElapsed\" value=\"1\"/>"
        "        <OnCondition test=\"refractoryTimeElapsed .geq. t_ref\">"
        "          <Transition regime=\"integrating\"/>"
        "        </OnCondition>"
        "      </Regime>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF1Cell id=\"glif1ExcCellInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\" t_ref=\"2ms\"/>"
        "  <GLIF1Cell id=\"glif1InhCellInstance\" C=\"1.0e-10\" gL=\"1.2e-8\" EL=\"-65mV\" vth=\"-50mV\" vreset=\"-65mV\" t_ref=\"2ms\"/>"
        "  <expOneSynapse id=\"glifExcSynapse\" gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\" weight=\"2\"/>"
        "  <alphaCurrentSynapse id=\"glifCurrSynapse\" tau=\"2ms\" ibase=\"0.1nA\" weight=\"2\"/>"
        "  <ComponentType name=\"TestNmdaSynapse\" extends=\"baseConductanceBasedSynapse\">"
        "    <Property name=\"weight\" dimension=\"none\" defaultValue=\"1\"/>"
        "    <Parameter name=\"tau\" dimension=\"time\"/>"
        "    <Children name=\"blockMechanisms\" type=\"baseBlockMechanism\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"g\" dimension=\"conductance\" exposure=\"g\"/>"
        "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g * (erev - v)\"/>"
        "      <TimeDerivative variable=\"g\" value=\"-g / tau\"/>"
        "      <OnStart>"
        "        <StateAssignment variable=\"g\" value=\"0\"/>"
        "      </OnStart>"
        "      <OnEvent port=\"in\">"
        "        <StateAssignment variable=\"g\" value=\"g + weight\"/>"
        "      </OnEvent>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <TestNmdaSynapse id=\"glifNmdaSynapse\" gbase=\"1nS\" erev=\"0mV\" tau=\"50ms\" weight=\"1\"/>"
        "  <network id=\"AllocatorNet\">"
        "    <population id=\"ExcPop\" component=\"glif1ExcCellInstance\" size=\"3\"/>"
        "    <population id=\"InhPop\" component=\"glif1InhCellInstance\" size=\"2\"/>"
        "    <projection id=\"ExcToInh\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"InhPop\" synapse=\"glifExcSynapse\">"
        "      <connection id=\"0\" preCellId=\"../ExcPop/0/glif1ExcCellInstance\" postCellId=\"../InhPop/0/glif1InhCellInstance\"/>"
        "    </projection>"
        "    <projection id=\"ExcRecurrent\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"ExcPop\" synapse=\"glifCurrSynapse\">"
        "      <connection id=\"0\" preCellId=\"../ExcPop/0/glif1ExcCellInstance\" postCellId=\"../ExcPop/2/glif1ExcCellInstance\"/>"
        "    </projection>"
        "    <projection id=\"ExcToInhNmda\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"InhPop\" synapse=\"glifNmdaSynapse\">"
        "      <connection id=\"0\" preCellId=\"../ExcPop/2/glif1ExcCellInstance\" postCellId=\"../InhPop/0/glif1InhCellInstance\"/>"
        "    </projection>"
        "  </network>"
        "</neuroml>");

    return write_temp_file("spikecorec_allocator_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"AllocatorTop\">"
        "  <include href=\"spikecorec_allocator_content.nml\"/>"
        "</neuroml>");
}

ModelSpecification build_allocator_test_model() {
    String path = write_allocator_fixture();
    NML_Parser parser;
    parser.parse(path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

} // namespace

// ── widened cell-state vector + O(P) boundary array (arch §4.1) ──────────

TEST(Allocator, sizes_widened_state_vector_and_boundary_array_from_alloc_state_directives) {
    ModelSpecification model = build_allocator_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    ModelAllocation allocation = allocate_model(model, programs);

    // ExcPop: size 3 * V_t 2 (v, refractoryTimeElapsed) = 6, base offset 0.
    // InhPop: size 2 * V_t 2 = 4, base offset 6.
    EXPECT_EQ(allocation.cell_state_element_count, 10);
    EXPECT_EQ(allocation.population_count, 2);

    ASSERT_NE(allocation.cell_type_boundaries.get_contents(), nullptr);
    const s64 *boundaries = allocation.cell_type_boundaries.get_contents();

    s32 exc_pop_index = -1, inh_pop_index = -1;
    for (s32 index = 0; index < (s32)model.populations.size(); ++index) {
        if (model.populations[(usize)index].id == "ExcPop") exc_pop_index = index;
        if (model.populations[(usize)index].id == "InhPop") inh_pop_index = index;
    }
    ASSERT_NE(exc_pop_index, -1);
    ASSERT_NE(inh_pop_index, -1);

    EXPECT_EQ(boundaries[exc_pop_index], 0);
    EXPECT_EQ(boundaries[inh_pop_index], 6);

    // Sanity: every cell_state element is zero-initialized.
    const f32 *state = allocation.cell_state.get_contents();
    for (s64 element_index = 0; element_index < allocation.cell_state_element_count; ++element_index) {
        EXPECT_FLOAT_EQ(state[element_index], 0.0f);
    }
}

// ── synapse state produces no aggregated accumulators ─────────────────────
//
// Arch §4.2's aggregated-per-neuron-accumulator synapse shape (originally
// ticket #57/D1) was dropped in favor of routing every synapse's state
// uniformly through per-edge storage (see synapse_lowering.h's header
// comment and CLAUDE.md's Phase-1 description) -- the per-edge/aggregatable
// distinction this test originally exercised (expOneSynapse/alphaCurrentSynapse
// as "aggregatable" with their own accumulators) no longer exists as a
// classification. `allocate_model`'s AccumDirective handling itself is
// generic and unchanged (it allocates one zeroed per-neuron buffer for
// whatever accum directives an IR program declares); this fixture's three
// synapse types now all declare their state via PeredgeDirective instead,
// so this test instead guards that no synapse type in this fixture
// spuriously produces an accumulator under the corrected design.
TEST(Allocator, synapse_state_produces_no_accumulators_under_per_edge_design) {
    ModelSpecification model = build_allocator_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    ModelAllocation allocation = allocate_model(model, programs);

    EXPECT_EQ(allocation.accumulators.size(), 0u);
}

// ── regime index (arch §4.5) ──────────────────────────────────────────────

TEST(Allocator, allocates_one_shared_regime_index_array_when_any_type_declares_regimes) {
    ModelSpecification model = build_allocator_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    ModelAllocation allocation = allocate_model(model, programs);

    ASSERT_TRUE(allocation.has_regime_index);
    ASSERT_NE(allocation.regime_indices.get_contents(), nullptr);
    const s32 *regime_indices = allocation.regime_indices.get_contents();
    for (s64 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
        EXPECT_EQ(regime_indices[neuron_index], 0);
    }
}

// ── WeightMatrix per-edge variable count (arch §4.3) ──────────────────────

TEST(Allocator, sets_weight_matrix_per_edge_variable_count_from_peredge_directives) {
    ModelSpecification model = build_allocator_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    ModelAllocation allocation = allocate_model(model, programs);

    // Every synapse type now lowers per-edge (the aggregatable/per-edge
    // distinction was dropped, see the accumulators test above): expOneSynapse
    // contributes "g", alphaCurrentSynapse contributes "I" and "J" (its decay
    // shape isn't Phase-1-recognized, but the peredge slot is still declared),
    // and TestNmdaSynapse contributes "g" -- 4 peredge variables total.
    EXPECT_EQ(allocation.per_edge_variable_count, 4);
    ASSERT_TRUE(model.adjacency.has_value());
    EXPECT_EQ(model.adjacency->per_edge_variable_count, 4);
}

// ── require bindings (metadata only, no allocation) ───────────────────────

TEST(Allocator, records_require_bindings_with_no_allocation) {
    ModelSpecification model = build_allocator_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    s32 exp_one_index = type_library_index_for(model, "glifExcSynapse");
    s32 nmda_index = type_library_index_for(model, "glifNmdaSynapse");

    ModelAllocation allocation = allocate_model(model, programs);

    ASSERT_EQ(allocation.require_bindings.size(), 2u);
    bool found_exp_one = false, found_nmda = false;
    for (const auto &binding : allocation.require_bindings) {
        EXPECT_EQ(binding.name, "v");
        EXPECT_EQ(binding.scope, "postsynaptic");
        if (binding.type_library_index == exp_one_index) found_exp_one = true;
        if (binding.type_library_index == nmda_index) found_nmda = true;
    }
    EXPECT_TRUE(found_exp_one);
    EXPECT_TRUE(found_nmda);
}

// ── baked param constants, collected/validated (arch §3.1) ────────────────

TEST(Allocator, collects_baked_param_constants_per_type) {
    ModelSpecification model = build_allocator_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    s32 exc_cell_index = type_library_index_for(model, "glif1ExcCellInstance");
    s32 inh_cell_index = type_library_index_for(model, "glif1InhCellInstance");

    ModelAllocation allocation = allocate_model(model, programs);

    ASSERT_EQ(allocation.baked_param_constants.count(exc_cell_index), 1u);
    const auto &exc_constants = allocation.baked_param_constants.at(exc_cell_index);
    ASSERT_EQ(exc_constants.count("EL"), 1u);
    EXPECT_NEAR(exc_constants.at("EL"), -0.070, 1e-9);
    ASSERT_EQ(exc_constants.count("gL"), 1u);
    EXPECT_NEAR(exc_constants.at("gL"), 1.0e-8, 1e-15);

    ASSERT_EQ(allocation.baked_param_constants.count(inh_cell_index), 1u);
    const auto &inh_constants = allocation.baked_param_constants.at(inh_cell_index);
    ASSERT_EQ(inh_constants.count("EL"), 1u);
    EXPECT_NEAR(inh_constants.at("EL"), -0.065, 1e-9);
}

// ── recorded-derived-exposure scratch (arch §4.1/§4.5) ────────────────────

TEST(Allocator, allocates_scratch_for_a_cell_type_derived_exposure_with_no_matching_state_slot) {
    ModelSpecification model = build_allocator_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    s32 exc_cell_index = type_library_index_for(model, "glif1ExcCellInstance");
    s32 inh_cell_index = type_library_index_for(model, "glif1InhCellInstance");

    ModelAllocation allocation = allocate_model(model, programs);

    // `iSyn` is exposed but never a `state`/`accum` slot (it's a
    // select/reduce DerivedVariable aliased to network_inputs) -- one
    // per-neuron scratch buffer per Cell type-library entry.
    ASSERT_EQ(allocation.derived_exposure_scratch_buffers.size(), 2u);
    EXPECT_EQ(allocation.derived_exposure_scratch_buffers.count(type_scoped_key(exc_cell_index, "iSyn")), 1u);
    EXPECT_EQ(allocation.derived_exposure_scratch_buffers.count(type_scoped_key(inh_cell_index, "iSyn")), 1u);

    // `v`'s own exposure matches its `state` slot directly -- no scratch.
    EXPECT_EQ(allocation.derived_exposure_scratch_buffers.count(type_scoped_key(exc_cell_index, "v")), 0u);
}

// ── validation: mismatched parallel-array length ──────────────────────────

TEST(Allocator, throws_when_ir_program_count_does_not_match_type_library_size) {
    ModelSpecification model = build_allocator_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ASSERT_FALSE(programs.empty());
    programs.pop_back();

    EXPECT_THROW(allocate_model(model, programs), std::runtime_error);
}

// ── validation: non-f32 dtype (Phase-1 scope boundary) ────────────────────

TEST(Allocator, throws_on_non_f32_state_dtype) {
    ModelSpecification model;
    model.total_neuron_count = 1;

    TypeLibraryEntry cell_entry;
    cell_entry.component_type_name = "BadDtypeCell";
    cell_entry.bound_instance_id = "badDtypeCellInstance";
    cell_entry.category = TypeLibraryCategory::Cell;
    cell_entry.state_variable_count = 1;
    model.type_library.push_back(cell_entry);

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = 1;
    population.neuron_index_begin = 0;
    population.neuron_index_end = 1;
    model.populations.push_back(population);

    IrProgram program;
    program.component_type_name = "BadDtypeCell";
    program.alloc.push_back(StateDirective{"v", "f64", std::nullopt});
    Vector<IrProgram> programs{program};

    EXPECT_THROW(allocate_model(model, programs), std::runtime_error);
}
