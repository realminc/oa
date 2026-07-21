# ═══════════════════════════════════════════════════════════════════════════
# Examples — Quick tests and prototypes (not polished tutorials)
# ═══════════════════════════════════════════════════════════════════════════

# Examples are for rapid prototyping and testing, not production code.
# They use the same infrastructure as tutorials but with less polish.

# Example helper function (similar to oa_add_tutorial but for Examples)
function(oa_add_example TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${oa_SOURCE_DIR}/Examples/Ml/MlTrainingMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oa GTest::gtest)
	target_include_directories(${TARGET} PRIVATE ${oa_SOURCE_DIR}/Test)
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Example/${SUBDIR}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# ─── ML Examples ───────────────────────────────────────────────────────────

# OaMlTraining usage examples
oa_add_example(MlTrainingSimple Ml/MlTrainingSimple.cpp Ml)

# AlmAg — OaAlm autograd test suite
add_executable(AlmAg
	Ml/Alm/AlmAg.cpp
	${oa_SOURCE_DIR}/Examples/Ml/MlTrainingMain.cpp
)
target_link_libraries(AlmAg PRIVATE oa GTest::gtest)
target_include_directories(AlmAg PRIVATE ${oa_SOURCE_DIR}/Test)
set_target_properties(AlmAg PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Example/Ml")
add_test(NAME AlmAg COMMAND AlmAg)
set_tests_properties(AlmAg PROPERTIES WORKING_DIRECTORY "${oa_SOURCE_DIR}")
if(TARGET oa_shaders)
	add_dependencies(AlmAg oa_shaders)
endif()

# Uncomment to add more ML examples:
# add_executable(ExampleMl_QuickTrain Examples/Ml/QuickTrain.cpp)
# target_link_libraries(ExampleMl_QuickTrain PRIVATE oa GTest::gtest_main)
# add_test(NAME ExampleMl.QuickTrain COMMAND ExampleMl_QuickTrain)

# ─── Core Examples ─────────────────────────────────────────────────────────

# Uncomment to add Core examples:
# add_executable(ExampleCore_QuickCompute Examples/Core/QuickCompute.cpp)
# target_link_libraries(ExampleCore_QuickCompute PRIVATE oa GTest::gtest_main)
# add_test(NAME ExampleCore.QuickCompute COMMAND ExampleCore_QuickCompute)

# ─── Scratch (gitignored, for truly temporary files) ──────────────────────

# Scratch files are not tracked by git and can be deleted anytime.
# To use: create Examples/Scratch/test.cpp, uncomment below, build, run, delete.

# file(GLOB SCRATCH_SOURCES "Examples/Scratch/*.cpp")
# foreach(SCRATCH_FILE ${SCRATCH_SOURCES})
#     get_filename_component(SCRATCH_NAME ${SCRATCH_FILE} NAME_WE)
#     add_executable(Scratch_${SCRATCH_NAME} ${SCRATCH_FILE})
#     target_link_libraries(Scratch_${SCRATCH_NAME} PRIVATE oa GTest::gtest_main)
#     add_test(NAME Scratch.${SCRATCH_NAME} COMMAND Scratch_${SCRATCH_NAME})
# endforeach()

# ═══════════════════════════════════════════════════════════════════════════
# Usage:
#   1. Create Examples/Ml/MyTest.cpp
#   2. Uncomment/add build rules above
#   3. cmake --build Build/Release --target ExampleMl_MyTest
#   4. ctest -R ExampleMl.MyTest
#   5. Delete when done or move to Tutorial/ if valuable
# ═══════════════════════════════════════════════════════════════════════════
