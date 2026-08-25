# OA maintained public examples.
#
# A normal example is a small standalone application linked only to OA. It is
# registered with CTest and must validate its own result through its exit code.
function(oa_add_example TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE})
	target_link_libraries(${TARGET} PRIVATE oaSdkSupport)
	set_target_properties(${TARGET} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/example/${SUBDIR}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	set_tests_properties(${TARGET} PROPERTIES
		LABELS "example;gpu_bindless"
		ENVIRONMENT
			"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}")
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# Legacy ML examples still use the historical GTest fixture while useful
# single-concept coverage is migrated into standalone examples or test/.
function(oa_add_legacy_ml_example TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${oa_SOURCE_DIR}/sdk/cpp/examples/ml/itTrainingMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oaSdkSupport GTest::gtest)
	target_include_directories(${TARGET} PRIVATE ${oa_SOURCE_DIR}/test/cpp)
	set_target_properties(${TARGET} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/example/${SUBDIR}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	set_tests_properties(${TARGET} PROPERTIES
		LABELS "example;legacy;gpu_bindless"
		ENVIRONMENT
			"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}")
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# Schema-owned beginner examples.
include(${oa_SOURCE_DIR}/sdk/cpp/examples/gen/examples.cmake)

# Historical ML inventory pending conversion under this directory's README.
oa_add_legacy_ml_example(ItTraining ml/itTraining.cpp ml)

add_executable(AlmAg
	ml/alm/almAg.cpp
	${oa_SOURCE_DIR}/sdk/cpp/examples/ml/itTrainingMain.cpp
)
target_link_libraries(AlmAg PRIVATE oaSdkSupport GTest::gtest)
target_include_directories(AlmAg PRIVATE ${oa_SOURCE_DIR}/test/cpp)
set_target_properties(AlmAg PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/example/ml")
add_test(NAME AlmAg COMMAND AlmAg)
set_tests_properties(AlmAg PROPERTIES
	LABELS "example;legacy;gpu_bindless"
	ENVIRONMENT
		"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}"
	WORKING_DIRECTORY "${oa_SOURCE_DIR}")
if(TARGET oa_shaders)
	add_dependencies(AlmAg oa_shaders)
endif()
