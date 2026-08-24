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
	set_tests_properties(${TARGET} PROPERTIES LABELS "example;gpu_bindless")
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
	set_tests_properties(${TARGET} PROPERTIES LABELS "example;legacy;gpu_bindless")
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# Core — minimal GPU matrix operation with explicit engine submission.
oa_add_example(ExampleCoreMatrixAdd core/matrixAdd.cpp core)

# Audio — semantic waveform metadata around one GPU DSP operation.
oa_add_example(ExampleAudioClip audio/clip.cpp audio)

# Vision — semantic image metadata around one GPU resize operation.
oa_add_example(ExampleVisionResize vision/resize.cpp vision)

# Crypto — GPU SHAKE batch checked against the CPU cryptographic oracle.
if(OA_BUILD_CRYPTO)
	oa_add_example(ExampleCryptoShake256 crypto/shake256.cpp crypto)
endif()

# Plot — retained commands rendered through the headless Ui compositor.
oa_add_example(ExamplePlotLine plot/line.cpp plot)

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
	WORKING_DIRECTORY "${oa_SOURCE_DIR}")
if(TARGET oa_shaders)
	add_dependencies(AlmAg oa_shaders)
endif()
