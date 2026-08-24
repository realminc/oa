if(NOT OA_EMBED_SCRIPT OR NOT TEST_ROOT)
	message(FATAL_ERROR "test_embed_spirv.cmake requires OA_EMBED_SCRIPT and TEST_ROOT")
endif()

if(TEST_ROOT STREQUAL "/" OR TEST_ROOT STREQUAL "")
	message(FATAL_ERROR "refusing unsafe TEST_ROOT")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/spv/nested")
file(WRITE "${TEST_ROOT}/spv/Alpha.spv" "alpha-module")
file(WRITE "${TEST_ROOT}/spv/nested/Beta.spv" "beta-module")
# Deliberately omit a final newline: the production manifest historically did.
file(WRITE "${TEST_ROOT}/manifest.txt" "Alpha\nnested/Beta")

set(OUTPUT "${TEST_ROOT}/embedded.cpp")
execute_process(
	COMMAND "${CMAKE_COMMAND}"
		-DSPIRV_DIR=${TEST_ROOT}/spv
		-DOUTPUT=${OUTPUT}
		-DSHADER_LIST_FILE=${TEST_ROOT}/manifest.txt
		-P ${OA_EMBED_SCRIPT}
	RESULT_VARIABLE GENERATE_RESULT
	OUTPUT_VARIABLE GENERATE_STDOUT
	ERROR_VARIABLE GENERATE_STDERR)
if(NOT GENERATE_RESULT EQUAL 0)
	message(FATAL_ERROR
		"embed generator failed: ${GENERATE_STDOUT}\n${GENERATE_STDERR}")
endif()

file(SHA256 "${TEST_ROOT}/spv/Alpha.spv" ALPHA_SHA256)
file(SHA256 "${TEST_ROOT}/spv/nested/Beta.spv" BETA_SHA256)
string(SUBSTRING "${ALPHA_SHA256}" 0 16 ALPHA_CONTENT_ID)
string(SUBSTRING "${BETA_SHA256}" 0 16 BETA_CONTENT_ID)
file(READ "${OUTPUT}" GENERATED)

foreach(EXPECTED
		"{\"Alpha\", kSpv_Alpha, 12, 0x${ALPHA_CONTENT_ID}ULL}"
		"{\"nested/Beta\", kSpv_nested_Beta, 11, 0x${BETA_CONTENT_ID}ULL}"
		"oa::U32 oavk::spirvCount() { return 2; }")
	string(FIND "${GENERATED}" "${EXPECTED}" FOUND_AT)
	if(FOUND_AT EQUAL -1)
		message(FATAL_ERROR "generated registry is missing: ${EXPECTED}")
	endif()
endforeach()

file(SHA256 "${OUTPUT}" FIRST_OUTPUT_SHA256)
execute_process(
	COMMAND "${CMAKE_COMMAND}"
		-DSPIRV_DIR=${TEST_ROOT}/spv
		-DOUTPUT=${OUTPUT}
		-DSHADER_LIST_FILE=${TEST_ROOT}/manifest.txt
		-P ${OA_EMBED_SCRIPT}
	RESULT_VARIABLE REGENERATE_RESULT)
if(NOT REGENERATE_RESULT EQUAL 0)
	message(FATAL_ERROR "identical-input regeneration failed")
endif()
file(SHA256 "${OUTPUT}" SECOND_OUTPUT_SHA256)
if(NOT FIRST_OUTPUT_SHA256 STREQUAL SECOND_OUTPUT_SHA256)
	message(FATAL_ERROR "identical inputs produced different embedded output")
endif()

# A failed generation must not replace the last good output.
file(WRITE "${OUTPUT}" "last-good-output")
file(WRITE "${TEST_ROOT}/manifest.txt" "Missing")
execute_process(
	COMMAND "${CMAKE_COMMAND}"
		-DSPIRV_DIR=${TEST_ROOT}/spv
		-DOUTPUT=${OUTPUT}
		-DSHADER_LIST_FILE=${TEST_ROOT}/manifest.txt
		-P ${OA_EMBED_SCRIPT}
	RESULT_VARIABLE MISSING_RESULT
	OUTPUT_QUIET
	ERROR_QUIET)
if(MISSING_RESULT EQUAL 0)
	message(FATAL_ERROR "missing SPIR-V unexpectedly succeeded")
endif()
file(READ "${OUTPUT}" PRESERVED_OUTPUT)
if(NOT PRESERVED_OUTPUT STREQUAL "last-good-output")
	message(FATAL_ERROR "failed generation replaced the last good output")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
