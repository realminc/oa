execute_process(
	COMMAND ${Python_EXECUTABLE} -m pytest --version
	RESULT_VARIABLE OA_PYTEST_RESULT
	OUTPUT_QUIET
	ERROR_QUIET)
if(NOT OA_PYTEST_RESULT EQUAL 0)
	message(FATAL_ERROR
		"OA_BUILD_TESTS=ON with OA_BUILD_PYTHON=ON requires pytest for ${Python_EXECUTABLE}")
endif()

function(oa_add_python_profile TARGET PROFILE LABELS)
	add_test(
		NAME ${TARGET}
		COMMAND ${Python_EXECUTABLE} -m pytest
			-q --strict-config --strict-markers
			--oa-profile=${PROFILE} --oa-forbid-skips
			${CMAKE_CURRENT_SOURCE_DIR})
	set_tests_properties(${TARGET} PROPERTIES
		LABELS "python;${LABELS}"
		WORKING_DIRECTORY ${oa_SOURCE_DIR}
		ENVIRONMENT
			"OA_PYTHON_BUILD_DIR=$<TARGET_FILE_DIR:_oa>;PYTHONPATH=${OA_PYTHON_DIR}:$<TARGET_FILE_DIR:_oa>;OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}")
endfunction()

oa_add_python_profile(TestPythonHost host "host")
oa_add_python_profile(TestPythonGpu gpu "gpu_bindless")

if(OA_BUILD_CRYPTO)
	oa_add_python_profile(TestPythonCryptoHost crypto-host "crypto;host")
	oa_add_python_profile(TestPythonCryptoGpu crypto-gpu "crypto;gpu_bindless")
endif()

execute_process(
	COMMAND ${Python_EXECUTABLE} -c "import gymnasium"
	RESULT_VARIABLE OA_GYMNASIUM_RESULT
	OUTPUT_QUIET
	ERROR_QUIET)
if(OA_GYMNASIUM_RESULT EQUAL 0)
	oa_add_python_profile(TestPythonGymnasium external "external;gpu_bindless;rl")
else()
	message(STATUS "Python Gymnasium profile: UNAVAILABLE (gymnasium is not installed)")
endif()
