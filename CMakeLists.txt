# OA — REALM FOUNDATION LIBRARY
#
# Vulkan 1.4 compute. Slang shaders. One backend across NVIDIA, AMD, and Intel.
cmake_minimum_required(VERSION 3.20)

# ─── Version: single source of truth ───
# The top-level VERSION file is the ONE place the version lives — read here by CMake and
# by the Python package (source/py/oa/__init__.py). It may carry a pre-release suffix
# (e.g. "0.7.0-dev"); strip it to numeric X.Y.Z for project(), keep the full string for
# the OA_VERSION_STRING compile definition consumed by <oa/core/version.h>.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" OA_VERSION_FULL LIMIT_COUNT 1)
string(STRIP "${OA_VERSION_FULL}" OA_VERSION_FULL)
if(OA_VERSION_FULL MATCHES "^([0-9]+[.][0-9]+[.][0-9]+)")
	set(OA_VERSION_NUMERIC "${CMAKE_MATCH_1}")
else()
	set(OA_VERSION_NUMERIC "0.0.0")
	set(OA_VERSION_FULL "0.0.0-unknown")
endif()

# ─── vcpkg manifest features ───
# Optional dependencies live behind vcpkg.json features. vcpkg resolves the
# manifest during project() below, so feature selection MUST happen here —
# before project() — by reading the build options straight from the cache (they
# arrive via -D before option() formalizes them further down). A default
# configure enables no optional feature.
if(OA_BUILD_PYTHON)
	list(APPEND VCPKG_MANIFEST_FEATURES "python")
endif()
if(OA_BUILD_CRYPTO)
	list(APPEND VCPKG_MANIFEST_FEATURES "crypto")
endif()

project(oa
	VERSION ${OA_VERSION_NUMERIC}
	DESCRIPTION "Realm Foundation Library — Vulkan compute, ML, Crypto"
	LANGUAGES C CXX
)

# Full version string (incl. any pre-release suffix) available to all C/C++ sources.
add_compile_definitions(OA_VERSION_STRING="${OA_VERSION_FULL}")

# ─── Submodule Detection ───
# When oa is included via add_subdirectory() from ml/ or chain/, we ALWAYS embed
# OA's core kernels into liboa.a. Parent projects use --whole-archive linker flag
# to force-link all OA symbols (including embedded shaders) into their executables.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	message(STATUS "oa: running as submodule — OA kernels will be embedded in liboa (parent uses --whole-archive)")
else()
	message(STATUS "oa: running as standalone project — OA kernels will be embedded in liboa")
endif()

# OPTIONS
option(OA_BUILD_TESTS "Build unit tests" ON)
option(OA_OASTD_EXTRA_TARGETS "Optional: oa_oastd_headers (IDE file list) + oastd -> TestOaStd" ON)
option(OA_VULKAN_VALIDATION "Enable Vulkan validation layers" OFF)
option(OA_BUILD_CRYPTO "Build crypto implementations (requires pqcrypto)" OFF)
option(OA_EMBED_SHADERS "Embed OA compute and graphics shaders into liboa (required)" ON)
option(OA_BUILD_PYTHON "Build Python bindings via nanobind (vcpkg feature python)" OFF)
option(OA_BUILD_SHARED "Build liboa as a shared library (required for distributable runtime packages)" OFF)
option(OA_BUILD_TUTORIALS "Build the tutorial executables (installable as oa-* CLI tools)" OFF)
option(OA_ANDROID_ML "Build the Android-safe Runtime + Core + ML source profile" ${ANDROID})
option(OA_ENABLE_NATIVE_ARCH "Enable -march=native for local non-Android Release builds" ON)
set(OA_SANITIZERS "" CACHE STRING "Comma-separated host sanitizers: address, undefined, or address,undefined")
set_property(CACHE OA_SANITIZERS PROPERTY STRINGS "" "address" "undefined" "address,undefined")

if(NOT OA_EMBED_SHADERS)
	message(FATAL_ERROR
		"OA_EMBED_SHADERS=OFF is no longer supported: build-generated embedded SPIR-V is the sole runtime kernel source")
endif()

# COMPILER SETTINGS — CLANG ONLY
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	message(WARNING "OA requires Clang. Detected: ${CMAKE_CXX_COMPILER_ID}")
	if(OA_STRICT_COMPILER)
		message(FATAL_ERROR "GCC is not supported. Use Clang.")
	endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(OA_SANITIZERS)
	if(ANDROID)
		message(FATAL_ERROR "OA_SANITIZERS is a host validation profile and is not supported by the Android build")
	endif()
	if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
		message(FATAL_ERROR "OA_SANITIZERS requires Clang or GCC")
	endif()
	if(NOT OA_SANITIZERS MATCHES "^(address|undefined|address,undefined)$")
		message(FATAL_ERROR "Invalid OA_SANITIZERS='${OA_SANITIZERS}'")
	endif()
	add_compile_options("-fsanitize=${OA_SANITIZERS}" -fno-omit-frame-pointer)
	add_link_options("-fsanitize=${OA_SANITIZERS}")
	message(STATUS "Host sanitizers: ${OA_SANITIZERS}")
endif()

add_subdirectory(source)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=unused-result -Wno-error=unused-parameter -Wno-error=missing-field-initializers)
	if(CMAKE_BUILD_TYPE STREQUAL "Release")
		add_compile_options(-O3)
		if(NOT ANDROID AND OA_ENABLE_NATIVE_ARCH)
			add_compile_options(-march=native)
		endif()
	endif()
else()
	# GCC configuration
	add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=unused-result -Wno-error=unused-parameter -Wno-error=missing-field-initializers)
	if(CMAKE_BUILD_TYPE STREQUAL "Release")
		add_compile_options(-O3)
		if(NOT ANDROID AND OA_ENABLE_NATIVE_ARCH)
			add_compile_options(-march=native)
		endif()
	endif()
endif()

# ─── Binary output: bin/{preset}/{app,example,tutorial,test}/ ───────────────
# Native output is already identified by being an executable. Keep `cpp` only
# in authored source roots; repeating it below an artifact-kind directory adds
# no useful build axis.
cmake_path(GET CMAKE_BINARY_DIR FILENAME OA_PRESET_NAME)
set(OA_BIN_DIR "${CMAKE_SOURCE_DIR}/bin/${OA_PRESET_NAME}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}")
if(OA_BUILD_TESTS)
	file(MAKE_DIRECTORY "${OA_BIN_DIR}/test")
endif()

# CMake target identities remain stable because CTest, packaging, and downstream
# projects address them by name. Filesystem artifacts follow OA's camelCase file
# convention: TutorialNlpByteRnnAg -> tutorialNlpByteRnnAg, TestTypes -> testTypes.
function(oa_apply_executable_output_names DIRECTORY_PATH)
	get_property(_oa_targets DIRECTORY "${DIRECTORY_PATH}" PROPERTY BUILDSYSTEM_TARGETS)
	foreach(_oa_target IN LISTS _oa_targets)
		get_target_property(_oa_type ${_oa_target} TYPE)
		get_target_property(_oa_output_name ${_oa_target} OUTPUT_NAME)
		if(_oa_type STREQUAL "EXECUTABLE" AND
			(NOT _oa_output_name OR _oa_output_name MATCHES "-NOTFOUND$"))
			string(SUBSTRING "${_oa_target}" 0 1 _oa_first)
			string(TOLOWER "${_oa_first}" _oa_first)
			string(SUBSTRING "${_oa_target}" 1 -1 _oa_rest)
			set_target_properties(${_oa_target} PROPERTIES
				OUTPUT_NAME "${_oa_first}${_oa_rest}")
		endif()
	endforeach()

	get_property(_oa_subdirectories DIRECTORY "${DIRECTORY_PATH}" PROPERTY SUBDIRECTORIES)
	foreach(_oa_subdirectory IN LISTS _oa_subdirectories)
		oa_apply_executable_output_names("${_oa_subdirectory}")
	endforeach()
endfunction()

add_compile_definitions(OA_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
if(WIN32)
	add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

# VULKAN SDK + SLANG COMPILER — Required. No CPU-only builds.
if(NOT ANDROID)
	find_package(VulkanHeaders CONFIG)
	if(NOT VulkanHeaders_FOUND)
		find_package(Vulkan)
		if(Vulkan_FOUND)
			set(VULKAN_INCLUDE_DIR "${Vulkan_INCLUDE_DIRS}")
		else()
			message(FATAL_ERROR "Neither VulkanHeaders (vcpkg) nor Vulkan SDK found")
		endif()
	endif()
else()
	# The NDK provides core Vulkan headers but omits the standard vk_video
	# include set referenced by OA's generated loader. Vulkan-Headers is
	# architecture-neutral; use the workstation package for declarations only.
	unset(OA_ANDROID_VULKAN_HEADERS CACHE)
	find_path(OA_ANDROID_VULKAN_HEADERS vk_video/vulkan_video_codec_av1std.h
		PATHS
		"$ENV{VCPKG_INSTALLED_DIR}/x64-linux/include"
		"$ENV{HOME}/.vcpkg/oa/release/installed/x64-linux/include"
		"$ENV{HOME}/.vcpkg/installed/x64-linux/include"
		NO_DEFAULT_PATH
		NO_CMAKE_FIND_ROOT_PATH)
	if(NOT OA_ANDROID_VULKAN_HEADERS)
		message(FATAL_ERROR "Android OA build requires Vulkan-Headers (vk_video includes)")
	endif()
endif()

# Prefer a standalone Slang install (bf16/fp8 fixes land in shader-slang releases, often ahead of Vulkan SDK).
# https://github.com/shader-slang/slang/releases/ — extract e.g. slang-2026.5.2-linux-x86_64.tar.gz and set SLANG_ROOT or PATH.
find_program(SLANGC slangc HINTS
	"$ENV{SLANG_ROOT}/bin"
	"$ENV{HOME}/tools/slang/bin"
	"${VCPKG_INSTALLED_DIR}/x64-linux/tools/shader-slang"
	"$ENV{VCPKG_INSTALLED_DIR}/x64-linux/tools/shader-slang"
	"$ENV{HOME}/.vcpkg/vcpkg/installed/x64-linux/tools/shader-slang"
	"$ENV{VCPKG_ROOT}/installed/x64-windows/tools/shader-slang"
	"$ENV{VCPKG_ROOT}/installed/x64-windows-static/tools/shader-slang"
	"$ENV{VULKAN_SDK}/bin"
	"/usr/bin")
find_program(GLSLANG_VALIDATOR glslangValidator)
if(SLANGC)
	execute_process(COMMAND "${SLANGC}" -version
		OUTPUT_VARIABLE OA_SLANG_VERSION
		ERROR_VARIABLE OA_SLANG_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE OA_SLANG_VER_RC)
	string(STRIP "${OA_SLANG_VERSION}" OA_SLANG_VERSION)
	if(OA_SLANG_VER_RC EQUAL 0 AND OA_SLANG_VERSION)
		message(STATUS "slangc: ${SLANGC} (Slang ${OA_SLANG_VERSION})")
	else()
		message(STATUS "slangc: ${SLANGC}")
	endif()
elseif(OA_EMBED_SHADERS)
	message(FATAL_ERROR "OA_EMBED_SHADERS=ON but slangc not found. Install Slang or set -DSLANGC=<path>.")
endif()

if(OA_VULKAN_VALIDATION)
	add_compile_definitions(OA_VULKAN_VALIDATION)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Slang feature-test compiles (PR-1 item A from OaLlamaCppVulkanLessons.md §9).
# Each test compiles a minimal .slang file that requires exactly one feature.
# If slangc rejects it, the corresponding OA_HAVE_*_SLANG cache var is OFF and
# the dependent code paths (SPV embed list entries, runtime ext enable) drop
# out of the build cleanly. Same source tree builds correctly on a host with
# old slangc as a host with new slangc.
# Mirrors llama.cpp's test_shader_extension_support in
# ggml/src/ggml-vulkan/CMakeLists.txt:38-57.
# ─────────────────────────────────────────────────────────────────────────────
function(test_slang_extension_support FEATURE_NAME TEST_SHADER RESULT_VAR)
	if(NOT SLANGC)
		set(${RESULT_VAR} OFF PARENT_SCOPE)
		return()
	endif()
	set(_OA_SLANG_CAP_ARGS "")
	foreach(_CAP ${ARGN})
		list(APPEND _OA_SLANG_CAP_ARGS -capability ${_CAP})
	endforeach()
	# Need both include paths (umbrella + math modules) so basic imports resolve.
	set(_OA_FT_INC_UMBRELLA "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/include/oa/core")
	set(_OA_FT_INC_MODULES  "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/math")
	execute_process(
		COMMAND ${SLANGC} ${TEST_SHADER}
			-I ${_OA_FT_INC_UMBRELLA} -I ${_OA_FT_INC_MODULES}
			-target spirv -entry main ${_OA_SLANG_CAP_ARGS}
			-o ${CMAKE_BINARY_DIR}/_slang_feature_test.spv
		OUTPUT_VARIABLE _slang_out
		ERROR_VARIABLE  _slang_err
		RESULT_VARIABLE _slang_rc
	)
	if(_slang_rc EQUAL 0)
		message(STATUS "slang feature: ${FEATURE_NAME} supported (${RESULT_VAR}=ON)")
		set(${RESULT_VAR} ON PARENT_SCOPE)
		add_compile_definitions(${RESULT_VAR})
	else()
		message(STATUS "slang feature: ${FEATURE_NAME} NOT supported (${RESULT_VAR}=OFF)")
		set(${RESULT_VAR} OFF PARENT_SCOPE)
	endif()
endfunction()

function(test_glsl_extension_support FEATURE_NAME TEST_SHADER RESULT_VAR)
	if(NOT GLSLANG_VALIDATOR)
		message(STATUS "glsl feature: ${FEATURE_NAME} NOT supported (${RESULT_VAR}=OFF, glslangValidator not found)")
		set(${RESULT_VAR} OFF PARENT_SCOPE)
		return()
	endif()
	execute_process(
		COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.4 ${TEST_SHADER}
			-o ${CMAKE_BINARY_DIR}/_glsl_feature_test.spv
		OUTPUT_VARIABLE _glsl_out
		ERROR_VARIABLE  _glsl_err
		RESULT_VARIABLE _glsl_rc
	)
	if(_glsl_rc EQUAL 0)
		message(STATUS "glsl feature: ${FEATURE_NAME} supported (${RESULT_VAR}=ON)")
		set(${RESULT_VAR} ON PARENT_SCOPE)
		add_compile_definitions(${RESULT_VAR})
	else()
		message(STATUS "glsl feature: ${FEATURE_NAME} NOT supported (${RESULT_VAR}=OFF)")
		set(${RESULT_VAR} OFF PARENT_SCOPE)
	endif()
endfunction()

set(_OA_FEATURE_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/feature-tests")
test_slang_extension_support("VK_KHR_cooperative_matrix"
	"${_OA_FEATURE_TESTS_DIR}/coopMatKhr.slang"   OA_HAVE_COOPMAT_KHR_SLANG
	spvCooperativeMatrixKHR)
test_slang_extension_support("VK_KHR_shader_bfloat16 (type only)"
	"${_OA_FEATURE_TESTS_DIR}/shaderBfloat16.slang" OA_HAVE_BFLOAT16_SLANG)
test_slang_extension_support("VK_KHR_shader_bfloat16 + KHR_cooperative_matrix"
	"${_OA_FEATURE_TESTS_DIR}/coopMatBf16.slang"  OA_HAVE_COOPMAT_BF16_SLANG
	spvCooperativeMatrixKHR spvCooperativeMatrixConversionsNV)
test_slang_extension_support("VK_KHR_shader_integer_dot_product"
	"${_OA_FEATURE_TESTS_DIR}/integerDotProduct.slang" OA_HAVE_INTEGER_DOT_PRODUCT_SLANG
	spvIntegerDotProduct)
test_slang_extension_support("VK_NV_cooperative_vector"
	"${_OA_FEATURE_TESTS_DIR}/coopVecNv.slang"    OA_HAVE_COOPVEC_NV_SLANG
	spvCooperativeVectorNV)
test_slang_extension_support("VK_NV_cooperative_matrix2"
	"${_OA_FEATURE_TESTS_DIR}/coopMat2Nv.slang"   OA_HAVE_COOPMAT2_NV_SLANG
	SPV_NV_cooperative_matrix2 spvCooperativeMatrixTensorAddressingNV)
test_glsl_extension_support("GL_NV_cooperative_matrix2"
	"${_OA_FEATURE_TESTS_DIR}/coopMat2Nv.comp"    OA_HAVE_COOPMAT2_NV_GLSL)
# Pure-KHR bf16 CoopMat1 in GLSL — the portable SMEM GEMM's exact feature set
# (no NV extension). Available on NVIDIA and AMD RDNA3.5/Strix.
test_glsl_extension_support("GL_KHR_cooperative_matrix + GL_EXT_bfloat16"
	"${_OA_FEATURE_TESTS_DIR}/coopMat1Bf16Khr.comp" OA_HAVE_COOPMAT1_BF16_KHR_GLSL)


# NVTX profiling support (CUDA toolkit)
find_path(NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
	HINTS
		/opt/cuda/targets/x86_64-linux/include
		/usr/local/cuda/targets/x86_64-linux/include
		$ENV{CUDA_PATH}/include
)
if(NVTX_INCLUDE_DIR)
	message(STATUS "NVTX found: ${NVTX_INCLUDE_DIR}")
	include_directories(${NVTX_INCLUDE_DIR})
else()
	message(STATUS "NVTX not found - profiling markers will be disabled")
endif()

# DEPENDENCIES
if(NOT ANDROID)
	find_package(yaml-cpp CONFIG)
	find_package(CLI11 CONFIG)
	find_package(fmt CONFIG)
	find_package(hwy CONFIG)
	find_package(utf8proc CONFIG REQUIRED)
	# The vcpkg HarfBuzz config imports FreeType, whose exported static target
	# links ZLIB::ZLIB without declaring that dependency itself.
	find_package(ZLIB REQUIRED)
	find_package(harfbuzz CONFIG REQUIRED)
endif()

# HEADER-ONLY INTERFACE LIBRARY (transitive oa_api + optional CLI/YAML/fmt)
add_library(oa_headers INTERFACE)
target_link_libraries(oa_headers INTERFACE oa_api)
target_compile_features(oa_headers INTERFACE cxx_std_20)
if(yaml-cpp_FOUND)
	target_link_libraries(oa_headers INTERFACE yaml-cpp::yaml-cpp)
	target_compile_definitions(oa_headers INTERFACE OA_HAS_YAML_CPP)
endif()
if(CLI11_FOUND)
	target_link_libraries(oa_headers INTERFACE CLI11::CLI11)
	target_compile_definitions(oa_headers INTERFACE OA_HAS_CLI11)
endif()
if(fmt_FOUND)
	target_link_libraries(oa_headers INTERFACE fmt::fmt)
	target_compile_definitions(oa_headers INTERFACE OA_HAS_FMT)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/fnmatrix/fnMatrixSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/fnAutogenKernels.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/kernelManifest.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/ml/fnmatrix/fnMatrixSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/ml/nn/nnSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/audio/fnaudio/fnaudiosources.gen.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/vision/fnimage/fnImageSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/vision/fnvideo/fnvideosources.gen.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/ui/matrix/fnmatrixsources.gen.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/crypto/hash/fnsources.gen.cmake OPTIONAL)

# OA LIBRARY (implementation: source/cpp/lib/oa)
set(OA_SOURCES
	# Core
	source/cpp/lib/oa/core/envFlag.cpp
	source/cpp/lib/oa/core/log.cpp
	source/cpp/lib/oa/core/validation.cpp
	source/cpp/lib/oa/core/memory.cpp
	source/cpp/lib/oa/core/image.cpp
	source/cpp/lib/oa/core/color.cpp
	source/cpp/lib/oa/core/matrix.cpp
	source/cpp/lib/oa/core/op.cpp
	source/cpp/lib/oa/core/autograd.cpp
	source/cpp/lib/oa/core/autograd/autogradAttach.gen.cpp
	source/cpp/lib/oa/core/fnmatrix/fnMatrix.cpp
	source/cpp/lib/oa/core/fnmatrix/alloc/fnMatrixAlloc.cpp
	source/cpp/lib/oa/core/fnmatrix/view/fnMatrixView.cpp
	# FnMatrix — modularized by category (GPU-only, pure Vulkan)
	source/cpp/lib/oa/core/fnmatrix/elemwise/fnMatrixElemwise.cpp
	source/cpp/lib/oa/core/fnmatrix/blas/fnMatrixBlas.cpp
	source/cpp/lib/oa/core/fnmatrix/reduce/fnMatrixReduce.cpp
	source/cpp/lib/oa/core/fnmatrix/index/fnMatrixIndex.cpp
	source/cpp/lib/oa/core/fnmatrix/rng/fnMatrixRng.cpp

	# Runtime — fixed kernel metadata and public lookup implementation
	source/cpp/lib/oa/runtime/kernelRegistry.cpp
	# FnMatrixFused.cpp removed — fusion is a runtime/matcher concern (OaBlasGemm.md Phase 5)

	# FnMatrix — AUTOGENERATED from FnAutogen schemas (100% coverage)
	${OA_CORE_FN_MATRIX_GENERATED_SOURCES}
	source/cpp/lib/oa/runtime/gemm/dispatch.cpp
	source/cpp/lib/oa/runtime/gemm/graphLowering.cpp
	source/cpp/lib/oa/runtime/gemm/registry.cpp
	source/cpp/lib/oa/runtime/gemm/router.cpp
	source/cpp/lib/oa/runtime/gemm/tuner.cpp
	source/cpp/lib/oa/core/filesystem.cpp
	source/cpp/lib/oa/core/mappedFile.cpp
	source/cpp/lib/oa/core/perfStat.cpp
	source/cpp/lib/oa/core/perfStore.cpp
	source/cpp/lib/oa/core/cpuTopology.cpp
	source/cpp/lib/oa/core/threadPool.cpp
	source/cpp/lib/oa/core/simd.cpp
	source/cpp/lib/oa/crypto/hash.cpp
	source/cpp/lib/oa/crypto/keccak.cpp

	# ML — autograd (OaAutograd.md)
	source/cpp/lib/oa/ml/autograd/core/autogradCore.cpp
	source/cpp/lib/oa/ml/autograd/autogradAttach.gen.cpp

	# ML — utilities (data pipeline, encoding, checkpoints, .oam)
	# source/cpp/lib/oa/ml/adapterregistry.cpp  # stripped — apps construct OaModule directly
	# source/cpp/lib/oa/ml/modeladapter.cpp  # stripped — apps construct OaModule directly
	source/cpp/lib/oa/ml/checkpoint.cpp
	source/cpp/lib/oa/ml/budget.cpp
	source/cpp/lib/oa/ml/modelFile.cpp
	source/cpp/lib/oa/ml/optim/fnOptim.cpp
	source/cpp/lib/oa/ml/byte/encoder.cpp
	source/cpp/lib/oa/data/dataset.cpp
	source/cpp/lib/oa/data/dsSubset.cpp
	source/cpp/lib/oa/data/fnDataset.cpp

	# ML — I/O formats (SafeTensors, ONNX)
	source/cpp/lib/oa/ml/io/transferWeights.cpp
	source/cpp/lib/oa/ml/io/shardedWeightSource.cpp
	source/cpp/lib/oa/ml/io/safeTensorReader.cpp

	# Vulkan runtime
	source/cpp/lib/oa/runtime/oaVk.c
	source/cpp/lib/oa/runtime/oaVma.cpp
	source/cpp/lib/oa/runtime/device/device.cpp
	source/cpp/lib/oa/runtime/instance.cpp
	source/cpp/lib/oa/runtime/device/deviceUtils.cpp
	# Device modularization (PR-1)
	source/cpp/lib/oa/runtime/device/computeDevice.cpp
	source/cpp/lib/oa/runtime/device/renderDevice.cpp
	source/cpp/lib/oa/runtime/device/deviceBuilder.cpp
	source/cpp/lib/oa/runtime/device/features/coreFeatures.cpp
	source/cpp/lib/oa/runtime/device/features/mlFeatures.cpp
	source/cpp/lib/oa/runtime/device/features/visionFeatures.cpp
	source/cpp/lib/oa/runtime/device/features/audioFeatures.cpp
	source/cpp/lib/oa/runtime/device/features/renderFeatures.cpp
	source/cpp/lib/oa/runtime/allocator.cpp
	source/cpp/lib/oa/runtime/pipeline.cpp
	source/cpp/lib/oa/runtime/bindless.cpp
	source/cpp/lib/oa/runtime/cmd.cpp
	source/cpp/lib/oa/runtime/sync.cpp
	source/cpp/lib/oa/runtime/stream.cpp
	source/cpp/lib/oa/runtime/graphicsStream.cpp
	source/cpp/lib/oa/runtime/uploadRing.cpp
	source/cpp/lib/oa/runtime/timestamp.cpp
	source/cpp/lib/oa/runtime/timerRegistry.cpp
	source/cpp/lib/oa/runtime/timer.cpp
	source/cpp/lib/oa/runtime/dispatch.cpp
	source/cpp/lib/oa/runtime/imageDispatch.cpp
	source/cpp/lib/oa/runtime/texture.cpp
	source/cpp/lib/oa/runtime/semanticGraph.cpp
	source/cpp/lib/oa/runtime/executableGraph.cpp
	source/cpp/lib/oa/runtime/dnn.cpp
	source/cpp/lib/oa/runtime/dnn/graphLowering.cpp
	source/cpp/lib/oa/runtime/canary.cpp
	source/cpp/lib/oa/runtime/executionSessionCore.cpp
	source/cpp/lib/oa/runtime/executionSessionSubmit.cpp
	source/cpp/lib/oa/runtime/executableGraphBuilder.cpp
	source/cpp/lib/oa/runtime/executionSession.cpp
	source/cpp/lib/oa/runtime/executionPlan.cpp

	source/cpp/lib/oa/runtime/spirv/reflect.cpp
	source/cpp/lib/oa/runtime/engine/engine.cpp
	source/cpp/lib/oa/runtime/engine/presenter.cpp
	source/cpp/lib/oa/runtime/collective/hostCollectiveOracle.cpp
	source/cpp/lib/oa/runtime/externalMemory.cpp
	source/cpp/lib/oa/runtime/app.cpp
	source/cpp/lib/oa/runtime/window.cpp

	# ML — Matrix operations (oa::FnMatrix namespace extensions)
	source/cpp/lib/oa/ml/fnmatrix/blas/fnMatrixBlas.cpp
	source/cpp/lib/oa/ml/fnmatrix/norm/fnMatrixNorm.cpp
	source/cpp/lib/oa/ml/fnmatrix/norm/fnMatrixChannelNorm.cpp
	source/cpp/lib/oa/ml/fnmatrix/conv/fnMatrixConv1dGemm.cpp
	source/cpp/lib/oa/ml/fnmatrix/modules/fnMatrixModules.cpp
	source/cpp/lib/oa/ml/fnmatrix/pool/fnMatrixPool.cpp
	source/cpp/lib/oa/ml/fnmatrix/quant/fnMatrixQuant.cpp
	source/cpp/lib/oa/ml/fnmatrix/ssm/fnMatrixMamba3.cpp
	source/cpp/lib/oa/ml/fnmatrix/ssm/fnMatrixEmpyrealm.cpp
	source/cpp/lib/oa/ml/fnmatrix/vq/fnMatrixVq.cpp
	source/cpp/lib/oa/ml/fnmatrix/bmm/fnMatrixBmm.cpp
	source/cpp/lib/oa/ml/fnmatrix/attention/fnMatrixAttention.cpp
	source/cpp/lib/oa/ml/fnmatrix/attention/fnMatrixFlashAttention.cpp
	source/cpp/lib/oa/ml/fnmatrix/moe/fnMatrixMoe.cpp
	source/cpp/lib/oa/ml/fnmatrix/nn/fnMatrixGru.cpp
	source/cpp/lib/oa/ml/fnmatrix/nn/fnMatrixRnn.cpp
	source/cpp/lib/oa/ml/fnmatrix/backward/fnMatrixBackward.cpp
	# FnMatrixOptim.cpp removed - optimizer functions now in ml/optim/ (oa::FnOptim namespace)

	# ML — Loss functions (oa::FnLoss namespace) — all hand-written with fused kernels
	source/cpp/lib/oa/ml/fnloss/fnLoss.cpp

	# ML — reinforcement learning
	source/cpp/lib/oa/ml/environment.cpp
	source/cpp/lib/oa/ml/environmentExecution.cpp
	source/cpp/lib/oa/ml/actorCritic.cpp
	source/cpp/lib/oa/ml/rolloutCollector.cpp
	source/cpp/lib/oa/ml/dqnTrainer.cpp
	source/cpp/lib/oa/ml/advantage.cpp
	source/cpp/lib/oa/ml/policyEvaluator.cpp
	source/cpp/lib/oa/ml/itRolloutTraining.cpp
	source/cpp/lib/oa/ml/policy.cpp
	source/cpp/lib/oa/ml/ppoTrainer.cpp
	source/cpp/lib/oa/ml/replay.cpp
	source/cpp/lib/oa/ml/sacTrainer.cpp
	source/cpp/lib/oa/ml/fnEnvironment.cpp
	source/cpp/lib/oa/ml/fnloss/ppo.cpp
	source/cpp/lib/oa/ml/fnloss/dqn.cpp
	source/cpp/lib/oa/ml/fnloss/sac.cpp
	source/cpp/lib/oa/ml/rollout.cpp

	# ML — AUTOGENERATED from FnAutogen schemas
	${OA_ML_FN_MATRIX_GENERATED_SOURCES}

	# ML — nn modules (Vulkan dispatch)
	source/cpp/lib/oa/ml/nn/module.cpp
	source/cpp/lib/oa/ml/itTraining.cpp
	source/cpp/lib/oa/ml/trainingSession.cpp
	source/cpp/lib/oa/ml/mcpTraining.cpp
	source/cpp/lib/oa/ml/trainingProgram.cpp
	source/cpp/lib/oa/ml/fnflow/fnFlow.cpp
	source/cpp/lib/oa/ml/nn/flow/flowTimeEmbedding.cpp
	source/cpp/lib/oa/ml/nn/flow/flowTransformer.cpp
	source/cpp/lib/oa/ml/nn/flow/flowDenoiser.cpp
	source/cpp/lib/oa/ml/metric.cpp
	source/cpp/lib/oa/ml/tokenizer.cpp
	${OA_NN_GENERATED_SOURCES}
	source/cpp/lib/oa/ml/nn/dropout/dropout.cpp
	source/cpp/lib/oa/ml/nn/byte.cpp
	source/cpp/lib/oa/ml/nn/rope/rope.cpp
	source/cpp/lib/oa/ml/nn/mamba3/mamba3.cpp
	# Hand-written modules (NOT schema-generated) — must live here, not in
	# NnSources.gen.cmake, or a `--live` NN regen drops them from the build.
	source/cpp/lib/oa/ml/nn/ffn/ffn.cpp
	source/cpp/lib/oa/ml/nn/moe/moe.cpp
	source/cpp/lib/oa/ml/nn/gru/gru.cpp
	source/cpp/lib/oa/ml/nn/rnn/rnn.cpp
	source/cpp/lib/oa/ml/nn/transformer/transformer.cpp
	source/cpp/lib/oa/ml/nn/empyrealm/empyrealmCore.cpp
	source/cpp/lib/oa/ml/nn/empyrealm/empyrealmModule.cpp
	source/cpp/lib/oa/ml/nn/vq/vq.cpp
	# SeqVqTokenizer and ConvVqTokenizer — moved to Extensions/
	# Note: Sequential, Attention, Vision (BatchNorm2d, AdaptiveAvgPool2d), Upsample are now generated

	# Ui — widget layer, present, image loading
	source/cpp/lib/oa/ui/image.cpp
	source/cpp/lib/oa/ui/cv.cpp
	source/cpp/lib/oa/ui/detectionOverlay.cpp
	source/cpp/lib/oa/ui/canvas.cpp
	source/cpp/lib/oa/ui/input.cpp
	source/cpp/lib/oa/ui/style.cpp
	source/cpp/lib/oa/ui/text.cpp
	source/cpp/lib/oa/ui/ui.cpp
	source/cpp/lib/oa/ui/renderBackend.cpp
	source/cpp/lib/oa/ui/viewerPlatform.cpp
	source/cpp/lib/oa/ui/viewerApplication.cpp
	source/cpp/lib/oa/ui/viewerPresentation.cpp
	source/cpp/lib/oa/ui/viewport.cpp

	# Core — shared interaction/navigation primitives
	source/cpp/lib/oa/ui/platformInput.cpp
	source/cpp/lib/oa/ui/navigation.cpp
	# Ui — the single windowed/headless media viewer application
	source/cpp/lib/oa/ui/viewer.cpp
	source/cpp/lib/oa/ui/trainingViewer.cpp

	# Plot — matplotlib-style plotting over oa::ExecutionSession sinks
	source/cpp/lib/oa/ui/plot/axes.cpp
	source/cpp/lib/oa/ui/plot/figure.cpp

	# Audio
	source/cpp/lib/oa/audio/audioDecoder.cpp
	source/cpp/lib/oa/audio/audioEncoder.cpp
	source/cpp/lib/oa/audio/audioCapture.cpp
	source/cpp/lib/oa/audio/audioPlayer.cpp
	source/cpp/lib/oa/audio/fnaudio/fnAudio.cpp
	source/cpp/lib/oa/audio/fnaudio/signal/fnAudioSignal.cpp
	source/cpp/lib/oa/audio/fnaudio/transform/fnAudioTransform.cpp

	# Audio — AUTOGENERATED from FnAutogen schemas
	${OA_AUDIO_FN_AUDIO_GENERATED_SOURCES}

	# Render — reusable 3D render-to-texture infrastructure
	source/cpp/lib/oa/render/fnCamera.cpp
	source/cpp/lib/oa/render/fnMesh.cpp
	source/cpp/lib/oa/render/renderer.cpp
	source/cpp/lib/oa/render/renderer3d.cpp

	# Vision
	source/cpp/lib/oa/vision/detection.cpp
	source/cpp/lib/oa/vision/fndetection/fnDetection.cpp
	source/cpp/lib/oa/vision/fnimage/postprocess/fnImageSegmentation.cpp
	source/cpp/lib/oa/vision/fnImageSaveFile.cpp
	source/cpp/lib/oa/vision/image/codec/imageCodec.cpp
	source/cpp/lib/oa/vision/image/codec/imageDecoder.cpp
	source/cpp/lib/oa/vision/image/codec/imageEncoder.cpp
	source/cpp/lib/oa/vision/image/codec/stbImageCodec.cpp
	source/cpp/lib/oa/vision/image/codec/stbImageImpl.cpp
	source/cpp/lib/oa/vision/image/codec/webpImageCodec.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoder.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoderProfile.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoderOps.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoderInternal.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoderColor.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoderDecode.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoderFrame.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoderOperations.cpp
	source/cpp/lib/oa/vision/video/decoder/videoDecoderPreprocess.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/videoDecoderCodecAccess.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/videoDecoderRecordAccess.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/h264/videoDecoderH264.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/h264/videoDecoderH264Record.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/h265/videoDecoderH265.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/h265/videoDecoderH265Record.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/av1/videoDecoderAv1.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/av1/videoDecoderAv1Record.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/vp9/videoDecoderVp9.cpp
	source/cpp/lib/oa/vision/video/decoder/codec/vp9/videoDecoderVp9Record.cpp
	source/cpp/lib/oa/vision/video/encoder/videoEncoder.cpp
	source/cpp/lib/oa/vision/video/encoder/videoEncoderInternal.cpp
	source/cpp/lib/oa/vision/video/encoder/videoEncoderOperations.cpp
	source/cpp/lib/oa/vision/video/videoRecorder.cpp
	source/cpp/lib/oa/vision/capture/screenCapture.cpp
	source/cpp/lib/oa/vision/capture/cameraCapture.cpp
	source/cpp/lib/oa/vision/video/videoDemuxer.cpp
	source/cpp/lib/oa/vision/video/videoPlayer.cpp
	source/cpp/lib/oa/vision/video/videoMuxer.cpp

	# Vision — Video Codec Registry
	source/cpp/lib/oa/vision/video/codec/codecRegistry.cpp
	source/cpp/lib/oa/vision/video/codec/vcpH264.cpp
	source/cpp/lib/oa/vision/video/codec/vcpH265.cpp
	source/cpp/lib/oa/vision/video/codec/vcpAv1.cpp
	source/cpp/lib/oa/vision/video/codec/vcpVp9.cpp

	# Runtime — Vulkan Video Core (Layer 1)
	source/cpp/lib/oa/runtime/oaVkVideo.cpp

	# Vision Image Functions (GPU-only, pure Vulkan)
	source/cpp/lib/oa/vision/fnimage/fnImage.cpp
	source/cpp/lib/oa/vision/fnimage/fnImageConversion.cpp
	source/cpp/lib/oa/vision/fnimage/color/fnImagePixel.cpp
	source/cpp/lib/oa/vision/fnimage/geometric/fnImageGeometric.cpp
	source/cpp/lib/oa/vision/fnimage/filter/fnImageFilter.cpp
	source/cpp/lib/oa/vision/fnimage/fnImagePipeline.cpp

	# Vision video functions (stateless oa::FnVideo frame/NAL transformations)
	source/cpp/lib/oa/vision/fnvideo/nal/fnVideoNal.cpp
	source/cpp/lib/oa/vision/fnvideo/color/fnVideoColor.cpp

	# Vision — AUTOGENERATED from FnAutogen schemas
	${OA_VISION_FN_IMAGE_GENERATED_SOURCES}
	${OA_VISION_FN_VIDEO_GENERATED_SOURCES}

	source/cpp/lib/oa/ml/nn/byte.cpp

	# ML — optimizers
	source/cpp/lib/oa/ml/optim/optimizer.cpp
	source/cpp/lib/oa/ml/optim/sgd.cpp
	source/cpp/lib/oa/ml/optim/adam.cpp
	source/cpp/lib/oa/ml/optim/adamW.cpp
	source/cpp/lib/oa/ml/optim/muon.cpp
	source/cpp/lib/oa/ml/callback/lrScheduler.cpp



	# Network
	source/cpp/lib/oa/mcp/mcp.cpp
	source/cpp/lib/oa/network/tcp.cpp
	source/cpp/lib/oa/network/tcpFramed.cpp
	source/cpp/lib/oa/network/satelliteProtocol.cpp
	source/cpp/lib/oa/network/satelliteSession.cpp
)

if(OA_BUILD_CRYPTO)
	list(APPEND OA_SOURCES
		# Single-shot CPU keygen/sign/verify (liboqs)
		source/cpp/lib/oa/crypto/sign.cpp

		# oa::FnHash — GPU batch hash (auto-context); CPU primitives live in
		# Crypto/Keccak.cpp + Crypto/Hash.cpp (compiled unconditionally above).
		${OA_CRYPTO_HASH_FN_GENERATED_SOURCES}
		source/cpp/lib/oa/crypto/hash/fnHash.cpp

	)
endif()

# UI — AUTOGENERATED from FnAutogen schemas
list(APPEND OA_SOURCES
	${OA_UI_FN_MATRIX_GENERATED_SOURCES}
)

# Android ships the actual OA compute/ML stack, not the desktop product shell.
# Keep this as a source boundary instead of scattering __ANDROID__ through UI,
# capture, codec, and networking implementations.
if(OA_ANDROID_ML)
	list(FILTER OA_SOURCES EXCLUDE REGEX "/(audio|vision|ui|render|animation|network|crypto)/")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/data/")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/core/(simd|cpuTopology|perfStat|perfStore|mappedFile)[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/runtime/(app|window|imageDispatch|oaVkVideo)[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/runtime/engine/graphicsEngine[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/runtime/externalMemory[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/ml/(tokenizer|io|data)/")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/ml/(oad|mamba3)[.]cpp$")
	target_compile_definitions(oa_api INTERFACE OA_ANDROID_ML=1)
	message(STATUS "OA source profile: Android Runtime + Core + ML")
endif()

# SHARED when building Python bindings (avoids TLS relocation issues) or when explicitly
# requested for a distributable runtime package (OA_BUILD_SHARED); STATIC otherwise.
if(OA_BUILD_PYTHON OR OA_BUILD_SHARED)
	add_library(oa SHARED ${OA_SOURCES})
else()
	add_library(oa STATIC ${OA_SOURCES})
endif()
target_link_libraries(oa PUBLIC oa_headers)
if(ANDROID)
	target_include_directories(oa_api SYSTEM INTERFACE ${OA_ANDROID_VULKAN_HEADERS})
endif()
if(WIN32)
	target_link_libraries(oa PUBLIC ws2_32 PRIVATE bcrypt)
endif()
target_include_directories(oa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/thirdparty/miniaudio)
target_include_directories(oa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib)

# Highway SIMD
if(hwy_FOUND AND NOT ANDROID)
	target_link_libraries(oa PRIVATE hwy::hwy)
	message(STATUS "Highway SIMD: found")
elseif(NOT ANDROID)
	message(WARNING "Highway not found — OaSimd* functions will not compile")
endif()

if(NOT ANDROID)
	target_link_libraries(oa PRIVATE utf8proc::utf8proc)
	target_link_libraries(oa PRIVATE harfbuzz::harfbuzz)
endif()
get_target_property(_oa_library_type oa TYPE)
if(UNIX AND NOT APPLE AND _oa_library_type STREQUAL "SHARED_LIBRARY")
	# OA's private static text/image dependencies must not interpose the system
	# copies used by plug-ins loaded later by the host process (notably
	# libdecor's GTK/Pango stack). Localize the named C archives only; hiding
	# every archive also localizes C++ template ownership machinery and can cross
	# allocator boundaries at process teardown.
	target_link_options(oa PRIVATE
		"LINKER:--exclude-libs=libpng16.a:libharfbuzz.a:libfreetype.a:libz.a:libbz2.a:libbrotlidec.a:libbrotlicommon.a")
endif()
set(OA_PACKAGE_NEEDS_UTF8PROC OFF)
set(OA_PACKAGE_NEEDS_HARFBUZZ OFF)
if(NOT ANDROID AND _oa_library_type STREQUAL "STATIC_LIBRARY")
	set(OA_PACKAGE_NEEDS_UTF8PROC ON)
	set(OA_PACKAGE_NEEDS_HARFBUZZ ON)
endif()

# Still-image WebP is an optional internal codec backend. Desktop vcpkg
# manifests install libwebp; Linux developer builds may use pkg-config. Android
# keeps the stb JPEG/PNG/BMP/TGA slice and reports WebP unavailable.
set(OA_WEBP_DEPENDENCY_MODE "None")
if(NOT ANDROID)
	find_package(WebP CONFIG QUIET)
	if(TARGET WebP::webp)
		target_link_libraries(oa PRIVATE WebP::webp)
		target_compile_definitions(oa PRIVATE OA_HAS_WEBP=1)
		if(_oa_library_type STREQUAL "STATIC_LIBRARY")
			set(OA_WEBP_DEPENDENCY_MODE "CMake")
		endif()
		message(STATUS "Still-image WebP: libwebp")
	elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
		find_package(PkgConfig QUIET)
		if(PkgConfig_FOUND)
			pkg_check_modules(OA_WEBP QUIET IMPORTED_TARGET libwebp)
		endif()
		if(TARGET PkgConfig::OA_WEBP)
			target_link_libraries(oa PRIVATE PkgConfig::OA_WEBP)
			target_compile_definitions(oa PRIVATE OA_HAS_WEBP=1)
			if(_oa_library_type STREQUAL "STATIC_LIBRARY")
				set(OA_WEBP_DEPENDENCY_MODE "PkgConfig")
			endif()
			message(STATUS "Still-image WebP: libwebp (pkg-config)")
		else()
			target_compile_definitions(oa PRIVATE OA_HAS_WEBP=0)
			message(STATUS "Still-image WebP: unavailable")
		endif()
	else()
		target_compile_definitions(oa PRIVATE OA_HAS_WEBP=0)
		message(STATUS "Still-image WebP: unavailable")
	endif()
else()
	target_compile_definitions(oa PRIVATE OA_HAS_WEBP=0)
endif()

# SDL3 — required by oa::CameraCapture, oa::Window, and oa::Viewer
set(OA_PACKAGE_NEEDS_SDL3 OFF)
if(NOT ANDROID)
	find_package(SDL3 CONFIG QUIET)
endif()
if(TARGET SDL3::SDL3)
	target_link_libraries(oa PRIVATE SDL3::SDL3)
	if(_oa_library_type STREQUAL "STATIC_LIBRARY")
		set(OA_PACKAGE_NEEDS_SDL3 ON)
	endif()
	message(STATUS "SDL3: found (OaCameraCapture enabled)")
elseif(NOT ANDROID)
	message(WARNING "SDL3 not found — OaCameraCapture and windowed apps will not work")
endif()

# Wayland screencast portal + PipeWire. Optional outside Linux; the public
# OaScreenCapture surface remains available and reports Unavailable at runtime.
set(OA_PACKAGE_NEEDS_PIPEWIRE_CAPTURE OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	find_package(PkgConfig QUIET)
	if(PkgConfig_FOUND)
		pkg_check_modules(OA_PIPEWIRE QUIET IMPORTED_TARGET libpipewire-0.3)
		pkg_check_modules(OA_LIBPORTAL QUIET IMPORTED_TARGET libportal)
	endif()
	if(TARGET PkgConfig::OA_PIPEWIRE AND TARGET PkgConfig::OA_LIBPORTAL)
		target_link_libraries(oa PRIVATE PkgConfig::OA_PIPEWIRE PkgConfig::OA_LIBPORTAL)
		target_compile_definitions(oa PRIVATE OA_HAS_PIPEWIRE_SCREEN_CAPTURE=1)
		if(_oa_library_type STREQUAL "STATIC_LIBRARY")
			set(OA_PACKAGE_NEEDS_PIPEWIRE_CAPTURE ON)
		endif()
		message(STATUS "Wayland screen capture: libportal + PipeWire")
	else()
		message(STATUS "Wayland screen capture: unavailable (libportal/PipeWire not found)")
	endif()
endif()

# Vulkan: OaVk (forked volk) handles dynamic loading via dlopen — no static linking.
target_link_libraries(oa PRIVATE ${CMAKE_DL_LIBS})
if(TARGET Vulkan::Headers)
	target_link_libraries(oa PUBLIC Vulkan::Headers)
elseif(Vulkan_FOUND)
	target_include_directories(oa PUBLIC ${Vulkan_INCLUDE_DIRS})
endif()

# PQCrypto (liboqs — ML-DSA-65 / Dilithium-3)
if(OA_BUILD_CRYPTO)
	find_package(liboqs CONFIG REQUIRED)
	target_link_libraries(oa PRIVATE OQS::oqs)
	target_compile_definitions(oa PUBLIC OA_BUILD_CRYPTO)
	message(STATUS "Building with ML-DSA-65 signing (liboqs)")
endif()

# COMPUTE SHADERS — Slang → SPIR-V (ML primitives + crypto kernels)
if(SLANGC)
	set(OA_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/ml/shader/compute")
	set(OA_SPIRV_DIR "${CMAKE_CURRENT_BINARY_DIR}/spirv")
	file(MAKE_DIRECTORY ${OA_SPIRV_DIR})

	# Slang importable math modules (no entry points) + Public/Core/Math.slang umbrella
	set(OA_SLANG_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/math")
	set(OA_KERNEL_FLAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/flat")
	set(OA_KERNEL_MATRIX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/matrix")
	set(OA_KERNEL_BLAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/blas")
	set(OA_KERNEL_ML_BLAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/ml/shader/compute/blas")
	set(OA_KERNEL_SSM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/ml/shader/compute/nn/ssm")
	set(OA_SLANG_UMBRELLA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/include/oa/core")
	set(OA_ML_SLANG_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/ml/shader/compute/nn/math")
	set(OA_ML_SLANG_UMBRELLA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/include/oa/ml")
	file(GLOB OA_SLANG_MODULES "${OA_SLANG_MODULE_DIR}/*.slang")
	list(APPEND OA_SLANG_MODULES "${OA_SLANG_UMBRELLA_DIR}/math.slang")
	file(GLOB OA_ML_SLANG_MODULES "${OA_ML_SLANG_MODULE_DIR}/*.slang")
	list(APPEND OA_SLANG_MODULES ${OA_ML_SLANG_MODULES})
	list(APPEND OA_SLANG_MODULES "${OA_ML_SLANG_UMBRELLA_DIR}/math.slang")

	# Standalone fixed kernels are generated from tools/kernelAutogen/schema.toml.
	# FnAutogen and OaTile compile their own operation and GEMM families below;
	# OA_ALL_SPV_OUTPUTS composes all three into the exact embedded manifest.
	set(OA_ML_SPV_OUTPUTS "")
	list(LENGTH OA_STANDALONE_KERNEL_NAMES _OA_STANDALONE_KERNEL_COUNT)
	foreach(_FIELD SOURCES PROFILES STAGES CONDITIONS)
		list(LENGTH OA_STANDALONE_KERNEL_${_FIELD} _OA_STANDALONE_FIELD_COUNT)
		if(NOT _OA_STANDALONE_FIELD_COUNT EQUAL _OA_STANDALONE_KERNEL_COUNT)
			message(FATAL_ERROR
				"KernelManifest.gen.cmake field ${_FIELD} has ${_OA_STANDALONE_FIELD_COUNT} "
				"rows; expected ${_OA_STANDALONE_KERNEL_COUNT}")
		endif()
	endforeach()
	foreach(_I RANGE 1 ${_OA_STANDALONE_KERNEL_COUNT})
		math(EXPR _KERNEL_INDEX "${_I} - 1")
		list(GET OA_STANDALONE_KERNEL_NAMES ${_KERNEL_INDEX} _NAME)
		list(GET OA_STANDALONE_KERNEL_SOURCES ${_KERNEL_INDEX} _SOURCE_REL)
		list(GET OA_STANDALONE_KERNEL_PROFILES ${_KERNEL_INDEX} _PROFILE)
		list(GET OA_STANDALONE_KERNEL_STAGES ${_KERNEL_INDEX} _STAGE)
		list(GET OA_STANDALONE_KERNEL_CONDITIONS ${_KERNEL_INDEX} _CONDITION)
		if(_CONDITION STREQUAL "crypto" AND NOT OA_BUILD_CRYPTO)
			continue()
		endif()

		set(_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE_REL}")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		get_filename_component(_SPV_PARENT "${_SPV}" DIRECTORY)
		set(_KERNEL_ARGS
			-I ${OA_SLANG_UMBRELLA_DIR}
			-I ${OA_SLANG_MODULE_DIR})
		set(_KERNEL_DEPS "")
		if(_PROFILE STREQUAL "ssm")
			list(APPEND _KERNEL_ARGS
				-I ${OA_ML_SLANG_UMBRELLA_DIR}
				-I ${OA_ML_SLANG_MODULE_DIR})
		elseif(_PROFILE STREQUAL "vision")
			set(_VISION_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/vision/shader/compute")
			list(APPEND _KERNEL_ARGS
				-I ${_VISION_SHADER_DIR}/conversion
				-I ${_VISION_SHADER_DIR}/math)
			list(APPEND _KERNEL_DEPS
				${_VISION_SHADER_DIR}/conversion/rgbToYuv.slang
				${_VISION_SHADER_DIR}/math/image.slang)
		endif()
		if(NOT _PROFILE STREQUAL "ssm")
			list(APPEND _KERNEL_ARGS -target spirv)
			if(NOT _STAGE STREQUAL "compute")
				list(APPEND _KERNEL_ARGS -stage ${_STAGE})
			endif()
			list(APPEND _KERNEL_ARGS -entry main)
		endif()
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${_SPV_PARENT}
			COMMAND ${SLANGC} ${_SRC} ${_KERNEL_ARGS} -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES} ${_KERNEL_DEPS}
			COMMENT "slangc: ${_SOURCE_REL} → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	function(oa_compile_slang TARGET SOURCE)
		get_filename_component(NAME ${SOURCE} NAME_WE)
		set(SPV "${OA_SPIRV_DIR}/${NAME}.spv")
		add_custom_command(
			OUTPUT ${SPV}
			COMMAND ${SLANGC} ${SOURCE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${SPV}
			DEPENDS ${SOURCE} ${OA_SLANG_MODULES}
			COMMENT "slangc: ${NAME}.slang → ${NAME}.spv"
		)
		target_sources(${TARGET} PRIVATE ${SPV})
	endfunction()

	# Operation-owned ML kernels are emitted by FnAutogen from [ops.kernel].
	# Standalone fixed kernels were already compiled from KernelManifest.gen.cmake.
	set(OA_ML_SHADER_REG ${OA_FN_AUTOGEN_ML_KERNEL_NAMES})
	set(OA_ML_SHADER_SRC ${OA_FN_AUTOGEN_ML_KERNEL_SOURCES})
	list(LENGTH OA_ML_SHADER_REG _OA_ML_REG_N)
	list(LENGTH OA_ML_SHADER_SRC _OA_ML_SRC_N)
	if(NOT _OA_ML_REG_N EQUAL _OA_ML_SRC_N)
		message(FATAL_ERROR "OA_ML_SHADER_REG (${_OA_ML_REG_N}) and OA_ML_SHADER_SRC (${_OA_ML_SRC_N}) length mismatch! "
			"These lists must be kept in 1:1 order. "
			"Last REG entry: ${OA_ML_SHADER_REG}; "
			"Last SRC entry: ${OA_ML_SHADER_SRC}")
	endif()
	list(LENGTH OA_ML_SHADER_REG _OA_ML_N)
	math(EXPR _OA_ML_LAST "${_OA_ML_N} - 1")
	foreach(_I RANGE ${_OA_ML_LAST})
		list(GET OA_ML_SHADER_REG ${_I} _REG)
		list(GET OA_ML_SHADER_SRC ${_I} _SRCREL)
		if(_SRCREL MATCHES "^ssm/")
			string(REGEX REPLACE "^ssm/" "" _SSM_SRCREL "${_SRCREL}")
			set(_SRC "${OA_KERNEL_SSM_DIR}/${_SSM_SRCREL}.slang")
			set(_SPV "${OA_SPIRV_DIR}/${_REG}.spv")
			add_custom_command(
				OUTPUT ${_SPV}
				COMMAND ${SLANGC} ${SLANGC_FLAGS} ${_SRC}
					-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
					-I ${OA_ML_SLANG_UMBRELLA_DIR} -I ${OA_ML_SLANG_MODULE_DIR}
					-o ${_SPV}
				DEPENDS ${_SRC} ${OA_SLANG_MODULES}
				COMMENT "slangc: Nn/Ssm/${_SSM_SRCREL}.slang → ${_REG}.spv"
			)
			list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
			continue()
		elseif(_SRCREL MATCHES "^flat/")
			string(REGEX REPLACE "^flat/" "" _FLAT_SRCREL "${_SRCREL}")
			# Prefer .gen.slang (auto-generated) over manual .slang
			if(EXISTS "${OA_KERNEL_FLAT_DIR}/${_FLAT_SRCREL}.gen.slang")
				set(_SRC "${OA_KERNEL_FLAT_DIR}/${_FLAT_SRCREL}.gen.slang")
			else()
				set(_SRC "${OA_KERNEL_FLAT_DIR}/${_FLAT_SRCREL}.slang")
			endif()
		else()
			set(_SRC "${OA_SHADER_DIR}/${_SRCREL}.slang")
		endif()
		# Preserve the logical registry name in the output path. Namespaced kernels
		# such as Quantize/QuantizeQ4 must not be flattened: the embedded registry
		# is the runtime authority and oavk::findSpirv consumes this exact name.
		set(_SPV "${OA_SPIRV_DIR}/${_REG}.spv")
		get_filename_component(_SPV_DIR "${_SPV}" DIRECTORY)
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${_SPV_DIR}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: ${_SRCREL}.slang → ${_REG}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# ═══════════════════════════════════════════════════════════════════════
	# BLAS Kernels — Unified GEMM/BLAS directory structure
	# ═══════════════════════════════════════════════════════════════════════

	# Generated from the OaTile schemas. Base owns the correctness and CoopVec
	# routes; FP32 and BF16 own their bounded specialization lattices. These files
	# drive compilation from the same records that emit fixed IDs and routing rows.
	include("${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/blas/tile/oaTileBaseVariants.gen.cmake")
	include("${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/blas/tile/oaTileFp32Variants.gen.cmake")
	include("${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/blas/tile/oaTileSmallMVariants.gen.cmake")
	include("${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/core/shader/compute/blas/tile/oaTileBf16Variants.gen.cmake")

	set(OA_BLAS_GLSL_COMP_KERNELS "")
	set(OA_ML_BLAS_GLSL_COMP_KERNELS "")

	# Compile core GEMM kernels. Logical registry names are stable PascalCase
	# identities; source paths follow the repository's camelCase layout.
	list(LENGTH OA_BLAS_GEMM_KERNEL_NAMES _OA_BASE_GEMM_LEN)
	math(EXPR _OA_BASE_GEMM_LAST "${_OA_BASE_GEMM_LEN} - 1")
	foreach(_I RANGE ${_OA_BASE_GEMM_LAST})
		list(GET OA_BLAS_GEMM_KERNEL_NAMES ${_I} _NAME)
		list(GET OA_BLAS_GEMM_KERNEL_SOURCES ${_I} _KERNEL)
		set(_SRC "${OA_KERNEL_BLAS_DIR}/${_KERNEL}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")

		# Determine if cooperative matrix capability is needed
		if(_NAME MATCHES "CoopMat")
			add_custom_command(
				OUTPUT ${_SPV}
				COMMAND ${SLANGC} ${_SRC}
					-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
					-target spirv -entry main
					-capability spvCooperativeMatrixKHR
					-capability spvCooperativeMatrixConversionsNV
					-o ${_SPV}
				DEPENDS ${_SRC} ${OA_SLANG_MODULES}
				COMMENT "slangc: Blas/${_KERNEL}.slang → ${_NAME}.spv"
			)
		else()
			add_custom_command(
				OUTPUT ${_SPV}
				COMMAND ${SLANGC} ${_SRC}
					-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
					-target spirv -entry main -o ${_SPV}
				DEPENDS ${_SRC} ${OA_SLANG_MODULES}
				COMMENT "slangc: Blas/${_KERNEL}.slang → ${_NAME}.spv"
			)
		endif()
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	set(_GEMM_FP32_TEMPLATE "${OA_KERNEL_BLAS_DIR}/gemm/gemmTiled.slang")
	list(LENGTH OA_TILE_FP32_NAMES _OA_FP32_LEN)
	set(_OA_FP32_I 0)
	while(_OA_FP32_I LESS _OA_FP32_LEN)
		list(GET OA_TILE_FP32_NAMES ${_OA_FP32_I} _FP32_NAME)
		list(GET OA_TILE_FP32_TILE_M ${_OA_FP32_I} _FP32_TILE_M)
		list(GET OA_TILE_FP32_TILE_N ${_OA_FP32_I} _FP32_TILE_N)
		list(GET OA_TILE_FP32_TILE_K ${_OA_FP32_I} _FP32_TILE_K)
		list(GET OA_TILE_FP32_THREAD_M ${_OA_FP32_I} _FP32_THREAD_M)
		list(GET OA_TILE_FP32_THREAD_N ${_OA_FP32_I} _FP32_THREAD_N)
		list(GET OA_TILE_FP32_WORKGROUP ${_OA_FP32_I} _FP32_WORKGROUP)
		list(GET OA_TILE_FP32_GROUP_M ${_OA_FP32_I} _FP32_GROUP_M)
		list(GET OA_TILE_FP32_STRIDED ${_OA_FP32_I} _FP32_STRIDED)
		list(GET OA_TILE_FP32_EPILOGUE ${_OA_FP32_I} _FP32_EPILOGUE)
		set(_FP32_SPV "${OA_SPIRV_DIR}/${_FP32_NAME}.spv")
		add_custom_command(
			OUTPUT ${_FP32_SPV}
			COMMAND ${SLANGC} ${_GEMM_FP32_TEMPLATE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-D OA_TILE_M=${_FP32_TILE_M}
				-D OA_TILE_N=${_FP32_TILE_N}
				-D OA_TILE_K=${_FP32_TILE_K}
				-D OA_THREAD_TILE_M=${_FP32_THREAD_M}
				-D OA_THREAD_TILE_N=${_FP32_THREAD_N}
				-D OA_TILE_WORKGROUP=${_FP32_WORKGROUP}
				-D OA_TILE_GROUP_M=${_FP32_GROUP_M}
				-D OA_TILE_STRIDED=${_FP32_STRIDED}
				-D OA_TILE_EPILOGUE=${_FP32_EPILOGUE}
				-target spirv -entry main -o ${_FP32_SPV}
			DEPENDS ${_GEMM_FP32_TEMPLATE} ${OA_SLANG_MODULES}
			COMMENT "slangc: GemmTiled.slang -> ${_FP32_NAME}.spv (${_FP32_TILE_M}x${_FP32_TILE_N}x${_FP32_TILE_K}, thread ${_FP32_THREAD_M}x${_FP32_THREAD_N}, group-M ${_FP32_GROUP_M}, strided ${_FP32_STRIDED}, epilogue ${_FP32_EPILOGUE})"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_FP32_SPV})
		math(EXPR _OA_FP32_I "${_OA_FP32_I} + 1")
	endwhile()

	set(_GEMM_SMALL_M_TEMPLATE "${OA_KERNEL_BLAS_DIR}/gemm/gemmSmallM.slang")
	list(LENGTH OA_TILE_SMALL_M_NAMES _OA_SMALL_M_LEN)
	set(_OA_SMALL_M_I 0)
	while(_OA_SMALL_M_I LESS _OA_SMALL_M_LEN)
		list(GET OA_TILE_SMALL_M_NAMES ${_OA_SMALL_M_I} _SMALL_M_NAME)
		list(GET OA_TILE_SMALL_M_OUTPUTS ${_OA_SMALL_M_I} _SMALL_M_OUTPUTS)
		list(GET OA_TILE_SMALL_M_REDUCTION ${_OA_SMALL_M_I} _SMALL_M_REDUCTION)
		list(GET OA_TILE_SMALL_M_WORKGROUP ${_OA_SMALL_M_I} _SMALL_M_WORKGROUP)
		list(GET OA_TILE_SMALL_M_EPILOGUE ${_OA_SMALL_M_I} _SMALL_M_EPILOGUE)
		set(_SMALL_M_SPV "${OA_SPIRV_DIR}/${_SMALL_M_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SMALL_M_SPV}
			COMMAND ${SLANGC} ${_GEMM_SMALL_M_TEMPLATE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-D OA_SMALL_M_OUTPUTS=${_SMALL_M_OUTPUTS}
				-D OA_SMALL_M_REDUCTION=${_SMALL_M_REDUCTION}
				-D OA_SMALL_M_WORKGROUP=${_SMALL_M_WORKGROUP}
				-D OA_SMALL_M_EPILOGUE=${_SMALL_M_EPILOGUE}
				-target spirv -entry main -o ${_SMALL_M_SPV}
			DEPENDS ${_GEMM_SMALL_M_TEMPLATE} ${OA_SLANG_MODULES}
			COMMENT "slangc: GemmSmallM.slang -> ${_SMALL_M_NAME}.spv (outputs ${_SMALL_M_OUTPUTS}, reduction ${_SMALL_M_REDUCTION}, epilogue ${_SMALL_M_EPILOGUE})"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SMALL_M_SPV})
		math(EXPR _OA_SMALL_M_I "${_OA_SMALL_M_I} + 1")
	endwhile()

	# GemmCmSgBf16 variants from one template source.
	set(_GEMM_BF16_TEMPLATE "${OA_KERNEL_BLAS_DIR}/gemm/gemmCmSgBf16.slang")
	list(LENGTH OA_GEMM_BF16_VARIANTS _OA_BF16_LEN)
	math(EXPR _OA_BF16_LAST "${_OA_BF16_LEN} - 1")
	set(_OA_BF16_I 0)
	while(_OA_BF16_I LESS _OA_BF16_LAST)
		list(GET OA_GEMM_BF16_VARIANTS ${_OA_BF16_I} _BF16_NAME)
		math(EXPR _OA_BF16_J "${_OA_BF16_I} + 1")
		list(GET OA_GEMM_BF16_VARIANTS ${_OA_BF16_J} _BF16_ACT)
		set(_BF16_SPV "${OA_SPIRV_DIR}/${_BF16_NAME}.spv")
		add_custom_command(
			OUTPUT ${_BF16_SPV}
			COMMAND ${SLANGC} ${_GEMM_BF16_TEMPLATE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-D OA_ACT=${_BF16_ACT}
				-target spirv -entry main
				-capability spvCooperativeMatrixKHR
				-capability spvCooperativeMatrixConversionsNV
				-o ${_BF16_SPV}
			DEPENDS ${_GEMM_BF16_TEMPLATE} ${OA_SLANG_MODULES}
			COMMENT "slangc: GemmCmSgBf16.slang → ${_BF16_NAME}.spv (OA_ACT=${_BF16_ACT})"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_BF16_SPV})
		math(EXPR _OA_BF16_I "${_OA_BF16_I} + 2")
	endwhile()

	# GemmCmWgBf16 variants from one template source.
	set(_GEMM_WG_BF16_TEMPLATE "${OA_KERNEL_BLAS_DIR}/gemm/gemmCmWgBf16.slang")
	list(LENGTH OA_GEMM_WG_BF16_VARIANTS _OA_WG_BF16_LEN)
	math(EXPR _OA_WG_BF16_LAST "${_OA_WG_BF16_LEN} - 1")
	set(_OA_WG_BF16_I 0)
	while(_OA_WG_BF16_I LESS ${_OA_WG_BF16_LAST})
		list(GET OA_GEMM_WG_BF16_VARIANTS ${_OA_WG_BF16_I} _WG_BF16_NAME)
		math(EXPR _OA_WG_BF16_J "${_OA_WG_BF16_I} + 1")
		list(GET OA_GEMM_WG_BF16_VARIANTS ${_OA_WG_BF16_J} _WG_BF16_ACT)
		set(_WG_BF16_SPV "${OA_SPIRV_DIR}/${_WG_BF16_NAME}.spv")
		add_custom_command(
			OUTPUT ${_WG_BF16_SPV}
			COMMAND ${SLANGC} ${_GEMM_WG_BF16_TEMPLATE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-D OA_ACT=${_WG_BF16_ACT}
				-target spirv -entry main
				-capability spvCooperativeMatrixKHR
				-capability spvCooperativeMatrixConversionsNV
				-o ${_WG_BF16_SPV}
			DEPENDS ${_GEMM_WG_BF16_TEMPLATE} ${OA_SLANG_MODULES}
			COMMENT "slangc: GemmCmWgBf16.slang → ${_WG_BF16_NAME}.spv (OA_ACT=${_WG_BF16_ACT})"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_WG_BF16_SPV})
		math(EXPR _OA_WG_BF16_I "${_OA_WG_BF16_I} + 2")
	endwhile()

	# Compile GLSL .comp kernels.
	#
	# Raw GEMMs live in source/cpp/lib/oa/core/shader/compute/blas; fused NN-epilogue
	# variants live in source/cpp/lib/oa/ml/shader/compute/blas to mirror the .slang
	# fused-kernel layout.
	foreach(_KERNEL ${OA_BLAS_GLSL_COMP_KERNELS})
		get_filename_component(_NAME ${_KERNEL} NAME)
		set(_SRC "${OA_KERNEL_BLAS_DIR}/${_KERNEL}.comp")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.4 ${_SRC}
				-o ${_SPV}
			DEPENDS ${_SRC}
			COMMENT "glslangValidator: Blas/${_KERNEL}.comp → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()
	foreach(_KERNEL ${OA_ML_BLAS_GLSL_COMP_KERNELS})
		get_filename_component(_NAME ${_KERNEL} NAME)
		set(_SRC "${OA_KERNEL_ML_BLAS_DIR}/${_KERNEL}.comp")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.4 ${_SRC}
				-o ${_SPV}
			DEPENDS ${_SRC}
			COMMENT "glslangValidator: Ml/Shader/Compute/Blas/${_KERNEL}.comp → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Compile utility kernels (GemmCoopVec needs CooperativeVector).
	list(LENGTH OA_BLAS_UTIL_KERNEL_NAMES _OA_BASE_UTIL_LEN)
	math(EXPR _OA_BASE_UTIL_LAST "${_OA_BASE_UTIL_LEN} - 1")
	foreach(_I RANGE ${_OA_BASE_UTIL_LAST})
		list(GET OA_BLAS_UTIL_KERNEL_NAMES ${_I} _NAME)
		list(GET OA_BLAS_UTIL_KERNEL_SOURCES ${_I} _KERNEL)
		set(_SRC "${OA_KERNEL_BLAS_DIR}/${_KERNEL}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main
				-capability spvCooperativeVectorNV
				-o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Blas/${_KERNEL}.slang → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Operation-owned Vision kernels are emitted by FnAutogen.
	set(OA_KERNEL_VISION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/vision/shader/compute")
	list(LENGTH OA_FN_AUTOGEN_VISION_KERNEL_NAMES _OA_VISION_NAME_COUNT)
	list(LENGTH OA_FN_AUTOGEN_VISION_KERNEL_SOURCES _OA_VISION_SOURCE_COUNT)
	if(NOT _OA_VISION_NAME_COUNT EQUAL _OA_VISION_SOURCE_COUNT)
		message(FATAL_ERROR "FnAutogen Vision kernel name/source count mismatch")
	endif()
	math(EXPR _OA_VISION_LAST "${_OA_VISION_NAME_COUNT} - 1")
	foreach(_I RANGE ${_OA_VISION_LAST})
		list(GET OA_FN_AUTOGEN_VISION_KERNEL_NAMES ${_I} _NAME)
		list(GET OA_FN_AUTOGEN_VISION_KERNEL_SOURCES ${_I} _SOURCE_REL)
		set(_SRC "${OA_KERNEL_VISION_DIR}/${_SOURCE_REL}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-I ${OA_KERNEL_VISION_DIR}/conversion
				-I ${OA_KERNEL_VISION_DIR}/math
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
				${OA_KERNEL_VISION_DIR}/conversion/rgbToYuv.slang
				${OA_KERNEL_VISION_DIR}/math/image.slang
			COMMENT "slangc: Vision/Shader/Compute/${_SOURCE_REL}.slang → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Ui blit shaders (compose image → swapchain)
	set(OA_KERNEL_UI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib/oa/ui/shader/compute")
	list(LENGTH OA_FN_AUTOGEN_UI_KERNEL_NAMES _OA_UI_NAME_COUNT)
	list(LENGTH OA_FN_AUTOGEN_UI_KERNEL_SOURCES _OA_UI_SOURCE_COUNT)
	if(NOT _OA_UI_NAME_COUNT EQUAL _OA_UI_SOURCE_COUNT)
		message(FATAL_ERROR "FnAutogen Ui kernel name/source count mismatch")
	endif()
	math(EXPR _OA_UI_LAST "${_OA_UI_NAME_COUNT} - 1")
	foreach(_I RANGE ${_OA_UI_LAST})
		list(GET OA_FN_AUTOGEN_UI_KERNEL_NAMES ${_I} _NAME)
		list(GET OA_FN_AUTOGEN_UI_KERNEL_SOURCES ${_I} _SOURCE_REL)
		set(_SRC "${OA_KERNEL_UI_DIR}/${_SOURCE_REL}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Ui/Shader/Compute/${_SOURCE_REL}.slang → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	set(OA_ALL_SPV_OUTPUTS ${OA_ML_SPV_OUTPUTS})
	list(APPEND OA_ALL_SPV_OUTPUTS ${OA_BLAS_SPV_OUTPUTS})
	list(SORT OA_ALL_SPV_OUTPUTS)

	# SPIR-V EMBEDDING — bake every declared shader output into liboa.a. The
	# output path relative to OA_SPIRV_DIR is the exact runtime lookup key. There
	# is deliberately no second hand-assembled name list: if a shader compiles,
	# it is embedded under that exact logical name or configuration fails.
	set(OA_ALL_SPV_NAMES "")
	foreach(_SPV IN LISTS OA_ALL_SPV_OUTPUTS)
		file(RELATIVE_PATH _SPV_REL "${OA_SPIRV_DIR}" "${_SPV}")
		if(_SPV_REL MATCHES "^\\.\\./" OR IS_ABSOLUTE "${_SPV_REL}")
			message(FATAL_ERROR
				"Shader output is outside OA_SPIRV_DIR and cannot define an embedded key: ${_SPV}")
		endif()
		if(NOT _SPV_REL MATCHES "[.]spv$")
			message(FATAL_ERROR "Shader output does not end in .spv: ${_SPV}")
		endif()
		string(REGEX REPLACE "[.]spv$" "" _SPV_NAME "${_SPV_REL}")
		if(_SPV_NAME IN_LIST OA_ALL_SPV_NAMES)
			message(FATAL_ERROR "Duplicate embedded shader name derived from outputs: ${_SPV_NAME}")
		endif()
		list(APPEND OA_ALL_SPV_NAMES "${_SPV_NAME}")
	endforeach()
	list(LENGTH OA_ALL_SPV_OUTPUTS _OA_SPV_OUTPUT_COUNT)
	list(LENGTH OA_ALL_SPV_NAMES _OA_SPV_NAME_COUNT)
	if(NOT _OA_SPV_OUTPUT_COUNT EQUAL _OA_SPV_NAME_COUNT)
		message(FATAL_ERROR
			"Shader output/name manifest mismatch: ${_OA_SPV_OUTPUT_COUNT} outputs, ${_OA_SPV_NAME_COUNT} names")
	endif()
	message(STATUS "SPIR-V manifest: ${_OA_SPV_NAME_COUNT} compiled outputs define the embedded registry")

	# Unified shader target — compute and graphics shaders.
	add_custom_target(oa_shaders ALL DEPENDS ${OA_ALL_SPV_OUTPUTS})

	# Remove orphan .spv files from previous builds (renamed/removed shaders)
	if(EXISTS "${OA_SPIRV_DIR}")
		file(GLOB_RECURSE OA_EXISTING_SPVS "${OA_SPIRV_DIR}/*.spv")
		foreach(SPV ${OA_EXISTING_SPVS})
			if(NOT SPV IN_LIST OA_ALL_SPV_OUTPUTS)
				file(REMOVE ${SPV})
				message(STATUS "Removed stale SPIR-V: ${SPV}")
			endif()
		endforeach()
	endif()

	# ninja clean_shaders → wipe all SPIR-V, forces full recompile
	add_custom_target(clean_shaders
		COMMAND ${CMAKE_COMMAND} -E rm -rf "${OA_SPIRV_DIR}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${OA_SPIRV_DIR}"
		COMMENT "Cleaning all compiled OA shaders"
	)

	# ninja clean_cache → wipe pipeline cache (use after shader changes)
	add_custom_target(clean_cache
		COMMAND ${CMAKE_COMMAND} -E rm -f "${CMAKE_SOURCE_DIR}/var/vk/pipeline.vcache"
		COMMENT "Cleaning Vulkan pipeline cache"
	)

	# Set pipeline cache as additional clean file so 'ninja clean' removes it
	set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
		"${CMAKE_SOURCE_DIR}/var/vk/pipeline.vcache"
	)

	# Shader list file — only rewrite when content changes (avoids spurious embed rebuilds)
	set(OA_SHADER_LIST_FILE "${CMAKE_CURRENT_BINARY_DIR}/spirv_shader_list.txt")
	list(JOIN OA_ALL_SPV_NAMES "\n" OA_SHADER_LIST_CONTENT)
	set(OA_SHADER_LIST_CHANGED FALSE)
	if(EXISTS "${OA_SHADER_LIST_FILE}")
		file(READ "${OA_SHADER_LIST_FILE}" OA_SHADER_LIST_OLD)
		if(NOT OA_SHADER_LIST_OLD STREQUAL OA_SHADER_LIST_CONTENT)
			set(OA_SHADER_LIST_CHANGED TRUE)
		endif()
	else()
		set(OA_SHADER_LIST_CHANGED TRUE)
	endif()
	if(OA_SHADER_LIST_CHANGED)
		file(WRITE ${OA_SHADER_LIST_FILE} "${OA_SHADER_LIST_CONTENT}")
	endif()

	set(OA_EMBED_CPP "${CMAKE_CURRENT_BINARY_DIR}/oa_spirv_embed.cpp")
	add_custom_command(
		OUTPUT ${OA_EMBED_CPP}
		COMMAND ${CMAKE_COMMAND}
			-DSPIRV_DIR=${OA_SPIRV_DIR}
			-DOUTPUT=${OA_EMBED_CPP}
			-DSHADER_LIST_FILE=${OA_SHADER_LIST_FILE}
			-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
		DEPENDS
			${OA_ALL_SPV_OUTPUTS}
			${OA_SHADER_LIST_FILE}
			${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
		COMMENT "Embedding SPIR-V shaders into liboa"
	)
	target_sources(oa PRIVATE ${OA_EMBED_CPP})
	target_compile_definitions(oa PUBLIC OA_EMBED_SHADERS)
	message(STATUS "SPIR-V embedding: ON (build-generated registry is the sole runtime source)")
endif()

# TESTS
# Tutorials use the GTest harness (MlTestMain) + add_test, so Gtest/enable_testing are
# needed whenever tutorials OR tests are built.
if(OA_BUILD_TESTS OR OA_BUILD_TUTORIALS)
	enable_testing()
	find_package(GTest CONFIG REQUIRED)
endif()

# DEVELOPER KIT
# Sample-domain support, applications, examples and tutorials are consumers of
# the OA library. They live outside source/ so they cannot become installed API
# accidentally.
if(NOT OA_ANDROID_ML)
	add_subdirectory(sdk)
endif()

if(OA_BUILD_TESTS)
	# Test suite (unit tests, benchmarks, kernel correctness)
	add_subdirectory(test)
endif()

# INSTALL
include(GNUInstallDirs)

# License material is part of the runtime artifact, not an SDK afterthought.
# vcpkg's `copyright` files are the exact notices for the dependency versions
# selected by the lock/baseline used for this build.
set(OA_LICENSE_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/licenses/oa")
file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" OA_PROJECT_LICENSE_FILE)
install(FILES "${OA_PROJECT_LICENSE_FILE}"
	DESTINATION "${OA_LICENSE_INSTALL_DIR}"
	RENAME LICENSE
	COMPONENT oa
)
install(FILES NOTICE.md
	DESTINATION "${OA_LICENSE_INSTALL_DIR}"
	COMPONENT oa
)
install(FILES
	source/cpp/thirdparty/licenses/volk.txt
	source/cpp/thirdparty/licenses/vma.txt
	source/cpp/thirdparty/licenses/miniaudio.txt
	source/cpp/thirdparty/licenses/stb.txt
	DESTINATION "${OA_LICENSE_INSTALL_DIR}/third-party"
	COMPONENT oa
)
if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
	file(GLOB _OA_VCPKG_NOTICE_FILES
		"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/*/copyright")
	foreach(_OA_NOTICE_FILE IN LISTS _OA_VCPKG_NOTICE_FILES)
		get_filename_component(_OA_NOTICE_DIR "${_OA_NOTICE_FILE}" DIRECTORY)
		get_filename_component(_OA_NOTICE_PACKAGE "${_OA_NOTICE_DIR}" NAME)
		install(FILES "${_OA_NOTICE_FILE}"
			DESTINATION "${OA_LICENSE_INSTALL_DIR}/third-party"
			RENAME "${_OA_NOTICE_PACKAGE}.txt"
			COMPONENT oa)
	endforeach()
endif()

# Install components (consumed by the AUR split packages):
#   oa      → runtime: the shared library only. SPIR-V is baked into the .so,
#             so the runtime package needs nothing else.
#   oa_sdk  → dev: headers, Slang shader sources, compiled SPIR-V, CMake package config.
install(TARGETS oa oa_headers oa_api
	EXPORT oa-targets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT oa
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT oa_sdk
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(TARGET oa-satellite)
	install(TARGETS oa-satellite
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
		COMPONENT oa)
endif()

# Public API headers (source/cpp/include/oa/{core,runtime,…}/ → include/oa/{core,runtime,…}/)
install(DIRECTORY source/cpp/include/oa/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oa
	COMPONENT oa_sdk
)

# Older operation generators still place declaration fragments in
# source/cpp/lib while exposing them through public umbrella headers. Install
# only those declaration families; NN declarations are generated directly into
# source/cpp/include and are covered by the public-header install above.
install(FILES source/cpp/lib/oa/core/opRegistry.gen.h
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oa/core
	COMPONENT oa_sdk
)
install(DIRECTORY source/cpp/lib/oa/core/fnmatrix/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oa/core/fnmatrix
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.gen.h"
)
install(DIRECTORY source/cpp/lib/oa/ml/fnmatrix/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oa/ml/fnmatrix
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.gen.h"
)
install(DIRECTORY source/cpp/lib/oa/ml/fnloss/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oa/ml/fnloss
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.gen.h"
)
install(DIRECTORY source/cpp/lib/oa/audio/fnaudio/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oa/audio/fnaudio
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.gen.h"
)
install(DIRECTORY source/cpp/lib/oa/vision/fnimage/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oa/vision/fnimage
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.gen.h"
)

# Slang math modules + core matrix kernels (sources for consumer -I paths)
install(DIRECTORY source/cpp/lib/oa/core/shader/compute/math/
	DESTINATION lib/oa/core/shader/compute/math
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)
install(DIRECTORY source/cpp/lib/oa/core/shader/compute/matrix/
	DESTINATION lib/oa/core/shader/compute/matrix
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)
install(DIRECTORY source/cpp/lib/oa/core/shader/compute/flat/
	DESTINATION lib/oa/core/shader/compute/flat
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)

# Install Slang shader sources + compiled SPIR-V
install(DIRECTORY source/cpp/lib/oa/ml/shaders/
	DESTINATION lib/oa/ml/shaders
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)
install(DIRECTORY source/cpp/lib/oa/crypto/shader/compute/
	DESTINATION lib/oa/crypto/shader/compute
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/spirv/
	DESTINATION lib/oa/ml/shaders
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.spv"
)

install(EXPORT oa-targets
	FILE oa-targets.cmake
	NAMESPACE oa::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oa
	COMPONENT oa_sdk
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/oa-config-version.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/oa-config.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/oa-config.cmake"
	@ONLY
)

install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/oa-config.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/oa-config-version.cmake"
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oa
	COMPONENT oa_sdk
)

# ─────────────────────────────────────────────────────────────────────────────
# Python Bindings (nanobind)
# ─────────────────────────────────────────────────────────────────────────────
# nanobind is a C++ dependency: prefer the vcpkg 'python' feature (vcpkg.json →
# nanobind, auto-selected when OA_BUILD_PYTHON is ON via VCPKG_MANIFEST_FEATURES
# above). Fall back to a pip-installed nanobind for builds outside the vcpkg
# toolchain.
if(OA_BUILD_PYTHON)
	# Link the extension against the interpreter the user actually imports from
	# (the newest system Python), NOT the private CPython that the vcpkg nanobind
	# port builds purely to satisfy its own dependency. That private copy sits on
	# CMAKE_PREFIX_PATH under the vcpkg toolchain, so pin the VERSION strategy:
	# FindPython picks the HIGHEST version found (system 3.14) over vcpkg's older
	# 3.12, regardless of search-path priority. Callers can still override with
	# -DPython_EXECUTABLE=... for a specific venv/conda target.
	if(NOT DEFINED Python_EXECUTABLE AND NOT DEFINED Python_ROOT_DIR)
		set(Python_FIND_STRATEGY VERSION)
		set(Python_FIND_VIRTUALENV STANDARD)
	endif()

	# nanobind requires Python (not Python3) to be found first
	find_package(Python COMPONENTS Interpreter Development REQUIRED)
	message(STATUS "Python bindings: using interpreter ${Python_EXECUTABLE} (${Python_VERSION})")

	# vcpkg (or any CMAKE_PREFIX_PATH) copy first — this is the supported path.
	find_package(nanobind CONFIG QUIET)

	if(nanobind_FOUND)
		message(STATUS "Found nanobind via vcpkg / CMAKE_PREFIX_PATH")
	else()
		# Fallback: locate a pip-installed nanobind and point CMake at its config.
		execute_process(
			COMMAND ${Python_EXECUTABLE} -m pip show nanobind
			OUTPUT_VARIABLE NANOBIND_PIP_INFO
			ERROR_QUIET
			RESULT_VARIABLE NANOBIND_PIP_RC
		)

		if(NANOBIND_PIP_RC EQUAL 0)
			message(STATUS "Found nanobind via pip")
			string(REGEX REPLACE ".*Location: ([^\r\n]+).*" "\\1" NANOBIND_PIP_DIR "${NANOBIND_PIP_INFO}")
			set(nanobind_DIR "${NANOBIND_PIP_DIR}/nanobind/cmake")
			find_package(nanobind CONFIG REQUIRED)
		else()
			message(FATAL_ERROR "nanobind not found. Enable the vcpkg 'python' feature "
				"(configure with -DOA_BUILD_PYTHON=ON under the vcpkg toolchain) or install "
				"it for the active interpreter: pip install nanobind")
		endif()
	endif()

	# Python native module. Binding sources mirror the public OA modules; one
	# private extension preserves a single nanobind type registry across them.
	set(OA_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/py")

	set(OA_PYTHON_SOURCES
		${OA_PYTHON_DIR}/module.cpp
		${OA_PYTHON_DIR}/core/core.cpp
		${OA_PYTHON_DIR}/core/type.cpp
		${OA_PYTHON_DIR}/core/filesystem.cpp
		${OA_PYTHON_DIR}/core/factory.cpp
		${OA_PYTHON_DIR}/core/fnMatrix.cpp
		${OA_PYTHON_DIR}/core/backward.cpp
		${OA_PYTHON_DIR}/runtime/runtime.cpp
		${OA_PYTHON_DIR}/ml/ml.cpp
		${OA_PYTHON_DIR}/ml/fnMatrix.cpp
		${OA_PYTHON_DIR}/ml/module.cpp
		${OA_PYTHON_DIR}/ml/nn.cpp
		${OA_PYTHON_DIR}/ml/loss.cpp
		${OA_PYTHON_DIR}/ml/autograd.cpp
		${OA_PYTHON_DIR}/ml/optim.cpp
		${OA_PYTHON_DIR}/ml/training.cpp
		${OA_PYTHON_DIR}/ml/metric.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/sdk/py/lib/ml/nlp.cpp
		${OA_PYTHON_DIR}/ml/reinforcement.cpp
		${OA_PYTHON_DIR}/audio/audio.cpp
		${OA_PYTHON_DIR}/audio/type.cpp
		${OA_PYTHON_DIR}/audio/session.cpp
		${OA_PYTHON_DIR}/audio/codec.cpp
		${OA_PYTHON_DIR}/audio/fnAudio.cpp
		${OA_PYTHON_DIR}/vision/vision.cpp
		${OA_PYTHON_DIR}/vision/type.cpp
		${OA_PYTHON_DIR}/vision/image.cpp
		${OA_PYTHON_DIR}/vision/detection.cpp
		${OA_PYTHON_DIR}/vision/codec.cpp
		${OA_PYTHON_DIR}/vision/video.cpp
		${OA_PYTHON_DIR}/ui/viewer.cpp
		${OA_PYTHON_DIR}/ui/plot.cpp
	)
	# Crypto binding only when the crypto backend is built (oa::FnHash/sign +
	# liboqs). Module.cpp guards the BindCrypto() call with #ifdef OA_BUILD_CRYPTO.
	if(OA_BUILD_CRYPTO)
		list(APPEND OA_PYTHON_SOURCES
			${OA_PYTHON_DIR}/crypto/crypto.cpp
			${OA_PYTHON_DIR}/crypto/hash.cpp
			${OA_PYTHON_DIR}/crypto/sign.cpp
			${OA_PYTHON_DIR}/crypto/fnHash.cpp
		)
	endif()

	# `_oa` is deliberately private. Users import the public `oa` package and its
	# `core`, `runtime`, `ml`, `audio`, `vision`, and optional `crypto` modules.
	nanobind_add_module(_oa
		${OA_PYTHON_SOURCES}
	)
	target_link_libraries(_oa PRIVATE oa oaSdkSupport)
	target_include_directories(_oa PRIVATE
		${OA_PYTHON_DIR}
		${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/include
		${CMAKE_CURRENT_SOURCE_DIR}/source/cpp/lib
		${CMAKE_CURRENT_SOURCE_DIR}/sdk/cpp/include
	)
	# Fix TLS relocation error for shared library
	target_compile_options(_oa PRIVATE -ftls-model=initial-exec)

	if(SKBUILD)
		# Keep native implementation details inside the `oa` package. Both DSOs live
		# together, so the extension resolves liboa through $ORIGIN.
		set_target_properties(_oa PROPERTIES INSTALL_RPATH "$ORIGIN")
		install(TARGETS _oa LIBRARY DESTINATION oa COMPONENT PythonWheel)
		install(TARGETS oa LIBRARY DESTINATION oa COMPONENT PythonWheel)
		# Wheels can vendor native dependency binaries during auditwheel repair.
		# Carry the attribution closure inside the package instead of relying on
		# host package-manager metadata.
		install(FILES "${OA_PROJECT_LICENSE_FILE}" NOTICE.md
			DESTINATION oa/licenses
			COMPONENT PythonWheel)
		install(FILES
			source/cpp/thirdparty/licenses/volk.txt
			source/cpp/thirdparty/licenses/vma.txt
			source/cpp/thirdparty/licenses/miniaudio.txt
			source/cpp/thirdparty/licenses/stb.txt
			DESTINATION oa/licenses/third-party
			COMPONENT PythonWheel)
		foreach(_OA_NOTICE_FILE IN LISTS _OA_VCPKG_NOTICE_FILES)
			get_filename_component(_OA_NOTICE_DIR "${_OA_NOTICE_FILE}" DIRECTORY)
			get_filename_component(_OA_NOTICE_PACKAGE "${_OA_NOTICE_DIR}" NAME)
			install(FILES "${_OA_NOTICE_FILE}"
				DESTINATION oa/licenses/third-party
				RENAME "${_OA_NOTICE_PACKAGE}.txt"
				COMPONENT PythonWheel)
		endforeach()
	else()
		# Dev/system install uses the same package layout as the wheel.
		install(TARGETS _oa LIBRARY DESTINATION ${Python_SITEARCH}/oa)
		install(DIRECTORY ${OA_PYTHON_DIR}/oa/
			DESTINATION ${Python_SITEARCH}/oa
			FILES_MATCHING
				PATTERN "*.py"
				PATTERN "*.pyi"
				PATTERN "py.typed")
	endif()

	if(OA_BUILD_TESTS)
		add_subdirectory(test/py)
	endif()

	message(STATUS "Python bindings: ENABLED (nanobind)")
else()
	message(STATUS "Python bindings: DISABLED (set -DOA_BUILD_PYTHON=ON to enable)")
endif()

oa_apply_executable_output_names("${CMAKE_CURRENT_SOURCE_DIR}")

# SUMMARY
message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════════════════╗")
message(STATUS "║             OA — REALM FOUNDATION LIBRARY                ")
message(STATUS "╠═══════════════════════════════════════════════════════════════╣")
message(STATUS "║  Version:     ${PROJECT_VERSION}                                        ")
message(STATUS "║  Build type:  ${CMAKE_BUILD_TYPE}                                     ")
message(STATUS "║  Validation:  ${OA_VULKAN_VALIDATION}                                       ")
message(STATUS "║  Crypto:      ${OA_BUILD_CRYPTO}                                       ")
message(STATUS "║  Python:      ${OA_BUILD_PYTHON}                                       ")
message(STATUS "║  Tests:       ${OA_BUILD_TESTS}                                       ")
message(STATUS "╚═══════════════════════════════════════════════════════════════╝")
message(STATUS "")
